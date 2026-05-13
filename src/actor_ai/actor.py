# Future imports (must occur at the beginning of the file):
from __future__ import annotations

# Third party imports:
import pykka

from .accounting import Ledger, MonitoringContext, UsageSummary, new_session_id
from .messages import Forget, Instruct, Remember
from .providers import Claude, LLMProvider
from .tools import extract_tools


class AIActor(pykka.ThreadingActor):
    """A pykka ThreadingActor that can be instructed in natural language.

    Features
    --------
    * **Multi-provider** — set ``provider`` to ``Claude``, ``GPT``, ``Gemini``,
      ``Mistral``, or ``DeepSeek`` (default: ``Claude()``).
    * **Session** — ``instruct()`` calls are automatically chained: the actor
      accumulates a rolling conversation so each turn has full context.
      Control the window with ``max_history`` (number of *turns*, i.e.
      user+assistant pairs; 0 = unlimited).
    * **Memory** — persist named facts across sessions via ``Remember``/
      ``Forget`` messages or ``remember()``/``forget()`` methods.  Facts are
      appended to the system prompt automatically.
    * **Tools** — decorate methods with ``@tool`` to expose them to the LLM.
    * **Accounting** — attach a ``Ledger`` instance to track token usage and
      calculate cost.  Optionally override ``actor_name`` to give the actor a
      human-readable label in accounting reports.

    Example::

        from actor_ai import AIActor, tool, Claude, GPT, Ledger, Rates

        ledger = Ledger()

        class AssistantActor(AIActor):
            system_prompt = "You are a helpful assistant."
            provider = GPT("gpt-4o")
            max_history = 20
            ledger = ledger
            actor_name = "assistant"

            @tool
            def get_time(self) -> str:
                "Return the current UTC time."
                import datetime
                return datetime.datetime.utcnow().isoformat()

        ref = AssistantActor.start()

        ref.proxy().instruct("My name is Alice.").get()
        ref.proxy().instruct("What is my name?").get()  # → "Alice"

        # Inspect spend after the calls:
        print(ledger.total_usage())
        print(ledger.total_cost(Rates.default()))

        ref.stop()
    """

    system_prompt: str = "You are a helpful AI agent."
    max_tokens: int = 4096
    max_history: int = 0
    provider: LLMProvider = Claude()
    ledger: Ledger | None = None
    actor_name: str | None = None
    monitoring: bool = False

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def on_start(self) -> None:
        self._session: list[dict] = []
        self._memory: dict[str, str] = {}
        self._session_id: str = new_session_id()

    # ------------------------------------------------------------------ #
    # Message routing                                                      #
    # ------------------------------------------------------------------ #

    def on_receive(self, message: object) -> object:
        if isinstance(message, Instruct):
            return self.instruct(
                message.instruction,
                history=message.history or None,
                use_session=message.use_session,
            )
        if isinstance(message, Remember):
            self.remember(message.key, message.value)
            return None
        if isinstance(message, Forget):
            self.forget(message.key)
            return None
        return super().on_receive(message)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def instruct(
        self,
        instruction: str,
        history: list[dict] | None = None,
        use_session: bool = True,
    ) -> str:
        """Process a natural language instruction, invoking @tool methods as needed."""
        if history is not None:
            messages = list(history)
        elif use_session:
            messages = list(self._session)
        else:
            messages = []

        messages.append({"role": "user", "content": instruction})

        accumulated = UsageSummary()

        def _on_usage(usage: UsageSummary) -> None:
            nonlocal accumulated
            accumulated += usage

        monitoring_ctx: MonitoringContext | None = None
        if self.monitoring:
            monitoring_ctx = MonitoringContext(
                actor_name=self.actor_name or type(self).__name__,
                session_id=self._session_id,
            )

        reply = self.provider.run(
            system=self._effective_system_prompt(),
            messages=messages,
            tools=extract_tools(self),
            dispatcher=self._dispatch_tool,
            max_tokens=self.max_tokens,
            on_usage=_on_usage if self.ledger is not None else None,
            monitoring_context=monitoring_ctx,
        )

        if self.ledger is not None:
            self.ledger.record(
                actor_name=self.actor_name or type(self).__name__,
                model=self.provider.model,
                input_tokens=accumulated.input_tokens,
                output_tokens=accumulated.output_tokens,
                session_id=self._session_id,
            )

        if use_session and history is None:
            self._session.append({"role": "user", "content": instruction})
            self._session.append({"role": "assistant", "content": reply})
            self._trim_session()

        return reply

    def remember(self, key: str, value: str) -> None:
        """Persist a named fact that will be included in every system prompt."""
        self._memory[key] = value

    def forget(self, key: str) -> None:
        """Remove a previously remembered fact."""
        self._memory.pop(key, None)

    def clear_session(self) -> None:
        """Discard the current conversation session and start a new session ID."""
        self._session.clear()
        self._session_id = new_session_id()

    def get_session(self) -> list[dict]:
        """Return a copy of the current session history."""
        return list(self._session)

    def get_memory(self) -> dict[str, str]:
        """Return a copy of the current memory store."""
        return dict(self._memory)

    def get_session_id(self) -> str:
        """Return the current session ID (changes after every ``clear_session()``)."""
        return self._session_id

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _effective_system_prompt(self) -> str:
        if not self._memory:
            return self.system_prompt
        facts = "\n".join(f"- {k}: {v}" for k, v in self._memory.items())
        return f"{self.system_prompt}\n\nKnown facts:\n{facts}"

    def _trim_session(self) -> None:
        if self.max_history > 0:
            keep = self.max_history * 2
            if len(self._session) > keep:
                self._session = self._session[-keep:]

    def _dispatch_tool(self, name: str, args: dict) -> object:
        cls_attr = getattr(type(self), name, None)
        if cls_attr is None or not getattr(cls_attr, "_is_ai_tool", False):
            raise ValueError(f"Tool {name!r} not found or not decorated with @tool")
        return getattr(self, name)(**args)
