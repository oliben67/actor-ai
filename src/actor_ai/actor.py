# Future imports (must occur at the beginning of the file):
from __future__ import annotations

# Standard library imports:
from collections.abc import AsyncGenerator, Callable, Generator
from contextlib import asynccontextmanager, contextmanager

# Third party imports:
import pykka

from .accounting import Ledger, MonitoringContext, UsageSummary, new_session_id
from .messages import Forget, Instruct, Remember
from .providers import LLMProvider
from .tools import extract_tools, tool


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
    provider: LLMProvider | None = None
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
        if self.provider is None:
            raise RuntimeError(
                "No provider configured. Set a provider class attribute:\n"
                "    class MyActor(AIActor):\n"
                "        provider = Claude()  # or GPT(), Gemini(), etc."
            )
        provider = self.provider
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

        reply = provider.run(
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
                model=provider.model,
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

    @classmethod
    @contextmanager
    def get_proxy(cls) -> Generator[pykka.ActorProxy]:
        """Context manager that starts the actor, yields its proxy, then stops it.

        Example::

            with MyAgent.get_proxy() as proxy:
                reply = proxy.instruct("Hello!").get()
        """
        ref = cls.start()
        try:
            yield ref.proxy()
        finally:
            ref.stop()

    @classmethod
    @asynccontextmanager
    async def aget_proxy(cls) -> AsyncGenerator[pykka.ActorProxy]:
        """Async context manager that starts the actor, yields its proxy, then stops it.

        The actor itself runs in a thread (pykka's standard model). Use
        ``asyncio.to_thread`` to call blocking proxy methods without blocking
        the event loop.

        Example::

            async with MyAgent.aget_proxy() as proxy:
                reply = await asyncio.to_thread(proxy.instruct("Hello!").get)
        """
        ref = cls.start()
        try:
            yield ref.proxy()
        finally:
            ref.stop()


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def _make_sub_agent_tool(cls: type[AIActor], agent_name: str) -> Callable:
    """Return a @tool-decorated method that delegates instruct() to a sub-agent class."""

    def delegate(self, instruction: str) -> str:
        with cls.get_proxy() as proxy:
            return proxy.instruct(instruction).get()

    delegate.__name__ = agent_name
    delegate.__doc__ = f"Delegate to the {agent_name} specialist agent."
    return tool(delegate)


def make_agent(
    name: str,
    system_prompt: str,
    provider: LLMProvider | None = None,
    *,
    tools: list[Callable] | None = None,
    sub_agents: dict[str, type[AIActor]] | None = None,
    max_tokens: int = 4096,
    max_history: int = 0,
    ledger: Ledger | None = None,
    actor_name: str | None = None,
    monitoring: bool = False,
) -> type[AIActor]:
    """Create a configured AIActor subclass without writing a class definition.

    Returns a class that can be started with ``.start()`` or used with
    ``get_proxy()`` / ``aget_proxy()``.

    Args:
        name: Class name and default ``actor_name`` label in accounting reports.
        system_prompt: System prompt sent on every ``instruct()`` call.
        provider: LLM backend to use. ``None`` means no provider (pure actor).
        tools: Callables to expose to the LLM. Functions already decorated with
            ``@tool`` are used as-is; plain functions are auto-decorated.
        sub_agents: Mapping of ``method_name → AIActor subclass``. Each entry
            is auto-wired as a ``@tool`` method; when the LLM calls it the
            sub-agent is started, instructed, and stopped automatically.
        max_tokens: Maximum completion tokens (default 4096).
        max_history: Rolling session window in turns; 0 = unlimited.
        ledger: Attach a ``Ledger`` instance for token accounting.
        actor_name: Human-readable label in accounting reports. Defaults to *name*.
        monitoring: Forward metadata to LiteLLM when ``True``.

    Example::

        from actor_ai import make_agent, Claude, GPT

        Researcher = make_agent(
            "Researcher",
            "You are a deep research specialist. Cite sources.",
            Claude(),
        )

        Writer = make_agent(
            "Writer",
            "You write clear, concise summaries.",
            GPT("gpt-4o"),
        )

        Orchestrator = make_agent(
            "Orchestrator",
            "Coordinate research and writing tasks. Use available tools.",
            Claude(),
            sub_agents={"researcher": Researcher, "writer": Writer},
        )

        with Orchestrator.get_proxy() as proxy:
            reply = proxy.instruct("Write a report on climate change.").get()
    """
    attrs: dict[str, object] = {
        "system_prompt": system_prompt,
        "provider": provider,
        "max_tokens": max_tokens,
        "max_history": max_history,
        "ledger": ledger,
        "actor_name": actor_name or name,
        "monitoring": monitoring,
    }

    if tools:
        for fn in tools:
            if not getattr(fn, "_is_ai_tool", False):
                fn = tool(fn)
            attrs[fn.__name__] = fn

    if sub_agents:
        for agent_name, agent_cls in sub_agents.items():
            attrs[agent_name] = _make_sub_agent_tool(agent_cls, agent_name)

    return type(name, (AIActor,), attrs)
