# Future imports (must occur at the beginning of the file):
from __future__ import annotations

# Standard library imports:
import os
from collections.abc import AsyncGenerator, Callable, Generator, Iterable
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import IO, Any, TypeVar, overload

# Third party imports:
import pykka

from .accounting import Ledger, MonitoringContext, UsageSummary, new_session_id
from .context import SharedContext
from .messages import Forget, Instruct, Remember
from .providers import LLMProvider
from .tools import extract_tools, tool

type InstructionInput = str | os.PathLike[str] | IO[str]

_EvalT = TypeVar("_EvalT")


def _resolve_instruction(instruction: InstructionInput) -> str:
    """Resolve an instruction to a plain string.

    Accepts a str, any os.PathLike (e.g. pathlib.Path), or a readable stream.
    Binary streams and Path reads are decoded as UTF-8.
    """
    if isinstance(instruction, str):
        return instruction
    if isinstance(instruction, os.PathLike):
        return Path(instruction).read_text(encoding="utf-8")
    if hasattr(instruction, "read"):
        content = instruction.read()
        return content.decode("utf-8") if isinstance(content, bytes) else content
    raise TypeError(
        f"instruction must be a str, path-like object, or readable stream; "
        f"got {type(instruction).__name__!r}"
    )


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
    context: SharedContext | None = None

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def on_start(self) -> None:
        self._session: list[dict] = []
        self._memory: dict[str, str] = {}
        self._working_memory: dict[str, str] = {}
        self._session_id: str = new_session_id()
        self._usage: UsageSummary = UsageSummary()

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
        instruction: str | os.PathLike[str] | IO[str],
        history: list[dict] | None = None,
        use_session: bool = True,
    ) -> str:
        """Process a natural language instruction, invoking @tool methods as needed.

        ``instruction`` may be a plain string, a :class:`pathlib.Path` (or any
        ``os.PathLike``), or a readable text/binary stream.  Paths are read with
        UTF-8 encoding; streams are read and decoded if they return bytes.
        """
        text = _resolve_instruction(instruction)
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

        messages.append({"role": "user", "content": text})

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
            on_usage=_on_usage,
            monitoring_context=monitoring_ctx,
        )

        self._usage += accumulated

        if self.ledger is not None:
            self.ledger.record(
                actor_name=self.actor_name or type(self).__name__,
                model=provider.model,
                input_tokens=accumulated.input_tokens,
                output_tokens=accumulated.output_tokens,
                session_id=self._session_id,
            )

        if use_session and history is None:
            self._session.append({"role": "user", "content": text})
            self._session.append({"role": "assistant", "content": reply})
            self._trim_session()

        if self.context is not None:
            label = self.actor_name or type(self).__name__
            self.context.append_log(label, "user", text)
            self.context.append_log(label, "assistant", reply)

        return reply

    @overload
    def instruct_many(
        self,
        instructions: Iterable[InstructionInput],
        evaluate: None = ...,
        *,
        use_session: bool = ...,
    ) -> list[str]: ...

    @overload
    def instruct_many(
        self,
        instructions: Iterable[InstructionInput],
        evaluate: Callable[[list[tuple[str, str]]], _EvalT],
        *,
        use_session: bool = ...,
    ) -> _EvalT: ...

    def instruct_many(
        self,
        instructions: Iterable[InstructionInput],
        evaluate: Callable[[list[tuple[str, str]]], Any] | None = None,
        *,
        use_session: bool = False,
    ) -> list[str] | Any:
        """Run multiple instructions and return the replies, optionally evaluated.

        Each instruction is resolved (str / Path / stream → text) then sent to the
        LLM in collection order.

        Args:
            instructions: Any iterable of ``InstructionInput`` values.
            evaluate: Optional callable that receives a ``list[tuple[str, str]]``
                of *(resolved instruction text, reply)* pairs in order and returns
                an arbitrary result.  When ``None`` the raw ``list[str]`` of
                replies is returned.
            use_session: When ``False`` (default) each call is independent — the
                actor session is neither read nor written.  When ``True`` all
                instructions run in the shared session so each turn has the
                context of the previous ones.

        Returns:
            ``list[str]`` when *evaluate* is ``None``; otherwise whatever
            *evaluate* returns.
        """
        resolved = [_resolve_instruction(i) for i in instructions]
        replies = [self.instruct(text, use_session=use_session) for text in resolved]
        pairs = list(zip(resolved, replies))
        return evaluate(pairs) if evaluate is not None else replies

    def remember(self, key: str, value: str) -> None:
        """Persist a named fact that will be included in every system prompt."""
        if self.context is not None:
            self.context.remember(key, value)
        else:
            self._memory[key] = value

    def forget(self, key: str) -> None:
        """Remove a previously remembered fact."""
        if self.context is not None:
            self.context.forget(key)
        else:
            self._memory.pop(key, None)

    def clear_session(self) -> None:
        """Discard the current conversation session and start a new session ID.

        When no shared context is attached, also clears working memory (task-scoped
        facts that do not survive a session reset).  When a :class:`SharedContext`
        is attached, working memory is NOT cleared — shared state belongs to the
        context and must be reset explicitly via ``clear_working_memory()``.
        Long-term memory (``remember()``/``forget()``) is always unaffected.
        """
        self._session.clear()
        if self.context is None:
            self._working_memory.clear()
        self._session_id = new_session_id()

    def get_session(self) -> list[dict]:
        """Return a copy of the current session history."""
        return list(self._session)

    def get_memory(self) -> dict[str, str]:
        """Return a copy of the current memory store."""
        if self.context is not None:
            return self.context.get_memory()
        return dict(self._memory)

    def get_session_id(self) -> str:
        """Return the current session ID (changes after every ``clear_session()``)."""
        return self._session_id

    # Working memory ---------------------------------------------------- #

    def remember_working(self, key: str, value: str) -> None:
        """Store a task-scoped fact injected into the system prompt for this session.

        Without a shared context: cleared by ``clear_session()``.
        With a shared context: delegates to the context and is visible to all
        agents sharing it; use ``clear_working_memory()`` to reset.
        For durable facts use ``remember()`` instead.
        """
        if self.context is not None:
            self.context.remember_working(key, value)
        else:
            self._working_memory[key] = value

    def forget_working(self, key: str) -> None:
        """Remove a working-memory fact (no-op if the key does not exist)."""
        if self.context is not None:
            self.context.forget_working(key)
        else:
            self._working_memory.pop(key, None)

    def get_working_memory(self) -> dict[str, str]:
        """Return a copy of the current working-memory store."""
        if self.context is not None:
            return self.context.get_working_memory()
        return dict(self._working_memory)

    def clear_working_memory(self) -> None:
        """Remove all working-memory facts without resetting the session."""
        if self.context is not None:
            self.context.clear_working_memory()
        else:
            self._working_memory.clear()

    # Usage tracking ---------------------------------------------------- #

    def get_usage(self) -> UsageSummary:
        """Return a snapshot of token usage accumulated since the last ``reset_usage()``."""
        return UsageSummary(
            input_tokens=self._usage.input_tokens,
            output_tokens=self._usage.output_tokens,
        )

    def reset_usage(self) -> None:
        """Reset the internal usage counter to zero."""
        self._usage = UsageSummary()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _effective_system_prompt(self) -> str:
        result = self.system_prompt
        memory = self.context.get_memory() if self.context else self._memory
        working = self.context.get_working_memory() if self.context else self._working_memory
        if memory:
            facts = "\n".join(f"- {k}: {v}" for k, v in memory.items())
            result = f"{result}\n\nKnown facts:\n{facts}"
        if working:
            wm = "\n".join(f"- {k}: {v}" for k, v in working.items())
            result = f"{result}\n\nWorking memory:\n{wm}"
        return result

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
    context: SharedContext | None = None,
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
        "context": context,
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
