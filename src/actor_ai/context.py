# Future imports (must occur at the beginning of the file):
from __future__ import annotations

# Standard library imports:
import threading


class SharedContext:
    """Thread-safe shared memory and conversation log for multi-agent sessions.

    Multiple :class:`AIActor` instances can reference the same ``SharedContext``
    to share long-term memory, working memory, and an append-only conversation
    log.  All methods are safe to call concurrently from different actor threads.

    Memory tiers
    ------------
    * **Long-term** — facts that outlive any single session (``remember``/``forget``).
    * **Working** — task-scoped facts cleared explicitly via
      ``clear_working_memory()`` (agents do NOT clear shared working memory on
      ``clear_session()`` — use ``clear_working_memory()`` to reset it).
    * **Conversation log** — append-only record of every ``(agent, role, content)``
      entry written by agents during ``instruct()`` calls.

    Example::

        from actor_ai import make_agent, SharedContext, Claude

        ctx = SharedContext()
        ctx.remember("project", "Q3 planning")

        Analyst = make_agent("Analyst", "You are a data analyst.", Claude(), context=ctx)
        Writer = make_agent("Writer", "You write summaries.", Claude(), context=ctx)

        with Analyst.get_proxy() as a, Writer.get_proxy() as w:
            analysis = a.instruct("Analyse the latest data.").get()
            summary = w.instruct("Summarise the analysis above.").get()

        for entry in ctx.get_log():
            print(entry["agent"], entry["role"], entry["content"][:60])
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._memory: dict[str, str] = {}
        self._working_memory: dict[str, str] = {}
        self._log: list[dict] = []

    # ------------------------------------------------------------------ #
    # Long-term memory                                                     #
    # ------------------------------------------------------------------ #

    def remember(self, key: str, value: str) -> None:
        """Store a long-term fact visible to all agents sharing this context."""
        with self._lock:
            self._memory[key] = value

    def forget(self, key: str) -> None:
        """Remove a long-term fact (no-op if the key does not exist)."""
        with self._lock:
            self._memory.pop(key, None)

    def get_memory(self) -> dict[str, str]:
        """Return a snapshot of the shared long-term memory."""
        with self._lock:
            return dict(self._memory)

    # ------------------------------------------------------------------ #
    # Working memory                                                       #
    # ------------------------------------------------------------------ #

    def remember_working(self, key: str, value: str) -> None:
        """Store a task-scoped fact in shared working memory."""
        with self._lock:
            self._working_memory[key] = value

    def forget_working(self, key: str) -> None:
        """Remove a working-memory fact (no-op if the key does not exist)."""
        with self._lock:
            self._working_memory.pop(key, None)

    def get_working_memory(self) -> dict[str, str]:
        """Return a snapshot of the shared working memory."""
        with self._lock:
            return dict(self._working_memory)

    def clear_working_memory(self) -> None:
        """Remove all working-memory facts."""
        with self._lock:
            self._working_memory.clear()

    # ------------------------------------------------------------------ #
    # Conversation log                                                     #
    # ------------------------------------------------------------------ #

    def append_log(self, agent_name: str, role: str, content: str) -> None:
        """Append an entry to the shared conversation log.

        Called automatically by :meth:`AIActor.instruct` when the actor has a
        context attached.  You can also call it directly to inject synthetic
        entries.
        """
        with self._lock:
            self._log.append({"agent": agent_name, "role": role, "content": content})

    def get_log(self) -> list[dict]:
        """Return a snapshot of the shared conversation log."""
        with self._lock:
            return list(self._log)

    def clear_log(self) -> None:
        """Clear the shared conversation log."""
        with self._lock:
            self._log.clear()
