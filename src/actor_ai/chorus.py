# Future imports (must occur at the beginning of the file):
from __future__ import annotations

# Third party imports:
import pykka

from .messages import Forget, Remember


class Chorus(pykka.ThreadingActor):
    """A group of named AIActors that can be instructed individually or together.

    The Chorus itself is a pykka actor — all its methods are accessible via
    proxy and all interaction is thread-safe.

    Example::

        from actor_ai import AIActor, Chorus, Claude, GPT

        class Researcher(AIActor):
            system_prompt = "You are a research specialist."
            provider = GPT("gpt-4o")

        class Writer(AIActor):
            system_prompt = "You are a creative writer."
            provider = Claude()

        # Build the chorus from running actor refs:
        researcher_ref = Researcher.start()
        writer_ref = Writer.start()

        chorus_ref = Chorus.start(
            agents={"researcher": researcher_ref, "writer": writer_ref}
        )

        # Instruct one agent:
        result = chorus_ref.proxy().instruct("researcher", "Find facts about Mars.").get()

        # Broadcast to all agents, collect replies keyed by name:
        results = chorus_ref.proxy().broadcast("Introduce yourself.").get()

        # Pipeline: chain output of one agent as input to the next:
        result = chorus_ref.proxy().pipeline(
            ["researcher", "writer"],
            "Write a short story about Mars exploration.",
        ).get()

        chorus_ref.stop()
    """

    def __init__(self, agents: dict[str, pykka.ActorRef] | None = None) -> None:
        super().__init__()
        self._agents: dict[str, pykka.ActorRef] = dict(agents or {})

    # ------------------------------------------------------------------ #
    # Agent management                                                     #
    # ------------------------------------------------------------------ #

    def add(self, name: str, ref: pykka.ActorRef) -> None:
        """Register a new agent under *name*."""
        self._agents[name] = ref

    def remove(self, name: str) -> None:
        """Unregister an agent (does not stop it)."""
        self._agents.pop(name, None)

    def agents(self) -> list[str]:
        """Return the names of all registered agents."""
        return list(self._agents.keys())

    # ------------------------------------------------------------------ #
    # Instruction routing                                                  #
    # ------------------------------------------------------------------ #

    def instruct(self, name: str, instruction: str, **kwargs) -> str:
        """Send an instruction to the named agent and return its reply.

        Keyword args (``use_session``, ``history``) are forwarded to the
        agent's ``instruct()`` method via the proxy.
        """
        ref = self._get(name)
        return ref.proxy().instruct(instruction, **kwargs).get()

    def broadcast(self, instruction: str, **kwargs) -> dict[str, str]:
        """Send the same instruction to all agents in parallel.

        Returns a dict mapping agent name → reply.
        """
        futures = {
            name: ref.proxy().instruct(instruction, **kwargs) for name, ref in self._agents.items()
        }
        return {name: f.get() for name, f in futures.items()}

    def pipeline(self, names: list[str], instruction: str, **kwargs) -> str:
        """Pass output of each agent as input to the next in *names*.

        The first agent receives *instruction*; each subsequent agent receives
        the previous agent's reply.  Returns the final agent's reply.
        """
        if not names:
            raise ValueError("pipeline requires at least one agent name")
        message = instruction
        for name in names:
            message = self.instruct(name, message, **kwargs)
        return message

    # ------------------------------------------------------------------ #
    # Memory broadcast                                                     #
    # ------------------------------------------------------------------ #

    def remember(self, key: str, value: str, names: list[str] | None = None) -> None:
        """Store a fact in one or all agents' memory.

        Args:
            key: Fact key.
            value: Fact value.
            names: Agent names to update; ``None`` means all agents.
        """
        targets = self._resolve(names)
        for ref in targets.values():
            ref.tell(Remember(key, value))

    def forget(self, key: str, names: list[str] | None = None) -> None:
        """Remove a fact from one or all agents' memory.

        Args:
            key: Fact key to remove.
            names: Agent names to update; ``None`` means all agents.
        """
        targets = self._resolve(names)
        for ref in targets.values():
            ref.tell(Forget(key))

    # ------------------------------------------------------------------ #
    # Lifecycle helpers                                                    #
    # ------------------------------------------------------------------ #

    def stop_agents(self, names: list[str] | None = None) -> None:
        """Stop one or all registered agents (and remove them from the chorus)."""
        targets = self._resolve(names)
        for name, ref in targets.items():
            ref.stop()
            self._agents.pop(name, None)

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _get(self, name: str) -> pykka.ActorRef:
        try:
            return self._agents[name]
        except KeyError:
            available = ", ".join(self._agents) or "(none)"
            raise KeyError(f"Agent {name!r} not found. Available: {available}") from None

    def _resolve(self, names: list[str] | None) -> dict[str, pykka.ActorRef]:
        if names is None:
            return dict(self._agents)
        return {n: self._get(n) for n in names}
