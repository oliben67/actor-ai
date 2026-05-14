# Future imports (must occur at the beginning of the file):
from __future__ import annotations

# Standard library imports:
from typing import Literal

# Third party imports:
import pykka

from .messages import Forget, Remember

ChorusType = Literal["system", "project", "team", "department", "custom"]


class Chorus(pykka.ThreadingActor):
    """A group of named actors that can be instructed individually or together.

    The Chorus itself is a pykka actor — all its methods are accessible via
    proxy and all interaction is thread-safe.  Members can be ``AIActor``
    instances, other ``Chorus`` instances, or any plain pykka actor that
    exposes an ``instruct(instruction)`` method.

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
            type="team",
            agents={"researcher": researcher_ref, "writer": writer_ref},
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

    def __init__(
        self,
        agents: dict[str, pykka.ActorRef] | None = None,
        type: ChorusType = "custom",
    ) -> None:
        super().__init__()
        self._agents: dict[str, pykka.ActorRef] = dict(agents or {})
        self.type: ChorusType = type

    # ------------------------------------------------------------------ #
    # Actor management                                                     #
    # ------------------------------------------------------------------ #

    def add(self, name: str, ref: pykka.ActorRef) -> None:
        """Register a new actor under *name*."""
        self._agents[name] = ref

    def remove(self, name: str) -> None:
        """Unregister an actor (does not stop it)."""
        self._agents.pop(name, None)

    def join(self, name: str, ref: pykka.ActorRef) -> None:
        """An actor joins the chorus under *name*."""
        self._agents[name] = ref

    def leave(self, name: str) -> None:
        """An actor leaves the chorus (does not stop it)."""
        self._agents.pop(name, None)

    def agents(self) -> list[str]:
        """Return the names of all registered actors."""
        return list(self._agents.keys())

    # ------------------------------------------------------------------ #
    # Instruction routing                                                  #
    # ------------------------------------------------------------------ #

    def instruct(self, name_or_instruction: str, instruction: str | None = None, **kwargs) -> str:
        """Send an instruction to a named actor or broadcast to all actors.

        Single-argument form — ``instruct(instruction)`` broadcasts to all
        members and returns each reply formatted as ``"name: reply\\n..."``
        per member.  This form enables a Chorus to act as a member inside
        another Chorus (``broadcast`` always calls ``proxy().instruct(text)``).

        Two-argument form — ``instruct(name, instruction)`` routes to the
        named member and returns its reply directly.
        """
        if instruction is None:
            results = self.broadcast(name_or_instruction, **kwargs)
            return "\n".join(f"{n}: {r}" for n, r in results.items())
        ref = self._get(name_or_instruction)
        return ref.proxy().instruct(instruction, **kwargs).get()

    def broadcast(self, instruction: str, **kwargs) -> dict[str, str]:
        """Send the same instruction to all actors in parallel.

        Returns a dict mapping actor name → reply.
        """
        futures = {
            name: ref.proxy().instruct(instruction, **kwargs) for name, ref in self._agents.items()
        }
        return {name: f.get() for name, f in futures.items()}

    def pipeline(self, names: list[str], instruction: str, **kwargs) -> str:
        """Pass output of each actor as input to the next in *names*.

        The first actor receives *instruction*; each subsequent actor receives
        the previous actor's reply.  Returns the final actor's reply.
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
        """Store a fact in one or all actors' memory.

        Args:
            key: Fact key.
            value: Fact value.
            names: Actor names to update; ``None`` means all actors.
        """
        targets = self._resolve(names)
        for ref in targets.values():
            ref.tell(Remember(key, value))

    def forget(self, key: str, names: list[str] | None = None) -> None:
        """Remove a fact from one or all actors' memory.

        Args:
            key: Fact key to remove.
            names: Actor names to update; ``None`` means all actors.
        """
        targets = self._resolve(names)
        for ref in targets.values():
            ref.tell(Forget(key))

    # ------------------------------------------------------------------ #
    # Message handling (enables use as a sub-member in another Chorus)    #
    # ------------------------------------------------------------------ #

    def on_receive(self, message: object) -> object:
        if isinstance(message, Remember):
            self.remember(message.key, message.value)
            return None
        if isinstance(message, Forget):
            self.forget(message.key)
            return None
        return super().on_receive(message)

    # ------------------------------------------------------------------ #
    # Lifecycle helpers                                                    #
    # ------------------------------------------------------------------ #

    def stop_agents(self, names: list[str] | None = None) -> None:
        """Stop one or all registered actors (and remove them from the chorus)."""
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
