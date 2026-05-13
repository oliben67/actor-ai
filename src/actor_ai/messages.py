# Future imports (must occur at the beginning of the file):
from __future__ import annotations

# Standard library imports:
from dataclasses import dataclass, field


@dataclass
class Instruct:
    """Message that instructs an AIActor with a natural language prompt.

    Send via ``tell()`` for fire-and-forget, or ``ask()`` to receive the
    response::

        actor_ref.tell(Instruct("summarise the last report"))
        result = actor_ref.ask(Instruct("what is 2 + 2?"))

    When ``use_session=True`` (default) the actor's in-memory session history
    is used automatically; pass ``use_session=False`` to make a stateless
    one-off call.
    """

    instruction: str
    history: list[dict] = field(default_factory=list)
    use_session: bool = True


@dataclass
class Remember:
    """Store a key/value fact in the actor's long-term memory.

    The fact is injected into the system prompt on every subsequent call::

        actor_ref.tell(Remember("user_name", "Alice"))
    """

    key: str
    value: str


@dataclass
class Forget:
    """Remove a previously remembered fact by key::

    actor_ref.tell(Forget("user_name"))
    """

    key: str
