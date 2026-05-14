# Future imports (must occur at the beginning of the file):
from __future__ import annotations

# Standard library imports:
import threading
from collections.abc import Callable
from dataclasses import dataclass

# Third party imports:
import pykka


@dataclass
class WorkflowState:
    """One state in a :class:`Workflow` state machine.

    The ``chorus`` is invoked each time the workflow enters this state.
    Pass a single :class:`pykka.ActorRef` to route to one actor, or a
    ``dict[str, ActorRef]`` to dispatch the instruction to **all listed
    actors in parallel** and collect their replies as a formatted
    ``"name: reply\\n..."`` string.

    ``instruction`` is a template string where ``{input}`` expands to the
    original instruction passed to :meth:`Workflow.run` or
    :meth:`Workflow.step`, and ``{output}`` expands to the reply produced
    by the previous state.  Literal braces that are not template markers
    must be doubled (``{{`` / ``}}``).

    Examples::

        # Single actor (sequential):
        WorkflowState(chorus=review_ref, instruction="Review this:\\n{output}")

        # Parallel actors — all fired simultaneously, replies combined:
        WorkflowState(
            chorus={"researcher": researcher_ref, "critic": critic_ref},
            instruction="Analyse: {input}",
        )
    """

    chorus: pykka.ActorRef | dict[str, pykka.ActorRef]
    instruction: str = "{output}"


@dataclass
class WorkflowTransition:
    """A directed edge in a :class:`Workflow` state machine.

    Either ``on_event`` or ``guard`` (or both) must be provided:

    * ``on_event`` — fires when :meth:`Workflow.event` is called with the
      matching name while the workflow is in ``source``.
    * ``guard`` — a callable that receives the Chorus reply from ``source``
      and returns ``True`` when the transition should fire.  Evaluated
      automatically after each :meth:`Workflow.step` / :meth:`Workflow.run`
      execution.

    Example::

        WorkflowTransition("draft", "review", guard=lambda r: "ready" in r)
        WorkflowTransition("review", "draft", on_event="reject")
    """

    source: str
    target: str
    on_event: str | None = None
    guard: Callable[[str], bool] | None = None


class Workflow(pykka.ThreadingActor):
    """State-machine workflow that orchestrates actors and Choruses.

    Each registered state routes to an actor, a :class:`~actor_ai.Chorus`,
    or a dict of parallel actors.  Transitions fire either when a reply
    matches a guard predicate or when an explicit named event is dispatched.
    States and transitions can be added, modified, or removed at runtime.

    **Async execution** — :meth:`run` and :meth:`step` block the calling
    thread (via pykka's proxy ``Future.get()``).  Use :meth:`run_detached`
    to execute the workflow loop in a background OS thread so the workflow
    actor's mailbox stays free for :meth:`event` and management calls
    during the run.  Both patterns are idiomatic pykka:

    * ``wf.proxy().run("x").get()``  — synchronous (blocking the caller)
    * ``wf.proxy().run_detached("x").get()``  — async (actor stays responsive)

    Example::

        from actor_ai import AIActor, Chorus, Claude
        from actor_ai import Workflow, WorkflowState, WorkflowTransition

        draft_chorus  = Chorus.start(agents={"drafter": drafter_ref})
        review_chorus = Chorus.start(agents={"reviewer": reviewer_ref})

        wf = Workflow.start(
            states={
                "draft":  WorkflowState(draft_chorus,  instruction="{input}"),
                "review": WorkflowState(review_chorus, instruction="Review:\\n{output}"),
                # Parallel actors within a single state:
                "analyse": WorkflowState(
                    chorus={"researcher": r_ref, "critic": c_ref},
                    instruction="Analyse: {input}",
                ),
            },
            transitions=[
                WorkflowTransition("draft",  "review",  guard=lambda r: len(r) > 100),
                WorkflowTransition("review", "draft",   on_event="reject"),
            ],
            initial_state="draft",
        )

        # Blocking run:
        output = wf.proxy().run("Draft a proposal.").get()

        # Non-blocking run (actor stays responsive to events):
        wf.proxy().run_detached("Draft a proposal.", on_complete=print).get()
        wf.proxy().event("reject").get()  # fires while run_detached is in progress
    """

    def __init__(
        self,
        states: dict[str, WorkflowState] | None = None,
        transitions: list[WorkflowTransition] | None = None,
        initial_state: str | None = None,
    ) -> None:
        super().__init__()
        self._states: dict[str, WorkflowState] = dict(states or {})
        self._transitions: list[WorkflowTransition] = list(transitions or [])
        self._current: str | None = initial_state
        self._last_output: str = ""

    # ------------------------------------------------------------------ #
    # State machine management                                             #
    # ------------------------------------------------------------------ #

    def add_state(self, name: str, state: WorkflowState) -> None:
        """Register (or replace) the state *name*."""
        self._states[name] = state

    def remove_state(self, name: str) -> None:
        """Unregister *name* (no-op if unknown)."""
        self._states.pop(name, None)

    def add_transition(self, transition: WorkflowTransition) -> None:
        """Append a transition to the state machine."""
        self._transitions.append(transition)

    def remove_transitions(self, source: str, target: str | None = None) -> None:
        """Remove transitions originating from *source*.

        If *target* is given, only the transition(s) leading to that state
        are removed; otherwise all transitions from *source* are removed.
        """
        self._transitions = [
            t
            for t in self._transitions
            if not (t.source == source and (target is None or t.target == target))
        ]

    def set_state(self, name: str) -> None:
        """Force the workflow into *name*, which must be a registered state."""
        if name not in self._states:
            raise KeyError(f"Unknown state {name!r}")
        self._current = name

    def states(self) -> list[str]:
        """Return the names of all registered states."""
        return list(self._states.keys())

    def current_state(self) -> str | None:
        """Return the name of the current state, or ``None`` if unset."""
        return self._current

    def last_output(self) -> str:
        """Return the reply produced by the most recent execution."""
        return self._last_output

    # ------------------------------------------------------------------ #
    # Execution                                                            #
    # ------------------------------------------------------------------ #

    def step(self, instruction: str | None = None) -> str:
        """Execute the current state once and apply guard transitions.

        ``instruction`` is used as ``{input}`` in the state's instruction
        template.  When omitted, ``{input}`` falls back to the last output.
        After execution, the first matching guard transition (if any) advances
        the current state automatically.

        Returns the combined reply from the state's actor(s).
        """
        if self._current is None:
            raise ValueError("No current state — call set_state() or pass initial_state")
        state = self._get_state(self._current)
        effective = instruction if instruction is not None else self._last_output
        formatted = self._format(state.instruction, effective, self._last_output)
        output = self._execute(state.chorus, formatted)
        self._last_output = output
        next_name = self._match_guard(self._current, output)
        if next_name is not None:
            self._current = next_name
        return output

    def run(self, instruction: str | None = None) -> str:
        """Execute from the current state, following guard transitions, until terminal.

        A state is *terminal* (within this call) when no guard transition
        matches its reply.  Event-only transitions do not block ``run()``.

        ``{input}`` in every state's instruction template expands to the
        original *instruction* passed here; ``{output}`` expands to the
        reply from the immediately preceding state.

        This method blocks the workflow actor's thread until it completes.
        Use :meth:`run_detached` to keep the actor responsive during execution.

        Returns the final reply.
        """
        if self._current is None:
            raise ValueError("No current state — call set_state() or pass initial_state")
        initial = instruction if instruction is not None else self._last_output
        output = self._last_output
        while True:
            state = self._get_state(self._current)
            formatted = self._format(state.instruction, initial, output)
            output = self._execute(state.chorus, formatted)
            self._last_output = output
            next_name = self._match_guard(self._current, output)
            if next_name is None:
                break
            self._current = next_name
        return output

    def run_detached(
        self,
        instruction: str | None = None,
        on_complete: Callable[[str], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        """Execute the workflow loop in a background thread.

        The workflow actor's mailbox remains free during execution so
        :meth:`event` and management calls can be dispatched while the
        run is in progress.

        Each step is split into two phases:

        1. **Prepare** — the actor atomically reads the current state and
           formats the instruction (fast, holds the actor briefly).
        2. **Execute** — the actor(s) in the state are invoked outside the
           workflow actor's thread (slow, actor is free).
        3. **Commit** — the actor atomically updates ``last_output`` and
           advances the state machine (fast, holds the actor briefly).

        Args:
            instruction: Initial instruction (``{input}`` in templates).
            on_complete: Called with the final output when the run finishes.
            on_error: Called with the exception if the run fails.
        """
        ref = self.actor_ref

        def _loop() -> None:
            inp = instruction
            try:
                while True:
                    action = ref.proxy().prepare_step(inp).get()
                    if action is None:
                        break
                    chorus, formatted = action
                    output = Workflow._execute(chorus, formatted)
                    continued = ref.proxy().commit_step(output).get()
                    if not continued:
                        break
                    inp = None
                if on_complete is not None:
                    on_complete(ref.proxy().last_output().get())
            except Exception as exc:  # noqa: BLE001
                if on_error is not None:
                    on_error(exc)

        threading.Thread(target=_loop, daemon=True).start()

    def event(self, name: str) -> bool:
        """Fire a named event and transition to the matching target state.

        Searches transitions originating from the current state for one
        whose ``on_event`` equals *name*.  The first match wins.

        Returns ``True`` if a transition was taken, ``False`` otherwise.
        """
        for t in self._transitions:
            if t.source == self._current and t.on_event == name:
                self._current = t.target
                return True
        return False

    # ------------------------------------------------------------------ #
    # Coordination methods for run_detached's background thread           #
    # ------------------------------------------------------------------ #

    def prepare_step(
        self, instruction: str | None
    ) -> tuple[pykka.ActorRef | dict[str, pykka.ActorRef], str] | None:
        """Return ``(chorus, formatted_instruction)`` for the current state, or ``None``.

        Called by the background thread in :meth:`run_detached` to atomically
        read state before the slow actor-execution phase.
        """
        if self._current is None:
            return None
        state = self._get_state(self._current)  # KeyError propagates to caller
        effective = instruction if instruction is not None else self._last_output
        formatted = self._format(state.instruction, effective, self._last_output)
        return state.chorus, formatted

    def commit_step(self, output: str) -> bool:
        """Store *output*, apply guard transitions, return ``True`` if state advanced.

        Called by the background thread in :meth:`run_detached` to atomically
        update state after the slow actor-execution phase.
        """
        self._last_output = output
        if self._current is None:
            return False
        next_name = self._match_guard(self._current, output)
        if next_name is None:
            return False
        self._current = next_name
        return True

    # ------------------------------------------------------------------ #
    # Internal — shared helpers                                            #
    # ------------------------------------------------------------------ #

    def _get_state(self, name: str) -> WorkflowState:
        try:
            return self._states[name]
        except KeyError:
            available = ", ".join(self._states) or "(none)"
            raise KeyError(f"State {name!r} not found. Registered: {available}") from None

    def _match_guard(self, state_name: str, output: str) -> str | None:
        for t in self._transitions:
            if t.source == state_name and t.guard is not None and t.guard(output):
                return t.target
        return None

    @staticmethod
    def _format(template: str, input_text: str, output_text: str) -> str:
        return template.format_map({"input": input_text, "output": output_text})

    @staticmethod
    def _execute(chorus: pykka.ActorRef | dict[str, pykka.ActorRef], instruction: str) -> str:
        """Invoke one actor or a dict of parallel actors with *instruction*."""
        if isinstance(chorus, dict):
            futures = {name: ref.proxy().instruct(instruction) for name, ref in chorus.items()}
            results = {name: f.get() for name, f in futures.items()}
            return "\n".join(f"{n}: {r}" for n, r in results.items())
        return chorus.proxy().instruct(instruction).get()
