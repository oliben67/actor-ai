"""14 – Workflow: parallel actors and async execution

Demonstrates parallel actor states and non-blocking execution:
  - Parallel actors in a state: dict-form chorus fires all actors simultaneously
  - Guard on combined output: transition fires on a keyword from one parallel actor
  - Mixed states: single-actor state followed by parallel-actor state
  - run_detached() with on_complete callback: non-blocking execution
  - run_detached() with on_error callback: error handling in background runs
  - event() while run_detached() is running: actor mailbox stays free
  - prepare_step() / commit_step() for custom orchestration

Real-LLM swap
-------------
Replace ScriptedProvider with:
    from actor_ai import Claude, GPT
    researcher_provider = Claude()      # requires ANTHROPIC_API_KEY
    critic_provider     = GPT("gpt-4o") # requires OPENAI_API_KEY
"""

# Future imports (must occur at the beginning of the file):
from __future__ import annotations

# Standard library imports:
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Third party imports:
from fake_provider import ScriptedProvider

# Local imports:
from actor_ai import AIActor, Chorus, Workflow, WorkflowState, WorkflowTransition

# ── Agent factories ────────────────────────────────────────────────────────
# Each section gets its own actor instances so scripted-reply indices are
# independent.  Real-LLM examples would simply reuse the same provider.


def _make_researcher(replies: list[str]) -> type:
    """Return a fresh AIActor subclass backed by its own ScriptedProvider."""
    return type(
        f"Researcher_{id(replies)}",
        (AIActor,),
        {
            "system_prompt": "You are a researcher. Analyse and summarise findings.",
            "provider": ScriptedProvider(replies),
        },
    )


def _make_critic(replies: list[str]) -> type:
    return type(
        f"Critic_{id(replies)}",
        (AIActor,),
        {
            "system_prompt": "You are a critical reviewer. Identify risks and gaps.",
            "provider": ScriptedProvider(replies),
        },
    )


def _make_summariser(replies: list[str]) -> type:
    return type(
        f"Summariser_{id(replies)}",
        (AIActor,),
        {
            "system_prompt": "You are a summariser. Produce a concise executive summary.",
            "provider": ScriptedProvider(replies),
        },
    )


def _make_gatekeeper(replies: list[str]) -> type:
    return type(
        f"Gatekeeper_{id(replies)}",
        (AIActor,),
        {
            "system_prompt": "You are a gatekeeper. Confirm the recommendation or reject it.",
            "provider": ScriptedProvider(replies),
        },
    )


# ── Helpers ────────────────────────────────────────────────────────────────


def _divider(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:  # noqa: PLR0915 (many sections)
    # Standard library imports:
    import contextlib

    # Third party imports:
    import pykka

    # ── 1. Parallel actors in a state ─────────────────────────────────────
    _divider("1. Parallel actors in a single state (dict form)")

    r1 = _make_researcher(
        ["Researcher: The data shows strong positive signals — approved for next phase."]
    ).start()
    c1 = _make_critic(
        ["Critic: Some gaps remain but nothing blocking. Risks are manageable."]
    ).start()

    wf1_ref = Workflow.start(
        states={
            "analyse": WorkflowState(
                chorus={"researcher": r1, "critic": c1},
                instruction="Analyse: {input}",
            ),
        },
        initial_state="analyse",
    )
    try:
        combined = wf1_ref.proxy().run("Should we proceed with the new feature?").get()
        print("  Combined reply from parallel actors:")
        for line in combined.splitlines():
            print(f"    {line}")
    finally:
        wf1_ref.stop()
        r1.stop()
        c1.stop()

    # ── 2. Guard on combined output ────────────────────────────────────────
    _divider("2. Guard transition fires on keyword in combined parallel output")

    r2 = _make_researcher(["Researcher: Analysis done — approved for the next phase."]).start()
    c2 = _make_critic(["Critic: Risks are acceptable. No blockers identified."]).start()
    s2 = _make_summariser(
        ["Summary: Positive signals; manageable risks — recommend proceeding."]
    ).start()
    chorus_s2 = Chorus.start(agents={"summariser": s2})

    wf2_ref = Workflow.start(
        states={
            "analyse": WorkflowState(
                chorus={"researcher": r2, "critic": c2},
                instruction="Analyse: {input}",
            ),
            "summarise": WorkflowState(
                chorus=chorus_s2,
                instruction="Summarise the analysis:\n{output}",
            ),
        },
        transitions=[
            # "approved" appears in the Researcher's scripted reply
            WorkflowTransition("analyse", "summarise", guard=lambda r: "approved" in r),
        ],
        initial_state="analyse",
    )
    try:
        output2 = wf2_ref.proxy().run("Evaluate the Q3 roadmap.").get()
        print(f"  state after run()  : {wf2_ref.proxy().current_state().get()!r}")
        print(f"  final output       : {output2}")
    finally:
        wf2_ref.stop()
        chorus_s2.stop()
        r2.stop()
        c2.stop()
        s2.stop()

    # ── 3. Mixed states: single then parallel ──────────────────────────────
    _divider("3. Mixed states — single actor, then parallel actors")

    s3 = _make_summariser(["Summary: Last quarter showed steady growth in all metrics."]).start()
    r3 = _make_researcher(
        ["Researcher: Detailed analysis confirms growth is driven by new markets."]
    ).start()
    c3 = _make_critic(["Critic: Watch out for margin compression — warrants monitoring."]).start()
    chorus_s3 = Chorus.start(agents={"summariser": s3})

    wf3_ref = Workflow.start(
        states={
            "summarise": WorkflowState(chorus=chorus_s3, instruction="{input}"),
            "analyse": WorkflowState(
                chorus={"researcher": r3, "critic": c3},
                instruction="Deep-dive on: {output}",
            ),
        },
        transitions=[
            WorkflowTransition("summarise", "analyse", guard=lambda r: True),
        ],
        initial_state="summarise",
    )
    try:
        output3 = wf3_ref.proxy().run("Provide an overview of last quarter.").get()
        print(f"  final state  : {wf3_ref.proxy().current_state().get()!r}")
        print("  final output (combined parallel):")
        for line in output3.splitlines():
            print(f"    {line}")
    finally:
        wf3_ref.stop()
        chorus_s3.stop()
        s3.stop()
        r3.stop()
        c3.stop()

    # ── 4. run_detached() with on_complete callback ────────────────────────
    _divider("4. run_detached() with on_complete — non-blocking execution")

    class DetachedWorker(AIActor):
        system_prompt = "You are a background processing agent."
        provider = ScriptedProvider(["Background task complete — output ready."])

    worker4 = DetachedWorker.start()
    chorus4 = Chorus.start(agents={"worker": worker4})
    wf4_ref = Workflow.start(
        states={"process": WorkflowState(chorus4, instruction="{input}")},
        initial_state="process",
    )
    wf4 = wf4_ref.proxy()

    result_holder: list[str] = []
    done_event = threading.Event()

    def on_complete(output: str) -> None:
        result_holder.append(output)
        done_event.set()

    # Fire run_detached — returns immediately
    wf4.run_detached("Process this data in the background.", on_complete=on_complete).get()
    print("  run_detached() returned immediately — doing other work …")
    print("  (main thread is free to do other things here)")

    completed = done_event.wait(timeout=10)
    print(f"  callback received  : {completed}")
    print(f"  background output  : {result_holder[0] if result_holder else '(none)'}")
    wf4_ref.stop()
    chorus4.stop()
    worker4.stop()

    # ── 5. run_detached() with on_error callback ───────────────────────────
    _divider("5. run_detached() with on_error — background error handling")

    # Pointing to a non-existent initial state triggers a KeyError in the thread
    wf5_ref = Workflow.start(initial_state="nonexistent_state")

    error_holder: list[Exception] = []
    error_event = threading.Event()

    def on_error(exc: Exception) -> None:
        error_holder.append(exc)
        error_event.set()

    wf5_ref.proxy().run_detached("trigger error", on_error=on_error).get()
    error_caught = error_event.wait(timeout=10)
    print(f"  on_error callback received : {error_caught}")
    print(
        f"  exception type             : "
        f"{type(error_holder[0]).__name__ if error_holder else '(none)'}"
    )
    wf5_ref.stop()

    # ── 6. event() while run_detached() is in progress ────────────────────
    _divider("6. event() fired while run_detached() is running")

    # Two states connected only by an event transition ("advance").
    # run_detached starts in "waiting" — the state executes once, then the
    # event fires (actor mailbox is free) and the workflow advances.

    class WaitingActor(AIActor):
        system_prompt = "You are a waiting actor."
        provider = ScriptedProvider(["Waiting for event."])

    class AdvancedActor(AIActor):
        system_prompt = "You are an advanced actor."
        provider = ScriptedProvider(["Advanced state reached."])

    waiting_ref = WaitingActor.start()
    advanced_ref = AdvancedActor.start()
    waiting_chorus = Chorus.start(agents={"waiter": waiting_ref})
    advanced_chorus = Chorus.start(agents={"advancer": advanced_ref})

    wf6_ref = Workflow.start(
        states={
            "waiting": WorkflowState(waiting_chorus, instruction="{input}"),
            "advanced": WorkflowState(advanced_chorus, instruction="{input}"),
        },
        transitions=[
            WorkflowTransition("waiting", "advanced", on_event="advance"),
        ],
        initial_state="waiting",
    )
    wf6 = wf6_ref.proxy()

    # Use a small barrier so the event fires after run_detached has started
    # but we don't need to sleep — just fire it immediately; the actor mailbox
    # accepts both messages concurrently.
    done6 = threading.Event()
    wf6.run_detached("Start and wait.", on_complete=lambda _: done6.set()).get()

    # Actor mailbox is free — fire an event while the run is in progress
    fired6 = wf6.event("advance").get()
    print(f"  event('advance') fired during detached run : {fired6}")

    done6.wait(timeout=10)
    print(f"  final state after detached run : {wf6.current_state().get()!r}")
    wf6_ref.stop()
    waiting_chorus.stop()
    advanced_chorus.stop()
    waiting_ref.stop()
    advanced_ref.stop()

    # ── 7. prepare_step() / commit_step() custom orchestration ────────────
    _divider("7. prepare_step() / commit_step() — custom orchestration")

    class CustomWorker(AIActor):
        system_prompt = "You are a custom worker."
        provider = ScriptedProvider(["Custom step complete — ready for gate."])

    class GatekeeperActor(AIActor):
        system_prompt = "You are a gatekeeper."
        provider = ScriptedProvider(["Gatekeeper: cleared to proceed."])

    custom_worker = CustomWorker.start()
    gatekeeper7 = GatekeeperActor.start()
    custom_chorus = Chorus.start(agents={"worker": custom_worker})
    gate_chorus = Chorus.start(agents={"gatekeeper": gatekeeper7})

    wf7_ref = Workflow.start(
        states={
            "work": WorkflowState(custom_chorus, instruction="{input}"),
            "gate": WorkflowState(gate_chorus, instruction="Gate-check:\n{output}"),
        },
        transitions=[
            WorkflowTransition("work", "gate", guard=lambda r: True),
        ],
        initial_state="work",
    )
    wf7 = wf7_ref.proxy()

    print(f"  current state before manual step : {wf7.current_state().get()!r}")

    # Phase 1: prepare — atomically reads current state, formats instruction
    action = wf7.prepare_step("Custom orchestration input.").get()
    chorus_obj, formatted_instruction = action
    print(f"  prepare_step() formatted instruction : {formatted_instruction!r}")

    # Phase 2: execute — call the chorus directly outside the workflow thread
    output7 = Workflow._execute(chorus_obj, formatted_instruction)
    print(f"  direct execute output : {output7}")

    # Phase 3: commit — store output and advance state machine
    advanced7 = wf7.commit_step(output7).get()
    print(f"  commit_step() advanced state : {advanced7}")
    print(f"  current state after commit   : {wf7.current_state().get()!r}")
    print(f"  last_output() after commit   : {wf7.last_output().get()}")

    wf7_ref.stop()
    custom_chorus.stop()
    gate_chorus.stop()
    with contextlib.suppress(pykka.ActorDeadError):
        custom_worker.stop()
    with contextlib.suppress(pykka.ActorDeadError):
        gatekeeper7.stop()


if __name__ == "__main__":
    main()
