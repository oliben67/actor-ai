"""13 – Workflow: state machine orchestration

Demonstrates core Workflow features:
  - Basic run(): guard transition fires automatically when reply matches
  - step(): manual stepping through states, current_state() changes
  - event(): explicit named transitions independent of reply content
  - Both guard and event transitions on the same state
  - Runtime modification: add_state() / add_transition() after workflow starts
  - set_state() to force-jump to any registered state
  - current_state() and last_output() inspection at any point

Real-LLM swap
-------------
Replace ScriptedProvider with:
    from actor_ai import Claude
    provider = Claude()          # requires ANTHROPIC_API_KEY
"""

# Future imports (must occur at the beginning of the file):
from __future__ import annotations

# Standard library imports:
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Third party imports:
from fake_provider import ScriptedProvider

# Local imports:
from actor_ai import AIActor, Chorus, Workflow, WorkflowState, WorkflowTransition

# ── Agent definitions ──────────────────────────────────────────────────────


class WFDrafter(AIActor):
    system_prompt = "You are a document drafter."
    provider = ScriptedProvider(["Here is the draft — it is ready for review."])


class WFReviewer(AIActor):
    system_prompt = "You are a document reviewer."
    provider = ScriptedProvider(["Review complete. The document looks good."])


class WFApprover(AIActor):
    system_prompt = "You are a final approver."
    provider = ScriptedProvider(["Approved and signed off."])


class WFStepper(AIActor):
    """Used for step()-by-step examples; has two distinct replies."""

    system_prompt = "You are a multi-step assistant."
    provider = ScriptedProvider(["Step 1 complete.", "Step 2 complete."])


class WFEventActor(AIActor):
    """Used for event-transition examples."""

    system_prompt = "You are an event-driven assistant."
    provider = ScriptedProvider(["Draft submitted.", "Rework done — ready again."])


class WFBothActor(AIActor):
    """State that has both a guard and an event transition."""

    system_prompt = "You are an analysis assistant."
    provider = ScriptedProvider(["Analysis done — approved.", "Analysis done — approved."])


class WFRuntimeActor(AIActor):
    """Used for the runtime-modification example."""

    system_prompt = "You are an initial-state assistant."
    provider = ScriptedProvider(["First pass complete — needs polish."])


class WFPolishActor(AIActor):
    """Added at runtime as a second state."""

    system_prompt = "You are a polish assistant."
    provider = ScriptedProvider(["Polished and publication-ready."])


# ── Helpers ────────────────────────────────────────────────────────────────


def _divider(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    # Start all actor refs
    drafter_ref = WFDrafter.start()
    reviewer_ref = WFReviewer.start()
    approver_ref = WFApprover.start()
    stepper_ref = WFStepper.start()
    event_ref = WFEventActor.start()
    both_ref = WFBothActor.start()
    runtime_ref = WFRuntimeActor.start()
    polish_ref = WFPolishActor.start()

    # Wrap plain actor refs in single-actor Choruses so Workflow can call them
    drafter_chorus = Chorus.start(agents={"drafter": drafter_ref})
    reviewer_chorus = Chorus.start(agents={"reviewer": reviewer_ref})
    approver_chorus = Chorus.start(agents={"approver": approver_ref})
    stepper_chorus = Chorus.start(agents={"stepper": stepper_ref})
    event_chorus = Chorus.start(agents={"event_actor": event_ref})
    both_chorus = Chorus.start(agents={"both_actor": both_ref})
    runtime_chorus = Chorus.start(agents={"runtime_actor": runtime_ref})
    polish_chorus = Chorus.start(agents={"polish_actor": polish_ref})

    try:
        # ── 1. Basic run() with guard transition ──────────────────────────
        _divider("1. Basic run() — guard fires when reply contains 'ready'")

        wf1_ref = Workflow.start(
            states={
                "draft": WorkflowState(drafter_chorus, instruction="{input}"),
                "review": WorkflowState(reviewer_chorus, instruction="Review this:\n{output}"),
            },
            transitions=[
                WorkflowTransition("draft", "review", guard=lambda r: "ready" in r),
            ],
            initial_state="draft",
        )
        wf1 = wf1_ref.proxy()

        print(f"  initial state : {wf1.current_state().get()!r}")
        output = wf1.run("Please write a short proposal.").get()
        print(f"  final state   : {wf1.current_state().get()!r}")
        print(f"  final output  : {output}")
        wf1_ref.stop()

        # ── 2. step() manual stepping ─────────────────────────────────────
        _divider("2. step() — manual stepping, current_state() changes")

        wf2_ref = Workflow.start(
            states={
                "step_a": WorkflowState(stepper_chorus, instruction="{input}"),
                "step_b": WorkflowState(stepper_chorus, instruction="{output}"),
            },
            transitions=[
                WorkflowTransition("step_a", "step_b", guard=lambda r: "complete" in r),
            ],
            initial_state="step_a",
        )
        wf2 = wf2_ref.proxy()

        print(f"  state before step 1 : {wf2.current_state().get()!r}")
        out1 = wf2.step("Begin the process.").get()
        print(f"  state after  step 1 : {wf2.current_state().get()!r}  | output: {out1}")
        out2 = wf2.step().get()
        print(f"  state after  step 2 : {wf2.current_state().get()!r}  | output: {out2}")
        wf2_ref.stop()

        # ── 3. event() transition ─────────────────────────────────────────
        _divider("3. event() — named event sends workflow back from review to draft")

        wf3_ref = Workflow.start(
            states={
                "draft": WorkflowState(event_chorus, instruction="{input}"),
                "review": WorkflowState(reviewer_chorus, instruction="Review:\n{output}"),
            },
            transitions=[
                WorkflowTransition("draft", "review", guard=lambda r: "submitted" in r),
                WorkflowTransition("review", "draft", on_event="reject"),
            ],
            initial_state="draft",
        )
        wf3 = wf3_ref.proxy()

        # Run forward: draft → review (guard fires on "submitted")
        wf3.run("Write the initial draft.").get()
        print(f"  state after run()         : {wf3.current_state().get()!r}")

        # Fire a named event to go back to draft
        fired = wf3.event("reject").get()
        print(f"  event('reject') fired     : {fired}")
        print(f"  state after reject event  : {wf3.current_state().get()!r}")

        # Run again from draft (second scripted reply)
        wf3.run("Revise the draft.").get()
        print(f"  state after second run()  : {wf3.current_state().get()!r}")
        wf3_ref.stop()

        # ── 4. Both guard and event transitions defined ───────────────────
        _divider("4. Both guard and event transitions on the same state")

        wf4_ref = Workflow.start(
            states={
                "analyse": WorkflowState(both_chorus, instruction="{input}"),
                "approve": WorkflowState(approver_chorus, instruction="Finalise:\n{output}"),
                "escalate": WorkflowState(reviewer_chorus, instruction="Escalate:\n{output}"),
            },
            transitions=[
                WorkflowTransition("analyse", "approve", guard=lambda r: "approved" in r),
                WorkflowTransition("analyse", "escalate", on_event="escalate"),
            ],
            initial_state="analyse",
        )
        wf4 = wf4_ref.proxy()

        # Path 1: guard fires → analyse → approve
        output4 = wf4.run("Analyse the situation.").get()
        print(f"  guard path — final state  : {wf4.current_state().get()!r}")
        print(f"  guard path — final output : {output4}")

        # Path 2: reset to analyse, fire event instead → escalate
        wf4.set_state("analyse").get()
        wf4.event("escalate").get()
        print(f"  event path — state after escalate event : {wf4.current_state().get()!r}")
        wf4_ref.stop()

        # ── 5. Runtime modification: add_state() / add_transition() ───────
        _divider("5. Runtime modification — add state and transition after start")

        wf5_ref = Workflow.start(
            states={
                "draft": WorkflowState(runtime_chorus, instruction="{input}"),
            },
            transitions=[],
            initial_state="draft",
        )
        wf5 = wf5_ref.proxy()

        print(f"  registered states before add : {wf5.states().get()}")

        # Add a new state and transition at runtime
        wf5.add_state("polish", WorkflowState(polish_chorus, instruction="Polish:\n{output}")).get()
        wf5.add_transition(
            WorkflowTransition("draft", "polish", guard=lambda r: "polish" in r)
        ).get()

        print(f"  registered states after add  : {wf5.states().get()}")

        output5 = wf5.run("Write a first draft.").get()
        print(f"  final state   : {wf5.current_state().get()!r}")
        print(f"  final output  : {output5}")
        wf5_ref.stop()

        # ── 6. set_state() to force-jump ──────────────────────────────────
        _divider("6. set_state() — force-jump to any registered state")

        wf6_ref = Workflow.start(
            states={
                "draft": WorkflowState(drafter_chorus, instruction="{input}"),
                "review": WorkflowState(reviewer_chorus, instruction="{output}"),
                "approve": WorkflowState(approver_chorus, instruction="{output}"),
            },
            initial_state="draft",
        )
        wf6 = wf6_ref.proxy()

        print(f"  current state before set_state : {wf6.current_state().get()!r}")
        wf6.set_state("approve").get()
        print(f"  current state after set_state  : {wf6.current_state().get()!r}")

        output6 = wf6.run("Give final approval.").get()
        print(f"  output from 'approve' state    : {output6}")
        wf6_ref.stop()

        # ── 7. current_state() and last_output() inspection ───────────────
        _divider("7. current_state() and last_output() inspection")

        wf7_ref = Workflow.start(
            states={
                "start": WorkflowState(drafter_chorus, instruction="{input}"),
            },
            initial_state="start",
        )
        wf7 = wf7_ref.proxy()

        print(f"  last_output() before run : {wf7.last_output().get()!r}")
        wf7.run("Draft something brief.").get()
        print(f"  current_state() after run : {wf7.current_state().get()!r}")
        print(f"  last_output() after run   : {wf7.last_output().get()}")
        wf7_ref.stop()

    finally:
        for chorus in [
            drafter_chorus,
            reviewer_chorus,
            approver_chorus,
            stepper_chorus,
            event_chorus,
            both_chorus,
            runtime_chorus,
            polish_chorus,
        ]:
            # Standard library imports:
            import contextlib

            # Third party imports:
            import pykka

            with contextlib.suppress(pykka.ActorDeadError):
                chorus.stop()

        for ref in [
            drafter_ref,
            reviewer_ref,
            approver_ref,
            stepper_ref,
            event_ref,
            both_ref,
            runtime_ref,
            polish_ref,
        ]:
            # Standard library imports:
            import contextlib

            # Third party imports:
            import pykka

            with contextlib.suppress(pykka.ActorDeadError):
                ref.stop()


if __name__ == "__main__":
    main()
