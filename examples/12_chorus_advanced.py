"""12 – Chorus: type, join/leave, nested choruses, non-AI actors

Demonstrates advanced Chorus features:
  - ChorusType: create choruses with explicit type and inspect it
  - join(name, ref) / leave(name): dynamic membership changes
  - instruct(instruction) single-arg form: broadcasts to all, returns formatted string
  - Nested Chorus as member: inner Chorus inside outer Chorus
  - Non-AI pykka actor: plain actor participates alongside AI actors
  - on_receive memory propagation: outer.remember() reaches the inner agent

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
import pykka
from fake_provider import ScriptedProvider

# Local imports:
from actor_ai import AIActor, Chorus, ChorusType

# ── Agent definitions ──────────────────────────────────────────────────────


class DesignLead(AIActor):
    system_prompt = "You are a design lead. Respond concisely."
    provider = ScriptedProvider(
        [
            "Design lead here — focused on user experience.",
            "I approve the direction.",
        ]
    )


class DevLead(AIActor):
    system_prompt = "You are a development lead. Respond concisely."
    provider = ScriptedProvider(
        [
            "Dev lead here — focused on implementation.",
            "I confirm it is technically feasible.",
        ]
    )


class QALead(AIActor):
    system_prompt = "You are a QA lead. Respond concisely."
    provider = ScriptedProvider(
        [
            "QA lead here — focused on quality and testing.",
        ]
    )


class ProductManager(AIActor):
    system_prompt = "You are a product manager. Respond concisely."
    provider = ScriptedProvider(
        [
            "PM here — focused on product strategy.",
        ]
    )


# ── Non-AI pykka actor ─────────────────────────────────────────────────────


class StatusBotActor(pykka.ThreadingActor):
    """A plain pykka actor that echoes a canned status reply.

    It participates in a Chorus broadcast just like an AIActor — it only
    needs an ``instruct(instruction)`` method.
    """

    def instruct(self, instruction: str, **kwargs) -> str:  # noqa: ARG002
        return f"[STATUS BOT] All systems nominal. (received: {instruction!r})"


# ── Helpers ────────────────────────────────────────────────────────────────


def _divider(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    # Start all actor refs up front
    design_ref = DesignLead.start()
    dev_ref = DevLead.start()
    qa_ref = QALead.start()
    pm_ref = ProductManager.start()
    bot_ref = StatusBotActor.start()

    # ── 1. ChorusType ──────────────────────────────────────────────────────
    _divider("1. ChorusType — team and department")

    team_ref = Chorus.start(
        type="team",
        agents={"design": design_ref, "dev": dev_ref},
    )
    team = team_ref.proxy()

    dept_ref = Chorus.start(
        type="department",
        agents={"team": team_ref},
    )
    dept = dept_ref.proxy()

    print(f"  team chorus type  : {team.type.get()!r}")
    print(f"  dept chorus type  : {dept.type.get()!r}")

    # ── 2. join() / leave() ────────────────────────────────────────────────
    _divider("2. join() and leave()")

    print(f"  members before join : {team.agents().get()}")
    team.join("qa", qa_ref).get()
    print(f"  members after join  : {team.agents().get()}")

    team.leave("qa").get()
    print(f"  members after leave : {team.agents().get()}")
    # qa_ref is still alive — leave() never stops the actor
    print(f"  qa actor alive      : {pykka.ActorRegistry.get_by_urn(qa_ref.actor_urn) is not None}")

    # ── 3. instruct(instruction) — single-arg broadcast form ──────────────
    _divider("3. instruct(instruction) — single-arg broadcast")

    result = team.instruct("Introduce yourself in one sentence.").get()
    print("  Formatted output (name: reply per line):")
    for line in result.splitlines():
        print(f"    {line}")

    # ── 4. Nested Chorus as member ─────────────────────────────────────────
    _divider("4. Nested Chorus as member of outer Chorus")

    inner_ref = Chorus.start(
        type="team",
        agents={"design": design_ref, "dev": dev_ref},
    )
    outer_ref = Chorus.start(
        type="department",
        agents={"engineering_team": inner_ref, "pm": pm_ref},
    )
    outer = outer_ref.proxy()

    broadcast_result = outer.broadcast("Give a one-line status update.").get()
    print("  outer.broadcast() results:")
    for name, reply in broadcast_result.items():
        print(f"    [{name}] {reply}")

    # ── 5. Non-AI pykka actor in a Chorus ─────────────────────────────────
    _divider("5. Non-AI pykka actor participates in broadcast")

    mixed_ref = Chorus.start(
        agents={
            "ai_agent": design_ref,
            "status_bot": bot_ref,
        }
    )
    mixed = mixed_ref.proxy()

    mixed_result = mixed.broadcast("What is your current status?").get()
    print("  mixed broadcast (AI + plain actor):")
    for name, reply in mixed_result.items():
        print(f"    [{name:12}] {reply}")

    # ── 6. on_receive memory propagation through nested choruses ──────────
    _divider("6. on_receive — remember() propagates to inner agent")

    # Build a fresh inner agent so we can inspect its memory cleanly
    class InnerAgent(AIActor):
        system_prompt = "You are an assistant."
        provider = ScriptedProvider(["ready"])

    inner_agent_ref = InnerAgent.start()

    propagation_inner_ref = Chorus.start(
        type="team",
        agents={"member": inner_agent_ref},
    )
    propagation_outer_ref = Chorus.start(
        type="department",
        agents={"inner_team": propagation_inner_ref},
    )
    propagation_outer = propagation_outer_ref.proxy()

    # Broadcast a fact from the outer chorus
    propagation_outer.remember("project", "actor-ai").get()

    # Flush the inner chorus's mailbox (ensures the tell() has been processed)
    propagation_inner_ref.proxy().agents().get()

    # Now the fact should have reached the inner agent
    memory = inner_agent_ref.proxy().get_memory().get()
    print("  outer.remember('project', 'actor-ai') called")
    print(f"  inner_agent memory : {memory}")
    print(f"  propagated OK      : {memory.get('project') == 'actor-ai'}")

    # ── Cleanup ────────────────────────────────────────────────────────────
    try:
        team_ref.stop()
        dept_ref.stop()
        inner_ref.stop()
        outer_ref.stop()
        mixed_ref.stop()
        propagation_inner_ref.stop()
        propagation_outer_ref.stop()
    finally:
        # Standard library imports:
        import contextlib

        for ref in [design_ref, dev_ref, qa_ref, pm_ref, bot_ref, inner_agent_ref]:
            with contextlib.suppress(pykka.ActorDeadError):
                ref.stop()


if __name__ == "__main__":
    main()
