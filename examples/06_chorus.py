"""06 – Chorus

A Chorus groups named AIActors and lets you:
  - instruct(name, …)  — single agent
  - broadcast(…)       — all agents in parallel, returns {name: reply}
  - pipeline([…], …)   — chain agents; each receives the previous reply
  - remember/forget(…) — broadcast facts to one or all agents
  - stop_agents(…)     — shut down selected (or all) agents

Real-LLM swap
-------------
    from actor_ai import Claude, GPT
    class Researcher(AIActor):
        provider = Claude()
    class Writer(AIActor):
        provider = GPT("gpt-4o")
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
from actor_ai import AIActor, Chorus

# ── Define three specialist agents ────────────────────────────────────────


class Researcher(AIActor):
    system_prompt = "You are a research specialist. Provide concise factual summaries."
    provider = ScriptedProvider(
        [
            # broadcast reply
            "I am the Researcher — I gather and summarise facts.",
            # pipeline step 1: expand the brief
            "Mars is the fourth planet from the Sun. "
            "It has two moons (Phobos and Deimos), a thin CO₂ atmosphere, "
            "and surface temperatures ranging from −125 °C to 20 °C. "
            "NASA's Perseverance rover is currently exploring Jezero Crater.",
        ]
    )


class Writer(AIActor):
    system_prompt = "You are a creative science writer. Turn research notes into vivid prose."
    provider = ScriptedProvider(
        [
            # broadcast reply
            "I am the Writer — I craft engaging narratives from raw facts.",
            # pipeline step 2: transform into a paragraph
            "Beneath a rust-coloured sky, the iron plains of Mars stretch endlessly. "
            "Two tiny moons wheel overhead while Perseverance inches forward, "
            "its wheels etching history into Jezero's ancient lake bed.",
        ]
    )


class Reviewer(AIActor):
    system_prompt = "You are an editorial reviewer. Check tone, clarity, and accuracy."
    provider = ScriptedProvider(
        [
            # broadcast reply
            "I am the Reviewer — I ensure quality and consistency.",
            # pipeline step 3: final polish
            "Reviewed and approved. The paragraph is vivid, accurate, and publication-ready.",
        ]
    )


def _divider(title: str) -> None:
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print(f"{'─' * 55}")


def main() -> None:
    researcher_ref = Researcher.start()
    writer_ref = Writer.start()
    reviewer_ref = Reviewer.start()

    chorus_ref = Chorus.start(
        agents={
            "researcher": researcher_ref,
            "writer": writer_ref,
            "reviewer": reviewer_ref,
        }
    )
    chorus = chorus_ref.proxy()

    try:
        # ── List agents ────────────────────────────────────────────────
        _divider("Registered agents")
        print("  " + ", ".join(chorus.agents().get()))

        # ── Single agent instruct ──────────────────────────────────────
        _divider("instruct('researcher', …)")
        reply = chorus.instruct(
            "researcher",
            "Give me a brief overview of Mars.",
        ).get()
        print(f"  [researcher] {reply[:80]}…")

        # ── Broadcast ─────────────────────────────────────────────────
        _divider("broadcast('Introduce yourself in one sentence.')")
        replies = chorus.broadcast("Introduce yourself in one sentence.").get()
        for name, r in replies.items():
            print(f"  [{name:10}] {r}")

        # ── Pipeline ──────────────────────────────────────────────────
        _divider("pipeline(['researcher','writer','reviewer'], …)")
        final = chorus.pipeline(
            ["researcher", "writer", "reviewer"],
            "Summarise the latest Mars exploration news.",
        ).get()
        print(f"  Final output:\n  {final}")

        # ── Remember / Forget broadcast ───────────────────────────────
        _divider("remember() — broadcast to all agents")
        chorus.remember("audience", "general public").get()
        chorus.remember("max_length", "100 words").get()
        print("  Fact 'audience' and 'max_length' stored in all agents.")

        # Forget from a single agent only
        chorus.forget("max_length", names=["writer"]).get()
        print("  Fact 'max_length' forgotten by 'writer' only.")

        # ── Dynamic agent management ───────────────────────────────────
        _divider("add / remove agents")

        class Editor(AIActor):
            system_prompt = "You are a copy editor."
            provider = ScriptedProvider(["I am the Editor — I fix grammar and style."])

        editor_ref = Editor.start()
        chorus.add("editor", editor_ref).get()
        print(f"  Agents now: {', '.join(chorus.agents().get())}")

        chorus.remove("editor").get()
        editor_ref.stop()
        print(f"  Agents after remove: {', '.join(chorus.agents().get())}")

        # ── stop_agents ────────────────────────────────────────────────
        _divider("stop_agents(['reviewer'])")
        chorus.stop_agents(names=["reviewer"]).get()
        print(f"  Agents still running: {', '.join(chorus.agents().get())}")

    finally:
        chorus_ref.stop()
        # researcher_ref and writer_ref are still alive (chorus.stop() doesn't stop them)
        # Standard library imports:
        import contextlib

        # Third party imports:
        import pykka

        with contextlib.suppress(pykka.ActorDeadError):
            researcher_ref.stop()
        with contextlib.suppress(pykka.ActorDeadError):
            writer_ref.stop()


if __name__ == "__main__":
    main()
