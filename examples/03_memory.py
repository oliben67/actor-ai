"""03 – Memory tiers

actor-ai has three memory tiers:

  1. Short-term (session history)  — rolling conversation turns, trimmed by
     max_history.  Cleared by clear_session().

  2. Working memory                — task-scoped key/value facts injected into
     the system prompt as "Working memory:".  Cleared by clear_session() or
     clear_working_memory().  Use for ephemeral task context.

  3. Long-term memory              — durable key/value facts injected as
     "Known facts:".  Survives clear_session().  Use for user preferences and
     persistent context.

This example runs without an API key — the fake provider echoes preset replies.

Real-LLM swap
-------------
    from actor_ai import Claude
    provider = Claude()
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
from actor_ai import AIActor


class PersonalAssistant(AIActor):
    system_prompt = "You are a personal assistant. Always use the stored facts when replying."
    provider = ScriptedProvider(
        [
            # Long-term: who am I?
            "You are Alice, a Software Engineer at Acme Corp.",
            # Working: current task
            "Your current task is the Q3 report, due this Friday.",
            # Working: project context was cleared — no task context available
            "I see you are Alice at Acme Corp, but I have no task context right now.",
            # Long-term survives clear_session; working was reset
            "Welcome back, Alice from Acme Corp! What can I help you with?",
            # Working memory set again in the new session
            "You are reviewing the Orion roadmap draft.",
        ]
    )


def _show_memory(label: str, proxy) -> None:
    mem = proxy.get_memory().get()
    working = proxy.get_working_memory().get()
    print(f"\n  [{label}]")
    print(f"  Long-term  : {mem or '(empty)'}")
    print(f"  Working    : {working or '(empty)'}")


def _divider(title: str) -> None:
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print(f"{'─' * 55}")


def main() -> None:
    ref = PersonalAssistant.start()
    proxy = ref.proxy()

    try:
        # ── Tier 3: Long-term memory ─────────────────────────────────────
        _divider("Tier 3 — Long-term memory (remember / forget)")

        proxy.remember("name", "Alice").get()
        proxy.remember("role", "Software Engineer").get()
        proxy.remember("company", "Acme Corp").get()

        _show_memory("after remember()", proxy)

        r1 = proxy.instruct("Who am I?").get()
        print(f"\n  [Q] Who am I?\n  [A] {r1}")

        # ── Tier 2: Working memory ────────────────────────────────────────
        _divider("Tier 2 — Working memory (remember_working / forget_working)")

        proxy.remember_working("current_task", "Q3 report due Friday").get()
        proxy.remember_working("priority_project", "Orion").get()

        _show_memory("after remember_working()", proxy)

        r2 = proxy.instruct("What am I working on?").get()
        print(f"\n  [Q] What am I working on?\n  [A] {r2}")

        # Working memory can be cleared alone
        proxy.clear_working_memory().get()
        _show_memory("after clear_working_memory()", proxy)

        r3 = proxy.instruct("What is my current task?").get()
        print(f"\n  [Q] What is my current task?\n  [A] {r3}")

        # ── clear_session() — working reset, long-term survives ───────────
        _divider("clear_session() — working memory reset, long-term survives")

        proxy.remember_working("current_task", "reviewing Orion roadmap").get()
        old_sid = proxy.get_session_id().get()
        proxy.clear_session().get()
        new_sid = proxy.get_session_id().get()

        print(f"\n  Session: {old_sid[:8]}… → {new_sid[:8]}…")
        _show_memory("after clear_session()", proxy)

        r4 = proxy.instruct("Hello! Remember me?").get()
        print(f"\n  [Q] Hello! Remember me?\n  [A] {r4}")

        # Rebuild working memory in new session
        proxy.remember_working("current_task", "reviewing Orion roadmap draft").get()
        r5 = proxy.instruct("What's my task for today?").get()
        print(f"\n  [Q] What's my task for today?\n  [A] {r5}")

        _show_memory("final state", proxy)

        # ── Tier 1: Session history (short-term) ─────────────────────────
        _divider("Tier 1 — Session history (get_session)")

        session = proxy.get_session().get()
        print(f"\n  Current session turns: {len(session) // 2}")
        for i, msg in enumerate(session):
            role_label = "  [user]     " if msg["role"] == "user" else "  [assistant]"
            print(f"{role_label} {msg['content'][:60]!r}")

    finally:
        ref.stop()


if __name__ == "__main__":
    main()
