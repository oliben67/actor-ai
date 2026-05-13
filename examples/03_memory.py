"""03 – Long-Term Memory

Memory persists named facts *across* turns and survives clear_session().
Facts are automatically injected into the system prompt so the LLM sees them
on every call.

This example shows:
  - remember(key, value): store a fact
  - forget(key): remove a fact
  - get_memory(): inspect the current memory store
  - Facts visible even after clear_session()
  - Memory shared across sessions via the same actor instance

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
            # Q1: who am I?
            "You are Alice, a Software Engineer at Acme Corp.",
            # Q2: current task?
            "Your current task is the Q3 report, due this Friday.",
            # Q3: project?
            "Project Orion is your top priority right now.",
            # After forgetting 'name':
            "I know you work at Acme Corp as a Software Engineer, "
            "but I don't have your name on file.",
            # After clear_session (memory persists):
            "Welcome back! You're Alice from Acme Corp — Project Orion is your priority.",
        ]
    )


def _show_memory(proxy) -> None:
    mem = proxy.get_memory().get()
    if not mem:
        print("  (empty)")
    else:
        for k, v in mem.items():
            print(f"  {k!r:20} → {v!r}")


def _divider(title: str) -> None:
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")


def main() -> None:
    ref = PersonalAssistant.start()
    proxy = ref.proxy()

    try:
        # ── Store initial facts ──────────────────────────────────────────
        _divider("Store facts with remember()")

        proxy.remember("name", "Alice").get()
        proxy.remember("role", "Software Engineer").get()
        proxy.remember("company", "Acme Corp").get()

        print("Memory after initial setup:")
        _show_memory(proxy)

        r1 = proxy.instruct("Who am I?").get()
        print("\n[Q] Who am I?")
        print(f"[A] {r1}")

        # ── Add more facts ───────────────────────────────────────────────
        _divider("Add more facts")

        proxy.remember("current_task", "Q3 report due Friday").get()
        r2 = proxy.instruct("What is my current task?").get()
        print("[Q] What is my current task?")
        print(f"[A] {r2}")

        proxy.remember("priority_project", "Orion").get()
        r3 = proxy.instruct("Which project should I focus on?").get()
        print("\n[Q] Which project should I focus on?")
        print(f"[A] {r3}")

        # ── Forget a fact ────────────────────────────────────────────────
        _divider("forget() a single fact")

        proxy.forget("name").get()
        print("Memory after forgetting 'name':")
        _show_memory(proxy)

        r4 = proxy.instruct("Do you know my name?").get()
        print("\n[Q] Do you know my name?")
        print(f"[A] {r4}")

        # ── Memory survives clear_session() ──────────────────────────────
        _divider("Memory survives clear_session()")

        proxy.remember("name", "Alice").get()  # restore it
        old_sid = proxy.get_session_id().get()
        proxy.clear_session().get()
        new_sid = proxy.get_session_id().get()

        print(f"Session cleared: {old_sid[:8]}… → {new_sid[:8]}…")
        print("Memory after clear:")
        _show_memory(proxy)

        r5 = proxy.instruct("Hello! Remember me?").get()
        print("\n[Q] Hello! Remember me?")
        print(f"[A] {r5}")

    finally:
        ref.stop()


if __name__ == "__main__":
    main()
