"""02 – Session Management

AIActor keeps a rolling conversation history so each turn has full context.
This example shows:
  - Multi-turn conversation (session accumulation)
  - max_history: cap the window to N user+assistant turns
  - clear_session(): discard history and start a fresh session ID
  - use_session=False: one-shot query that never touches the session

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


class ChatBot(AIActor):
    system_prompt = "You are a friendly conversational assistant."
    provider = ScriptedProvider(
        [
            "Hi Alice! Great to meet you.",
            "Of course! You told me your name is Alice.",
            "Got it — noted for next time.",
            # after clear_session():
            "Hello! I'm afraid I don't know your name yet.",
            # use_session=False (stateless query):
            "Paris is the capital of France.",
        ]
    )
    max_history = 3  # keep at most 3 turns (6 messages) in memory


def _divider(title: str) -> None:
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")


def main() -> None:
    ref = ChatBot.start()
    proxy = ref.proxy()

    try:
        # ── Multi-turn conversation ──────────────────────────────────────
        _divider("Multi-turn conversation")

        r1 = proxy.instruct("Hi! My name is Alice.").get()
        print("[user]      Hi! My name is Alice.")
        print(f"[assistant] {r1}")

        r2 = proxy.instruct("Do you remember my name?").get()
        print("\n[user]      Do you remember my name?")
        print(f"[assistant] {r2}")

        r3 = proxy.instruct("Please remember I prefer short answers.").get()
        print("\n[user]      Please remember I prefer short answers.")
        print(f"[assistant] {r3}")

        # ── Inspect the session ──────────────────────────────────────────
        _divider("Current session")

        session = proxy.get_session().get()
        print(f"Session length : {len(session)} messages ({len(session) // 2} turns)")
        for msg in session:
            excerpt = msg["content"][:60] + ("…" if len(msg["content"]) > 60 else "")
            print(f"  [{msg['role']:9}] {excerpt}")

        # ── max_history in action ────────────────────────────────────────
        # The window is capped at max_history=3 turns; adding a 4th will
        # push the oldest turn out.  The session already has 3 turns, so
        # the next call will trim to 3.
        # (ScriptedProvider keeps returning the last reply here.)

        # ── clear_session ────────────────────────────────────────────────
        _divider("clear_session()")

        old_session_id = proxy.get_session_id().get()
        proxy.clear_session().get()
        new_session_id = proxy.get_session_id().get()

        r4 = proxy.instruct("Hello! Do you know who I am?").get()
        print(f"Old session ID : {old_session_id}")
        print(f"New session ID : {new_session_id}")
        print("[user]         Hello! Do you know who I am?")
        print(f"[assistant]    {r4}")

        # ── use_session=False (stateless one-shot) ───────────────────────
        _divider("use_session=False (stateless one-shot)")

        session_before = len(proxy.get_session().get())
        r5 = proxy.instruct("What is the capital of France?", use_session=False).get()
        session_after = len(proxy.get_session().get())

        print("[user]      What is the capital of France?")
        print(f"[assistant] {r5}")
        print(f"\nSession length before : {session_before}")
        print(f"Session length after  : {session_after}  (unchanged — stateless call)")

    finally:
        ref.stop()


if __name__ == "__main__":
    main()
