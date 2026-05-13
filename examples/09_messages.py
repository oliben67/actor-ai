"""09 – Message API

Every AIActor is a pykka ThreadingActor, so besides the proxy API you can
send raw message objects with:

  ref.ask(message)        — synchronous, blocks and returns the reply
  ref.tell(message)       — fire-and-forget, no return value

Exported message types:
  Instruct(instruction, history=[], use_session=True)
  Remember(key, value)
  Forget(key)

This example shows both the message API and the equivalent proxy calls side
by side so you can choose the style that fits your code.
"""

# Future imports (must occur at the beginning of the file):
from __future__ import annotations

# Standard library imports:
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Third party imports:
from fake_provider import ScriptedProvider

# Local imports:
from actor_ai import AIActor, Forget, Instruct, Remember


class Assistant(AIActor):
    system_prompt = "You are a concise assistant."
    provider = ScriptedProvider(
        [
            "The speed of light is approximately 299,792,458 m/s.",
            "You are Alice, a physicist.",
            "Your name is Alice.",
        ]
    )


def _divider(title: str) -> None:
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print(f"{'─' * 55}")


def main() -> None:
    ref = Assistant.start()

    try:
        # ── Instruct via ask() ────────────────────────────────────────
        _divider("Instruct via ref.ask()")

        reply = ref.ask(Instruct("What is the speed of light?"))
        print(f"  [ask → Instruct] {reply}")

        # Equivalent proxy form:
        # reply = ref.proxy().instruct("What is the speed of light?").get()

        # ── Remember via tell() ───────────────────────────────────────
        _divider("Remember via ref.tell()")

        ref.tell(Remember("name", "Alice"))
        ref.tell(Remember("occupation", "physicist"))

        # tell() is fire-and-forget; give the actor time to process before
        # we query memory. In practice, the next ask() will wait in the queue
        # *after* the tell()s, so ordering is guaranteed.

        reply2 = ref.ask(Instruct("Who am I?"))
        print(f"  [ask → Instruct after tell → Remember] {reply2}")

        # ── Inspect memory after tell(Remember) ───────────────────────
        _divider("Memory state after tell(Remember)")

        mem = ref.proxy().get_memory().get()
        print(f"  Memory: {mem}")

        # ── Forget via tell() ─────────────────────────────────────────
        _divider("Forget via ref.tell()")

        ref.tell(Forget("occupation"))
        mem_after = ref.proxy().get_memory().get()
        print(f"  Memory after forgetting 'occupation': {mem_after}")

        # ── Instruct with explicit history (no session) ───────────────
        _divider("Instruct with explicit history")

        explicit_history = [
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Nice to meet you, Alice!"},
        ]
        reply3 = ref.ask(
            Instruct(
                "What is my name?",
                history=explicit_history,
                use_session=False,
            )
        )
        print(f"  [explicit history, no session] {reply3}")

        # ── use_session=False via Instruct ────────────────────────────
        _divider("use_session=False — stateless one-shot")

        session_before = ref.proxy().get_session().get()
        ref.ask(Instruct("One quick stateless question.", use_session=False))
        session_after = ref.proxy().get_session().get()

        print(f"  Session length before: {len(session_before)}")
        print(f"  Session length after : {len(session_after)}  (unchanged)")

        # ── Proxy API equivalent summary ──────────────────────────────
        _divider("Proxy API equivalents (reference)")

        print(
            "  ref.ask(Instruct('…'))       ↔  ref.proxy().instruct('…').get()\n"
            "  ref.tell(Remember('k', 'v')) ↔  ref.proxy().remember('k', 'v').get()\n"
            "  ref.tell(Forget('k'))        ↔  ref.proxy().forget('k').get()\n"
            "\n"
            "  tell() is fire-and-forget and never blocks.\n"
            "  ask() blocks until the actor returns a value.\n"
            "  proxy().method().get() is the most common async-friendly style."
        )

    finally:
        ref.stop()


if __name__ == "__main__":
    main()
