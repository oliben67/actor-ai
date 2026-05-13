"""01 – Hello World

Simplest possible usage: define an AIActor, start it, send one instruction,
print the reply, stop the actor.

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
from actor_ai import AIActor


class Assistant(AIActor):
    system_prompt = "You are a helpful assistant. Answer concisely."
    provider = ScriptedProvider(["The capital of France is Paris."])


def main() -> None:
    ref = Assistant.start()
    try:
        question = "What is the capital of France?"
        reply = ref.proxy().instruct(question).get()
        print(f"Question : {question}")
        print(f"Reply    : {reply}")
    finally:
        ref.stop()


if __name__ == "__main__":
    main()
