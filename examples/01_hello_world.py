"""01 – Hello World

Simplest possible usage: define an AIActor, start it, send one instruction,
print the reply, stop the actor.  Three patterns are shown:

1. Manual start / stop
2. get_proxy()  — synchronous context manager
3. aget_proxy() — async context manager (asyncio.to_thread for the blocking call)

Real-LLM swap
-------------
Replace ScriptedProvider with:
    from actor_ai import Claude
    provider = Claude()          # requires ANTHROPIC_API_KEY
"""

# Future imports (must occur at the beginning of the file):
from __future__ import annotations

# Standard library imports:
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Third party imports:
from fake_provider import ScriptedProvider

# Local imports:
from actor_ai import AIActor


class Assistant(AIActor):
    system_prompt = "You are a helpful assistant. Answer concisely."
    provider = ScriptedProvider([
        "The capital of France is Paris.",
        "The capital of Germany is Berlin.",
        "The capital of Spain is Madrid.",
    ])


def main() -> None:
    # 1. Manual start / stop
    ref = Assistant.start()
    try:
        question = "What is the capital of France?"
        reply = ref.proxy().instruct(question).get()
        print(f"Question : {question}")
        print(f"Reply    : {reply}")
    finally:
        ref.stop()

    # 2. get_proxy() — synchronous context manager
    question = "What is the capital of Germany?"
    with Assistant.get_proxy() as proxy:
        reply = proxy.instruct(question).get()
    print(f"Question : {question}")
    print(f"Reply    : {reply}")

    # 3. aget_proxy() — async context manager
    asyncio.run(_async_example())


async def _async_example() -> None:
    question = "What is the capital of Spain?"
    async with Assistant.aget_proxy() as proxy:
        # pykka proxy calls are blocking; use asyncio.to_thread to avoid
        # blocking the event loop
        reply = await asyncio.to_thread(proxy.instruct(question).get)
    print(f"Question : {question}")
    print(f"Reply    : {reply}")


if __name__ == "__main__":
    main()
