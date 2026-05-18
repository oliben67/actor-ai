"""17 - GitHub Copilot SDK provider

This example uses ``Copilot(use_sdk=True)``, which keeps actor-ai's normal
synchronous provider interface while calling the GitHub Copilot SDK's async
session API under the hood.

Requirements:
  - GitHub Copilot CLI/SDK authentication available, or pass ``api_key=...``
  - A GitHub account with an active Copilot subscription

Run:
    uv run python examples/17_copilot_sdk.py
"""

# Future imports (must occur at the beginning of the file):
from __future__ import annotations

# Local imports:
from actor_ai import AIActor, Copilot, tool


class MathAssistant(AIActor):
    system_prompt = "You are a concise assistant. Use tools when they help."
    provider = Copilot("claude-sonnet-4.5", use_sdk=True, timeout=60.0)

    @tool
    def add(self, a: int, b: int) -> int:
        """Add two integers."""
        return a + b


if __name__ == "__main__":
    ref = MathAssistant.start()
    try:
        answer = ref.proxy().instruct("What is 21 + 21? Use the add tool.").get()
        print(answer)
    finally:
        ref.stop()
