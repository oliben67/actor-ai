"""04 – Tool Calling

Methods decorated with @tool are exposed to the LLM as callable functions.
The LLM can invoke them mid-generation; the actor dispatches the call and
feeds the result back to the LLM to compose its final reply.

This example shows:
  - @tool with a docstring description
  - @tool("explicit description") override
  - Tools with typed parameters (exposed as JSON Schema)
  - Optional tool parameters
  - Inspecting the generated tool specs
  - Simulating a tool-call round-trip with ToolCallingProvider

Real-LLM swap
-------------
    from actor_ai import Claude
    provider = Claude()        # Claude natively supports tool_use
"""

# Future imports (must occur at the beginning of the file):
from __future__ import annotations

# Standard library imports:
import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Third party imports:
from fake_provider import ToolCallingProvider

# Local imports:
from actor_ai import AIActor, tool
from actor_ai.tools import extract_tools

# ── Actor with several @tool methods ──────────────────────────────────────


class Calculator(AIActor):
    """An actor that exposes arithmetic and utility tools to the LLM."""

    system_prompt = (
        "You are a calculator assistant. "
        "Always use the provided tools for every computation; never guess."
    )

    @tool("Add two integers and return their sum.")
    def add(self, x: int, y: int) -> int:
        return x + y

    @tool("Multiply two numbers and return the product.")
    def multiply(self, x: float, y: float) -> float:
        return x * y

    @tool
    def power(self, base: float, exponent: float) -> float:
        """Raise base to the power of exponent."""
        return base**exponent

    @tool("Return the square root of a non-negative number.")
    def sqrt(self, n: float) -> float:
        return n**0.5

    @tool("Return the current UTC date-time as an ISO-8601 string.")
    def current_time(self) -> str:
        return datetime.datetime.now(datetime.UTC).isoformat()

    def list_tools(self) -> list[str]:
        """Return a human-readable summary of every @tool method."""
        result = []
        for spec in extract_tools(self):
            props = spec["input_schema"]["properties"]
            req = set(spec["input_schema"]["required"])
            params = ", ".join(
                f"{k}: {v['type']}" + ("" if k in req else "?") for k, v in props.items()
            )
            result.append(f"{spec['name']}({params}) — {spec['description']}")
        return result


def _divider(title: str) -> None:
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print(f"{'─' * 55}")


def _demo(question: str, tool_name: str, tool_args: dict, expected_reply: str) -> None:
    """Spin up a Calculator with a ToolCallingProvider, run one instruction."""
    prov = ToolCallingProvider(
        tool_name=tool_name,
        tool_args=tool_args,
        final_reply=expected_reply,
    )

    class DemoCalc(AIActor):
        system_prompt = Calculator.system_prompt
        provider = prov

        @tool("Add two integers and return their sum.")
        def add(self, x: int, y: int) -> int:
            return x + y

        @tool("Multiply two numbers and return the product.")
        def multiply(self, x: float, y: float) -> float:
            return x * y

        @tool
        def power(self, base: float, exponent: float) -> float:
            """Raise base to the power of exponent."""
            return base**exponent

        @tool("Return the square root of a non-negative number.")
        def sqrt(self, n: float) -> float:
            return n**0.5

        @tool("Return the current UTC date-time as an ISO-8601 string.")
        def current_time(self) -> str:
            return datetime.datetime.now(datetime.UTC).isoformat()

    ref = DemoCalc.start()
    try:
        reply = ref.proxy().instruct(question).get()
        print(f"  Question    : {question}")
        print(f"  Tool called : {tool_name}({tool_args!r})")
        print(f"  Tool result : {prov.tool_result}")
        print(f"  LLM reply   : {reply}")
    finally:
        ref.stop()


def main() -> None:
    # ── Show generated tool specs ──────────────────────────────────────
    _divider("Generated tool specs (sent to the LLM)")

    ref = Calculator.start()
    try:
        tools = ref.proxy().list_tools().get()
        for t in tools:
            print(f"  {t}")
    finally:
        ref.stop()

    # ── Tool-call demos ────────────────────────────────────────────────
    _divider("add(x=15, y=27)")
    _demo("What is 15 + 27?", "add", {"x": 15, "y": 27}, "15 + 27 = 42")

    _divider("multiply(x=6.0, y=7.0)")
    _demo("What is 6 times 7?", "multiply", {"x": 6.0, "y": 7.0}, "6 × 7 = 42.0")

    _divider("power(base=2.0, exponent=10.0)")
    _demo(
        "What is 2 to the power of 10?",
        "power",
        {"base": 2.0, "exponent": 10.0},
        "2¹⁰ = 1024.0",
    )

    _divider("sqrt(n=144.0)")
    _demo("What is the square root of 144?", "sqrt", {"n": 144.0}, "√144 = 12.0")

    _divider("current_time() — no parameters")
    _demo(
        "What time is it now?",
        "current_time",
        {},
        "It is currently [current UTC time].",
    )


if __name__ == "__main__":
    main()
