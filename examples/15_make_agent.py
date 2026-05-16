"""15 – make_agent(): Agent Factory

``make_agent()`` creates a fully configured AIActor subclass without writing a
class definition. It is the recommended way to define agents, especially when
you need many agents with different roles or when agents delegate to each other
as sub-agents.

Key patterns shown
------------------
1. Simple agents — role, prompt, provider in one call
2. Tools — plain callables injected without subclassing
3. Sub-agents — agents wired to other agents via @tool auto-delegation
4. Class-based equivalent — when you still need a class

Real-LLM swap
-------------
Replace ScriptedProvider / ToolCallingProvider with:
    from actor_ai import Claude, GPT
    provider = Claude()
    provider = GPT("gpt-4o")
"""

# Future imports (must occur at the beginning of the file):
from __future__ import annotations

# Standard library imports:
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Third party imports:
from fake_provider import ScriptedProvider, ToolCallingProvider

# Local imports:
from actor_ai import make_agent, tool

# ---------------------------------------------------------------------------
# 1. Simple agents — no class needed
# ---------------------------------------------------------------------------


def example_simple_agents() -> None:
    print("=== 1. Simple agents ===")

    Researcher = make_agent(
        "Researcher",
        "You are a deep research specialist. Be thorough and cite sources.",
        ScriptedProvider(["Mars has two moons: Phobos and Deimos."]),
    )

    Writer = make_agent(
        "Writer",
        "You write clear, concise summaries for a general audience.",
        ScriptedProvider(["Mars has two tiny moons, Phobos and Deimos."]),
    )

    with Researcher.get_proxy() as proxy:
        reply = proxy.instruct("Tell me about Mars's moons.").get()
    print(f"Researcher: {reply}")

    with Writer.get_proxy() as proxy:
        reply = proxy.instruct("Summarise Mars's moons.").get()
    print(f"Writer    : {reply}")


# ---------------------------------------------------------------------------
# 2. Injecting tools
# ---------------------------------------------------------------------------


def example_with_tools() -> None:
    print("\n=== 2. Injecting tools ===")

    def word_count(self, text: str) -> int:
        "Count the number of words in a text."
        return len(text.split())

    @tool("Convert a temperature from Celsius to Fahrenheit.")
    def celsius_to_fahrenheit(self, celsius: float) -> float:
        return celsius * 9 / 5 + 32

    provider = ToolCallingProvider(
        tool_name="word_count",
        tool_args={"text": "The quick brown fox"},
        final_reply="The text contains 4 words.",
    )

    Analyst = make_agent(
        "Analyst",
        "Use tools for all calculations. Never guess.",
        provider,
        tools=[word_count, celsius_to_fahrenheit],
    )

    with Analyst.get_proxy() as proxy:
        reply = proxy.instruct("How many words are in 'The quick brown fox'?").get()
    print(f"Analyst: {reply}")
    print(f"  tool_result={provider.tool_result}")


# ---------------------------------------------------------------------------
# 3. Sub-agents — automatic @tool delegation
# ---------------------------------------------------------------------------


def example_sub_agents() -> None:
    print("\n=== 3. Sub-agents ===")

    # Specialist agents — each does one thing well
    Researcher = make_agent(
        "Researcher",
        "You are a research specialist. Return concise facts.",
        ScriptedProvider(["Key fact: Mars is the fourth planet from the Sun."]),
    )

    Writer = make_agent(
        "Writer",
        "You write polished prose from supplied facts.",
        ScriptedProvider(["Mars, the fourth planet, captivates scientists worldwide."]),
    )

    Critic = make_agent(
        "Critic",
        "You critique drafts and suggest improvements.",
        ScriptedProvider(["The draft is solid but could add emotional depth."]),
    )

    # Orchestrator — sub_agents are auto-wired as @tool methods
    # The LLM decides when to call researcher, writer, or critic
    Orchestrator = make_agent(
        "Orchestrator",
        (
            "You coordinate research and writing tasks. "
            "Use researcher to gather facts, writer to draft prose, "
            "and critic to review the result. "
            "Return the final polished text."
        ),
        # Simulates the LLM calling the 'researcher' tool
        ToolCallingProvider(
            tool_name="researcher",
            tool_args={"instruction": "Find facts about Mars."},
            final_reply="Here is the final report on Mars.",
        ),
        sub_agents={
            "researcher": Researcher,
            "writer": Writer,
            "critic": Critic,
        },
    )

    with Orchestrator.get_proxy() as proxy:
        reply = proxy.instruct("Write a short report about Mars.").get()
    print(f"Orchestrator: {reply}")


# ---------------------------------------------------------------------------
# 4. Class-based equivalent (when you need it)
# ---------------------------------------------------------------------------


def example_class_vs_factory() -> None:
    print("\n=== 4. Class-based equivalent ===")

    # This class definition …
    # Local imports:
    from actor_ai import AIActor

    class Summariser(AIActor):
        system_prompt = "You summarise text concisely."
        provider = ScriptedProvider(["Short summary here."])
        actor_name = "Summariser"

    # … is equivalent to:
    Summariser2 = make_agent(
        "Summariser",
        "You summarise text concisely.",
        ScriptedProvider(["Short summary here."]),
    )

    # Both behave identically
    for cls in (Summariser, Summariser2):
        with cls.get_proxy() as proxy:
            reply = proxy.instruct("Summarise this long document.").get()
        print(f"  [{cls.__name__}] {reply}")

    # Use a class only when you need @tool instance methods with complex logic,
    # access to `self`, or on_start() lifecycle hooks — otherwise make_agent()
    # is cleaner and shorter.


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    example_simple_agents()
    example_with_tools()
    example_sub_agents()
    example_class_vs_factory()


if __name__ == "__main__":
    main()
