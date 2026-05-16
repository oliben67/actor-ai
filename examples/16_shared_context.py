"""16 – SharedContext: Multi-agent shared memory and conversation log

``SharedContext`` lets multiple agents running concurrently share:

* **Long-term memory** — facts written by any agent visible to all (``remember``/``forget``).
* **Working memory** — task-scoped facts injected into every agent's system prompt
  (``remember_working``/``forget_working``/``clear_working_memory``).
* **Conversation log** — append-only record of every ``(agent, role, content)`` turn,
  written automatically during ``instruct()`` calls.

Each agent still keeps its own private session history (the back-and-forth messages
sent to its LLM).  The shared context is the cross-cutting layer on top.

Key behaviours shown
---------------------
1. Two agents reading shared long-term memory set before they start.
2. One agent writing a fact that the other immediately sees.
3. Shared working memory injected into both system prompts.
4. ``clear_session()`` does NOT clear shared working memory — reset it
   explicitly with ``clear_working_memory()``.
5. Inspecting the interleaved conversation log after both agents have run.

Real-LLM swap
-------------
Replace ScriptedProvider with:
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
from fake_provider import ScriptedProvider

# Local imports:
from actor_ai import AIActor, SharedContext, make_agent

# ---------------------------------------------------------------------------
# 1. Shared long-term memory — facts visible to all agents
# ---------------------------------------------------------------------------


def example_shared_long_term_memory() -> None:
    print("=== 1. Shared long-term memory ===")

    ctx = SharedContext()
    ctx.remember("project", "Apollo")
    ctx.remember("language", "Python")

    Analyst = make_agent(
        "Analyst",
        "You are a data analyst.",
        ScriptedProvider(["Apollo project uses Python — analysis complete."]),
        context=ctx,
    )
    Writer = make_agent(
        "Writer",
        "You write concise summaries.",
        ScriptedProvider(["Apollo is a Python project. Summary done."]),
        context=ctx,
    )

    with Analyst.get_proxy() as a:
        reply = a.instruct("Analyse the project setup.").get()
    print(f"Analyst : {reply}")
    print(f"  log entries so far: {len(ctx.get_log())}")

    with Writer.get_proxy() as w:
        reply = w.instruct("Write a one-liner about the project.").get()
    print(f"Writer  : {reply}")


# ---------------------------------------------------------------------------
# 2. One agent writes; the other reads
# ---------------------------------------------------------------------------


def example_cross_agent_memory_write() -> None:
    print("\n=== 2. Cross-agent memory write ===")

    ctx = SharedContext()

    Researcher = make_agent(
        "Researcher",
        "You research topics and remember findings.",
        ScriptedProvider(["Mars has two moons: Phobos and Deimos."]),
        context=ctx,
    )
    Reporter = make_agent(
        "Reporter",
        "You report on remembered findings.",
        ScriptedProvider(["Based on research: Mars — two moons."]),
        context=ctx,
    )

    with Researcher.get_proxy() as r:
        r.remember("mars_moons", "Phobos and Deimos")
        reply = r.instruct("Tell me about Mars moons.").get()
    print(f"Researcher: {reply}")

    with Reporter.get_proxy() as p:
        memory = p.get_memory().get()
        reply = p.instruct("Report on our Mars findings.").get()
    print(f"Reporter  : {reply}")
    print(f"  Reporter saw memory: {memory}")


# ---------------------------------------------------------------------------
# 3. Shared working memory injected into system prompts
# ---------------------------------------------------------------------------


def example_shared_working_memory() -> None:
    print("\n=== 3. Shared working memory ===")

    ctx = SharedContext()
    ctx.remember_working("current_task", "Q3 budget review")
    ctx.remember_working("deadline", "Friday")

    Analyst = make_agent(
        "Analyst",
        "You analyse financial data.",
        ScriptedProvider(["Q3 budget analysis complete — within target."]),
        context=ctx,
    )
    Reviewer = make_agent(
        "Reviewer",
        "You review financial analyses.",
        ScriptedProvider(["Analysis reviewed. Approved for Friday submission."]),
        context=ctx,
    )

    with Analyst.get_proxy() as a:
        reply = a.instruct("Analyse Q3 budget.").get()
    print(f"Analyst : {reply}")

    with Reviewer.get_proxy() as r:
        reply = r.instruct("Review the analysis.").get()
    print(f"Reviewer: {reply}")

    # clear_session() does NOT touch shared working memory
    with Analyst.get_proxy() as a:
        a.clear_session().get()
        wm = a.get_working_memory().get()
    print(f"  Working memory after clear_session: {wm}")

    # Explicitly reset shared working memory
    ctx.clear_working_memory()
    print(f"  Working memory after clear_working_memory: {ctx.get_working_memory()}")


# ---------------------------------------------------------------------------
# 4. Conversation log — full interleaved history
# ---------------------------------------------------------------------------


def example_conversation_log() -> None:
    print("\n=== 4. Conversation log ===")

    ctx = SharedContext()

    Planner = make_agent(
        "Planner",
        "You plan tasks.",
        ScriptedProvider(["Plan: research → draft → review."]),
        context=ctx,
        actor_name="Planner",
    )
    Executor = make_agent(
        "Executor",
        "You execute plans.",
        ScriptedProvider(["Executed: research done. Draft ready."]),
        context=ctx,
        actor_name="Executor",
    )

    with Planner.get_proxy() as p:
        p.instruct("Create a plan.").get()

    with Executor.get_proxy() as e:
        e.instruct("Execute the plan.").get()

    print("  Conversation log:")
    for entry in ctx.get_log():
        snippet = entry["content"][:50].replace("\n", " ")
        print(f"    [{entry['agent']}] {entry['role']}: {snippet}")


# ---------------------------------------------------------------------------
# 5. Class-based actor with shared context
# ---------------------------------------------------------------------------


def example_class_based_with_context() -> None:
    print("\n=== 5. Class-based actor with shared context ===")

    ctx = SharedContext()
    ctx.remember("domain", "healthcare")

    class SpecialistActor(AIActor):
        system_prompt = "You are a domain specialist."
        provider = ScriptedProvider(["Healthcare domain analysis complete."])
        context = ctx

    with SpecialistActor.get_proxy() as proxy:
        reply = proxy.instruct("Analyse the domain.").get()
    print(f"Specialist: {reply}")
    print(f"  Log entries: {len(ctx.get_log())}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    example_shared_long_term_memory()
    example_cross_agent_memory_write()
    example_shared_working_memory()
    example_conversation_log()
    example_class_based_with_context()


if __name__ == "__main__":
    main()
