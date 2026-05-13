"""10 – Advanced: Full Pipeline

Combines everything into a realistic multi-agent research pipeline:
  - Two specialist actors (Researcher + Writer) with @tool methods
  - Long-term memory shared per actor
  - Session management with history trimming
  - Shared Ledger + Rates for cost tracking
  - A Chorus that orchestrates the pipeline
  - Post-run accounting report

The Researcher has tools to look up facts; the Writer has a tool to count words.
The pipeline: Researcher gathers facts → Writer crafts a report.

Real-LLM swap
-------------
    from actor_ai import Claude, GPT
    # Researcher.provider = Claude()
    # Writer.provider     = GPT("gpt-4o")
"""

# Future imports (must occur at the beginning of the file):
from __future__ import annotations

# Standard library imports:
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Third party imports:
from fake_provider import ScriptedProvider, ToolCallingProvider

# Local imports:
from actor_ai import AIActor, Chorus, Ledger, Rates, tool
from actor_ai.accounting import UsageSummary

# ── Shared accounting objects ──────────────────────────────────────────────

ledger = Ledger()
rates = Rates.default()
rates.set("scripted/demo", input_per_million=1.00, output_per_million=2.00)


# ── Researcher ─────────────────────────────────────────────────────────────


class Researcher(AIActor):
    """Gathers factual information and summarises it.

    Exposes two tools:
      lookup_planet  — return key facts about a named planet
      list_missions  — list recent missions to a named destination
    """

    system_prompt = (
        "You are a space research specialist. "
        "Use your tools to look up accurate facts before summarising."
    )
    provider = ToolCallingProvider(
        tool_name="lookup_planet",
        tool_args={"name": "Mars"},
        final_reply=(
            "Mars is the fourth planet from the Sun. Diameter: 6,779 km. "
            "Two moons: Phobos and Deimos. Surface gravity: 3.72 m/s². "
            "NASA's Perseverance rover is exploring Jezero Crater since 2021."
        ),
        usage=UsageSummary(input_tokens=220, output_tokens=95),
    )
    actor_name = "researcher"
    ledger = ledger
    max_history = 5

    @tool("Look up key facts about a named planet.")
    def lookup_planet(self, name: str) -> str:
        facts = {
            "Mars": "4th planet, 2 moons, diameter 6779 km, gravity 3.72 m/s²",
            "Venus": "2nd planet, no moons, hottest surface (465 °C), retrograde rotation",
            "Jupiter": "5th planet, 95 moons, largest planet, Great Red Spot storm",
        }
        return facts.get(name, f"No data available for {name!r}.")

    @tool("List recent space missions to a named destination.")
    def list_missions(self, destination: str) -> str:
        missions = {
            "Mars": "Perseverance (2021), Ingenuity (2021), Tianwen-1 (2021)",
            "Moon": "Artemis I (2022), SLIM (2024), Chandrayaan-3 (2023)",
            "Jupiter": "Juno (ongoing), Europa Clipper (2024 launch)",
        }
        return missions.get(destination, f"No missions listed for {destination!r}.")


# ── Writer ─────────────────────────────────────────────────────────────────


class Writer(AIActor):
    """Turns research notes into polished prose.

    Exposes a tool to count words in a draft.
    """

    system_prompt = (
        "You are a science communicator. "
        "Use the word_count tool to verify your draft stays under the limit."
    )
    provider = ToolCallingProvider(
        tool_name="word_count",
        tool_args={"text": "Mars is a fascinating world of iron plains and ancient craters."},
        final_reply=(
            "**Red Planet Rising**\n\n"
            "Mars — the fourth planet, ruddy and remote — has captivated humanity "
            "for centuries. With a diameter of 6,779 kilometres and surface gravity "
            "barely a third of Earth's, its iron-rich plains stretch beneath a thin "
            "carbon-dioxide sky. Two tiny moons, Phobos and Deimos, race overhead "
            "while NASA's Perseverance rover inches through Jezero Crater, sampling "
            "ancient lake sediments for signs of past life. Mars is not just a "
            "destination — it is our next chapter."
        ),
        usage=UsageSummary(input_tokens=310, output_tokens=160),
    )
    actor_name = "writer"
    ledger = ledger
    max_history = 5

    @tool("Count the number of words in a text string.")
    def word_count(self, text: str) -> int:
        return len(text.split())


# ── Pipeline orchestration ─────────────────────────────────────────────────


def _divider(title: str) -> None:
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")


def _section(title: str) -> None:
    print(f"\n  ── {title}")


def main() -> None:
    researcher_ref = Researcher.start()
    writer_ref = Writer.start()

    chorus_ref = Chorus.start(agents={"researcher": researcher_ref, "writer": writer_ref})
    chorus = chorus_ref.proxy()

    try:
        # ── Seed memory in both agents ─────────────────────────────────
        _divider("Step 1: Seed shared context")

        chorus.remember("topic", "Mars exploration").get()
        chorus.remember("audience", "general public").get()
        chorus.remember("max_words", "150").get()

        print("  Researcher memory:")
        for k, v in researcher_ref.proxy().get_memory().get().items():
            print(f"    {k}: {v}")

        # ── Researcher gathers facts (calls lookup_planet tool) ────────
        _divider("Step 2: Researcher gathers facts (tool call)")

        research_reply = chorus.instruct(
            "researcher",
            "Research the key facts about Mars and recent missions.",
        ).get()

        print("  Tool called : lookup_planet(name='Mars')")
        print(f"  Tool result : {Researcher.provider.tool_result}")
        print("\n  Research summary (excerpt):")
        print(f"  {research_reply[:120]}…")

        # Session now has 1 turn
        session_len = len(researcher_ref.proxy().get_session().get())
        print(f"\n  Researcher session: {session_len} messages")

        # ── Writer crafts the report (calls word_count tool) ──────────
        _divider("Step 3: Writer crafts report (tool call)")

        writer_brief = f"Write a 100–150 word public brief about: {research_reply}"

        # Feed the research output directly to the writer via pipeline
        final_report = chorus.instruct("writer", writer_brief).get()

        print("  Tool called : word_count(text='…')")
        print(f"  Word count  : {Writer.provider.tool_result}")
        print("\n  Final report:\n")
        for line in final_report.split("\n"):
            print(f"  {line}")

        # ── Accounting report ──────────────────────────────────────────
        _divider("Step 4: Accounting report")

        _section("Ledger entries")
        for entry in ledger.entries():
            print(
                f"    [{entry.actor_name:10}]  "
                f"in={entry.input_tokens:4}  out={entry.output_tokens:4}  "
                f"total={entry.usage.total_tokens:5}  "
                f"session={entry.session_id[:8]}…"
            )

        _section("Usage by actor")
        for name, u in ledger.usage_by_actor().items():
            print(f"    {name:12}: {u.total_tokens:5} tokens total")

        total = ledger.total_usage()
        print(
            f"\n    Grand total : {total.total_tokens} tokens"
            f" ({total.input_tokens} in + {total.output_tokens} out)"
        )

        _section("Cost breakdown (scripted/demo rate: $1/$2 per M)")
        for name, cost in ledger.cost_by_actor(rates).items():
            print(f"    {name:12}: ${cost:.6f}")
        print(f"\n    Total cost  : ${ledger.total_cost(rates):.6f}")

        # ── Summary dict ──────────────────────────────────────────────
        _section("summary() dict")
        s = ledger.summary(rates)
        print(f"    entries          : {s['entries']}")
        print(f"    total_tokens     : {s['total_tokens']}")
        print(f"    total_cost_usd   : ${s['total_cost_usd']:.6f}")

        # ── Session info ───────────────────────────────────────────────
        _divider("Step 5: Session info")

        for name, ref in [("researcher", researcher_ref), ("writer", writer_ref)]:
            sid = ref.proxy().get_session_id().get()
            slen = len(ref.proxy().get_session().get())
            print(f"  {name:12}: session_id={sid[:8]}…  messages={slen}")

    finally:
        chorus_ref.stop()
        researcher_ref.stop()
        writer_ref.stop()


if __name__ == "__main__":
    main()
