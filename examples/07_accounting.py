"""07 – Accounting

The Ledger records every instruct() call (tokens in/out, model, actor name,
session ID, timestamp).  Rates maps model names to USD/million-token prices
so you can calculate cost.

This example runs entirely without an API key — the fake provider reports
realistic token counts via on_usage.

Topics covered:
  - Attaching a Ledger to one or more actors
  - Rates: default(), from_dict(), set()
  - Ledger: entries, total_usage, usage_by_actor, usage_by_model, cost
  - summary() for a quick overview
  - Thread-safety: shared Ledger across two concurrent actors
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
from actor_ai import AIActor, Ledger, ModelRate, Rates, UsageSummary

# ── Module-level shared objects ────────────────────────────────────────────
# Actor classes are defined at module level so their class attributes can
# reference module-level variables (Python class bodies do not close over
# enclosing function locals).

shared_ledger = Ledger()

rates = Rates.default()
rates.set("scripted/demo", input_per_million=1.00, output_per_million=2.00)
rates.set("my-fine-tune", input_per_million=0.50, output_per_million=1.00)


class AnalystActor(AIActor):
    system_prompt = "You are a financial analyst."
    provider = ScriptedProvider(
        ["Revenue grew 12 % YoY driven by cloud subscriptions."],
        usage=UsageSummary(input_tokens=200, output_tokens=80),
    )
    actor_name = "analyst"
    ledger = shared_ledger


class WriterActor(AIActor):
    system_prompt = "You are a business writer."
    provider = ScriptedProvider(
        [
            "Cloud adoption continues to accelerate, lifting annual revenue by a solid 12 percent.",
            "Investors welcomed the results; shares rose in early trading.",
        ],
        usage=UsageSummary(input_tokens=150, output_tokens=120),
    )
    actor_name = "writer"
    ledger = shared_ledger


# ── Helpers ────────────────────────────────────────────────────────────────


def _divider(title: str) -> None:
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print(f"{'─' * 55}")


def main() -> None:

    # ── Setup ──────────────────────────────────────────────────────────
    _divider("Setup: Ledger + Rates")

    print(f"  Ledger  : {shared_ledger}")
    print(f"  Models in rates table: {len(rates.models())}")

    # ── Make a few LLM calls ───────────────────────────────────────────
    _divider("Two actors sharing one Ledger")

    analyst_ref = AnalystActor.start()
    writer_ref = WriterActor.start()

    try:
        analyst_ref.proxy().instruct("Summarise Q3 earnings.").get()
        writer_ref.proxy().instruct("Turn the analyst notes into a press snippet.").get()
        writer_ref.proxy().instruct("Add a closing sentence about the stock reaction.").get()

        # ── All entries ────────────────────────────────────────────────
        _divider("All ledger entries")

        print(f"  Total entries: {len(shared_ledger)}")
        for entry in shared_ledger.entries():
            session_str = entry.session_id[:8] if entry.session_id else "N/A"
            print(
                f"  [{entry.actor_name:8}]  model={entry.model!r:15}"
                f"  in={entry.input_tokens:4}  out={entry.output_tokens:4}"
                f"  session={session_str}…"
            )

        # ── Usage aggregates ───────────────────────────────────────────
        _divider("Usage aggregates")

        total = shared_ledger.total_usage()
        print(
            f"  Total : {total.input_tokens} in + {total.output_tokens} out"
            f" = {total.total_tokens} tokens"
        )

        print("\n  By actor:")
        for name, u in shared_ledger.usage_by_actor().items():
            print(f"    {name:10}:  {u.input_tokens:4} in  {u.output_tokens:4} out")

        print("\n  By model:")
        for model, u in shared_ledger.usage_by_model().items():
            print(f"    {model!r:20}: {u.total_tokens:4} total tokens")

        # ── Cost ───────────────────────────────────────────────────────
        _divider("Cost calculation")

        total_cost = shared_ledger.total_cost(rates)
        print(f"  Total cost : ${total_cost:.6f}")

        print("\n  By actor:")
        for name, cost in shared_ledger.cost_by_actor(rates).items():
            print(f"    {name:10}: ${cost:.6f}")

        print("\n  By model:")
        for model, cost in shared_ledger.cost_by_model(rates).items():
            print(f"    {model!r:20}: ${cost:.6f}")

        # ── Filter by session ──────────────────────────────────────────
        _divider("Filter by session (analyst only)")

        analyst_sid = analyst_ref.proxy().get_session_id().get()
        session_entries = shared_ledger.entries_for_session(analyst_sid)
        session_usage = shared_ledger.usage_by_session().get(analyst_sid, UsageSummary())
        print(
            f"  Analyst session {analyst_sid[:8]}…  →"
            f"  {len(session_entries)} entries,"
            f"  {session_usage.total_tokens} tokens"
        )

        # ── Filter by actor ────────────────────────────────────────────
        writer_entries = shared_ledger.entries_for_actor("writer")
        print(f"  Writer entries: {len(writer_entries)}")

        # ── Summary dict ───────────────────────────────────────────────
        _divider("Ledger.summary(rates)")

        summary = shared_ledger.summary(rates)
        for key, val in summary.items():
            if isinstance(val, dict):
                print(f"  {key}:")
                for k, v in val.items():
                    print(f"    {k!r}: {v}")
            else:
                print(f"  {key}: {val}")

        # ── ModelRate standalone ───────────────────────────────────────
        _divider("ModelRate standalone")

        custom_rate = ModelRate(input_per_million=3.00, output_per_million=15.00)
        usage = UsageSummary(input_tokens=1_000, output_tokens=500)
        cost = custom_rate.cost(usage)
        print("  Rate  : $3.00 in / $15.00 out per million tokens")
        print(f"  Usage : {usage.input_tokens} in + {usage.output_tokens} out")
        print(f"  Cost  : ${cost:.6f}")

        # ── Rates.from_dict ────────────────────────────────────────────
        _divider("Rates.from_dict()")

        custom_rates = Rates.from_dict(
            {
                "my-model-v1": {"input": 1.50, "output": 3.00},
                "my-model-v2": {"input": 0.80, "output": 1.60},
            }
        )
        print(f"  Models: {custom_rates.models()}")
        u = UsageSummary(input_tokens=10_000, output_tokens=5_000)
        print(f"  my-model-v1  10K in / 5K out → ${custom_rates.cost('my-model-v1', u):.4f}")
        print(
            f"  unknown-model              → "
            f"${custom_rates.cost('unknown-model', u):.4f}  (returns 0.0 if unknown)"
        )

        # ── Ledger.clear ───────────────────────────────────────────────
        _divider("Ledger.clear()")

        print(f"  Before clear: {len(shared_ledger)} entries")
        shared_ledger.clear()
        print(f"  After  clear: {len(shared_ledger)} entries")

    finally:
        analyst_ref.stop()
        writer_ref.stop()


if __name__ == "__main__":
    main()
