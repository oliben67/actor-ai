"""Tests for the accounting layer: UsageSummary, Ledger, ModelRate, Rates."""

# Future imports (must occur at the beginning of the file):
from __future__ import annotations

# Standard library imports:
from datetime import UTC, datetime
from unittest.mock import patch

# Third party imports:
import pytest

# Local imports:
from actor_ai import AIActor, Ledger, LedgerEntry, ModelRate, Rates, UsageSummary
from actor_ai.accounting import _DEFAULT_RATES, new_session_id
from tests.conftest import FakeProvider

# ---------------------------------------------------------------------------
# UsageSummary
# ---------------------------------------------------------------------------


class TestUsageSummary:
    def test_defaults_to_zero(self):
        u = UsageSummary()
        assert u.input_tokens == 0
        assert u.output_tokens == 0

    def test_total_tokens_sum(self):
        u = UsageSummary(input_tokens=100, output_tokens=50)
        assert u.total_tokens == 150

    def test_total_tokens_zero(self):
        assert UsageSummary().total_tokens == 0

    def test_add_two_summaries(self):
        a = UsageSummary(10, 5)
        b = UsageSummary(20, 15)
        c = a + b
        assert c.input_tokens == 30
        assert c.output_tokens == 20

    def test_add_does_not_mutate_operands(self):
        a = UsageSummary(10, 5)
        b = UsageSummary(20, 15)
        _ = a + b
        assert a.input_tokens == 10
        assert b.input_tokens == 20

    def test_iadd(self):
        u = UsageSummary(10, 5)
        u += UsageSummary(3, 2)
        assert u.input_tokens == 13
        assert u.output_tokens == 7

    def test_equality(self):
        assert UsageSummary(1, 2) == UsageSummary(1, 2)
        assert UsageSummary(1, 2) != UsageSummary(1, 3)


# ---------------------------------------------------------------------------
# ModelRate
# ---------------------------------------------------------------------------


class TestModelRate:
    def test_cost_with_zero_usage(self):
        rate = ModelRate(input_per_million=3.0, output_per_million=15.0)
        assert rate.cost(UsageSummary(0, 0)) == 0.0

    def test_cost_one_million_input_tokens(self):
        rate = ModelRate(input_per_million=3.0, output_per_million=15.0)
        cost = rate.cost(UsageSummary(input_tokens=1_000_000, output_tokens=0))
        assert cost == pytest.approx(3.0)

    def test_cost_one_million_output_tokens(self):
        rate = ModelRate(input_per_million=3.0, output_per_million=15.0)
        cost = rate.cost(UsageSummary(input_tokens=0, output_tokens=1_000_000))
        assert cost == pytest.approx(15.0)

    def test_cost_mixed(self):
        rate = ModelRate(input_per_million=2.0, output_per_million=10.0)
        # 500k input + 100k output
        cost = rate.cost(UsageSummary(input_tokens=500_000, output_tokens=100_000))
        assert cost == pytest.approx(1.0 + 1.0)  # 2.0 * 0.5 + 10.0 * 0.1

    def test_frozen(self):
        rate = ModelRate(1.0, 2.0)
        with pytest.raises(Exception):
            rate.input_per_million = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Rates
# ---------------------------------------------------------------------------


class TestRates:
    def test_default_contains_known_models(self):
        rates = Rates.default()
        assert "claude-sonnet-4-6" in rates
        assert "gpt-4o" in rates
        assert "gemini-2.0-flash" in rates
        assert "mistral-large-latest" in rates
        assert "deepseek-chat" in rates

    def test_from_dict(self):
        rates = Rates.from_dict({"my-model": {"input": 1.0, "output": 2.0}})
        assert "my-model" in rates
        assert rates.get_rate("my-model") == ModelRate(1.0, 2.0)

    def test_cost_known_model(self):
        rates = Rates.from_dict({"m": {"input": 4.0, "output": 8.0}})
        cost = rates.cost("m", UsageSummary(input_tokens=1_000_000, output_tokens=0))
        assert cost == pytest.approx(4.0)

    def test_cost_unknown_model_returns_zero(self):
        rates = Rates.default()
        cost = rates.cost("unknown-model-xyz", UsageSummary(100, 50))
        assert cost == 0.0

    def test_set_adds_new_model(self):
        rates = Rates.default()
        rates.set("new-model", input_per_million=5.0, output_per_million=20.0)
        assert "new-model" in rates
        assert rates.get_rate("new-model") == ModelRate(5.0, 20.0)

    def test_set_replaces_existing_model(self):
        rates = Rates.from_dict({"m": {"input": 1.0, "output": 2.0}})
        rates.set("m", input_per_million=9.0, output_per_million=99.0)
        assert rates.get_rate("m") == ModelRate(9.0, 99.0)

    def test_get_rate_missing_returns_none(self):
        rates = Rates.default()
        assert rates.get_rate("nonexistent") is None

    def test_models_returns_all_configured(self):
        rates = Rates.from_dict({"a": {"input": 1, "output": 1}})
        assert rates.models() == ["a"]

    def test_repr(self):
        rates = Rates.from_dict({"x": {"input": 1, "output": 1}})
        assert "Rates" in repr(rates)

    def test_default_rates_covers_all_known(self):
        rates = Rates.default()
        for model in _DEFAULT_RATES:
            assert model in rates, f"Missing default rate for {model}"

    def test_empty_rates(self):
        rates = Rates({})
        assert rates.models() == []
        assert rates.cost("any", UsageSummary(100, 50)) == 0.0


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


class TestLedger:
    def _entry(self, actor="a", model="m", inp=100, out=50, session="s1"):
        return LedgerEntry(
            actor_name=actor,
            model=model,
            input_tokens=inp,
            output_tokens=out,
            timestamp=datetime.now(UTC),
            session_id=session,
        )

    def test_starts_empty(self):
        ledger = Ledger()
        assert len(ledger) == 0
        assert ledger.entries() == []

    def test_record_adds_entry(self):
        ledger = Ledger()
        ledger.record("actor_a", "gpt-4o", 100, 50, session_id="s1")
        assert len(ledger) == 1

    def test_entries_returns_copy(self):
        ledger = Ledger()
        ledger.record("a", "m", 10, 5)
        entries = ledger.entries()
        entries.clear()
        assert len(ledger) == 1

    def test_clear_removes_all(self):
        ledger = Ledger()
        ledger.record("a", "m", 10, 5)
        ledger.clear()
        assert len(ledger) == 0

    def test_entry_timestamp_is_utc(self):
        ledger = Ledger()
        ledger.record("a", "m", 10, 5)
        ts = ledger.entries()[0].timestamp
        assert ts.tzinfo == UTC

    def test_total_usage_single_entry(self):
        ledger = Ledger()
        ledger.record("a", "m", 100, 50)
        u = ledger.total_usage()
        assert u.input_tokens == 100
        assert u.output_tokens == 50

    def test_total_usage_multiple_entries(self):
        ledger = Ledger()
        ledger.record("a", "m", 100, 50)
        ledger.record("b", "m", 200, 100)
        u = ledger.total_usage()
        assert u.input_tokens == 300
        assert u.output_tokens == 150

    def test_total_usage_empty(self):
        u = Ledger().total_usage()
        assert u == UsageSummary(0, 0)

    def test_usage_by_actor(self):
        ledger = Ledger()
        ledger.record("alice", "m", 100, 50)
        ledger.record("alice", "m", 50, 25)
        ledger.record("bob", "m", 200, 100)
        by_actor = ledger.usage_by_actor()
        assert by_actor["alice"] == UsageSummary(150, 75)
        assert by_actor["bob"] == UsageSummary(200, 100)

    def test_usage_by_model(self):
        ledger = Ledger()
        ledger.record("a", "gpt-4o", 100, 50)
        ledger.record("a", "gpt-4o", 100, 50)
        ledger.record("a", "claude-sonnet-4-6", 200, 80)
        by_model = ledger.usage_by_model()
        assert by_model["gpt-4o"] == UsageSummary(200, 100)
        assert by_model["claude-sonnet-4-6"] == UsageSummary(200, 80)

    def test_usage_by_session(self):
        ledger = Ledger()
        ledger.record("a", "m", 100, 50, session_id="sess-1")
        ledger.record("a", "m", 200, 80, session_id="sess-1")
        ledger.record("a", "m", 50, 20, session_id="sess-2")
        by_session = ledger.usage_by_session()
        assert by_session["sess-1"] == UsageSummary(300, 130)
        assert by_session["sess-2"] == UsageSummary(50, 20)

    def test_usage_by_session_none_mapped_to_key(self):
        ledger = Ledger()
        ledger.record("a", "m", 10, 5, session_id=None)
        by_session = ledger.usage_by_session()
        assert "no_session" in by_session

    def test_total_cost(self):
        ledger = Ledger()
        ledger.record("a", "gpt-4o", 1_000_000, 0)
        rates = Rates.from_dict({"gpt-4o": {"input": 2.50, "output": 10.00}})
        assert ledger.total_cost(rates) == pytest.approx(2.50)

    def test_total_cost_unknown_model_is_zero(self):
        ledger = Ledger()
        ledger.record("a", "unknown", 1_000_000, 1_000_000)
        assert ledger.total_cost(Rates.default()) == 0.0

    def test_cost_by_actor(self):
        ledger = Ledger()
        ledger.record("alice", "gpt-4o", 1_000_000, 0)
        ledger.record("bob", "gpt-4o", 0, 1_000_000)
        rates = Rates.from_dict({"gpt-4o": {"input": 2.50, "output": 10.00}})
        by_actor = ledger.cost_by_actor(rates)
        assert by_actor["alice"] == pytest.approx(2.50)
        assert by_actor["bob"] == pytest.approx(10.00)

    def test_cost_by_model(self):
        ledger = Ledger()
        ledger.record("a", "gpt-4o", 1_000_000, 0)
        ledger.record("a", "claude-sonnet-4-6", 1_000_000, 0)
        rates = Rates.from_dict(
            {
                "gpt-4o": {"input": 2.50, "output": 10.00},
                "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
            }
        )
        by_model = ledger.cost_by_model(rates)
        assert by_model["gpt-4o"] == pytest.approx(2.50)
        assert by_model["claude-sonnet-4-6"] == pytest.approx(3.00)

    def test_cost_by_session(self):
        ledger = Ledger()
        ledger.record("a", "gpt-4o", 1_000_000, 0, session_id="s1")
        ledger.record("a", "gpt-4o", 0, 1_000_000, session_id="s2")
        rates = Rates.from_dict({"gpt-4o": {"input": 2.50, "output": 10.00}})
        by_sess = ledger.cost_by_session(rates)
        assert by_sess["s1"] == pytest.approx(2.50)
        assert by_sess["s2"] == pytest.approx(10.00)

    def test_entries_for_actor(self):
        ledger = Ledger()
        ledger.record("alice", "m", 10, 5)
        ledger.record("bob", "m", 20, 10)
        assert len(ledger.entries_for_actor("alice")) == 1
        assert ledger.entries_for_actor("alice")[0].actor_name == "alice"

    def test_entries_for_session(self):
        ledger = Ledger()
        ledger.record("a", "m", 10, 5, session_id="s1")
        ledger.record("a", "m", 10, 5, session_id="s2")
        assert len(ledger.entries_for_session("s1")) == 1

    def test_entries_for_model(self):
        ledger = Ledger()
        ledger.record("a", "gpt-4o", 10, 5)
        ledger.record("a", "claude-sonnet-4-6", 10, 5)
        assert len(ledger.entries_for_model("gpt-4o")) == 1

    def test_summary_without_rates(self):
        ledger = Ledger()
        ledger.record("a", "gpt-4o", 100, 50)
        s = ledger.summary()
        assert s["entries"] == 1
        assert s["input_tokens"] == 100
        assert s["output_tokens"] == 50
        assert s["total_tokens"] == 150
        assert "total_cost_usd" not in s

    def test_summary_with_rates(self):
        ledger = Ledger()
        ledger.record("a", "gpt-4o", 1_000_000, 0)
        rates = Rates.from_dict({"gpt-4o": {"input": 2.50, "output": 10.00}})
        s = ledger.summary(rates)
        assert "total_cost_usd" in s
        assert s["total_cost_usd"] == pytest.approx(2.50)

    def test_repr(self):
        ledger = Ledger()
        ledger.record("a", "m", 10, 5)
        assert "Ledger" in repr(ledger)
        assert "1" in repr(ledger)

    def test_ledger_entry_usage_property(self):
        entry = LedgerEntry("a", "m", 100, 50, datetime.now(UTC))
        assert entry.usage == UsageSummary(100, 50)

    def test_thread_safety(self):
        """Multiple threads can record simultaneously without losing entries."""
        # Standard library imports:
        import threading

        ledger = Ledger()
        errors = []

        def record_many():
            try:
                for _ in range(100):
                    ledger.record("t", "m", 1, 1)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_many) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert len(ledger) == 500


# ---------------------------------------------------------------------------
# new_session_id
# ---------------------------------------------------------------------------


class TestNewSessionId:
    def test_returns_string(self):
        assert isinstance(new_session_id(), str)

    def test_unique_each_call(self):
        ids = {new_session_id() for _ in range(50)}
        assert len(ids) == 50


# ---------------------------------------------------------------------------
# AIActor accounting integration
# ---------------------------------------------------------------------------


class TestActorAccounting:
    def test_ledger_receives_entry_after_instruct(self, actor_factory):
        ledger = Ledger()
        usage = UsageSummary(input_tokens=100, output_tokens=50)
        provider = FakeProvider(["reply"], usage=usage)

        class Actor(AIActor):
            pass

        Actor.provider = provider
        Actor.ledger = ledger
        ref = actor_factory(Actor)
        ref.proxy().instruct("hello").get()

        assert len(ledger) == 1
        entry = ledger.entries()[0]
        assert entry.actor_name == "Actor"
        assert entry.input_tokens == 100
        assert entry.output_tokens == 50

    def test_actor_name_used_in_entry(self, actor_factory):
        ledger = Ledger()
        provider = FakeProvider(["r"])

        class Agent(AIActor):
            actor_name = "my_agent"

        Agent.provider = provider
        Agent.ledger = ledger
        ref = actor_factory(Agent)
        ref.proxy().instruct("hi").get()

        assert ledger.entries()[0].actor_name == "my_agent"

    def test_model_recorded_in_entry(self, actor_factory):
        ledger = Ledger()
        provider = FakeProvider(["r"])
        provider.model = "test-model-xyz"

        class Actor(AIActor):
            pass

        Actor.provider = provider
        Actor.ledger = ledger
        ref = actor_factory(Actor)
        ref.proxy().instruct("hi").get()

        assert ledger.entries()[0].model == "test-model-xyz"

    def test_session_id_recorded_in_entry(self, actor_factory):
        ledger = Ledger()
        provider = FakeProvider(["r"])

        class Actor(AIActor):
            pass

        Actor.provider = provider
        Actor.ledger = ledger
        ref = actor_factory(Actor)
        session_id = ref.proxy().get_session_id().get()
        ref.proxy().instruct("hi").get()

        assert ledger.entries()[0].session_id == session_id

    def test_multiple_calls_accumulate_entries(self, actor_factory):
        ledger = Ledger()
        provider = FakeProvider(["r1", "r2", "r3"])

        class Actor(AIActor):
            pass

        Actor.provider = provider
        Actor.ledger = ledger
        ref = actor_factory(Actor)
        for _ in range(3):
            ref.proxy().instruct("q").get()

        assert len(ledger) == 3

    def test_clear_session_changes_session_id(self, actor_factory):
        ledger = Ledger()
        provider = FakeProvider(["r1", "r2"])

        class Actor(AIActor):
            pass

        Actor.provider = provider
        Actor.ledger = ledger
        ref = actor_factory(Actor)
        ref.proxy().instruct("q1").get()
        ref.proxy().clear_session().get()
        ref.proxy().instruct("q2").get()

        entries = ledger.entries()
        assert entries[0].session_id != entries[1].session_id

    def test_no_ledger_no_recording(self, actor_factory):
        """When ledger is None, no accounting happens and instruct still works."""
        provider = FakeProvider(["r"])

        class Actor(AIActor):
            pass

        Actor.provider = provider
        Actor.ledger = None
        ref = actor_factory(Actor)
        assert ref.proxy().instruct("hi").get() == "r"

    def test_total_cost_calculation(self, actor_factory):
        ledger = Ledger()
        provider = FakeProvider(["r"], usage=UsageSummary(1_000_000, 0))

        class Actor(AIActor):
            pass

        Actor.provider = provider
        Actor.ledger = ledger
        Actor.provider.model = "gpt-4o"
        ref = actor_factory(Actor)
        ref.proxy().instruct("q").get()

        rates = Rates.from_dict({"gpt-4o": {"input": 2.50, "output": 10.00}})
        assert ledger.total_cost(rates) == pytest.approx(2.50)

    def test_cost_by_actor_with_multiple_actors(self, actor_factory):
        ledger = Ledger()
        prov_a = FakeProvider(["ra"], usage=UsageSummary(input_tokens=1_000_000, output_tokens=0))
        prov_b = FakeProvider(["rb"], usage=UsageSummary(input_tokens=0, output_tokens=1_000_000))
        prov_a.model = "gpt-4o"
        prov_b.model = "gpt-4o"

        class ActorA(AIActor):
            actor_name = "actor_a"

        class ActorB(AIActor):
            actor_name = "actor_b"

        ActorA.provider = prov_a
        ActorA.ledger = ledger
        ActorB.provider = prov_b
        ActorB.ledger = ledger
        ref_a = actor_factory(ActorA)
        ref_b = actor_factory(ActorB)
        ref_a.proxy().instruct("q").get()
        ref_b.proxy().instruct("q").get()

        rates = Rates.from_dict({"gpt-4o": {"input": 2.50, "output": 10.00}})
        by_actor = ledger.cost_by_actor(rates)
        assert by_actor["actor_a"] == pytest.approx(2.50)
        assert by_actor["actor_b"] == pytest.approx(10.00)

    def test_get_session_id_returns_string(self, actor_factory):
        provider = FakeProvider(["r"])

        class Actor(AIActor):
            pass

        Actor.provider = provider
        ref = actor_factory(Actor)
        sid = ref.proxy().get_session_id().get()
        assert isinstance(sid, str)
        assert len(sid) > 0

    def test_session_id_stable_within_session(self, actor_factory):
        provider = FakeProvider(["r1", "r2"])

        class Actor(AIActor):
            pass

        Actor.provider = provider
        ref = actor_factory(Actor)
        sid1 = ref.proxy().get_session_id().get()
        ref.proxy().instruct("q").get()
        sid2 = ref.proxy().get_session_id().get()
        assert sid1 == sid2


# ---------------------------------------------------------------------------
# Provider on_usage callbacks (Claude + GPT)
# ---------------------------------------------------------------------------


class TestProviderUsageCallback:
    def test_claude_reports_usage_on_end_turn(self):
        # Standard library imports:
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        # Local imports:
        from actor_ai.providers.anthropic import Claude

        with patch("actor_ai.providers.anthropic.Anthropic"):
            provider = Claude()

        usage_obj = SimpleNamespace(input_tokens=120, output_tokens=60)
        text_block = SimpleNamespace(text="done")
        response = SimpleNamespace(stop_reason="end_turn", content=[text_block], usage=usage_obj)
        provider._client = MagicMock()
        provider._client.messages.create.return_value = response

        received: list[UsageSummary] = []
        provider.run("s", [], [], lambda n, a: None, 100, on_usage=received.append)

        assert len(received) == 1
        assert received[0].input_tokens == 120
        assert received[0].output_tokens == 60

    def test_claude_no_callback_when_no_usage_attr(self):
        # Standard library imports:
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        # Local imports:
        from actor_ai.providers.anthropic import Claude

        with patch("actor_ai.providers.anthropic.Anthropic"):
            provider = Claude()

        # Response with no usage attribute
        response = SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(text="ok")])
        provider._client = MagicMock()
        provider._client.messages.create.return_value = response

        received: list[UsageSummary] = []
        provider.run("s", [], [], lambda n, a: None, 100, on_usage=received.append)
        assert received == []

    def test_gpt_reports_usage_on_stop(self):
        # Standard library imports:
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        # Local imports:
        from actor_ai.providers.openai import GPT

        with patch("actor_ai.providers.openai.OpenAI"):
            provider = GPT()

        usage_obj = SimpleNamespace(prompt_tokens=80, completion_tokens=40)
        choice = SimpleNamespace(
            finish_reason="stop",
            message=SimpleNamespace(content="done", tool_calls=[]),
        )
        response = SimpleNamespace(choices=[choice], usage=usage_obj)
        provider._client = MagicMock()
        provider._client.chat.completions.create.return_value = response

        received: list[UsageSummary] = []
        provider.run("s", [], [], lambda n, a: None, 100, on_usage=received.append)

        assert len(received) == 1
        assert received[0].input_tokens == 80
        assert received[0].output_tokens == 40

    def test_gpt_no_callback_when_usage_is_none(self):
        # Standard library imports:
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        # Local imports:
        from actor_ai.providers.openai import GPT

        with patch("actor_ai.providers.openai.OpenAI"):
            provider = GPT()

        choice = SimpleNamespace(
            finish_reason="stop",
            message=SimpleNamespace(content="ok", tool_calls=[]),
        )
        response = SimpleNamespace(choices=[choice], usage=None)
        provider._client = MagicMock()
        provider._client.chat.completions.create.return_value = response

        received: list[UsageSummary] = []
        provider.run("s", [], [], lambda n, a: None, 100, on_usage=received.append)
        assert received == []

    def test_accumulated_usage_across_tool_loop(self, actor_factory):
        """Usage from each API call in the agentic loop is summed into one entry."""
        # Local imports:
        from tests.conftest import ToolCallingFakeProvider

        ledger = Ledger()
        # The ToolCallingFakeProvider reports usage=UsageSummary(20, 10)
        provider = ToolCallingFakeProvider(
            "noop", {}, usage=UsageSummary(input_tokens=50, output_tokens=25)
        )

        class Actor(AIActor):
            # Local imports:
            from actor_ai.tools import tool as _tool

            @_tool
            def noop(self) -> str:
                return "ok"

        Actor.provider = provider
        Actor.ledger = ledger
        ref = actor_factory(Actor)
        ref.proxy().instruct("go").get()

        assert ledger.entries()[0].input_tokens == 50
        assert ledger.entries()[0].output_tokens == 25
