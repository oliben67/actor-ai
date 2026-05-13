"""Tests for Chorus: agent management, routing, broadcasting, pipeline, memory."""

# Future imports (must occur at the beginning of the file):
from __future__ import annotations

# Third party imports:
import pytest

# Local imports:
from actor_ai import AIActor, Chorus
from tests.conftest import FakeProvider

# ---------------------------------------------------------------------------
# Helpers — cheap agent factories
# ---------------------------------------------------------------------------


def make_agent(replies: list[str] | None = None, name: str = "Agent"):
    """Return a started AIActor ref backed by a FakeProvider."""
    provider = FakeProvider(replies or ["default reply"])

    cls = type(name, (AIActor,), {"provider": provider})
    return cls.start(), provider


def make_chorus(agents: dict | None = None):
    """Return a started Chorus ref."""
    return Chorus.start(agents=agents or {})


# ---------------------------------------------------------------------------
# Agent management
# ---------------------------------------------------------------------------


class TestAgentManagement:
    def test_empty_chorus_agents_returns_empty_list(self, actor_factory):
        ref = actor_factory(Chorus)
        assert ref.proxy().agents().get() == []

    def test_agents_list_after_construction(self, actor_factory):
        ag_ref, _ = make_agent()
        ref = actor_factory(Chorus, agents={"alpha": ag_ref})
        assert ref.proxy().agents().get() == ["alpha"]

    def test_add_agent(self, actor_factory):
        ref = actor_factory(Chorus)
        ag_ref, _ = make_agent()
        ref.proxy().add("new", ag_ref).get()
        assert "new" in ref.proxy().agents().get()

    def test_add_multiple_agents(self, actor_factory):
        ref = actor_factory(Chorus)
        for name in ["a", "b", "c"]:
            ag_ref, _ = make_agent(name=name)
            ref.proxy().add(name, ag_ref).get()
        assert set(ref.proxy().agents().get()) == {"a", "b", "c"}

    def test_remove_agent(self, actor_factory):
        ag_ref, _ = make_agent()
        ref = actor_factory(Chorus, agents={"target": ag_ref})
        ref.proxy().remove("target").get()
        assert "target" not in ref.proxy().agents().get()

    def test_remove_nonexistent_is_no_op(self, actor_factory):
        ref = actor_factory(Chorus)
        ref.proxy().remove("ghost").get()  # must not raise

    def test_remove_does_not_stop_agent(self, actor_factory):
        # Third party imports:
        import pykka

        ag_ref, _ = make_agent()
        ref = actor_factory(Chorus, agents={"ag": ag_ref})
        ref.proxy().remove("ag").get()
        assert pykka.ActorRegistry.get_by_urn(ag_ref.actor_urn) is not None
        ag_ref.stop()


# ---------------------------------------------------------------------------
# instruct() routing
# ---------------------------------------------------------------------------


class TestInstruct:
    def test_routes_to_named_agent(self, actor_factory):
        ag_ref, _ = make_agent(replies=["agent reply"])
        ref = actor_factory(Chorus, agents={"worker": ag_ref})
        result = ref.proxy().instruct("worker", "do something").get()
        assert result == "agent reply"

    def test_returns_agent_reply(self, actor_factory):
        ag_ref, _ = make_agent(replies=["hello back"])
        ref = actor_factory(Chorus, agents={"bot": ag_ref})
        assert ref.proxy().instruct("bot", "hello").get() == "hello back"

    def test_unknown_agent_raises_key_error(self, actor_factory):
        ref = actor_factory(Chorus)
        with pytest.raises(KeyError):
            ref.proxy().instruct("ghost", "hi").get()

    def test_error_message_includes_agent_name(self, actor_factory):
        ref = actor_factory(Chorus)
        with pytest.raises(KeyError, match="ghost"):
            ref.proxy().instruct("ghost", "hi").get()

    def test_routes_to_correct_agent_among_many(self, actor_factory):
        ref_a, _ = make_agent(replies=["from_a"], name="A")
        ref_b, _ = make_agent(replies=["from_b"], name="B")
        ref = actor_factory(Chorus, agents={"a": ref_a, "b": ref_b})
        assert ref.proxy().instruct("b", "hi").get() == "from_b"
        assert ref.proxy().instruct("a", "hi").get() == "from_a"


# ---------------------------------------------------------------------------
# broadcast()
# ---------------------------------------------------------------------------


class TestBroadcast:
    def test_empty_chorus_returns_empty_dict(self, actor_factory):
        ref = actor_factory(Chorus)
        assert ref.proxy().broadcast("hello").get() == {}

    def test_single_agent_broadcast(self, actor_factory):
        ag_ref, _ = make_agent(replies=["I got it"], name="Solo")
        ref = actor_factory(Chorus, agents={"solo": ag_ref})
        result = ref.proxy().broadcast("ping").get()
        assert result == {"solo": "I got it"}

    def test_multiple_agents_all_receive(self, actor_factory):
        ref_a, _ = make_agent(replies=["ack_a"], name="AA")
        ref_b, _ = make_agent(replies=["ack_b"], name="BB")
        ref = actor_factory(Chorus, agents={"a": ref_a, "b": ref_b})
        result = ref.proxy().broadcast("ping").get()
        assert result == {"a": "ack_a", "b": "ack_b"}

    def test_broadcast_returns_dict_keyed_by_name(self, actor_factory):
        ag_ref, _ = make_agent(replies=["r"], name="X")
        ref = actor_factory(Chorus, agents={"x": ag_ref})
        result = ref.proxy().broadcast("q").get()
        assert "x" in result

    def test_all_agents_receive_same_instruction(self, actor_factory):
        prov_a = FakeProvider(["ra"])
        prov_b = FakeProvider(["rb"])
        cls_a = type("A", (AIActor,), {"provider": prov_a})
        cls_b = type("B", (AIActor,), {"provider": prov_b})
        ref_a = cls_a.start()
        ref_b = cls_b.start()
        ref = actor_factory(Chorus, agents={"a": ref_a, "b": ref_b})
        ref.proxy().broadcast("broadcast msg").get()
        assert prov_a.calls[0]["messages"][-1]["content"] == "broadcast msg"
        assert prov_b.calls[0]["messages"][-1]["content"] == "broadcast msg"


# ---------------------------------------------------------------------------
# pipeline()
# ---------------------------------------------------------------------------


class TestPipeline:
    def test_empty_names_raises(self, actor_factory):
        ref = actor_factory(Chorus)
        with pytest.raises(ValueError, match="at least one"):
            ref.proxy().pipeline([], "start").get()

    def test_single_agent_returns_its_reply(self, actor_factory):
        ag_ref, _ = make_agent(replies=["single result"])
        ref = actor_factory(Chorus, agents={"only": ag_ref})
        assert ref.proxy().pipeline(["only"], "input").get() == "single result"

    def test_two_agents_chain_output(self, actor_factory):
        prov_a = FakeProvider(["step_one_output"])
        prov_b = FakeProvider(["final_output"])
        cls_a = type("PA", (AIActor,), {"provider": prov_a})
        cls_b = type("PB", (AIActor,), {"provider": prov_b})
        ref_a = cls_a.start()
        ref_b = cls_b.start()
        ref = actor_factory(Chorus, agents={"first": ref_a, "second": ref_b})
        result = ref.proxy().pipeline(["first", "second"], "initial").get()
        assert result == "final_output"
        # second agent received the first agent's reply as its instruction
        assert prov_b.calls[0]["messages"][-1]["content"] == "step_one_output"

    def test_three_agent_chain(self, actor_factory):
        prov1 = FakeProvider(["out1"])
        prov2 = FakeProvider(["out2"])
        prov3 = FakeProvider(["out3"])
        refs = []
        for i, prov in enumerate([prov1, prov2, prov3]):
            cls = type(f"Pn{i}", (AIActor,), {"provider": prov})
            refs.append(cls.start())
        agents = {"s1": refs[0], "s2": refs[1], "s3": refs[2]}
        ref = actor_factory(Chorus, agents=agents)
        result = ref.proxy().pipeline(["s1", "s2", "s3"], "begin").get()
        assert result == "out3"
        assert prov2.calls[0]["messages"][-1]["content"] == "out1"
        assert prov3.calls[0]["messages"][-1]["content"] == "out2"

    def test_unknown_agent_in_pipeline_raises(self, actor_factory):
        ref = actor_factory(Chorus)
        with pytest.raises(KeyError):
            ref.proxy().pipeline(["ghost"], "start").get()


# ---------------------------------------------------------------------------
# remember() / forget() memory broadcast
# ---------------------------------------------------------------------------


class TestMemoryBroadcast:
    def test_remember_reaches_all_agents(self, actor_factory):
        prov_a = FakeProvider(["r"])
        prov_b = FakeProvider(["r"])
        cls_a = type("MA", (AIActor,), {"provider": prov_a})
        cls_b = type("MB", (AIActor,), {"provider": prov_b})
        ref_a = cls_a.start()
        ref_b = cls_b.start()
        ref = actor_factory(Chorus, agents={"a": ref_a, "b": ref_b})
        ref.proxy().remember("project", "Mars").get()
        mem_a = ref_a.proxy().get_memory().get()
        mem_b = ref_b.proxy().get_memory().get()
        assert mem_a.get("project") == "Mars"
        assert mem_b.get("project") == "Mars"

    def test_remember_to_subset_of_agents(self, actor_factory):
        prov_a = FakeProvider(["r"])
        prov_b = FakeProvider(["r"])
        cls_a = type("SA", (AIActor,), {"provider": prov_a})
        cls_b = type("SB", (AIActor,), {"provider": prov_b})
        ref_a = cls_a.start()
        ref_b = cls_b.start()
        ref = actor_factory(Chorus, agents={"a": ref_a, "b": ref_b})
        ref.proxy().remember("k", "v", names=["a"]).get()
        assert ref_a.proxy().get_memory().get().get("k") == "v"
        assert "k" not in ref_b.proxy().get_memory().get()

    def test_forget_reaches_all_agents(self, actor_factory):
        prov_a = FakeProvider(["r"])
        prov_b = FakeProvider(["r"])
        cls_a = type("FA", (AIActor,), {"provider": prov_a})
        cls_b = type("FB", (AIActor,), {"provider": prov_b})
        ref_a = cls_a.start()
        ref_b = cls_b.start()
        ref_a.proxy().remember("key", "val").get()
        ref_b.proxy().remember("key", "val").get()
        ref = actor_factory(Chorus, agents={"a": ref_a, "b": ref_b})
        ref.proxy().forget("key").get()
        assert "key" not in ref_a.proxy().get_memory().get()
        assert "key" not in ref_b.proxy().get_memory().get()

    def test_forget_to_subset_of_agents(self, actor_factory):
        prov_a = FakeProvider(["r"])
        prov_b = FakeProvider(["r"])
        cls_a = type("FsA", (AIActor,), {"provider": prov_a})
        cls_b = type("FsB", (AIActor,), {"provider": prov_b})
        ref_a = cls_a.start()
        ref_b = cls_b.start()
        ref_a.proxy().remember("k", "v").get()
        ref_b.proxy().remember("k", "v").get()
        ref = actor_factory(Chorus, agents={"a": ref_a, "b": ref_b})
        ref.proxy().forget("k", names=["a"]).get()
        assert "k" not in ref_a.proxy().get_memory().get()
        assert ref_b.proxy().get_memory().get().get("k") == "v"

    def test_remember_unknown_name_raises(self, actor_factory):
        ref = actor_factory(Chorus)
        with pytest.raises(KeyError):
            ref.proxy().remember("k", "v", names=["ghost"]).get()

    def test_forget_unknown_name_raises(self, actor_factory):
        ref = actor_factory(Chorus)
        with pytest.raises(KeyError):
            ref.proxy().forget("k", names=["ghost"]).get()


# ---------------------------------------------------------------------------
# stop_agents()
# ---------------------------------------------------------------------------


class TestStopAgents:
    def test_stop_named_agent_removes_it(self, actor_factory):
        ag_ref, _ = make_agent()
        ref = actor_factory(Chorus, agents={"victim": ag_ref})
        ref.proxy().stop_agents(["victim"]).get()
        assert "victim" not in ref.proxy().agents().get()

    def test_stop_all_agents_with_none(self, actor_factory):
        ref_a, _ = make_agent(name="StA")
        ref_b, _ = make_agent(name="StB")
        ref = actor_factory(Chorus, agents={"a": ref_a, "b": ref_b})
        ref.proxy().stop_agents().get()
        assert ref.proxy().agents().get() == []

    def test_untouched_agents_remain(self, actor_factory):
        ref_a, _ = make_agent(name="KA")
        ref_b, _ = make_agent(name="KB")
        ref = actor_factory(Chorus, agents={"keep": ref_a, "remove": ref_b})
        ref.proxy().stop_agents(["remove"]).get()
        remaining = ref.proxy().agents().get()
        assert "keep" in remaining
        assert "remove" not in remaining
