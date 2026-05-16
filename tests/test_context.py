"""Tests for SharedContext and AIActor/make_agent integration with shared context."""

# Future imports (must occur at the beginning of the file):
from __future__ import annotations

# Standard library imports:
import threading

# Third party imports:
import pytest

# Local imports:
from actor_ai import AIActor, SharedContext, make_agent
from tests.conftest import FakeProvider


def make_actor_with_context(ctx: SharedContext, replies: list[str] | None = None):
    provider = FakeProvider(replies or ["reply"])
    cls = type(
        "CtxActor",
        (AIActor,),
        {"provider": provider, "system_prompt": "Test", "context": ctx},
    )
    return cls, provider


# ---------------------------------------------------------------------------
# SharedContext — long-term memory
# ---------------------------------------------------------------------------


class TestSharedContextMemory:
    def test_empty_on_init(self):
        ctx = SharedContext()
        assert ctx.get_memory() == {}

    def test_remember_stores_fact(self):
        ctx = SharedContext()
        ctx.remember("key", "value")
        assert ctx.get_memory() == {"key": "value"}

    def test_forget_removes_fact(self):
        ctx = SharedContext()
        ctx.remember("k", "v")
        ctx.forget("k")
        assert ctx.get_memory() == {}

    def test_forget_missing_key_is_noop(self):
        ctx = SharedContext()
        ctx.forget("nonexistent")  # must not raise

    def test_overwrite_existing_key(self):
        ctx = SharedContext()
        ctx.remember("k", "first")
        ctx.remember("k", "second")
        assert ctx.get_memory()["k"] == "second"

    def test_get_memory_returns_copy(self):
        ctx = SharedContext()
        ctx.remember("k", "v")
        m = ctx.get_memory()
        m["k"] = "mutated"
        assert ctx.get_memory()["k"] == "v"


# ---------------------------------------------------------------------------
# SharedContext — working memory
# ---------------------------------------------------------------------------


class TestSharedContextWorkingMemory:
    def test_empty_on_init(self):
        ctx = SharedContext()
        assert ctx.get_working_memory() == {}

    def test_remember_working_stores_fact(self):
        ctx = SharedContext()
        ctx.remember_working("task", "summarise")
        assert ctx.get_working_memory() == {"task": "summarise"}

    def test_forget_working_removes_fact(self):
        ctx = SharedContext()
        ctx.remember_working("k", "v")
        ctx.forget_working("k")
        assert ctx.get_working_memory() == {}

    def test_forget_working_missing_key_is_noop(self):
        ctx = SharedContext()
        ctx.forget_working("nonexistent")

    def test_clear_working_memory(self):
        ctx = SharedContext()
        ctx.remember_working("a", "1")
        ctx.remember_working("b", "2")
        ctx.clear_working_memory()
        assert ctx.get_working_memory() == {}

    def test_get_working_memory_returns_copy(self):
        ctx = SharedContext()
        ctx.remember_working("k", "v")
        m = ctx.get_working_memory()
        m["k"] = "mutated"
        assert ctx.get_working_memory()["k"] == "v"


# ---------------------------------------------------------------------------
# SharedContext — conversation log
# ---------------------------------------------------------------------------


class TestSharedContextLog:
    def test_empty_on_init(self):
        ctx = SharedContext()
        assert ctx.get_log() == []

    def test_append_log_adds_entry(self):
        ctx = SharedContext()
        ctx.append_log("Agent", "user", "hello")
        log = ctx.get_log()
        assert len(log) == 1
        assert log[0] == {"agent": "Agent", "role": "user", "content": "hello"}

    def test_multiple_entries_in_order(self):
        ctx = SharedContext()
        ctx.append_log("A", "user", "q1")
        ctx.append_log("A", "assistant", "a1")
        ctx.append_log("B", "user", "q2")
        log = ctx.get_log()
        assert [e["content"] for e in log] == ["q1", "a1", "q2"]

    def test_clear_log_empties_it(self):
        ctx = SharedContext()
        ctx.append_log("A", "user", "hi")
        ctx.clear_log()
        assert ctx.get_log() == []

    def test_get_log_returns_copy(self):
        ctx = SharedContext()
        ctx.append_log("A", "user", "hi")
        log = ctx.get_log()
        log.clear()
        assert len(ctx.get_log()) == 1


# ---------------------------------------------------------------------------
# SharedContext — thread safety
# ---------------------------------------------------------------------------


class TestSharedContextThreadSafety:
    def test_concurrent_remember_does_not_corrupt(self):
        ctx = SharedContext()
        errors: list[Exception] = []

        def write(i: int) -> None:
            try:
                for _ in range(100):
                    ctx.remember(f"k{i}", f"v{i}")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=write, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(ctx.get_memory()) == 10

    def test_concurrent_append_log_does_not_corrupt(self):
        ctx = SharedContext()
        errors: list[Exception] = []

        def log_entries(agent: str) -> None:
            try:
                for i in range(50):
                    ctx.append_log(agent, "user", f"msg{i}")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=log_entries, args=(f"Agent{i}",)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(ctx.get_log()) == 200


# ---------------------------------------------------------------------------
# AIActor — integration with SharedContext
# ---------------------------------------------------------------------------


class TestActorWithSharedContext:
    def test_remember_writes_to_context(self, actor_factory):
        ctx = SharedContext()
        cls, _ = make_actor_with_context(ctx)
        ref = actor_factory(cls)
        ref.proxy().remember("name", "Alice").get()
        assert ctx.get_memory() == {"name": "Alice"}

    def test_forget_removes_from_context(self, actor_factory):
        ctx = SharedContext()
        cls, _ = make_actor_with_context(ctx)
        ref = actor_factory(cls)
        ctx.remember("name", "Alice")
        ref.proxy().forget("name").get()
        assert ctx.get_memory() == {}

    def test_get_memory_reads_from_context(self, actor_factory):
        ctx = SharedContext()
        ctx.remember("lang", "Python")
        cls, _ = make_actor_with_context(ctx)
        ref = actor_factory(cls)
        assert ref.proxy().get_memory().get() == {"lang": "Python"}

    def test_two_actors_share_memory(self, actor_factory):
        ctx = SharedContext()
        cls_a, _ = make_actor_with_context(ctx, ["r1"])
        cls_b, _ = make_actor_with_context(ctx, ["r2"])
        ref_a = actor_factory(cls_a)
        ref_b = actor_factory(cls_b)
        ref_a.proxy().remember("shared", "fact").get()
        assert ref_b.proxy().get_memory().get() == {"shared": "fact"}

    def test_memory_injected_into_system_prompt(self, actor_factory):
        ctx = SharedContext()
        ctx.remember("project", "Iris")
        cls, provider = make_actor_with_context(ctx)
        ref = actor_factory(cls)
        ref.proxy().instruct("hi").get()
        assert "Iris" in provider.calls[0]["system"]

    def test_remember_working_writes_to_context(self, actor_factory):
        ctx = SharedContext()
        cls, _ = make_actor_with_context(ctx)
        ref = actor_factory(cls)
        ref.proxy().remember_working("task", "draft").get()
        assert ctx.get_working_memory() == {"task": "draft"}

    def test_two_actors_share_working_memory(self, actor_factory):
        ctx = SharedContext()
        cls_a, _ = make_actor_with_context(ctx, ["r1"])
        cls_b, _ = make_actor_with_context(ctx, ["r2"])
        ref_a = actor_factory(cls_a)
        ref_b = actor_factory(cls_b)
        ref_a.proxy().remember_working("goal", "analyse").get()
        assert ref_b.proxy().get_working_memory().get() == {"goal": "analyse"}

    def test_working_memory_injected_into_system_prompt(self, actor_factory):
        ctx = SharedContext()
        ctx.remember_working("task", "summarise")
        cls, provider = make_actor_with_context(ctx)
        ref = actor_factory(cls)
        ref.proxy().instruct("hi").get()
        assert "summarise" in provider.calls[0]["system"]

    def test_clear_session_does_not_clear_shared_working_memory(self, actor_factory):
        ctx = SharedContext()
        ctx.remember_working("task", "draft")
        cls, _ = make_actor_with_context(ctx)
        ref = actor_factory(cls)
        ref.proxy().clear_session().get()
        assert ctx.get_working_memory() == {"task": "draft"}

    def test_clear_working_memory_via_actor_clears_context(self, actor_factory):
        ctx = SharedContext()
        ctx.remember_working("task", "draft")
        cls, _ = make_actor_with_context(ctx)
        ref = actor_factory(cls)
        ref.proxy().clear_working_memory().get()
        assert ctx.get_working_memory() == {}

    def test_forget_working_via_actor_removes_from_context(self, actor_factory):
        ctx = SharedContext()
        ctx.remember_working("k", "v")
        cls, _ = make_actor_with_context(ctx)
        ref = actor_factory(cls)
        ref.proxy().forget_working("k").get()
        assert ctx.get_working_memory() == {}

    def test_instruct_appends_to_log(self, actor_factory):
        ctx = SharedContext()
        cls, _ = make_actor_with_context(ctx, ["pong"])
        ref = actor_factory(cls)
        ref.proxy().instruct("ping").get()
        log = ctx.get_log()
        assert len(log) == 2
        assert log[0] == {"agent": "CtxActor", "role": "user", "content": "ping"}
        assert log[1] == {"agent": "CtxActor", "role": "assistant", "content": "pong"}

    def test_log_uses_actor_name_when_set(self, actor_factory):
        ctx = SharedContext()
        provider = FakeProvider(["r"])
        cls = type(
            "Named",
            (AIActor,),
            {"provider": provider, "system_prompt": "s", "context": ctx, "actor_name": "MyAgent"},
        )
        ref = actor_factory(cls)
        ref.proxy().instruct("hi").get()
        assert ctx.get_log()[0]["agent"] == "MyAgent"

    def test_two_actors_log_interleaved(self, actor_factory):
        ctx = SharedContext()
        cls_a, _ = make_actor_with_context(ctx, ["a_reply"])
        cls_b, _ = make_actor_with_context(ctx, ["b_reply"])
        ref_a = actor_factory(cls_a)
        ref_b = actor_factory(cls_b)
        ref_a.proxy().instruct("from_a").get()
        ref_b.proxy().instruct("from_b").get()
        contents = [e["content"] for e in ctx.get_log()]
        assert "from_a" in contents
        assert "a_reply" in contents
        assert "from_b" in contents
        assert "b_reply" in contents

    def test_instruct_use_session_false_still_logs(self, actor_factory):
        ctx = SharedContext()
        cls, _ = make_actor_with_context(ctx, ["r"])
        ref = actor_factory(cls)
        ref.proxy().instruct("q", use_session=False).get()
        assert len(ctx.get_log()) == 2

    def test_no_context_uses_local_memory(self, actor_factory):
        provider = FakeProvider(["r"])
        cls = type("Local", (AIActor,), {"provider": provider, "system_prompt": "s"})
        ref = actor_factory(cls)
        ref.proxy().remember("k", "v").get()
        assert ref.proxy().get_memory().get() == {"k": "v"}

    def test_no_context_clear_session_clears_local_working_memory(self, actor_factory):
        provider = FakeProvider(["r"])
        cls = type("Local", (AIActor,), {"provider": provider, "system_prompt": "s"})
        ref = actor_factory(cls)
        ref.proxy().remember_working("task", "draft").get()
        ref.proxy().clear_session().get()
        assert ref.proxy().get_working_memory().get() == {}


# ---------------------------------------------------------------------------
# make_agent() — context parameter
# ---------------------------------------------------------------------------


class TestMakeAgentWithContext:
    def test_make_agent_passes_context(self, actor_factory):
        ctx = SharedContext()
        ctx.remember("env", "test")
        Agent = make_agent("Agent", "You help.", FakeProvider(["hi"]), context=ctx)
        ref = actor_factory(Agent)
        assert ref.proxy().get_memory().get() == {"env": "test"}

    def test_make_agent_logs_to_context(self, actor_factory):
        ctx = SharedContext()
        Agent = make_agent("Agent", "You help.", FakeProvider(["hi"]), context=ctx)
        ref = actor_factory(Agent)
        ref.proxy().instruct("hello").get()
        log = ctx.get_log()
        assert any(e["agent"] == "Agent" for e in log)

    def test_two_make_agents_share_context(self, actor_factory):
        ctx = SharedContext()
        A = make_agent("A", "Agent A.", FakeProvider(["ra"]), context=ctx)
        B = make_agent("B", "Agent B.", FakeProvider(["rb"]), context=ctx)
        ref_a = actor_factory(A)
        ref_b = actor_factory(B)
        ref_a.proxy().remember("shared", "data").get()
        assert ref_b.proxy().get_memory().get() == {"shared": "data"}

    def test_make_agent_without_context_is_independent(self, actor_factory):
        A = make_agent("A", "Agent A.", FakeProvider(["ra"]))
        B = make_agent("B", "Agent B.", FakeProvider(["rb"]))
        ref_a = actor_factory(A)
        ref_b = actor_factory(B)
        ref_a.proxy().remember("k", "v").get()
        assert ref_b.proxy().get_memory().get() == {}

    @pytest.mark.parametrize("context", [None])
    def test_make_agent_default_context_is_none(self, context):
        Agent = make_agent("Agent", "s.", FakeProvider(["r"]))
        assert Agent.context is None
