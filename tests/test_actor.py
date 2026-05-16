"""Tests for AIActor: provider wiring, session, memory, tool dispatch, messages."""

# Future imports (must occur at the beginning of the file):
from __future__ import annotations

# Standard library imports:
import io

# Third party imports:
import pytest

# Local imports:
from actor_ai import AIActor, Forget, Instruct, Remember, tool
from tests.conftest import FakeProvider, ToolCallingFakeProvider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_actor(cls_attrs: dict | None = None, replies: list[str] | None = None):
    """Return (ActorClass, provider) — does NOT start the actor."""
    provider = FakeProvider(replies or ["reply"])
    attrs = {"provider": provider, "system_prompt": "Test system"}
    attrs.update(cls_attrs or {})
    cls = type("TestActor", (AIActor,), attrs)
    return cls, provider


# ---------------------------------------------------------------------------
# Basic instruct()
# ---------------------------------------------------------------------------


class TestInstructBasic:
    def test_instruct_via_proxy(self, actor_factory):
        cls, _ = make_actor(replies=["pong"])
        ref = actor_factory(cls)
        assert ref.proxy().instruct("ping").get() == "pong"

    def test_instruct_via_ask_message(self, actor_factory):
        cls, _ = make_actor(replies=["ack"])
        ref = actor_factory(cls)
        assert ref.ask(Instruct("go")) == "ack"

    def test_instruct_via_tell_message_no_error(self, actor_factory):
        cls, _ = make_actor()
        ref = actor_factory(cls)
        ref.tell(Instruct("fire and forget"))
        # Synchronise — session is updated only after message is processed.
        ref.proxy().get_session().get()

    def test_provider_receives_user_message(self, actor_factory):
        cls, provider = make_actor()
        ref = actor_factory(cls)
        ref.proxy().instruct("hello world").get()
        messages = provider.calls[0]["messages"]
        assert messages[-1] == {"role": "user", "content": "hello world"}

    def test_provider_receives_system_prompt(self, actor_factory):
        cls, provider = make_actor()
        ref = actor_factory(cls)
        ref.proxy().instruct("hi").get()
        assert provider.calls[0]["system"] == "Test system"

    def test_provider_receives_max_tokens(self, actor_factory):
        provider = FakeProvider(["r"])

        class Actor(AIActor):
            max_tokens = 2048

        Actor.provider = provider
        ref = actor_factory(Actor)
        ref.proxy().instruct("hi").get()
        assert provider.calls[0]["max_tokens"] == 2048

    def test_provider_receives_tool_specs(self, actor_factory):
        provider = FakeProvider(["r"])

        class Actor(AIActor):
            @tool
            def my_tool(self, x: int) -> str:
                "A tool."
                return str(x)

        Actor.provider = provider
        ref = actor_factory(Actor)
        ref.proxy().instruct("hi").get()
        names = [t["name"] for t in provider.calls[0]["tools"]]
        assert "my_tool" in names

    def test_no_tools_when_none_decorated(self, actor_factory):
        cls, provider = make_actor()
        ref = actor_factory(cls)
        ref.proxy().instruct("hi").get()
        assert provider.calls[0]["tools"] == []


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------


class TestSession:
    def test_session_empty_on_start(self, actor_factory):
        cls, _ = make_actor()
        ref = actor_factory(cls)
        assert ref.proxy().get_session().get() == []

    def test_session_updated_after_instruct(self, actor_factory):
        cls, _ = make_actor(replies=["A1"])
        ref = actor_factory(cls)
        ref.proxy().instruct("Q1").get()
        session = ref.proxy().get_session().get()
        assert session == [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
        ]

    def test_session_accumulates_across_calls(self, actor_factory):
        cls, _ = make_actor(replies=["A1", "A2"])
        ref = actor_factory(cls)
        ref.proxy().instruct("Q1").get()
        ref.proxy().instruct("Q2").get()
        session = ref.proxy().get_session().get()
        assert len(session) == 4
        contents = [m["content"] for m in session]
        assert contents == ["Q1", "A1", "Q2", "A2"]

    def test_session_passed_to_provider_on_second_call(self, actor_factory):
        cls, provider = make_actor(replies=["A1", "A2"])
        ref = actor_factory(cls)
        ref.proxy().instruct("Q1").get()
        ref.proxy().instruct("Q2").get()
        # Second call's messages must include the first turn
        second_messages = provider.calls[1]["messages"]
        contents = [m["content"] for m in second_messages]
        assert "Q1" in contents
        assert "A1" in contents
        assert "Q2" in contents

    def test_clear_session_empties_history(self, actor_factory):
        cls, _ = make_actor(replies=["A1", "A2"])
        ref = actor_factory(cls)
        ref.proxy().instruct("Q1").get()
        ref.proxy().clear_session().get()
        assert ref.proxy().get_session().get() == []

    def test_after_clear_next_call_has_no_history(self, actor_factory):
        cls, provider = make_actor(replies=["A1", "A2"])
        ref = actor_factory(cls)
        ref.proxy().instruct("Q1").get()
        ref.proxy().clear_session().get()
        ref.proxy().instruct("Q2").get()
        third_call_messages = provider.calls[1]["messages"]
        contents = [m["content"] for m in third_call_messages]
        assert "Q1" not in contents
        assert "A1" not in contents

    def test_get_session_returns_copy(self, actor_factory):
        cls, _ = make_actor(replies=["A"])
        ref = actor_factory(cls)
        ref.proxy().instruct("Q").get()
        s1 = ref.proxy().get_session().get()
        s1.clear()
        s2 = ref.proxy().get_session().get()
        assert len(s2) == 2

    def test_use_session_false_does_not_update_session(self, actor_factory):
        cls, _ = make_actor()
        ref = actor_factory(cls)
        ref.proxy().instruct("hi", use_session=False).get()
        assert ref.proxy().get_session().get() == []

    def test_explicit_history_used_over_session(self, actor_factory):
        cls, provider = make_actor(replies=["A1", "A2"])
        ref = actor_factory(cls)
        ref.proxy().instruct("Q1").get()  # builds session
        explicit = [{"role": "user", "content": "explicit prior"}]
        ref.proxy().instruct("Q2", history=explicit).get()
        second_messages = provider.calls[1]["messages"]
        contents = [m["content"] for m in second_messages]
        assert "explicit prior" in contents
        assert "Q1" not in contents

    def test_explicit_history_does_not_update_session(self, actor_factory):
        cls, _ = make_actor()
        ref = actor_factory(cls)
        explicit = [{"role": "user", "content": "prior"}]
        ref.proxy().instruct("Q", history=explicit).get()
        assert ref.proxy().get_session().get() == []

    def test_instruct_message_use_session_false(self, actor_factory):
        cls, _ = make_actor()
        ref = actor_factory(cls)
        ref.ask(Instruct("hi", use_session=False))
        assert ref.proxy().get_session().get() == []

    def test_instruct_message_with_history(self, actor_factory):
        cls, provider = make_actor()
        ref = actor_factory(cls)
        history = [{"role": "user", "content": "custom prior"}]
        ref.ask(Instruct("new q", history=history, use_session=False))
        messages = provider.calls[0]["messages"]
        assert any(m["content"] == "custom prior" for m in messages)


# ---------------------------------------------------------------------------
# max_history trimming
# ---------------------------------------------------------------------------


class TestMaxHistory:
    def test_unlimited_when_zero(self, actor_factory):
        provider = FakeProvider(["r"] * 10)

        class Actor(AIActor):
            max_history = 0

        Actor.provider = provider
        ref = actor_factory(Actor)
        for i in range(5):
            ref.proxy().instruct(f"q{i}").get()
        assert len(ref.proxy().get_session().get()) == 10

    def test_trim_to_one_turn(self, actor_factory):
        provider = FakeProvider(["r1", "r2"])

        class Actor(AIActor):
            max_history = 1

        Actor.provider = provider
        ref = actor_factory(Actor)
        ref.proxy().instruct("q1").get()
        ref.proxy().instruct("q2").get()
        session = ref.proxy().get_session().get()
        assert len(session) == 2
        assert session[0]["content"] == "q2"
        assert session[1]["content"] == "r2"

    def test_trim_to_two_turns(self, actor_factory):
        provider = FakeProvider(["r1", "r2", "r3"])

        class Actor(AIActor):
            max_history = 2

        Actor.provider = provider
        ref = actor_factory(Actor)
        for i, q in enumerate(["q1", "q2", "q3"], 1):
            ref.proxy().instruct(q).get()
        session = ref.proxy().get_session().get()
        assert len(session) == 4
        contents = [m["content"] for m in session]
        assert "q1" not in contents
        assert "r1" not in contents
        assert contents[:2] == ["q2", "r2"]

    def test_provider_still_receives_full_context_before_trim(self, actor_factory):
        """Trimming happens AFTER provider.run(), so turn 3 can see turn 1 in messages."""
        provider = FakeProvider(["r1", "r2", "r3"])

        class Actor(AIActor):
            max_history = 2

        Actor.provider = provider
        ref = actor_factory(Actor)
        ref.proxy().instruct("q1").get()
        ref.proxy().instruct("q2").get()
        ref.proxy().instruct("q3").get()
        third_call_messages = provider.calls[2]["messages"]
        contents = [m["content"] for m in third_call_messages]
        assert "q1" in contents

    def test_no_trim_when_at_exact_limit(self, actor_factory):
        provider = FakeProvider(["r1", "r2"])

        class Actor(AIActor):
            max_history = 2

        Actor.provider = provider
        ref = actor_factory(Actor)
        ref.proxy().instruct("q1").get()
        ref.proxy().instruct("q2").get()
        session = ref.proxy().get_session().get()
        assert len(session) == 4


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


class TestMemory:
    def test_memory_empty_on_start(self, actor_factory):
        cls, _ = make_actor()
        ref = actor_factory(cls)
        assert ref.proxy().get_memory().get() == {}

    def test_remember_via_method(self, actor_factory):
        cls, _ = make_actor()
        ref = actor_factory(cls)
        ref.proxy().remember("name", "Alice").get()
        assert ref.proxy().get_memory().get() == {"name": "Alice"}

    def test_remember_via_message(self, actor_factory):
        cls, _ = make_actor()
        ref = actor_factory(cls)
        ref.tell(Remember("color", "blue"))
        memory = ref.proxy().get_memory().get()
        assert memory.get("color") == "blue"

    def test_forget_via_method(self, actor_factory):
        cls, _ = make_actor()
        ref = actor_factory(cls)
        ref.proxy().remember("key", "val").get()
        ref.proxy().forget("key").get()
        assert ref.proxy().get_memory().get() == {}

    def test_forget_via_message(self, actor_factory):
        cls, _ = make_actor()
        ref = actor_factory(cls)
        ref.proxy().remember("key", "val").get()
        ref.tell(Forget("key"))
        memory = ref.proxy().get_memory().get()
        assert "key" not in memory

    def test_forget_missing_key_is_no_op(self, actor_factory):
        cls, _ = make_actor()
        ref = actor_factory(cls)
        ref.proxy().forget("nonexistent").get()  # must not raise

    def test_multiple_facts_stored(self, actor_factory):
        cls, _ = make_actor()
        ref = actor_factory(cls)
        ref.proxy().remember("a", "1").get()
        ref.proxy().remember("b", "2").get()
        memory = ref.proxy().get_memory().get()
        assert memory == {"a": "1", "b": "2"}

    def test_remember_overwrites_existing_key(self, actor_factory):
        cls, _ = make_actor()
        ref = actor_factory(cls)
        ref.proxy().remember("k", "first").get()
        ref.proxy().remember("k", "second").get()
        assert ref.proxy().get_memory().get()["k"] == "second"

    def test_get_memory_returns_copy(self, actor_factory):
        cls, _ = make_actor()
        ref = actor_factory(cls)
        ref.proxy().remember("k", "v").get()
        m = ref.proxy().get_memory().get()
        m["k"] = "mutated"
        assert ref.proxy().get_memory().get()["k"] == "v"

    def test_memory_injected_into_system_prompt(self, actor_factory):
        cls, provider = make_actor()
        ref = actor_factory(cls)
        ref.proxy().remember("user", "Alice").get()
        ref.proxy().instruct("hi").get()
        system = provider.calls[0]["system"]
        assert "user" in system
        assert "Alice" in system

    def test_multiple_memory_items_all_in_prompt(self, actor_factory):
        cls, provider = make_actor()
        ref = actor_factory(cls)
        ref.proxy().remember("lang", "Python").get()
        ref.proxy().remember("role", "engineer").get()
        ref.proxy().instruct("hi").get()
        system = provider.calls[0]["system"]
        assert "Python" in system
        assert "engineer" in system

    def test_empty_memory_does_not_alter_system_prompt(self, actor_factory):
        cls, provider = make_actor()
        ref = actor_factory(cls)
        ref.proxy().instruct("hi").get()
        assert provider.calls[0]["system"] == "Test system"

    def test_base_system_prompt_always_present(self, actor_factory):
        cls, provider = make_actor()
        ref = actor_factory(cls)
        ref.proxy().remember("k", "v").get()
        ref.proxy().instruct("hi").get()
        assert "Test system" in provider.calls[0]["system"]


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------


class TestToolDispatch:
    def test_dispatch_calls_tool_method(self, actor_factory):
        results = []

        provider = ToolCallingFakeProvider("capture", {"value": "xyz"})

        class Actor(AIActor):
            @tool
            def capture(self, value: str) -> str:
                results.append(value)
                return value

        Actor.provider = provider
        ref = actor_factory(Actor)
        ref.proxy().instruct("go").get()
        assert results == ["xyz"]

    def test_dispatch_returns_tool_result_to_provider(self, actor_factory):
        provider = ToolCallingFakeProvider("double", {"x": 7}, "done")

        class Actor(AIActor):
            @tool
            def double(self, x: int) -> int:
                return x * 2

        Actor.provider = provider
        ref = actor_factory(Actor)
        ref.proxy().instruct("go").get()
        assert provider.tool_result == 14

    def test_dispatch_raises_for_unknown_tool_via_instruct(self, actor_factory):
        """_dispatch_tool's ValueError path must be reachable through instruct()."""
        provider = ToolCallingFakeProvider("ghost_tool", {})

        class Actor(AIActor):
            pass

        Actor.provider = provider
        ref = actor_factory(Actor)
        with pytest.raises(Exception, match="ghost_tool"):
            ref.proxy().instruct("go").get()

    def test_dispatch_raises_for_non_tool_method_via_instruct(self, actor_factory):
        """Calling a method not decorated with @tool through the dispatcher raises."""
        provider = ToolCallingFakeProvider("plain_method", {})

        class Actor(AIActor):
            def plain_method(self):
                return "nope"

        Actor.provider = provider
        ref = actor_factory(Actor)
        with pytest.raises(Exception):
            ref.proxy().instruct("go").get()

    def test_provider_reply_is_final_answer(self, actor_factory):
        provider = ToolCallingFakeProvider("noop", {}, final_reply="final answer")

        class Actor(AIActor):
            @tool
            def noop(self) -> str:
                return "ignored"

        Actor.provider = provider
        ref = actor_factory(Actor)
        assert ref.proxy().instruct("go").get() == "final answer"


# ---------------------------------------------------------------------------
# Class-level attributes and defaults
# ---------------------------------------------------------------------------


class TestClassAttributes:
    def test_default_system_prompt(self):
        assert AIActor.system_prompt == "You are a helpful AI agent."

    def test_default_max_tokens(self):
        assert AIActor.max_tokens == 4096

    def test_default_max_history(self):
        assert AIActor.max_history == 0

    def test_custom_system_prompt(self, actor_factory):
        provider = FakeProvider(["r"])

        class Actor(AIActor):
            system_prompt = "Custom prompt"

        Actor.provider = provider
        ref = actor_factory(Actor)
        ref.proxy().instruct("hi").get()
        assert provider.calls[0]["system"] == "Custom prompt"

    def test_provider_can_be_replaced_per_instance(self, actor_factory):
        """pykka proxy attribute assignment (ProxySetAttr) replaces the instance attr."""

        class Actor(AIActor):
            pass

        a_provider = FakeProvider(["from a"])
        b_provider = FakeProvider(["from b"])

        Actor.provider = a_provider
        ref = actor_factory(Actor)
        # pykka's proxy supports attribute setting via ProxySetAttr
        ref.proxy().provider = b_provider
        result = ref.proxy().instruct("hi").get()
        assert result == "from b"


# ---------------------------------------------------------------------------
# on_receive — unknown message type
# ---------------------------------------------------------------------------


class TestOnReceive:
    def test_unknown_message_is_handled_gracefully(self, actor_factory):
        """Unrecognised messages must not crash the actor."""
        cls, _ = make_actor()
        ref = actor_factory(cls)
        ref.tell("this is not a known message type")
        # Actor should still respond normally afterwards
        result = ref.proxy().instruct("alive?").get()
        assert result == "reply"


# ---------------------------------------------------------------------------
# AIActor without a provider (plain pykka actor behaviour)
# ---------------------------------------------------------------------------


class TestNoProvider:
    def test_actor_starts_without_provider(self, actor_factory):
        cls = type("NoProvActor", (AIActor,), {})
        ref = actor_factory(cls)
        assert ref is not None

    def test_instruct_raises_runtime_error_without_provider(self, actor_factory):
        cls = type("NoProvActor2", (AIActor,), {})
        ref = actor_factory(cls)
        with pytest.raises(RuntimeError, match="No provider configured"):
            ref.proxy().instruct("hello").get()

    def test_remember_and_forget_work_without_provider(self, actor_factory):
        cls = type("NoProvActor3", (AIActor,), {})
        ref = actor_factory(cls)
        ref.proxy().remember("k", "v").get()
        assert ref.proxy().get_memory().get() == {"k": "v"}
        ref.proxy().forget("k").get()
        assert ref.proxy().get_memory().get() == {}

    def test_session_methods_work_without_provider(self, actor_factory):
        cls = type("NoProvActor4", (AIActor,), {})
        ref = actor_factory(cls)
        assert ref.proxy().get_session().get() == []
        ref.proxy().clear_session().get()

    def test_provider_none_is_default(self):
        assert AIActor.provider is None


# ---------------------------------------------------------------------------
# Working memory
# ---------------------------------------------------------------------------


class TestWorkingMemory:
    def test_working_memory_empty_on_start(self, actor_factory):
        cls, _ = make_actor()
        ref = actor_factory(cls)
        assert ref.proxy().get_working_memory().get() == {}

    def test_remember_working_stores_fact(self, actor_factory):
        cls, _ = make_actor()
        ref = actor_factory(cls)
        ref.proxy().remember_working("task", "summarise Q3").get()
        assert ref.proxy().get_working_memory().get() == {"task": "summarise Q3"}

    def test_forget_working_removes_fact(self, actor_factory):
        cls, _ = make_actor()
        ref = actor_factory(cls)
        ref.proxy().remember_working("k", "v").get()
        ref.proxy().forget_working("k").get()
        assert ref.proxy().get_working_memory().get() == {}

    def test_forget_working_missing_key_is_no_op(self, actor_factory):
        cls, _ = make_actor()
        ref = actor_factory(cls)
        ref.proxy().forget_working("nonexistent").get()  # must not raise

    def test_clear_working_memory_removes_all(self, actor_factory):
        cls, _ = make_actor()
        ref = actor_factory(cls)
        ref.proxy().remember_working("a", "1").get()
        ref.proxy().remember_working("b", "2").get()
        ref.proxy().clear_working_memory().get()
        assert ref.proxy().get_working_memory().get() == {}

    def test_get_working_memory_returns_copy(self, actor_factory):
        cls, _ = make_actor()
        ref = actor_factory(cls)
        ref.proxy().remember_working("k", "v").get()
        m = ref.proxy().get_working_memory().get()
        m["k"] = "mutated"
        assert ref.proxy().get_working_memory().get()["k"] == "v"

    def test_clear_session_clears_working_memory(self, actor_factory):
        cls, _ = make_actor()
        ref = actor_factory(cls)
        ref.proxy().remember_working("task", "draft").get()
        ref.proxy().clear_session().get()
        assert ref.proxy().get_working_memory().get() == {}

    def test_clear_session_preserves_long_term_memory(self, actor_factory):
        cls, _ = make_actor()
        ref = actor_factory(cls)
        ref.proxy().remember("name", "Alice").get()
        ref.proxy().remember_working("task", "draft").get()
        ref.proxy().clear_session().get()
        assert ref.proxy().get_memory().get() == {"name": "Alice"}
        assert ref.proxy().get_working_memory().get() == {}

    def test_working_memory_injected_into_system_prompt(self, actor_factory):
        cls, provider = make_actor()
        ref = actor_factory(cls)
        ref.proxy().remember_working("goal", "summarise report").get()
        ref.proxy().instruct("hi").get()
        system = provider.calls[0]["system"]
        assert "Working memory" in system
        assert "summarise report" in system

    def test_working_memory_section_absent_when_empty(self, actor_factory):
        cls, provider = make_actor()
        ref = actor_factory(cls)
        ref.proxy().instruct("hi").get()
        assert "Working memory" not in provider.calls[0]["system"]

    def test_both_memory_tiers_in_prompt(self, actor_factory):
        cls, provider = make_actor()
        ref = actor_factory(cls)
        ref.proxy().remember("user", "Alice").get()
        ref.proxy().remember_working("task", "draft report").get()
        ref.proxy().instruct("hi").get()
        system = provider.calls[0]["system"]
        assert "Known facts" in system
        assert "Working memory" in system
        assert "Alice" in system
        assert "draft report" in system


# ---------------------------------------------------------------------------
# Usage tracking
# ---------------------------------------------------------------------------


class TestUsageTracking:
    def test_usage_zero_on_start(self, actor_factory):
        cls, _ = make_actor()
        ref = actor_factory(cls)
        usage = ref.proxy().get_usage().get()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0

    def test_usage_accumulated_after_instruct(self, actor_factory):
        cls, _ = make_actor()
        ref = actor_factory(cls)
        ref.proxy().instruct("hello").get()
        usage = ref.proxy().get_usage().get()
        assert usage.input_tokens > 0
        assert usage.output_tokens > 0

    def test_usage_accumulates_across_calls(self, actor_factory):
        cls, _ = make_actor(replies=["r1", "r2"])
        ref = actor_factory(cls)
        ref.proxy().instruct("first").get()
        u1 = ref.proxy().get_usage().get()
        ref.proxy().instruct("second").get()
        u2 = ref.proxy().get_usage().get()
        assert u2.input_tokens >= u1.input_tokens
        assert u2.total_tokens > u1.total_tokens

    def test_reset_usage_clears_counters(self, actor_factory):
        cls, _ = make_actor()
        ref = actor_factory(cls)
        ref.proxy().instruct("hello").get()
        ref.proxy().reset_usage().get()
        usage = ref.proxy().get_usage().get()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0

    def test_get_usage_returns_snapshot_not_reference(self, actor_factory):
        cls, _ = make_actor()
        ref = actor_factory(cls)
        ref.proxy().instruct("hello").get()
        u = ref.proxy().get_usage().get()
        original_input = u.input_tokens
        ref.proxy().instruct("second call").get()
        assert u.input_tokens == original_input  # snapshot is immutable

    def test_usage_tracked_without_ledger(self, actor_factory):
        cls, _ = make_actor()  # no ledger attached
        ref = actor_factory(cls)
        ref.proxy().instruct("hello").get()
        usage = ref.proxy().get_usage().get()
        assert usage.total_tokens > 0


# ---------------------------------------------------------------------------
# instruct() input types (Path and streams)
# ---------------------------------------------------------------------------


class TestInstructInputTypes:
    def test_plain_string(self, actor_factory):
        cls, provider = make_actor()
        ref = actor_factory(cls)
        ref.proxy().instruct("hello").get()
        assert provider.calls[0]["messages"][-1]["content"] == "hello"

    def test_path_object(self, actor_factory, tmp_path):
        instruction_file = tmp_path / "instruction.txt"
        instruction_file.write_text("hello from file", encoding="utf-8")
        cls, provider = make_actor()
        ref = actor_factory(cls)
        ref.proxy().instruct(instruction_file).get()
        assert provider.calls[0]["messages"][-1]["content"] == "hello from file"

    def test_text_stream(self, actor_factory):
        stream = io.StringIO("hello from stream")
        cls, provider = make_actor()
        ref = actor_factory(cls)
        ref.proxy().instruct(stream).get()
        assert provider.calls[0]["messages"][-1]["content"] == "hello from stream"

    def test_binary_stream(self, actor_factory):
        stream = io.BytesIO(b"hello from bytes")
        cls, provider = make_actor()
        ref = actor_factory(cls)
        ref.proxy().instruct(stream).get()
        assert provider.calls[0]["messages"][-1]["content"] == "hello from bytes"

    def test_path_content_stored_in_session(self, actor_factory, tmp_path):
        instruction_file = tmp_path / "task.txt"
        instruction_file.write_text("file instruction", encoding="utf-8")
        cls, _ = make_actor()
        ref = actor_factory(cls)
        ref.proxy().instruct(instruction_file).get()
        session = ref.proxy().get_session().get()
        assert session[0]["content"] == "file instruction"

    def test_invalid_type_raises_type_error(self, actor_factory):
        cls, _ = make_actor()
        ref = actor_factory(cls)
        with pytest.raises(TypeError, match="instruction must be"):
            ref.proxy().instruct(12345).get()  # type: ignore[arg-type]
