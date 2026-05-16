"""Tests for make_agent() factory function."""

# Future imports (must occur at the beginning of the file):
from __future__ import annotations

# Standard library imports:
import asyncio

# Third party imports:
import pytest

# Local imports:
from actor_ai import AIActor, make_agent, tool
from tests.conftest import FakeProvider, ToolCallingFakeProvider

# ---------------------------------------------------------------------------
# Class structure
# ---------------------------------------------------------------------------


class TestMakeAgentClass:
    def test_returns_aiactor_subclass(self):
        cls = make_agent("Bot", "You are a bot.")
        assert issubclass(cls, AIActor)

    def test_class_name(self):
        cls = make_agent("MyBot", "prompt")
        assert cls.__name__ == "MyBot"

    def test_system_prompt_set(self):
        cls = make_agent("Bot", "Custom prompt")
        assert cls.system_prompt == "Custom prompt"

    def test_provider_set(self):
        provider = FakeProvider(["hi"])
        cls = make_agent("Bot", "prompt", provider)
        assert cls.provider is provider

    def test_provider_defaults_to_none(self):
        cls = make_agent("Bot", "prompt")
        assert cls.provider is None

    def test_actor_name_defaults_to_name(self):
        cls = make_agent("MyBot", "prompt")
        assert cls.actor_name == "MyBot"

    def test_actor_name_explicit_overrides_name(self):
        cls = make_agent("MyBot", "prompt", actor_name="custom-label")
        assert cls.actor_name == "custom-label"

    def test_max_history_forwarded(self):
        cls = make_agent("Bot", "prompt", max_history=5)
        assert cls.max_history == 5

    def test_max_tokens_forwarded(self):
        cls = make_agent("Bot", "prompt", max_tokens=1024)
        assert cls.max_tokens == 1024

    def test_monitoring_forwarded(self):
        cls = make_agent("Bot", "prompt", monitoring=True)
        assert cls.monitoring is True

    def test_each_call_returns_independent_class(self):
        cls_a = make_agent("A", "prompt a")
        cls_b = make_agent("B", "prompt b")
        assert cls_a is not cls_b
        assert cls_a.system_prompt != cls_b.system_prompt


# ---------------------------------------------------------------------------
# Basic instruct
# ---------------------------------------------------------------------------


class TestMakeAgentInstruct:
    def test_can_start_and_instruct(self, actor_factory):
        cls = make_agent("Bot", "Be helpful.", FakeProvider(["hello"]))
        ref = actor_factory(cls)
        assert ref.proxy().instruct("hi").get() == "hello"

    def test_system_prompt_reaches_provider(self, actor_factory):
        provider = FakeProvider(["ok"])
        cls = make_agent("Bot", "Unique prompt xyz", provider)
        ref = actor_factory(cls)
        ref.proxy().instruct("hi").get()
        assert provider.calls[0]["system"] == "Unique prompt xyz"

    def test_no_provider_raises_runtime_error(self, actor_factory):
        cls = make_agent("NoProv", "prompt")
        ref = actor_factory(cls)
        with pytest.raises(RuntimeError, match="No provider configured"):
            ref.proxy().instruct("hi").get()

    def test_session_persists_across_calls(self, actor_factory):
        cls = make_agent("Bot", "prompt", FakeProvider(["a1", "a2"]))
        ref = actor_factory(cls)
        ref.proxy().instruct("q1").get()
        ref.proxy().instruct("q2").get()
        session = ref.proxy().get_session().get()
        assert len(session) == 4

    def test_get_proxy_context_manager(self):
        cls = make_agent("Bot", "prompt", FakeProvider(["ctx reply"]))
        with cls.get_proxy() as proxy:
            reply = proxy.instruct("hello").get()
        assert reply == "ctx reply"

    def test_aget_proxy_async_context_manager(self):
        cls = make_agent("Bot", "prompt", FakeProvider(["async reply"]))

        async def _run():
            async with cls.aget_proxy() as proxy:
                return await asyncio.to_thread(proxy.instruct("hello").get)

        assert asyncio.run(_run()) == "async reply"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class TestMakeAgentTools:
    def test_tool_decorated_function_is_wired(self, actor_factory):
        @tool
        def double(self, x: int) -> int:
            "Double a number."
            return x * 2

        provider = ToolCallingFakeProvider("double", {"x": 5})
        cls = make_agent("Calculator", "Use tools.", provider, tools=[double])
        ref = actor_factory(cls)
        ref.proxy().instruct("go").get()
        assert provider.tool_result == 10

    def test_undecorated_function_auto_wrapped(self, actor_factory):
        def multiply(self, x: int, y: int) -> int:
            "Multiply two numbers."
            return x * y

        provider = ToolCallingFakeProvider("multiply", {"x": 3, "y": 4})
        cls = make_agent("Calculator", "Use tools.", provider, tools=[multiply])
        ref = actor_factory(cls)
        ref.proxy().instruct("go").get()
        assert provider.tool_result == 12

    def test_tool_appears_in_extract_tools(self, actor_factory):
        @tool
        def greet(self, name: str) -> str:
            "Greet by name."
            return f"Hello {name}"

        provider = FakeProvider(["ok"])
        cls = make_agent("Greeter", "prompt", provider, tools=[greet])
        ref = actor_factory(cls)
        ref.proxy().instruct("hi").get()
        tool_names = [t["name"] for t in provider.calls[0]["tools"]]
        assert "greet" in tool_names

    def test_multiple_tools(self, actor_factory):
        @tool
        def add(self, x: int, y: int) -> int:
            "Add."
            return x + y

        @tool
        def subtract(self, x: int, y: int) -> int:
            "Subtract."
            return x - y

        provider = FakeProvider(["ok"])
        cls = make_agent("Math", "Use tools.", provider, tools=[add, subtract])
        ref = actor_factory(cls)
        ref.proxy().instruct("hi").get()
        tool_names = [t["name"] for t in provider.calls[0]["tools"]]
        assert "add" in tool_names
        assert "subtract" in tool_names


# ---------------------------------------------------------------------------
# Sub-agents
# ---------------------------------------------------------------------------


class TestMakeAgentSubAgents:
    def test_sub_agent_wired_as_tool_attribute(self):
        SubCls = make_agent("Sub", "I am sub.", FakeProvider(["sub reply"]))
        cls = make_agent("Parent", "Orchestrate.", None, sub_agents={"helper": SubCls})
        assert hasattr(cls, "helper")
        assert getattr(cls.helper, "_is_ai_tool", False)

    def test_sub_agent_appears_in_tool_specs(self, actor_factory):
        SubCls = make_agent("Sub", "I specialize.", FakeProvider(["sub"]))
        provider = FakeProvider(["ok"])
        cls = make_agent("Parent", "Orchestrate.", provider, sub_agents={"specialist": SubCls})
        ref = actor_factory(cls)
        ref.proxy().instruct("hi").get()
        tool_names = [t["name"] for t in provider.calls[0]["tools"]]
        assert "specialist" in tool_names

    def test_sub_agent_tool_delegates_and_returns_reply(self, actor_factory):
        SubCls = make_agent("Sub", "I research.", FakeProvider(["sub answer"]))
        provider = ToolCallingFakeProvider("researcher", {"instruction": "find it"}, "final")
        cls = make_agent("Parent", "Orchestrate.", provider, sub_agents={"researcher": SubCls})
        ref = actor_factory(cls)
        reply = ref.proxy().instruct("start").get()
        assert reply == "final"
        assert provider.tool_result == "sub answer"

    def test_multiple_sub_agents(self):
        SubA = make_agent("SubA", "A specialist.", FakeProvider(["a"]))
        SubB = make_agent("SubB", "B specialist.", FakeProvider(["b"]))
        cls = make_agent("Parent", "Orchestrate.", None, sub_agents={"a": SubA, "b": SubB})
        assert getattr(cls.a, "_is_ai_tool", False)
        assert getattr(cls.b, "_is_ai_tool", False)

    def test_each_sub_agent_call_is_independent(self, actor_factory):
        calls = []

        @tool
        def record(self, instruction: str) -> str:
            "Record the call."
            calls.append(instruction)
            return "recorded"

        SubCls = make_agent(
            "Sub",
            "Record.",
            ToolCallingFakeProvider("record", {"instruction": "x"}, "done"),
            tools=[record],
        )

        provider = ToolCallingFakeProvider("helper", {"instruction": "task1"}, "final")
        cls = make_agent("Parent", "Use helper.", provider, sub_agents={"helper": SubCls})
        ref = actor_factory(cls)
        ref.proxy().instruct("go").get()
        assert provider.tool_result == "done"

    def test_tools_and_sub_agents_combined(self, actor_factory):
        @tool
        def local_tool(self, x: int) -> int:
            "Local computation."
            return x * 10

        SubCls = make_agent("Sub", "Sub.", FakeProvider(["sub"]))
        provider = FakeProvider(["ok"])
        cls = make_agent(
            "Mixed",
            "Orchestrate.",
            provider,
            tools=[local_tool],
            sub_agents={"delegate": SubCls},
        )
        ref = actor_factory(cls)
        ref.proxy().instruct("hi").get()
        tool_names = [t["name"] for t in provider.calls[0]["tools"]]
        assert "local_tool" in tool_names
        assert "delegate" in tool_names
