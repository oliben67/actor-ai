"""Tests for monitoring integration: MonitoringContext, LiteLLM provider, AIActor.monitoring."""

# Future imports (must occur at the beginning of the file):
from __future__ import annotations

# Standard library imports:
import json
from unittest.mock import MagicMock, patch

# Local imports:
from actor_ai.accounting import MonitoringContext, UsageSummary
from actor_ai.actor import AIActor
from actor_ai.providers.litellm import LiteLLM, _to_openai_tool
from tests.conftest import FakeProvider

# ---------------------------------------------------------------------------
# MonitoringContext
# ---------------------------------------------------------------------------


class TestMonitoringContext:
    def test_fields(self):
        ctx = MonitoringContext(actor_name="bot", session_id="s1")
        assert ctx.actor_name == "bot"
        assert ctx.session_id == "s1"
        assert ctx.metadata == {}

    def test_custom_metadata(self):
        ctx = MonitoringContext(actor_name="bot", session_id="s1", metadata={"env": "prod"})
        assert ctx.metadata == {"env": "prod"}

    def test_metadata_default_is_independent(self):
        a = MonitoringContext(actor_name="a", session_id="x")
        b = MonitoringContext(actor_name="b", session_id="y")
        a.metadata["key"] = "val"
        assert b.metadata == {}


# ---------------------------------------------------------------------------
# AIActor.monitoring flag
# ---------------------------------------------------------------------------


class TestAIActorMonitoring:
    def test_monitoring_off_by_default(self, actor_factory):
        class Agent(AIActor):
            provider = FakeProvider()

        ref = actor_factory(Agent)
        assert ref.proxy().monitoring.get() is False

    def test_monitoring_on_passes_context_to_provider(self, actor_factory):
        fake = FakeProvider()

        class Agent(AIActor):
            provider = fake
            monitoring = True

        ref = actor_factory(Agent)
        ref.proxy().instruct("hello").get()

        ctx = fake.calls[0]["monitoring_context"]
        assert ctx is not None
        assert isinstance(ctx, MonitoringContext)
        assert ctx.actor_name == "Agent"
        assert ctx.session_id != ""

    def test_monitoring_off_passes_none_context(self, actor_factory):
        fake = FakeProvider()

        class Agent(AIActor):
            provider = fake
            monitoring = False

        ref = actor_factory(Agent)
        ref.proxy().instruct("hello").get()

        assert fake.calls[0]["monitoring_context"] is None

    def test_monitoring_uses_actor_name_when_set(self, actor_factory):
        fake = FakeProvider()

        class Agent(AIActor):
            provider = fake
            monitoring = True
            actor_name = "my-bot"

        ref = actor_factory(Agent)
        ref.proxy().instruct("hello").get()

        ctx = fake.calls[0]["monitoring_context"]
        assert ctx.actor_name == "my-bot"

    def test_monitoring_context_session_id_matches_actor_session(self, actor_factory):
        fake = FakeProvider()

        class Agent(AIActor):
            provider = fake
            monitoring = True

        ref = actor_factory(Agent)
        ref.proxy().instruct("hello").get()
        session_id = ref.proxy().get_session_id().get()

        assert fake.calls[0]["monitoring_context"].session_id == session_id

    def test_monitoring_context_session_id_updates_after_clear(self, actor_factory):
        fake = FakeProvider()

        class Agent(AIActor):
            provider = fake
            monitoring = True

        ref = actor_factory(Agent)
        ref.proxy().instruct("first").get()
        first_ctx = fake.calls[0]["monitoring_context"]

        ref.proxy().clear_session().get()
        ref.proxy().instruct("second").get()
        second_ctx = fake.calls[1]["monitoring_context"]

        assert first_ctx.session_id != second_ctx.session_id


# ---------------------------------------------------------------------------
# LiteLLM provider — unit tests (litellm.completion mocked)
# ---------------------------------------------------------------------------


def _make_litellm_response(content: str, finish_reason: str = "stop", usage=None):
    """Build a minimal mock response that looks like a litellm/openai response."""
    message = MagicMock()
    message.content = content
    message.tool_calls = None

    choice = MagicMock()
    choice.finish_reason = finish_reason
    choice.message = message

    resp = MagicMock()
    resp.choices = [choice]
    if usage is None:
        resp.usage = MagicMock()
        resp.usage.prompt_tokens = 10
        resp.usage.completion_tokens = 5
    else:
        resp.usage = usage
    return resp


def _make_tool_call_response(tool_name: str, tool_args: dict, call_id: str = "call_1"):
    tc = MagicMock()
    tc.id = call_id
    tc.function = MagicMock()
    tc.function.name = tool_name
    tc.function.arguments = json.dumps(tool_args)

    message = MagicMock()
    message.content = None
    message.tool_calls = [tc]

    choice = MagicMock()
    choice.finish_reason = "tool_calls"
    choice.message = message

    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = MagicMock()
    resp.usage.prompt_tokens = 15
    resp.usage.completion_tokens = 8
    return resp


class TestLiteLLMProvider:
    def _provider(self, **kwargs):
        with patch("litellm.completion"):
            return LiteLLM("openai/gpt-4o", **kwargs)

    def test_model_stored(self):
        p = self._provider()
        assert p.model == "openai/gpt-4o"

    def test_simple_reply(self):
        p = self._provider()
        resp = _make_litellm_response("hello back")
        with patch("litellm.completion", return_value=resp) as mock_completion:
            result = p.run("sys", [{"role": "user", "content": "hi"}], [], lambda n, a: None, 100)

        assert result == "hello back"
        mock_completion.assert_called_once()

    def test_on_usage_callback_called(self):
        p = self._provider()
        resp = _make_litellm_response("hi")
        received: list[UsageSummary] = []
        with patch("litellm.completion", return_value=resp):
            p.run("s", [], [], lambda n, a: None, 100, on_usage=received.append)

        assert len(received) == 1
        assert received[0].input_tokens == 10
        assert received[0].output_tokens == 5

    def test_on_usage_none_skipped(self):
        p = self._provider()
        resp = _make_litellm_response("hi")
        with patch("litellm.completion", return_value=resp):
            p.run("s", [], [], lambda n, a: None, 100, on_usage=None)

    def test_usage_none_skipped(self):
        p = self._provider()
        resp = _make_litellm_response("hi")
        resp.usage = None
        received: list[UsageSummary] = []
        with patch("litellm.completion", return_value=resp):
            p.run("s", [], [], lambda n, a: None, 100, on_usage=received.append)
        assert received == []

    def test_monitoring_context_forwarded_as_metadata(self):
        p = self._provider()
        resp = _make_litellm_response("ok")
        ctx = MonitoringContext(actor_name="bot", session_id="sid-1", metadata={"env": "test"})
        with patch("litellm.completion", return_value=resp) as mock_c:
            p.run("s", [], [], lambda n, a: None, 100, monitoring_context=ctx)

        call_kwargs = mock_c.call_args.kwargs
        assert call_kwargs["metadata"] == {
            "actor_name": "bot",
            "session_id": "sid-1",
            "env": "test",
        }

    def test_no_monitoring_context_no_metadata_key(self):
        p = self._provider()
        resp = _make_litellm_response("ok")
        with patch("litellm.completion", return_value=resp) as mock_c:
            p.run("s", [], [], lambda n, a: None, 100)

        call_kwargs = mock_c.call_args.kwargs
        assert "metadata" not in call_kwargs

    def test_tool_call_dispatches_and_continues(self):
        p = self._provider()
        tool_resp = _make_tool_call_response("add", {"x": 1, "y": 2})
        final_resp = _make_litellm_response("result is 3")

        dispatcher = MagicMock(return_value=3)

        with patch("litellm.completion", side_effect=[tool_resp, final_resp]):
            result = p.run("s", [], [], dispatcher, 100)

        dispatcher.assert_called_once_with("add", {"x": 1, "y": 2})
        assert result == "result is 3"

    def test_unknown_finish_reason_returns_empty(self):
        p = self._provider()
        resp = _make_litellm_response("ignored", finish_reason="length")
        with patch("litellm.completion", return_value=resp):
            result = p.run("s", [], [], lambda n, a: None, 100)
        assert result == ""

    def test_system_prompt_prepended(self):
        p = self._provider()
        resp = _make_litellm_response("ok")
        with patch("litellm.completion", return_value=resp) as mock_c:
            p.run("my system", [{"role": "user", "content": "hi"}], [], lambda n, a: None, 100)

        messages = mock_c.call_args.kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "my system"}
        assert messages[1] == {"role": "user", "content": "hi"}

    def test_tools_converted_to_openai_format(self):
        p = self._provider()
        resp = _make_litellm_response("ok")
        tool_spec = {
            "name": "my_tool",
            "description": "does stuff",
            "input_schema": {"type": "object", "properties": {}},
        }
        with patch("litellm.completion", return_value=resp) as mock_c:
            p.run("s", [], [tool_spec], lambda n, a: None, 100)

        tools = mock_c.call_args.kwargs["tools"]
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "my_tool"

    def test_no_tools_no_tools_kwarg(self):
        p = self._provider()
        resp = _make_litellm_response("ok")
        with patch("litellm.completion", return_value=resp) as mock_c:
            p.run("s", [], [], lambda n, a: None, 100)

        assert "tools" not in mock_c.call_args.kwargs

    def test_optional_params_forwarded(self):
        with patch("litellm.completion"):
            p = LiteLLM("openai/gpt-4o", temperature=0.5, top_p=0.9, timeout=30.0, max_retries=2)
        resp = _make_litellm_response("ok")
        with patch("litellm.completion", return_value=resp) as mock_c:
            p.run("s", [], [], lambda n, a: None, 100)

        kw = mock_c.call_args.kwargs
        assert kw["temperature"] == 0.5
        assert kw["top_p"] == 0.9
        assert kw["timeout"] == 30.0
        assert kw["num_retries"] == 2

    def test_api_key_forwarded(self):
        with patch("litellm.completion"):
            p = LiteLLM("openai/gpt-4o", api_key="sk-test")
        resp = _make_litellm_response("ok")
        with patch("litellm.completion", return_value=resp) as mock_c:
            p.run("s", [], [], lambda n, a: None, 100)

        assert mock_c.call_args.kwargs["api_key"] == "sk-test"

    def test_success_callbacks_registered(self):
        # Third party imports:
        import litellm as _litellm

        with patch("litellm.completion"):
            LiteLLM("openai/gpt-4o", success_callbacks=["langfuse"])
        assert "langfuse" in _litellm.success_callback

    def test_failure_callbacks_registered(self):
        # Third party imports:
        import litellm as _litellm

        with patch("litellm.completion"):
            LiteLLM("openai/gpt-4o", failure_callbacks=["sentry"])
        assert "sentry" in _litellm.failure_callback

    def test_none_optional_params_not_forwarded(self):
        p = self._provider()
        resp = _make_litellm_response("ok")
        with patch("litellm.completion", return_value=resp) as mock_c:
            p.run("s", [], [], lambda n, a: None, 100)

        kw = mock_c.call_args.kwargs
        assert "temperature" not in kw
        assert "top_p" not in kw
        assert "timeout" not in kw
        assert "num_retries" not in kw
        assert "api_key" not in kw


# ---------------------------------------------------------------------------
# _to_openai_tool helper
# ---------------------------------------------------------------------------


class TestToOpenAITool:
    def test_converts_spec(self):
        spec = {
            "name": "greet",
            "description": "say hello",
            "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}},
        }
        result = _to_openai_tool(spec)
        assert result == {
            "type": "function",
            "function": {
                "name": "greet",
                "description": "say hello",
                "parameters": spec["input_schema"],
            },
        }

    def test_missing_description_defaults_to_empty(self):
        spec = {"name": "noop", "input_schema": {"type": "object"}}
        result = _to_openai_tool(spec)
        assert result["function"]["description"] == ""


# ---------------------------------------------------------------------------
# Integration: LiteLLM provider via AIActor
# ---------------------------------------------------------------------------


class TestLiteLLMWithAIActor:
    def test_instruct_returns_reply(self, actor_factory):
        resp = _make_litellm_response("42")

        with patch("litellm.completion"):
            p = LiteLLM("openai/gpt-4o")

        class Agent(AIActor):
            provider = p
            monitoring = True

        ref = actor_factory(Agent)
        with patch("litellm.completion", return_value=resp):
            result = ref.proxy().instruct("what is 6x7?").get()

        assert result == "42"

    def test_monitoring_metadata_passed_through_actor(self, actor_factory):
        resp = _make_litellm_response("hi")

        with patch("litellm.completion"):
            p = LiteLLM("openai/gpt-4o")

        class Agent(AIActor):
            provider = p
            actor_name = "integration-agent"
            monitoring = True

        ref = actor_factory(Agent)
        with patch("litellm.completion", return_value=resp) as mock_c:
            ref.proxy().instruct("hello").get()

        metadata = mock_c.call_args.kwargs.get("metadata", {})
        assert metadata["actor_name"] == "integration-agent"
        assert "session_id" in metadata

    def test_no_monitoring_no_metadata(self, actor_factory):
        resp = _make_litellm_response("hi")

        with patch("litellm.completion"):
            p = LiteLLM("openai/gpt-4o")

        class Agent(AIActor):
            provider = p
            monitoring = False

        ref = actor_factory(Agent)
        with patch("litellm.completion", return_value=resp) as mock_c:
            ref.proxy().instruct("hello").get()

        assert "metadata" not in mock_c.call_args.kwargs
