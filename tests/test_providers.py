"""Tests for all LLM provider implementations."""

# Future imports (must occur at the beginning of the file):
from __future__ import annotations

# Standard library imports:
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Third party imports:
import pytest

# Local imports:
from actor_ai.providers.anthropic import Claude
from actor_ai.providers.base import LLMProvider
from actor_ai.providers.openai import (
    GPT,
    Copilot,
    DeepSeek,
    Gemini,
    Mistral,
    _to_openai_tool,
    _token_from_vscode_keyring,
)

# ---------------------------------------------------------------------------
# Helpers — build lightweight mock responses without using the real SDKs
# ---------------------------------------------------------------------------


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(text=text)


def tool_use_block(name: str, input: dict, id: str = "tu_1") -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", name=name, input=input, id=id)


def anthropic_response(stop_reason: str, *blocks) -> SimpleNamespace:
    return SimpleNamespace(stop_reason=stop_reason, content=list(blocks))


def openai_choice(finish_reason: str, content: str | None = None, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    return SimpleNamespace(finish_reason=finish_reason, message=message)


def openai_response(*choices) -> SimpleNamespace:
    return SimpleNamespace(choices=list(choices))


def openai_tool_call(name: str, args: dict, id: str = "call_1"):
    fn = SimpleNamespace(name=name, arguments=json.dumps(args))
    return SimpleNamespace(id=id, function=fn)


# ---------------------------------------------------------------------------
# LLMProvider ABC
# ---------------------------------------------------------------------------


class TestLLMProviderABC:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            LLMProvider()  # type: ignore[abstract]

    def test_concrete_subclass_must_implement_run(self):
        class Incomplete(LLMProvider):
            model = "x"

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_concrete_subclass_can_be_instantiated(self):
        class Complete(LLMProvider):
            model = "x"

            def run(self, system, messages, tools, dispatcher, max_tokens):
                return "ok"

        assert Complete().run("s", [], [], lambda n, a: None, 100) == "ok"


# ---------------------------------------------------------------------------
# Claude provider
# ---------------------------------------------------------------------------


class TestClaude:
    def _provider(self, replies):
        """Build a Claude provider with a mocked Anthropic client."""
        with patch("actor_ai.providers.anthropic.Anthropic"):
            provider = Claude()
        provider._client = MagicMock()
        provider._client.messages.create.side_effect = replies
        return provider

    def test_default_model(self):
        with patch("actor_ai.providers.anthropic.Anthropic"):
            p = Claude()
        assert p.model == "claude-sonnet-4-6"

    def test_custom_model(self):
        with patch("actor_ai.providers.anthropic.Anthropic"):
            p = Claude("claude-opus-4-7")
        assert p.model == "claude-opus-4-7"

    def test_is_llm_provider(self):
        with patch("actor_ai.providers.anthropic.Anthropic"):
            p = Claude()
        assert isinstance(p, LLMProvider)

    def test_run_end_turn_returns_text(self):
        provider = self._provider([anthropic_response("end_turn", text_block("Hello!"))])
        result = provider.run(
            "sys", [{"role": "user", "content": "hi"}], [], lambda n, a: None, 1024
        )
        assert result == "Hello!"

    def test_run_returns_empty_string_when_no_text_block(self):
        provider = self._provider([anthropic_response("end_turn")])
        result = provider.run(
            "sys", [{"role": "user", "content": "hi"}], [], lambda n, a: None, 1024
        )
        assert result == ""

    def test_run_tool_use_then_end_turn(self):
        dispatcher = MagicMock(return_value=42)
        provider = self._provider(
            [
                anthropic_response("tool_use", tool_use_block("add", {"a": 1, "b": 2})),
                anthropic_response("end_turn", text_block("The answer is 42.")),
            ]
        )
        result = provider.run("sys", [{"role": "user", "content": "add"}], [], dispatcher, 1024)
        assert result == "The answer is 42."
        dispatcher.assert_called_once_with("add", {"a": 1, "b": 2})

    def test_run_multiple_tool_calls_in_one_turn(self):
        dispatcher = MagicMock(return_value="done")
        provider = self._provider(
            [
                anthropic_response(
                    "tool_use",
                    tool_use_block("tool_a", {}, "id_a"),
                    tool_use_block("tool_b", {"x": 1}, "id_b"),
                ),
                anthropic_response("end_turn", text_block("Both done.")),
            ]
        )
        provider.run("sys", [], [], dispatcher, 1024)
        assert dispatcher.call_count == 2

    def test_run_does_not_pass_tools_when_empty(self):
        provider = self._provider([anthropic_response("end_turn", text_block("ok"))])
        provider.run("sys", [], [], lambda n, a: None, 1024)
        call_kwargs = provider._client.messages.create.call_args[1]
        assert "tools" not in call_kwargs

    def test_run_passes_tools_when_present(self):
        spec = {
            "name": "f",
            "description": "d",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        }
        provider = self._provider([anthropic_response("end_turn", text_block("ok"))])
        provider.run("sys", [], [spec], lambda n, a: None, 1024)
        call_kwargs = provider._client.messages.create.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] == [spec]

    def test_run_does_not_mutate_input_messages(self):
        provider = self._provider([anthropic_response("end_turn", text_block("ok"))])
        original = [{"role": "user", "content": "hello"}]
        snapshot = list(original)
        provider.run("sys", original, [], lambda n, a: None, 1024)
        assert original == snapshot

    def test_run_unknown_stop_reason_returns_empty(self):
        provider = self._provider([anthropic_response("unknown_reason")])
        result = provider.run("sys", [], [], lambda n, a: None, 1024)
        assert result == ""

    def test_run_passes_system_and_model(self):
        provider = self._provider([anthropic_response("end_turn", text_block("ok"))])
        provider.run("My system prompt", [], [], lambda n, a: None, 512)
        kw = provider._client.messages.create.call_args[1]
        assert kw["system"] == "My system prompt"
        assert kw["model"] == "claude-sonnet-4-6"
        assert kw["max_tokens"] == 512

    def test_run_tool_result_message_structure(self):
        """Tool results must be structured as Anthropic tool_result content blocks."""
        dispatcher = MagicMock(return_value="result_value")
        provider = self._provider(
            [
                anthropic_response("tool_use", tool_use_block("my_tool", {}, "tu_abc")),
                anthropic_response("end_turn", text_block("ok")),
            ]
        )
        provider.run("sys", [], [], dispatcher, 1024)
        second_call_messages = provider._client.messages.create.call_args_list[1][1]["messages"]
        tool_result_msg = second_call_messages[-1]
        assert tool_result_msg["role"] == "user"
        assert tool_result_msg["content"][0]["type"] == "tool_result"
        assert tool_result_msg["content"][0]["tool_use_id"] == "tu_abc"
        assert tool_result_msg["content"][0]["content"] == "result_value"


# ---------------------------------------------------------------------------
# _to_openai_tool conversion
# ---------------------------------------------------------------------------


class TestToOpenAITool:
    def test_wraps_in_function_type(self):
        spec = {
            "name": "f",
            "description": "d",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        }
        result = _to_openai_tool(spec)
        assert result["type"] == "function"

    def test_name_preserved(self):
        spec = {
            "name": "my_func",
            "description": "",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        }
        assert _to_openai_tool(spec)["function"]["name"] == "my_func"

    def test_description_preserved(self):
        spec = {
            "name": "f",
            "description": "Does something",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        }
        assert _to_openai_tool(spec)["function"]["description"] == "Does something"

    def test_input_schema_becomes_parameters(self):
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
        spec = {"name": "f", "description": "", "input_schema": schema}
        result = _to_openai_tool(spec)
        assert result["function"]["parameters"] == schema
        assert "input_schema" not in result["function"]

    def test_missing_description_defaults_to_empty(self):
        spec = {"name": "f", "input_schema": {"type": "object", "properties": {}, "required": []}}
        result = _to_openai_tool(spec)
        assert result["function"]["description"] == ""


# ---------------------------------------------------------------------------
# _OpenAICompatible / GPT provider
# ---------------------------------------------------------------------------


class TestGPT:
    def _provider(self, responses):
        with patch("actor_ai.providers.openai.OpenAI"):
            provider = GPT()
        provider._client = MagicMock()
        provider._client.chat.completions.create.side_effect = responses
        return provider

    def test_default_model(self):
        with patch("actor_ai.providers.openai.OpenAI"):
            p = GPT()
        assert p.model == "gpt-4o"

    def test_custom_model(self):
        with patch("actor_ai.providers.openai.OpenAI"):
            p = GPT("gpt-4o-mini")
        assert p.model == "gpt-4o-mini"

    def test_is_llm_provider(self):
        with patch("actor_ai.providers.openai.OpenAI"):
            p = GPT()
        assert isinstance(p, LLMProvider)

    def test_run_stop_returns_content(self):
        provider = self._provider([openai_response(openai_choice("stop", content="Hello!"))])
        result = provider.run(
            "sys", [{"role": "user", "content": "hi"}], [], lambda n, a: None, 1024
        )
        assert result == "Hello!"

    def test_run_stop_none_content_returns_empty(self):
        provider = self._provider([openai_response(openai_choice("stop", content=None))])
        result = provider.run("sys", [], [], lambda n, a: None, 1024)
        assert result == ""

    def test_run_tool_calls_then_stop(self):
        dispatcher = MagicMock(return_value=99)
        tc = openai_tool_call("multiply", {"a": 3, "b": 33})
        provider = self._provider(
            [
                openai_response(openai_choice("tool_calls", tool_calls=[tc])),
                openai_response(openai_choice("stop", content="99")),
            ]
        )
        result = provider.run("sys", [], [], dispatcher, 1024)
        assert result == "99"
        dispatcher.assert_called_once_with("multiply", {"a": 3, "b": 33})

    def test_run_multiple_tool_calls(self):
        dispatcher = MagicMock(return_value="r")
        tc1 = openai_tool_call("a", {}, "c1")
        tc2 = openai_tool_call("b", {"x": 1}, "c2")
        provider = self._provider(
            [
                openai_response(openai_choice("tool_calls", tool_calls=[tc1, tc2])),
                openai_response(openai_choice("stop", content="done")),
            ]
        )
        provider.run("sys", [], [], dispatcher, 1024)
        assert dispatcher.call_count == 2

    def test_run_prepends_system_message(self):
        provider = self._provider([openai_response(openai_choice("stop", content="ok"))])
        provider.run("MySys", [{"role": "user", "content": "hi"}], [], lambda n, a: None, 1024)
        messages = provider._client.chat.completions.create.call_args[1]["messages"]
        assert messages[0] == {"role": "system", "content": "MySys"}

    def test_run_does_not_pass_tools_when_empty(self):
        provider = self._provider([openai_response(openai_choice("stop", content="ok"))])
        provider.run("sys", [], [], lambda n, a: None, 1024)
        kw = provider._client.chat.completions.create.call_args[1]
        assert "tools" not in kw

    def test_run_passes_converted_tools(self):
        spec = {
            "name": "f",
            "description": "d",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        }
        provider = self._provider([openai_response(openai_choice("stop", content="ok"))])
        provider.run("sys", [], [spec], lambda n, a: None, 1024)
        kw = provider._client.chat.completions.create.call_args[1]
        assert "tools" in kw
        assert kw["tools"][0]["type"] == "function"

    def test_run_does_not_mutate_input_messages(self):
        provider = self._provider([openai_response(openai_choice("stop", content="ok"))])
        original = [{"role": "user", "content": "hi"}]
        snapshot = list(original)
        provider.run("sys", original, [], lambda n, a: None, 1024)
        assert original == snapshot

    def test_run_unknown_finish_reason_returns_empty(self):
        provider = self._provider([openai_response(openai_choice("content_filter", content=None))])
        result = provider.run("sys", [], [], lambda n, a: None, 1024)
        assert result == ""

    def test_run_tool_result_message_structure(self):
        """Tool results must be {'role': 'tool', 'tool_call_id': ..., 'content': ...}."""
        dispatcher = MagicMock(return_value="42")
        tc = openai_tool_call("f", {}, "call_xyz")
        provider = self._provider(
            [
                openai_response(openai_choice("tool_calls", tool_calls=[tc])),
                openai_response(openai_choice("stop", content="done")),
            ]
        )
        provider.run("sys", [], [], dispatcher, 1024)
        second_messages = provider._client.chat.completions.create.call_args_list[1][1]["messages"]
        tool_result = second_messages[-1]
        assert tool_result["role"] == "tool"
        assert tool_result["tool_call_id"] == "call_xyz"
        assert tool_result["content"] == "42"


# ---------------------------------------------------------------------------
# OpenAI-compatible subclasses — model/URL defaults
# ---------------------------------------------------------------------------


class TestProviderDefaults:
    @pytest.mark.parametrize(
        "cls, expected_model",
        [
            (GPT, "gpt-4o"),
            (Gemini, "gemini-2.0-flash"),
            (Mistral, "mistral-large-latest"),
            (DeepSeek, "deepseek-chat"),
            (Copilot, "gpt-4o"),
        ],
    )
    def test_default_model(self, cls, expected_model):
        with patch("actor_ai.providers.openai.OpenAI"):
            p = cls()
        assert p.model == expected_model

    def test_gemini_base_url(self):
        captured = {}
        with patch("actor_ai.providers.openai.OpenAI") as mock_openai:
            mock_openai.side_effect = lambda **kw: captured.update(kw) or MagicMock()
            Gemini()
        assert "generativelanguage.googleapis.com" in captured.get("base_url", "")

    def test_mistral_base_url(self):
        captured = {}
        with patch("actor_ai.providers.openai.OpenAI") as mock_openai:
            mock_openai.side_effect = lambda **kw: captured.update(kw) or MagicMock()
            Mistral()
        assert "mistral.ai" in captured.get("base_url", "")

    def test_deepseek_base_url(self):
        captured = {}
        with patch("actor_ai.providers.openai.OpenAI") as mock_openai:
            mock_openai.side_effect = lambda **kw: captured.update(kw) or MagicMock()
            DeepSeek()
        assert "deepseek.com" in captured.get("base_url", "")

    def test_gemini_uses_google_api_key_env(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "gkey-test")
        captured = {}
        with patch("actor_ai.providers.openai.OpenAI") as mock_openai:
            mock_openai.side_effect = lambda **kw: captured.update(kw) or MagicMock()
            Gemini()
        assert captured.get("api_key") == "gkey-test"

    def test_mistral_uses_mistral_api_key_env(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "mkey-test")
        captured = {}
        with patch("actor_ai.providers.openai.OpenAI") as mock_openai:
            mock_openai.side_effect = lambda **kw: captured.update(kw) or MagicMock()
            Mistral()
        assert captured.get("api_key") == "mkey-test"

    def test_deepseek_uses_deepseek_api_key_env(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "dkey-test")
        captured = {}
        with patch("actor_ai.providers.openai.OpenAI") as mock_openai:
            mock_openai.side_effect = lambda **kw: captured.update(kw) or MagicMock()
            DeepSeek()
        assert captured.get("api_key") == "dkey-test"

    def test_explicit_api_key_takes_precedence(self):
        captured = {}
        with patch("actor_ai.providers.openai.OpenAI") as mock_openai:
            mock_openai.side_effect = lambda **kw: captured.update(kw) or MagicMock()
            GPT(api_key="explicit-key")
        assert captured.get("api_key") == "explicit-key"

    def test_claude_uses_anthropic_api_key_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-test")
        captured = {}
        with patch("actor_ai.providers.anthropic.Anthropic") as mock_ant:
            mock_ant.side_effect = lambda **kw: captured.update(kw) or MagicMock()
            Claude()
        assert captured.get("api_key") == "ant-test"

    def test_claude_explicit_api_key(self):
        captured = {}
        with patch("actor_ai.providers.anthropic.Anthropic") as mock_ant:
            mock_ant.side_effect = lambda **kw: captured.update(kw) or MagicMock()
            Claude(api_key="explicit-ant-key")
        assert captured.get("api_key") == "explicit-ant-key"


# ---------------------------------------------------------------------------
# Provider configuration parameters
# ---------------------------------------------------------------------------


class TestClaudeConfiguration:
    def _make(self, **kwargs):
        with patch("actor_ai.providers.anthropic.Anthropic"):
            p = Claude(**kwargs)
        p._client = MagicMock()
        p._client.messages.create.return_value = anthropic_response("end_turn", text_block("ok"))
        return p

    def test_temperature_forwarded(self):
        p = self._make(temperature=0.3)
        p.run("s", [], [], lambda n, a: None, 100)
        assert p._client.messages.create.call_args[1]["temperature"] == 0.3

    def test_top_p_forwarded(self):
        p = self._make(top_p=0.9)
        p.run("s", [], [], lambda n, a: None, 100)
        assert p._client.messages.create.call_args[1]["top_p"] == 0.9

    def test_top_k_forwarded(self):
        p = self._make(top_k=40)
        p.run("s", [], [], lambda n, a: None, 100)
        assert p._client.messages.create.call_args[1]["top_k"] == 40

    def test_stop_sequences_forwarded(self):
        p = self._make(stop_sequences=["STOP", "END"])
        p.run("s", [], [], lambda n, a: None, 100)
        assert p._client.messages.create.call_args[1]["stop_sequences"] == ["STOP", "END"]

    def test_none_params_not_forwarded(self):
        p = self._make()
        p.run("s", [], [], lambda n, a: None, 100)
        kw = p._client.messages.create.call_args[1]
        for param in ("temperature", "top_p", "top_k", "stop_sequences"):
            assert param not in kw

    def test_timeout_passed_to_client_constructor(self):
        captured = {}
        with patch("actor_ai.providers.anthropic.Anthropic") as mock_ant:
            mock_ant.side_effect = lambda **kw: captured.update(kw) or MagicMock()
            Claude(timeout=15.0)
        assert captured.get("timeout") == 15.0


class TestGPTConfiguration:
    def _make(self, **kwargs):
        with patch("actor_ai.providers.openai.OpenAI"):
            p = GPT(**kwargs)
        p._client = MagicMock()
        p._client.chat.completions.create.return_value = openai_response(
            openai_choice("stop", content="ok")
        )
        return p

    def test_temperature_forwarded(self):
        p = self._make(temperature=0.7)
        p.run("s", [], [], lambda n, a: None, 100)
        assert p._client.chat.completions.create.call_args[1]["temperature"] == 0.7

    def test_top_p_forwarded(self):
        p = self._make(top_p=0.95)
        p.run("s", [], [], lambda n, a: None, 100)
        assert p._client.chat.completions.create.call_args[1]["top_p"] == 0.95

    def test_frequency_penalty_forwarded(self):
        p = self._make(frequency_penalty=0.5)
        p.run("s", [], [], lambda n, a: None, 100)
        assert p._client.chat.completions.create.call_args[1]["frequency_penalty"] == 0.5

    def test_presence_penalty_forwarded(self):
        p = self._make(presence_penalty=0.2)
        p.run("s", [], [], lambda n, a: None, 100)
        assert p._client.chat.completions.create.call_args[1]["presence_penalty"] == 0.2

    def test_seed_forwarded(self):
        p = self._make(seed=42)
        p.run("s", [], [], lambda n, a: None, 100)
        assert p._client.chat.completions.create.call_args[1]["seed"] == 42

    def test_stop_forwarded(self):
        p = self._make(stop=["###"])
        p.run("s", [], [], lambda n, a: None, 100)
        assert p._client.chat.completions.create.call_args[1]["stop"] == ["###"]

    def test_response_format_forwarded(self):
        fmt = {"type": "json_object"}
        p = self._make(response_format=fmt)
        p.run("s", [], [], lambda n, a: None, 100)
        assert p._client.chat.completions.create.call_args[1]["response_format"] == fmt

    def test_none_params_not_forwarded(self):
        p = self._make()
        p.run("s", [], [], lambda n, a: None, 100)
        kw = p._client.chat.completions.create.call_args[1]
        for param in (
            "temperature",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "seed",
            "stop",
            "response_format",
        ):
            assert param not in kw

    def test_timeout_passed_to_client_constructor(self):
        captured = {}
        with patch("actor_ai.providers.openai.OpenAI") as mock_oai:
            mock_oai.side_effect = lambda **kw: captured.update(kw) or MagicMock()
            GPT(timeout=20.0)
        assert captured.get("timeout") == 20.0


# ---------------------------------------------------------------------------
# _token_from_vscode_keyring
# ---------------------------------------------------------------------------


class TestTokenFromVscodeKeyring:
    def test_returns_none_when_keyring_not_installed(self):
        with patch.dict("sys.modules", {"keyring": None}):
            assert _token_from_vscode_keyring() is None

    def test_returns_none_when_no_entry_found(self):
        mock_keyring = MagicMock()
        mock_keyring.get_password.return_value = None
        with patch.dict("sys.modules", {"keyring": mock_keyring}):
            assert _token_from_vscode_keyring() is None

    def test_returns_plain_token_string(self):
        mock_keyring = MagicMock()
        mock_keyring.get_password.side_effect = lambda svc, acct: (
            "ghp_plain_token" if svc == "vscode.github-authentication" else None
        )
        with patch.dict("sys.modules", {"keyring": mock_keyring}):
            assert _token_from_vscode_keyring() == "ghp_plain_token"

    def test_extracts_password_field_from_json(self):
        import json

        mock_keyring = MagicMock()
        mock_keyring.get_password.return_value = json.dumps(
            {"account": "oliben67", "password": "ghp_from_json_password"}
        )
        with patch.dict("sys.modules", {"keyring": mock_keyring}):
            assert _token_from_vscode_keyring() == "ghp_from_json_password"

    def test_extracts_token_field_from_json(self):
        import json

        mock_keyring = MagicMock()
        mock_keyring.get_password.return_value = json.dumps({"token": "ghp_from_json_token"})
        with patch.dict("sys.modules", {"keyring": mock_keyring}):
            assert _token_from_vscode_keyring() == "ghp_from_json_token"

    def test_tries_second_service_when_first_returns_none(self):
        mock_keyring = MagicMock()
        mock_keyring.get_password.side_effect = lambda svc, acct: (
            "ghp_second_service" if svc == "GitHub.github.com" else None
        )
        with patch.dict("sys.modules", {"keyring": mock_keyring}):
            assert _token_from_vscode_keyring() == "ghp_second_service"


# ---------------------------------------------------------------------------
# Copilot provider
# ---------------------------------------------------------------------------


class TestCopilotProvider:
    def _make(self, **kwargs) -> Copilot:
        with patch("actor_ai.providers.openai.OpenAI"):
            p = Copilot(**kwargs)
        p._client = MagicMock()
        p._client.chat.completions.create.return_value = openai_response(
            openai_choice("stop", content="ok")
        )
        return p

    def _captured_init(self, **kwargs) -> dict:
        captured: dict = {}
        with patch("actor_ai.providers.openai.OpenAI") as mock_oai:
            mock_oai.side_effect = lambda **kw: captured.update(kw) or MagicMock()
            Copilot(**kwargs)
        return captured

    # -- model validation --

    def test_invalid_model_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported Copilot model"):
            with patch("actor_ai.providers.openai.OpenAI"):
                Copilot("gpt-3.5-turbo")

    def test_error_message_lists_valid_models(self):
        with pytest.raises(ValueError) as exc:
            with patch("actor_ai.providers.openai.OpenAI"):
                Copilot("bad-model")
        msg = str(exc.value)
        for m in Copilot.MODELS:
            assert m in msg

    def test_all_valid_models_accepted(self):
        for model in Copilot.MODELS:
            with patch("actor_ai.providers.openai.OpenAI"):
                p = Copilot(model)
            assert p.model == model

    def test_models_class_attribute_is_frozenset(self):
        assert isinstance(Copilot.MODELS, frozenset)
        assert len(Copilot.MODELS) > 0

    def test_copilot_model_literal_importable(self):
        # Local imports:
        from actor_ai import CopilotModel as CM

        assert CM is not None

    # -- constructor / identity --

    def test_default_model(self):
        p = self._make()
        assert p.model == "gpt-4o"

    def test_custom_model(self):
        p = self._make(model="claude-sonnet-4-5")
        assert p.model == "claude-sonnet-4-5"

    def test_base_url_is_githubcopilot(self):
        captured = self._captured_init()
        assert captured.get("base_url") == "https://api.githubcopilot.com"

    def test_integration_header_sent(self):
        captured = self._captured_init()
        headers = captured.get("default_headers", {})
        assert headers.get("Copilot-Integration-Id") == "vscode-chat"

    def test_uses_github_token_env(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_test123")
        with patch("actor_ai.providers.openai.subprocess.run") as mock_sub:
            captured = self._captured_init()
        assert captured.get("api_key") == "ghp_test123"
        mock_sub.assert_not_called()

    def test_explicit_api_key_takes_precedence(self, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        with patch("actor_ai.providers.openai.subprocess.run") as mock_sub:
            captured = self._captured_init(api_key="ghp_explicit")
        assert captured.get("api_key") == "ghp_explicit"
        mock_sub.assert_not_called()

    def test_falls_back_to_gh_cli(self, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "ghp_from_cli\n"
        captured: dict = {}
        with patch("actor_ai.providers.openai.subprocess.run", return_value=mock_result):
            with patch("actor_ai.providers.openai.OpenAI") as mock_oai:
                mock_oai.side_effect = lambda **kw: captured.update(kw) or MagicMock()
                Copilot()
        assert captured.get("api_key") == "ghp_from_cli"

    def test_gh_cli_not_found_raises_clear_error(self, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        with patch("actor_ai.providers.openai.subprocess.run", side_effect=FileNotFoundError):
            with patch("actor_ai.providers.openai._token_from_vscode_keyring", return_value=None):
                with pytest.raises(ValueError, match="No GitHub token found for Copilot"):
                    with patch("actor_ai.providers.openai.OpenAI"):
                        Copilot()

    def test_gh_cli_nonzero_exit_raises_clear_error(self, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        with patch("actor_ai.providers.openai.subprocess.run", return_value=mock_result):
            with patch("actor_ai.providers.openai._token_from_vscode_keyring", return_value=None):
                with pytest.raises(ValueError, match="No GitHub token found for Copilot"):
                    with patch("actor_ai.providers.openai.OpenAI"):
                        Copilot()

    def test_error_message_lists_all_auth_options(self, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        with patch("actor_ai.providers.openai.subprocess.run", side_effect=FileNotFoundError):
            with patch("actor_ai.providers.openai._token_from_vscode_keyring", return_value=None):
                with pytest.raises(ValueError) as exc:
                    with patch("actor_ai.providers.openai.OpenAI"):
                        Copilot()
        msg = str(exc.value)
        assert "GITHUB_TOKEN" in msg
        assert "gh auth login" in msg
        assert "keyring" in msg

    def test_falls_back_to_vscode_keyring(self, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        captured: dict = {}
        with patch("actor_ai.providers.openai.subprocess.run", return_value=mock_result):
            with patch(
                "actor_ai.providers.openai._token_from_vscode_keyring",
                return_value="ghp_from_vscode",
            ):
                with patch("actor_ai.providers.openai.OpenAI") as mock_oai:
                    mock_oai.side_effect = lambda **kw: captured.update(kw) or MagicMock()
                    Copilot()
        assert captured.get("api_key") == "ghp_from_vscode"

    def test_timeout_passed_to_client_constructor(self):
        captured = self._captured_init(timeout=30.0)
        assert captured.get("timeout") == 30.0

    # -- run() forwards parameters correctly --

    def test_run_returns_reply(self):
        p = self._make()
        assert p.run("sys", [], [], lambda n, a: None, 100) == "ok"

    def test_temperature_forwarded(self):
        p = self._make(temperature=0.3)
        p.run("s", [], [], lambda n, a: None, 100)
        assert p._client.chat.completions.create.call_args[1]["temperature"] == 0.3

    def test_top_p_forwarded(self):
        p = self._make(top_p=0.9)
        p.run("s", [], [], lambda n, a: None, 100)
        assert p._client.chat.completions.create.call_args[1]["top_p"] == 0.9

    def test_seed_forwarded(self):
        p = self._make(seed=7)
        p.run("s", [], [], lambda n, a: None, 100)
        assert p._client.chat.completions.create.call_args[1]["seed"] == 7

    def test_stop_forwarded(self):
        p = self._make(stop=["END"])
        p.run("s", [], [], lambda n, a: None, 100)
        assert p._client.chat.completions.create.call_args[1]["stop"] == ["END"]

    def test_none_params_not_forwarded(self):
        p = self._make()
        p.run("s", [], [], lambda n, a: None, 100)
        kw = p._client.chat.completions.create.call_args[1]
        for param in ("temperature", "top_p", "seed", "stop"):
            assert param not in kw

    def test_system_message_prepended(self):
        p = self._make()
        p.run("be helpful", [{"role": "user", "content": "hi"}], [], lambda n, a: None, 100)
        messages = p._client.chat.completions.create.call_args[1]["messages"]
        assert messages[0] == {"role": "system", "content": "be helpful"}
        assert messages[1] == {"role": "user", "content": "hi"}

    def test_tool_call_round_trip(self):
        tool_call = SimpleNamespace(
            id="tc1",
            function=SimpleNamespace(name="double", arguments='{"n": 5}'),
        )
        tool_resp = openai_response(openai_choice("tool_calls", tool_calls=[tool_call]))
        final_resp = openai_response(openai_choice("stop", content="result is 10"))

        with patch("actor_ai.providers.openai.OpenAI"):
            p = Copilot()
        p._client = MagicMock()
        p._client.chat.completions.create.side_effect = [tool_resp, final_resp]

        result = p.run("s", [], [], lambda n, a: a["n"] * 2, 100)
        assert result == "result is 10"
        assert p._client.chat.completions.create.call_count == 2

    def test_usage_callback_called(self):
        p = self._make()
        resp = openai_response(openai_choice("stop", content="ok"))
        resp.usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
        p._client.chat.completions.create.return_value = resp
        received = []
        p.run("s", [], [], lambda n, a: None, 100, on_usage=received.append)
        assert len(received) == 1
        assert received[0].input_tokens == 10
        assert received[0].output_tokens == 5

    def test_monitoring_context_ignored(self):
        # Local imports:
        from actor_ai.accounting import MonitoringContext

        p = self._make()
        ctx = MonitoringContext(actor_name="test", session_id="s1")
        result = p.run("s", [], [], lambda n, a: None, 100, monitoring_context=ctx)
        assert result == "ok"

    # -- exported from top-level package --

    def test_copilot_importable_from_actor_ai(self):
        # Local imports:
        from actor_ai import Copilot as CopilotPublic

        assert CopilotPublic is Copilot
