# Future imports (must occur at the beginning of the file):
from __future__ import annotations

# Standard library imports:
import json
import os
import subprocess
from collections.abc import Callable
from typing import Literal

# Third party imports:
from openai import OpenAI

from ..accounting import MonitoringContext, UsageSummary
from .base import LLMProvider


class _OpenAICompatible(LLMProvider):
    """Base for all OpenAI-compatible providers (GPT, Gemini, Mistral, DeepSeek).

    Converts tool specs from Anthropic's canonical ``input_schema`` format to
    OpenAI's ``parameters`` format internally.

    Configuration parameters
    ------------------------
    temperature : float, optional
        Sampling temperature in ``[0, 2]``.
    top_p : float, optional
        Nucleus-sampling probability mass.
    frequency_penalty : float, optional
        Penalise tokens based on their frequency in the text so far.
    presence_penalty : float, optional
        Penalise tokens based on whether they have appeared in the text.
    timeout : float, optional
        HTTP request timeout in seconds.
    stop : list[str] | str, optional
        Sequences that stop generation.
    seed : int, optional
        Random seed for deterministic sampling (best-effort).
    response_format : dict, optional
        e.g. ``{"type": "json_object"}`` to force JSON output.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        timeout: float | None = None,
        stop: list[str] | str | None = None,
        seed: int | None = None,
        response_format: dict | None = None,
        default_headers: dict | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop
        self.seed = seed
        self.response_format = response_format
        self._client = OpenAI(
            api_key=api_key or os.environ.get(api_key_env),
            base_url=base_url,
            timeout=timeout,
            default_headers=default_headers,
        )

    def run(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
        dispatcher: Callable[[str, dict], object],
        max_tokens: int,
        *,
        on_usage: Callable[[UsageSummary], None] | None = None,
        monitoring_context: MonitoringContext | None = None,
    ) -> str:
        oai_messages: list = [{"role": "system", "content": system}, *messages]
        oai_tools = [_to_openai_tool(t) for t in tools] if tools else []

        while True:
            kwargs: dict = {
                "model": self.model,
                "messages": oai_messages,
                "max_tokens": max_tokens,
            }
            if oai_tools:
                kwargs["tools"] = oai_tools
            if self.temperature is not None:
                kwargs["temperature"] = self.temperature
            if self.top_p is not None:
                kwargs["top_p"] = self.top_p
            if self.frequency_penalty is not None:
                kwargs["frequency_penalty"] = self.frequency_penalty
            if self.presence_penalty is not None:
                kwargs["presence_penalty"] = self.presence_penalty
            if self.stop is not None:
                kwargs["stop"] = self.stop
            if self.seed is not None:
                kwargs["seed"] = self.seed
            if self.response_format is not None:
                kwargs["response_format"] = self.response_format

            response = self._client.chat.completions.create(**kwargs)

            if on_usage is not None and response.usage is not None:
                on_usage(
                    UsageSummary(
                        input_tokens=response.usage.prompt_tokens or 0,
                        output_tokens=response.usage.completion_tokens or 0,
                    )
                )

            choice = response.choices[0]

            if choice.finish_reason == "stop":
                return choice.message.content or ""

            if choice.finish_reason == "tool_calls":
                oai_messages.append(choice.message)
                for tc in choice.message.tool_calls or []:
                    args = json.loads(tc.function.arguments)
                    result = dispatcher(tc.function.name, args)
                    oai_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": str(result),
                        }
                    )
            else:
                break

        return ""


def _to_openai_tool(spec: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": spec["name"],
            "description": spec.get("description", ""),
            "parameters": spec["input_schema"],
        },
    }


class GPT(_OpenAICompatible):
    """OpenAI GPT provider.

    Reads ``OPENAI_API_KEY`` from the environment by default.

    Example::

        class MyActor(AIActor):
            provider = GPT()                           # gpt-4o
            provider = GPT("gpt-4o-mini",
                           temperature=0.0,
                           seed=42)
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        *,
        api_key: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        timeout: float | None = None,
        stop: list[str] | str | None = None,
        seed: int | None = None,
        response_format: dict | None = None,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key,
            api_key_env="OPENAI_API_KEY",
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            timeout=timeout,
            stop=stop,
            seed=seed,
            response_format=response_format,
        )


class Gemini(_OpenAICompatible):
    """Google Gemini provider via its OpenAI-compatible endpoint.

    Reads ``GOOGLE_API_KEY`` from the environment by default.

    Example::

        class MyActor(AIActor):
            provider = Gemini()                        # gemini-2.0-flash
            provider = Gemini("gemini-1.5-pro",
                              temperature=0.4)
    """

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        *,
        api_key: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        timeout: float | None = None,
        stop: list[str] | str | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key,
            api_key_env="GOOGLE_API_KEY",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            temperature=temperature,
            top_p=top_p,
            timeout=timeout,
            stop=stop,
            seed=seed,
        )


class Mistral(_OpenAICompatible):
    """Mistral AI provider via its OpenAI-compatible endpoint.

    Reads ``MISTRAL_API_KEY`` from the environment by default.

    Example::

        class MyActor(AIActor):
            provider = Mistral()                       # mistral-large-latest
            provider = Mistral("mistral-small-latest",
                               temperature=0.7)
    """

    def __init__(
        self,
        model: str = "mistral-large-latest",
        *,
        api_key: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        timeout: float | None = None,
        stop: list[str] | str | None = None,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key,
            api_key_env="MISTRAL_API_KEY",
            base_url="https://api.mistral.ai/v1",
            temperature=temperature,
            top_p=top_p,
            timeout=timeout,
            stop=stop,
        )


class DeepSeek(_OpenAICompatible):
    """DeepSeek provider via its OpenAI-compatible endpoint.

    Reads ``DEEPSEEK_API_KEY`` from the environment by default.

    Example::

        class MyActor(AIActor):
            provider = DeepSeek()                      # deepseek-chat
            provider = DeepSeek("deepseek-reasoner",
                                temperature=0.0)
    """

    def __init__(
        self,
        model: str = "deepseek-chat",
        *,
        api_key: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        timeout: float | None = None,
        stop: list[str] | str | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key,
            api_key_env="DEEPSEEK_API_KEY",
            base_url="https://api.deepseek.com/v1",
            temperature=temperature,
            top_p=top_p,
            timeout=timeout,
            stop=stop,
            seed=seed,
        )


_VSCODE_KEYRING_SERVICES = (
    "vscode.github-authentication",
    "GitHub.github.com",
)


def _token_from_vscode_keyring() -> str | None:
    """Try to read a GitHub token from VS Code's OS keyring entry.

    VS Code stores GitHub auth sessions via SecretStorage (libsecret on Linux,
    Keychain on macOS, Windows Credential Manager on Windows).  The ``keyring``
    package is an optional dependency; if it is not installed, or if no matching
    entry exists, this returns ``None`` silently.
    """
    try:
        import keyring  # optional dependency

        for service in _VSCODE_KEYRING_SERVICES:
            token = keyring.get_password(service, "github.com")
            if token:
                # Stored value may be JSON {"account":…,"password":"ghp_…"}
                try:
                    import json as _json

                    data = _json.loads(token)
                    if isinstance(data, dict):
                        token = data.get("password") or data.get("token") or token
                except (ValueError, TypeError):
                    pass
                if token:
                    return str(token)
    except ImportError:
        pass
    return None


def _resolve_github_token(explicit_key: str | None) -> str | None:
    """Return a GitHub token using the first available source.

    Resolution order:
    1. ``explicit_key`` argument (passed directly to the constructor)
    2. ``GITHUB_TOKEN`` environment variable
    3. ``gh auth token`` CLI output (works when ``gh`` is authenticated)
    4. OS keyring entry written by the VS Code GitHub Copilot extension
       (requires the optional ``keyring`` package)
    """
    if explicit_key:
        return explicit_key
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    try:
        result = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            token = result.stdout.strip()
            if token:
                return token
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return _token_from_vscode_keyring()


# Literal type for IDE autocomplete — keep in sync with Copilot.MODELS below.
CopilotModel = Literal[
    "gpt-4o",
    "gpt-4o-mini",
    "o1",
    "o1-mini",
    "o3-mini",
    "claude-sonnet-4-5",
    "gemini-2.0-flash",
]


class Copilot(_OpenAICompatible):
    """GitHub Copilot provider via its OpenAI-compatible endpoint.

    Reads ``GITHUB_TOKEN`` from the environment by default.  Requires a
    GitHub account with an active Copilot subscription (Individual, Business,
    or Enterprise).

    Passing an unsupported model string raises ``ValueError`` immediately at
    construction time so the mistake is caught before any network call is made.
    Use ``Copilot.MODELS`` to inspect the full set of valid model strings, or
    rely on the ``CopilotModel`` ``Literal`` type for IDE autocompletion.

    Supported models (as of mid-2025)
    ----------------------------------
    * ``gpt-4o`` (default)
    * ``gpt-4o-mini``
    * ``o1``, ``o1-mini``, ``o3-mini``
    * ``claude-sonnet-4-5``
    * ``gemini-2.0-flash``

    The ``Copilot-Integration-Id`` header is sent automatically to identify
    the integration to GitHub's backend.

    Configuration parameters
    ------------------------
    model : CopilotModel
        One of the supported Copilot model strings (see ``Copilot.MODELS``).
    api_key : str, optional
        GitHub token.  Resolution order: explicit argument →
        ``GITHUB_TOKEN`` environment variable → ``gh auth token`` CLI
        (works when the GitHub CLI is authenticated).
    temperature : float, optional
        Sampling temperature.
    top_p : float, optional
        Nucleus-sampling probability mass.
    timeout : float, optional
        HTTP request timeout in seconds.
    stop : list[str] | str, optional
        Sequences that stop generation.
    seed : int, optional
        Random seed for deterministic sampling (best-effort).

    Example::

        # IDE shows valid model choices via CopilotModel Literal
        class MyActor(AIActor):
            provider = Copilot()                       # gpt-4o (default)
            provider = Copilot("claude-sonnet-4-5")    # Claude via Copilot
            provider = Copilot("gemini-2.0-flash",
                               temperature=0.2)

        # Inspect valid models at runtime
        print(Copilot.MODELS)
    """

    MODELS: frozenset[str] = frozenset(
        {
            "gpt-4o",
            "gpt-4o-mini",
            "o1",
            "o1-mini",
            "o3-mini",
            "claude-sonnet-4-5",
            "gemini-2.0-flash",
        }
    )

    _BASE_URL = "https://api.githubcopilot.com"
    _INTEGRATION_HEADER = {"Copilot-Integration-Id": "vscode-chat"}  # noqa: RUF012

    def __init__(
        self,
        model: CopilotModel = "gpt-4o",
        *,
        api_key: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        timeout: float | None = None,
        stop: list[str] | str | None = None,
        seed: int | None = None,
    ) -> None:
        if model not in self.MODELS:
            valid = ", ".join(sorted(self.MODELS))
            raise ValueError(f"Unsupported Copilot model {model!r}. Valid models: {valid}")
        resolved = _resolve_github_token(api_key)
        if resolved is None:
            raise ValueError(
                "No GitHub token found for Copilot. Supply one via:\n"
                "  - api_key='ghp_...' constructor argument\n"
                "  - GITHUB_TOKEN environment variable\n"
                "  - gh auth login  (GitHub CLI)\n"
                "  - VS Code GitHub Copilot extension + pip install keyring"
            )
        super().__init__(
            model=model,
            api_key=resolved,
            api_key_env="GITHUB_TOKEN",
            base_url=self._BASE_URL,
            temperature=temperature,
            top_p=top_p,
            timeout=timeout,
            stop=stop,
            seed=seed,
            default_headers=self._INTEGRATION_HEADER,
        )
