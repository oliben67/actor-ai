# Future imports (must occur at the beginning of the file):
from __future__ import annotations

# Standard library imports:
import json
import os
import subprocess
from collections.abc import Callable

# Third party imports:
from cachetools import TTLCache, cached
from cachetools.keys import hashkey
from openai import OpenAI

from ..accounting import MonitoringContext, UsageSummary
from .base import LLMProvider

_MODELS_CACHE: TTLCache = TTLCache(maxsize=8, ttl=3600 * 6)


class _OpenAICompatible(LLMProvider):
    _API_KEY_ENV: str = "OPENAI_API_KEY"
    _BASE_URL: str | None = None
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

    @classmethod
    def available_models(cls, refresh: bool = False) -> list[str]:
        if refresh:
            _MODELS_CACHE.pop(hashkey(cls), None)
        return cls._fetch_models()

    @classmethod
    @cached(_MODELS_CACHE)
    def _fetch_models(cls) -> list[str]:
        client = OpenAI(
            api_key=os.environ.get(cls._API_KEY_ENV),
            base_url=cls._BASE_URL,
        )
        return sorted(m.id for m in client.models.list())

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
                        reasoning_tokens=_usage_detail_token_count(
                            response.usage,
                            "completion_tokens_details",
                            "reasoning_tokens",
                        ),
                        cache_read_tokens=_usage_detail_token_count(
                            response.usage,
                            "prompt_tokens_details",
                            "cached_tokens",
                        ),
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


def _usage_detail_token_count(usage: object, detail_name: str, token_name: str) -> int:
    details = getattr(usage, detail_name, None)
    if details is None:
        return 0
    if isinstance(details, dict):
        value = details.get(token_name)
    else:
        value = getattr(details, token_name, None)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    return 0


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

    _API_KEY_ENV = "GOOGLE_API_KEY"
    _BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

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

    _API_KEY_ENV = "MISTRAL_API_KEY"
    _BASE_URL = "https://api.mistral.ai/v1"

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

    _API_KEY_ENV = "DEEPSEEK_API_KEY"
    _BASE_URL = "https://api.deepseek.com/v1"

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


# Service names searched in the OS keyring. Tried in order; first match wins.
# - gh:github.com        → GitHub CLI (gh auth login)
# - vscode.github-*      → VS Code built-in GitHub auth (older versions / macOS / Windows)
# - GitHub.github.com    → VS Code GitHub auth (alternate name seen in some versions)
_KEYRING_SERVICES = (
    "gh:github.com",
    "vscode.github-authentication",
    "GitHub.github.com",
)


def _decode_keyring_secret(raw: str) -> str | None:
    """Return a usable token from a plain or JSON-encoded keyring value."""
    if not raw or not raw.isprintable():
        return None
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            raw = str(data.get("password") or data.get("token") or "")
    except ValueError, TypeError:
        pass
    return raw.strip() or None


def _token_from_keyring() -> str | None:
    """Search the OS keyring for a usable GitHub token.

    On Linux uses ``secretstorage`` (a ``keyring`` dependency) to search by
    service attribute — so the GitHub username does not need to be known in
    advance.  On macOS and Windows falls back to ``keyring.get_password`` with
    the known account name ``"github.com"``.  Returns ``None`` silently if no
    matching entry is found or if the backend is unavailable.
    """
    # Linux: secretstorage allows attribute-based search without knowing the account name.
    # Resolve the D-Bus exception class first so it is always bound before the try block.
    _SSError: type[Exception]
    try:
        # Third party imports:
        from secretstorage.exceptions import SecretServiceNotAvailableException as _SSError
    except ImportError:
        _SSError = OSError  # placeholder; secretstorage not installed, won't be raised

    try:
        # Third party imports:
        import secretstorage

        bus = secretstorage.dbus_init()
        collection = secretstorage.get_default_collection(bus)
        for service in _KEYRING_SERVICES:
            for item in collection.search_items({"service": service}):
                secret_bytes = item.get_secret()
                if secret_bytes:
                    token = _decode_keyring_secret(secret_bytes.decode("utf-8", errors="ignore"))
                    if token:
                        return token
    except ImportError, _SSError:  # type: ignore[misc]
        pass

    # macOS / Windows: keyring.get_password requires an account name.
    # Third party imports:
    import keyring

    for service in _KEYRING_SERVICES:
        secret_str = keyring.get_password(service, "github.com")
        if secret_str:
            token = _decode_keyring_secret(secret_str)
            if token:
                return token
    return None


def _resolve_github_token(explicit_key: str | None) -> str | None:
    """Return a GitHub token using the first available source.

    Resolution order:
    1. ``explicit_key`` argument (passed directly to the constructor)
    2. ``GITHUB_TOKEN`` environment variable
    3. ``gh auth token`` CLI (works when the GitHub CLI is authenticated)
    4. OS keyring — searches entries written by the GitHub CLI and the VS Code
       GitHub Copilot extension (via the bundled ``keyring`` package)
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
    except FileNotFoundError, subprocess.TimeoutExpired, OSError:
        pass
    return _token_from_keyring()
