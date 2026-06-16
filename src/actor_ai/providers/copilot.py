# Future imports (must occur at the beginning of the file):
from __future__ import annotations

# Standard library imports:
import asyncio
import inspect
from collections.abc import Callable
from typing import Literal

# Third party imports:
from cachetools import TTLCache, cached
from cachetools.keys import hashkey
from copilot import CopilotClient, ExternalServerConfig, SubprocessConfig
from copilot.generated.session_events import AssistantMessageData, AssistantUsageData, SessionModelChangeData, SessionResumeData, SessionStartData
from copilot.session import PermissionHandler
from copilot.tools import Tool, ToolInvocation, ToolResult
from openai import OpenAI

# Local imports:
import logging

from actor_ai.accounting import MonitoringContext, UsageSummary
from actor_ai.providers.openai import _OpenAICompatible, _resolve_github_token

_log = logging.getLogger(__name__)

# Literal type for IDE autocomplete — kept in sync with Copilot.MODELS below.
CopilotModel = Literal[
    "auto",
    "claude-haiku-4.5",
    "claude-opus-4.7",
    "claude-opus-4.8",
    "claude-sonnet-4.5",
    "claude-sonnet-4.6",
    "gemini-3.1-pro-preview",
    "gemini-3.5-flash",
    "gpt-5-mini",
    "gpt-5.3-codex",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.5",
]


_MODELS_CACHE: TTLCache = TTLCache(maxsize=2, ttl=3600 * 6)
_SDK_MODELS_CACHE: TTLCache = TTLCache(maxsize=1, ttl=3600 * 6)


class Copilot(_OpenAICompatible):
    """GitHub Copilot provider.

    Uses GitHub Copilot's OpenAI-compatible endpoint by default, or the native
    Copilot SDK when ``use_sdk=True``.

    Passing an unsupported model string raises ``ValueError`` immediately at
    construction time so the mistake is caught before any network call is made.
    Use ``Copilot.MODELS`` to inspect the full set of valid model strings, or
    rely on the ``CopilotModel`` ``Literal`` type for IDE autocompletion.

    Supported models (as of mid-2025)
    ----------------------------------
    * ``gpt-4o`` (default)
    * ``gpt-4o-mini``
    * ``gpt-5``
    * ``o1``
    * ``o1-mini``
    * ``o3-mini``
    * ``claude-sonnet-4-5``
    * ``claude-sonnet-4-6``
    * ``claude-sonnet-4-7``
    * ``gemini-2.0-flash``

    The OpenAI-compatible path reads ``GITHUB_TOKEN`` by default and sends the
    ``Copilot-Integration-Id`` header automatically. The SDK path uses Copilot
    SDK/CLI authentication, or the explicit ``api_key`` when provided.

    Configuration parameters
    ------------------------
    model : CopilotModel
        One of the supported Copilot model strings (see ``Copilot.MODELS``).
    use_sdk : bool, optional
        When ``True``, call Copilot through the native async SDK session API.
    api_key : str, optional
        GitHub token.  Resolution order: explicit argument →
        ``GITHUB_TOKEN`` environment variable → ``gh auth token`` CLI
        (works when the GitHub CLI is authenticated).
    cli_url : str, optional
        Existing Copilot CLI server URL for SDK mode.
    cli_path : str, optional
        Copilot CLI executable path for SDK mode.
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
            provider = Copilot(use_sdk=True)           # native Copilot SDK
            provider = Copilot("gemini-2.0-flash",
                               temperature=0.2)

        # Inspect valid models at runtime
        print(Copilot.MODELS)
    """

    MODELS: frozenset[str] = frozenset(
        {arg for arg in getattr(CopilotModel, "__args__", []) if isinstance(arg, str)}
    )

    _API_KEY_ENV = "GITHUB_TOKEN"
    _BASE_URL = "https://api.githubcopilot.com"
    _INTEGRATION_HEADER = {"Copilot-Integration-Id": "vscode-chat"}  # noqa: RUF012

    def __init__(
        self,
        model: CopilotModel = "auto",
        *,
        use_sdk: bool = False,
        api_key: str | None = None,
        cli_url: str | None = None,
        cli_path: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        timeout: float | None = None,
        stop: list[str] | str | None = None,
        seed: int | None = None,
    ) -> None:
        if model not in self.MODELS:
            valid = ", ".join(sorted(self.MODELS))
            raise ValueError(f"Unsupported Copilot model {model!r}. Valid models: {valid}")
        self.use_sdk = use_sdk
        self._cli_url = cli_url
        self._cli_path = cli_path

        if use_sdk:
            self.model = model
            self.temperature = temperature
            self.top_p = top_p
            self.stop = stop
            self.seed = seed
            self.timeout = timeout or 60.0
            self._api_key = _resolve_github_token(api_key)
            self.resolved_model: str | None = None
            return

        resolved = _resolve_github_token(api_key)
        if resolved is None:
            raise ValueError(
                "No GitHub token found for Copilot. Supply one via:\n"
                "  - api_key='ghp_...' constructor argument\n"
                "  - GITHUB_TOKEN environment variable\n"
                "  - gh auth login  (GitHub CLI)\n"
                "  - VS Code GitHub Copilot extension (token read automatically from OS keyring)"
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

    async def _run_sdk(
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
        del max_tokens, monitoring_context

        prompt = _to_copilot_prompt(messages)
        sdk_tools = [_to_copilot_tool(tool, dispatcher) for tool in tools] if tools else []
        config = self._client_config()
        input_tokens = 0
        output_tokens = 0
        reasoning_tokens = 0
        cache_read_tokens = 0
        cache_write_tokens = 0
        usage_seen = False
        captured_model: str | None = None

        def capture_usage(event) -> None:
            nonlocal cache_read_tokens, cache_write_tokens, input_tokens, output_tokens
            nonlocal reasoning_tokens, usage_seen, captured_model
            match event.data:
                case SessionStartData() as data:
                    # selected_model is the actual model Copilot chose for "auto" at session start
                    if captured_model is None and data.selected_model and data.selected_model != "auto":
                        captured_model = data.selected_model
                        _log.info("Copilot session.start → selected_model=%r", captured_model)
                case SessionResumeData() as data:
                    if captured_model is None and data.selected_model and data.selected_model != "auto":
                        captured_model = data.selected_model
                        _log.info("Copilot session.resume → selected_model=%r", captured_model)
                case SessionModelChangeData() as data:
                    # Fires when Copilot resolves "auto" to a specific model
                    if captured_model is None:
                        captured_model = data.new_model
                        _log.info("Copilot session.model_change → %r", captured_model)
                case AssistantUsageData() as data:
                    usage_seen = True
                    input_tokens += _usage_token_count(data.input_tokens)
                    output_tokens += _usage_token_count(data.output_tokens)
                    reasoning_tokens += _usage_token_count(data.reasoning_tokens)
                    cache_read_tokens += _usage_token_count(data.cache_read_tokens)
                    cache_write_tokens += _usage_token_count(data.cache_write_tokens)
                    if captured_model is None and data.model:
                        captured_model = data.model
                        _log.info("Copilot assistant.usage → model=%r", captured_model)

        async with CopilotClient(config) as client:
            async with await client.create_session(
                on_permission_request=PermissionHandler.approve_all,
                model=self.model,
                tools=sdk_tools,
                system_message={"mode": "replace", "content": system},
                on_event=capture_usage,
            ) as session:
                reply = await session.send_and_wait(prompt, timeout=self.timeout)

        _log.info("Copilot captured_model=%r (self.model=%r)", captured_model, self.model)
        if self.model == "auto" and captured_model:
            self.resolved_model = captured_model
            _log.info("Copilot resolved_model set to %r", self.resolved_model)

        if usage_seen and on_usage is not None:
            on_usage(
                UsageSummary(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    reasoning_tokens=reasoning_tokens,
                    cache_read_tokens=cache_read_tokens,
                    cache_write_tokens=cache_write_tokens,
                )
            )

        if reply:
            match reply.data:
                case AssistantMessageData() as data:
                    return data.content
        return ""

    def _client_config(self) -> ExternalServerConfig | SubprocessConfig | None:
        if self._cli_url is not None:
            return ExternalServerConfig(url=self._cli_url)
        if self._api_key is not None or self._cli_path is not None:
            return SubprocessConfig(cli_path=self._cli_path, github_token=self._api_key)
        return None

    async def _probe_auto_model(self) -> str | None:
        """Send a minimal probe message to let Copilot select the model for 'auto'."""
        config = self._client_config()
        captured: str | None = None

        def capture(event) -> None:
            nonlocal captured
            match event.data:
                case SessionStartData() as data:
                    if captured is None and data.selected_model and data.selected_model != "auto":
                        captured = data.selected_model
                case SessionResumeData() as data:
                    if captured is None and data.selected_model and data.selected_model != "auto":
                        captured = data.selected_model
                case SessionModelChangeData() as data:
                    if captured is None and data.new_model:
                        captured = data.new_model
                case AssistantUsageData() as data:
                    # AssistantUsageData is the most reliable source: Copilot only populates
                    # selected_model on SessionStartData after a message exchange has started.
                    if captured is None and data.model:
                        captured = data.model

        try:
            async with CopilotClient(config) as client:
                async with await client.create_session(
                    on_permission_request=PermissionHandler.approve_all,
                    model=self.model,
                    on_event=capture,
                ) as session:
                    # A minimal send is required — Copilot only resolves "auto" once it
                    # processes a real request and emits AssistantUsageData.
                    await session.send_and_wait(".", timeout=self.timeout)
        except Exception as exc:
            _log.debug("Copilot probe_resolved_model failed: %s", exc)

        if captured:
            self.resolved_model = captured
            _log.info("Copilot probe resolved 'auto' → %r", captured)
        return captured

    def probe_resolved_model(self) -> str | None:
        """Synchronously probe the resolved model for 'auto' (used at actor startup)."""
        if not self.use_sdk or self.model != "auto":
            return None
        return asyncio.run(self._probe_auto_model())

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
        if not self.use_sdk:
            return super().run(
                system=system,
                messages=messages,
                tools=tools,
                dispatcher=dispatcher,
                max_tokens=max_tokens,
                on_usage=on_usage,
                monitoring_context=monitoring_context,
            )

        return asyncio.run(
            self._run_sdk(
                system=system,
                messages=messages,
                tools=tools,
                dispatcher=dispatcher,
                max_tokens=max_tokens,
                on_usage=on_usage,
                monitoring_context=monitoring_context,
            )
        )

    @classmethod
    def available_models(cls, refresh: bool = False, *, use_sdk: bool = False) -> list[str]:  # pyright: ignore[override]
        """Return a sorted list of model IDs available through the Copilot endpoint.

        Results are cached for 6 hours; pass ``refresh=True`` to force a live call.

        Parameters
        ----------
        refresh:
            Clear the cache entry and force a new API call.
        use_sdk:
            When ``True``, query via the native Copilot SDK
            (``CopilotClient.list_models()``).  When ``False`` (default),
            query the OpenAI-compatible ``/v1/models`` REST endpoint.
        """
        if use_sdk:
            if refresh:
                _SDK_MODELS_CACHE.pop(hashkey(cls), None)
            return cls._fetch_models_sdk()
        if refresh:
            _MODELS_CACHE.pop(hashkey(cls), None)
        return cls._fetch_models()

    @classmethod
    @cached(_MODELS_CACHE)
    def _fetch_models(cls) -> list[str]:  # pyright: ignore[override]
        client = OpenAI(
            api_key=_resolve_github_token(None),
            base_url=cls._BASE_URL,
            default_headers=cls._INTEGRATION_HEADER,
        )
        return sorted(m.id for m in client.models.list())

    @classmethod
    @cached(_SDK_MODELS_CACHE)
    def _fetch_models_sdk(cls) -> list[str]:
        async def _fetch() -> list[str]:
            client = CopilotClient()
            try:
                await client.start()
                models = await client.list_models()
            finally:
                await client.stop()
            return sorted(m.id for m in models)

        return asyncio.run(_fetch())


class Copilot_SDK(Copilot):
    """Compatibility alias for ``Copilot(use_sdk=True)``."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["use_sdk"] = True
        super().__init__(*args, **kwargs)


def _to_copilot_prompt(messages: list[dict]) -> str:
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _usage_token_count(value: float | None) -> int:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    return 0


def _to_copilot_tool(
    spec: dict,
    dispatcher: Callable[[str, dict], object],
) -> Tool:
    async def handler(invocation: ToolInvocation) -> ToolResult:
        arguments = invocation.arguments or {}
        loop = asyncio.get_running_loop()
        # Run dispatcher in a thread so blocking tool implementations (e.g. nested
        # instruct calls that call asyncio.run()) don't block the SDK event loop.
        result = await loop.run_in_executor(None, dispatcher, invocation.tool_name, arguments)
        if inspect.isawaitable(result):
            result = await result
        return ToolResult(text_result_for_llm=str(result), result_type="success")

    return Tool(
        name=spec["name"],
        description=spec.get("description", ""),
        parameters=spec.get("input_schema"),
        handler=handler,
        skip_permission=True,
    )
