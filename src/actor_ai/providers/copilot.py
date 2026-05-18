# Future imports (must occur at the beginning of the file):
from __future__ import annotations

# Standard library imports:
import asyncio
import inspect
from collections.abc import Callable
from typing import Literal

# Third party imports:
from copilot import CopilotClient, ExternalServerConfig, SubprocessConfig
from copilot.generated.session_events import AssistantMessageData, AssistantUsageData
from copilot.session import PermissionHandler
from copilot.tools import Tool, ToolInvocation, ToolResult

# Local imports:
from actor_ai.accounting import MonitoringContext, UsageSummary
from actor_ai.providers.openai import _OpenAICompatible, _resolve_github_token

# Literal type for IDE autocomplete — keep in sync with Copilot.MODELS below.
CopilotModel = Literal[
    "gpt-4o",
    "gpt-4o-mini",
    "o1",
    "o1-mini",
    "o3-mini",
    "gpt-5",
    "claude-sonnet-4.5",
    "claude-sonnet-4-5",
    "gemini-2.0-flash",
]


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
    * ``o1``, ``o1-mini``, ``o3-mini``
    * ``claude-sonnet-4.5``
    * ``claude-sonnet-4-5``
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
        {
            "gpt-4o",
            "gpt-4o-mini",
            "o1",
            "o1-mini",
            "o3-mini",
            "gpt-5",
            "claude-sonnet-4.5",
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

        def capture_usage(event) -> None:
            nonlocal cache_read_tokens, cache_write_tokens, input_tokens, output_tokens
            nonlocal reasoning_tokens, usage_seen
            match event.data:
                case AssistantUsageData() as data:
                    usage_seen = True
                    input_tokens += _usage_token_count(data.input_tokens)
                    output_tokens += _usage_token_count(data.output_tokens)
                    reasoning_tokens += _usage_token_count(data.reasoning_tokens)
                    cache_read_tokens += _usage_token_count(data.cache_read_tokens)
                    cache_write_tokens += _usage_token_count(data.cache_write_tokens)

        async with CopilotClient(config) as client:
            async with await client.create_session(
                on_permission_request=PermissionHandler.approve_all,
                model=self.model,
                tools=sdk_tools,
                system_message={"mode": "replace", "content": system},
            ) as session:
                unsubscribe = session.on(capture_usage)
                try:
                    reply = await session.send_and_wait(prompt, timeout=self.timeout)
                finally:
                    unsubscribe()

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
        result = dispatcher(invocation.tool_name, arguments)
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
