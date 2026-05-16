# Future imports (must occur at the beginning of the file):
from __future__ import annotations

# Standard library imports:
import json
from collections.abc import Callable

# Third party imports:
import litellm

from ..accounting import MonitoringContext, UsageSummary
from .base import LLMProvider


class LiteLLM(LLMProvider):
    """Provider backed by LiteLLM, enabling unified monitoring across all models.

    LiteLLM supports 100+ models using a single ``model`` string such as
    ``"openai/gpt-4o"``, ``"anthropic/claude-sonnet-4-6"``,
    ``"gemini/gemini-2.0-flash"``, etc.

    When ``AIActor.monitoring = True``, actor metadata (actor name, session ID)
    is forwarded to LiteLLM via the ``metadata`` keyword, making it visible in
    any callbacks registered via ``success_callbacks`` / ``failure_callbacks``.

    Configuration parameters
    ------------------------
    model : str
        LiteLLM model string (e.g. ``"openai/gpt-4o"``).
    api_key : str, optional
        Explicit API key; falls back to the appropriate env variable.
    temperature : float, optional
        Sampling temperature.
    top_p : float, optional
        Nucleus-sampling probability mass.
    timeout : float, optional
        HTTP request timeout in seconds.
    max_retries : int, optional
        Number of automatic retries on transient errors.
    success_callbacks : list[str | callable], optional
        LiteLLM success callback names / callables (e.g. ``["langfuse"]``).
    failure_callbacks : list[str | callable], optional
        LiteLLM failure callback names / callables.

    Example::

        import litellm

        class MyActor(AIActor):
            monitoring = True
            provider = LiteLLM(
                "openai/gpt-4o",
                success_callbacks=["langfuse"],
            )
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        success_callbacks: list | None = None,
        failure_callbacks: list | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._temperature = temperature
        self._top_p = top_p
        self._timeout = timeout
        self._max_retries = max_retries

        if success_callbacks is not None:
            litellm.success_callback = list(success_callbacks)
        if failure_callbacks is not None:
            litellm.failure_callback = list(failure_callbacks)

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
        lm_messages: list = [{"role": "system", "content": system}, *messages]
        lm_tools = [_to_openai_tool(t) for t in tools] if tools else []

        metadata: dict = {}
        if monitoring_context is not None:
            metadata = {
                "actor_name": monitoring_context.actor_name,
                "session_id": monitoring_context.session_id,
                **monitoring_context.metadata,
            }

        while True:
            kwargs: dict = {
                "model": self.model,
                "messages": lm_messages,
                "max_tokens": max_tokens,
            }
            if lm_tools:
                kwargs["tools"] = lm_tools
            if self._api_key is not None:
                kwargs["api_key"] = self._api_key
            if self._temperature is not None:
                kwargs["temperature"] = self._temperature
            if self._top_p is not None:
                kwargs["top_p"] = self._top_p
            if self._timeout is not None:
                kwargs["timeout"] = self._timeout
            if self._max_retries is not None:
                kwargs["num_retries"] = self._max_retries
            if metadata:
                kwargs["metadata"] = metadata

            response = litellm.completion(**kwargs)

            if on_usage is not None:
                usage = getattr(response, "usage", None)
                if usage is not None:
                    on_usage(
                        UsageSummary(
                            input_tokens=getattr(usage, "prompt_tokens", None) or 0,
                            output_tokens=getattr(usage, "completion_tokens", None) or 0,
                        )
                    )

            choices = getattr(response, "choices", None)
            if choices is None:
                break
            choice = choices[0]

            if choice.finish_reason == "stop":
                return choice.message.content or ""

            if choice.finish_reason == "tool_calls":
                lm_messages.append(choice.message)
                for tc in choice.message.tool_calls or []:
                    args = json.loads(tc.function.arguments)
                    result = dispatcher(tc.function.name, args)
                    lm_messages.append(
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
