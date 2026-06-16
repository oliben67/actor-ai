# Future imports (must occur at the beginning of the file):
from __future__ import annotations

# Standard library imports:
import os
from collections.abc import Callable

# Third party imports:
from anthropic import Anthropic
from cachetools import TTLCache, cached
from cachetools.keys import hashkey

from ..accounting import MonitoringContext, UsageSummary
from .base import LLMProvider

_MODELS_CACHE: TTLCache = TTLCache(maxsize=1, ttl=3600 * 6)


class Claude(LLMProvider):
    """Anthropic Claude provider.

    Reads ``ANTHROPIC_API_KEY`` from the environment by default.

    Configuration parameters
    ------------------------
    temperature : float, optional
        Sampling temperature in ``[0, 1]``.  Higher values produce more varied
        output; lower values are more deterministic.
    top_p : float, optional
        Nucleus-sampling probability mass.  Mutually exclusive with
        ``top_k`` per Anthropic's API.
    top_k : int, optional
        Sample from the top-K tokens at each step.
    timeout : float, optional
        HTTP request timeout in seconds.
    stop_sequences : list[str], optional
        Additional sequences that will stop generation.

    Example::

        class MyActor(AIActor):
            provider = Claude()                          # defaults
            provider = Claude("claude-opus-4-7",
                              temperature=0.2,
                              timeout=30.0)
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        *,
        api_key: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        timeout: float | None = None,
        stop_sequences: list[str] | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.stop_sequences = stop_sequences
        self._client = Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
            timeout=timeout,
        )

    @classmethod
    def available_models(cls, refresh: bool = False) -> list[str]:
        if refresh:
            _MODELS_CACHE.pop(hashkey(cls), None)
        return cls._fetch_models()

    @classmethod
    @cached(_MODELS_CACHE)
    def _fetch_models(cls) -> list[str]:
        client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
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
        kwargs: dict = {
            "model": self.model,
            "system": system,
            "messages": list(messages),
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.top_k is not None:
            kwargs["top_k"] = self.top_k
        if self.stop_sequences:
            kwargs["stop_sequences"] = self.stop_sequences

        while True:
            response = self._client.messages.create(**kwargs)

            if on_usage is not None and hasattr(response, "usage") and response.usage:
                on_usage(
                    UsageSummary(
                        input_tokens=response.usage.input_tokens or 0,
                        output_tokens=response.usage.output_tokens or 0,
                        cache_read_tokens=getattr(
                            response.usage,
                            "cache_read_input_tokens",
                            None,
                        )
                        or 0,
                        cache_write_tokens=getattr(
                            response.usage,
                            "cache_creation_input_tokens",
                            None,
                        )
                        or 0,
                    )
                )

            if response.stop_reason == "end_turn":
                return next(
                    (block.text for block in response.content if hasattr(block, "text")),
                    "",
                )

            if response.stop_reason == "tool_use":
                kwargs["messages"].append({"role": "assistant", "content": response.content})
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = dispatcher(block.name, block.input)
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": str(result),
                            }
                        )
                kwargs["messages"].append({"role": "user", "content": tool_results})
            else:
                break

        return ""
