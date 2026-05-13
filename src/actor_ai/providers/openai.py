# Future imports (must occur at the beginning of the file):
from __future__ import annotations

# Standard library imports:
import json
import os
from collections.abc import Callable

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
