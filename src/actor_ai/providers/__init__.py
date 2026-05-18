# Future imports (must occur at the beginning of the file):
from __future__ import annotations

from .anthropic import Claude
from .base import LLMProvider
from .copilot import Copilot, Copilot_SDK, CopilotModel
from .openai import GPT, DeepSeek, Gemini, Mistral

__all__ = [
    "GPT",
    "Claude",
    "Copilot",
    "CopilotModel",
    "Copilot_SDK",
    "DeepSeek",
    "Gemini",
    "LLMProvider",
    "LiteLLM",
    "Mistral",
]


def __getattr__(name: str):
    if name == "LiteLLM":
        from .litellm import LiteLLM

        return LiteLLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
