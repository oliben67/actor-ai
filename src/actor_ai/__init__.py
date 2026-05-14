# Future imports (must occur at the beginning of the file):
from __future__ import annotations

from .accounting import Ledger, LedgerEntry, ModelRate, MonitoringContext, Rates, UsageSummary
from .actor import AIActor
from .chorus import Chorus
from .messages import Forget, Instruct, Remember
from .providers import (
    GPT,
    Claude,
    Copilot,
    CopilotModel,
    DeepSeek,
    Gemini,
    LiteLLM,
    LLMProvider,
    Mistral,
)
from .tools import tool

__all__ = [
    "AIActor",
    "Chorus",
    "Claude",
    "Copilot",
    "CopilotModel",
    "DeepSeek",
    "Forget",
    "GPT",
    "Gemini",
    "Instruct",
    "Ledger",
    "LedgerEntry",
    "LiteLLM",
    "LLMProvider",
    "Mistral",
    "ModelRate",
    "MonitoringContext",
    "Rates",
    "Remember",
    "UsageSummary",
    "tool",
]


def main() -> None:
    print("actor-ai: multi-provider AI agents built on pykka.")
    print("Providers: Claude, Copilot, GPT, Gemini, Mistral, DeepSeek")
