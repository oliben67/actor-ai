# Future imports (must occur at the beginning of the file):
from __future__ import annotations

from .accounting import Ledger, LedgerEntry, ModelRate, MonitoringContext, Rates, UsageSummary
from .actor import AIActor, InstructionInput, make_agent
from .chorus import Chorus, ChorusType
from .context import SharedContext
from .messages import Forget, Instruct, Remember
from .providers import (
    GPT,
    Claude,
    DeepSeek,
    Gemini,
    LLMProvider,
    Mistral,
)
from .providers.copilot import Copilot, Copilot_SDK, CopilotModel
from .tools import tool
from .workflow import Workflow, WorkflowState, WorkflowTransition

__all__ = [
    "AIActor",
    "Chorus",
    "ChorusType",
    "Claude",
    "Copilot",
    "CopilotModel",
    "Copilot_SDK",
    "DeepSeek",
    "Forget",
    "GPT",
    "Gemini",
    "Instruct",
    "InstructionInput",
    "LLMProvider",
    "Ledger",
    "LedgerEntry",
    "LiteLLM",
    "Mistral",
    "ModelRate",
    "MonitoringContext",
    "Rates",
    "Remember",
    "SharedContext",
    "UsageSummary",
    "Workflow",
    "WorkflowState",
    "WorkflowTransition",
    "make_agent",
    "tool",
]


def main() -> None:
    print("actor-ai: multi-provider AI agents built on pykka.")
    print("Providers: Claude, Copilot, GPT, Gemini, Mistral, DeepSeek")


def __getattr__(name: str):
    if name == "LiteLLM":
        from .providers.litellm import LiteLLM

        return LiteLLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
