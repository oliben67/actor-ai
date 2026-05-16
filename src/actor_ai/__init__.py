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
    Copilot,
    CopilotModel,
    DeepSeek,
    Gemini,
    LiteLLM,
    LLMProvider,
    Mistral,
)
from .tools import tool
from .workflow import Workflow, WorkflowState, WorkflowTransition

__all__ = [
    "AIActor",
    "Chorus",
    "ChorusType",
    "Claude",
    "Copilot",
    "CopilotModel",
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
