"""05 – Provider Configuration

actor_ai ships six provider backends. This example shows how to configure
each one, what parameters are available, and how to swap the provider on an
actor class or per instance.

No LLM calls are made here — provider objects are instantiated with a dummy
API key so we can inspect their configuration attributes.

Providers
---------
  Claude          Anthropic API         ANTHROPIC_API_KEY
  GPT             OpenAI API            OPENAI_API_KEY
  Gemini          Google AI (OpenAI-compat)  GOOGLE_API_KEY
  Mistral         Mistral AI            MISTRAL_API_KEY
  DeepSeek        DeepSeek API          DEEPSEEK_API_KEY
  LiteLLM         Any model via LiteLLM  (depends on model)
"""

# Future imports (must occur at the beginning of the file):
from __future__ import annotations

# Standard library imports:
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))

# Third party imports:
from fake_provider import ScriptedProvider

# Local imports:
from actor_ai import GPT, AIActor, Claude, DeepSeek, Gemini, LiteLLM, Mistral

# ── Helpers ────────────────────────────────────────────────────────────────


def _show(provider_name: str, provider) -> None:
    print(f"\n  {provider_name}")
    print(f"    model       : {provider.model}")
    for attr in (
        "temperature",
        "top_p",
        "top_k",
        "seed",
        "frequency_penalty",
        "presence_penalty",
        "stop_sequences",
        "stop",
        "max_retries",
    ):
        val = getattr(provider, f"_{attr}", getattr(provider, attr, None))
        if val is not None:
            print(f"    {attr:<20}: {val}")


def _divider(title: str) -> None:
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print(f"{'─' * 55}")


def main() -> None:

    # ── Claude ────────────────────────────────────────────────────────
    _divider("Claude (Anthropic)")

    with patch("actor_ai.providers.anthropic.Anthropic"):
        c_default = Claude()
        c_opus = Claude("claude-opus-4-7", temperature=0.2, top_k=40, timeout=30.0)
        c_haiku = Claude(
            "claude-haiku-4-5-20251001",
            top_p=0.9,
            stop_sequences=["STOP", "END"],
        )

    _show("Claude() — defaults", c_default)
    _show("Claude('claude-opus-4-7', temperature=0.2, top_k=40)", c_opus)
    _show("Claude('claude-haiku-4-5-20251001', top_p=0.9, stop_sequences=[…])", c_haiku)

    # ── GPT ───────────────────────────────────────────────────────────
    _divider("GPT (OpenAI)")

    with patch("actor_ai.providers.openai.OpenAI"):
        g_default = GPT()
        g_mini = GPT("gpt-4o-mini", temperature=0.0, seed=42, frequency_penalty=0.3)
        g_json = GPT(
            "gpt-4o",
            response_format={"type": "json_object"},
            presence_penalty=0.5,
        )

    _show("GPT() — defaults", g_default)
    _show("GPT('gpt-4o-mini', temperature=0.0, seed=42)", g_mini)
    _show("GPT('gpt-4o', response_format={'type': 'json_object'})", g_json)
    print(f"    response_format : {g_json.response_format}")

    # ── Gemini ────────────────────────────────────────────────────────
    _divider("Gemini (Google AI via OpenAI-compatible endpoint)")

    with patch("actor_ai.providers.openai.OpenAI"):
        gem_default = Gemini()
        gem_pro = Gemini("gemini-1.5-pro", temperature=0.4, top_p=0.95)

    _show("Gemini() — defaults", gem_default)
    _show("Gemini('gemini-1.5-pro', temperature=0.4, top_p=0.95)", gem_pro)

    # ── Mistral ───────────────────────────────────────────────────────
    _divider("Mistral (Mistral AI)")

    with patch("actor_ai.providers.openai.OpenAI"):
        mist_default = Mistral()
        mist_small = Mistral("mistral-small-latest", temperature=0.7, top_p=0.85)

    _show("Mistral() — defaults", mist_default)
    _show("Mistral('mistral-small-latest', temperature=0.7)", mist_small)

    # ── DeepSeek ──────────────────────────────────────────────────────
    _divider("DeepSeek")

    with patch("actor_ai.providers.openai.OpenAI"):
        ds_default = DeepSeek()
        ds_reasoner = DeepSeek("deepseek-reasoner", temperature=0.0, seed=0)

    _show("DeepSeek() — defaults", ds_default)
    _show("DeepSeek('deepseek-reasoner', temperature=0.0, seed=0)", ds_reasoner)

    # ── LiteLLM ───────────────────────────────────────────────────────
    _divider("LiteLLM (unified proxy — 100+ models)")

    with patch("litellm.completion"):
        ll_gpt = LiteLLM("openai/gpt-4o", temperature=0.3)
        ll_claude = LiteLLM("anthropic/claude-sonnet-4-6", max_retries=3)
        ll_gemini = LiteLLM("gemini/gemini-2.0-flash", top_p=0.9, timeout=20.0)

    _show("LiteLLM('openai/gpt-4o', temperature=0.3)", ll_gpt)
    _show("LiteLLM('anthropic/claude-sonnet-4-6', max_retries=3)", ll_claude)
    _show("LiteLLM('gemini/gemini-2.0-flash', top_p=0.9)", ll_gemini)

    # ── Swapping providers on an actor ────────────────────────────────
    _divider("Swapping providers on an actor")

    class MyAgent(AIActor):
        system_prompt = "You are a helpful assistant."
        provider = ScriptedProvider(["Hello from provider A."])

    ref = MyAgent.start()
    try:
        r1 = ref.proxy().instruct("Hi!").get()
        print(f"  [provider A] {r1}")

        # Replace the provider on the running instance via proxy.
        new_provider = ScriptedProvider(["Hello from provider B."])
        ref.proxy().provider = new_provider  # pykka ProxySetAttr

        r2 = ref.proxy().instruct("Hi again!").get()
        print(f"  [provider B] {r2}")
    finally:
        ref.stop()


if __name__ == "__main__":
    main()
