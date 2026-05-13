"""08 – Monitoring with LiteLLM

When AIActor.monitoring = True, each instruct() call creates a
MonitoringContext (actor name + session ID + optional metadata) and passes it
to the provider.

The LiteLLM provider forwards that context as the ``metadata`` kwarg to
litellm.completion(), making it visible to any registered callbacks
(Langfuse, Helicone, custom callbacks, etc.).

This example mocks litellm.completion to show exactly what metadata is
forwarded without needing a real API key.

Topics covered:
  - AIActor.monitoring = True / False
  - MonitoringContext fields
  - LiteLLM provider: model string, temperature, max_retries, callbacks
  - How metadata flows from actor → provider → litellm.completion
  - Registering a custom success callback
"""

# Future imports (must occur at the beginning of the file):
from __future__ import annotations

# Standard library imports:
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent))

# Third party imports:
import litellm

# Local imports:
from actor_ai import AIActor, LiteLLM
from actor_ai.accounting import MonitoringContext

# ── Build a minimal litellm-compatible mock response ──────────────────────


def _fake_litellm_response(content: str = "Done.") -> MagicMock:
    usage = MagicMock(prompt_tokens=60, completion_tokens=25)
    message = MagicMock(content=content, tool_calls=None)
    choice = MagicMock(finish_reason="stop", message=message)
    resp = MagicMock(choices=[choice], usage=usage)
    return resp


def _divider(title: str) -> None:
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print(f"{'─' * 55}")


def main() -> None:

    # ── 1. monitoring = False (default) ───────────────────────────────
    _divider("monitoring = False — no metadata forwarded")

    with patch("litellm.completion"):
        prov = LiteLLM("openai/gpt-4o", temperature=0.3)

    class AgentOff(AIActor):
        system_prompt = "You are a helpful agent."
        provider = prov
        monitoring = False

    ref = AgentOff.start()
    try:
        with patch("litellm.completion", return_value=_fake_litellm_response()) as mock_c:
            ref.proxy().instruct("Hello!").get()

        kw = mock_c.call_args.kwargs
        print(f"  'metadata' in call kwargs: {'metadata' in kw}")
        print(f"  model forwarded          : {kw['model']}")
    finally:
        ref.stop()

    # ── 2. monitoring = True ──────────────────────────────────────────
    _divider("monitoring = True — metadata forwarded to litellm")

    with patch("litellm.completion"):
        prov2 = LiteLLM("openai/gpt-4o")

    class AgentOn(AIActor):
        system_prompt = "You are a helpful agent."
        provider = prov2
        actor_name = "my-assistant"
        monitoring = True

    ref2 = AgentOn.start()
    try:
        session_id = ref2.proxy().get_session_id().get()

        with patch("litellm.completion", return_value=_fake_litellm_response("42")) as mock_c:
            reply = ref2.proxy().instruct("What is 6 × 7?").get()

        kw = mock_c.call_args.kwargs
        metadata = kw.get("metadata", {})
        print(f"  reply                    : {reply}")
        print(f"  metadata.actor_name      : {metadata.get('actor_name')}")
        print(f"  metadata.session_id      : {metadata.get('session_id')}")
        print(f"  session_id matches actor : {metadata.get('session_id') == session_id}")
    finally:
        ref2.stop()

    # ── 3. Custom metadata on MonitoringContext ───────────────────────
    _divider("MonitoringContext with extra metadata")

    ctx = MonitoringContext(
        actor_name="billing-agent",
        session_id="sess-abc123",
        metadata={"environment": "production", "user_id": "u42", "tier": "premium"},
    )
    print(f"  actor_name  : {ctx.actor_name}")
    print(f"  session_id  : {ctx.session_id}")
    print(f"  metadata    : {ctx.metadata}")

    # Simulate what LiteLLM would forward
    forwarded = {
        "actor_name": ctx.actor_name,
        "session_id": ctx.session_id,
        **ctx.metadata,
    }
    print("\n  Forwarded to litellm.completion(metadata=…):")
    for k, v in forwarded.items():
        print(f"    {k}: {v!r}")

    # ── 4. Registering success / failure callbacks ─────────────────────
    _divider("Registering LiteLLM callbacks")

    captured_events: list[dict] = []

    def my_success_callback(kwargs, response, start_time, end_time):
        captured_events.append(
            {
                "type": "success",
                "model": kwargs.get("model"),
                "actor_name": (kwargs.get("metadata") or {}).get("actor_name"),
                "latency_ms": round((end_time - start_time).total_seconds() * 1000),
            }
        )

    with patch("litellm.completion"):
        LiteLLM(
            "openai/gpt-4o",
            success_callbacks=[my_success_callback],
            failure_callbacks=["sentry"],  # named integrations also supported
        )

    print(f"  success_callback list : {litellm.success_callback}")
    print(f"  failure_callback list : {litellm.failure_callback}")

    # ── 5. LiteLLM provider configuration summary ─────────────────────
    _divider("LiteLLM provider attributes")

    with patch("litellm.completion"):
        prov4 = LiteLLM(
            "anthropic/claude-sonnet-4-6",
            api_key="sk-test",
            temperature=0.1,
            top_p=0.95,
            timeout=30.0,
            max_retries=2,
        )

    print(f"  model      : {prov4.model}")
    print(f"  temperature: {prov4._temperature}")
    print(f"  top_p      : {prov4._top_p}")
    print(f"  timeout    : {prov4._timeout}")
    print(f"  max_retries: {prov4._max_retries}")


if __name__ == "__main__":
    main()
