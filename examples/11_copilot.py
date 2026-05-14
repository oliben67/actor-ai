"""11 – GitHub Copilot Provider

The ``Copilot`` provider connects to GitHub Copilot's OpenAI-compatible
endpoint (https://api.githubcopilot.com) using a GitHub token.

It requires a GitHub account with an active Copilot subscription
(Individual, Business, or Enterprise).

Supported models (mid-2025)
----------------------------
    gpt-4o             (default)
    gpt-4o-mini
    o1, o3-mini
    claude-sonnet-4-5
    gemini-2.0-flash

The provider sends the ``Copilot-Integration-Id: vscode-chat`` header
automatically so GitHub routes the request correctly.

This example mocks the underlying OpenAI client to run without a real
GitHub token, while demonstrating the full API surface.

Topics covered:
  - Copilot() default and custom model strings
  - GITHUB_TOKEN environment variable lookup
  - Explicit api_key override
  - The integration header forwarded to GitHub's API
  - Tool calling via Copilot
  - Using Copilot with AIActor and Chorus

Real-Copilot swap
-----------------
    import os
    os.environ["GITHUB_TOKEN"] = "ghp_your_token_here"
    # then remove all `patch(...)` wrappers below
"""

# Future imports (must occur at the beginning of the file):
from __future__ import annotations

# Standard library imports:
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent))

# Third party imports:
from fake_provider import ScriptedProvider, ToolCallingProvider

# Local imports:
from actor_ai import AIActor, Chorus, Copilot, CopilotModel, tool

# ── Helpers ───────────────────────────────────────────────────────────────────


def _mock_openai_response(content: str = "Done.") -> MagicMock:
    usage = MagicMock(prompt_tokens=80, completion_tokens=30)
    message = MagicMock(content=content, tool_calls=None)
    choice = MagicMock(finish_reason="stop", message=message)
    return MagicMock(choices=[choice], usage=usage)


def _divider(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ── 1. Provider construction and defaults ─────────────────────────────────────


def _section_construction() -> None:
    _divider("1. Copilot provider construction")

    captured: dict = {}

    def _capture(**kw):
        captured.update(kw)
        return MagicMock()

    with patch("actor_ai.providers.openai.OpenAI", side_effect=_capture):
        prov = Copilot()

    print(f"  model               : {prov.model}")
    print(f"  base_url            : {captured['base_url']}")
    print(f"  integration header  : {captured['default_headers']}")

    with patch("actor_ai.providers.openai.OpenAI", side_effect=_capture):
        prov2 = Copilot("claude-sonnet-4-5", temperature=0.2, seed=42)

    print(f"\n  custom model        : {prov2.model}")
    print(f"  temperature stored  : {prov2.temperature}")
    print(f"  seed stored         : {prov2.seed}")


# ── 2. Token resolution ───────────────────────────────────────────────────────


def _section_token(monkeypatch_env: dict) -> None:
    _divider("2. Token resolution — GITHUB_TOKEN env var")

    # Standard library imports:
    import os

    os.environ["GITHUB_TOKEN"] = "ghp_demo_token_from_env"

    captured: dict = {}
    with patch("actor_ai.providers.openai.OpenAI") as mock_oai:
        mock_oai.side_effect = lambda **kw: captured.update(kw) or MagicMock()
        Copilot()

    print(f"  api_key resolved    : {captured.get('api_key')}")

    captured2: dict = {}
    with patch("actor_ai.providers.openai.OpenAI") as mock_oai:
        mock_oai.side_effect = lambda **kw: captured2.update(kw) or MagicMock()
        Copilot(api_key="ghp_explicit_override")

    print(f"  explicit api_key    : {captured2.get('api_key')}")

    del os.environ["GITHUB_TOKEN"]


# ── 3. Simple instruction with AIActor ────────────────────────────────────────


def _section_instruct() -> None:
    _divider("3. AIActor with Copilot provider — simple instruction")

    with patch("actor_ai.providers.openai.OpenAI"):
        prov = Copilot()

    class Assistant(AIActor):
        system_prompt = "You are a helpful coding assistant powered by GitHub Copilot."
        provider = prov

    ref = Assistant.start()
    try:
        with patch.object(prov, "_client") as mock_client:
            mock_client.chat.completions.create.return_value = _mock_openai_response(
                "Use `git log --oneline -10` to see the last 10 commits."
            )
            reply = ref.proxy().instruct("How do I view recent git commits?").get()

        print("  [user]      How do I view recent git commits?")
        print(f"  [assistant] {reply}")
    finally:
        ref.stop()


# ── 4. Valid models — runtime enumeration and IDE type ────────────────────────


def _section_models() -> None:
    _divider("4. Valid Copilot models — Copilot.MODELS + CopilotModel Literal")

    # Copilot.MODELS: frozenset[str] — for runtime checks and iteration
    print("  Copilot.MODELS (runtime frozenset):")
    for m in sorted(Copilot.MODELS):
        marker = " ← default" if m == "gpt-4o" else ""
        print(f"    {m}{marker}")

    # CopilotModel: Literal[...] — for IDE autocompletion / type checking
    print(f"\n  CopilotModel is a Literal type alias: {CopilotModel}")

    # All valid models accepted without error
    print("\n  Instantiating each valid model (no errors expected):")
    for m in sorted(Copilot.MODELS):
        with patch("actor_ai.providers.openai.OpenAI"):
            Copilot(m)
        print(f"    Copilot({m!r}) ✓")

    # Invalid model raises ValueError immediately — before any network call
    _divider("4b. Invalid model → ValueError at construction time")
    try:
        with patch("actor_ai.providers.openai.OpenAI"):
            Copilot("gpt-3.5-turbo")  # type: ignore[arg-type]
    except ValueError as exc:
        print(f"  ValueError: {exc}")


# ── 5. Tool calling through Copilot ──────────────────────────────────────────


def _section_tools() -> None:
    _divider("5. Tool calling via Copilot")

    prov = ToolCallingProvider(
        tool_name="list_files",
        tool_args={"path": "src/"},
        final_reply="Found 11 Python source files in src/actor_ai/.",
    )

    class FileAgent(AIActor):
        system_prompt = (
            "You are a GitHub Copilot assistant. "
            "Use tools to answer questions about the repository."
        )
        provider = prov

        @tool("List Python files in a directory path.")
        def list_files(self, path: str) -> list[str]:
            # Standard library imports:
            import glob

            return glob.glob(f"{path}**/*.py", recursive=True)

    ref = FileAgent.start()
    try:
        reply = ref.proxy().instruct("How many Python files are in src/?").get()
        print("  Tool called : list_files(path='src/')")
        print(f"  Tool result : {prov.tool_result}")
        print(f"  Reply       : {reply}")
    finally:
        ref.stop()


# ── 6. Copilot in a Chorus ────────────────────────────────────────────────────


def _section_chorus() -> None:
    _divider("6. Copilot agents in a Chorus")

    class CopilotReviewer(AIActor):
        system_prompt = "You are a senior code reviewer using GitHub Copilot."
        provider = ScriptedProvider(
            [
                "I am the Copilot Reviewer — I flag issues and suggest improvements.",
                "The implementation looks solid. Minor suggestion: add type hints "
                "to the helper functions for better IDE support.",
            ]
        )

    class CopilotDocWriter(AIActor):
        system_prompt = "You are a technical writer using GitHub Copilot."
        provider = ScriptedProvider(
            [
                "I am the Copilot Doc Writer — I produce clear, concise docs.",
                "**Helper Functions** — process input data and return validated "
                "results. Add `:param x:` and `:returns:` tags for full coverage.",
            ]
        )

    reviewer_ref = CopilotReviewer.start()
    doc_writer_ref = CopilotDocWriter.start()

    chorus_ref = Chorus.start(agents={"reviewer": reviewer_ref, "doc_writer": doc_writer_ref})
    chorus = chorus_ref.proxy()

    try:
        print(f"  Agents: {', '.join(chorus.agents().get())}")

        replies = chorus.broadcast("Introduce yourself in one sentence.").get()
        for name, r in replies.items():
            print(f"\n  [{name}] {r}")

        final = chorus.pipeline(
            ["reviewer", "doc_writer"],
            "Review and then document the helper functions in utils.py.",
        ).get()
        print(f"\n  Pipeline output:\n  {final}")
    finally:
        chorus_ref.stop()
        reviewer_ref.stop()
        doc_writer_ref.stop()


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    _section_construction()
    _section_token({})
    _section_instruct()
    _section_models()
    _section_tools()
    _section_chorus()


if __name__ == "__main__":
    main()
