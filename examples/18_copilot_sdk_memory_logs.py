"""18 - GitHub Copilot SDK memory and logs

This example runs prompts through ``Copilot(use_sdk=True)``, then prints
actor memory, working memory, session history, shared conversation logs, and
ledger usage.

Requirements:
  - GitHub Copilot CLI/SDK authentication available, or pass ``api_key=...``
  - A GitHub account with an active Copilot subscription

Run:
    uv run python examples/18_copilot_sdk_memory_logs.py
"""

# Future imports (must occur at the beginning of the file):
from __future__ import annotations

# Standard library imports:
from textwrap import shorten

# Local imports:
from actor_ai import AIActor, Copilot, Ledger, SharedContext

context = SharedContext()
ledger = Ledger()

COPILOT_TIMEOUT = 60.0
PROXY_TIMEOUT = COPILOT_TIMEOUT + 10.0


class CopilotMemoryAssistant(AIActor):
    system_prompt = (
        "You are a concise assistant helping test actor-ai memory and logs. "
        "Use known facts and working memory when they are relevant."
    )
    provider = Copilot("claude-sonnet-4.5", use_sdk=True, timeout=COPILOT_TIMEOUT)
    actor_name = "CopilotMemoryAssistant"
    context = context
    ledger = ledger
    max_history = 10


def _print_mapping(title: str, values: dict[str, str]) -> None:
    print(f"\n{title}")
    if not values:
        print("  (empty)")
        return
    for key, value in values.items():
        print(f"  {key}: {value}")


def _print_log(log: list[dict]) -> None:
    print("\nConversation log")
    if not log:
        print("  (empty)")
        return
    for index, entry in enumerate(log, start=1):
        content = shorten(entry["content"].replace("\n", " "), width=120, placeholder="...")
        print(f"  {index:02}. [{entry['agent']}] {entry['role']}: {content}")


def _print_session(session: list[dict]) -> None:
    print("\nActor session")
    if not session:
        print("  (empty)")
        return
    for index, message in enumerate(session, start=1):
        content = shorten(message["content"].replace("\n", " "), width=120, placeholder="...")
        print(f"  {index:02}. {message['role']}: {content}")


def _print_usage() -> None:
    usage = ledger.total_usage()
    print("\nLedger usage")
    print(f"  entries: {len(ledger)}")
    print(f"  input tokens: {usage.input_tokens}")
    print(f"  output tokens: {usage.output_tokens}")
    print(f"  reasoning tokens: {usage.reasoning_tokens}")
    print(f"  cache read tokens: {usage.cache_read_tokens}")
    print(f"  cache write tokens: {usage.cache_write_tokens}")
    print(f"  cache tokens: {usage.cache_tokens}")
    print(f"  total tokens: {usage.total_tokens}")


def main() -> None:
    prompts = {
        "Remember that the project codename is Project The Long and Winding Road and confirm it in one sentence.": PROXY_TIMEOUT,
        "What project codename do you know about? Keep the answer short.": PROXY_TIMEOUT,
        "Remember that today's task is validating memory and logs.": PROXY_TIMEOUT,
        "Summarize the project codename and today's task as two bullets.": PROXY_TIMEOUT,
        "Paint me like one of your French girls.": PROXY_TIMEOUT,
        """Create a script in python that collects the current temperature in Celsius for the 10 largest cities in the
world using a weather API. Don't worry about API keys or actual implementation details, just provide a plausible script.
Display your script in your reply.""": PROXY_TIMEOUT * 5,
        "Give a final one-sentence status update for this memory/log test.": PROXY_TIMEOUT,
    }

    ref = CopilotMemoryAssistant.start()
    proxy = ref.proxy()

    try:
        proxy.remember("project_codename", "Atlas").get(timeout=PROXY_TIMEOUT)
        proxy.remember_working("current_task", "validating memory and logs").get(
            timeout=PROXY_TIMEOUT
        )

        print(
            f"Running {len(prompts)} prompts through Copilot SDK "
            f"(Copilot timeout: {COPILOT_TIMEOUT}s, proxy timeout: {PROXY_TIMEOUT}s)...\n"
        )
        for index, (prompt, proxy_timeout) in enumerate(prompts.items(), start=1):
            reply = proxy.instruct(prompt).get(timeout=proxy_timeout)
            print(f"Prompt {index} (proxy timeout: {proxy_timeout}s): {prompt}")
            print(f"Reply  {index}: {reply}\n")

        _print_mapping("Long-term memory", proxy.get_memory().get(timeout=PROXY_TIMEOUT))
        _print_mapping("Working memory", proxy.get_working_memory().get(timeout=PROXY_TIMEOUT))
        _print_session(proxy.get_session().get(timeout=PROXY_TIMEOUT))
        _print_log(context.get_log())
        _print_usage()
    finally:
        ref.stop()


if __name__ == "__main__":
    main()
