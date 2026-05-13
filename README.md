# actor-ai

**Multi-provider AI agents built on [pykka](https://github.com/jodal/pykka).**

`actor-ai` extends pykka's actor framework so each actor can process natural-language instructions, call tools, maintain conversation sessions, remember facts, track token spend, and monitor traffic — all with a clean, provider-agnostic API.

## Installation

```bash
pip install actor-ai          # or: uv add actor-ai
```

Requires Python ≥ 3.14 and one (or more) provider API keys.

## Quick start

```python
from actor_ai import AIActor, Claude

class Assistant(AIActor):
    system_prompt = "You are a helpful assistant."
    provider = Claude()          # requires ANTHROPIC_API_KEY

ref = Assistant.start()
reply = ref.proxy().instruct("What is the capital of France?").get()
print(reply)   # → "The capital of France is Paris."
ref.stop()
```

## Providers

| Class | Backend | Environment variable |
|---|---|---|
| `Claude` | Anthropic | `ANTHROPIC_API_KEY` |
| `GPT` | OpenAI | `OPENAI_API_KEY` |
| `Gemini` | Google AI (OpenAI-compat) | `GOOGLE_API_KEY` |
| `Mistral` | Mistral AI | `MISTRAL_API_KEY` |
| `DeepSeek` | DeepSeek | `DEEPSEEK_API_KEY` |
| `LiteLLM` | Any (100+ models) | depends on model |

## Key features

- **Multi-turn sessions** — rolling conversation history, configurable window (`max_history`)
- **Long-term memory** — `remember(key, value)` / `forget(key)` facts injected into every system prompt
- **Tool calling** — decorate methods with `@tool` to expose them to the LLM
- **Chorus** — orchestrate groups of agents (`broadcast`, `pipeline`)
- **Accounting** — track token usage and cost per actor, model, and session (`Ledger`, `Rates`)
- **Monitoring** — forward per-call metadata to LiteLLM callbacks (Langfuse, Helicone, custom)

## Examples

The [`examples/`](examples/) directory contains ten self-contained, runnable scripts.  
Each example uses a fake scripted provider so no API key is required to run them.

```
examples/
  01_hello_world.py    — One-turn interaction
  02_session.py        — Session history, max_history, clear_session
  03_memory.py         — remember() / forget()
  04_tools.py          — @tool decorator and tool calling
  05_providers.py      — All six provider configurations
  06_chorus.py         — Chorus: instruct, broadcast, pipeline
  07_accounting.py     — Ledger and Rates
  08_monitoring.py     — LiteLLM monitoring
  09_messages.py       — Instruct / Remember / Forget message objects
  10_advanced.py       — Full pipeline combining all features
```

Run any example:

```bash
uv run python examples/01_hello_world.py
```

## User Manual

See [MANUAL.md](MANUAL.md) for the complete reference.

## License

MIT
