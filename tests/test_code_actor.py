"""Tests for actor_ai.code_actor — decorators, module discovery, PythonCodeActor."""

# Future imports (must occur at the beginning of the file):
from __future__ import annotations

# Standard library imports:
import threading
import time
from pathlib import Path
from unittest.mock import patch

# Third party imports:
import pytest

# Local imports:
from actor_ai.code_actor import (
    CodeActorContext,
    PythonCodeActor,
    _load_module,
    cron,
    event,
    loop,
    on_message,
)

# ── Decorator attribute marking ───────────────────────────────────────────────


def test_event_decorator_sets_attribute():
    @event("my-event")
    def handler(context: str) -> str:
        return context

    assert getattr(handler, "_code_actor_event") == "my-event"


def test_cron_decorator_sets_attribute():
    @cron("0 9 * * *")
    def job() -> str:
        return "done"

    meta = getattr(job, "_code_actor_cron")
    assert meta["schedule"] == "0 9 * * *"
    assert meta["name"] == "job"


def test_cron_decorator_custom_name():
    @cron("0 9 * * *", name="morning")
    def job() -> str:
        return "done"

    assert getattr(job, "_code_actor_cron")["name"] == "morning"


def test_on_message_decorator():
    @on_message
    def handle(text: str) -> str:
        return text

    assert getattr(handle, "_code_actor_message") is True


def test_loop_decorator():
    @loop
    def run(ctx):
        pass

    assert getattr(run, "_code_actor_loop") is True


# ── Module discovery ──────────────────────────────────────────────────────────


def _write_module(tmp_path: Path, code: str) -> Path:
    p = tmp_path / "actor_module.py"
    p.write_text(code, encoding="utf-8")
    return p


def test_load_module_discovers_event(tmp_path):
    p = _write_module(
        tmp_path,
        """
from actor_ai.code_actor import event

@event("greet")
def greet(context):
    return f"hello {context}"
""",
    )
    info = _load_module(p)
    assert len(info.events) == 1
    assert info.events[0].name == "greet"
    assert info.events[0].handler("world") == "hello world"


def test_load_module_discovers_cron(tmp_path):
    p = _write_module(
        tmp_path,
        """
from actor_ai.code_actor import cron

@cron("*/5 * * * *")
def tick():
    return "tick"
""",
    )
    info = _load_module(p)
    assert len(info.crons) == 1
    assert info.crons[0].schedule == "*/5 * * * *"
    assert info.crons[0].name == "tick"


def test_load_module_discovers_message_handler(tmp_path):
    p = _write_module(
        tmp_path,
        """
from actor_ai.code_actor import on_message

@on_message
def handle(text):
    return "ack: " + text
""",
    )
    info = _load_module(p)
    assert info.message_handler is not None
    assert info.message_handler("ping") == "ack: ping"


def test_load_module_discovers_loop(tmp_path):
    p = _write_module(
        tmp_path,
        """
from actor_ai.code_actor import loop

@loop
def run(ctx):
    ctx.stop.wait()
""",
    )
    info = _load_module(p)
    assert info.loop_fn is not None


def test_load_module_missing_file():
    with pytest.raises(FileNotFoundError):
        _load_module("/nonexistent/path/actor.py")


def test_load_module_bad_spec(tmp_path):
    # Standard library imports:
    import importlib.util

    p = _write_module(tmp_path, "# module")
    with patch.object(importlib.util, "spec_from_file_location", return_value=None):
        with pytest.raises(ImportError):
            _load_module(p)


def test_load_module_empty_module(tmp_path):
    p = _write_module(tmp_path, "# no decorators")
    info = _load_module(p)
    assert info.events == []
    assert info.crons == []
    assert info.message_handler is None
    assert info.loop_fn is None


# ── CodeActorContext ──────────────────────────────────────────────────────────


def test_context_log_appends():
    # Standard library imports:
    from collections import deque

    buf: deque[str] = deque()
    lock = threading.Lock()
    ctx = CodeActorContext(threading.Event(), buf, lock)
    ctx.log("hello")
    ctx.log("world")
    assert list(buf) == ["hello", "world"]


def test_context_stop_event():
    stop = threading.Event()
    ctx = CodeActorContext(stop, deque(), threading.Lock())
    assert not ctx.stop.is_set()
    stop.set()
    assert ctx.stop.is_set()


# Standard library imports:
from collections import deque  # noqa: E402  (after test helpers that use it)

# ── PythonCodeActor (full lifecycle) ─────────────────────────────────────────


def _make_actor_cls(script_path: str, name: str = "test-actor") -> type:
    return type(name, (PythonCodeActor,), {"script_path": script_path, "actor_name": name})


def test_python_code_actor_instruct(tmp_path, actor_factory):
    p = _write_module(
        tmp_path,
        """
from actor_ai.code_actor import on_message

@on_message
def handle(text):
    return "reply: " + text
""",
    )
    cls = _make_actor_cls(str(p))
    ref = actor_factory(cls)
    result = ref.proxy().instruct("ping").get(timeout=5)
    assert result == "reply: ping"


def test_python_code_actor_no_message_handler_returns_empty(tmp_path, actor_factory):
    p = _write_module(tmp_path, "# no handler")
    cls = _make_actor_cls(str(p))
    ref = actor_factory(cls)
    result = ref.proxy().instruct("anything").get(timeout=5)
    assert result == ""


def test_python_code_actor_fire_event(tmp_path, actor_factory):
    p = _write_module(
        tmp_path,
        """
from actor_ai.code_actor import event

@event("process")
def process(context):
    return "done: " + context
""",
    )
    cls = _make_actor_cls(str(p))
    ref = actor_factory(cls)
    result = ref.proxy().fire_event("process", "data").get(timeout=5)
    assert result == "done: data"


def test_python_code_actor_unknown_event_raises(tmp_path, actor_factory):
    p = _write_module(tmp_path, "# no events")
    cls = _make_actor_cls(str(p))
    ref = actor_factory(cls)
    with pytest.raises(ValueError, match="Unknown event"):
        ref.proxy().fire_event("ghost").get(timeout=5)


def test_python_code_actor_run_cron(tmp_path, actor_factory):
    p = _write_module(
        tmp_path,
        """
from actor_ai.code_actor import cron

@cron("0 * * * *", name="hourly")
def hourly():
    return "hourly done"
""",
    )
    cls = _make_actor_cls(str(p))
    ref = actor_factory(cls)
    result = ref.proxy().run_cron("hourly").get(timeout=5)
    assert result == "hourly done"


def test_python_code_actor_unknown_cron_raises(tmp_path, actor_factory):
    p = _write_module(tmp_path, "# no crons")
    cls = _make_actor_cls(str(p))
    ref = actor_factory(cls)
    with pytest.raises(ValueError, match="Unknown cron"):
        ref.proxy().run_cron("ghost").get(timeout=5)


def test_python_code_actor_get_events(tmp_path, actor_factory):
    p = _write_module(
        tmp_path,
        """
from actor_ai.code_actor import event

@event("alpha")
def alpha(ctx): return ""

@event("beta")
def beta(ctx): return ""
""",
    )
    cls = _make_actor_cls(str(p))
    ref = actor_factory(cls)
    events = ref.proxy().get_events().get(timeout=5)
    names = {e["name"] for e in events}
    assert names == {"alpha", "beta"}


def test_python_code_actor_get_crons(tmp_path, actor_factory):
    p = _write_module(
        tmp_path,
        """
from actor_ai.code_actor import cron

@cron("0 9 * * *", name="morning")
def morning(): return ""
""",
    )
    cls = _make_actor_cls(str(p))
    ref = actor_factory(cls)
    crons = ref.proxy().get_crons().get(timeout=5)
    assert crons == [{"name": "morning", "schedule": "0 9 * * *"}]


def test_python_code_actor_loop_runs_and_stops(tmp_path, actor_factory):
    p = _write_module(
        tmp_path,
        """
from actor_ai.code_actor import loop

@loop
def run(ctx):
    ctx.log("started")
    ctx.stop.wait()
    ctx.log("stopped")
""",
    )
    cls = _make_actor_cls(str(p))
    ref = actor_factory(cls)
    time.sleep(0.2)  # let the loop thread start
    ref.stop()
    # Actor stopped cleanly — no error raised
    time.sleep(0.1)


def test_python_code_actor_logs(tmp_path, actor_factory):
    p = _write_module(
        tmp_path,
        """
from actor_ai.code_actor import loop

@loop
def run(ctx):
    ctx.log("line-one")
    ctx.log("line-two")
    ctx.stop.wait()
""",
    )
    cls = _make_actor_cls(str(p))
    ref = actor_factory(cls)
    time.sleep(0.2)  # let the loop run
    logs = ref.proxy().get_logs().get(timeout=5)
    assert "line-one" in logs
    assert "line-two" in logs


def test_python_code_actor_none_return_becomes_empty_string(tmp_path, actor_factory):
    p = _write_module(
        tmp_path,
        """
from actor_ai.code_actor import on_message

@on_message
def handle(text):
    return None
""",
    )
    cls = _make_actor_cls(str(p))
    ref = actor_factory(cls)
    result = ref.proxy().instruct("x").get(timeout=5)
    assert result == ""


def test_python_code_actor_loop_exception_logged(tmp_path, actor_factory):
    p = _write_module(
        tmp_path,
        """
from actor_ai.code_actor import loop

@loop
def run(ctx):
    raise RuntimeError("boom")
""",
    )
    cls = _make_actor_cls(str(p))
    ref = actor_factory(cls)
    time.sleep(0.2)  # let the loop thread run (and raise immediately)
    logs = ref.proxy().get_logs().get(timeout=5)
    assert "boom" in logs
