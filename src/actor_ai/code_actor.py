"""PythonCodeActor — non-AI actor backed by a Python module."""

# Future imports (must occur at the beginning of the file):
from __future__ import annotations

# Standard library imports:
import importlib.util
import logging
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# Third party imports:
import pykka

_log = logging.getLogger(__name__)

# Sentinel attribute names placed on decorated functions.
_ATTR_EVENT = "_code_actor_event"
_ATTR_CRON = "_code_actor_cron"
_ATTR_MESSAGE = "_code_actor_message"
_ATTR_LOOP = "_code_actor_loop"

_MAX_LOG_LINES = 200


# ── Public decorators ─────────────────────────────────────────────────────────


def event(name: str) -> Callable[[Callable], Callable]:
    """Register a function as a handler for the named event.

    The function signature must be ``(context: str) -> str``.
    """

    def decorator(fn: Callable) -> Callable:
        setattr(fn, _ATTR_EVENT, name)
        return fn

    return decorator


def cron(schedule: str, *, name: str | None = None) -> Callable[[Callable], Callable]:
    """Register a function as a cron job handler.

    ``schedule`` is a standard 5-field cron expression.  ``name`` defaults to
    the function name and is used when triggering the job from the runtime.

    The function signature must be ``() -> str``.
    """

    def decorator(fn: Callable) -> Callable:
        setattr(fn, _ATTR_CRON, {"schedule": schedule, "name": name or fn.__name__})
        return fn

    return decorator


def on_message(fn: Callable) -> Callable:
    """Register a function as the actor's instruct/message handler.

    The function signature must be ``(text: str) -> str``.
    """
    setattr(fn, _ATTR_MESSAGE, True)
    return fn


def loop(fn: Callable) -> Callable:
    """Register a function as the actor's main background loop.

    The function receives a single ``ctx: CodeActorContext`` argument.  It
    should loop until ``ctx.stop.is_set()`` and may call ``ctx.log(msg)`` to
    write to the actor's log buffer.
    """
    setattr(fn, _ATTR_LOOP, True)
    return fn


# ── Runtime context ───────────────────────────────────────────────────────────


class CodeActorContext:
    """Passed to the loop function, providing the kill switch and log helper."""

    def __init__(self, stop: threading.Event, buf: deque[str], lock: threading.Lock) -> None:
        self.stop = stop
        self._buf = buf
        self._lock = lock

    def log(self, message: str) -> None:
        """Append a line to the actor's log buffer (visible via get_logs())."""
        with self._lock:
            self._buf.append(str(message))


# ── Internal dataclasses ──────────────────────────────────────────────────────


@dataclass
class CodeEventDef:
    name: str
    handler: Callable


@dataclass
class CodeCronDef:
    name: str
    schedule: str
    handler: Callable


@dataclass
class _ModuleInfo:
    events: list[CodeEventDef]
    crons: list[CodeCronDef]
    message_handler: Callable | None
    loop_fn: Callable | None


# ── Module loader ─────────────────────────────────────────────────────────────


def _load_module(path: str | Path) -> _ModuleInfo:
    """Import a Python file and discover its events, crons, handler, and loop."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Code actor script not found: {p}")

    spec = importlib.util.spec_from_file_location(p.stem, p)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec for {p}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]

    events: list[CodeEventDef] = []
    crons: list[CodeCronDef] = []
    message_handler: Callable | None = None
    loop_fn: Callable | None = None

    for attr_name in dir(mod):
        obj = getattr(mod, attr_name)
        if not callable(obj):
            continue
        if hasattr(obj, _ATTR_EVENT):
            events.append(CodeEventDef(name=getattr(obj, _ATTR_EVENT), handler=obj))
        if hasattr(obj, _ATTR_CRON):
            meta = getattr(obj, _ATTR_CRON)
            crons.append(CodeCronDef(name=meta["name"], schedule=meta["schedule"], handler=obj))
        if getattr(obj, _ATTR_MESSAGE, False):
            message_handler = obj
        if getattr(obj, _ATTR_LOOP, False):
            loop_fn = obj

    return _ModuleInfo(events=events, crons=crons, message_handler=message_handler, loop_fn=loop_fn)


# ── PythonCodeActor ───────────────────────────────────────────────────────────


class PythonCodeActor(pykka.ThreadingActor):
    """A non-AI pykka actor whose behaviour is defined by a Python module.

    The module declares its interface using the decorators in this module:

    * :func:`on_message` — handles ``instruct()`` calls
    * :func:`event` — registers named event handlers
    * :func:`cron` — registers cron job handlers
    * :func:`loop` — optional long-running background loop

    Example module::

        from actor_ai.code_actor import event, cron, on_message, loop

        @on_message
        def handle(text: str) -> str:
            return f"echo: {text}"

        @event("analyze")
        def analyze(context: str) -> str:
            return f"analyzed: {context}"

        @cron("0 9 * * 1-5")
        def morning_brief() -> str:
            return "Good morning!"

        @loop
        def run(ctx) -> None:
            import time
            while not ctx.stop.is_set():
                ctx.log("tick")
                time.sleep(60)

    Subclass attributes (set by ``type()`` in ActorRuntime):

    * ``script_path`` — absolute path to the ``.py`` module file.
    * ``actor_name`` — human-readable name for log messages.
    """

    script_path: str = ""
    actor_name: str = ""

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def on_start(self) -> None:
        self._log_lock = threading.Lock()
        self._log_buf: deque[str] = deque(maxlen=_MAX_LOG_LINES)
        self._stop_event = threading.Event()
        self._loop_thread: threading.Thread | None = None

        info = _load_module(self.script_path)
        self._events: dict[str, CodeEventDef] = {e.name: e for e in info.events}
        self._crons: list[CodeCronDef] = info.crons
        self._message_handler = info.message_handler
        self._ctx = CodeActorContext(self._stop_event, self._log_buf, self._log_lock)

        if info.loop_fn is not None:
            self._loop_thread = threading.Thread(
                target=self._run_loop,
                args=(info.loop_fn,),
                daemon=True,
                name=f"{self.actor_name or 'code'}-loop",
            )
            self._loop_thread.start()

        _log.info(
            "Code actor %r started (events=%r, crons=%r)",
            self.actor_name,
            list(self._events),
            [c.name for c in self._crons],
        )

    def on_stop(self) -> None:
        self._stop_event.set()
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=5)
        _log.info("Code actor %r stopped", self.actor_name)

    def _run_loop(self, loop_fn: Callable) -> None:
        try:
            loop_fn(self._ctx)
        except Exception as exc:
            _log.error("Loop in code actor %r raised: %s", self.actor_name, exc)
            with self._log_lock:
                self._log_buf.append(f"[error] {exc}")

    # ── Messaging ──────────────────────────────────────────────────────────

    def instruct(self, text: str) -> str:
        if self._message_handler is None:
            return ""
        result = self._message_handler(text)
        return str(result) if result is not None else ""

    def fire_event(self, event_name: str, context: str = "") -> str:
        evt = self._events.get(event_name)
        if evt is None:
            raise ValueError(f"Unknown event {event_name!r} on actor {self.actor_name!r}")
        result = evt.handler(context)
        return str(result) if result is not None else ""

    def run_cron(self, cron_name: str) -> str:
        crn = next((c for c in self._crons if c.name == cron_name), None)
        if crn is None:
            raise ValueError(f"Unknown cron {cron_name!r} on actor {self.actor_name!r}")
        result = crn.handler()
        return str(result) if result is not None else ""

    # ── Introspection ──────────────────────────────────────────────────────

    def get_events(self) -> list[dict]:
        return [{"name": name} for name in self._events]

    def get_crons(self) -> list[dict]:
        return [{"name": c.name, "schedule": c.schedule} for c in self._crons]

    def get_logs(self) -> str:
        with self._log_lock:
            return "\n".join(self._log_buf)
