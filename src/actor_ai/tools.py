# Future imports (must occur at the beginning of the file):
from __future__ import annotations

# Standard library imports:
import inspect
from collections.abc import Callable
from typing import Any, get_type_hints


def tool(func_or_desc: Callable | str | None = None) -> Any:
    """Mark an AIActor method as callable by the AI.

    Usage:
        @tool
        def my_method(self, x: int) -> str: ...

        @tool("Custom description")
        def my_method(self, x: int) -> str: ...
    """
    if callable(func_or_desc):
        func_or_desc._is_ai_tool = True  # type: ignore[attr-defined]
        func_or_desc._tool_description = (func_or_desc.__doc__ or "").strip()  # type: ignore[attr-defined]
        return func_or_desc

    description = func_or_desc or ""

    def decorator(f: Callable) -> Callable:
        f._is_ai_tool = True  # type: ignore[attr-defined]
        f._tool_description = description or (f.__doc__ or "").strip()  # type: ignore[attr-defined]
        return f

    return decorator


def extract_tools(actor_instance: Any) -> list[dict]:
    """Return Claude-compatible tool specs for all @tool-decorated methods."""
    tools = []
    for name in dir(type(actor_instance)):
        if name.startswith("_"):
            continue
        cls_attr = getattr(type(actor_instance), name, None)
        if cls_attr is not None and getattr(cls_attr, "_is_ai_tool", False):
            tools.append(_build_tool_spec(name, cls_attr))
    return tools


def _build_tool_spec(name: str, func: Callable) -> dict:
    try:
        hints = get_type_hints(func)
    except Exception:  # noqa: BLE001
        hints = {}

    sig = inspect.signature(func)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        prop: dict[str, Any] = {"type": _to_json_type(hints.get(param_name))}
        # Include return annotation in description if helpful
        properties[param_name] = prop
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "name": name,
        "description": getattr(func, "_tool_description", None) or (func.__doc__ or "").strip(),
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


def _to_json_type(python_type: Any) -> str:
    mapping: dict[Any, str] = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }
    return mapping.get(python_type, "string")
