"""Tests for the @tool decorator, extract_tools(), and schema building."""

# Future imports (must occur at the beginning of the file):
from __future__ import annotations

# Third party imports:
import pytest

# Local imports:
from actor_ai.tools import _build_tool_spec, _to_json_type, extract_tools, tool

# ---------------------------------------------------------------------------
# @tool decorator
# ---------------------------------------------------------------------------


class TestToolDecorator:
    def test_marks_function_as_tool(self):
        @tool
        def my_func(self):
            pass

        assert my_func._is_ai_tool is True

    def test_uses_docstring_as_description(self):
        @tool
        def my_func(self):
            "This is the docstring."
            pass

        assert my_func._tool_description == "This is the docstring."

    def test_strips_docstring_whitespace(self):
        @tool
        def my_func(self):
            """padded"""
            pass

        assert my_func._tool_description == "padded"

    def test_empty_description_when_no_docstring(self):
        @tool
        def my_func(self):
            pass

        assert my_func._tool_description == ""

    def test_explicit_description_string(self):
        @tool("Custom description")
        def my_func(self):
            "Docstring that should be ignored."
            pass

        assert my_func._is_ai_tool is True
        assert my_func._tool_description == "Custom description"

    def test_explicit_description_overrides_docstring(self):
        @tool("Explicit")
        def my_func(self):
            "Docstring"
            pass

        assert my_func._tool_description == "Explicit"

    def test_empty_string_arg_falls_back_to_docstring(self):
        @tool("")
        def my_func(self):
            "Fallback"
            pass

        assert my_func._tool_description == "Fallback"

    def test_none_arg_falls_back_to_docstring(self):
        @tool(None)
        def my_func(self):
            "From docstring"
            pass

        assert my_func._tool_description == "From docstring"

    def test_returns_same_function_when_used_bare(self):
        def original(self):
            pass

        decorated = tool(original)
        assert decorated is original

    def test_returns_wrapped_function_with_string_arg(self):
        @tool("desc")
        def my_func(self):
            pass

        assert my_func._is_ai_tool is True

    def test_non_tool_function_has_no_marker(self):
        def plain(self):
            pass

        assert not getattr(plain, "_is_ai_tool", False)


# ---------------------------------------------------------------------------
# _to_json_type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "python_type, expected",
    [
        (str, "string"),
        (int, "integer"),
        (float, "number"),
        (bool, "boolean"),
        (list, "array"),
        (dict, "object"),
        (None, "string"),
        (bytes, "string"),
        (tuple, "string"),
        (set, "string"),
    ],
)
def test_to_json_type(python_type, expected):
    assert _to_json_type(python_type) == expected


# ---------------------------------------------------------------------------
# _build_tool_spec
# ---------------------------------------------------------------------------


class TestBuildToolSpec:
    def _decorated(self, func, description=""):
        func._is_ai_tool = True
        func._tool_description = description
        return func

    def test_name_in_spec(self):
        def func(self): ...

        spec = _build_tool_spec("my_name", self._decorated(func))
        assert spec["name"] == "my_name"

    def test_description_from_tool_description_attr(self):
        def func(self): ...

        spec = _build_tool_spec("f", self._decorated(func, "Explicit desc"))
        assert spec["description"] == "Explicit desc"

    def test_description_falls_back_to_docstring(self):
        def func(self):
            "Docstring desc"
            ...

        func._tool_description = ""
        spec = _build_tool_spec("f", func)
        assert spec["description"] == "Docstring desc"

    def test_no_params_beyond_self(self):
        @tool
        def func(self): ...

        spec = _build_tool_spec("func", func)
        assert spec["input_schema"]["properties"] == {}
        assert spec["input_schema"]["required"] == []

    def test_required_param_no_default(self):
        @tool
        def func(self, x: int): ...

        spec = _build_tool_spec("func", func)
        assert "x" in spec["input_schema"]["required"]
        assert spec["input_schema"]["properties"]["x"] == {"type": "integer"}

    def test_optional_param_with_default_not_required(self):
        @tool
        def func(self, x: str = "default"): ...

        spec = _build_tool_spec("func", func)
        assert "x" not in spec["input_schema"]["required"]
        assert "x" in spec["input_schema"]["properties"]

    def test_mixed_required_and_optional(self):
        @tool
        def func(self, required: str, optional: int = 0): ...

        spec = _build_tool_spec("func", func)
        assert spec["input_schema"]["required"] == ["required"]
        assert set(spec["input_schema"]["properties"]) == {"required", "optional"}

    def test_multiple_required_params(self):
        @tool
        def func(self, a: str, b: int, c: float): ...

        spec = _build_tool_spec("func", func)
        assert set(spec["input_schema"]["required"]) == {"a", "b", "c"}

    def test_type_annotations_mapped(self):
        @tool
        def func(self, s: str, i: int, f: float, b: bool, lst: list, d: dict): ...

        spec = _build_tool_spec("func", func)
        props = spec["input_schema"]["properties"]
        assert props["s"]["type"] == "string"
        assert props["i"]["type"] == "integer"
        assert props["f"]["type"] == "number"
        assert props["b"]["type"] == "boolean"
        assert props["lst"]["type"] == "array"
        assert props["d"]["type"] == "object"

    def test_unannotated_param_defaults_to_string(self):
        @tool
        def func(self, x): ...

        spec = _build_tool_spec("func", func)
        assert spec["input_schema"]["properties"]["x"]["type"] == "string"

    def test_input_schema_type_is_object(self):
        @tool
        def func(self): ...

        spec = _build_tool_spec("func", func)
        assert spec["input_schema"]["type"] == "object"

    def test_self_is_excluded_from_properties(self):
        @tool
        def func(self, x: int): ...

        spec = _build_tool_spec("func", func)
        assert "self" not in spec["input_schema"]["properties"]


# ---------------------------------------------------------------------------
# extract_tools
# ---------------------------------------------------------------------------


class TestExtractTools:
    def test_empty_when_no_tool_methods(self):
        # Local imports:
        from actor_ai import AIActor

        class Actor(AIActor):
            def not_a_tool(self):
                pass

        actor = Actor.__new__(Actor)
        assert extract_tools(actor) == []

    def test_finds_single_tool(self):
        # Local imports:
        from actor_ai import AIActor

        class Actor(AIActor):
            @tool
            def my_tool(self, x: int) -> str:
                "Does something."
                return str(x)

        actor = Actor.__new__(Actor)
        tools = extract_tools(actor)
        assert len(tools) == 1
        assert tools[0]["name"] == "my_tool"

    def test_finds_multiple_tools(self):
        # Local imports:
        from actor_ai import AIActor

        class Actor(AIActor):
            @tool
            def tool_a(self): ...

            @tool
            def tool_b(self): ...

            @tool
            def tool_c(self): ...

        actor = Actor.__new__(Actor)
        names = {t["name"] for t in extract_tools(actor)}
        assert names == {"tool_a", "tool_b", "tool_c"}

    def test_excludes_private_methods(self):
        # Local imports:
        from actor_ai import AIActor

        class Actor(AIActor):
            @tool
            def public_tool(self): ...

        # Manually mark a private method — should still be excluded by name filter
        Actor._private = lambda self: None
        Actor._private._is_ai_tool = True

        actor = Actor.__new__(Actor)
        tools = extract_tools(actor)
        assert all(t["name"] == "public_tool" for t in tools)

    def test_does_not_include_non_tool_methods(self):
        # Local imports:
        from actor_ai import AIActor

        class Actor(AIActor):
            @tool
            def is_tool(self): ...

            def is_not_tool(self): ...

        actor = Actor.__new__(Actor)
        names = {t["name"] for t in extract_tools(actor)}
        assert "is_tool" in names
        assert "is_not_tool" not in names

    def test_inherited_tools_included(self):
        # Local imports:
        from actor_ai import AIActor

        class Base(AIActor):
            @tool
            def base_tool(self): ...

        class Child(Base):
            @tool
            def child_tool(self): ...

        actor = Child.__new__(Child)
        names = {t["name"] for t in extract_tools(actor)}
        assert "base_tool" in names
        assert "child_tool" in names

    def test_get_type_hints_failure_is_silently_ignored(self):
        """If get_type_hints() raises (e.g. unresolvable forward ref), fall back gracefully."""
        # Local imports:
        from actor_ai.tools import _build_tool_spec

        def func(self, x: CompletelyNonExistentType123) -> None: ...  # type: ignore[name-defined]  # noqa: F821

        func._is_ai_tool = True
        func._tool_description = "desc"
        # Must not raise; unknown type falls back to "string"
        spec = _build_tool_spec("func", func)
        assert spec["input_schema"]["properties"]["x"]["type"] == "string"

    def test_spec_shape(self):
        # Local imports:
        from actor_ai import AIActor

        class Actor(AIActor):
            @tool("A helpful tool")
            def compute(self, value: int) -> str:
                return str(value)

        actor = Actor.__new__(Actor)
        tools = extract_tools(actor)
        spec = tools[0]
        assert spec["name"] == "compute"
        assert spec["description"] == "A helpful tool"
        assert "input_schema" in spec
        assert spec["input_schema"]["type"] == "object"
