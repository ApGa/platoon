"""Typed tool-call arguments from the request's tool schemas, for formats that sample argument values as text.

In XML-style formats (``qwen3_coder``, ``glm45``) every argument value is text: the chat template writes a string as
it is and anything else as JSON (Qwen3.5's template uses Python's ``str`` instead, so ``True`` and ``None``), which
makes ``3`` the integer or the string.  :class:`ToolSchemas` types each value by its parameter's JSON schema in the
request's tools.  A value stays text when the schema does not describe the parameter, does not say its type (a
``$ref``, say) or allows text and nothing else, and when the text does not read as any type the schema allows.
JSON-native formats need none of this: their arguments are typed already.

Arguments are JSON values, which the gateway serves as JSON text: :data:`STRICT_JSON` and :data:`LENIENT_JSON`
refuse what JSON has no value for (``NaN``, ``Infinity``, and numbers out of a float's range, which Python would read
as ``inf``) and values nested more than :data:`MAX_JSON_DEPTH` levels deep, and :func:`json_compatible` checks a
Python literal the same way.

Typing text values by the tool's schema follows SGLang's ``qwen3_coder`` and ``glm4_moe`` detectors (Apache-2.0;
see ``NOTICE`` in this directory); the code is written for platoon.
"""

from __future__ import annotations

import ast
import json
import math
import re
from collections.abc import Iterable, Mapping
from typing import Any

_INTEGER = re.compile(r"[+-]?\d+")
_NUMBER = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_JSON_TYPES = ((bool, "boolean"), (int, "integer"), (float, "number"), (str, "string"), (list, "array"))


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"the number {text} is out of range")
    return value


def _no_constant(name: str) -> Any:
    raise ValueError(f"{name} is not a JSON value")


#: How deep a JSON value may nest arrays and objects. Python's json module decodes about 10,000 levels on 3.12 and
#: 3.13, and on 3.14 as many as the thread's stack holds (some 16,000 in a thread on Linux, millions in a main thread
#: whose stack is unlimited); a limit below all of these makes a response parse the same on every Python and thread.
MAX_JSON_DEPTH = 1000


def _deeper_than(value: Any, limit: int) -> bool:
    """Whether ``value`` nests arrays and objects more than ``limit`` levels deep, read level by level (no
    recursion)."""

    level = [value] if isinstance(value, (dict, list)) else []
    for _ in range(limit):
        level = [
            child
            for container in level
            for child in (container.values() if isinstance(container, dict) else container)
            if isinstance(child, (dict, list))
        ]
        if not level:
            return False
    return bool(level)


class _Decoder(json.JSONDecoder):
    """A JSON decoder that refuses a value nested more than :data:`MAX_JSON_DEPTH` levels deep."""

    def raw_decode(self, s: str, idx: int = 0) -> tuple[Any, int]:
        value, end = super().raw_decode(s, idx)
        # Only a value with that many brackets can be that deep: most never need the walk.
        if s.count("[", idx, end) + s.count("{", idx, end) > MAX_JSON_DEPTH and _deeper_than(value, MAX_JSON_DEPTH):
            raise ValueError(f"the JSON nests more than {MAX_JSON_DEPTH} levels deep")
        return value, end


# JSON decoders that refuse NaN, Infinity, numbers out of a float's range and values nested more than MAX_JSON_DEPTH
# levels deep; LENIENT_JSON also takes raw control characters (newlines) inside strings, which strict JSON forbids.
STRICT_JSON = _Decoder(parse_float=_finite_float, parse_constant=_no_constant)
LENIENT_JSON = _Decoder(strict=False, parse_float=_finite_float, parse_constant=_no_constant)


class ToolSchemas:
    """The parameter schemas of the request's tools, by tool name.

    ``tools`` are OpenAI tool definitions (``{"type": "function", "function": {...}}``) or flat function
    definitions (``{"name": ..., "parameters": ...}``); entries without a name are skipped.
    """

    def __init__(self, tools: Iterable[Any] | None = None) -> None:
        self._parameters: dict[str, Mapping[str, Any]] = {}
        for tool in tools or ():
            function = _field(tool, "function") or tool
            name = _field(function, "name")
            if isinstance(name, str) and name:
                parameters = _field(function, "parameters")
                self._parameters[name] = parameters if isinstance(parameters, Mapping) else {}

    @classmethod
    def of(cls, tools: ToolSchemas | Iterable[Any] | None) -> ToolSchemas:
        return tools if isinstance(tools, ToolSchemas) else cls(tools)

    def __bool__(self) -> bool:
        return bool(self._parameters)

    def __contains__(self, name: object) -> bool:
        return name in self._parameters

    @property
    def names(self) -> list[str]:
        return list(self._parameters)

    def properties(self, name: str) -> Mapping[str, Any]:
        """The ``properties`` of tool ``name``'s parameters; empty for an unknown tool."""

        properties = self._parameters.get(name, {}).get("properties")
        return properties if isinstance(properties, Mapping) else {}

    def declares(self, name: str, key: str) -> bool | None:
        """Whether tool ``name`` declares the parameter ``key``; None when the tool is unknown."""

        if name not in self._parameters:
            return None
        return key in self.properties(name)

    def coerce(self, name: str, values: Mapping[str, str]) -> dict[str, Any]:
        """``values`` (argument texts) typed by tool ``name``'s parameter schemas; see :func:`coerce_value`."""

        properties = self.properties(name)
        return {key: coerce_value(value, properties.get(key)) for key, value in values.items()}


def coerce_value(text: str, schema: Any) -> Any:
    """``text`` as the JSON type ``schema`` allows, else ``text`` itself.

    Types are tried in a fixed order, and the first the text reads as wins: ``null`` (``null`` or ``None``),
    ``boolean`` (``true`` or ``false`` in any case), ``integer``, ``number`` (an integer when it has no fraction or
    exponent, as JSON writes one), then ``object`` and ``array`` (JSON, else a Python literal).  Surrounding
    whitespace is ignored for these; a value that stays text keeps its whitespace.  A number JSON cannot hold (out of
    a float's range) or an integer with more digits than Python converts stays text, and so does a literal that holds
    what JSON cannot (a set, bytes).
    """

    types = schema_types(schema)
    if not types or types == {"string"}:
        return text
    stripped = text.strip()
    if "null" in types and stripped in ("null", "None"):
        return None
    if "boolean" in types and stripped.lower() in ("true", "false"):
        return stripped.lower() == "true"
    if types & {"integer", "number"}:
        number = _number(stripped, integer_only="number" not in types)
        if number is not None:
            return number
    if types & {"object", "array"}:
        value = _structured(stripped)
        if (isinstance(value, dict) and "object" in types) or (isinstance(value, list) and "array" in types):
            return value
    return text


def schema_types(schema: Any) -> set[str]:
    """The JSON types a parameter schema allows: its ``type`` (one or a list) and those of its ``anyOf``,
    ``oneOf`` and ``allOf`` members; without any, the types of its ``enum`` or ``const`` values, or ``object`` or
    ``array`` from ``properties`` or ``items``.  Empty when the schema says nothing about types."""

    if not isinstance(schema, Mapping):
        return set()
    types: set[str] = set()
    declared = schema.get("type")
    if isinstance(declared, str):
        types.add(declared)
    elif isinstance(declared, list):
        types.update(item for item in declared if isinstance(item, str))
    for key in ("anyOf", "oneOf", "allOf"):
        members = schema.get(key)
        if isinstance(members, list):
            for member in members:
                types |= schema_types(member)
    if types:
        return types
    if isinstance(schema.get("enum"), list):
        return {_json_type(value) for value in schema["enum"]}
    if "const" in schema:
        return {_json_type(schema["const"])}
    if "properties" in schema:
        return {"object"}
    if "items" in schema:
        return {"array"}
    return set()


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    for python_type, name in _JSON_TYPES:
        if isinstance(value, python_type):
            return name
    return "object"


def json_compatible(value: Any) -> Any:
    """``value``, a Python literal, as a JSON value (tuples become lists).

    Raises ``ValueError`` when JSON cannot hold it: a set, bytes, a complex number, a float out of range (``inf``,
    ``nan``), or an object key that is not a string.
    """

    if value is None or isinstance(value, (str, int)):  # bool is an int
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        raise ValueError(f"{value} is not a JSON number")
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("an object key is not a string")
        return {key: json_compatible(item) for key, item in value.items()}
    raise ValueError(f"a {type(value).__name__} is not a JSON value")


def _number(text: str, *, integer_only: bool) -> int | float | None:
    """``text`` as an integer when it has no fraction or exponent, else (unless ``integer_only``) as a float; None
    when it is neither, has more digits than ``int`` converts, or is out of a float's range."""

    try:
        if _INTEGER.fullmatch(text):
            return int(text)
        if not integer_only and _NUMBER.fullmatch(text):
            return _finite_float(text)
    except ValueError:
        pass
    return None


def _structured(text: str) -> Any:
    """A JSON object or array, else a Python literal one (``{'a': 1}``) that JSON can hold; None when the text is
    neither."""

    if text[:1] not in ("{", "["):
        return None
    try:
        return STRICT_JSON.decode(text)
    except (ValueError, RecursionError):
        pass
    try:
        return json_compatible(ast.literal_eval(text))
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return None


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)
