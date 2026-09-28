"""LLM abstraction.

The rest of the codebase only knows :class:`LLMClient`.  Swapping Ollama for
llama.cpp / an API provider later means adding one file here, nothing else.
"""

from __future__ import annotations

import inspect
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Message:
    role: str  # system | user | assistant
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(slots=True)
class LLMResponse:
    content: str
    raw: dict[str, Any] = field(default_factory=dict)
    model: str = ""
    duration_ms: float = 0.0


class LLMError(Exception):
    """Any failure while talking to the model backend."""


class LLMClient(ABC):
    """Minimal interface every backend must implement."""

    name: str = "base"

    @abstractmethod
    async def chat(self, messages: list[Message], *, temperature: float | None = None) -> LLMResponse:
        ...

    @abstractmethod
    async def health(self) -> dict[str, Any]:
        ...

    async def close(self) -> None:  # pragma: no cover - default no-op
        return None

    # ------------------------------------------------------------------ #
    async def chat_json(
        self, messages: list[Message], *, temperature: float | None = None
    ) -> dict[str, Any]:
        """Chat and parse a JSON object out of the reply.

        Backends that can CONSTRAIN generation to JSON (ollama's format=json,
        the OpenAI response_format) are asked to - a 3B local model told
        "reply with one JSON object" in the prompt will still return prose
        often enough to matter, and that failure surfaces as the router
        falling back to "spawn a task" for a plain question.

        Prose is still handled: local models wrap JSON in ```json fences, so
        the first balanced object is extracted rather than trusting
        ``json.loads`` blindly.
        """
        response = await self.chat(
            messages, temperature=temperature, **_json_kwargs(self.chat)
        )
        data = extract_json(response.content)
        if data is None:
            raise LLMError(f"model did not return JSON: {response.content[:400]!r}")
        return data


def _json_kwargs(chat_callable: Any) -> dict[str, Any]:
    """``{"json_mode": True}`` if this backend accepts it, else ``{}``.

    Checked by signature rather than by catching TypeError: a TypeError
    raised INSIDE a backend's chat() would otherwise be silently retried as
    "this backend is old", hiding a real bug.

    Cached on the underlying function, because this runs on every single
    chat_json() call and ``inspect.signature`` is not cheap - the answer
    depends only on the method's declaration, which never changes at runtime.
    """
    func = getattr(chat_callable, "__func__", chat_callable)
    cached = _JSON_KWARGS_CACHE.get(func)
    if cached is not None:
        return cached

    try:
        parameters = inspect.signature(chat_callable).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        result: dict[str, Any] = {}
    else:
        supported = "json_mode" in parameters or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
        )
        result = {"json_mode": True} if supported else {}

    try:
        _JSON_KWARGS_CACHE[func] = result
    except TypeError:  # pragma: no cover - unhashable callable
        pass
    return result


# Keyed on the function object, so it is bounded by the number of backend
# classes (a handful) rather than growing per client instance.
_JSON_KWARGS_CACHE: dict[Any, dict[str, Any]] = {}


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of the first JSON object in ``text``."""
    if not text:
        return None

    candidates: list[str] = []
    stripped = text.strip()
    candidates.append(stripped)
    candidates.extend(match.strip() for match in _FENCE_RE.findall(text))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass

    # Balanced-brace scan (handles strings/escapes) over the raw text.
    for source in [text, *_FENCE_RE.findall(text)]:
        depth = 0
        start = -1
        in_string = False
        escape = False
        for index, char in enumerate(source):
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                if depth == 0:
                    start = index
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0 and start >= 0:
                    chunk = source[start : index + 1]
                    try:
                        parsed = json.loads(chunk)
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        start = -1
    return None
