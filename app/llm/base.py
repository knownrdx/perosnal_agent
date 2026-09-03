"""LLM abstraction.

The rest of the codebase only knows :class:`LLMClient`.  Swapping Ollama for
llama.cpp / an API provider later means adding one file here, nothing else.
"""

from __future__ import annotations

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

        Local models often wrap JSON in prose or ```json fences, so we extract
        the first balanced object rather than trusting ``json.loads`` blindly.
        """
        response = await self.chat(messages, temperature=temperature)
        data = extract_json(response.content)
        if data is None:
            raise LLMError(f"model did not return JSON: {response.content[:400]!r}")
        return data


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
