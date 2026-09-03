"""Deterministic offline LLM stub.

Used by the test-suite and by ``LLM_PROVIDER=echo`` so the whole agent loop can
be exercised end to end without a model server.  It understands a couple of
simple intents so the primary success test (§30) is verifiable offline.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from app.llm.base import LLMClient, LLMResponse, Message


class EchoClient(LLMClient):
    name = "echo"

    def __init__(self, script: list[dict[str, Any]] | None = None) -> None:
        # ``script`` lets tests queue exact decisions, popped one per call.
        self.script = list(script or [])
        self.calls: list[list[Message]] = []
        self.handler: Callable[[list[Message]], dict[str, Any]] | None = None

    async def chat(
        self, messages: list[Message], *, temperature: float | None = None
    ) -> LLMResponse:
        self.calls.append(messages)

        # Planning and self-verification are auxiliary calls, not decisions.
        # They must not consume a scripted decision, or every test would have
        # to hand-write plans and verdicts it does not care about.
        auxiliary = self._auxiliary(messages)
        if auxiliary is not None:
            return LLMResponse(content=json.dumps(auxiliary), model=self.name)

        if self.script:
            decision = self.script.pop(0)
        elif self.handler is not None:
            decision = self.handler(messages)
        else:
            decision = self._decide(messages)
        return LLMResponse(content=json.dumps(decision), model=self.name)

    async def health(self) -> dict[str, Any]:
        return {"ok": True, "provider": self.name, "model": self.name, "model_available": True}

    # ------------------------------------------------------------------ #
    @staticmethod
    def _auxiliary(messages: list[Message]) -> dict[str, Any] | None:
        """Answer planner/verifier/summariser prompts in character.

        Returns None for ordinary decision calls so the script still drives the
        agent loop itself.
        """
        system = "\n".join(m.content for m in messages if m.role == "system")

        if "You plan work for an autonomous agent" in system:
            # Offline default: no plan, so the loop behaves exactly as before.
            return {"complexity": "simple", "steps": [], "success_criteria": "", "risks": []}

        if "You are the final check" in system:
            # Trust the agent offline; dedicated tests drive rejection directly.
            return {"verified": True, "reason": "offline verifier", "missing": []}

        if "You compress an agent's execution history" in system:
            return {"summary": "offline summary"}

        return None

    def _decide(self, messages: list[Message]) -> dict[str, Any]:
        text = "\n".join(m.content for m in messages if m.role == "user")
        lowered = text.lower()

        # Already produced an observation for a write? then finish.
        if "tool_result" in lowered and "file_write" in lowered:
            match = re.search(r'"path":\s*"([^"]+)"', text)
            path = match.group(1) if match else "output/echo.txt"
            return {
                "thought": "file written, reporting result",
                "action": "final",
                "final_answer": f"Done. File available at {path}",
                "output_files": [path],
            }

        if "write" in lowered and ("file" in lowered or ".txt" in lowered):
            return {
                "thought": "user asked for a file",
                "action": "tool",
                "tool": "file_write",
                "args": {"path": "output/echo.txt", "content": "hello from the echo model"},
            }

        return {
            "thought": "no tool needed",
            "action": "final",
            "final_answer": "Echo model: no tool needed for this request.",
        }
