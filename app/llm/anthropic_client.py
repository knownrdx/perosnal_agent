"""Anthropic (Claude) backend.

Uses the Messages API.  Claude requires the system prompt as a top-level
``system`` field rather than a message with role=system, so the client splits
it out automatically.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from app.config import get_settings
from app.llm.base import LLMClient, LLMError, LLMResponse, Message
from app.logging_conf import get_logger

log = get_logger(__name__)

API_VERSION = "2023-06-01"


def _normalise_base(url: str) -> str:
    """Accept both ``https://host`` and ``https://host/v1`` as a base URL.

    Gateways are usually advertised with the ``/v1`` already attached (that is
    what an OpenAI-style config looks like), while the official Anthropic base
    is bare. Requests always append ``/v1/messages``, so a base that already
    ends in ``/v1`` would produce ``/v1/v1/messages`` and 404. Strip it once
    here rather than making every caller remember the difference.
    """
    trimmed = (url or "").strip().rstrip("/")
    if trimmed.endswith("/v1"):
        trimmed = trimmed[: -len("/v1")]
    return trimmed


class AnthropicClient(LLMClient):
    name = "anthropic"
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_s: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.anthropic_api_key
        self.base_url = _normalise_base(base_url or settings.anthropic_base_url)
        self.model = model or settings.anthropic_model
        self.timeout_s = timeout_s or settings.llm_timeout_s
        self.temperature = settings.llm_temperature if temperature is None else temperature
        self.max_tokens = max_tokens or settings.llm_max_tokens
        self._client: httpx.AsyncClient | None = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout_s, connect=15.0),
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": API_VERSION,
                    "Content-Type": "application/json",
                },
            )
        return self._client

    @staticmethod
    def _split_system(messages: list[Message]) -> tuple[str, list[dict[str, str]]]:
        """Claude wants system separately and alternating user/assistant turns."""
        system_parts = [m.content for m in messages if m.role == "system"]
        turns: list[dict[str, str]] = []
        for message in messages:
            if message.role == "system":
                continue
            role = "assistant" if message.role == "assistant" else "user"
            if turns and turns[-1]["role"] == role:
                turns[-1]["content"] += "\n\n" + message.content
            else:
                turns.append({"role": role, "content": message.content})
        if not turns:
            turns = [{"role": "user", "content": "continue"}]
        if turns[0]["role"] != "user":
            turns.insert(0, {"role": "user", "content": "continue"})
        return "\n\n".join(system_parts), turns

    async def chat(
        self, messages: list[Message], *, temperature: float | None = None
    ) -> LLMResponse:
        if not self.api_key:
            raise LLMError("anthropic: API key is not configured")

        system, turns = self._split_system(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": turns,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature if temperature is None else temperature,
        }
        if system:
            payload["system"] = system

        last_error: Exception | None = None
        for attempt in range(1, 4):
            started = time.perf_counter()
            try:
                response = await self._http().post("/v1/messages", json=payload)

                if response.status_code == 429:
                    raise LLMError("rate limited")
                if response.status_code in {401, 403}:
                    raise LLMError("anthropic: authentication failed")
                if response.status_code == 529:
                    raise LLMError("anthropic: service overloaded")
                if response.status_code >= 500:
                    raise LLMError(f"anthropic: server error {response.status_code}")
                if response.status_code >= 400:
                    raise LLMError(f"anthropic: {response.status_code} {_detail(response)}")

                data = response.json()
                blocks = data.get("content") or []
                content = "".join(
                    block.get("text", "") for block in blocks if block.get("type") == "text"
                )
                if not content.strip():
                    raise LLMError("empty response content")

                duration_ms = (time.perf_counter() - started) * 1000
                usage = data.get("usage") or {}
                log.info(
                    "llm_response",
                    extra={
                        "tool": "llm",
                        "provider": self.name,
                        "model": self.model,
                        "duration": round(duration_ms, 1),
                        "prompt_tokens": usage.get("input_tokens"),
                        "completion_tokens": usage.get("output_tokens"),
                    },
                )
                return LLMResponse(
                    content=content, raw=data, model=self.model, duration_ms=duration_ms
                )

            except (httpx.HTTPError, LLMError, ValueError) as exc:
                last_error = exc
                if "authentication failed" in str(exc):
                    raise LLMError(str(exc)) from exc
                log.warning(
                    "llm_attempt_failed",
                    extra={"tool": "llm", "provider": self.name, "attempt": attempt,
                           "error": str(exc)[:300]},
                )
                if attempt < 3:
                    await asyncio.sleep(2 ** attempt)

        raise LLMError(f"anthropic chat failed after 3 attempts: {last_error}")

    async def health(self) -> dict[str, Any]:
        if not self.api_key:
            return {"ok": False, "provider": self.name, "model": self.model,
                    "error": "API key not configured"}
        try:
            # /v1/models is the cheapest authenticated call.
            response = await self._http().get("/v1/models", timeout=15.0)
            if response.status_code in {401, 403}:
                return {"ok": False, "provider": self.name, "model": self.model,
                        "error": "invalid API key"}
            if response.status_code == 404:
                # Older gateways may not expose /v1/models; key still works.
                return {"ok": True, "provider": self.name, "model": self.model,
                        "model_available": True, "models": []}
            response.raise_for_status()
            models = [m.get("id") for m in (response.json().get("data") or [])]
            return {
                "ok": True,
                "provider": self.name,
                "model": self.model,
                "model_available": (self.model in models) if models else True,
                "models": models[:30],
            }
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "provider": self.name, "model": self.model,
                    "error": str(exc)[:200]}

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()


def _detail(response: httpx.Response) -> str:
    try:
        error = response.json().get("error") or {}
        return str(error.get("message", ""))[:200]
    except Exception:  # noqa: BLE001
        return response.text[:200]
