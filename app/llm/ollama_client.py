"""Ollama backend (V1 inference engine).

Uses the /api/chat endpoint with ``stream=false`` and ``format=json`` friendly
options.  Retries transient network failures with exponential backoff.
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


class OllamaClient(LLMClient):
    name = "ollama"

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout_s: int | None = None,
        num_ctx: int | None = None,
        temperature: float | None = None,
    ) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.llm_model
        self.timeout_s = timeout_s or settings.llm_timeout_s
        self.num_ctx = num_ctx or settings.llm_num_ctx
        self.temperature = settings.llm_temperature if temperature is None else temperature
        # How long ollama keeps the weights in memory after a request. The
        # default 5m expires between messages and the reload dominates reply
        # time on a CPU-only host, so hold them much longer.
        self.keep_alive = getattr(settings, "ollama_keep_alive", "2h")
        self._client: httpx.AsyncClient | None = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout_s, connect=15.0),
            )
        return self._client

    async def chat(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.as_dict() for m in messages],
            "stream": False,
            # Keep the weights resident between messages. Ollama's default
            # (5 min) expires during normal gaps in a conversation, and the
            # reload costs 18-60s on this CPU-only box - which is most of
            # what "the bot is slow" actually was.
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature if temperature is None else temperature,
                "num_ctx": self.num_ctx,
            },
        }
        if json_mode:
            # Ollama constrains generation to valid JSON at the sampler, so a
            # small local model cannot wander off into prose or a markdown
            # fence. Asking a 3B model nicely in the prompt does not work;
            # this does, and it is what makes the local fallback usable for
            # routing and planning at all.
            payload["format"] = "json"

        last_error: Exception | None = None
        for attempt in range(1, 4):
            started = time.perf_counter()
            try:
                response = await self._http().post("/api/chat", json=payload)
                if response.status_code >= 500:
                    raise LLMError(f"ollama {response.status_code}: {response.text[:200]}")
                response.raise_for_status()
                data = response.json()
                content = (data.get("message") or {}).get("content", "")
                if not content:
                    raise LLMError("empty response from model")
                duration_ms = (time.perf_counter() - started) * 1000
                log.info(
                    "llm_response",
                    extra={
                        "tool": "llm",
                        "duration": round(duration_ms, 1),
                        "model": self.model,
                        "chars": len(content),
                    },
                )
                return LLMResponse(
                    content=content, raw=data, model=self.model, duration_ms=duration_ms
                )
            except (httpx.HTTPError, LLMError, ValueError) as exc:
                last_error = exc
                log.warning(
                    "llm_attempt_failed",
                    extra={"tool": "llm", "attempt": attempt, "error": str(exc)[:300]},
                )
                if attempt < 3:
                    await asyncio.sleep(2 ** attempt)

        raise LLMError(f"ollama chat failed after 3 attempts: {last_error}")

    async def health(self) -> dict[str, Any]:
        try:
            response = await self._http().get("/api/tags", timeout=10.0)
            response.raise_for_status()
            models = [m.get("name") for m in response.json().get("models", [])]
            return {
                "ok": True,
                "provider": self.name,
                "model": self.model,
                "model_available": self.model in models,
                "models": models[:20],
            }
        except Exception as exc:  # noqa: BLE001 - health must never raise
            return {"ok": False, "provider": self.name, "model": self.model, "error": str(exc)[:200]}

    async def ensure_model(self) -> bool:
        """Pull the configured model if it is not present yet."""
        info = await self.health()
        if info.get("model_available"):
            return True
        log.info("llm_pull_start", extra={"tool": "llm", "model": self.model})
        try:
            async with self._http().stream(
                "POST", "/api/pull", json={"model": self.model, "stream": True}, timeout=None
            ) as response:
                response.raise_for_status()
                async for _ in response.aiter_lines():
                    pass
            log.info("llm_pull_done", extra={"tool": "llm", "model": self.model})
            return True
        except Exception as exc:  # noqa: BLE001
            log.error("llm_pull_failed", extra={"tool": "llm", "error": str(exc)[:300]})
            return False

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
