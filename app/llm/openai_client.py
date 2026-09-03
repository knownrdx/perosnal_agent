"""OpenAI-compatible backend.

Covers ChatGPT (api.openai.com) and every provider that speaks the same
/chat/completions schema: OpenRouter, Groq, DeepSeek, Together, xAI, Mistral,
vLLM, LM Studio, llama.cpp server.  One class, different base_url + key.
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

# Models that reject temperature / use max_completion_tokens instead.
_REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5")


class OpenAIClient(LLMClient):
    name = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_s: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        provider_label: str | None = None,
    ) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.openai_api_key
        self.base_url = (base_url or settings.openai_base_url).rstrip("/")
        self.model = model or settings.openai_model
        self.timeout_s = timeout_s or settings.llm_timeout_s
        self.temperature = settings.llm_temperature if temperature is None else temperature
        self.max_tokens = max_tokens or settings.llm_max_tokens
        if provider_label:
            self.name = provider_label
        self._client: httpx.AsyncClient | None = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout_s, connect=15.0),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    def _is_reasoning_model(self) -> bool:
        return self.model.lower().startswith(_REASONING_PREFIXES)

    def _payload(self, messages: list[Message], temperature: float | None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.as_dict() for m in messages],
            "stream": False,
        }
        if self._is_reasoning_model():
            payload["max_completion_tokens"] = self.max_tokens
        else:
            payload["temperature"] = self.temperature if temperature is None else temperature
            payload["max_tokens"] = self.max_tokens
        return payload

    async def chat(
        self, messages: list[Message], *, temperature: float | None = None
    ) -> LLMResponse:
        if not self.api_key:
            raise LLMError(f"{self.name}: API key is not configured")

        payload = self._payload(messages, temperature)
        last_error: Exception | None = None

        for attempt in range(1, 4):
            started = time.perf_counter()
            try:
                response = await self._http().post("/chat/completions", json=payload)

                if response.status_code == 429:
                    retry_after = float(response.headers.get("retry-after", 2 ** attempt))
                    raise LLMError(f"rate limited (retry after {retry_after:.0f}s)")
                if response.status_code in {401, 403}:
                    raise LLMError(f"{self.name}: authentication failed ({response.status_code})")
                if response.status_code >= 500:
                    raise LLMError(f"{self.name}: server error {response.status_code}")
                if response.status_code >= 400:
                    detail = _error_detail(response)
                    raise LLMError(f"{self.name}: {response.status_code} {detail}")

                data = response.json()
                choices = data.get("choices") or []
                if not choices:
                    raise LLMError("no choices returned")
                content = (choices[0].get("message") or {}).get("content") or ""
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
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens"),
                    },
                )
                return LLMResponse(
                    content=content, raw=data, model=self.model, duration_ms=duration_ms
                )

            except (httpx.HTTPError, LLMError, ValueError) as exc:
                last_error = exc
                message = str(exc)
                # Auth / invalid model are permanent: fail fast.
                if "authentication failed" in message or "invalid_request" in message:
                    raise LLMError(message) from exc
                log.warning(
                    "llm_attempt_failed",
                    extra={"tool": "llm", "provider": self.name, "attempt": attempt,
                           "error": message[:300]},
                )
                if attempt < 3:
                    await asyncio.sleep(2 ** attempt)

        raise LLMError(f"{self.name} chat failed after 3 attempts: {last_error}")

    async def health(self) -> dict[str, Any]:
        if not self.api_key:
            return {"ok": False, "provider": self.name, "model": self.model,
                    "error": "API key not configured"}
        try:
            response = await self._http().get("/models", timeout=15.0)
            if response.status_code in {401, 403}:
                # Some gateways (e.g. OmniRoute) gate /models behind a dashboard
                # key while /chat/completions stays open. Probe what we use.
                if await self._chat_reachable():
                    return {"ok": True, "provider": self.name, "model": self.model,
                            "model_available": True, "models": [],
                            "note": "/models requires a key; chat endpoint reachable"}
                return {"ok": False, "provider": self.name, "model": self.model,
                        "error": "invalid API key"}
            response.raise_for_status()
            models = [m.get("id") for m in (response.json().get("data") or [])]
            return {
                "ok": True,
                "provider": self.name,
                "model": self.model,
                "model_available": (self.model in models) if models else True,
                "models": models[:30],
            }
        except Exception as exc:  # noqa: BLE001 - health must never raise
            return {"ok": False, "provider": self.name, "model": self.model,
                    "error": str(exc)[:200]}

    async def _chat_reachable(self) -> bool:
        """Cheapest possible call against the endpoint we actually use."""
        try:
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
            }
            if self._is_reasoning_model():
                payload.pop("max_tokens")
                payload["max_completion_tokens"] = 1
            response = await self._http().post("/chat/completions", json=payload, timeout=25.0)
            return response.status_code < 400
        except Exception:  # noqa: BLE001
            return False

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()


def _error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
        error = payload.get("error")
        if isinstance(error, dict):
            return str(error.get("message", ""))[:200]
        return str(error or payload)[:200]
    except Exception:  # noqa: BLE001
        return response.text[:200]
