"""Telegram tools + a thin verified API client.

Sending is *verified*: the Bot API response must contain a message_id before we
report success (master prompt sections 20 and 35).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx

from app.config import get_settings
from app.logging_conf import get_logger
from app.security import Permission, UnsafePath, rel_path, safe_path
from app.tools.base import (
    Arg,
    AuthError,
    InvalidInput,
    PermanentToolError,
    RateLimited,
    TemporaryToolError,
    ToolContext,
)
from app.tools.registry import tool

log = get_logger(__name__)

# Telegram bot API upload cap for documents sent by bots.
TELEGRAM_MAX_UPLOAD = 50 * 1024 * 1024


class TelegramAPI:
    """Minimal Bot API client used by tools and the notifier."""

    def __init__(self, token: str | None = None, base: str | None = None) -> None:
        settings = get_settings()
        self.token = token or settings.telegram_bot_token
        self.base = (base or settings.telegram_api_base).rstrip("/")
        self._client: httpx.AsyncClient | None = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=15.0))
        return self._client

    def _url(self, method: str) -> str:
        return f"{self.base}/bot{self.token}/{method}"

    async def call(
        self,
        method: str,
        data: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.token:
            raise AuthError("TELEGRAM_BOT_TOKEN is not configured")
        try:
            response = await self._http().post(self._url(method), data=data or {}, files=files)
        except httpx.HTTPError as exc:
            raise TemporaryToolError(f"telegram network error: {exc}") from exc

        if response.status_code == 429:
            retry_after = 5
            try:
                retry_after = int(response.json().get("parameters", {}).get("retry_after", 5))
            except Exception:  # noqa: BLE001
                pass
            await asyncio.sleep(min(retry_after, 30))
            raise RateLimited(f"telegram rate limited, retry after {retry_after}s")
        if response.status_code in {401, 403}:
            raise AuthError(f"telegram auth error: {response.text[:200]}")
        if response.status_code >= 500:
            raise TemporaryToolError(f"telegram server error {response.status_code}")

        try:
            payload = response.json()
        except ValueError as exc:
            raise TemporaryToolError("telegram returned a non-JSON response") from exc

        if not payload.get("ok"):
            raise PermanentToolError(f"telegram API error: {str(payload.get('description'))[:200]}")
        return payload.get("result") or {}

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()


_api: TelegramAPI | None = None


def get_api() -> TelegramAPI:
    global _api
    if _api is None:
        _api = TelegramAPI()
    return _api


def set_api(api: TelegramAPI | None) -> None:
    """Injection point for tests."""
    global _api
    _api = api


async def close_api() -> None:
    global _api
    if _api is not None:
        await _api.close()
    _api = None


def _target_chat(ctx: ToolContext | None, chat_id: int | None) -> int:
    if chat_id:
        return int(chat_id)
    if ctx is not None and ctx.chat_id:
        return int(ctx.chat_id)
    owner = get_settings().owner_chat_id
    if owner:
        return int(owner)
    raise InvalidInput("no chat_id available (set TELEGRAM_OWNER_CHAT_ID)")


def _verify_message(data: dict[str, Any]) -> bool:
    return bool(data.get("message_id"))


@tool(
    "telegram_send_message",
    description="Send a text message to the owner's Telegram chat. Verified via message_id.",
    permission=Permission.WRITE,
    args={
        "text": Arg("string", True, "Message text (max 4096 chars)"),
        "chat_id": Arg("integer", False, "Override chat id; defaults to the task's chat"),
    },
    timeout_s=90,
    max_retries=2,
    side_effect=True,
    verify=_verify_message,
)
async def telegram_send_message(
    text: str, chat_id: int | None = None, ctx: ToolContext | None = None
) -> dict[str, Any]:
    if not text.strip():
        raise InvalidInput("text must not be empty")
    target = _target_chat(ctx, chat_id)
    result = await get_api().call(
        "sendMessage",
        {"chat_id": target, "text": text[:4096], "disable_web_page_preview": "true"},
    )
    log.info(
        "telegram_sent",
        extra={"tool": "telegram_send_message", "task_id": ctx.task_id if ctx else None,
               "message_id": result.get("message_id")},
    )
    return {"sent": True, "chat_id": target, "message_id": result.get("message_id")}


@tool(
    "telegram_send_file",
    description="Send a workspace file to the owner's Telegram chat. Verified via message_id.",
    permission=Permission.WRITE,
    args={
        "path": Arg("string", True, "Workspace-relative file to send"),
        "caption": Arg("string", False, "Optional caption", default=""),
        "chat_id": Arg("integer", False, "Override chat id; defaults to the task's chat"),
    },
    timeout_s=600,
    max_retries=2,
    side_effect=True,
    verify=_verify_message,
)
async def telegram_send_file(
    path: str, caption: str = "", chat_id: int | None = None, ctx: ToolContext | None = None
) -> dict[str, Any]:
    try:
        target_file: Path = safe_path(path, must_exist=True)
    except UnsafePath as exc:
        raise InvalidInput(str(exc)) from exc
    if target_file.is_dir():
        raise InvalidInput("cannot send a directory")

    size = target_file.stat().st_size
    if size == 0:
        raise InvalidInput("refusing to send an empty file")
    if size > TELEGRAM_MAX_UPLOAD:
        raise PermanentToolError(
            f"file is {size / 1048576:.1f} MB; Telegram bots cannot upload more than 50 MB"
        )

    target = _target_chat(ctx, chat_id)
    with target_file.open("rb") as handle:
        result = await get_api().call(
            "sendDocument",
            {"chat_id": target, "caption": caption[:1024]},
            {"document": (target_file.name, handle)},
        )

    document = result.get("document") or {}
    log.info(
        "telegram_file_sent",
        extra={
            "tool": "telegram_send_file",
            "task_id": ctx.task_id if ctx else None,
            "path": rel_path(target_file),
            "message_id": result.get("message_id"),
        },
    )
    return {
        "sent": True,
        "chat_id": target,
        "message_id": result.get("message_id"),
        "file_id": document.get("file_id"),
        "file_name": document.get("file_name", target_file.name),
        "size_bytes": size,
        "path": rel_path(target_file),
    }
