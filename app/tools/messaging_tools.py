"""WhatsApp and Microsoft Teams tools.

Both talk to their Go bridge over HTTP.  WhatsApp uses the WhatsApp Web
multi-device protocol (whatsmeow) with the owner's own linked device; Teams
uses the official Microsoft Graph API with application permissions.

Sends are *verified*: the bridge must return a message id before the tool
reports success, exactly like the Telegram tools.
"""

from __future__ import annotations

from typing import Any

from app.config import get_settings
from app.integrations import BridgeError, teams_bridge, whatsapp_bridge
from app.logging_conf import get_logger
from app.security import Permission, UnsafePath, rel_path, safe_path
from app.tools.base import (
    Arg,
    InvalidInput,
    PermanentToolError,
    TemporaryToolError,
    ToolContext,
)
from app.tools.registry import tool

log = get_logger(__name__)

_settings = get_settings()
WHATSAPP_ENABLED = _settings.whatsapp_enabled
TEAMS_ENABLED = _settings.teams_enabled


def _raise(exc: BridgeError) -> None:
    """Map a bridge failure onto the agent's retry taxonomy."""
    if exc.temporary:
        raise TemporaryToolError(str(exc)) from exc
    if exc.status in {400, 422}:
        raise InvalidInput(str(exc)) from exc
    raise PermanentToolError(str(exc)) from exc


def _verify_sent(data: dict[str, Any]) -> bool:
    return bool(data.get("message_id"))


# --------------------------------------------------------------------------- #
# WhatsApp
# --------------------------------------------------------------------------- #
@tool(
    "whatsapp_send_message",
    description=(
        "Send a WhatsApp text message. 'to' is a phone number with country code "
        "(e.g. 8801712345678) or a full JID. Verified by message id."
    ),
    permission=Permission.WRITE,
    args={
        "to": Arg("string", True, "Phone number with country code, or a WhatsApp JID"),
        "text": Arg("string", True, "Message text"),
    },
    timeout_s=120,
    max_retries=2,
    side_effect=True,
    verify=_verify_sent,
    enabled=WHATSAPP_ENABLED,
)
async def whatsapp_send_message(
    to: str, text: str, ctx: ToolContext | None = None
) -> dict[str, Any]:
    if not text.strip():
        raise InvalidInput("text must not be empty")
    try:
        result = await whatsapp_bridge().send_text(to, text[:4000])
    except BridgeError as exc:
        _raise(exc)
    log.info(
        "whatsapp_sent",
        extra={"tool": "whatsapp_send_message", "task_id": ctx.task_id if ctx else None,
               "message_id": result.get("message_id")},
    )
    return {"sent": True, "to": to, "message_id": result.get("message_id")}


@tool(
    "whatsapp_send_file",
    description="Send a workspace file over WhatsApp. Verified by message id.",
    permission=Permission.WRITE,
    args={
        "to": Arg("string", True, "Phone number with country code, or a WhatsApp JID"),
        "path": Arg("string", True, "Workspace-relative file to send"),
        "caption": Arg("string", False, "Optional caption", default=""),
    },
    timeout_s=600,
    max_retries=2,
    side_effect=True,
    verify=_verify_sent,
    enabled=WHATSAPP_ENABLED,
)
async def whatsapp_send_file(
    to: str, path: str, caption: str = "", ctx: ToolContext | None = None
) -> dict[str, Any]:
    try:
        target = safe_path(path, must_exist=True)
    except UnsafePath as exc:
        raise InvalidInput(str(exc)) from exc
    if target.is_dir():
        raise InvalidInput("cannot send a directory")
    size = target.stat().st_size
    if size == 0:
        raise InvalidInput("refusing to send an empty file")

    try:
        result = await whatsapp_bridge().send_file(to, rel_path(target), caption)
    except BridgeError as exc:
        _raise(exc)
    return {
        "sent": True,
        "to": to,
        "message_id": result.get("message_id"),
        "path": rel_path(target),
        "size_bytes": size,
    }


@tool(
    "whatsapp_read_messages",
    description="Read recent WhatsApp messages received by the agent.",
    permission=Permission.READ,
    args={"limit": Arg("integer", False, "How many messages", default=20)},
    timeout_s=60,
    max_retries=1,
    enabled=WHATSAPP_ENABLED,
)
async def whatsapp_read_messages(limit: int = 20) -> dict[str, Any]:
    try:
        result = await whatsapp_bridge().messages(limit=max(1, min(limit, 100)))
    except BridgeError as exc:
        _raise(exc)
    return {"count": result.get("count", 0), "messages": result.get("messages", [])}


@tool(
    "whatsapp_status",
    description="Check whether the WhatsApp account is linked and connected.",
    permission=Permission.READ,
    args={},
    timeout_s=30,
    max_retries=1,
    enabled=WHATSAPP_ENABLED,
)
async def whatsapp_status() -> dict[str, Any]:
    try:
        return await whatsapp_bridge().status()
    except BridgeError as exc:
        _raise(exc)


# --------------------------------------------------------------------------- #
# Microsoft Teams
# --------------------------------------------------------------------------- #
@tool(
    "teams_send_message",
    description=(
        "Send a message to a Microsoft Teams chat or channel. 'chat' is a chat id, "
        "or 'teamId/channelId' for a channel. Verified by message id."
    ),
    permission=Permission.WRITE,
    args={
        "text": Arg("string", True, "Message text"),
        "chat": Arg("string", False, "Chat id or teamId/channelId; blank = default chat",
                    default=""),
    },
    timeout_s=120,
    max_retries=2,
    side_effect=True,
    verify=_verify_sent,
    enabled=TEAMS_ENABLED,
)
async def teams_send_message(
    text: str, chat: str = "", ctx: ToolContext | None = None
) -> dict[str, Any]:
    if not text.strip():
        raise InvalidInput("text must not be empty")
    try:
        result = await teams_bridge().send_text(chat, text[:8000])
    except BridgeError as exc:
        _raise(exc)
    log.info(
        "teams_sent",
        extra={"tool": "teams_send_message", "task_id": ctx.task_id if ctx else None,
               "message_id": result.get("message_id")},
    )
    return {"sent": True, "chat": result.get("chat", chat), "message_id": result.get("message_id")}


@tool(
    "teams_read_messages",
    description="Read recent messages from a Microsoft Teams chat or channel.",
    permission=Permission.READ,
    args={
        "chat": Arg("string", False, "Chat id or teamId/channelId; blank = default", default=""),
        "limit": Arg("integer", False, "How many messages", default=20),
    },
    timeout_s=90,
    max_retries=1,
    enabled=TEAMS_ENABLED,
)
async def teams_read_messages(chat: str = "", limit: int = 20) -> dict[str, Any]:
    try:
        result = await teams_bridge().messages(chat=chat, limit=max(1, min(limit, 50)))
    except BridgeError as exc:
        _raise(exc)
    return {
        "count": result.get("count", 0),
        "chat": result.get("chat", chat),
        "messages": result.get("messages", []),
    }


@tool(
    "teams_download_file",
    description="Download a Teams-hosted file (Graph download URL) into the workspace.",
    permission=Permission.WRITE,
    args={
        "url": Arg("string", True, "Microsoft Graph download URL from a message attachment"),
        "path": Arg("string", False, "Destination workspace path", default=""),
    },
    timeout_s=600,
    max_retries=2,
    verify=lambda data: bool(data.get("size_bytes", 0) > 0),
    enabled=TEAMS_ENABLED,
)
async def teams_download_file(url: str, path: str = "") -> dict[str, Any]:
    if not path:
        name = url.split("?")[0].rstrip("/").split("/")[-1] or "teams_file.bin"
        path = f"downloads/teams/{name}"
    try:
        safe_path(path)
    except UnsafePath as exc:
        raise InvalidInput(str(exc)) from exc
    try:
        result = await teams_bridge().download(url, path)
    except BridgeError as exc:
        _raise(exc)
    return {"downloaded": True, "path": result.get("path", path),
            "size_bytes": result.get("size_bytes", 0)}


@tool(
    "teams_status",
    description="Check whether the Teams integration is authenticated.",
    permission=Permission.READ,
    args={},
    timeout_s=30,
    max_retries=1,
    enabled=TEAMS_ENABLED,
)
async def teams_status() -> dict[str, Any]:
    try:
        return await teams_bridge().status()
    except BridgeError as exc:
        _raise(exc)
