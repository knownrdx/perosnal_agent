"""Tools backed by the owner's Telegram account (userbot).

These do what a bot token cannot: read the owner's own chats, message any
user, and drive BotFather to manage the owner's other bots.
"""

from __future__ import annotations

from typing import Any

from app.integrations.telegram_user import UserbotError, get_userbot
from app.logging_conf import get_logger
from app.security import Permission, UnsafePath, rel_path, safe_path
from app.tools.base import Arg, InvalidInput, PermanentToolError, TemporaryToolError, ToolContext
from app.tools.registry import tool

log = get_logger(__name__)


def _fail(exc: Exception) -> None:
    message = str(exc)
    if "not linked" in message or "no longer valid" in message:
        raise PermanentToolError(f"{message} (owner must run /tglogin)") from exc
    raise TemporaryToolError(message) from exc


@tool(
    "tg_send_message",
    description=(
        "Send a Telegram message from the OWNER'S OWN account (not the bot). "
        "Target: @username, numeric id, or 'me' for Saved Messages."
    ),
    permission=Permission.WRITE,
    args={
        "to": Arg("string", True, "@username, chat id, or 'me'"),
        "text": Arg("string", True, "Message text"),
    },
    timeout_s=120,
    max_retries=1,
    side_effect=True,
    verify=lambda data: bool(data.get("message_id")),
)
async def tg_send_message(to: str, text: str, ctx: ToolContext | None = None) -> dict[str, Any]:
    if not text.strip():
        raise InvalidInput("text must not be empty")
    try:
        result = await get_userbot().send_message(to, text[:4000])
    except UserbotError as exc:
        _fail(exc)
    log.info("tg_user_sent", extra={"tool": "tg_send_message",
                                   "task_id": ctx.task_id if ctx else None})
    return result


@tool(
    "tg_send_file",
    description="Send a workspace file from the owner's own Telegram account.",
    permission=Permission.WRITE,
    args={
        "to": Arg("string", True, "@username, chat id, or 'me'"),
        "path": Arg("string", True, "Workspace-relative file"),
        "caption": Arg("string", False, "Optional caption", default=""),
    },
    timeout_s=600,
    max_retries=1,
    side_effect=True,
    verify=lambda data: bool(data.get("message_id")),
)
async def tg_send_file(to: str, path: str, caption: str = "") -> dict[str, Any]:
    try:
        target = safe_path(path, must_exist=True)
    except UnsafePath as exc:
        raise InvalidInput(str(exc)) from exc
    if target.is_dir():
        raise InvalidInput("cannot send a directory")
    if target.stat().st_size == 0:
        raise InvalidInput("refusing to send an empty file")
    try:
        result = await get_userbot().send_file(to, str(target), caption)
    except UserbotError as exc:
        _fail(exc)
    result["path"] = rel_path(target)
    return result


@tool(
    "tg_read_messages",
    description="Read recent messages from one of the owner's Telegram chats.",
    permission=Permission.READ,
    args={
        "chat": Arg("string", True, "@username, chat id, or 'me'"),
        "limit": Arg("integer", False, "How many messages", default=20),
    },
    timeout_s=120,
    max_retries=1,
)
async def tg_read_messages(chat: str, limit: int = 20) -> dict[str, Any]:
    try:
        messages = await get_userbot().read_messages(chat, max(1, min(limit, 100)))
    except UserbotError as exc:
        _fail(exc)
    return {"chat": chat, "count": len(messages), "messages": messages}


@tool(
    "tg_list_chats",
    description="List the owner's Telegram chats, groups and channels.",
    permission=Permission.READ,
    args={"limit": Arg("integer", False, "How many dialogs", default=30)},
    timeout_s=120,
    max_retries=1,
)
async def tg_list_chats(limit: int = 30) -> dict[str, Any]:
    try:
        dialogs = await get_userbot().list_dialogs(max(1, min(limit, 100)))
    except UserbotError as exc:
        _fail(exc)
    return {"count": len(dialogs), "chats": dialogs}


@tool(
    "tg_search_messages",
    description="Search across the owner's Telegram message history.",
    permission=Permission.READ,
    args={
        "query": Arg("string", True, "Text to search for"),
        "limit": Arg("integer", False, "Max results", default=20),
    },
    timeout_s=120,
    max_retries=1,
)
async def tg_search_messages(query: str, limit: int = 20) -> dict[str, Any]:
    if not query.strip():
        raise InvalidInput("query must not be empty")
    try:
        results = await get_userbot().search(query, max(1, min(limit, 100)))
    except UserbotError as exc:
        _fail(exc)
    return {"query": query, "count": len(results), "results": results}


@tool(
    "tg_bot_admin",
    description=(
        "Manage the owner's Telegram bots through BotFather. "
        "command is a BotFather command such as /mybots, /setdescription, /token; "
        "replies are the follow-up answers BotFather asks for, in order."
    ),
    permission=Permission.HIGH_RISK,
    args={
        "command": Arg("string", True, "BotFather command, e.g. /mybots"),
        "replies": Arg("array", False, "Follow-up answers in order", default=None),
    },
    timeout_s=180,
    max_retries=0,
    side_effect=True,
)
async def tg_bot_admin(command: str, replies: list[str] | None = None) -> dict[str, Any]:
    command = command.strip()
    if not command.startswith("/"):
        raise InvalidInput("command must be a BotFather command starting with '/'")
    try:
        answers = await get_userbot().botfather(command, *(replies or []))
    except UserbotError as exc:
        _fail(exc)
    # BotFather replies can contain a fresh bot token: never log them.
    return {"command": command, "replies": [a[:1500] for a in answers]}


@tool(
    "tg_account_status",
    description="Check whether the owner's Telegram account is linked to the agent.",
    permission=Permission.READ,
    args={},
    timeout_s=60,
    max_retries=0,
)
async def tg_account_status() -> dict[str, Any]:
    return await get_userbot().status()
