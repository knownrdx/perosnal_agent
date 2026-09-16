"""Deterministic, LLM-free automation for OTP-number distribution bots.

Why this exists (do not route this through the normal LLM task engine): the
target bot (e.g. @PBDxbot) has a strict, undocumented-by-us FSM: a file must
be sent, then the actual command must be sent as a REPLY to that file
message, or the bot silently rejects it ("No valid phone numbers found").
Getting an LLM to reliably reproduce that exact two-step reply sequence,
every single cycle, forever, is a bad bet - and if the LLM provider is slow
or down (it has been, repeatedly), a normal Task-based job would simply never
run. Everything here calls the owner's userbot directly; no LLM in the loop.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from app.db import repo
from app.db.base import session_scope
from app.integrations.telegram_user import UserbotError, get_userbot
from app.logging_conf import get_logger
from app.security import safe_path

log = get_logger(__name__)

SETTING_KEY = "otp_bot_automation"

DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": False,
    "target_bot": "@PBDxbot",
    "file_path": "",            # workspace-relative path, or "" = latest uploads/* file
    "add_command_template": "/fan -t {tag} -l {limit} -c {count}",
    "tag": "General",
    "limit": 4,
    "count": 4,
    "quota_command": "/myquota",
    "quota_threshold": 0,       # trigger cleanup+re-add when active quota <= this
    "cleanup_command": "/useddelete",
    "interval_minutes": 10,
}

_QUOTA_RE = re.compile(r"active\s*[:\-]?\s*(\d+)", re.IGNORECASE)


@dataclass(slots=True)
class CycleResult:
    ok: bool
    action: str                     # "skipped" | "added" | "error"
    quota_reply: str = ""
    active_quota: int | None = None
    add_reply: str = ""
    error: str = ""
    ran_at: str = field(default_factory=lambda: "")


async def get_config() -> dict[str, Any]:
    async with session_scope() as session:
        stored = await repo.get_setting(session, SETTING_KEY)
    merged = dict(DEFAULT_CONFIG)
    if stored:
        merged.update(stored)
    return merged


async def save_config(patch: dict[str, Any]) -> dict[str, Any]:
    current = await get_config()
    current.update({k: v for k, v in patch.items() if k in DEFAULT_CONFIG})
    async with session_scope() as session:
        await repo.set_setting(session, SETTING_KEY, current)
    return current


async def get_last_result() -> dict[str, Any] | None:
    async with session_scope() as session:
        return await repo.get_setting(session, f"{SETTING_KEY}_last_result")


async def _save_last_result(result: CycleResult) -> None:
    async with session_scope() as session:
        await repo.set_setting(session, f"{SETTING_KEY}_last_result", asdict(result))


def _resolve_file(file_path: str):
    """The configured file, or the most recently uploaded file if unset."""
    from app.config import get_settings

    if file_path:
        return safe_path(file_path, must_exist=True)

    uploads_dir = get_settings().workspace / "uploads"
    if not uploads_dir.is_dir():
        raise FileNotFoundError("no uploads/ directory yet - upload a file first")
    candidates = [p for p in uploads_dir.iterdir() if p.is_file()]
    if not candidates:
        raise FileNotFoundError("uploads/ is empty - upload a numbers file first")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _parse_quota(text: str) -> int | None:
    match = _QUOTA_RE.search(text)
    return int(match.group(1)) if match else None


async def _last_bot_message(target: str, *, limit: int = 5) -> dict[str, Any] | None:
    messages = await get_userbot().read_messages(target, limit)
    for message in reversed(messages):
        if not message.get("out"):
            return message
    return None


async def run_cycle(config: dict[str, Any] | None = None) -> CycleResult:
    """One full check-and-fix pass. Never raises - always returns a result,
    even on failure, so a scheduled caller can log/notify without a try/except
    of its own.
    """
    from app.db.models import utcnow

    cfg = config or await get_config()
    target = cfg["target_bot"]
    result = CycleResult(ok=False, action="error", ran_at=utcnow().isoformat())

    try:
        await get_userbot().send_message(target, cfg["quota_command"])
        await asyncio.sleep(3)
        quota_msg = await _last_bot_message(target)
        if quota_msg is None:
            result.error = "no reply from the bot to the quota command"
            await _save_last_result(result)
            return result
        result.quota_reply = quota_msg["text"]
        active = _parse_quota(quota_msg["text"])
        result.active_quota = active

        if active is None:
            result.error = "could not parse an 'Active' count from the quota reply"
            await _save_last_result(result)
            return result

        if active > cfg["quota_threshold"]:
            result.ok = True
            result.action = "skipped"
            await _save_last_result(result)
            return result

        # Quota exhausted: clean up used numbers, then re-add from the file.
        await get_userbot().send_message(target, cfg["cleanup_command"])
        await asyncio.sleep(2)

        file_target = _resolve_file(cfg["file_path"])
        sent_file = await get_userbot().send_file(target, str(file_target))
        await asyncio.sleep(2)

        command_text = cfg["add_command_template"].format(
            tag=cfg["tag"], limit=cfg["limit"], count=cfg["count"]
        )
        await get_userbot().send_message(
            target, command_text, reply_to=sent_file.get("message_id")
        )
        await asyncio.sleep(3)

        add_msg = await _last_bot_message(target)
        result.add_reply = add_msg["text"] if add_msg else ""
        result.ok = True
        result.action = "added"
        await _save_last_result(result)
        return result

    except UserbotError as exc:
        result.error = f"telegram account not linked or errored: {exc}"
        await _save_last_result(result)
        return result
    except FileNotFoundError as exc:
        result.error = str(exc)
        await _save_last_result(result)
        return result
    except Exception as exc:  # noqa: BLE001 - a cycle must never crash the runner
        log.exception("otp_bot_cycle_error")
        result.error = f"unexpected error: {exc}"
        await _save_last_result(result)
        return result
