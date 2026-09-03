"""Inbound message handling.

The Go bridges POST every received WhatsApp / Teams message to
``/webhooks/{channel}``.  This module stores it (deduplicated by the channel's
own message id) and optionally turns it into an agent task when the sender is
on the allowlist.
"""

from __future__ import annotations

from typing import Any

from app.config import get_settings
from app.db import repo
from app.db.base import session_scope
from app.logging_conf import get_logger

log = get_logger(__name__)

SUPPORTED_CHANNELS = {"whatsapp", "teams"}


def _normalise(channel: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Map a bridge payload onto the common inbound shape."""
    message = payload.get("message") if isinstance(payload.get("message"), dict) else payload

    if channel == "whatsapp":
        return {
            "external_id": str(message.get("id") or ""),
            "chat": str(message.get("chat") or ""),
            "sender": str(message.get("sender") or ""),
            "sender_name": str(message.get("sender_name") or ""),
            "text": str(message.get("text") or ""),
            "media_path": str(message.get("media_path") or ""),
            "from_me": bool(message.get("from_me")),
            "data": {"timestamp": message.get("timestamp")},
        }

    return {
        "external_id": str(message.get("id") or ""),
        "chat": str(message.get("chat") or ""),
        "sender": str(message.get("sender") or ""),
        "sender_name": str(message.get("sender_name") or ""),
        "text": str(message.get("text") or ""),
        "media_path": "",
        "from_me": False,
        "data": {"created_at": message.get("created_at"), "has_files": message.get("has_files")},
    }


async def handle_inbound(
    channel: str, payload: dict[str, Any], notifier: Any | None = None
) -> dict[str, Any]:
    """Persist an inbound message and optionally spawn a task.

    Returns a small dict describing what happened. Never raises for ordinary
    bad input: the bridge should not retry forever on a malformed message.
    """
    channel = channel.lower().strip()
    if channel not in SUPPORTED_CHANNELS:
        return {"stored": False, "reason": f"unknown channel '{channel}'"}

    fields = _normalise(channel, payload)
    if not fields["external_id"]:
        return {"stored": False, "reason": "missing message id"}
    if fields["from_me"]:
        return {"stored": False, "reason": "own message ignored"}
    if not fields["text"] and not fields["media_path"]:
        return {"stored": False, "reason": "empty message ignored"}

    settings = get_settings()

    async with session_scope() as session:
        record = await repo.record_inbound(
            session,
            channel=channel,
            external_id=fields["external_id"],
            chat=fields["chat"],
            sender=fields["sender"],
            sender_name=fields["sender_name"],
            text=fields["text"],
            media_path=fields["media_path"],
            data=fields["data"],
        )
        if record is None:
            return {"stored": False, "reason": "duplicate", "duplicate": True}
        message_id = record.id

    log.info(
        "inbound_message",
        extra={"channel": channel, "sender": fields["sender"][:40],
               "has_media": bool(fields["media_path"])},
    )

    allowed = settings.sender_allowed(fields["sender"])
    task_id: str | None = None

    if settings.inbound_auto_task and allowed:
        instruction = (
            f"A message arrived on {channel} from "
            f"{fields['sender_name'] or fields['sender']}:\n\n{fields['text']}\n\n"
            "Decide whether it needs action. If a reply is appropriate, send it back on "
            f"{channel} to {fields['sender']}. If nothing is needed, finish and say so."
        )
        if fields["media_path"]:
            instruction += f"\n\nAn attached file was saved at: {fields['media_path']}"

        async with session_scope() as session:
            task = await repo.create_task(
                session,
                user_request=instruction,
                title=f"{channel}: {fields['text'][:60]}",
                chat_id=settings.owner_chat_id,
                permission="WRITE",
                max_steps=settings.max_task_steps,
                max_retries=settings.max_task_retries,
                context={"source": channel, "sender": fields["sender"],
                         "inbound_id": message_id},
            )
            task_id = task.id
            await repo.mark_inbound_handled(session, message_id, task_id)

        log.info("inbound_task_created", extra={"channel": channel, "task_id": task_id})

    if notifier is not None and not settings.inbound_auto_task:
        icon = "\U0001F4AC" if channel == "whatsapp" else "\U0001F465"
        who = fields["sender_name"] or fields["sender"]
        extra = f"\n\U0001F4C1 {fields['media_path']}" if fields["media_path"] else ""
        await notifier.send(
            settings.owner_chat_id,
            f"{icon} New {channel} message\n\nFrom: {who}\n\n{fields['text'][:1500]}{extra}",
            dedupe_key=f"inbound:{channel}:{fields['external_id']}",
        )

    return {
        "stored": True,
        "id": message_id,
        "channel": channel,
        "task_id": task_id,
        "auto_task": bool(task_id),
        "sender_allowed": allowed,
    }
