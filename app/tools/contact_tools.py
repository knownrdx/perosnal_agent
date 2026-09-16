"""Contacts/members: people the agent has seen across Telegram and WhatsApp.

Syncing pulls Telegram dialogs from the owner's userbot (when linked) and
WhatsApp contacts from the bridge's ``/contacts`` endpoint (when reachable).
Either half can be unavailable without breaking the other - a fresh install
with no WhatsApp bridge yet should still be able to sync Telegram contacts,
and vice versa.
"""

from __future__ import annotations

from typing import Any

from app.db.base import session_scope
from app.db import repo
from app.integrations import BridgeError, whatsapp_bridge
from app.logging_conf import get_logger
from app.security import Permission
from app.tools.base import Arg
from app.tools.registry import tool

log = get_logger(__name__)


async def _sync_telegram() -> dict[str, Any]:
    """Pull the owner's Telegram dialogs and upsert each as a contact."""
    try:
        from app.integrations.telegram_user import UserbotError, get_userbot

        userbot = get_userbot()
        if not userbot.has_session():
            return {"ok": False, "synced": 0, "reason": "telegram account not linked"}
        dialogs = await userbot.list_dialogs(limit=200)
    except Exception as exc:  # noqa: BLE001 - never let one half break the other
        log.warning("contacts_sync_telegram_failed", extra={"error": str(exc)[:200]})
        return {"ok": False, "synced": 0, "reason": str(exc)[:200]}

    synced = 0
    async with session_scope() as session:
        for dialog in dialogs:
            if not dialog.get("is_user"):
                # Groups/channels are places people are *seen in*, not people.
                continue
            external_id = str(dialog.get("id") or "")
            if not external_id:
                continue
            await repo.upsert_contact(
                session,
                channel="telegram",
                external_id=external_id,
                display_name=str(dialog.get("name") or ""),
                username_or_phone=str(dialog.get("username") or ""),
                seen_in=[dialog.get("name")] if dialog.get("name") else None,
            )
            synced += 1
    return {"ok": True, "synced": synced}


async def _sync_whatsapp() -> dict[str, Any]:
    """Pull contacts from the WhatsApp bridge's /contacts endpoint."""
    try:
        result = await whatsapp_bridge().request("GET", "/contacts")
    except BridgeError as exc:
        log.warning("contacts_sync_whatsapp_failed", extra={"error": str(exc)[:200]})
        return {"ok": False, "synced": 0, "reason": str(exc)[:200]}
    except Exception as exc:  # noqa: BLE001 - bridge not implemented yet, etc.
        log.warning("contacts_sync_whatsapp_failed", extra={"error": str(exc)[:200]})
        return {"ok": False, "synced": 0, "reason": str(exc)[:200]}

    contacts = result.get("contacts") if isinstance(result, dict) else None
    if not isinstance(contacts, list):
        return {"ok": False, "synced": 0, "reason": "bridge returned no contact list"}

    synced = 0
    async with session_scope() as session:
        for entry in contacts:
            if not isinstance(entry, dict):
                continue
            external_id = str(entry.get("jid") or entry.get("id") or "")
            if not external_id:
                continue
            await repo.upsert_contact(
                session,
                channel="whatsapp",
                external_id=external_id,
                display_name=str(entry.get("name") or entry.get("display_name") or ""),
                username_or_phone=str(entry.get("phone") or entry.get("number") or ""),
            )
            synced += 1
    return {"ok": True, "synced": synced}


@tool(
    "contacts_sync",
    description=(
        "Sync people the agent has seen on Telegram (owner's own dialogs) and "
        "WhatsApp (bridge contacts) into a durable contacts store, so they do "
        "not need rediscovering next time."
    ),
    permission=Permission.READ,
    args={},
    timeout_s=120,
    max_retries=1,
)
async def contacts_sync() -> dict[str, Any]:
    telegram = await _sync_telegram()
    whatsapp = await _sync_whatsapp()
    return {
        "telegram": telegram,
        "whatsapp": whatsapp,
        "total_synced": telegram.get("synced", 0) + whatsapp.get("synced", 0),
    }


@tool(
    "contacts_list",
    description="List recently seen contacts, optionally filtered by channel.",
    permission=Permission.READ,
    args={
        "channel": Arg("string", False, "telegram | whatsapp | blank for all", default=""),
        "limit": Arg("integer", False, "Max results", default=50),
    },
    timeout_s=30,
    max_retries=0,
)
async def contacts_list(channel: str = "", limit: int = 50) -> dict[str, Any]:
    async with session_scope() as session:
        rows = await repo.list_contacts(
            session, channel=channel or None, limit=max(1, min(limit, 200))
        )
    return {
        "count": len(rows),
        "contacts": [
            {
                "channel": c.channel,
                "external_id": c.external_id,
                "display_name": c.display_name,
                "username_or_phone": c.username_or_phone or "",
                "seen_in": list(c.seen_in or []),
            }
            for c in rows
        ],
    }


@tool(
    "contacts_search",
    description="Search contacts by name, username or phone number.",
    permission=Permission.READ,
    args={
        "query": Arg("string", True, "Text to search for"),
        "limit": Arg("integer", False, "Max results", default=20),
    },
    timeout_s=30,
    max_retries=0,
)
async def contacts_search(query: str, limit: int = 20) -> dict[str, Any]:
    async with session_scope() as session:
        rows = await repo.search_contacts(session, query, limit=max(1, min(limit, 50)))
    return {
        "count": len(rows),
        "contacts": [
            {
                "channel": c.channel,
                "external_id": c.external_id,
                "display_name": c.display_name,
                "username_or_phone": c.username_or_phone or "",
            }
            for c in rows
        ],
    }
