"""Telegram user account (userbot) via Telethon or Pyrogram.

This is the owner's own Telegram account, not the bot.  It lets the agent do
things a bot cannot: read the owner's chats, message any user, and manage the
owner's other bots through BotFather.

Login is driven entirely from the Telegram bot chat:

    /tglogin <api_id> <api_hash> <phone>   -> Telegram sends a code
    /tgcode <code>                          -> signs in
    /tg2fa <password>                       -> only if 2FA is enabled

or by pasting a session string exported from another tool. Two session
string formats exist in the wild: Telethon's ``StringSession`` and
Pyrogram's ``session_string``. ``link_string_session`` tries Telethon first
(unchanged behaviour for existing users) and falls back to Pyrogram if that
fails, so either format works. Which library authenticated the stored
session is remembered in the vault so that a restart reconnects with the
right one.

The resulting session string is stored ENCRYPTED in the credential vault, so a
restart does not require logging in again and the session never touches disk
in plaintext.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from app.logging_conf import get_logger
from app.security.vault import get_vault

log = get_logger(__name__)

SESSION_KEY = "telegram_user_session"
API_ID_KEY = "telegram_user_api_id"
API_HASH_KEY = "telegram_user_api_hash"
TELEGRAM_USER_BACKEND_KEY = "telegram_user_backend"

BACKEND_TELETHON = "telethon"
BACKEND_PYROGRAM = "pyrogram"

try:
    from telethon import TelegramClient, functions
    from telethon.errors import (
        PhoneCodeExpiredError,
        PhoneCodeInvalidError,
        SessionPasswordNeededError,
    )
    from telethon.sessions import StringSession

    TELETHON_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    TelegramClient = None  # type: ignore[assignment]
    StringSession = None  # type: ignore[assignment]
    functions = None  # type: ignore[assignment]
    SessionPasswordNeededError = Exception  # type: ignore[assignment,misc]
    PhoneCodeInvalidError = Exception  # type: ignore[assignment,misc]
    PhoneCodeExpiredError = Exception  # type: ignore[assignment,misc]
    TELETHON_AVAILABLE = False

try:
    from pyrogram import Client as PyrogramClient

    PYROGRAM_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    PyrogramClient = None  # type: ignore[assignment]
    PYROGRAM_AVAILABLE = False


class UserbotError(Exception):
    """Any failure in the Telegram user account integration."""


class TwoFactorRequired(UserbotError):
    """The account has a 2FA password; /tg2fa is needed to finish signing in."""


@dataclass
class PendingLogin:
    """State between /tglogin and /tgcode."""

    api_id: int
    api_hash: str
    phone: str
    phone_code_hash: str = ""
    client: Any = None
    awaiting_password: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class TelegramUserbot:
    """Owns one client (Telethon or Pyrogram) for the owner's account."""

    def __init__(self) -> None:
        self._client: Any = None
        self._pending: PendingLogin | None = None
        self._lock = asyncio.Lock()
        self._backend: str = BACKEND_TELETHON

    # ------------------------------------------------------------------ #
    # Availability / state
    # ------------------------------------------------------------------ #
    @staticmethod
    def available() -> bool:
        return TELETHON_AVAILABLE or PYROGRAM_AVAILABLE

    def _require_telethon(self) -> None:
        if not TELETHON_AVAILABLE:
            raise UserbotError(
                "the 'telethon' package is not installed; add it to requirements and rebuild"
            )

    def _require_pyrogram(self) -> None:
        if not PYROGRAM_AVAILABLE:
            raise UserbotError(
                "the 'pyrogram' package is not installed; add it to requirements and rebuild"
            )

    def has_session(self) -> bool:
        return bool(get_vault().get(SESSION_KEY))

    def _stored_backend(self) -> str:
        """Which library authenticated the stored session (defaults to telethon)."""
        return get_vault().get(TELEGRAM_USER_BACKEND_KEY) or BACKEND_TELETHON

    # ------------------------------------------------------------------ #
    # Connection
    # ------------------------------------------------------------------ #
    async def client(self) -> Any:
        """Return a connected, authorised client using the backend the stored
        session was linked with (Telethon or Pyrogram)."""
        async with self._lock:
            backend = self._stored_backend()

            if backend == BACKEND_PYROGRAM:
                return await self._client_pyrogram()
            return await self._client_telethon()

    async def _client_telethon(self) -> Any:
        self._require_telethon()
        if (
            self._client is not None
            and self._backend == BACKEND_TELETHON
            and self._client.is_connected()
        ):
            return self._client

        vault = get_vault()
        session = vault.get(SESSION_KEY)
        api_id = vault.get(API_ID_KEY)
        api_hash = vault.get(API_HASH_KEY)
        if not (session and api_id and api_hash):
            raise UserbotError("Telegram account is not linked. Use /tglogin first.")

        client = TelegramClient(StringSession(session), int(api_id), api_hash)
        await client.connect()
        if not await client.is_user_authorized():
            await client.disconnect()
            raise UserbotError("stored Telegram session is no longer valid; run /tglogin again")
        self._client = client
        self._backend = BACKEND_TELETHON
        return client

    async def _client_pyrogram(self) -> Any:
        self._require_pyrogram()
        if self._client is not None and self._backend == BACKEND_PYROGRAM and self._client.is_connected:
            return self._client

        vault = get_vault()
        session = vault.get(SESSION_KEY)
        api_id = vault.get(API_ID_KEY)
        api_hash = vault.get(API_HASH_KEY)
        if not (session and api_id and api_hash):
            raise UserbotError("Telegram account is not linked. Use /tglogin first.")

        app = PyrogramClient(
            name=":memory:",
            session_string=session,
            api_id=int(api_id),
            api_hash=api_hash,
            in_memory=True,
        )
        try:
            await app.start()
            await app.get_me()
        except Exception as exc:  # noqa: BLE001
            try:
                await app.stop()
            except Exception:  # noqa: BLE001
                pass
            raise UserbotError(
                f"stored Telegram session is no longer valid; run /tglogin again ({str(exc)[:200]})"
            ) from exc

        self._client = app
        self._backend = BACKEND_PYROGRAM
        return app

    async def status(self) -> dict[str, Any]:
        if not self.available():
            return {"available": False, "linked": False, "error": "telethon not installed"}
        if not self.has_session():
            return {"available": True, "linked": False}
        try:
            client = await self.client()
            me = await client.get_me()
            phone_number = getattr(me, "phone", None) or getattr(me, "phone_number", None)
            return {
                "available": True,
                "linked": True,
                "user_id": me.id,
                "username": me.username or "",
                "name": " ".join(filter(None, [me.first_name, me.last_name])),
                "phone": f"+{phone_number}" if phone_number else "",
            }
        except UserbotError as exc:
            return {"available": True, "linked": False, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            return {"available": True, "linked": False, "error": str(exc)[:200]}

    # ------------------------------------------------------------------ #
    # Login flow
    # ------------------------------------------------------------------ #
    async def start_login(self, api_id: int, api_hash: str, phone: str) -> dict[str, Any]:
        """Step 1: ask Telegram to send the login code."""
        self._require_telethon()
        phone = phone.strip()
        if not phone.startswith("+"):
            phone = "+" + phone.lstrip("+")

        client = TelegramClient(StringSession(), int(api_id), api_hash)
        await client.connect()
        try:
            sent = await client.send_code_request(phone)
        except Exception as exc:  # noqa: BLE001
            await client.disconnect()
            raise UserbotError(f"could not send the login code: {str(exc)[:200]}") from exc

        self._pending = PendingLogin(
            api_id=int(api_id),
            api_hash=api_hash,
            phone=phone,
            phone_code_hash=sent.phone_code_hash,
            client=client,
        )
        log.info("userbot_code_sent", extra={"phone": phone[:5] + "***"})
        return {"code_sent": True, "phone": phone}

    async def submit_code(self, code: str) -> dict[str, Any]:
        """Step 2: complete sign-in with the code Telegram sent."""
        if self._pending is None:
            raise UserbotError("no login in progress; start with /tglogin")

        pending = self._pending
        code = code.strip().replace(" ", "").replace("-", "")
        try:
            await pending.client.sign_in(
                phone=pending.phone, code=code, phone_code_hash=pending.phone_code_hash
            )
        except SessionPasswordNeededError:
            pending.awaiting_password = True
            raise TwoFactorRequired("this account has 2FA; send /tg2fa <password>") from None
        except PhoneCodeInvalidError as exc:
            raise UserbotError("that code is not correct") from exc
        except PhoneCodeExpiredError as exc:
            raise UserbotError("that code expired; run /tglogin again") from exc
        except Exception as exc:  # noqa: BLE001
            raise UserbotError(f"sign-in failed: {str(exc)[:200]}") from exc

        return await self._finish_login()

    async def submit_password(self, password: str) -> dict[str, Any]:
        """Step 3 (optional): 2FA password."""
        if self._pending is None or not self._pending.awaiting_password:
            raise UserbotError("no 2FA step in progress")
        try:
            await self._pending.client.sign_in(password=password)
        except Exception as exc:  # noqa: BLE001
            raise UserbotError(f"2FA failed: {str(exc)[:200]}") from exc
        return await self._finish_login()

    async def _finish_login(self) -> dict[str, Any]:
        pending = self._pending
        assert pending is not None
        client = pending.client

        session_string = client.session.save()
        vault = get_vault()
        await vault.set(SESSION_KEY, session_string)
        await vault.set(API_ID_KEY, str(pending.api_id))
        await vault.set(API_HASH_KEY, pending.api_hash)
        await vault.set(TELEGRAM_USER_BACKEND_KEY, BACKEND_TELETHON)

        me = await client.get_me()
        self._client = client
        self._backend = BACKEND_TELETHON
        self._pending = None

        log.info("userbot_linked", extra={"user_id": me.id})
        return {
            "linked": True,
            "user_id": me.id,
            "username": me.username or "",
            "name": " ".join(filter(None, [me.first_name, me.last_name])),
        }

    async def link_string_session(
        self, session_string: str, api_id: int = 0, api_hash: str = ""
    ) -> dict[str, Any]:
        """Attach an account using a session string produced elsewhere.

        Phone + code is the friendlier path, but it fails in two common cases:
        Telegram refuses to deliver a login code to a datacentre IP, and some
        accounts are already exported as a string from another machine. Pasting
        the string skips the code entirely.

        Two session string formats exist in the wild: Telethon's and
        Pyrogram's. This tries Telethon first (unchanged behaviour for
        existing users); if that fails it tries the string as a Pyrogram
        session. The string is verified against Telegram before it is
        stored, so a typo or a revoked session fails here rather than
        silently later.
        """
        session_string = session_string.strip()
        if len(session_string) < 40:
            raise UserbotError("that does not look like a Telegram session string")

        vault = get_vault()
        await vault.load()
        api_id = api_id or int(vault.get(API_ID_KEY, "0") or 0)
        api_hash = api_hash or vault.get(API_HASH_KEY, "")
        if not api_id or not api_hash:
            raise UserbotError(
                "api_id and api_hash are required the first time; get them from my.telegram.org"
            )

        telethon_error: str | None = None
        if TELETHON_AVAILABLE:
            try:
                return await self._link_telethon_string(session_string, api_id, api_hash)
            except Exception as exc:  # noqa: BLE001 - fall through to pyrogram
                telethon_error = str(exc)[:200]
        else:
            telethon_error = "telethon is not installed"

        pyrogram_error: str | None = None
        if PYROGRAM_AVAILABLE:
            try:
                return await self._link_pyrogram_string(session_string, api_id, api_hash)
            except Exception as exc:  # noqa: BLE001
                pyrogram_error = str(exc)[:200]
        else:
            pyrogram_error = "pyrogram is not installed"

        raise UserbotError(
            "that session string was not recognised by Telethon "
            f"({telethon_error}) or Pyrogram ({pyrogram_error})"
        )

    async def _link_telethon_string(
        self, session_string: str, api_id: int, api_hash: str
    ) -> dict[str, Any]:
        vault = get_vault()
        try:
            client = TelegramClient(StringSession(session_string), api_id, api_hash)
            await client.connect()
        except Exception as exc:  # noqa: BLE001 - the string may be malformed
            raise UserbotError(f"could not use that session string: {str(exc)[:200]}") from exc

        try:
            authorised = await client.is_user_authorized()
        except Exception as exc:  # noqa: BLE001
            await self._safe_disconnect(client)
            raise UserbotError(f"session check failed: {str(exc)[:200]}") from exc

        if not authorised:
            await self._safe_disconnect(client)
            raise UserbotError(
                "that session string is not authorised (expired, revoked, or from a "
                "different api_id)"
            )

        me = await client.get_me()

        await vault.set(SESSION_KEY, session_string)
        await vault.set(API_ID_KEY, str(api_id))
        await vault.set(API_HASH_KEY, api_hash)
        await vault.set(TELEGRAM_USER_BACKEND_KEY, BACKEND_TELETHON)

        self._client = client
        self._backend = BACKEND_TELETHON
        self._pending = None

        log.info("userbot_linked_via_string", extra={"user_id": me.id, "backend": BACKEND_TELETHON})
        return {
            "linked": True,
            "method": "string_session",
            "user_id": me.id,
            "username": me.username or "",
            "name": " ".join(filter(None, [me.first_name, me.last_name])),
        }

    async def _link_pyrogram_string(
        self, session_string: str, api_id: int, api_hash: str
    ) -> dict[str, Any]:
        vault = get_vault()
        app = PyrogramClient(
            name=":memory:",
            session_string=session_string,
            api_id=api_id,
            api_hash=api_hash,
            in_memory=True,
        )
        try:
            await app.start()
            me = await app.get_me()
        except Exception as exc:  # noqa: BLE001 - the string may be malformed/unauthorised
            try:
                await app.stop()
            except Exception:  # noqa: BLE001
                pass
            raise UserbotError(f"could not use that session string: {str(exc)[:200]}") from exc

        await vault.set(SESSION_KEY, session_string)
        await vault.set(API_ID_KEY, str(api_id))
        await vault.set(API_HASH_KEY, api_hash)
        await vault.set(TELEGRAM_USER_BACKEND_KEY, BACKEND_PYROGRAM)

        self._client = app
        self._backend = BACKEND_PYROGRAM
        self._pending = None

        log.info("userbot_linked_via_string", extra={"user_id": me.id, "backend": BACKEND_PYROGRAM})
        return {
            "linked": True,
            "method": "string_session",
            "user_id": me.id,
            "username": me.username or "",
            "name": " ".join(filter(None, [me.first_name, me.last_name])),
        }

    async def export_session_string(self) -> str:
        """Return the current session string so the owner can back it up."""
        client = await self.client()
        if self._backend == BACKEND_PYROGRAM:
            return await client.export_session_string()
        return StringSession.save(client.session)

    @staticmethod
    async def _safe_disconnect(client: Any) -> None:
        try:
            await client.disconnect()
        except Exception:  # noqa: BLE001 - cleanup must not mask the real error
            pass

    async def logout(self) -> bool:
        vault = get_vault()
        if self._client is not None:
            try:
                await self._client.log_out()
            except Exception:  # noqa: BLE001
                pass
            if self._backend == BACKEND_PYROGRAM:
                try:
                    await self._client.stop()
                except Exception:  # noqa: BLE001
                    pass
            else:
                try:
                    await self._client.disconnect()
                except Exception:  # noqa: BLE001
                    pass
        self._client = None
        self._pending = None
        removed = await vault.delete(SESSION_KEY)
        await vault.delete(API_ID_KEY)
        await vault.delete(API_HASH_KEY)
        await vault.delete(TELEGRAM_USER_BACKEND_KEY)
        self._backend = BACKEND_TELETHON
        log.info("userbot_logged_out")
        return removed

    async def close(self) -> None:
        if self._client is not None:
            try:
                if self._backend == BACKEND_PYROGRAM:
                    await self._client.stop()
                else:
                    await self._client.disconnect()
            except Exception:  # noqa: BLE001
                pass
        self._client = None

    # ------------------------------------------------------------------ #
    # Actions
    # ------------------------------------------------------------------ #
    async def send_message(self, target: str, text: str) -> dict[str, Any]:
        client = await self.client()
        if self._backend == BACKEND_PYROGRAM:
            peer = self._resolve_pyrogram(target)
            sent = await client.send_message(peer, text)
            return {"sent": True, "message_id": sent.id, "to": str(target)}
        entity = await self._resolve(client, target)
        sent = await client.send_message(entity, text)
        return {"sent": True, "message_id": sent.id, "to": str(target)}

    async def send_file(self, target: str, path: str, caption: str = "") -> dict[str, Any]:
        client = await self.client()
        if self._backend == BACKEND_PYROGRAM:
            peer = self._resolve_pyrogram(target)
            sent = await client.send_document(peer, path, caption=caption or "")
            return {"sent": True, "message_id": sent.id, "to": str(target)}
        entity = await self._resolve(client, target)
        sent = await client.send_file(entity, path, caption=caption or None)
        return {"sent": True, "message_id": sent.id, "to": str(target)}

    async def read_messages(self, target: str, limit: int = 20) -> list[dict[str, Any]]:
        client = await self.client()
        out: list[dict[str, Any]] = []
        if self._backend == BACKEND_PYROGRAM:
            peer = self._resolve_pyrogram(target)
            async for message in client.get_chat_history(peer, limit=limit):
                message_date = getattr(message, "date", None)
                out.append(
                    {
                        "id": message.id,
                        "text": (message.text or message.caption or "")[:2000],
                        "out": bool(getattr(message, "outgoing", False)),
                        "date": message_date.isoformat() if message_date else None,
                        "sender_id": getattr(message.from_user, "id", None)
                        if message.from_user
                        else None,
                    }
                )
            return out
        entity = await self._resolve(client, target)
        async for message in client.iter_messages(entity, limit=limit):
            out.append(
                {
                    "id": message.id,
                    "text": (message.message or "")[:2000],
                    "out": bool(message.out),
                    "date": message.date.isoformat() if message.date else None,
                    "sender_id": getattr(message, "sender_id", None),
                }
            )
        return out

    async def list_dialogs(self, limit: int = 30) -> list[dict[str, Any]]:
        client = await self.client()
        out: list[dict[str, Any]] = []
        if self._backend == BACKEND_PYROGRAM:
            async for dialog in client.get_dialogs(limit=limit):
                chat = dialog.chat
                chat_type = str(getattr(chat, "type", "")).lower()
                is_group = "group" in chat_type
                is_channel = "channel" in chat_type
                is_user = "private" in chat_type or "bot" in chat_type
                name = chat.title or " ".join(
                    filter(None, [getattr(chat, "first_name", None), getattr(chat, "last_name", None)])
                )
                out.append(
                    {
                        "id": chat.id,
                        "name": name or "",
                        "username": getattr(chat, "username", "") or "",
                        "unread": dialog.unread_messages_count,
                        "is_group": is_group,
                        "is_channel": is_channel,
                        "is_user": is_user,
                    }
                )
            return out
        async for dialog in client.iter_dialogs(limit=limit):
            out.append(
                {
                    "id": dialog.id,
                    "name": dialog.name or "",
                    "username": getattr(dialog.entity, "username", "") or "",
                    "unread": dialog.unread_count,
                    "is_group": bool(dialog.is_group),
                    "is_channel": bool(dialog.is_channel),
                    "is_user": bool(dialog.is_user),
                }
            )
        return out

    async def search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        client = await self.client()
        out: list[dict[str, Any]] = []
        if self._backend == BACKEND_PYROGRAM:
            async for message in client.search_global(query, limit=limit):
                message_date = getattr(message, "date", None)
                chat_id = getattr(message.chat, "id", None) if message.chat else None
                out.append(
                    {
                        "id": message.id,
                        "chat_id": chat_id,
                        "text": (message.text or message.caption or "")[:500],
                        "date": message_date.isoformat() if message_date else None,
                    }
                )
            return out
        async for message in client.iter_messages(None, search=query, limit=limit):
            out.append(
                {
                    "id": message.id,
                    "chat_id": getattr(message, "chat_id", None),
                    "text": (message.message or "")[:500],
                    "date": message.date.isoformat() if message.date else None,
                }
            )
        return out

    # ------------------------------------------------------------------ #
    # Bot management through BotFather
    # ------------------------------------------------------------------ #
    async def botfather(self, command: str, *replies: str, wait_s: float = 3.0) -> list[str]:
        """Send a command to @BotFather and collect its answers.

        ``replies`` are follow-up messages sent one after another, which is how
        BotFather's multi-step dialogues (e.g. /newbot -> name -> username)
        work.

        Only supported on a Telethon-linked session: Pyrogram has no
        equivalent of Telethon's ``client.conversation()`` context manager.
        """
        client = await self.client()
        if self._backend == BACKEND_PYROGRAM:
            raise UserbotError(
                "bot management via BotFather requires a Telethon-linked session; "
                "use /tglogin or paste a Telethon session string for this feature"
            )
        conversation_replies: list[str] = []

        async with client.conversation("BotFather", timeout=max(10.0, wait_s * (len(replies) + 2))) as conv:
            await conv.send_message(command)
            response = await conv.get_response()
            conversation_replies.append(response.message or "")

            for reply in replies:
                await conv.send_message(reply)
                response = await conv.get_response()
                conversation_replies.append(response.message or "")

        return conversation_replies

    async def list_bots(self) -> list[str]:
        """Ask BotFather which bots the account owns."""
        replies = await self.botfather("/mybots")
        return replies

    @staticmethod
    async def _resolve(client: Any, target: str) -> Any:
        """Accept @username, numeric id, phone or 'me'."""
        raw = str(target).strip()
        if raw.lower() in {"me", "self", "saved"}:
            return "me"
        if raw.lstrip("-").isdigit():
            return int(raw)
        return raw

    @staticmethod
    def _resolve_pyrogram(target: str) -> Any:
        """Accept @username, numeric id, phone or 'me' (Pyrogram peer form)."""
        raw = str(target).strip()
        if raw.lower() in {"me", "self", "saved"}:
            return "me"
        if raw.lstrip("-").isdigit():
            return int(raw)
        return raw


_userbot: TelegramUserbot | None = None


def get_userbot() -> TelegramUserbot:
    global _userbot
    if _userbot is None:
        _userbot = TelegramUserbot()
    return _userbot


def set_userbot(instance: TelegramUserbot | None) -> None:
    """Injection point for tests."""
    global _userbot
    _userbot = instance


async def close_userbot() -> None:
    global _userbot
    if _userbot is not None:
        await _userbot.close()
    _userbot = None
