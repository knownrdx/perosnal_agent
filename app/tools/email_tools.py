"""Multi-account email tools: IMAP list/read/search plus SMTP send.

WHY stdlib-only IMAP/SMTP instead of the Gmail REST API: the agent already owns
an encrypted credential vault, so a Gmail *App Password* buys the same reach as
OAuth without a Google Cloud project, a consent screen or a refresh-token dance
for every mailbox the owner holds.  It also generalises for free - Fastmail,
Zoho, a company Exchange box - which is why host and port are stored per
account rather than hard-coded to Gmail.

WHY every socket call is wrapped in ``asyncio.to_thread``: imaplib/smtplib are
blocking, and this process runs the Telegram bot, the scheduler and the task
worker on a single event loop.  A slow mailbox must never stall the bot.

Credentials live in the vault under ``email:<alias>`` as JSON, are decrypted
only inside this module, and are never logged, never returned by a tool and
never handed to the LLM.  Reads use ``BODY.PEEK`` with ``readonly=True`` so the
agent glancing at the inbox never silently marks the owner's mail as read.
"""

from __future__ import annotations

import asyncio
import email
import imaplib
import json
import mimetypes
import re
import smtplib
import ssl
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from email.header import decode_header
from email.message import EmailMessage, Message
from email.utils import formataddr, formatdate, getaddresses, make_msgid
from html import unescape as html_unescape
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.logging_conf import get_logger
from app.security import Permission, UnsafePath, get_vault, safe_path
from app.tools.base import (
    Arg,
    AuthError,
    InvalidInput,
    PermanentToolError,
    TemporaryToolError,
)
from app.tools.registry import tool

log = get_logger(__name__)

DEFAULT_ALIAS = "default"
VAULT_PREFIX = "email:"

IMAP_TIMEOUT_S = 30
SMTP_TIMEOUT_S = 45

MAX_LIST_LIMIT = 50
SNIPPET_CHARS = 220
MAX_TOTAL_ATTACHMENT_BYTES = 24 * 1024 * 1024       # providers reject above ~25 MB

APP_PASSWORD_HINT = (
    "Gmail (and any provider with 2FA on) rejects the normal account password over "
    "IMAP/SMTP. Turn on 2-Step Verification, create a 16-character App Password at "
    "myaccount.google.com/apppasswords, and save that as the account password."
)

# Network failures worth retrying. TimeoutError is socket.timeout on 3.10+, and
# both it and ConnectionError are OSError subclasses - listed for readability.
_NETWORK_ERRORS = (TimeoutError, ConnectionError, ssl.SSLError, OSError)


# --------------------------------------------------------------------------- #
# Account storage
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class Account:
    """One mailbox. ``password`` is repr-suppressed so it cannot leak via logs."""

    alias: str
    address: str
    password: str = field(repr=False, default="")
    imap_host: str = "imap.gmail.com"
    imap_port: int = 993
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    display_name: str = ""

    def public(self) -> dict[str, Any]:
        """Everything that is safe to show the owner, the LLM or a log line."""
        return {
            "alias": self.alias,
            "address": self.address,
            "imap_host": self.imap_host,
            "imap_port": self.imap_port,
            "smtp_host": self.smtp_host,
            "smtp_port": self.smtp_port,
            "display_name": self.display_name,
        }


def _norm_alias(alias: str | None) -> str:
    return (alias or DEFAULT_ALIAS).strip().lower() or DEFAULT_ALIAS


def _vault_key(alias: str) -> str:
    return VAULT_PREFIX + _norm_alias(alias)


async def _vault() -> Any:
    vault = get_vault()
    # The singleton may be brand new (fresh process, or the owner just added an
    # account from another worker); loading is idempotent and cheap.
    if not getattr(vault, "_loaded", False):
        await vault.load()
    return vault


def _known_aliases(vault: Any) -> list[str]:
    return sorted(
        name[len(VAULT_PREFIX):]
        for name in vault.names()
        if name.startswith(VAULT_PREFIX) and len(name) > len(VAULT_PREFIX)
    )


def _decode_account(alias: str, blob: str) -> Account:
    try:
        raw = json.loads(blob)
    except (TypeError, ValueError) as exc:
        raise PermanentToolError(
            f"stored credentials for account '{alias}' are corrupt; re-add the account"
        ) from exc
    if not isinstance(raw, dict):
        raise PermanentToolError(f"stored credentials for account '{alias}' are corrupt")
    return Account(
        alias=alias,
        address=str(raw.get("address", "")),
        password=str(raw.get("password", "")),
        imap_host=str(raw.get("imap_host") or "imap.gmail.com"),
        imap_port=int(raw.get("imap_port") or 993),
        smtp_host=str(raw.get("smtp_host") or "smtp.gmail.com"),
        smtp_port=int(raw.get("smtp_port") or 587),
        display_name=str(raw.get("display_name") or ""),
    )


async def _load_account(alias: str | None) -> Account:
    wanted = _norm_alias(alias)
    vault = await _vault()
    blob = vault.get(_vault_key(wanted), fallback="")
    if not blob:
        known = _known_aliases(vault)
        listing = ", ".join(known) if known else "(none configured yet)"
        raise InvalidInput(
            f"unknown email account '{wanted}'. Known aliases: {listing}. "
            "Add one from Telegram with an address and a Gmail App Password."
        )
    account = _decode_account(wanted, blob)
    if not account.address or not account.password:
        raise InvalidInput(
            f"email account '{wanted}' is missing an address or password; re-add it"
        )
    return account


# --------------------------------------------------------------------------- #
# Header / body decoding
# --------------------------------------------------------------------------- #
def _to_text(value: Any) -> str:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")
    return "" if value is None else str(value)


def _decode_mime_header(raw: Any) -> str:
    """Turn ``=?utf-8?B?...?=`` words into readable text (RFC 2047)."""
    if raw is None:
        return ""
    text = _to_text(raw)
    if not text:
        return ""
    try:
        parts = decode_header(text)
    except Exception:  # noqa: BLE001 - malformed headers must never break a read
        return text
    out: list[str] = []
    for value, charset in parts:
        if isinstance(value, (bytes, bytearray)):
            try:
                out.append(bytes(value).decode(charset or "utf-8", errors="replace"))
            except (LookupError, UnicodeDecodeError):
                out.append(bytes(value).decode("utf-8", errors="replace"))
        else:
            out.append(str(value))
    return "".join(out).strip()


_SCRIPT_RE = re.compile(r"(?is)<(script|style)\b[^>]*>.*?</\1\s*>")
_BREAK_RE = re.compile(r"(?i)<\s*(?:br\s*/?|/p|/div|/tr|/li|/h[1-6]|/table)\s*>")
_TAG_RE = re.compile(r"(?s)<[^>]*>")
_SPACES_RE = re.compile(r"[ \t\x0b\x0c\r]+")
_BLANKLINES_RE = re.compile(r"\n\s*\n\s*\n+")


def _strip_html(raw: str) -> str:
    """Good-enough HTML to text: no dependency, no script/style leakage."""
    text = _SCRIPT_RE.sub(" ", raw)
    text = _BREAK_RE.sub("\n", text)
    text = _TAG_RE.sub(" ", text)
    text = html_unescape(text)
    text = _SPACES_RE.sub(" ", text)
    return _BLANKLINES_RE.sub("\n\n", text).strip()


def _part_text(part: Message) -> str:
    payload = part.get_payload(decode=True)
    if payload is None:
        return _to_text(part.get_payload())
    charset = part.get_content_charset() or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except (LookupError, UnicodeDecodeError):
        return payload.decode("utf-8", errors="replace")


def _is_attachment(part: Message) -> bool:
    if part.get_content_maintype() == "multipart":
        return False
    disposition = _to_text(part.get("Content-Disposition")).lower()
    return "attachment" in disposition or bool(part.get_filename())


def _attachment_names(msg: Message) -> list[str]:
    names: list[str] = []
    for part in msg.walk():
        if not _is_attachment(part):
            continue
        name = _decode_mime_header(part.get_filename()) or part.get_content_type()
        names.append(name)
    return names


def _extract_body(msg: Message) -> str:
    """Prefer text/plain; fall back to text/html with the tags stripped."""
    plain: list[str] = []
    html: list[str] = []
    for part in msg.walk():
        if part.get_content_maintype() == "multipart" or _is_attachment(part):
            continue
        content_type = part.get_content_type()
        if content_type == "text/plain":
            plain.append(_part_text(part))
        elif content_type == "text/html":
            html.append(_part_text(part))
    joined = "\n".join(plain).strip()
    if joined:
        return joined
    if html:
        return _strip_html("\n".join(html))
    return ""


def _snippet(body: str, limit: int = SNIPPET_CHARS) -> str:
    collapsed = _SPACES_RE.sub(" ", body.replace("\n", " ")).strip()
    return collapsed[:limit]


# --------------------------------------------------------------------------- #
# IMAP plumbing
# --------------------------------------------------------------------------- #
def _quote_mailbox(folder: str) -> str:
    name = (folder or "INBOX").strip().strip('"').strip() or "INBOX"
    return '"' + name + '"'


def _clean_folder(folder: str) -> str:
    return (folder or "INBOX").strip().strip('"').strip() or "INBOX"


def _auth_error(account: Account, detail: str) -> AuthError:
    # ``detail`` is server text only - the password is never interpolated.
    return AuthError(
        f"login rejected for '{account.alias}' ({account.address}): {detail}. "
        + APP_PASSWORD_HINT
    )


def _imap_connect(account: Account) -> Any:
    # Explicit default context: imaplib's built-in fallback context does not
    # verify certificates, which would make the App Password interceptable.
    context = ssl.create_default_context()
    try:
        conn = imaplib.IMAP4_SSL(
            account.imap_host,
            account.imap_port,
            ssl_context=context,
            timeout=IMAP_TIMEOUT_S,
        )
    except _NETWORK_ERRORS as exc:
        raise TemporaryToolError(
            f"cannot reach IMAP {account.imap_host}:{account.imap_port}: {exc}"
        ) from exc
    except imaplib.IMAP4.error as exc:
        raise TemporaryToolError(f"IMAP handshake failed: {exc}") from exc

    try:
        conn.login(account.address, account.password)
    except imaplib.IMAP4.error as exc:
        _close_quietly(conn)
        raise _auth_error(account, str(exc)) from exc
    except _NETWORK_ERRORS as exc:
        _close_quietly(conn)
        raise TemporaryToolError(f"IMAP login connection failed: {exc}") from exc
    return conn


def _close_quietly(conn: Any) -> None:
    for method in ("logout", "shutdown"):
        closer = getattr(conn, method, None)
        if closer is None:
            continue
        try:
            closer()
            return
        except Exception:  # noqa: BLE001 - a broken socket must not mask the real error
            continue


@contextmanager
def _imap_session(account: Account, folder: str | None) -> Iterator[Any]:
    conn = _imap_connect(account)
    try:
        if folder is not None:
            _select(conn, folder)
        yield conn
    finally:
        _close_quietly(conn)


def _select(conn: Any, folder: str) -> None:
    try:
        typ, data = conn.select(_quote_mailbox(folder), readonly=True)
    except imaplib.IMAP4.error as exc:
        raise InvalidInput(f"cannot open folder '{_clean_folder(folder)}': {exc}") from exc
    except _NETWORK_ERRORS as exc:
        raise TemporaryToolError(f"IMAP select failed: {exc}") from exc
    if typ != "OK":
        raise InvalidInput(
            f"cannot open folder '{_clean_folder(folder)}': {_first_text(data)}"
        )


def _first_text(data: Any) -> str:
    if isinstance(data, (list, tuple)) and data:
        return _to_text(data[0])
    return _to_text(data)


def _is_ascii(value: str) -> bool:
    try:
        value.encode("ascii")
    except UnicodeEncodeError:
        return False
    return True


def _uid_search(conn: Any, criteria: str) -> list[str]:
    try:
        if _is_ascii(criteria):
            typ, data = conn.uid("SEARCH", None, criteria)
        else:
            # Non-ASCII terms need an explicit charset and raw bytes; imaplib
            # would otherwise try to ASCII-encode the command and blow up.
            typ, data = conn.uid("SEARCH", "CHARSET", "UTF-8", criteria.encode("utf-8"))
    except imaplib.IMAP4.error as exc:
        raise InvalidInput(f"mailbox rejected the search: {exc}") from exc
    except _NETWORK_ERRORS as exc:
        raise TemporaryToolError(f"IMAP search failed: {exc}") from exc
    if typ != "OK":
        raise InvalidInput(f"mailbox rejected the search: {_first_text(data)}")
    return _to_text(_first_text(data)).split()


def _escape_imap(value: str) -> str:
    """Quote-safe IMAP string literal body (backslash first, then quote)."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _search_criteria(query: str) -> str:
    """FROM/SUBJECT/BODY union. IMAP OR is binary, so two ORs cover three terms."""
    term = _escape_imap(query.strip())
    return f'OR OR FROM "{term}" SUBJECT "{term}" BODY "{term}"'


_FLAGS_RE = re.compile(r"FLAGS\s*\(([^)]*)\)", re.IGNORECASE)
_UID_RE = re.compile(r"UID\s+(\d+)", re.IGNORECASE)


def _split_fetch(data: Any) -> tuple[bytes, str]:
    """Flatten an imaplib FETCH payload into (raw rfc822 bytes, metadata text)."""
    raw = bytearray()
    meta: list[str] = []
    for item in data or []:
        if isinstance(item, tuple) and len(item) >= 2:
            meta.append(_to_text(item[0]))
            if isinstance(item[1], (bytes, bytearray)):
                raw.extend(bytes(item[1]))
            elif item[1] is not None:
                raw.extend(str(item[1]).encode("utf-8", errors="replace"))
        elif item is not None:
            meta.append(_to_text(item))
    return bytes(raw), " ".join(meta)


def _fetch_one(conn: Any, uid: str) -> tuple[Message | None, str]:
    """Fetch a full message by UID without touching the \\Seen flag."""
    try:
        typ, data = conn.uid("FETCH", str(uid), "(FLAGS BODY.PEEK[])")
    except imaplib.IMAP4.error as exc:
        raise PermanentToolError(f"IMAP fetch failed for uid {uid}: {exc}") from exc
    except _NETWORK_ERRORS as exc:
        raise TemporaryToolError(f"IMAP fetch failed for uid {uid}: {exc}") from exc
    if typ != "OK":
        return None, ""
    raw, meta = _split_fetch(data)
    if not raw:
        return None, meta
    return email.message_from_bytes(raw), meta


def _summarise(uid: str, msg: Message, meta: str) -> dict[str, Any]:
    flags_match = _FLAGS_RE.search(meta)
    flags = flags_match.group(1).lower() if flags_match else ""
    uid_match = _UID_RE.search(meta)
    body = _extract_body(msg)
    return {
        "uid": uid_match.group(1) if uid_match else str(uid),
        "from": _decode_mime_header(msg.get("From")),
        "subject": _decode_mime_header(msg.get("Subject")),
        "date": _decode_mime_header(msg.get("Date")),
        "snippet": _snippet(body),
        "unread": "\\seen" not in flags,
        "has_attachments": bool(_attachment_names(msg)),
    }


def _collect(conn: Any, uids: list[str], limit: int) -> list[dict[str, Any]]:
    """Newest first. UIDs ascend, so take the tail then reverse."""
    chosen = uids[-limit:] if limit > 0 else []
    chosen.reverse()
    out: list[dict[str, Any]] = []
    for uid in chosen:
        msg, meta = _fetch_one(conn, uid)
        if msg is None:
            continue
        out.append(_summarise(uid, msg, meta))
    return out


def _clamp_limit(limit: int) -> int:
    try:
        value = int(limit)
    except (TypeError, ValueError):
        value = 20
    return max(1, min(value, MAX_LIST_LIMIT))


def _browse(
    account: Account, folder: str, limit: int, *, criteria: str, unread_only: bool
) -> dict[str, Any]:
    """Shared blocking body for email_list / email_search - runs in a thread."""
    with _imap_session(account, folder) as conn:
        unread_uids = _uid_search(conn, "UNSEEN")
        if unread_only:
            wanted = unread_uids
        elif criteria == "ALL":
            wanted = _uid_search(conn, "ALL")
        else:
            wanted = _uid_search(conn, criteria)
        messages = _collect(conn, wanted, limit)
    return {
        "account": account.alias,
        "folder": _clean_folder(folder),
        "messages": messages,
        "count": len(messages),
        "unread_total": len(unread_uids),
    }


# --------------------------------------------------------------------------- #
# Tools: reading
# --------------------------------------------------------------------------- #
@tool(
    "email_list",
    description=(
        "List recent emails from a configured mailbox (newest first). Returns sender, "
        "subject, date, a short snippet and unread/attachment flags. Reading never "
        "marks messages as read."
    ),
    permission=Permission.READ,
    args={
        "account": Arg("string", False, "Account alias, e.g. default or work",
                       default=DEFAULT_ALIAS),
        "folder": Arg("string", False, "IMAP folder, e.g. INBOX", default="INBOX"),
        "limit": Arg("integer", False, "How many messages (max 50)", default=20),
        "unread_only": Arg("boolean", False, "Only unread messages", default=False),
    },
    timeout_s=120,
    max_retries=1,
)
async def email_list(
    account: str = DEFAULT_ALIAS,
    folder: str = "INBOX",
    limit: int = 20,
    unread_only: bool = False,
) -> dict[str, Any]:
    acct = await _load_account(account)
    capped = _clamp_limit(limit)
    result = await asyncio.to_thread(
        _browse, acct, folder, capped, criteria="ALL", unread_only=bool(unread_only)
    )
    log.info(
        "email_list",
        extra={
            "tool": "email_list",
            "alias": acct.alias,
            "folder": result["folder"],
            "count": result["count"],
            "unread_total": result["unread_total"],
        },
    )
    return result


@tool(
    "email_search",
    description=(
        "Search a mailbox. The query is matched against sender, subject and body "
        "(IMAP OR search). Same result shape as email_list."
    ),
    permission=Permission.READ,
    args={
        "query": Arg("string", True, "Text to look for in FROM, SUBJECT or BODY"),
        "account": Arg("string", False, "Account alias", default=DEFAULT_ALIAS),
        "folder": Arg("string", False, "IMAP folder", default="INBOX"),
        "limit": Arg("integer", False, "How many messages (max 50)", default=20),
    },
    timeout_s=120,
    max_retries=1,
)
async def email_search(
    query: str,
    account: str = DEFAULT_ALIAS,
    folder: str = "INBOX",
    limit: int = 20,
) -> dict[str, Any]:
    if not query or not query.strip():
        raise InvalidInput("query must not be empty")
    acct = await _load_account(account)
    capped = _clamp_limit(limit)
    criteria = _search_criteria(query)
    result = await asyncio.to_thread(
        _browse, acct, folder, capped, criteria=criteria, unread_only=False
    )
    result["query"] = query.strip()
    log.info(
        "email_search",
        extra={
            "tool": "email_search",
            "alias": acct.alias,
            "folder": result["folder"],
            "count": result["count"],
        },
    )
    return result


def _read_blocking(account: Account, folder: str, uid: str, max_chars: int) -> dict[str, Any]:
    with _imap_session(account, folder) as conn:
        msg, meta = _fetch_one(conn, uid)
    if msg is None:
        raise InvalidInput(
            f"no message with uid {uid} in folder '{_clean_folder(folder)}' "
            f"for account '{account.alias}'"
        )
    body = _extract_body(msg)
    truncated = len(body) > max_chars
    uid_match = _UID_RE.search(meta)
    return {
        "uid": uid_match.group(1) if uid_match else str(uid),
        "account": account.alias,
        "folder": _clean_folder(folder),
        "from": _decode_mime_header(msg.get("From")),
        "to": _decode_mime_header(msg.get("To")),
        "cc": _decode_mime_header(msg.get("Cc")),
        "subject": _decode_mime_header(msg.get("Subject")),
        "date": _decode_mime_header(msg.get("Date")),
        "body": body[:max_chars],
        "attachments": _attachment_names(msg),
        "truncated": truncated,
    }


@tool(
    "email_read",
    description=(
        "Read one email by its UID (from email_list or email_search). Prefers the "
        "plain-text part and falls back to HTML with tags stripped."
    ),
    permission=Permission.READ,
    args={
        "uid": Arg("string", True, "Message UID returned by email_list/email_search"),
        "account": Arg("string", False, "Account alias", default=DEFAULT_ALIAS),
        "folder": Arg("string", False, "IMAP folder", default="INBOX"),
        "max_chars": Arg("integer", False, "Body characters to return", default=4000),
    },
    timeout_s=120,
    max_retries=1,
)
async def email_read(
    uid: str,
    account: str = DEFAULT_ALIAS,
    folder: str = "INBOX",
    max_chars: int = 4000,
) -> dict[str, Any]:
    clean_uid = str(uid).strip()
    if not clean_uid:
        raise InvalidInput("uid must not be empty")
    acct = await _load_account(account)
    limit = max(1, min(int(max_chars or 4000), 200_000))
    result = await asyncio.to_thread(_read_blocking, acct, folder, clean_uid, limit)
    log.info(
        "email_read",
        extra={
            "tool": "email_read",
            "alias": acct.alias,
            "folder": result["folder"],
            "uid": result["uid"],
            "body_chars": len(result["body"]),
            "attachments": len(result["attachments"]),
        },
    )
    return result


# --------------------------------------------------------------------------- #
# Tools: sending
# --------------------------------------------------------------------------- #
def _recipients(*headers: str) -> list[str]:
    pairs = getaddresses([h for h in headers if h])
    return [addr for _name, addr in pairs if addr and "@" in addr]


def _load_attachments(paths: list[str]) -> list[tuple[str, bytes, str, str]]:
    settings = get_settings()
    loaded: list[tuple[str, bytes, str, str]] = []
    total = 0
    for raw in paths:
        try:
            target: Path = safe_path(raw, must_exist=True)
        except UnsafePath as exc:
            raise InvalidInput(str(exc)) from exc
        if target.is_dir():
            raise InvalidInput(f"cannot attach a directory: {raw}")
        size = target.stat().st_size
        if size > settings.max_file_bytes:
            raise InvalidInput(
                f"attachment {target.name} exceeds max file size "
                f"({settings.max_file_mb} MB)"
            )
        total += size
        if total > MAX_TOTAL_ATTACHMENT_BYTES:
            raise InvalidInput("attachments exceed the 24 MB total the provider accepts")
        guessed = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        maintype, _, subtype = guessed.partition("/")
        loaded.append((target.name, target.read_bytes(), maintype, subtype or "octet-stream"))
    return loaded


def _build_message(
    account: Account,
    to: str,
    subject: str,
    body: str,
    cc: str,
    attachments: list[tuple[str, bytes, str, str]],
) -> EmailMessage:
    msg = EmailMessage()
    msg["From"] = (
        formataddr((account.display_name, account.address))
        if account.display_name
        else account.address
    )
    msg["To"] = to
    if cc:
        msg["Cc"] = cc
    msg["Subject"] = subject
    msg["Date"] = formatdate(localtime=True)
    domain = account.address.rpartition("@")[2] or None
    msg["Message-ID"] = make_msgid(domain=domain)
    msg.set_content(body)
    for name, data, maintype, subtype in attachments:
        msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=name)
    return msg


def _send_blocking(account: Account, msg: EmailMessage, targets: list[str]) -> None:
    context = ssl.create_default_context()
    implicit_tls = int(account.smtp_port) == 465
    try:
        if implicit_tls:
            server = smtplib.SMTP_SSL(
                account.smtp_host, account.smtp_port,
                context=context, timeout=SMTP_TIMEOUT_S,
            )
        else:
            server = smtplib.SMTP(
                account.smtp_host, account.smtp_port, timeout=SMTP_TIMEOUT_S
            )
    except _NETWORK_ERRORS as exc:
        raise TemporaryToolError(
            f"cannot reach SMTP {account.smtp_host}:{account.smtp_port}: {exc}"
        ) from exc
    except smtplib.SMTPException as exc:
        raise TemporaryToolError(f"SMTP connection failed: {exc}") from exc

    try:
        server.ehlo()
        if not implicit_tls:
            server.starttls(context=context)
            server.ehlo()
        try:
            server.login(account.address, account.password)
        except smtplib.SMTPAuthenticationError as exc:
            raise _auth_error(account, _to_text(exc.smtp_error) or str(exc.smtp_code)) from exc
        except smtplib.SMTPNotSupportedError as exc:
            raise _auth_error(account, str(exc)) from exc
        try:
            server.send_message(msg, from_addr=account.address, to_addrs=targets)
        except (smtplib.SMTPRecipientsRefused, smtplib.SMTPSenderRefused) as exc:
            raise InvalidInput(f"the mail server refused a recipient: {exc}") from exc
        except smtplib.SMTPDataError as exc:
            raise PermanentToolError(f"the mail server rejected the message: {exc}") from exc
    except _NETWORK_ERRORS as exc:
        raise TemporaryToolError(f"SMTP send failed: {exc}") from exc
    except smtplib.SMTPException as exc:
        # AuthError / InvalidInput raised above are not SMTPException, so they
        # keep their precise classification and pass straight through here.
        raise TemporaryToolError(f"SMTP conversation failed: {exc}") from exc
    finally:
        try:
            server.quit()
        except Exception:  # noqa: BLE001 - the mail is already handed over
            pass


@tool(
    "email_send",
    description=(
        "Send an email from a configured account. 'to' and 'cc' accept comma-separated "
        "addresses; attachments are workspace-relative file paths."
    ),
    permission=Permission.WRITE,
    args={
        "to": Arg("string", True, "Recipient address(es), comma separated"),
        "subject": Arg("string", True, "Subject line"),
        "body": Arg("string", True, "Plain-text message body"),
        "account": Arg("string", False, "Account alias to send from", default=DEFAULT_ALIAS),
        "cc": Arg("string", False, "CC address(es), comma separated", default=""),
        "attachments": Arg("array", False, "Workspace-relative file paths", default=None),
    },
    timeout_s=180,
    max_retries=1,
    side_effect=True,
    verify=lambda data: bool(data.get("sent")),
)
async def email_send(
    to: str,
    subject: str,
    body: str,
    account: str = DEFAULT_ALIAS,
    cc: str = "",
    attachments: list[str] | None = None,
) -> dict[str, Any]:
    if not to or not to.strip():
        raise InvalidInput("to must not be empty")
    acct = await _load_account(account)

    targets = _recipients(to, cc)
    if not targets:
        raise InvalidInput(f"no valid recipient address in '{to}'")

    paths = [str(p).strip() for p in (attachments or []) if str(p).strip()]
    files = await asyncio.to_thread(_load_attachments, paths) if paths else []

    msg = _build_message(
        acct, to.strip(), subject or "", body or "", (cc or "").strip(), files
    )
    message_id = msg["Message-ID"]

    await asyncio.to_thread(_send_blocking, acct, msg, targets)

    log.info(
        "email_sent",
        extra={
            "tool": "email_send",
            "alias": acct.alias,
            "recipients": len(targets),
            "attachments": len(files),
            "message_id": message_id,
        },
    )
    return {
        "sent": True,
        "message_id": message_id,
        "to": to.strip(),
        "cc": (cc or "").strip(),
        "account": acct.alias,
        "recipients": targets,
        "attachments": [name for name, _data, _main, _sub in files],
    }


# --------------------------------------------------------------------------- #
# Tools: account inventory
# --------------------------------------------------------------------------- #
@tool(
    "email_accounts",
    description="List the configured email accounts (aliases and addresses, no secrets).",
    permission=Permission.READ,
    args={},
    timeout_s=30,
    max_retries=0,
)
async def email_accounts() -> dict[str, Any]:
    vault = await _vault()
    accounts: list[dict[str, Any]] = []
    for alias in _known_aliases(vault):
        blob = vault.get(_vault_key(alias), fallback="")
        if not blob:
            continue
        try:
            acct = _decode_account(alias, blob)
        except PermanentToolError:
            accounts.append({"alias": alias, "address": "(corrupt entry)", "imap_host": ""})
            continue
        # Deliberately a subset of Account.public(): no password, ever.
        accounts.append(
            {"alias": acct.alias, "address": acct.address, "imap_host": acct.imap_host}
        )
    log.info("email_accounts", extra={"tool": "email_accounts", "count": len(accounts)})
    return {"accounts": accounts, "count": len(accounts)}


# --------------------------------------------------------------------------- #
# Non-tool helpers for the Telegram setup flow
# --------------------------------------------------------------------------- #
async def save_account(
    alias: str,
    address: str,
    password: str,
    *,
    imap_host: str = "imap.gmail.com",
    imap_port: int = 993,
    smtp_host: str = "smtp.gmail.com",
    smtp_port: int = 587,
    display_name: str = "",
) -> None:
    """Store (or replace) one mailbox in the encrypted vault."""
    name = _norm_alias(alias)
    address = (address or "").strip()
    password = (password or "").strip()
    if "@" not in address:
        raise InvalidInput(f"'{address}' is not a valid email address")
    if not password:
        raise InvalidInput("a password (Gmail App Password) is required")

    payload = {
        "address": address,
        "password": password,
        "imap_host": (imap_host or "imap.gmail.com").strip(),
        "imap_port": int(imap_port or 993),
        "smtp_host": (smtp_host or "smtp.gmail.com").strip(),
        "smtp_port": int(smtp_port or 587),
        "display_name": (display_name or "").strip(),
    }
    vault = await _vault()
    await vault.set(_vault_key(name), json.dumps(payload))
    log.info(
        "email_account_saved",
        extra={"alias": name, "imap_host": payload["imap_host"],
               "smtp_host": payload["smtp_host"]},
    )


async def delete_account(alias: str) -> bool:
    """Remove a mailbox. Returns False when the alias was not configured."""
    name = _norm_alias(alias)
    vault = await _vault()
    removed = await vault.delete(_vault_key(name))
    if removed:
        log.info("email_account_deleted", extra={"alias": name})
    return bool(removed)


def _probe_blocking(account: Account) -> list[str]:
    with _imap_session(account, None) as conn:
        try:
            typ, data = conn.list()
        except imaplib.IMAP4.error as exc:
            raise PermanentToolError(f"could not list mailboxes: {exc}") from exc
        except _NETWORK_ERRORS as exc:
            raise TemporaryToolError(f"could not list mailboxes: {exc}") from exc
    if typ != "OK":
        return []
    names: list[str] = []
    for row in data or []:
        text = _to_text(row)
        quoted = re.findall(r'"([^"]*)"', text)
        if quoted:
            names.append(quoted[-1])
        elif text:
            names.append(text.split()[-1])
    return names


async def test_account(alias: str = DEFAULT_ALIAS) -> dict[str, Any]:
    """Actually connect and log in, so the owner learns immediately if it works."""
    account = await _load_account(alias)
    try:
        mailboxes = await asyncio.to_thread(_probe_blocking, account)
    except (AuthError, TemporaryToolError, PermanentToolError, InvalidInput) as exc:
        log.warning("email_account_test_failed", extra={"alias": account.alias})
        return {"ok": False, "reason": str(exc), "mailboxes": []}
    log.info(
        "email_account_test_ok",
        extra={"alias": account.alias, "mailboxes": len(mailboxes)},
    )
    return {
        "ok": True,
        "reason": f"connected to {account.imap_host} as {account.address}",
        "mailboxes": mailboxes,
    }


__all__ = [
    "APP_PASSWORD_HINT",
    "DEFAULT_ALIAS",
    "Account",
    "delete_account",
    "email_accounts",
    "email_list",
    "email_read",
    "email_search",
    "email_send",
    "save_account",
    "test_account",
]
