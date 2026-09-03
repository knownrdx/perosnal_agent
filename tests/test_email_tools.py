"""Email tools: fully offline.

No socket is ever opened: ``imaplib.IMAP4_SSL`` and ``smtplib.SMTP`` are
replaced with in-memory fakes that speak the same (awkward) response shapes the
real libraries return - nested tuples for FETCH, space-joined UID lists for
SEARCH - so the parsing code is genuinely exercised rather than bypassed.
"""

from __future__ import annotations

import imaplib
import json
import logging
import re
import smtplib
import socket
from email.message import EmailMessage
from typing import Any

import pytest

from app.security.vault import reset_vault
from app.tools import email_tools
from app.tools.base import AuthError, InvalidInput, TemporaryToolError

PASSWORD = "abcdefghijklmnop"          # shape of a real 16-char Gmail App Password
PASSWORD_2 = "zyxwvutsrqponmlk"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def vault(environment):
    reset_vault()
    yield
    reset_vault()


def log_blob(caplog: pytest.LogCaptureFixture) -> str:
    """Message plus every ``extra`` field: everything a handler could emit."""
    parts: list[str] = []
    for record in caplog.records:
        parts.append(str(record.getMessage()))
        for key, value in record.__dict__.items():
            parts.append(f"{key}={value!r}")
    return " || ".join(parts)


@pytest.fixture
def secret_log(caplog):
    """Capture everything this module logs, at DEBUG.

    Scoped to the module logger rather than the root on purpose: raising the
    ROOT level to DEBUG also enables ``app.security.vault``, whose log calls
    pass ``extra={"name": ...}`` - a reserved LogRecord attribute that makes
    logging raise KeyError. That is a pre-existing bug in a file this suite
    does not own, so we simply do not enable that logger here.
    """
    caplog.set_level(logging.DEBUG, logger="app.tools.email_tools")
    return caplog


# --------------------------------------------------------------------------- #
# Sample messages
# --------------------------------------------------------------------------- #
PLAIN_RAW = (
    b"From: Alice <alice@example.com>\r\n"
    b"To: owner@gmail.com\r\n"
    b"Subject: Quarterly report\r\n"
    b"Date: Mon, 01 Sep 2025 10:00:00 +0000\r\n"
    b"Content-Type: text/plain; charset=utf-8\r\n"
    b"\r\n"
    b"The numbers are in and they look fine.\r\n"
)

# Subject encoded per RFC 2047 (base64 utf-8): "Rechnung fur Marz - uber 50 EUR"
# written with real umlauts, so decode_header must be doing actual work.
ENCODED_RAW = (
    b"From: =?utf-8?B?SsO2cmcgTcO8bGxlcg==?= <joerg@example.de>\r\n"
    b"To: owner@gmail.com\r\n"
    b"Subject: =?utf-8?B?UmVjaG51bmcgZsO8ciBNw6RyeiDDvGJlciA1MCDigqw=?=\r\n"
    b"Date: Tue, 02 Sep 2025 08:30:00 +0000\r\n"
    b"Content-Type: text/plain; charset=utf-8\r\n"
    b"\r\n"
    b"Anbei die Rechnung.\r\n"
)

MULTIPART_RAW = (
    b"From: Bob <bob@example.com>\r\n"
    b"To: owner@gmail.com\r\n"
    b"Subject: Invitation\r\n"
    b"Date: Wed, 03 Sep 2025 09:00:00 +0000\r\n"
    b'Content-Type: multipart/mixed; boundary="OUTER"\r\n'
    b"\r\n"
    b"--OUTER\r\n"
    b'Content-Type: multipart/alternative; boundary="INNER"\r\n'
    b"\r\n"
    b"--INNER\r\n"
    b"Content-Type: text/plain; charset=utf-8\r\n"
    b"\r\n"
    b"PLAIN VERSION please read me\r\n"
    b"--INNER\r\n"
    b"Content-Type: text/html; charset=utf-8\r\n"
    b"\r\n"
    b"<html><body><p>HTML VERSION</p></body></html>\r\n"
    b"--INNER--\r\n"
    b"--OUTER\r\n"
    b"Content-Type: application/pdf\r\n"
    b'Content-Disposition: attachment; filename="agenda.pdf"\r\n'
    b"Content-Transfer-Encoding: base64\r\n"
    b"\r\n"
    b"JVBERi0xLjQK\r\n"
    b"--OUTER--\r\n"
)

HTML_ONLY_RAW = (
    b"From: News <news@example.com>\r\n"
    b"To: owner@gmail.com\r\n"
    b"Subject: Weekly digest\r\n"
    b"Date: Thu, 04 Sep 2025 07:00:00 +0000\r\n"
    b"Content-Type: text/html; charset=utf-8\r\n"
    b"\r\n"
    b"<html><head><style>p{color:red}</style>"
    b"<script>alert('xss')</script></head>"
    b"<body><h1>Headline</h1><p>Body &amp; more</p>"
    b"<a href='http://x'>link</a></body></html>\r\n"
)


def default_mailbox() -> list[dict[str, Any]]:
    """UID -> raw bytes + flags, ascending UID like a real mailbox."""
    return [
        {"uid": "101", "raw": PLAIN_RAW, "seen": True},
        {"uid": "102", "raw": ENCODED_RAW, "seen": False},
        {"uid": "103", "raw": MULTIPART_RAW, "seen": False},
        {"uid": "104", "raw": HTML_ONLY_RAW, "seen": True},
    ]


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeIMAP:
    """Minimal IMAP4_SSL stand-in returning imaplib-shaped tuples."""

    instances: list["FakeIMAP"] = []
    login_error: BaseException | None = None
    connect_error: BaseException | None = None
    mailbox: list[dict[str, Any]] = []

    def __init__(self, host, port=993, ssl_context=None, timeout=None, **kwargs):
        if type(self).connect_error is not None:
            raise type(self).connect_error
        self.host = host
        self.port = port
        self.timeout = timeout
        self.user = ""
        self.password = ""
        self.selected: str | None = None
        self.readonly: bool | None = None
        self.fetch_commands: list[tuple[str, ...]] = []
        self.search_criteria: list[str] = []
        self.logged_out = False
        self.messages = [dict(m) for m in (type(self).mailbox or default_mailbox())]
        type(self).instances.append(self)

    # -- auth ------------------------------------------------------------- #
    def login(self, user, password):
        if type(self).login_error is not None:
            raise type(self).login_error
        self.user = user
        self.password = password
        return ("OK", [b"LOGIN completed"])

    def logout(self):
        self.logged_out = True
        return ("BYE", [b"logging out"])

    # -- mailbox ---------------------------------------------------------- #
    def select(self, mailbox='"INBOX"', readonly=False):
        name = mailbox.strip('"')
        if name.lower() == "nosuchfolder":
            return ("NO", [b"[NONEXISTENT] Unknown Mailbox"])
        self.selected = name
        self.readonly = readonly
        return ("OK", [str(len(self.messages)).encode()])

    def list(self, directory='""', pattern="*"):
        return ("OK", [b'(\\HasNoChildren) "/" "INBOX"', b'(\\HasNoChildren) "/" "Sent"'])

    # -- uid dispatch ----------------------------------------------------- #
    def uid(self, command, *args):
        upper = command.upper()
        if upper == "SEARCH":
            return self._search(args)
        if upper == "FETCH":
            return self._fetch(args)
        raise imaplib.IMAP4.error(f"unsupported UID command {command}")

    def _search(self, args):
        parts = [a for a in args if a is not None]
        criteria = parts[-1]
        if isinstance(criteria, bytes):
            criteria = criteria.decode("utf-8")
        self.search_criteria.append(criteria)

        if criteria == "UNSEEN":
            hits = [m for m in self.messages if not m["seen"]]
        elif criteria == "ALL":
            hits = list(self.messages)
        else:
            hits = [m for m in self.messages if self._matches(m, criteria)]
        return ("OK", [" ".join(m["uid"] for m in hits).encode()])

    @staticmethod
    def _matches(message: dict[str, Any], criteria: str) -> bool:
        """Honour the FROM/SUBJECT/BODY terms the tool builds."""
        terms = re.findall(r'"((?:[^"\\]|\\.)*)"', criteria)
        haystack = message["raw"].decode("utf-8", errors="replace").lower()
        return any(term.replace('\\"', '"').lower() in haystack for term in terms if term)

    def _fetch(self, args):
        uid = str(args[0])
        spec = tuple(str(a) for a in args)
        self.fetch_commands.append(spec)
        for message in self.messages:
            if message["uid"] != uid:
                continue
            flags = "\\Seen" if message["seen"] else ""
            header = (
                f"1 (UID {uid} FLAGS ({flags}) BODY[] "
                f"{{{len(message['raw'])}}}"
            ).encode()
            return ("OK", [(header, message["raw"]), b")"])
        return ("NO", [b"uid not found"])


class FakeSMTP:
    """Minimal SMTP stand-in that records the fully built message."""

    instances: list["FakeSMTP"] = []
    login_error: BaseException | None = None
    connect_error: BaseException | None = None

    def __init__(self, host, port=587, timeout=None, **kwargs):
        if type(self).connect_error is not None:
            raise type(self).connect_error
        self.host = host
        self.port = port
        self.timeout = timeout
        self.user = ""
        self.password = ""
        self.started_tls = False
        self.ehlo_count = 0
        self.quit_called = False
        self.sent: list[dict[str, Any]] = []
        type(self).instances.append(self)

    def ehlo(self, name=""):
        self.ehlo_count += 1
        return (250, b"ok")

    def starttls(self, context=None, **kwargs):
        self.started_tls = True
        return (220, b"ready to start TLS")

    def login(self, user, password, **kwargs):
        if type(self).login_error is not None:
            raise type(self).login_error
        self.user = user
        self.password = password
        return (235, b"accepted")

    def send_message(self, msg, from_addr=None, to_addrs=None):
        self.sent.append({"msg": msg, "from": from_addr, "to": list(to_addrs or [])})
        return {}

    def quit(self):
        self.quit_called = True
        return (221, b"bye")


@pytest.fixture
def fake_mail(monkeypatch):
    """Patch the network classes inside the tool module's imports."""
    FakeIMAP.instances = []
    FakeIMAP.login_error = None
    FakeIMAP.connect_error = None
    FakeIMAP.mailbox = default_mailbox()
    FakeSMTP.instances = []
    FakeSMTP.login_error = None
    FakeSMTP.connect_error = None

    monkeypatch.setattr(email_tools.imaplib, "IMAP4_SSL", FakeIMAP)
    monkeypatch.setattr(email_tools.smtplib, "SMTP", FakeSMTP)
    monkeypatch.setattr(email_tools.smtplib, "SMTP_SSL", FakeSMTP)
    yield
    FakeIMAP.instances = []
    FakeSMTP.instances = []
    FakeIMAP.login_error = None
    FakeIMAP.connect_error = None
    FakeSMTP.login_error = None
    FakeSMTP.connect_error = None


async def save_default(alias: str = "default", password: str = PASSWORD, **kwargs) -> None:
    await email_tools.save_account(
        alias, kwargs.pop("address", "owner@gmail.com"), password, **kwargs
    )


# --------------------------------------------------------------------------- #
# Accounts
# --------------------------------------------------------------------------- #
async def test_save_account_then_list_never_exposes_the_password(vault, secret_log):
    await email_tools.save_account(
        "default", "owner@gmail.com", PASSWORD, display_name="The Owner"
    )

    result = await email_tools.email_accounts()

    assert result["count"] == 1
    entry = result["accounts"][0]
    assert entry == {
        "alias": "default",
        "address": "owner@gmail.com",
        "imap_host": "imap.gmail.com",
    }
    # The password must be absent from the entire returned structure, at any depth.
    assert PASSWORD not in json.dumps(result)
    assert PASSWORD not in log_blob(secret_log)
    assert secret_log.records, "no log records captured - the assertion would be vacuous"


async def test_saved_account_is_json_in_the_vault_and_defaults_to_gmail(vault):
    await save_default()

    from app.security import get_vault

    blob = get_vault().get("email:default")
    stored = json.loads(blob)
    assert stored["imap_host"] == "imap.gmail.com" and stored["imap_port"] == 993
    assert stored["smtp_host"] == "smtp.gmail.com" and stored["smtp_port"] == 587
    assert stored["password"] == PASSWORD          # readable to the agent, encrypted at rest


async def test_save_account_rejects_a_bad_address_or_empty_password(vault):
    with pytest.raises(InvalidInput):
        await email_tools.save_account("default", "not-an-address", PASSWORD)
    with pytest.raises(InvalidInput):
        await email_tools.save_account("default", "owner@gmail.com", "   ")


async def test_delete_account_roundtrip(vault):
    await save_default()
    assert await email_tools.delete_account("default") is True
    assert (await email_tools.email_accounts())["count"] == 0
    assert await email_tools.delete_account("default") is False


async def test_account_repr_hides_the_password():
    account = email_tools.Account(alias="a", address="x@y.z", password=PASSWORD)
    assert PASSWORD not in repr(account)
    assert PASSWORD not in json.dumps(account.public())


# --------------------------------------------------------------------------- #
# email_list
# --------------------------------------------------------------------------- #
async def test_email_list_parses_fake_imap_into_the_documented_shape(vault, fake_mail):
    await save_default()

    result = await email_tools.email_list(limit=10)

    assert result["account"] == "default"
    assert result["folder"] == "INBOX"
    assert result["count"] == 4
    assert result["unread_total"] == 2

    # Newest first: UID 104 down to 101.
    assert [m["uid"] for m in result["messages"]] == ["104", "103", "102", "101"]

    first = result["messages"][0]
    assert set(first) == {
        "uid", "from", "subject", "date", "snippet", "unread", "has_attachments"
    }

    plain = result["messages"][-1]
    assert plain["from"] == "Alice <alice@example.com>"
    assert plain["subject"] == "Quarterly report"
    assert plain["date"] == "Mon, 01 Sep 2025 10:00:00 +0000"
    assert plain["snippet"].startswith("The numbers are in")
    assert plain["unread"] is False
    assert plain["has_attachments"] is False

    with_pdf = next(m for m in result["messages"] if m["uid"] == "103")
    assert with_pdf["has_attachments"] is True
    assert with_pdf["unread"] is True


async def test_email_list_unread_only_and_limit(vault, fake_mail):
    await save_default()

    unread = await email_tools.email_list(unread_only=True)
    assert [m["uid"] for m in unread["messages"]] == ["103", "102"]
    assert all(m["unread"] for m in unread["messages"])

    limited = await email_tools.email_list(limit=2)
    assert limited["count"] == 2
    assert [m["uid"] for m in limited["messages"]] == ["104", "103"]
    assert limited["unread_total"] == 2       # total is independent of the page size


async def test_email_list_does_not_mark_messages_as_read(vault, fake_mail):
    await save_default()
    await email_tools.email_list()

    conn = FakeIMAP.instances[-1]
    assert conn.readonly is True
    assert all("BODY.PEEK" in " ".join(cmd) for cmd in conn.fetch_commands)
    assert conn.logged_out is True


async def test_mime_encoded_non_ascii_subject_is_decoded(vault, fake_mail):
    await save_default()

    result = await email_tools.email_list()
    encoded = next(m for m in result["messages"] if m["uid"] == "102")

    assert encoded["subject"] == "Rechnung f\u00fcr M\u00e4rz \u00fcber 50 \u20ac"
    assert "=?utf-8?" not in encoded["subject"]
    assert encoded["from"] == "J\u00f6rg M\u00fcller <joerg@example.de>"


async def test_unknown_folder_is_invalid_input(vault, fake_mail):
    await save_default()
    with pytest.raises(InvalidInput, match="nosuchfolder"):
        await email_tools.email_list(folder="nosuchfolder")


# --------------------------------------------------------------------------- #
# email_read
# --------------------------------------------------------------------------- #
async def test_email_read_prefers_text_plain(vault, fake_mail):
    await save_default()

    result = await email_tools.email_read("103")

    assert result["uid"] == "103"
    assert result["subject"] == "Invitation"
    assert result["to"] == "owner@gmail.com"
    assert "PLAIN VERSION please read me" in result["body"]
    assert "HTML VERSION" not in result["body"]
    assert "<p>" not in result["body"]
    assert result["attachments"] == ["agenda.pdf"]
    assert result["truncated"] is False


async def test_email_read_falls_back_to_html_with_tags_stripped(vault, fake_mail):
    await save_default()

    result = await email_tools.email_read("104")

    body = result["body"]
    assert "Headline" in body and "Body & more" in body
    assert "<" not in body and ">" not in body
    assert "alert('xss')" not in body          # <script> contents are dropped
    assert "color:red" not in body             # <style> contents are dropped
    assert result["attachments"] == []


async def test_email_read_truncates_and_flags_it(vault, fake_mail):
    await save_default()

    full = await email_tools.email_read("103")
    assert full["truncated"] is False

    clipped = await email_tools.email_read("103", max_chars=10)
    assert clipped["body"] == full["body"][:10]
    assert len(clipped["body"]) == 10
    assert clipped["truncated"] is True


async def test_email_read_unknown_uid_is_invalid_input(vault, fake_mail):
    await save_default()
    with pytest.raises(InvalidInput, match="999"):
        await email_tools.email_read("999")


# --------------------------------------------------------------------------- #
# email_search
# --------------------------------------------------------------------------- #
async def test_email_search_builds_a_from_subject_body_query(vault, fake_mail):
    await save_default()

    result = await email_tools.email_search("Quarterly")

    assert result["count"] == 1
    assert result["messages"][0]["uid"] == "101"
    assert set(result["messages"][0]) == {
        "uid", "from", "subject", "date", "snippet", "unread", "has_attachments"
    }

    criteria = FakeIMAP.instances[-1].search_criteria[-1]
    assert criteria == 'OR OR FROM "Quarterly" SUBJECT "Quarterly" BODY "Quarterly"'


async def test_email_search_escapes_quotes_in_the_query(vault, fake_mail):
    await save_default()

    await email_tools.email_search('say "hi"')

    criteria = FakeIMAP.instances[-1].search_criteria[-1]
    # The embedded quotes are escaped, so the IMAP string literal stays balanced.
    assert criteria.count('"') == 6 + 6          # 3 terms x 2 delimiters, 6 escaped
    assert '\\"hi\\"' in criteria


async def test_email_search_rejects_an_empty_query(vault, fake_mail):
    await save_default()
    with pytest.raises(InvalidInput, match="empty"):
        await email_tools.email_search("   ")


# --------------------------------------------------------------------------- #
# email_send
# --------------------------------------------------------------------------- #
async def test_email_send_builds_a_correct_message(vault, fake_mail, secret_log):
    await email_tools.save_account(
        "default", "owner@gmail.com", PASSWORD, display_name="The Owner"
    )

    result = await email_tools.email_send(
        to="friend@example.com",
        subject="Hello there",
        body="Short and plain.",
        cc="cc@example.com",
    )

    assert result["sent"] is True
    assert result["to"] == "friend@example.com"
    assert result["message_id"] and result["message_id"].endswith("@gmail.com>")
    assert sorted(result["recipients"]) == ["cc@example.com", "friend@example.com"]

    server = FakeSMTP.instances[-1]
    assert (server.host, server.port) == ("smtp.gmail.com", 587)
    assert server.started_tls is True
    assert server.user == "owner@gmail.com" and server.password == PASSWORD
    assert server.quit_called is True

    sent = server.sent[-1]
    msg: EmailMessage = sent["msg"]
    assert msg["From"] == "The Owner <owner@gmail.com>"
    assert msg["To"] == "friend@example.com"
    assert msg["Cc"] == "cc@example.com"
    assert msg["Subject"] == "Hello there"
    assert msg["Message-ID"] == result["message_id"]
    assert msg.get_content().strip() == "Short and plain."
    assert sent["from"] == "owner@gmail.com"
    assert sorted(sent["to"]) == ["cc@example.com", "friend@example.com"]

    assert PASSWORD not in log_blob(secret_log)


async def test_email_send_attaches_a_workspace_file(vault, fake_mail, environment):
    await save_default()
    (environment.workspace / "report.txt").write_text("hello attachment", encoding="utf-8")

    result = await email_tools.email_send(
        to="friend@example.com",
        subject="With file",
        body="see attached",
        attachments=["report.txt"],
    )

    assert result["sent"] is True
    assert result["attachments"] == ["report.txt"]

    msg: EmailMessage = FakeSMTP.instances[-1].sent[-1]["msg"]
    names = [p.get_filename() for p in msg.iter_attachments()]
    assert names == ["report.txt"]


async def test_email_send_rejects_an_attachment_outside_the_workspace(vault, fake_mail):
    await save_default()

    with pytest.raises(InvalidInput, match="escapes workspace"):
        await email_tools.email_send(
            to="friend@example.com",
            subject="Exfiltration attempt",
            body="here",
            attachments=["../../etc/passwd"],
        )
    assert FakeSMTP.instances == []          # nothing was sent


async def test_email_send_rejects_a_missing_attachment(vault, fake_mail):
    await save_default()
    with pytest.raises(InvalidInput, match="does not exist"):
        await email_tools.email_send(
            to="friend@example.com", subject="s", body="b", attachments=["nope.txt"]
        )


async def test_email_send_rejects_an_unusable_recipient(vault, fake_mail):
    await save_default()
    with pytest.raises(InvalidInput):
        await email_tools.email_send(to="   ", subject="s", body="b")
    with pytest.raises(InvalidInput, match="no valid recipient"):
        await email_tools.email_send(to="not-an-address", subject="s", body="b")


async def test_email_send_uses_implicit_tls_on_port_465(vault, fake_mail):
    await email_tools.save_account(
        "zoho", "owner@zoho.com", PASSWORD,
        imap_host="imap.zoho.com", smtp_host="smtp.zoho.com", smtp_port=465,
    )

    result = await email_tools.email_send(
        to="friend@example.com", subject="s", body="b", account="zoho"
    )

    assert result["sent"] is True
    server = FakeSMTP.instances[-1]
    assert (server.host, server.port) == ("smtp.zoho.com", 465)
    assert server.started_tls is False       # SMTP_SSL is already encrypted


# --------------------------------------------------------------------------- #
# Multi-account
# --------------------------------------------------------------------------- #
async def test_second_account_uses_its_own_credentials(vault, fake_mail):
    await email_tools.save_account("personal", "me@gmail.com", PASSWORD)
    await email_tools.save_account(
        "work", "me@company.com", PASSWORD_2,
        imap_host="imap.company.com", imap_port=1993,
        smtp_host="smtp.company.com", smtp_port=2587,
    )

    listing = await email_tools.email_accounts()
    assert {a["alias"] for a in listing["accounts"]} == {"personal", "work"}
    assert PASSWORD not in json.dumps(listing)
    assert PASSWORD_2 not in json.dumps(listing)

    result = await email_tools.email_list(account="work")
    assert result["account"] == "work"

    conn = FakeIMAP.instances[-1]
    assert (conn.host, conn.port) == ("imap.company.com", 1993)
    assert conn.user == "me@company.com"
    assert conn.password == PASSWORD_2       # NOT the personal account's password

    await email_tools.email_send(to="x@example.com", subject="s", body="b", account="work")
    server = FakeSMTP.instances[-1]
    assert (server.host, server.port) == ("smtp.company.com", 2587)
    assert server.user == "me@company.com" and server.password == PASSWORD_2


async def test_alias_is_case_insensitive(vault, fake_mail):
    await email_tools.save_account("Work", "me@company.com", PASSWORD_2)
    result = await email_tools.email_list(account="WORK")
    assert result["account"] == "work"


# --------------------------------------------------------------------------- #
# Error mapping
# --------------------------------------------------------------------------- #
async def test_unknown_alias_raises_invalid_input_naming_known_aliases(vault, fake_mail):
    await email_tools.save_account("personal", "me@gmail.com", PASSWORD)
    await email_tools.save_account("work", "me@company.com", PASSWORD_2)

    with pytest.raises(InvalidInput) as excinfo:
        await email_tools.email_list(account="holiday")

    message = str(excinfo.value)
    assert "holiday" in message
    assert "personal" in message and "work" in message
    assert FakeIMAP.instances == []          # never even dialled out


async def test_no_accounts_configured_says_so(vault, fake_mail):
    with pytest.raises(InvalidInput, match="none configured"):
        await email_tools.email_list()


async def test_imap_authentication_error_becomes_auth_error(vault, fake_mail, secret_log):
    await save_default()
    FakeIMAP.login_error = imaplib.IMAP4.error(
        "b'[AUTHENTICATIONFAILED] Invalid credentials (Failure)'"
    )

    with pytest.raises(AuthError) as excinfo:
        await email_tools.email_list()

    message = str(excinfo.value)
    assert "App Password" in message
    assert "myaccount.google.com/apppasswords" in message
    assert "2-Step Verification" in message
    assert PASSWORD not in message
    assert PASSWORD not in log_blob(secret_log)


async def test_smtp_authentication_error_becomes_auth_error(vault, fake_mail):
    await save_default()
    FakeSMTP.login_error = smtplib.SMTPAuthenticationError(
        535, b"5.7.8 Username and Password not accepted"
    )

    with pytest.raises(AuthError, match="App Password"):
        await email_tools.email_send(to="friend@example.com", subject="s", body="b")


async def test_imap_socket_timeout_becomes_temporary_error(vault, fake_mail):
    await save_default()
    FakeIMAP.connect_error = socket.timeout("timed out")

    with pytest.raises(TemporaryToolError, match="imap.gmail.com"):
        await email_tools.email_list()


async def test_smtp_connection_refused_becomes_temporary_error(vault, fake_mail):
    await save_default()
    FakeSMTP.connect_error = ConnectionRefusedError("connection refused")

    with pytest.raises(TemporaryToolError, match="smtp.gmail.com"):
        await email_tools.email_send(to="friend@example.com", subject="s", body="b")


async def test_corrupt_vault_entry_is_reported_not_crashed(vault, fake_mail):
    from app.security import get_vault

    await get_vault().set("email:broken", "not json at all")

    listing = await email_tools.email_accounts()
    assert listing["accounts"] == [
        {"alias": "broken", "address": "(corrupt entry)", "imap_host": ""}
    ]


# --------------------------------------------------------------------------- #
# test_account helper
# --------------------------------------------------------------------------- #
async def test_test_account_reports_success_with_mailboxes(vault, fake_mail):
    await save_default()

    result = await email_tools.test_account("default")

    assert result["ok"] is True
    assert "imap.gmail.com" in result["reason"]
    assert result["mailboxes"] == ["INBOX", "Sent"]
    assert PASSWORD not in json.dumps(result)


async def test_test_account_reports_failure_without_raising(vault, fake_mail, secret_log):
    await save_default()
    FakeIMAP.login_error = imaplib.IMAP4.error("[AUTHENTICATIONFAILED] Invalid credentials")

    result = await email_tools.test_account("default")

    assert result["ok"] is False
    assert "App Password" in result["reason"]
    assert result["mailboxes"] == []
    assert PASSWORD not in json.dumps(result)
    assert PASSWORD not in log_blob(secret_log)


# --------------------------------------------------------------------------- #
# Registry wiring + a whole-flow secret sweep
# --------------------------------------------------------------------------- #
async def test_tools_are_registered_with_the_right_permissions(environment):
    from app.security import Permission
    from app.tools.registry import registry

    expected = {
        "email_list": (Permission.READ, False),
        "email_read": (Permission.READ, False),
        "email_search": (Permission.READ, False),
        "email_accounts": (Permission.READ, False),
        "email_send": (Permission.WRITE, True),
    }
    for name, (permission, side_effect) in expected.items():
        registered = registry.get(name)
        assert registered is not None, f"missing tool: {name}"
        assert registered.permission is permission
        assert registered.side_effect is side_effect


async def test_email_send_runs_through_the_tool_framework(vault, fake_mail):
    from app.tools.base import ToolContext
    from app.tools.registry import registry

    await save_default()
    result = await registry.get("email_send").run(
        {"to": "friend@example.com", "subject": "s", "body": "b"}, ToolContext()
    )
    assert result.ok is True and result.data["sent"] is True


async def test_password_never_reaches_the_logs_across_every_flow(vault, fake_mail, secret_log):

    await email_tools.save_account("personal", "me@gmail.com", PASSWORD)
    await email_tools.save_account("work", "me@company.com", PASSWORD_2)
    await email_tools.email_accounts()
    await email_tools.email_list(account="personal")
    await email_tools.email_search("Quarterly", account="work")
    await email_tools.email_read("103", account="personal")
    await email_tools.email_send(
        to="friend@example.com", subject="s", body="b", account="personal"
    )
    await email_tools.test_account("work")
    await email_tools.delete_account("work")

    blob = log_blob(secret_log)
    assert secret_log.records, "no log records captured - the assertion would be vacuous"
    assert PASSWORD not in blob
    assert PASSWORD_2 not in blob
    # Body text of real mail must not be logged either, only counts.
    assert "The numbers are in" not in blob
    assert "PLAIN VERSION" not in blob
