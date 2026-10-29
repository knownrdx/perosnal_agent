"""Web-dashboard uploads into the OTP thread.

Mirrors the Telegram document handler: a file dropped into the OTP thread is
already queued for the target bot, so it must not also linger as the
"attach to the next instruction" file, and something that is not a list of
numbers must be refused rather than queued as noise.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api import create_app

HEADERS = {"X-API-Token": "test-api-token"}


@pytest.fixture
def client(environment):
    with TestClient(create_app()) as test_client:
        yield test_client


def _open_thread(client: TestClient) -> None:
    assert client.post("/api/otpbot/thread", headers=HEADERS).status_code == 200


def test_an_upload_in_the_otp_thread_is_not_left_as_a_pending_attachment(client):
    _open_thread(client)
    up = client.post(
        "/api/chat/upload",
        files={"file": ("bd.txt", b"+8801711000001\n+8801711000002\n", "text/plain")},
        data={"caption": "whatsapp 20h"},
        headers=HEADERS,
    )
    assert up.status_code == 200
    assert up.json()["queued_for_otpbot"] is True

    pending = client.get("/api/chat/pending_upload", headers=HEADERS).json()
    assert pending == {"pending_upload": None}


def test_a_spreadsheet_upload_in_the_otp_thread_is_refused(client):
    _open_thread(client)
    xlsx = b"PK\x03\x04\x14\x00\x06\x00\x08\x00\x00\x00!\x00[Content_Types].xml\n\x00 1234567"
    up = client.post(
        "/api/chat/upload",
        files={"file": ("numbers.xlsx", xlsx, "application/octet-stream")},
        headers=HEADERS,
    )
    assert up.status_code == 400
    assert ".txt" in up.json()["detail"]

    queue = client.get("/api/otpbot/queue", headers=HEADERS).json()["queue"]
    assert queue == []


def test_an_empty_upload_in_the_otp_thread_is_refused(client):
    _open_thread(client)
    up = client.post(
        "/api/chat/upload",
        files={"file": ("empty.txt", b"\n \n", "text/plain")},
        headers=HEADERS,
    )
    assert up.status_code == 400
    assert client.get("/api/otpbot/queue", headers=HEADERS).json()["queue"] == []


def test_a_spreadsheet_outside_the_otp_thread_is_still_just_saved(client):
    """Elsewhere an upload is an attachment for any task - an .xlsx there is
    perfectly legitimate and must keep working.
    """
    up = client.post(
        "/api/chat/upload",
        files={"file": ("report.xlsx", b"PK\x03\x04\x00\x00", "application/octet-stream")},
        headers=HEADERS,
    )
    assert up.status_code == 200
    assert up.json()["queued_for_otpbot"] is False


def test_an_upload_reports_which_question_is_open(client, monkeypatch):
    """The web upload asks the same questions in the same order as Telegram:
    start time, then run length - and only the first open one. There is no
    rate question: rates are set in the number bot itself.
    """
    from app.automation import otp_bot

    state: dict = {"start": None}

    async def _no_auto_start():
        return None

    async def needing_start(entries):
        return ["Bangladesh"]

    async def needing_runtime(entries):
        return ["Bangladesh"]

    async def set_start(countries):
        state["start"] = list(countries)

    monkeypatch.setattr(otp_bot, "maybe_auto_start", _no_auto_start)
    monkeypatch.setattr(otp_bot, "countries_needing_start", needing_start, raising=False)
    monkeypatch.setattr(otp_bot, "countries_needing_runtime", needing_runtime)
    monkeypatch.setattr(otp_bot, "set_awaiting_start", set_start, raising=False)
    monkeypatch.setattr(
        otp_bot, "start_question",
        lambda countries, default: f"When should {', '.join(countries)} start?", raising=False,
    )

    _open_thread(client)
    up = client.post(
        "/api/chat/upload",
        files={"file": ("bd.txt", b"+8801711000001\n+8801711000002\n", "text/plain")},
        data={"caption": "whatsapp 20h"},
        headers=HEADERS,
    )
    assert up.status_code == 200
    body = up.json()
    assert "needs_rate" not in body
    assert body["needs_start"]["countries"] == ["Bangladesh"]
    assert "When should" in body["needs_start"]["question"]
    # Only the first open question is asked; the run length waits its turn.
    assert body["needs_run_time"] is None
    assert state == {"start": ["Bangladesh"]}
