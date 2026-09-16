"""HTTP API: browser session-cookie auth for the web dashboard."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api import create_app


@pytest.fixture
def client(environment):
    with TestClient(create_app()) as test_client:
        yield test_client


def test_login_no_password_configured_returns_503(client):
    # environment fixture does not set WEB_UI_PASSWORD, so it defaults to "".
    resp = client.post("/api/auth/login", json={"password": "anything"})
    assert resp.status_code == 503


def test_login_wrong_password_returns_401(client, monkeypatch):
    monkeypatch.setenv("WEB_UI_PASSWORD", "correct-horse-battery-staple")
    from app.config import reload_settings

    reload_settings()

    resp = client.post("/api/auth/login", json={"password": "wrong"})
    assert resp.status_code == 401


def test_login_correct_password_sets_cookie(client, monkeypatch):
    monkeypatch.setenv("WEB_UI_PASSWORD", "correct-horse-battery-staple")
    from app.config import reload_settings

    reload_settings()

    resp = client.post("/api/auth/login", json={"password": "correct-horse-battery-staple"})
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    assert "agent_session" in resp.cookies


def test_auth_me_before_and_after_login(client, monkeypatch):
    monkeypatch.setenv("WEB_UI_PASSWORD", "hunter2")
    from app.config import reload_settings

    reload_settings()

    before = client.get("/api/auth/me")
    assert before.status_code == 200
    assert before.json() == {"authenticated": False}

    login = client.post("/api/auth/login", json={"password": "hunter2"})
    assert login.status_code == 200

    after = client.get("/api/auth/me")
    assert after.status_code == 200
    assert after.json() == {"authenticated": True}


def test_logout_clears_session(client, monkeypatch):
    monkeypatch.setenv("WEB_UI_PASSWORD", "hunter2")
    from app.config import reload_settings

    reload_settings()

    client.post("/api/auth/login", json={"password": "hunter2"})
    assert client.get("/api/auth/me").json() == {"authenticated": True}

    logout = client.post("/api/auth/logout")
    assert logout.status_code == 200
    assert logout.json() == {"ok": True}

    assert client.get("/api/auth/me").json() == {"authenticated": False}


def test_cookie_only_access_to_new_endpoints(client, monkeypatch):
    """Once logged in via cookie, GET endpoints work with NO X-API-Token header."""
    monkeypatch.setenv("WEB_UI_PASSWORD", "hunter2")
    from app.config import reload_settings

    reload_settings()

    login = client.post("/api/auth/login", json={"password": "hunter2"})
    assert login.status_code == 200

    # Deliberately send no X-API-Token header at all; rely on the cookie
    # TestClient stores cookies from the previous response automatically.
    jobs_resp = client.get("/api/jobs")
    assert jobs_resp.status_code == 200
    assert "count" in jobs_resp.json() and "jobs" in jobs_resp.json()

    autonomy_resp = client.get("/api/autonomy")
    assert autonomy_resp.status_code == 200
    assert "level" in autonomy_resp.json()

    contacts_resp = client.get("/api/contacts")
    assert contacts_resp.status_code == 200
    assert "count" in contacts_resp.json() and "contacts" in contacts_resp.json()


def test_new_endpoints_require_auth(client):
    """No X-API-Token header and no valid cookie -> 401 on the new endpoints."""
    assert client.get("/api/jobs").status_code == 401
    assert client.get("/api/autonomy").status_code == 401
    assert client.get("/api/contacts").status_code == 401
    assert client.get("/api/skills").status_code == 401


def test_new_endpoints_work_with_api_token(client):
    headers = {"X-API-Token": "test-api-token"}
    assert client.get("/api/jobs", headers=headers).status_code == 200
    assert client.get("/api/autonomy", headers=headers).status_code == 200
    assert client.get("/api/contacts", headers=headers).status_code == 200
    assert client.get("/api/skills", headers=headers).status_code == 200
    assert client.get("/api/memory", headers=headers).status_code == 200
    assert client.get("/api/bridges/status", headers=headers).status_code == 200
    assert client.get("/api/telegram_user/status", headers=headers).status_code == 200


def test_autonomy_set_valid_and_invalid(client):
    headers = {"X-API-Token": "test-api-token"}
    ok = client.post("/api/autonomy", json={"level": "high"}, headers=headers)
    assert ok.status_code == 200
    assert ok.json() == {"level": "high"}

    bad = client.post("/api/autonomy", json={"level": "chaos"}, headers=headers)
    assert bad.status_code == 400


def test_jobs_create_and_delete(client):
    headers = {"X-API-Token": "test-api-token"}
    created = client.post(
        "/api/jobs",
        json={"instruction": "say hi", "kind": "interval", "every": "5m"},
        headers=headers,
    )
    assert created.status_code == 200
    body = created.json()
    assert body["created"] is True
    job_id = body["job_id"]

    listed = client.get("/api/jobs", headers=headers).json()
    assert listed["count"] == 1

    bad = client.post(
        "/api/jobs", json={"instruction": "x", "kind": "once"}, headers=headers
    )
    assert bad.status_code == 400

    deleted = client.delete(f"/api/jobs/{job_id}", headers=headers)
    assert deleted.status_code == 200
    assert deleted.json() == {"removed": True, "job_id": job_id}

    assert client.delete("/api/jobs/doesnotexist", headers=headers).status_code == 404


def test_chat_send_and_history(client, monkeypatch):
    """Web chat uses the same handle_message brain as Telegram, same chat_id."""
    headers = {"X-API-Token": "test-api-token"}

    from app.agent.router import Decision, Intent

    async def fake_classify(*args, **kwargs):
        return Decision(Intent.CHAT, "test")

    async def fake_chat_reply(chat_id, text, llm=None, thread_id=None):
        return f"echo: {text}"

    monkeypatch.setattr("app.agent.conversation.classify", fake_classify)
    monkeypatch.setattr("app.agent.conversation.chat_reply", fake_chat_reply)

    resp = client.post("/api/chat", json={"message": "hello there"}, headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["reply"] == "echo: hello there"
    assert body["intent"] == "CHAT"

    history = client.get("/api/chat/history", headers=headers)
    assert history.status_code == 200
    roles = [m["role"] for m in history.json()["messages"]]
    assert roles[-2:] == ["user", "assistant"]
    assert history.json()["messages"][-1]["content"] == "echo: hello there"


def test_chat_requires_auth(client):
    assert client.post("/api/chat", json={"message": "hi"}).status_code == 401
    assert client.get("/api/chat/history").status_code == 401


def test_chat_works_with_cookie_only(client, monkeypatch):
    monkeypatch.setenv("WEB_UI_PASSWORD", "hunter2")
    from app.config import reload_settings

    reload_settings()

    from app.agent.router import Decision, Intent

    async def fake_classify(*args, **kwargs):
        return Decision(Intent.CHAT, "test")

    async def fake_chat_reply(chat_id, text, llm=None, thread_id=None):
        return "hi from the cookie session"

    monkeypatch.setattr("app.agent.conversation.classify", fake_classify)
    monkeypatch.setattr("app.agent.conversation.chat_reply", fake_chat_reply)

    login = client.post("/api/auth/login", json={"password": "hunter2"})
    assert login.status_code == 200

    resp = client.post("/api/chat", json={"message": "yo"})
    assert resp.status_code == 200
    assert resp.json()["reply"] == "hi from the cookie session"


def test_chat_upload_sets_pending_and_attaches(client):
    """Uploading via the web dashboard uses the exact same pending_upload
    mechanism as Telegram's F.document handler, so the next chat message
    auto-attaches the file just like it does on Telegram.
    """
    headers = {"X-API-Token": "test-api-token"}

    empty = client.get("/api/chat/pending_upload", headers=headers)
    assert empty.status_code == 200
    assert empty.json() == {"pending_upload": None}

    files = {"file": ("numbers.txt", b"12345\n67890\n", "text/plain")}
    up = client.post("/api/chat/upload", files=files, headers=headers)
    assert up.status_code == 200
    body = up.json()
    assert body["saved"] is True
    assert body["name"] == "numbers.txt"
    assert "uploads/" in body["path"]

    pending = client.get("/api/chat/pending_upload", headers=headers)
    assert pending.status_code == 200
    assert pending.json()["pending_upload"]["name"] == "numbers.txt"

    # The upload must show up as a permanent, reloadable chat message -
    # not just invisible session state that vanishes on the next page load.
    history = client.get("/api/chat/history", headers=headers).json()
    assert any("numbers.txt" in m["content"] for m in history["messages"])

    cleared = client.delete("/api/chat/pending_upload", headers=headers)
    assert cleared.status_code == 200
    assert cleared.json() == {"cleared": True}

    after = client.get("/api/chat/pending_upload", headers=headers)
    assert after.json() == {"pending_upload": None}


def test_chat_upload_requires_auth(client):
    files = {"file": ("x.txt", b"data", "text/plain")}
    assert client.post("/api/chat/upload", files=files).status_code == 401
    assert client.get("/api/chat/pending_upload").status_code == 401


def test_chat_threads_new_and_switch(client, monkeypatch):
    """New chat / thread switch mirrors Telegram's /new and /history commands."""
    headers = {"X-API-Token": "test-api-token"}

    from app.agent.router import Decision, Intent

    async def fake_classify(*args, **kwargs):
        return Decision(Intent.CHAT, "test")

    async def fake_chat_reply(chat_id, text, llm=None, thread_id=None):
        return f"echo: {text}"

    monkeypatch.setattr("app.agent.conversation.classify", fake_classify)
    monkeypatch.setattr("app.agent.conversation.chat_reply", fake_chat_reply)

    first = client.post("/api/chat", json={"message": "first thread message"}, headers=headers)
    assert first.status_code == 200

    threads_before = client.get("/api/chat/threads", headers=headers).json()
    assert threads_before["count"] == 1
    first_thread_id = threads_before["threads"][0]["thread_id"]

    new_thread = client.post("/api/chat/threads/new", headers=headers)
    assert new_thread.status_code == 200
    second_thread_id = new_thread.json()["thread_id"]
    assert second_thread_id != first_thread_id

    # New thread's history is empty; nothing from the old thread leaked in.
    history = client.get("/api/chat/history", headers=headers).json()
    assert history["thread_id"] == second_thread_id
    assert history["messages"] == []

    threads_after = client.get("/api/chat/threads", headers=headers).json()
    assert threads_after["count"] == 2
    current = next(t for t in threads_after["threads"] if t["is_current"])
    assert current["thread_id"] == second_thread_id

    # Switch back to the first thread - its history is untouched.
    switched = client.post(f"/api/chat/threads/{first_thread_id}/switch", headers=headers)
    assert switched.status_code == 200
    history_back = client.get("/api/chat/history", headers=headers).json()
    assert history_back["thread_id"] == first_thread_id
    assert len(history_back["messages"]) == 2  # user + assistant from the first send

    assert client.post(
        "/api/chat/threads/does-not-exist/switch", headers=headers
    ).status_code == 404


def test_chat_threads_require_auth(client):
    assert client.get("/api/chat/threads").status_code == 401
    assert client.post("/api/chat/threads/new").status_code == 401
    assert client.post("/api/chat/threads/x/switch").status_code == 401


def test_otpbot_config_and_status(client):
    headers = {"X-API-Token": "test-api-token"}

    default = client.get("/api/otpbot/config", headers=headers)
    assert default.status_code == 200
    assert default.json()["enabled"] is False

    updated = client.post(
        "/api/otpbot/config",
        json={"enabled": True, "target_bot": "@PBDxbot", "interval_minutes": 5},
        headers=headers,
    )
    assert updated.status_code == 200
    assert updated.json()["enabled"] is True
    assert updated.json()["interval_minutes"] == 5

    status = client.get("/api/otpbot/status", headers=headers)
    assert status.status_code == 200
    assert status.json()["config"]["enabled"] is True
    assert status.json()["last_result"] is None


def test_otpbot_requires_auth(client):
    assert client.get("/api/otpbot/config").status_code == 401
    assert client.post("/api/otpbot/config", json={}).status_code == 401
    assert client.get("/api/otpbot/status").status_code == 401
    assert client.post("/api/otpbot/run_now").status_code == 401
    assert client.get("/api/otpbot/queue").status_code == 401
    assert client.post("/api/otpbot/queue/x/tag", json={"tag": "BD"}).status_code == 401
    assert client.delete("/api/otpbot/queue/x").status_code == 401
    assert client.post("/api/otpbot/start").status_code == 401
    assert client.post("/api/otpbot/stop").status_code == 401
    assert client.post("/api/otpbot/thread").status_code == 401


def test_otpbot_queue_lifecycle(client, monkeypatch):
    headers = {"X-API-Token": "test-api-token"}

    from app.automation import otp_bot

    async def _no_sleep(seconds):
        return None

    monkeypatch.setattr(otp_bot, "_sleep", _no_sleep)

    class FakeUserbot:
        def __init__(self):
            self.sent_messages = []
            self.sent_files = []

        async def send_message(self, target, text, reply_to=None):
            self.sent_messages.append((target, text, reply_to))
            return {"sent": True, "message_id": len(self.sent_messages) + 100}

        async def send_file(self, target, path, caption="", reply_to=None):
            self.sent_files.append((target, path, reply_to))
            return {"sent": True, "message_id": 999}

        async def read_messages(self, target, limit=20):
            return [{"text": "Added.", "out": False}]

    from app.integrations.telegram_user import set_userbot

    fake = FakeUserbot()
    set_userbot(fake)
    try:
        empty = client.get("/api/otpbot/queue", headers=headers)
        assert empty.status_code == 200
        assert empty.json()["queue"] == []

        # Open the dedicated thread first - uploads only queue in there.
        opened = client.post("/api/otpbot/thread", headers=headers)
        assert opened.status_code == 200
        assert opened.json()["thread_id"]

        # Enqueue via the real upload endpoint so it goes through the same
        # code path a real "send file" does.
        upload = client.post(
            "/api/chat/upload",
            files={"file": ("plain_numbers.txt", b"+880***1111\n", "text/plain")},
            headers=headers,
        )
        assert upload.status_code == 200
        assert upload.json()["queued_for_otpbot"] is True

        queued = client.get("/api/otpbot/queue", headers=headers).json()["queue"]
        assert len(queued) == 1
        entry_id = queued[0]["id"]
        assert queued[0]["tag"] is None  # nothing to infer it from

        tagged = client.post(
            f"/api/otpbot/queue/{entry_id}/tag", json={"tag": "BD"}, headers=headers
        )
        assert tagged.status_code == 200
        assert tagged.json()["tag"] == "BD"

        started = client.post("/api/otpbot/start", headers=headers)
        assert started.status_code == 200
        assert started.json()["ok"] is True

        status = client.get("/api/otpbot/status", headers=headers).json()
        assert status["config"]["enabled"] is True
        assert len(status["active_files"]) == 1
        assert status["queue"] == []

        stopped = client.post("/api/otpbot/stop", headers=headers)
        assert stopped.status_code == 200
        assert stopped.json()["enabled"] is False
    finally:
        set_userbot(None)
