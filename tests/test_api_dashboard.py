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
