"""HTTP API: auth, health and the task control surface."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api import create_app


@pytest.fixture
def client(environment):
    with TestClient(create_app()) as test_client:
        yield test_client


def test_health_endpoints(client):
    assert client.get("/health/live").json() == {"status": "alive"}

    payload = client.get("/health").json()
    assert payload["checks"]["database"]["ok"] is True
    assert "resources" in payload and "stats" in payload

    assert client.get("/health/ready").status_code == 200


def test_api_requires_token(client):
    assert client.get("/api/tasks").status_code == 401
    assert client.get("/api/tasks", headers={"X-API-Token": "wrong"}).status_code == 401
    assert client.get("/api/tasks", headers={"X-API-Token": "test-api-token"}).status_code == 200


def test_task_crud_via_api(client):
    headers = {"X-API-Token": "test-api-token"}

    created = client.post(
        "/api/tasks", json={"instruction": "summarise the logs"}, headers=headers
    )
    assert created.status_code == 201
    task_id = created.json()["task_id"]

    detail = client.get(f"/api/tasks/{task_id}", headers=headers).json()
    assert detail["status"] == "PENDING"
    assert detail["request"] == "summarise the logs"

    listed = client.get("/api/tasks", headers=headers).json()
    assert listed["count"] == 1

    cancelled = client.post(f"/api/tasks/{task_id}/cancel", headers=headers).json()
    assert cancelled["cancelled"] is True

    after = client.get(f"/api/tasks/{task_id}", headers=headers).json()
    assert after["status"] == "CANCELLED"

    assert client.get("/api/tasks/deadbeef", headers=headers).status_code == 404


def test_tools_endpoint_lists_schemas(client):
    payload = client.get("/api/tools", headers={"X-API-Token": "test-api-token"}).json()
    assert payload["count"] >= 15
    names = {tool["name"] for tool in payload["tools"]}
    assert {"file_write", "telegram_send_file", "python_execute"} <= names
    delete_spec = next(t for t in payload["tools"] if t["name"] == "file_delete")
    assert delete_spec["permission"] == "HIGH_RISK"


def test_invalid_permission_rejected(client):
    response = client.post(
        "/api/tasks",
        json={"instruction": "x", "permission": "ROOT"},
        headers={"X-API-Token": "test-api-token"},
    )
    assert response.status_code == 422
