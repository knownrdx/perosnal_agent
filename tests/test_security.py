"""Security tests: workspace sandbox, permissions, secret redaction."""

from __future__ import annotations

import pytest

from app.logging_conf import redact
from app.security import (
    Permission,
    PermissionDenied,
    UnsafePath,
    check_permission,
    safe_path,
)


def test_relative_path_resolves_inside_workspace(environment):
    resolved = safe_path("downloads/report.pdf")
    assert str(resolved).startswith(str(environment.workspace))


@pytest.mark.parametrize(
    "bad",
    [
        "../../etc/passwd",
        "downloads/../../secret.txt",
        "/etc/passwd",
        "..",
        "downloads/../../../",
    ],
)
def test_path_traversal_is_rejected(environment, bad):
    with pytest.raises(UnsafePath):
        safe_path(bad)


def test_empty_path_rejected(environment):
    with pytest.raises(UnsafePath):
        safe_path("")


def test_must_exist_flag(environment):
    with pytest.raises(UnsafePath):
        safe_path("downloads/nope.txt", must_exist=True)


def test_permission_ordering():
    check_permission(Permission.READ, Permission.WRITE)
    check_permission(Permission.WRITE, Permission.WRITE)
    check_permission(Permission.HIGH_RISK, Permission.HIGH_RISK)
    with pytest.raises(PermissionDenied):
        check_permission(Permission.HIGH_RISK, Permission.WRITE)
    with pytest.raises(PermissionDenied):
        check_permission(Permission.WRITE, Permission.READ)


def test_secrets_are_redacted_from_logs():
    token = "123456789:AAHkq3vT7mVeryLongTelegramTokenValue0000"
    assert token not in redact(f"connecting with {token}")
    assert redact({"api_token": "supersecret"})["api_token"] == "***REDACTED***"
    assert redact({"password": "hunter2"})["password"] == "***REDACTED***"
    assert redact({"path": "downloads/a.txt"})["path"] == "downloads/a.txt"


def test_memory_refuses_secrets():
    from app.tools.memory_tools import looks_like_secret

    assert looks_like_secret("bot_token: 123456789:AAHkq3vT7mVeryLongTelegramTokenValue0000")
    assert looks_like_secret("password = hunter2")
    assert not looks_like_secret("owner prefers PDF reports in the morning")
