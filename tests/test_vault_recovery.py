"""The credential vault recovers from a failed boot load and is loud about a
master key that no longer matches.

Before: a database hiccup during ``load()`` at boot left the vault empty for
the life of the process, and losing ``/data/.secret_key`` (with no
AGENT_SECRET_KEY) only produced a WARNING per credential - so the Telegram
userbot just looked "not linked" with nothing pointing at the real cause.
"""

from __future__ import annotations

import logging

import pytest

from app.security.vault import CredentialVault, get_vault, reset_vault

SECRET = "super-secret-session-value-1234567890"


@pytest.fixture
def vault(environment, monkeypatch):
    monkeypatch.delenv("AGENT_SECRET_KEY", raising=False)
    reset_vault()
    yield get_vault()
    reset_vault()


def _break_db(monkeypatch):
    async def boom(_session):
        raise RuntimeError("database is not reachable yet")

    monkeypatch.setattr("app.db.repo.list_credentials", boom)


# --------------------------------------------------------------------------- #
# A failed load is retried lazily
# --------------------------------------------------------------------------- #
async def test_failed_load_is_retried_by_ensure_loaded(vault, monkeypatch):
    await vault.set("openai_api_key", SECRET)

    fresh = CredentialVault()
    with monkeypatch.context() as patch:
        _break_db(patch)
        await fresh.load()
    assert fresh.get("openai_api_key") == ""

    # Database is back: the next async access loads instead of staying empty.
    assert await fresh.ensure_loaded() is True
    assert fresh.get("openai_api_key") == SECRET


async def test_failed_load_is_retried_on_next_sync_get(vault, monkeypatch):
    await vault.set("openai_api_key", SECRET)

    fresh = CredentialVault()
    with monkeypatch.context() as patch:
        _break_db(patch)
        await fresh.load()

    # get() is synchronous, so it cannot wait for the database itself; it
    # starts a reload in the background and later calls see the values.
    fresh.get("openai_api_key")
    assert fresh._reload_task is not None
    await fresh._reload_task
    assert fresh.get("openai_api_key") == SECRET


async def test_successful_load_is_not_repeated(vault, monkeypatch):
    await vault.load()
    calls = {"n": 0}

    from app.db import repo

    original = repo.list_credentials

    async def counting(session):
        calls["n"] += 1
        return await original(session)

    monkeypatch.setattr("app.db.repo.list_credentials", counting)
    await vault.ensure_loaded()
    vault.get("anything")
    assert calls["n"] == 0
    assert vault._reload_task is None


async def test_userbot_recovers_from_failed_boot_load(vault, monkeypatch):
    from app.integrations.telegram_user import (
        API_HASH_KEY,
        API_ID_KEY,
        SESSION_KEY,
        TELETHON_AVAILABLE,
        TelegramUserbot,
    )

    if not TELETHON_AVAILABLE:
        pytest.skip("telethon is not installed")

    await vault.set(SESSION_KEY, "S" * 60)
    await vault.set(API_ID_KEY, "12345")
    await vault.set(API_HASH_KEY, "hash")

    # Simulate a boot where the database was not ready yet.
    reset_vault()
    with monkeypatch.context() as patch:
        _break_db(patch)
        await get_vault().load()

    class Client:
        async def connect(self):
            self.connected = True

        def is_connected(self):
            return True

        async def is_user_authorized(self):
            return True

        async def get_me(self):
            class Me:
                id = 1
                username = "owner"
                first_name = "Arif"
                last_name = ""
                phone = "880"

            return Me()

    monkeypatch.setattr("app.integrations.telegram_user.TelegramClient", lambda *a, **k: Client())
    monkeypatch.setattr("app.integrations.telegram_user.StringSession", lambda s="": s)

    status = await TelegramUserbot().status()
    assert status["linked"] is True, status


# --------------------------------------------------------------------------- #
# Decrypt failures are an ERROR that names the cause, never the value
# --------------------------------------------------------------------------- #
def _decrypt_errors(caplog) -> list[logging.LogRecord]:
    return [
        r for r in caplog.records
        if r.getMessage() == "vault_decrypt_failed" and r.levelno >= logging.ERROR
    ]


def _assert_no_secret_logged(caplog) -> None:
    for record in caplog.records:
        assert SECRET not in record.getMessage()
        assert SECRET not in repr(record.__dict__)


async def test_lost_key_file_is_reported_as_an_error(vault, environment, caplog):
    await vault.set("telegram_user_session", SECRET)
    (environment.workspace / ".secret_key").unlink()

    caplog.set_level(logging.INFO)
    fresh = CredentialVault()
    await fresh.load()

    errors = _decrypt_errors(caplog)
    assert len(errors) == 1
    record = errors[0]
    assert "telegram_user_session" in record.cred_names
    assert "missing" in record.cause and ".secret_key" in record.cause
    assert "AGENT_SECRET_KEY" in record.fix
    _assert_no_secret_logged(caplog)
    assert fresh.unreadable("telegram_user_session") is True


async def test_changed_env_key_is_reported_as_an_error(vault, monkeypatch, caplog):
    monkeypatch.setenv("AGENT_SECRET_KEY", "a" * 64)
    await vault.set("telegram_user_session", SECRET)
    monkeypatch.setenv("AGENT_SECRET_KEY", "b" * 64)

    caplog.set_level(logging.INFO)
    fresh = CredentialVault()
    await fresh.load()

    errors = _decrypt_errors(caplog)
    assert len(errors) == 1
    assert "AGENT_SECRET_KEY" in errors[0].cause
    _assert_no_secret_logged(caplog)
    for record in caplog.records:
        assert "a" * 64 not in repr(record.__dict__)
        assert "b" * 64 not in repr(record.__dict__)


async def test_userbot_says_why_the_session_is_unreadable(vault, environment):
    from app.integrations.telegram_user import SESSION_KEY, TelegramUserbot, UserbotError

    await vault.set(SESSION_KEY, SECRET)
    (environment.workspace / ".secret_key").unlink()
    reset_vault()
    await get_vault().load()

    userbot = TelegramUserbot()
    status = await userbot.status()
    assert status["linked"] is False
    assert "secret key" in status["error"] and "/tglogin" in status["error"]

    with pytest.raises(UserbotError, match="AGENT_SECRET_KEY"):
        await userbot.client()
