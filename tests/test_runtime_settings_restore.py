"""Owner choices made at runtime (autonomy, daily briefing) survive a restart.

They are saved to the database by /autonomy, /briefing and POST /api/autonomy
and re-applied by ``Application._restore_runtime_settings`` at boot. The
values are plain strings, so the restore must not depend on the dict-only
``repo.get_setting`` that the automation modules rely on.
"""

from __future__ import annotations

from types import SimpleNamespace

from aiogram import Dispatcher
from aiogram.filters import CommandObject

from app.config import get_settings, reload_settings
from app.db import repo
from app.db.base import session_scope


class FakeMessage:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(id=42)
        self.from_user = SimpleNamespace(id=42)
        self.answers: list[str] = []

    async def answer(self, text: str, **_: object) -> None:
        self.answers.append(text)


async def _allow(_message: object) -> bool:
    return True


def _handler(dp: Dispatcher, name: str):
    for handler in dp.message.handlers:
        for filt in handler.filters:
            if name in getattr(filt.callback, "commands", ()):
                return handler.callback
    raise AssertionError(f"/{name} is not registered")


async def _run(dp: Dispatcher, name: str, args: str | None) -> FakeMessage:
    message = FakeMessage()
    await _handler(dp, name)(message, CommandObject(prefix="/", command=name, args=args))
    return message


async def _restart():
    """Drop in-memory settings back to the .env values, as a new process would."""
    from app.main import Application

    reload_settings()
    application = Application()
    await application._restore_runtime_settings()
    return get_settings()


def _setup_dp() -> Dispatcher:
    from app.telegram.setup_commands import register_setup_handlers

    dp = Dispatcher()
    register_setup_handlers(SimpleNamespace(dp=dp, _guard=_allow, _pending_key=""))
    return dp


def _capability_dp() -> Dispatcher:
    from app.telegram.capability_commands import register_capability_commands

    dp = Dispatcher()
    register_capability_commands(dp, _allow)
    return dp


# --------------------------------------------------------------------------- #
async def test_autonomy_from_telegram_survives_restart(environment):
    assert get_settings().autonomy_level == "balanced"

    await _run(_setup_dp(), "autonomy", "high")
    assert get_settings().autonomy_level == "high"

    settings = await _restart()
    assert settings.autonomy_level == "high"


async def test_briefing_on_and_schedule_survive_restart(environment):
    dp = _capability_dp()
    await _run(dp, "briefing", "on")
    await _run(dp, "briefing", "cron 0 6 * * *")

    settings = await _restart()
    assert settings.briefing_enabled is True
    assert settings.briefing_cron == "0 6 * * *"


async def test_briefing_off_survives_restart_when_env_says_on(environment, monkeypatch):
    monkeypatch.setenv("BRIEFING_ENABLED", "true")
    reload_settings()
    assert get_settings().briefing_enabled is True

    await _run(_capability_dp(), "briefing", "off")

    settings = await _restart()
    assert settings.briefing_enabled is False


async def test_values_already_stored_by_older_builds_are_restored(environment):
    # Existing deployments already hold these rows as bare JSON strings.
    async with session_scope() as session:
        await repo.set_setting_value(session, "autonomy_level", "paranoid")
        await repo.set_setting_value(session, "briefing_enabled", "true")

    settings = await _restart()
    assert settings.autonomy_level == "paranoid"
    assert settings.briefing_enabled is True


async def test_garbage_values_are_ignored_on_restore(environment):
    async with session_scope() as session:
        await repo.set_setting_value(session, "autonomy_level", "reckless")
        await repo.set_setting_value(session, "briefing_enabled", "maybe")
        await repo.set_setting(session, "briefing_cron", {"not": "a string"})

    settings = await _restart()
    assert settings.autonomy_level == "balanced"
    assert settings.briefing_enabled is False
    assert settings.briefing_cron == "0 8 * * *"


async def test_get_setting_keeps_its_dict_only_contract(environment):
    """app/automation/* treats get_setting as dict-or-None; that must not change."""
    async with session_scope() as session:
        await repo.set_setting_value(session, "plain", "high")
        await repo.set_setting(session, "shaped", {"a": 1})

        assert await repo.get_setting(session, "plain") is None
        assert await repo.get_setting(session, "shaped") == {"a": 1}
        assert await repo.get_setting(session, "missing") is None

        assert await repo.get_setting_value(session, "plain") == "high"
        assert await repo.get_setting_value(session, "shaped") == {"a": 1}
        assert await repo.get_setting_value(session, "missing") is None


def test_autonomy_from_api_survives_restart(environment):
    from fastapi.testclient import TestClient

    from app.api import create_app
    from app.main import Application

    with TestClient(create_app()) as client:
        response = client.post(
            "/api/autonomy",
            json={"level": "paranoid"},
            headers={"X-API-Token": "test-api-token"},
        )
        assert response.status_code == 200

        reload_settings()
        application = Application()
        client.portal.call(application._restore_runtime_settings)

    assert get_settings().autonomy_level == "paranoid"
