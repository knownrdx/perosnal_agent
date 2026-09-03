"""Secure per-site login + session manager.

The point of ``app.integrations.web_session`` is that the model never sees a
password, so most of these tests are leak tests: the secret must not appear in
return values, exception messages or log records.
"""

from __future__ import annotations

import importlib
import json
import logging
import sys
from typing import Any

import pytest

from app.integrations.web_session import (
    AUTO_PASSWORD_SELECTORS,
    SiteProfile,
    SiteSessionManager,
    WebSessionError,
    get_session_manager,
    reset_session_manager,
)
from app.security.vault import reset_vault

PASSWORD = "hunter2-Sup3r-Secret!"
LOGIN_URL = "https://portal.example.test/login"
DASHBOARD_URL = "https://portal.example.test/dashboard"


# --------------------------------------------------------------------------- #
# Fake page - implements exactly the surface web_session is allowed to use
# --------------------------------------------------------------------------- #
class FakeContext:
    """Only ``storage_state()``; no ``add_cookies`` on purpose, so the module
    is proven not to depend on it."""

    def __init__(self) -> None:
        self.state: dict[str, Any] = {"cookies": [], "origins": []}

    async def storage_state(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.state))


class FakePage:
    """Simulates a login form.

    mode:
      "success"  -> submitting swaps the form for a dashboard
      "fail"     -> submitting keeps the form and shows an error
      "already"  -> the saved session is still valid, no form is ever rendered
    """

    LOGIN_FIELDS = ("input[name=user]", "input[type=password]", "button[type=submit]")

    def __init__(
        self, mode: str = "success", *, fields: tuple[str, ...] | None = None
    ) -> None:
        self.mode = mode
        self.context = FakeContext()
        self.navigations: list[str] = []
        self.fills: list[tuple[str, str]] = []
        self.clicks: list[str] = []
        self.waits: list[int] = []
        self.queries: list[str] = []

        if mode == "already":
            self._present: set[str] = {"#dashboard"}
            self._url = DASHBOARD_URL
            self._body = "Welcome back, tester"
            self._title = "Dashboard"
        else:
            self._present = set(fields if fields is not None else self.LOGIN_FIELDS)
            self._url = LOGIN_URL
            self._body = "Please sign in to continue"
            self._title = "Sign in"

    # -- PageLike surface ------------------------------------------------ #
    @property
    def url(self) -> str:
        return self._url

    async def goto(self, url: str) -> None:
        self.navigations.append(url)
        if self.mode != "already":
            self._url = url

    async def fill(self, selector: str, value: str) -> None:
        if selector not in self._present:
            raise RuntimeError(f"no element matching {selector}")
        self.fills.append((selector, value))

    async def click(self, selector: str) -> None:
        self.clicks.append(selector)
        if selector not in self._present:
            raise RuntimeError(f"no element matching {selector}")
        if self.mode == "success":
            self._present = {"#dashboard", "a#logout"}
            self._url = DASHBOARD_URL
            self._body = "Welcome back, tester"
            self._title = "Dashboard"
            self.context.state = {
                "cookies": [
                    {"name": "session", "value": "cookie-abc", "domain": "example.test"}
                ],
                "origins": [],
            }
        elif self.mode == "fail":
            self._body = "Invalid username or password"

    async def wait_for_timeout(self, milliseconds: int) -> None:
        self.waits.append(milliseconds)

    async def query_selector(self, selector: str) -> object | None:
        self.queries.append(selector)
        return object() if selector in self._present else None

    async def inner_text(self, selector: str) -> str:
        return self._body

    async def title(self) -> str:
        return self._title


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
@pytest.fixture
def manager(environment) -> Any:
    reset_vault()
    reset_session_manager()
    yield get_session_manager()
    reset_session_manager()
    reset_vault()


def make_profile(**overrides: Any) -> SiteProfile:
    data: dict[str, Any] = {
        "alias": "ipnr",
        "login_url": LOGIN_URL,
        "username": "tester@example.test",
        "password": PASSWORD,
        "username_selector": "input[name=user]",
        "password_selector": "input[type=password]",
        "submit_selector": "button[type=submit]",
        "success_selector": "#dashboard",
        "success_text": "",
    }
    data.update(overrides)
    return SiteProfile(**data)


def log_blob(caplog: pytest.LogCaptureFixture) -> str:
    """Everything a handler could possibly emit: message plus every extra."""
    parts: list[str] = []
    for record in caplog.records:
        parts.append(str(record.getMessage()))
        for key, value in record.__dict__.items():
            parts.append(f"{key}={value!r}")
    return " || ".join(parts)


# --------------------------------------------------------------------------- #
# Profiles
# --------------------------------------------------------------------------- #
async def test_save_profile_roundtrips_every_field(manager: SiteSessionManager) -> None:
    profile = make_profile(success_text="Welcome back")
    await manager.save_profile(profile)

    loaded = await manager.get_profile("ipnr")
    assert loaded is not None
    for name in SiteProfile.__dataclass_fields__:
        assert getattr(loaded, name) == getattr(profile, name), name


async def test_get_profile_is_case_insensitive_and_missing_returns_none(
    manager: SiteSessionManager,
) -> None:
    await manager.save_profile(make_profile())
    assert (await manager.get_profile("IPNR")) is not None
    assert (await manager.get_profile("nope")) is None


async def test_profile_is_encrypted_at_rest(manager: SiteSessionManager) -> None:
    from app.db import repo
    from app.db.base import session_scope

    await manager.save_profile(make_profile())
    async with session_scope() as session:
        rows = await repo.list_credentials(session)

    assert rows, "credentials row was not written"
    assert all(PASSWORD not in row.value for row in rows)


async def test_list_profiles_returns_aliases_and_never_a_password(
    manager: SiteSessionManager,
) -> None:
    await manager.save_profile(make_profile(alias="ipnr"))
    await manager.save_profile(make_profile(alias="bank", login_url="https://bank.test/login"))

    aliases = await manager.list_profiles()
    assert aliases == ["bank", "ipnr"]
    assert PASSWORD not in json.dumps(aliases)


async def test_public_dict_drops_the_password(manager: SiteSessionManager) -> None:
    public = make_profile().to_public_dict()
    assert "password" not in public
    assert PASSWORD not in json.dumps(public)
    assert PASSWORD not in repr(make_profile())


async def test_save_profile_rejects_bad_input(manager: SiteSessionManager) -> None:
    with pytest.raises(WebSessionError):
        await manager.save_profile(make_profile(alias="  "))
    with pytest.raises(WebSessionError):
        await manager.save_profile(make_profile(login_url="ftp://portal.test"))


# --------------------------------------------------------------------------- #
# Login
# --------------------------------------------------------------------------- #
async def test_successful_login_fills_submits_and_persists_state(
    manager: SiteSessionManager,
) -> None:
    await manager.save_profile(make_profile())
    page = FakePage("success")

    result = await manager.login("ipnr", page)

    assert result["logged_in"] is True
    assert result["reused_session"] is False
    assert result["url"] == DASHBOARD_URL
    assert page.navigations == [LOGIN_URL]
    assert ("input[name=user]", "tester@example.test") in page.fills
    assert ("input[type=password]", PASSWORD) in page.fills
    assert page.clicks == ["button[type=submit]"]

    saved = await manager.load_state("ipnr")
    assert saved is not None
    assert saved["cookies"][0]["name"] == "session"


async def test_auto_detection_picks_the_password_field(manager: SiteSessionManager) -> None:
    """Blank selectors must be resolved from the page itself."""
    await manager.save_profile(
        make_profile(
            username_selector="",
            password_selector="",
            submit_selector="",
            success_selector="",
            success_text="Welcome back",
        )
    )
    page = FakePage("success", fields=("input[type=email]", "input[type=password]", "button"))

    result = await manager.login("ipnr", page)

    assert result["logged_in"] is True
    filled = dict(page.fills)
    assert filled["input[type=password]"] == PASSWORD
    assert filled["input[type=email]"] == "tester@example.test"
    assert page.clicks == ["button"]
    assert "input[type=password]" in AUTO_PASSWORD_SELECTORS


async def test_auto_detection_failure_is_a_clear_error(manager: SiteSessionManager) -> None:
    await manager.save_profile(
        make_profile(username_selector="", password_selector="", success_selector="#dashboard")
    )
    page = FakePage("fail", fields=("div#nothing-useful",))

    with pytest.raises(WebSessionError, match="username"):
        await manager.login("ipnr", page)


async def test_failed_login_raises_without_the_password(
    manager: SiteSessionManager, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    await manager.save_profile(make_profile())
    page = FakePage("fail")

    with pytest.raises(WebSessionError) as excinfo:
        await manager.login("ipnr", page)

    message = str(excinfo.value)
    assert "ipnr" in message
    assert PASSWORD not in message
    assert PASSWORD not in repr(excinfo.value.args)
    assert PASSWORD not in log_blob(caplog)
    assert (await manager.load_state("ipnr")) is None


async def test_login_without_a_profile_raises(manager: SiteSessionManager) -> None:
    with pytest.raises(WebSessionError, match="no saved credentials"):
        await manager.login("unknown", FakePage("success"))


async def test_existing_session_is_reused_without_retyping_credentials(
    manager: SiteSessionManager,
) -> None:
    await manager.save_profile(make_profile())
    await manager.save_state("ipnr", {"cookies": [{"name": "session", "value": "old"}]})
    page = FakePage("already")

    result = await manager.login("ipnr", page)

    assert result["reused_session"] is True
    assert result["logged_in"] is True
    assert page.fills == []
    assert page.clicks == []
    assert page.navigations == [LOGIN_URL]


async def test_verify_logged_in(manager: SiteSessionManager) -> None:
    await manager.save_profile(make_profile())
    assert await manager.verify_logged_in("ipnr", FakePage("already")) is True
    assert await manager.verify_logged_in("ipnr", FakePage("fail")) is False
    assert await manager.verify_logged_in("missing", FakePage("already")) is False


# --------------------------------------------------------------------------- #
# Cookie persistence
# --------------------------------------------------------------------------- #
async def test_state_survives_a_save_load_roundtrip(manager: SiteSessionManager) -> None:
    state = {
        "cookies": [{"name": "sid", "value": "xyz", "domain": "example.test", "path": "/"}],
        "origins": [{"origin": "https://portal.example.test", "localStorage": []}],
    }
    await manager.save_state("ipnr", state)
    assert await manager.load_state("ipnr") == state


async def test_state_survives_a_restart(manager: SiteSessionManager) -> None:
    await manager.save_profile(make_profile())
    await manager.login("ipnr", FakePage("success"))

    reset_session_manager()
    reset_vault()
    fresh = get_session_manager()

    assert (await fresh.get_profile("ipnr")) is not None
    restored = await fresh.load_state("ipnr")
    assert restored is not None and restored["cookies"][0]["value"] == "cookie-abc"


async def test_clear_state_and_missing_state(manager: SiteSessionManager) -> None:
    await manager.save_state("ipnr", {"cookies": []})
    assert await manager.clear_state("ipnr") is True
    assert await manager.clear_state("ipnr") is False
    assert (await manager.load_state("ipnr")) is None


async def test_delete_profile_removes_credentials_and_cookies(
    manager: SiteSessionManager,
) -> None:
    await manager.save_profile(make_profile())
    await manager.login("ipnr", FakePage("success"))
    assert (await manager.load_state("ipnr")) is not None

    assert await manager.delete_profile("ipnr") is True

    assert (await manager.get_profile("ipnr")) is None
    assert (await manager.load_state("ipnr")) is None
    assert await manager.list_profiles() == []
    assert await manager.delete_profile("ipnr") is False


# --------------------------------------------------------------------------- #
# Leak guards
# --------------------------------------------------------------------------- #
async def test_password_never_appears_in_logs_for_any_flow(
    manager: SiteSessionManager, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)

    await manager.save_profile(make_profile())
    await manager.login("ipnr", FakePage("success"))
    await manager.login("ipnr", FakePage("already"))
    await manager.verify_logged_in("ipnr", FakePage("already"))
    with pytest.raises(WebSessionError):
        await manager.login("ipnr", FakePage("fail"))
    await manager.list_profiles()
    await manager.delete_profile("ipnr")

    blob = log_blob(caplog)
    assert caplog.records, "no log records captured - the assertion would be vacuous"
    assert PASSWORD not in blob
    assert "hunter2" not in blob


async def test_login_result_never_contains_the_password(manager: SiteSessionManager) -> None:
    await manager.save_profile(make_profile())
    result = await manager.login("ipnr", FakePage("success"))
    assert PASSWORD not in json.dumps(result)


# --------------------------------------------------------------------------- #
# Deployment guards
# --------------------------------------------------------------------------- #
async def test_works_under_production_logging_level(
    manager: SiteSessionManager, caplog: pytest.LogCaptureFixture
) -> None:
    """Regression guard: app/main.py calls setup_logging("INFO").

    app/security/vault.py logs with extra={"name": ...}, and "name" is a reserved
    LogRecord attribute, so logging raises KeyError at INFO level.  A full save ->
    login -> delete cycle must still work (and still not leak the password).
    """
    caplog.set_level(logging.INFO)

    await manager.save_profile(make_profile())
    assert (await manager.get_profile("ipnr")) is not None

    result = await manager.login("ipnr", FakePage("success"))
    assert result["logged_in"] is True
    assert (await manager.load_state("ipnr")) is not None

    assert await manager.delete_profile("ipnr") is True
    assert (await manager.get_profile("ipnr")) is None
    assert (await manager.load_state("ipnr")) is None
    assert PASSWORD not in log_blob(caplog)


def test_module_imports_without_playwright(monkeypatch: pytest.MonkeyPatch) -> None:
    """The VPS image may be built without browsers; importing must still work."""
    monkeypatch.setitem(sys.modules, "playwright", None)
    monkeypatch.setitem(sys.modules, "playwright.async_api", None)

    module = importlib.reload(importlib.import_module("app.integrations.web_session"))

    assert module.SiteSessionManager is not None
    assert not any(name.startswith("playwright") for name in vars(module))


def test_singleton_and_reset() -> None:
    reset_session_manager()
    first = get_session_manager()
    assert get_session_manager() is first
    reset_session_manager()
    assert get_session_manager() is not first
