"""Website login + session manager (credentials never reach the LLM).

WHY THIS EXISTS
---------------
The owner wants to say "log into site X and download the invoice".  With only
the raw browser tools that would mean the model itself typing the password into
``browser_type`` - which puts the secret into the prompt, the conversation
history, the tool-call log and any provider that sees them.  That is
unacceptable for a private agent.

So the login is performed *here* instead:

  * credentials live encrypted in the credential vault (``app.security.vault``);
  * the model only ever passes an ``alias`` ("ipnr"), never a secret;
  * the password is read from the vault, typed straight into the page and is
    never returned, never rendered into an exception message and never logged.

Two more things the raw tools cannot do:

  * cookies are persisted per site (Playwright ``storage_state``, stored
    encrypted in the same vault) so a restart does not force a fresh login and
    trip bot detection;
  * every site has its own login form, so the owner configures the selectors
    once per site; blank selectors fall back to auto-detection.

Playwright is deliberately NOT imported here: the VPS image may be built
without it.  The ``page`` object is injected by the caller, which also makes
the whole module testable with a fake page.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Protocol

from app.logging_conf import get_logger
from app.security.vault import get_vault

log = get_logger(__name__)

__all__ = [
    "AUTO_PASSWORD_SELECTORS",
    "AUTO_SUBMIT_SELECTORS",
    "AUTO_USERNAME_SELECTORS",
    "SiteProfile",
    "SiteSessionManager",
    "WebSessionError",
    "get_session_manager",
    "reset_session_manager",
]

MASK = "***"

# Vault key namespaces.  One entry per site for the credentials, one for the
# cookie jar, so deleting a profile can drop both.
CREDS_KEY = "site:{alias}:creds"
STATE_KEY = "site:{alias}:state"

# Tried in order when the profile leaves a selector blank.  Ordered from the
# most specific / least ambiguous to the loosest fallback.
AUTO_USERNAME_SELECTORS: tuple[str, ...] = (
    "input[autocomplete=username]",
    "input[type=email]",
    "input[name=username]",
    "input[name=email]",
    "input[name=user]",
    "input[name=login]",
    "#username",
    "#email",
    "#user",
    "#login",
    "form input[type=text]",
    "input[type=text]",
)

AUTO_PASSWORD_SELECTORS: tuple[str, ...] = (
    "input[type=password]",
    "input[autocomplete=current-password]",
    "input[name=password]",
    "input[name=pass]",
    "#password",
    "#pass",
)

AUTO_SUBMIT_SELECTORS: tuple[str, ...] = (
    "button[type=submit]",
    "input[type=submit]",
    "button[name=login]",
    "button[id=login]",
    "#login-button",
    "#submit",
    "form button",
    "button",
)

POST_SUBMIT_WAIT_MS = 1500
SUCCESS_POLL_ATTEMPTS = 3
SUCCESS_POLL_MS = 750


class PageLike(Protocol):
    """The only page surface this module is allowed to touch.

    Implementations may be sync or async (real Playwright is async, the test
    fake can be either) - every call goes through :func:`_resolve`.
    """

    url: str
    context: Any

    def goto(self, url: str) -> Any: ...
    def fill(self, selector: str, value: str) -> Any: ...
    def click(self, selector: str) -> Any: ...
    def wait_for_timeout(self, milliseconds: int) -> Any: ...
    def query_selector(self, selector: str) -> Any: ...
    def inner_text(self, selector: str) -> Any: ...
    def title(self) -> Any: ...


class WebSessionError(Exception):
    """A site login or session operation failed. Never carries the password."""


def _scrub(text: str, secret: str) -> str:
    """Belt-and-braces: strip the password out of anything we surface."""
    if secret and secret in text:
        return text.replace(secret, MASK)
    return text


async def _resolve(value: Any) -> Any:
    """Await ``value`` when the page implementation is async, else pass through."""
    if inspect.isawaitable(value):
        return await value
    return value


def _is_logrecord_clash(exc: BaseException) -> bool:
    """True for the vault's ``extra={"name": ...}`` logging bug.

    ``app.security.vault`` logs ``credential_set`` / ``credential_deleted`` with
    ``extra={"name": ...}``, but ``name`` is a reserved ``LogRecord`` attribute, so
    ``logging`` raises ``KeyError`` once the level is INFO or lower - which is what
    ``setup_logging("INFO")`` uses in production.  The raise happens *after* the row is
    committed and the cache updated, so the write itself did land.  Rather than losing
    every site profile to a log line, tolerate exactly this failure and verify the
    result instead.  Fixing vault.py is the real remedy.
    """
    return isinstance(exc, KeyError) and "LogRecord" in str(exc)


async def _vault_set(vault: Any, name: str, value: str) -> None:
    try:
        await vault.set(name, value)
    except Exception as exc:  # noqa: BLE001
        if not _is_logrecord_clash(exc):
            raise
        if vault.get(name) != value:  # the write really did fail
            raise


async def _vault_delete(vault: Any, name: str) -> bool:
    try:
        return bool(await vault.delete(name))
    except Exception as exc:  # noqa: BLE001
        if not _is_logrecord_clash(exc):
            raise
        # The vault only reaches that log line when the row was actually removed.
        return not vault.has(name)


@dataclass
class SiteProfile:
    """Everything needed to log into one site.

    ``password`` is ``repr=False`` so an accidental ``repr(profile)`` in a log
    line or traceback cannot leak it.
    """

    alias: str
    login_url: str
    username: str
    password: str = field(repr=False)
    username_selector: str = ""
    password_selector: str = ""
    submit_selector: str = ""
    success_selector: str = ""
    success_text: str = ""

    def to_public_dict(self) -> dict[str, Any]:
        """Safe to hand to the model, to the API and to logs: no password."""
        data = asdict(self)
        data.pop("password", None)
        return data

    def _to_vault_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def _from_vault_json(cls, raw: str) -> SiteProfile:
        data = json.loads(raw)
        allowed = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in allowed})


class SiteSessionManager:
    """Stores per-site credentials + cookies and performs the login itself."""

    def __init__(self) -> None:
        self._vault_loaded = False

    # ------------------------------------------------------------------ #
    # Vault plumbing
    # ------------------------------------------------------------------ #
    async def _vault(self) -> Any:
        vault = get_vault()
        if not self._vault_loaded:
            await vault.load()
            self._vault_loaded = True
        return vault

    @staticmethod
    def _norm(alias: str) -> str:
        # The vault lowercases credential names, so normalise here too or
        # list_profiles() and get_profile() would disagree about the alias.
        return alias.strip().lower()

    # ------------------------------------------------------------------ #
    # Profiles
    # ------------------------------------------------------------------ #
    async def save_profile(self, profile: SiteProfile) -> None:
        alias = self._norm(profile.alias)
        if not alias:
            raise WebSessionError("site alias must not be empty")
        if not profile.login_url.lower().startswith(("http://", "https://")):
            raise WebSessionError("login_url must be an http(s) URL")
        if not profile.username or not profile.password:
            raise WebSessionError(f"site '{alias}' needs both a username and a password")

        stored = SiteProfile(**{**asdict(profile), "alias": alias})
        vault = await self._vault()
        await _vault_set(vault, CREDS_KEY.format(alias=alias), stored._to_vault_json())
        log.info("site_profile_saved", extra={"alias": alias})

    async def get_profile(self, alias: str) -> SiteProfile | None:
        alias = self._norm(alias)
        if not alias:
            return None
        vault = await self._vault()
        raw = vault.get(CREDS_KEY.format(alias=alias))
        if not raw:
            return None
        try:
            return SiteProfile._from_vault_json(raw)
        except (ValueError, TypeError) as exc:
            # Never echo `raw` - it contains the password.
            log.warning(
                "site_profile_corrupt",
                extra={"alias": alias, "error": type(exc).__name__},
            )
            return None

    async def list_profiles(self) -> list[str]:
        """Aliases only - deliberately returns no credential material."""
        vault = await self._vault()
        aliases: list[str] = []
        for name in vault.names():
            if name.startswith("site:") and name.endswith(":creds"):
                aliases.append(name[len("site:") : -len(":creds")])
        return sorted(aliases)

    async def delete_profile(self, alias: str) -> bool:
        alias = self._norm(alias)
        if not alias:
            return False
        vault = await self._vault()
        removed = await _vault_delete(vault, CREDS_KEY.format(alias=alias))
        # Stale cookies for a deleted site would be both useless and a risk.
        await _vault_delete(vault, STATE_KEY.format(alias=alias))
        if removed:
            log.info("site_profile_deleted", extra={"alias": alias})
        return bool(removed)

    # ------------------------------------------------------------------ #
    # Cookie jar (Playwright storage_state)
    # ------------------------------------------------------------------ #
    async def save_state(self, alias: str, storage_state: dict[str, Any]) -> None:
        alias = self._norm(alias)
        if not alias:
            raise WebSessionError("site alias must not be empty")
        vault = await self._vault()
        payload = json.dumps(storage_state or {}, ensure_ascii=False, default=str)
        await _vault_set(vault, STATE_KEY.format(alias=alias), payload)
        cookies = (storage_state or {}).get("cookies") or []
        log.info("site_state_saved", extra={"alias": alias, "cookies": len(cookies)})

    async def load_state(self, alias: str) -> dict[str, Any] | None:
        alias = self._norm(alias)
        if not alias:
            return None
        vault = await self._vault()
        raw = vault.get(STATE_KEY.format(alias=alias))
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except ValueError:
            log.warning("site_state_corrupt", extra={"alias": alias})
            return None
        return data if isinstance(data, dict) else None

    async def clear_state(self, alias: str) -> bool:
        alias = self._norm(alias)
        if not alias:
            return False
        vault = await self._vault()
        removed = await _vault_delete(vault, STATE_KEY.format(alias=alias))
        if removed:
            log.info("site_state_cleared", extra={"alias": alias})
        return bool(removed)

    # ------------------------------------------------------------------ #
    # Page helpers - only the PageLike surface is used
    # ------------------------------------------------------------------ #
    async def _first_present(self, page: PageLike, selectors: Iterable[str]) -> str | None:
        for selector in selectors:
            try:
                found = await _resolve(page.query_selector(selector))
            except Exception:  # noqa: BLE001 - a bad selector must not abort detection
                continue
            if found is not None:
                return selector
        return None

    async def _resolve_selector(
        self, page: PageLike, configured: str, candidates: Iterable[str]
    ) -> str | None:
        if configured:
            return configured
        return await self._first_present(page, candidates)

    async def _success_check(self, profile: SiteProfile, page: PageLike) -> bool:
        if profile.success_selector:
            return (await _resolve(page.query_selector(profile.success_selector))) is not None
        if profile.success_text:
            text = await _resolve(page.inner_text("body"))
            return profile.success_text.lower() in str(text or "").lower()
        # Nothing explicit configured: the login form disappearing is the best
        # generic evidence that the credentials were accepted.
        return (await self._first_present(page, AUTO_PASSWORD_SELECTORS)) is None

    async def _wait_for_success(self, profile: SiteProfile, page: PageLike) -> bool:
        for attempt in range(SUCCESS_POLL_ATTEMPTS):
            try:
                if await self._success_check(profile, page):
                    return True
            except Exception:  # noqa: BLE001 - mid-navigation errors are transient
                pass
            if attempt < SUCCESS_POLL_ATTEMPTS - 1:
                await _resolve(page.wait_for_timeout(SUCCESS_POLL_MS))
        return False

    async def _restore_cookies(
        self, alias: str, page: PageLike, state: dict[str, Any]
    ) -> bool:
        """Best effort: real Playwright contexts expose add_cookies, fakes may not."""
        context = getattr(page, "context", None)
        add_cookies = getattr(context, "add_cookies", None)
        cookies = state.get("cookies") or []
        if context is None or add_cookies is None or not cookies:
            return False
        try:
            await _resolve(add_cookies(cookies))
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "site_cookie_restore_failed",
                extra={"alias": alias, "error": type(exc).__name__},
            )
            return False
        return True

    async def _capture_state(self, alias: str, page: PageLike) -> dict[str, Any] | None:
        context = getattr(page, "context", None)
        storage_state = getattr(context, "storage_state", None)
        if context is None or storage_state is None:
            return None
        try:
            state = await _resolve(storage_state())
        except Exception as exc:  # noqa: BLE001 - losing cookies must not fail the login
            log.warning(
                "site_state_capture_failed",
                extra={"alias": alias, "error": type(exc).__name__},
            )
            return None
        return state if isinstance(state, dict) else None

    # ------------------------------------------------------------------ #
    # The actual login
    # ------------------------------------------------------------------ #
    async def login(self, alias: str, page: PageLike) -> dict[str, Any]:
        """Log into ``alias`` using ``page``. The password never leaves this method."""
        alias = self._norm(alias)
        profile = await self.get_profile(alias)
        if profile is None:
            raise WebSessionError(
                f"no saved credentials for site '{alias}'; add the site profile first"
            )
        secret = profile.password

        state = await self.load_state(alias)
        if state:
            await self._restore_cookies(alias, page, state)

        try:
            await _resolve(page.goto(profile.login_url))
        except Exception as exc:  # noqa: BLE001
            detail = _scrub(str(exc), secret)[:200]
            raise WebSessionError(
                f"could not open the login page for site '{alias}': {detail}"
            ) from None

        if await self._success_check(profile, page):
            url = _scrub(str(getattr(page, "url", "")), secret)
            log.info("site_login_reused", extra={"alias": alias})
            return {"logged_in": True, "reused_session": True, "alias": alias, "url": url}

        username_selector = await self._resolve_selector(
            page, profile.username_selector, AUTO_USERNAME_SELECTORS
        )
        password_selector = await self._resolve_selector(
            page, profile.password_selector, AUTO_PASSWORD_SELECTORS
        )
        if not username_selector or not password_selector:
            missing = "username" if not username_selector else "password"
            raise WebSessionError(
                f"could not find the {missing} field for site '{alias}'; "
                "set the selector on the site profile"
            )

        await self._fill(alias, page, username_selector, profile.username, secret)
        await self._fill(alias, page, password_selector, secret, secret)

        submit_selector = await self._resolve_selector(
            page, profile.submit_selector, AUTO_SUBMIT_SELECTORS
        )
        if not submit_selector:
            raise WebSessionError(
                f"could not find the submit button for site '{alias}'; "
                "set submit_selector on the site profile"
            )
        try:
            await _resolve(page.click(submit_selector))
        except Exception as exc:  # noqa: BLE001
            detail = _scrub(str(exc), secret)[:200]
            raise WebSessionError(
                f"could not submit the login form for site '{alias}': {detail}"
            ) from None

        await _resolve(page.wait_for_timeout(POST_SUBMIT_WAIT_MS))

        if not await self._wait_for_success(profile, page):
            url = _scrub(str(getattr(page, "url", "")), secret)
            log.warning("site_login_failed", extra={"alias": alias})
            raise WebSessionError(
                f"login failed for site '{alias}': the success check did not pass "
                f"after submitting the form (url={url})"
            )

        captured = await self._capture_state(alias, page)
        if captured is not None:
            await self.save_state(alias, captured)

        url = _scrub(str(getattr(page, "url", "")), secret)
        title = ""
        try:
            title = _scrub(str(await _resolve(page.title()) or ""), secret)
        except Exception:  # noqa: BLE001 - the title is a nicety, not a result
            title = ""
        log.info("site_login_ok", extra={"alias": alias, "reused_session": False})
        return {
            "logged_in": True,
            "reused_session": False,
            "alias": alias,
            "url": url,
            "title": title,
        }

    async def _fill(
        self, alias: str, page: PageLike, selector: str, value: str, secret: str
    ) -> None:
        try:
            await _resolve(page.fill(selector, value))
        except Exception as exc:  # noqa: BLE001
            # `exc` could in theory echo the typed value back at us.
            detail = _scrub(str(exc), secret)[:200]
            raise WebSessionError(
                f"could not fill '{selector}' for site '{alias}': {detail}"
            ) from None

    async def verify_logged_in(self, alias: str, page: PageLike) -> bool:
        profile = await self.get_profile(alias)
        if profile is None:
            return False
        try:
            return await self._success_check(profile, page)
        except Exception:  # noqa: BLE001 - a broken page just means "not logged in"
            return False


_manager: SiteSessionManager | None = None


def get_session_manager() -> SiteSessionManager:
    global _manager
    if _manager is None:
        _manager = SiteSessionManager()
    return _manager


def reset_session_manager() -> None:
    global _manager
    _manager = None
