"""Logging into a website and acting as the owner.

The existing browser tools can click and type, but the model would have to type
the owner's password to use them - which puts the password in the prompt, the
history and the logs. That is not acceptable for a credential the owner reuses.

These tools close that gap: the login itself is performed by
``app.integrations.web_session``, which reads the password straight from the
encrypted vault. The model only ever names a site alias. It never sees, sends,
or is able to read the password.

Typical use, entirely from Telegram:

    /site add ipnr https://portal.example.com myuser mypass
    "log into ipnr and download this month's statement"
"""

from __future__ import annotations

from typing import Any

from app.config import get_settings
from app.integrations.web_session import WebSessionError, get_session_manager
from app.logging_conf import get_logger
from app.security import Permission, UnsafePath, file_info, safe_path
from app.tools.base import Arg, InvalidInput, PermanentToolError, TemporaryToolError
from app.tools.registry import tool

log = get_logger(__name__)

BROWSER_ENABLED = get_settings().enable_browser_tools


async def _page():
    """The shared browser page, started on demand."""
    from app.tools.browser_tools import _ensure_page

    return await _ensure_page()


async def _persist_cookies(alias: str, page) -> None:
    """Save the session so the next task does not log in again.

    Repeated logins are the fastest way to trip a site's bot detection, so this
    is a reliability measure as much as a speed one.
    """
    try:
        state = await page.context.storage_state()
        await get_session_manager().save_state(alias, state)
    except Exception as exc:  # noqa: BLE001 - never fail a task over a cookie save
        log.warning("cookie_save_failed", extra={"alias": alias, "error": str(exc)[:200]})


@tool(
    "site_login",
    description=(
        "Log into a saved website using stored credentials and keep the session open. "
        "Use the site alias the owner configured; you never need the password."
    ),
    permission=Permission.WRITE,
    args={"alias": Arg("string", True, "Saved site alias, e.g. 'ipnr'")},
    timeout_s=180,
    max_retries=1,
    enabled=BROWSER_ENABLED,
)
async def site_login(alias: str) -> dict[str, Any]:
    manager = get_session_manager()
    profile = await manager.get_profile(alias)
    if profile is None:
        known = await manager.list_profiles()
        raise InvalidInput(
            f"no saved site called {alias!r}. "
            f"Known sites: {', '.join(known) if known else '(none yet)'}. "
            "The owner adds one with /site add."
        )

    page = await _page()
    try:
        result = await manager.login(alias, page)
    except WebSessionError as exc:
        # The message is written by web_session and never contains the password.
        raise PermanentToolError(str(exc)) from exc

    log.info(
        "site_logged_in",
        extra={"alias": alias, "reused": result.get("reused_session", False)},
    )
    return {
        "alias": alias,
        "logged_in": True,
        "reused_session": result.get("reused_session", False),
        "url": result.get("url", ""),
        "title": result.get("title", ""),
    }


@tool(
    "site_navigate",
    description="Go to a URL inside an already logged-in site session and read the page text.",
    permission=Permission.WRITE,
    args={
        "alias": Arg("string", True, "Saved site alias"),
        "url": Arg("string", True, "URL to open (must belong to that site)"),
        "max_chars": Arg("integer", False, "Max characters of page text", default=4000),
    },
    timeout_s=180,
    max_retries=1,
    enabled=BROWSER_ENABLED,
)
async def site_navigate(alias: str, url: str, max_chars: int = 4000) -> dict[str, Any]:
    if not url.lower().startswith(("http://", "https://")):
        raise InvalidInput("only http(s) URLs are supported")

    manager = get_session_manager()
    if await manager.get_profile(alias) is None:
        raise InvalidInput(f"no saved site called {alias!r}")

    page = await _page()
    try:
        response = await page.goto(url, wait_until="domcontentloaded")
        text = await page.inner_text("body")
        title = await page.title()
    except Exception as exc:  # noqa: BLE001
        raise TemporaryToolError(f"navigation failed: {exc}") from exc

    await _persist_cookies(alias, page)
    return {
        "alias": alias,
        "url": page.url,
        "status": getattr(response, "status", None),
        "title": title,
        "text": text[: max(200, min(max_chars, 20000))],
    }


@tool(
    "site_download",
    description=(
        "Download a file from a logged-in site, either by clicking a link/button "
        "or by opening a direct file URL. Saves it into the workspace."
    ),
    permission=Permission.WRITE,
    args={
        "alias": Arg("string", True, "Saved site alias"),
        "selector": Arg("string", False, "CSS selector or text=Label of the download trigger",
                        default=""),
        "url": Arg("string", False, "Direct file URL instead of a selector", default=""),
        "path": Arg("string", False, "Destination workspace path", default=""),
        "timeout_s": Arg("integer", False, "How long to wait for the download", default=120),
    },
    timeout_s=300,
    max_retries=1,
    enabled=BROWSER_ENABLED,
    # A download only counts if bytes actually arrived.
    verify=lambda data: bool(data.get("size_bytes", 0) > 0),
)
async def site_download(
    alias: str,
    selector: str = "",
    url: str = "",
    path: str = "",
    timeout_s: int = 120,
) -> dict[str, Any]:
    if not selector and not url:
        raise InvalidInput("provide either a selector to click or a direct file url")

    manager = get_session_manager()
    if await manager.get_profile(alias) is None:
        raise InvalidInput(f"no saved site called {alias!r}")

    page = await _page()
    wait_ms = max(5, min(timeout_s, 600)) * 1000

    try:
        async with page.expect_download(timeout=wait_ms) as info:
            if selector:
                await page.click(selector)
            else:
                # Navigating to a file URL raises "Download is starting" in
                # Playwright; the download event is what we actually want.
                try:
                    await page.goto(url)
                except Exception:  # noqa: BLE001 - expected for direct downloads
                    pass
        download = await info.value
    except Exception as exc:  # noqa: BLE001
        raise TemporaryToolError(f"download did not start: {exc}") from exc

    destination = path or f"downloads/{download.suggested_filename}"
    try:
        target = safe_path(destination)
    except UnsafePath as exc:
        raise InvalidInput(str(exc)) from exc
    target.parent.mkdir(parents=True, exist_ok=True)
    await download.save_as(str(target))

    await _persist_cookies(alias, page)
    log.info("site_download", extra={"alias": alias, "file": target.name})
    return {"alias": alias, "downloaded": True, "source_url": download.url, **file_info(target)}


@tool(
    "site_action",
    description=(
        "Perform a click or fill on a logged-in site page. "
        "Never use this for passwords: credentials are handled by site_login."
    ),
    permission=Permission.WRITE,
    args={
        "alias": Arg("string", True, "Saved site alias"),
        "action": Arg("string", True, "click | fill | select", choices=["click", "fill", "select"]),
        "selector": Arg("string", True, "CSS selector or text=Label"),
        "value": Arg("string", False, "Value for fill/select", default=""),
        "wait_ms": Arg("integer", False, "Wait after acting", default=1000),
    },
    timeout_s=120,
    max_retries=1,
    enabled=BROWSER_ENABLED,
)
async def site_action(
    alias: str,
    action: str,
    selector: str,
    value: str = "",
    wait_ms: int = 1000,
) -> dict[str, Any]:
    manager = get_session_manager()
    if await manager.get_profile(alias) is None:
        raise InvalidInput(f"no saved site called {alias!r}")

    verb = action.strip().lower()
    if verb not in {"click", "fill", "select"}:
        raise InvalidInput("action must be one of: click, fill, select")
    if verb in {"fill", "select"} and not value:
        raise InvalidInput(f"{verb} requires a value")

    # Defence in depth: the model should never route a secret through here, and
    # a password field is a strong signal that something has gone wrong.
    if verb == "fill" and "password" in selector.lower():
        raise InvalidInput(
            "refusing to type into a password field; use site_login, which reads "
            "the stored credential without exposing it"
        )

    page = await _page()
    try:
        if verb == "click":
            await page.click(selector)
        elif verb == "fill":
            await page.fill(selector, value)
        else:
            await page.select_option(selector, value)
        await page.wait_for_timeout(max(0, min(wait_ms, 15000)))
    except Exception as exc:  # noqa: BLE001
        raise TemporaryToolError(f"{verb} failed for {selector!r}: {exc}") from exc

    await _persist_cookies(alias, page)
    return {"alias": alias, "action": verb, "selector": selector, "url": page.url}


@tool(
    "site_list",
    description="List the website aliases the owner has configured for automated login.",
    permission=Permission.READ,
    args={},
    timeout_s=30,
    max_retries=1,
    enabled=BROWSER_ENABLED,
)
async def site_list() -> dict[str, Any]:
    aliases = await get_session_manager().list_profiles()
    return {"sites": aliases, "count": len(aliases)}
