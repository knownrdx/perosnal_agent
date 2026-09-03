"""Browser tools (Playwright, optional).

Disabled by default (ENABLE_BROWSER_TOOLS=false).  API-first rule: only use
these when no official API exists.  One shared browser context per process,
closed on shutdown.  Downloads land in the workspace like any other file.
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.config import get_settings
from app.logging_conf import get_logger
from app.security import Permission, UnsafePath, file_info, safe_path
from app.tools.base import Arg, InvalidInput, PermanentToolError, TemporaryToolError
from app.tools.registry import tool

log = get_logger(__name__)

_settings = get_settings()
BROWSER_ENABLED = _settings.enable_browser_tools

_playwright = None
_browser = None
_context = None
_page = None
_lock = asyncio.Lock()


async def _ensure_page():
    """Lazily start Chromium and return the shared page."""
    global _playwright, _browser, _context, _page
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:  # pragma: no cover - depends on build flag
        raise PermanentToolError(
            "playwright is not installed; rebuild the image with INSTALL_BROWSER=true"
        ) from exc

    if _page is not None and not _page.is_closed():
        return _page

    _playwright = await async_playwright().start()
    _browser = await _playwright.chromium.launch(headless=True, args=["--no-sandbox"])
    _context = await _browser.new_context(accept_downloads=True)
    _context.set_default_timeout(45_000)
    _page = await _context.new_page()
    log.info("browser_started", extra={"tool": "browser"})
    return _page


async def shutdown_browser() -> None:
    global _playwright, _browser, _context, _page
    try:
        if _context is not None:
            await _context.close()
        if _browser is not None:
            await _browser.close()
        if _playwright is not None:
            await _playwright.stop()
    except Exception:  # noqa: BLE001 - shutdown must not raise
        pass
    _playwright = _browser = _context = _page = None


def _check_url(url: str) -> None:
    if not url.lower().startswith(("http://", "https://")):
        raise InvalidInput("only http(s) URLs are supported")


@tool(
    "browser_open",
    description="Open a URL in the headless browser and return the page title and visible text.",
    permission=Permission.WRITE,
    args={
        "url": Arg("string", True, "http(s) URL"),
        "max_chars": Arg("integer", False, "Max characters of page text", default=4000),
    },
    timeout_s=120,
    max_retries=1,
    enabled=BROWSER_ENABLED,
)
async def browser_open(url: str, max_chars: int = 4000) -> dict[str, Any]:
    _check_url(url)
    async with _lock:
        page = await _ensure_page()
        try:
            response = await page.goto(url, wait_until="domcontentloaded")
        except Exception as exc:  # noqa: BLE001
            raise TemporaryToolError(f"navigation failed: {exc}") from exc
        text = await page.inner_text("body")
        return {
            "url": page.url,
            "status": response.status if response else None,
            "title": await page.title(),
            "text": text[:max_chars],
        }


@tool(
    "browser_click",
    description="Click an element on the current page by CSS selector or visible text.",
    permission=Permission.WRITE,
    args={
        "selector": Arg("string", True, "CSS selector, or text=Some Label"),
        "wait_ms": Arg("integer", False, "Wait after clicking", default=1000),
    },
    timeout_s=90,
    max_retries=1,
    enabled=BROWSER_ENABLED,
)
async def browser_click(selector: str, wait_ms: int = 1000) -> dict[str, Any]:
    async with _lock:
        page = await _ensure_page()
        try:
            await page.click(selector)
        except Exception as exc:  # noqa: BLE001
            raise TemporaryToolError(f"click failed for {selector!r}: {exc}") from exc
        await page.wait_for_timeout(max(0, min(wait_ms, 15000)))
        return {"clicked": True, "selector": selector, "url": page.url, "title": await page.title()}


@tool(
    "browser_type",
    description="Type text into an input on the current page.",
    permission=Permission.WRITE,
    args={
        "selector": Arg("string", True, "CSS selector of the input"),
        "text": Arg("string", True, "Text to type"),
        "press_enter": Arg("boolean", False, "Press Enter afterwards", default=False),
    },
    timeout_s=90,
    max_retries=1,
    enabled=BROWSER_ENABLED,
)
async def browser_type(selector: str, text: str, press_enter: bool = False) -> dict[str, Any]:
    async with _lock:
        page = await _ensure_page()
        try:
            await page.fill(selector, text)
            if press_enter:
                await page.press(selector, "Enter")
                await page.wait_for_load_state("domcontentloaded")
        except Exception as exc:  # noqa: BLE001
            raise TemporaryToolError(f"type failed for {selector!r}: {exc}") from exc
        return {"typed": True, "selector": selector, "url": page.url}


@tool(
    "browser_read",
    description="Read the current page text without navigating.",
    permission=Permission.READ,
    args={
        "selector": Arg("string", False, "Optional CSS selector to scope the read", default="body"),
        "max_chars": Arg("integer", False, "Max characters", default=4000),
    },
    timeout_s=60,
    max_retries=1,
    enabled=BROWSER_ENABLED,
)
async def browser_read(selector: str = "body", max_chars: int = 4000) -> dict[str, Any]:
    async with _lock:
        page = await _ensure_page()
        text = await page.inner_text(selector)
        return {"url": page.url, "title": await page.title(), "text": text[:max_chars]}


@tool(
    "browser_download",
    description="Click an element that triggers a download and save the file into the workspace.",
    permission=Permission.WRITE,
    args={
        "selector": Arg("string", True, "Selector of the download trigger"),
        "path": Arg("string", False, "Destination workspace path", default=""),
        "timeout_s": Arg("integer", False, "How long to wait for the download", default=120),
    },
    timeout_s=300,
    max_retries=1,
    enabled=BROWSER_ENABLED,
    verify=lambda data: bool(data.get("size_bytes", 0) > 0),
)
async def browser_download(selector: str, path: str = "", timeout_s: int = 120) -> dict[str, Any]:
    async with _lock:
        page = await _ensure_page()
        try:
            async with page.expect_download(timeout=max(5, min(timeout_s, 600)) * 1000) as info:
                await page.click(selector)
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
        return {"downloaded": True, "url": download.url, **file_info(target)}


@tool(
    "browser_upload",
    description="Attach a workspace file to a file input on the current page.",
    permission=Permission.WRITE,
    args={
        "selector": Arg("string", True, "Selector of the <input type=file>"),
        "path": Arg("string", True, "Workspace file to upload"),
    },
    timeout_s=180,
    max_retries=1,
    enabled=BROWSER_ENABLED,
)
async def browser_upload(selector: str, path: str) -> dict[str, Any]:
    try:
        source = safe_path(path, must_exist=True)
    except UnsafePath as exc:
        raise InvalidInput(str(exc)) from exc
    async with _lock:
        page = await _ensure_page()
        try:
            await page.set_input_files(selector, str(source))
        except Exception as exc:  # noqa: BLE001
            raise TemporaryToolError(f"upload failed: {exc}") from exc
        return {"uploaded": True, "path": str(source.name), "selector": selector}
