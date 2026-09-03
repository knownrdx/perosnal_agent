"""Web search and generic HTTP/REST access.

The agent can already touch the filesystem, the database and a handful of
messaging bridges, but it had no way to look something up on the open web or
to call an API that has no dedicated tool.  These four tools close that gap
using only httpx (already a dependency) plus the standard library, so no HTML
parsing or search-client package is added to the image.

Every caller-supplied URL goes through :func:`_guard_url`.  This agent is
driven by an LLM that reads untrusted input (web pages, chat messages), so a
prompt injection could otherwise turn it into a port scanner for the host's
private network or a reader of the cloud metadata service at 169.254.169.254.
Private, loopback and link-local targets are therefore refused unless the
caller explicitly passes ``allow_private=True``.
"""

from __future__ import annotations

import html as html_lib
import ipaddress
import re
import socket
from typing import Any
from urllib.parse import unquote, urlparse

import httpx

from app.logging_conf import get_logger
from app.security import Permission
from app.tools.base import (
    Arg,
    InvalidInput,
    PermanentToolError,
    RateLimited,
    TemporaryToolError,
)
from app.tools.registry import tool

log = get_logger(__name__)

DUCKDUCKGO_HTML_URL = "https://html.duckduckgo.com/html/"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)

ALLOWED_METHODS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE"})
MAX_BODY_CHARS = 20_000
MAX_SEARCH_RESULTS = 25

# Injected by tests (httpx.MockTransport) so the suite never touches the network.
_transport: httpx.AsyncBaseTransport | None = None


def set_transport(transport: httpx.AsyncBaseTransport | None) -> None:
    """Override the httpx transport for every tool in this module."""
    global _transport
    _transport = transport


# --------------------------------------------------------------------------- #
# HTML helpers
# --------------------------------------------------------------------------- #
_ANCHOR_RE = re.compile(r"<a\s(?P<attrs>[^>]*?)>(?P<inner>.*?)</a>", re.I | re.S)
_ATTR_RE_CACHE: dict[str, re.Pattern[str]] = {}
_UDDG_RE = re.compile(r"[?&]uddg=([^&]+)")
_TAG_RE = re.compile(r"<[^>]+>")
_SPACES_RE = re.compile(r"[^\S\n]+")
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)
_H1_RE = re.compile(r"<h1[^>]*>(.*?)</h1>", re.I | re.S)
# Boilerplate containers whose text is never the answer the agent is after.
_DROP_BLOCK_RE = re.compile(
    r"<(script|style|noscript|nav|footer|header|aside|svg|form|template)\b[^>]*>.*?</\1>",
    re.I | re.S,
)
_BREAK_RE = re.compile(
    r"</?(br|p|div|li|tr|h[1-6]|section|article|ul|ol|table|blockquote)\b[^>]*>",
    re.I,
)


def _attr(attrs: str, name: str) -> str:
    pattern = _ATTR_RE_CACHE.get(name)
    if pattern is None:
        pattern = re.compile(name + r"""\s*=\s*(?:"([^"]*)"|'([^']*)')""", re.I)
        _ATTR_RE_CACHE[name] = pattern
    match = pattern.search(attrs)
    if not match:
        return ""
    return match.group(1) if match.group(1) is not None else match.group(2)


def _clean_inline(fragment: str) -> str:
    """Tags out, entities decoded, whitespace collapsed onto a single line."""
    text = html_lib.unescape(_TAG_RE.sub(" ", fragment or ""))
    return " ".join(text.split())


def _readable_text(page: str) -> str:
    body = _DROP_BLOCK_RE.sub(" ", page or "")
    body = _BREAK_RE.sub("\n", body)
    body = html_lib.unescape(_TAG_RE.sub(" ", body))
    lines = (_SPACES_RE.sub(" ", line).strip() for line in body.split("\n"))
    return "\n".join(line for line in lines if line)


def _page_title(page: str) -> str:
    for pattern in (_TITLE_RE, _H1_RE):
        match = pattern.search(page or "")
        if match:
            title = _clean_inline(match.group(1))
            if title:
                return title
    return ""


def _unwrap_ddg_url(href: str) -> str:
    """Turn a DuckDuckGo ``/l/?uddg=`` redirect wrapper into the real target."""
    raw = html_lib.unescape((href or "").strip())
    if not raw:
        return ""
    if raw.startswith("//"):
        raw = "https:" + raw
    wrapped = _UDDG_RE.search(raw)
    if wrapped:
        # Single decode only: parse_qs would decode twice and corrupt any
        # target URL that legitimately contains a percent sign.
        return unquote(wrapped.group(1))
    return raw if raw.lower().startswith(("http://", "https://")) else ""


def _parse_ddg_results(page: str, limit: int) -> list[dict[str, str]]:
    """Scan the HTML endpoint's anchors; they never nest, so one pass is safe."""
    results: list[dict[str, str]] = []
    for match in _ANCHOR_RE.finditer(page or ""):
        classes = _attr(match.group("attrs"), "class")
        inner = match.group("inner")
        if "result__a" in classes:
            if len(results) >= limit:
                break
            url = _unwrap_ddg_url(_attr(match.group("attrs"), "href"))
            title = _clean_inline(inner)
            if url and title:
                results.append({"title": title, "url": url, "snippet": ""})
        elif "result__snippet" in classes and results and not results[-1]["snippet"]:
            results[-1]["snippet"] = _clean_inline(inner)
    return results


# --------------------------------------------------------------------------- #
# SSRF guard
# --------------------------------------------------------------------------- #
def _resolve_addresses(host: str) -> list[str]:
    """Resolve a hostname to IP strings (separate function so tests can stub DNS)."""
    infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    return [str(info[4][0]) for info in infos]


def _as_ip(value: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(value.split("%")[0])
    except ValueError:
        return None


def _is_blocked_ip(raw: str) -> bool:
    ip = _as_ip(raw)
    if ip is None:
        return True  # unparsable address: fail closed
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def _guard_url(url: str, *, allow_private: bool = False) -> str:
    """Validate scheme and refuse internal network targets (SSRF protection)."""
    candidate = (url or "").strip()
    parsed = urlparse(candidate)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise InvalidInput(
            f"only http(s) URLs are supported, got {scheme or 'no'} scheme: {candidate[:120]}"
        )
    host = parsed.hostname
    if not host:
        raise InvalidInput(f"URL has no host: {candidate[:120]}")
    if allow_private:
        return candidate

    if _as_ip(host) is not None:
        addresses = [host]
    else:
        try:
            addresses = _resolve_addresses(host)
        except OSError as exc:
            raise TemporaryToolError(f"cannot resolve host {host}: {exc}") from exc
    blocked = sorted({addr for addr in addresses if _is_blocked_ip(addr)})
    if blocked:
        raise PermanentToolError(
            f"refusing to request {host} because it resolves to a private or "
            f"loopback address ({', '.join(blocked)}); pass allow_private=true "
            "only for a host you trust"
        )
    return candidate


# --------------------------------------------------------------------------- #
# HTTP plumbing
# --------------------------------------------------------------------------- #
def _client(timeout_s: float) -> httpx.AsyncClient:
    total = max(1.0, float(timeout_s))
    return httpx.AsyncClient(
        follow_redirects=True,
        timeout=httpx.Timeout(total, connect=min(15.0, total)),
        headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
        transport=_transport,
    )


def _raise_for_status(status: int, url: str) -> None:
    """Map HTTP status onto the framework's retry classification."""
    if status == 429:
        raise RateLimited(f"rate limited (HTTP 429) by {url}")
    if status >= 500:
        raise TemporaryToolError(f"server error (HTTP {status}) from {url}")
    if status in {401, 403}:
        raise PermanentToolError(f"access denied (HTTP {status}) by {url}")
    if status >= 400:
        raise PermanentToolError(f"request failed with HTTP {status}: {url}")


async def _send(
    method: str,
    url: str,
    *,
    headers: dict[str, Any] | None = None,
    json_body: Any = None,
    params: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
    timeout_s: float = 30,
    allow_private: bool = False,
) -> httpx.Response:
    target = _guard_url(url, allow_private=allow_private)
    try:
        async with _client(timeout_s) as client:
            response = await client.request(
                method,
                target,
                headers={str(k): str(v) for k, v in (headers or {}).items()} or None,
                json=json_body,
                params=params or None,
                data=data,
            )
    except httpx.TimeoutException as exc:
        raise TemporaryToolError(f"timed out after {timeout_s}s requesting {target}") from exc
    except httpx.HTTPError as exc:
        raise TemporaryToolError(f"network error requesting {target}: {exc}") from exc
    _raise_for_status(response.status_code, target)
    return response


def _body_payload(response: httpx.Response) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": response.status_code,
        "ok": 200 <= response.status_code < 400,
        "url": str(response.url),
        "headers": {key.lower(): value for key, value in response.headers.items()},
    }
    if "json" in response.headers.get("content-type", "").lower():
        try:
            payload["json"] = response.json()
            return payload
        except ValueError:
            pass  # mislabelled content type: fall back to raw text
    text = response.text
    payload["text"] = text[:MAX_BODY_CHARS]
    payload["truncated"] = len(text) > MAX_BODY_CHARS
    return payload


async def _http_call(
    method: str,
    url: str,
    headers: dict[str, Any] | None = None,
    json_body: Any = None,
    params: dict[str, Any] | None = None,
    timeout_s: int = 30,
    allow_private: bool = False,
) -> dict[str, Any]:
    verb = str(method or "").strip().upper()
    if verb not in ALLOWED_METHODS:
        allowed = ", ".join(sorted(ALLOWED_METHODS))
        raise InvalidInput(f"unsupported HTTP method '{method}'; allowed: {allowed}")
    if headers is not None and not isinstance(headers, dict):
        raise InvalidInput("headers must be an object of header name -> value")
    if params is not None and not isinstance(params, dict):
        raise InvalidInput("params must be an object of query name -> value")

    response = await _send(
        verb,
        url,
        headers=headers,
        json_body=json_body,
        params=params,
        timeout_s=timeout_s,
        allow_private=allow_private,
    )
    return _body_payload(response)


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
@tool(
    "web_search",
    description=(
        "Search the public web via DuckDuckGo and return ranked results with "
        "title, URL and snippet. Use this to find pages, facts or documentation."
    ),
    permission=Permission.READ,
    args={
        "query": Arg("string", True, "What to search for, as plain search terms"),
        "limit": Arg("integer", False, "Maximum number of results", default=5),
    },
    timeout_s=60,
    max_retries=2,
)
async def web_search(query: str, limit: int = 5) -> dict[str, Any]:
    term = (query or "").strip()
    if not term:
        raise InvalidInput("query must not be empty")
    limit = max(1, min(int(limit), MAX_SEARCH_RESULTS))

    response = await _send(
        "POST",
        DUCKDUCKGO_HTML_URL,
        data={"q": term},
        headers={"Referer": "https://duckduckgo.com/"},
        timeout_s=30,
    )
    results = _parse_ddg_results(response.text, limit)
    log.info("web_search", extra={"tool": "web_search", "count": len(results)})
    if not results:
        # An empty result set is a legitimate answer, not a tool failure.
        return {
            "results": [],
            "count": 0,
            "note": f"no results found for '{term}'; try different or broader terms",
        }
    return {"results": results, "count": len(results)}


@tool(
    "web_fetch",
    description=(
        "Fetch an http(s) web page and return its readable text with navigation, "
        "scripts and styling stripped. Use after web_search to read a result."
    ),
    permission=Permission.READ,
    args={
        "url": Arg("string", True, "Full http(s) URL of the page to read"),
        "max_chars": Arg("integer", False, "Maximum characters of text", default=6000),
        "allow_private": Arg(
            "boolean", False, "Allow private/loopback hosts (only if trusted)", default=False
        ),
    },
    timeout_s=90,
    max_retries=2,
)
async def web_fetch(
    url: str, max_chars: int = 6000, allow_private: bool = False
) -> dict[str, Any]:
    limit = max(200, int(max_chars))
    response = await _send("GET", url, timeout_s=45, allow_private=allow_private)
    page = response.text
    text = _readable_text(page)
    return {
        "url": str(response.url),
        "status": response.status_code,
        "title": _page_title(page),
        "text": text[:limit],
        "truncated": len(text) > limit,
    }


@tool(
    "http_request",
    description=(
        "Call any REST API with GET/POST/PUT/PATCH/DELETE and return the parsed "
        "JSON or text response. Use when no dedicated tool exists for the service."
    ),
    permission=Permission.WRITE,
    args={
        "method": Arg(
            "string", True, "HTTP method: GET, POST, PUT, PATCH or DELETE"
        ),
        "url": Arg("string", True, "Full http(s) URL of the endpoint"),
        "headers": Arg("object", False, "Extra request headers, e.g. Authorization"),
        "json_body": Arg("object", False, "JSON request body for POST/PUT/PATCH"),
        "params": Arg("object", False, "Query string parameters"),
        "timeout_s": Arg("integer", False, "Request timeout in seconds", default=30),
        "allow_private": Arg(
            "boolean", False, "Allow private/loopback hosts (only if trusted)", default=False
        ),
    },
    timeout_s=120,
    max_retries=2,
    # Idempotency is the caller's business here: the same handler serves safe
    # GETs and mutating POSTs, so the framework must not silently replay it.
    side_effect=False,
)
async def http_request(
    method: str,
    url: str,
    headers: dict[str, Any] | None = None,
    json_body: Any = None,
    params: dict[str, Any] | None = None,
    timeout_s: int = 30,
    allow_private: bool = False,
) -> dict[str, Any]:
    return await _http_call(
        method,
        url,
        headers=headers,
        json_body=json_body,
        params=params,
        timeout_s=timeout_s,
        allow_private=allow_private,
    )


@tool(
    "http_get",
    description=(
        "Read-only HTTP GET against a JSON or text API endpoint. Prefer this "
        "over http_request whenever the call only needs to read data."
    ),
    permission=Permission.READ,
    args={
        "url": Arg("string", True, "Full http(s) URL of the endpoint"),
        "headers": Arg("object", False, "Extra request headers, e.g. Authorization"),
        "params": Arg("object", False, "Query string parameters"),
        "timeout_s": Arg("integer", False, "Request timeout in seconds", default=30),
        "allow_private": Arg(
            "boolean", False, "Allow private/loopback hosts (only if trusted)", default=False
        ),
    },
    timeout_s=90,
    max_retries=2,
)
async def http_get(
    url: str,
    headers: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    timeout_s: int = 30,
    allow_private: bool = False,
) -> dict[str, Any]:
    return await _http_call(
        "GET",
        url,
        headers=headers,
        params=params,
        timeout_s=timeout_s,
        allow_private=allow_private,
    )
