"""Web/HTTP tool behaviour, fully offline.

Every request is served by an httpx.MockTransport and DNS is stubbed, so this
module never opens a socket. The SSRF cases matter most: they are the guard
between an injected prompt and the host's internal network.
"""

from __future__ import annotations

import json
from typing import Any, Callable

import httpx
import pytest

from app.security import Permission
from app.tools import registry
from app.tools.base import (
    InvalidInput,
    PermanentToolError,
    RateLimited,
    TemporaryToolError,
    ToolContext,
)
from app.tools.web_tools import (
    http_get,
    http_request,
    set_transport,
    web_fetch,
    web_search,
)
from app.tools import web_tools

PUBLIC_IP = "93.184.216.34"

_UDDG = "//duckduckgo.com/l/?uddg="
_DOCS_ENCODED = "https%3A%2F%2Fdocs.python.org%2F3%2Flibrary%2Fasyncio.html"
_REALPY_ENCODED = "https%3A%2F%2Freal-python.com%2Fasync-io%2F"

DDG_HTML = """
<html><body>
<div class="results">
  <div class="result results_links">
    <h2 class="result__title">
      <a rel="nofollow" class="result__a" href="{docs}&amp;rut=9f">
        asyncio &mdash; <b>Asynchronous</b> I/O
      </a>
    </h2>
    <a class="result__snippet" href="https://docs.python.org/3/library/asyncio.html">
      <b>asyncio</b> is a library to write concurrent code using the async/await syntax.
    </a>
    <a class="result__url" href="https://docs.python.org">docs.python.org</a>
  </div>
  <div class="result results_links">
    <h2 class="result__title">
      <a rel="nofollow" class="result__a" href="{realpy}&amp;rut=aa">
        Async IO in Python
      </a>
    </h2>
    <a class="result__snippet" href="https://real-python.com/async-io/">
      A complete walkthrough &amp; guide to concurrency.
    </a>
  </div>
  <div class="result results_links">
    <h2 class="result__title">
      <a rel="nofollow" class="result__a" href="https://example.org/plain">Plain link</a>
    </h2>
    <a class="result__snippet" href="https://example.org/plain">No redirect wrapper here.</a>
  </div>
</div>
</body></html>
""".format(docs=_UDDG + _DOCS_ENCODED, realpy=_UDDG + _REALPY_ENCODED)

DDG_NO_RESULTS = """
<html><body>
  <div class="no-results">No results found for that query.</div>
</body></html>
"""

ARTICLE_HTML = """
<html>
  <head>
    <title>Reactor Status &amp; Notes</title>
    <style>body { color: #fff; } .hidden { display: none; }</style>
    <script>window.tracker = "SHOULD_NOT_APPEAR"; alert(1);</script>
  </head>
  <body>
    <nav><a href="/home">HIDDEN_NAV_LINK</a></nav>
    <h1>Reactor Status</h1>
    <p>Coolant flow is <b>nominal</b> at 42&nbsp;L/s.</p>
    <p>Second      paragraph  with     sloppy whitespace.</p>
    <script>console.log("ALSO_HIDDEN");</script>
    <footer>HIDDEN_FOOTER_TEXT</footer>
  </body>
</html>
"""


@pytest.fixture(autouse=True)
def _stub_dns(monkeypatch) -> None:
    """No test may perform a real DNS lookup; IP literals bypass this anyway."""

    def _fake_resolve(host: str) -> list[str]:
        known = {"localhost": ["127.0.0.1"], "internal.corp": ["10.1.2.3"]}
        return known.get(host, [PUBLIC_IP])

    monkeypatch.setattr(web_tools, "_resolve_addresses", _fake_resolve)


@pytest.fixture
def mock_http() -> Any:
    """Install an httpx.MockTransport built from a request handler."""
    installed: list[bool] = []

    def install(handler: Callable[[httpx.Request], httpx.Response]):
        set_transport(httpx.MockTransport(handler))
        installed.append(True)

    yield install
    set_transport(None)


def _responder(body: str, status: int = 200, content_type: str = "text/html"):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text=body, headers={"content-type": content_type})

    return handler


# --------------------------------------------------------------------------- #
# web_search
# --------------------------------------------------------------------------- #
async def test_web_search_parses_duckduckgo_html(environment, mock_http):
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["method"] = request.method
        seen["body"] = request.content.decode()
        return httpx.Response(200, text=DDG_HTML, headers={"content-type": "text/html"})

    mock_http(handler)
    result = await web_search(query="python asyncio", limit=5)

    assert seen["method"] == "POST"
    assert seen["url"] == "https://html.duckduckgo.com/html/"
    assert "q=python+asyncio" in seen["body"]

    assert result["count"] == 3
    first = result["results"][0]
    # &mdash; must be unescaped to a real em dash, and <b> tags dropped.
    assert first["title"] == "asyncio \u2014 Asynchronous I/O"
    assert first["url"] == "https://docs.python.org/3/library/asyncio.html"
    assert "concurrent code" in first["snippet"]
    assert "<b>" not in first["title"] and "<b>" not in first["snippet"]
    assert result["results"][2]["url"] == "https://example.org/plain"


async def test_web_search_decodes_uddg_redirect_wrapper(environment, mock_http):
    wrapped = (
        '<a class="result__a" href="//duckduckgo.com/l/?uddg='
        "https%3A%2F%2Fexample.com%2Fa%2Fb%3Fq%3D1%26x%3D2&amp;rut=zz\">Wrapped</a>"
        '<a class="result__snippet" href="#">snip</a>'
    )
    mock_http(_responder(wrapped))
    result = await web_search(query="anything")

    assert result["results"][0]["url"] == "https://example.com/a/b?q=1&x=2"


async def test_web_search_empty_is_not_an_error(environment, mock_http):
    mock_http(_responder(DDG_NO_RESULTS))
    result = await web_search(query="zzzz no such thing zzzz")

    assert result["results"] == []
    assert result["count"] == 0
    assert result["note"]


async def test_web_search_respects_limit(environment, mock_http):
    mock_http(_responder(DDG_HTML))
    result = await web_search(query="python asyncio", limit=2)

    assert result["count"] == 2
    assert result["results"][1]["snippet"].startswith("A complete walkthrough & guide")


async def test_web_search_rejects_blank_query(environment, mock_http):
    mock_http(_responder(DDG_HTML))
    with pytest.raises(InvalidInput):
        await web_search(query="   ")


async def test_web_search_server_error_is_temporary(environment, mock_http):
    mock_http(_responder("boom", status=503))
    with pytest.raises(TemporaryToolError):
        await web_search(query="python")


# --------------------------------------------------------------------------- #
# web_fetch
# --------------------------------------------------------------------------- #
async def test_web_fetch_strips_script_style_nav_and_footer(environment, mock_http):
    mock_http(_responder(ARTICLE_HTML))
    result = await web_fetch(url="https://example.com/article")

    text = result["text"]
    assert result["status"] == 200
    assert result["title"] == "Reactor Status & Notes"
    assert "Coolant flow is nominal at 42 L/s." in text
    assert "Second paragraph with sloppy whitespace." in text
    hidden_markers = (
        "SHOULD_NOT_APPEAR", "ALSO_HIDDEN", "HIDDEN_NAV_LINK", "HIDDEN_FOOTER_TEXT",
    )
    for hidden in hidden_markers:
        assert hidden not in text
    assert "<" not in text and "&nbsp;" not in text
    assert result["truncated"] is False


async def test_web_fetch_truncates_long_pages(environment, mock_http):
    mock_http(_responder("<html><body><p>" + ("word " * 5000) + "</p></body></html>"))
    result = await web_fetch(url="https://example.com/long", max_chars=500)

    assert len(result["text"]) == 500
    assert result["truncated"] is True


@pytest.mark.parametrize(
    "bad_url",
    ["ftp://example.com/file.txt", "file:///etc/passwd", "javascript:alert(1)", "notaurl"],
)
async def test_web_fetch_rejects_non_http_schemes(environment, mock_http, bad_url):
    mock_http(_responder("should never be reached"))
    with pytest.raises(InvalidInput):
        await web_fetch(url=bad_url)


async def test_web_fetch_404_is_permanent(environment, mock_http):
    mock_http(_responder("nope", status=404))
    with pytest.raises(PermanentToolError):
        await web_fetch(url="https://example.com/missing")


# --------------------------------------------------------------------------- #
# http_request / http_get
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", ["TRACE", "CONNECT", "FETCH", "", "gett"])
async def test_http_request_rejects_unknown_method(environment, mock_http, method):
    mock_http(_responder("{}", content_type="application/json"))
    with pytest.raises(InvalidInput):
        await http_request(method=method, url="https://api.example.com/v1/things")


async def test_http_request_method_is_case_insensitive(environment, mock_http):
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        return httpx.Response(200, json={"ok": True})

    mock_http(handler)
    await http_request(method="post", url="https://api.example.com/v1/things", json_body={})
    assert seen["method"] == "POST"


async def test_http_request_sends_json_headers_and_params(environment, mock_http):
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["auth"] = request.headers.get("authorization")
        seen["query"] = str(request.url.query.decode())
        seen["body"] = json.loads(request.content or b"{}")
        return httpx.Response(201, json={"id": 7, "created": True})

    mock_http(handler)
    result = await http_request(
        method="POST",
        url="https://api.example.com/v1/things",
        headers={"Authorization": "Bearer t0ken"},
        json_body={"name": "widget"},
        params={"dry_run": "false"},
    )

    assert seen["method"] == "POST"
    assert seen["auth"] == "Bearer t0ken"
    assert "dry_run=false" in seen["query"]
    assert seen["body"] == {"name": "widget"}
    assert result["status"] == 201 and result["ok"] is True
    assert result["json"] == {"id": 7, "created": True}
    assert result["headers"]["content-type"].startswith("application/json")


async def test_http_get_returns_text_for_non_json(environment, mock_http):
    mock_http(_responder("plain body", content_type="text/plain"))
    result = await http_get(url="https://api.example.com/health")

    assert result["text"] == "plain body"
    assert "json" not in result
    assert result["ok"] is True


async def test_http_get_falls_back_to_text_on_broken_json(environment, mock_http):
    mock_http(_responder("{not json", content_type="application/json"))
    result = await http_get(url="https://api.example.com/broken")

    assert result["text"] == "{not json"
    assert "json" not in result


# --------------------------------------------------------------------------- #
# SSRF guard
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/admin",
        "http://127.0.0.1:8080/secret",
        "http://192.168.1.1/",
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata service
        "http://10.0.0.5/internal",
        "http://172.16.0.9/internal",
        "http://[::1]:9000/",
        "http://localhost:8000/",
        "http://internal.corp/wiki",  # resolves to a private address
    ],
)
async def test_ssrf_private_targets_are_blocked(environment, mock_http, url):
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, text="leaked")

    mock_http(handler)
    with pytest.raises(PermanentToolError):
        await http_get(url=url)
    assert calls == [], "SSRF guard must reject before any request is sent"


async def test_ssrf_guard_also_covers_http_request_and_web_fetch(environment, mock_http):
    mock_http(_responder("leaked"))
    with pytest.raises(PermanentToolError):
        await http_request(method="POST", url="http://169.254.169.254/latest/api/token")
    with pytest.raises(PermanentToolError):
        await web_fetch(url="http://127.0.0.1:3000/")


async def test_allow_private_permits_localhost(environment, mock_http):
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json={"status": "up"})

    mock_http(handler)
    result = await http_get(url="http://127.0.0.1:8080/health", allow_private=True)

    assert seen == ["http://127.0.0.1:8080/health"]
    assert result["json"] == {"status": "up"}

    result = await http_get(url="http://localhost:8080/health", allow_private=True)
    assert result["ok"] is True


async def test_public_host_is_allowed(environment, mock_http):
    mock_http(_responder("{}", content_type="application/json"))
    result = await http_get(url="https://api.example.com/v1/ping")
    assert result["ok"] is True


# --------------------------------------------------------------------------- #
# Status classification
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "status,expected",
    [
        (500, TemporaryToolError),
        (502, TemporaryToolError),
        (503, TemporaryToolError),
        (429, RateLimited),
        (404, PermanentToolError),
        (400, PermanentToolError),
        (401, PermanentToolError),
        (403, PermanentToolError),
    ],
)
async def test_status_codes_map_to_failure_kinds(environment, mock_http, status, expected):
    mock_http(_responder("body", status=status, content_type="text/plain"))
    with pytest.raises(expected):
        await http_get(url="https://api.example.com/thing")


async def test_rate_limited_is_a_temporary_error(environment, mock_http):
    mock_http(_responder("slow down", status=429, content_type="text/plain"))
    with pytest.raises(TemporaryToolError):  # RateLimited subclasses TemporaryToolError
        await http_get(url="https://api.example.com/thing")


async def test_network_failure_is_temporary(environment, mock_http):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    mock_http(handler)
    with pytest.raises(TemporaryToolError):
        await http_get(url="https://api.example.com/thing")


# --------------------------------------------------------------------------- #
# Registry integration
# --------------------------------------------------------------------------- #
async def test_tools_are_registered_with_expected_permissions(environment):
    expected = {
        "web_search": Permission.READ,
        "web_fetch": Permission.READ,
        "http_request": Permission.WRITE,
        "http_get": Permission.READ,
    }
    for name, permission in expected.items():
        registered = registry.get(name)
        assert registered is not None, f"missing tool: {name}"
        assert registered.permission is permission
        assert registered.description.strip()


async def test_http_get_runs_through_the_framework(environment, mock_http):
    mock_http(_responder('{"pong": true}', content_type="application/json"))
    result = await registry.get("http_get").run(
        {"url": "https://api.example.com/ping"}, ToolContext()
    )

    assert result.ok is True
    assert result.data["json"] == {"pong": True}


async def test_framework_reports_ssrf_block_as_permanent(environment, mock_http):
    from app.db.models import FailureKind

    mock_http(_responder("leaked"))
    result = await registry.get("http_get").run(
        {"url": "http://169.254.169.254/latest/meta-data/"}, ToolContext()
    )

    assert result.ok is False
    assert result.kind is FailureKind.PERMANENT
    assert result.attempts == 1, "a blocked address must not be retried"
