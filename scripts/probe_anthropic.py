"""Probe an Anthropic-compatible endpoint before wiring it in.

Usage:
    python scripts/probe_anthropic.py <base_url> <api_key>

Tries the sensible URL shapes and reports which one actually answers, so the
provider can be configured from evidence instead of assumption.
"""

from __future__ import annotations

import asyncio
import json
import sys

import httpx

API_VERSION = "2023-06-01"


def mask(value: str) -> str:
    return f"{value[:8]}...{value[-4:]}" if len(value) > 16 else "***"


async def try_messages(base: str, key: str) -> tuple[bool, str]:
    url = base.rstrip("/") + "/messages"
    payload = {
        "model": "claude-sonnet-4-5",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "Reply with the single word: pong"}],
    }
    headers = {
        "x-api-key": key,
        "anthropic-version": API_VERSION,
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json=payload, headers=headers)
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {str(exc)[:150]}"

    body = response.text[:300]
    if response.status_code != 200:
        return False, f"HTTP {response.status_code}: {body}"

    try:
        data = response.json()
        blocks = data.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        model = data.get("model", "?")
        usage = data.get("usage", {})
        return True, f"model={model} reply={text.strip()[:60]!r} usage={usage}"
    except Exception:  # noqa: BLE001
        return False, f"non-JSON: {body}"


async def try_models(base: str, key: str) -> tuple[bool, str]:
    url = base.rstrip("/") + "/models"
    headers = {"x-api-key": key, "anthropic-version": API_VERSION}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=headers)
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)[:150]
    if response.status_code != 200:
        return False, f"HTTP {response.status_code}"
    try:
        ids = [m.get("id") for m in (response.json().get("data") or [])]
        return True, ", ".join(str(i) for i in ids[:12]) or "(empty list)"
    except Exception:  # noqa: BLE001
        return False, response.text[:150]


async def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    raw_base, key = sys.argv[1].rstrip("/"), sys.argv[2]

    # The Vercel AI SDK appends /messages to baseURL, so the given URL is the
    # prefix. Also try it without a trailing /v1 in case it is doubled.
    candidates = [raw_base]
    if raw_base.endswith("/v1"):
        candidates.append(raw_base[: -len("/v1")])
    else:
        candidates.append(raw_base + "/v1")

    print(f"key: {mask(key)}\n")
    working = None
    for base in candidates:
        print(f"--- POST {base}/messages ---")
        ok, detail = await try_messages(base, key)
        print(f"  {'OK  ' if ok else 'FAIL'} {detail}\n")
        if ok and working is None:
            working = base

    if working:
        print(f"--- GET {working}/models ---")
        ok, detail = await try_models(working, key)
        print(f"  {'OK  ' if ok else 'FAIL'} {detail}\n")
        print(f"USE THIS BASE URL: {working}")
        return 0

    print("No working endpoint shape found.")
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
