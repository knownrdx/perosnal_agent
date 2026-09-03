"""List models on an Anthropic-compatible endpoint, then test the first one.

Usage:
    python scripts/probe_models.py <base_url> <api_key> [model]
"""

from __future__ import annotations

import asyncio
import json
import sys

import httpx

API_VERSION = "2023-06-01"


async def list_models(base: str, key: str) -> list[str]:
    headers = {"x-api-key": key, "anthropic-version": API_VERSION,
               "Authorization": f"Bearer {key}"}
    out: list[str] = []
    async with httpx.AsyncClient(timeout=30.0) as client:
        for path in ("/models", "/v1/models"):
            url = base.rstrip("/") + path
            try:
                response = await client.get(url, headers=headers)
            except Exception as exc:  # noqa: BLE001
                print(f"  {url} -> {type(exc).__name__}")
                continue
            print(f"  GET {url} -> HTTP {response.status_code}")
            if response.status_code != 200:
                print(f"    {response.text[:200]}")
                continue
            try:
                payload = response.json()
            except Exception:  # noqa: BLE001
                print(f"    non-JSON: {response.text[:150]}")
                continue
            items = payload.get("data") or payload.get("models") or []
            for item in items:
                name = item.get("id") if isinstance(item, dict) else str(item)
                if name:
                    out.append(str(name))
            if out:
                return out
    return out


async def test_model(base: str, key: str, model: str) -> None:
    url = base.rstrip("/") + "/messages"
    payload = {
        "model": model,
        "max_tokens": 24,
        "messages": [{"role": "user", "content": "Reply with the single word: pong"}],
    }
    headers = {
        "x-api-key": key,
        "anthropic-version": API_VERSION,
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=90.0) as client:
        response = await client.post(url, json=payload, headers=headers)
    print(f"  POST {url} model={model} -> HTTP {response.status_code}")
    if response.status_code != 200:
        print(f"    {response.text[:250]}")
        return
    data = response.json()
    blocks = data.get("content") or []
    text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    print(f"    OK  model={data.get('model')}  reply={text.strip()[:80]!r}")
    print(f"    usage={data.get('usage')}")


async def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    base, key = sys.argv[1].rstrip("/"), sys.argv[2]
    wanted = sys.argv[3] if len(sys.argv) > 3 else ""

    print("=== models ===")
    models = await list_models(base, key)
    if models:
        for name in models[:40]:
            print(f"    {name}")
    else:
        print("    (could not list models)")

    print("\n=== chat test ===")
    candidates = [wanted] if wanted else []
    candidates += [m for m in models if "claude" in m.lower()][:3]
    candidates += models[:2]
    seen = set()
    for model in [m for m in candidates if m and not (m in seen or seen.add(m))][:4]:
        await test_model(base, key, model)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
