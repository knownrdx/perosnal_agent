#!/usr/bin/env python
"""Bridge smoke test: run the real Go bridges and talk to them over HTTP.

Verifies the parts that unit tests mock out: that the compiled binaries boot,
enforce the shared token, reject bad input, and answer the exact JSON shapes
the Python client expects.

    python scripts/bridge_smoke.py            # expects the bridges running
    BRIDGE_TOKEN=... WA_URL=... TEAMS_URL=... python scripts/bridge_smoke.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

WA_URL = os.environ.get("WA_URL", "http://127.0.0.1:8081")
TEAMS_URL = os.environ.get("TEAMS_URL", "http://127.0.0.1:8082")
TOKEN = os.environ.get("BRIDGE_TOKEN", "")

checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


async def probe(client: httpx.AsyncClient, base: str, label: str) -> bool:
    try:
        response = await client.get(f"{base}/health", timeout=5.0)
    except Exception as exc:  # noqa: BLE001
        check(f"{label}: reachable", False, str(exc)[:80])
        return False

    payload = response.json()
    check(f"{label}: /health responds", response.status_code == 200 and payload.get("ok") is True,
          payload.get("service", ""))

    # Auth must be enforced.
    unauth = await client.get(f"{base}/status", timeout=5.0)
    check(f"{label}: rejects missing token", unauth.status_code == 401)

    wrong = await client.get(f"{base}/status", headers={"X-Bridge-Token": "wrong"}, timeout=5.0)
    check(f"{label}: rejects wrong token", wrong.status_code == 401)

    if not TOKEN:
        check(f"{label}: BRIDGE_TOKEN provided", False, "set BRIDGE_TOKEN to test authed calls")
        return True

    good = await client.get(f"{base}/status", headers={"X-Bridge-Token": TOKEN}, timeout=10.0)
    check(f"{label}: accepts valid token", good.status_code == 200, str(good.json())[:90])
    return True


async def main() -> int:
    headers = {"X-Bridge-Token": TOKEN} if TOKEN else {}
    print("\nProbing the real bridge services...\n")

    async with httpx.AsyncClient() as client:
        print("WhatsApp bridge:")
        wa_up = await probe(client, WA_URL, "whatsapp")
        if wa_up and TOKEN:
            bad = await client.post(f"{WA_URL}/send/text", headers=headers,
                                    json={"to": "", "text": "x"}, timeout=10.0)
            check("whatsapp: rejects empty recipient", bad.status_code >= 400,
                  str(bad.json().get("error", ""))[:70])

            escape = await client.post(f"{WA_URL}/send/file", headers=headers,
                                       json={"to": "8801700000000",
                                             "path": "../../etc/passwd"}, timeout=10.0)
            check("whatsapp: blocks path traversal", escape.status_code >= 400,
                  str(escape.json().get("error", ""))[:70])

            msgs = await client.get(f"{WA_URL}/messages?limit=5", headers=headers, timeout=10.0)
            check("whatsapp: /messages returns a list",
                  msgs.status_code == 200 and "messages" in msgs.json())

        print("\nTeams bridge:")
        teams_up = await probe(client, TEAMS_URL, "teams")
        if teams_up and TOKEN:
            bad = await client.post(f"{TEAMS_URL}/send/text", headers=headers,
                                    json={"chat": "", "text": ""}, timeout=10.0)
            check("teams: rejects empty text", bad.status_code >= 400,
                  str(bad.json().get("error", ""))[:70])

            escape = await client.post(f"{TEAMS_URL}/download", headers=headers,
                                       json={"url": "https://graph.microsoft.com/v1.0/x",
                                             "path": "../../etc/shadow"}, timeout=10.0)
            check("teams: blocks path traversal", escape.status_code >= 400,
                  str(escape.json().get("error", ""))[:70])

    failed = [name for name, ok, _ in checks if not ok]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
        return 1
    print("BRIDGE SMOKE OK\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
