#!/usr/bin/env bash
# Verify the Teams device-code sign-in endpoint from INSIDE the compose network.
# The bridge is intentionally not published to the host, so we exec from the
# agent container which shares the network.
set -uo pipefail
cd /opt/ai-agent

B=$(grep '^BRIDGE_TOKEN=' .env | cut -d= -f2)
URL=http://teams-bridge:8082

run() {
  docker compose exec -T agent python - "$1" "$2" "$B" "$URL" <<'PY'
import json, sys, urllib.request

method, path, token, base = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
req = urllib.request.Request(
    base + path, method=method,
    headers={"X-Bridge-Token": token, "Content-Type": "application/json"},
    data=b"" if method == "POST" else None,
)
try:
    with urllib.request.urlopen(req, timeout=40) as resp:
        body = resp.read().decode()
        print(f"http={resp.status}")
        try:
            print(json.dumps(json.loads(body), indent=2)[:600])
        except Exception:
            print(body[:400])
except Exception as exc:  # noqa: BLE001
    print("ERROR:", str(exc)[:300])
PY
}

echo "=== teams bridge auth mode ==="
docker compose logs teams-bridge 2>&1 | grep -iE 'auth mode|sign in' | tail -3

echo
echo "=== POST /login/start (real Microsoft device code) ==="
run POST /login/start

echo
echo "=== POST /login/poll (expect pending) ==="
run POST /login/poll

echo
echo "=== GET /status ==="
run GET /status
