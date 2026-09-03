#!/usr/bin/env bash
# Verify the new capability layer against the LIVE deployed agent.
set -uo pipefail
cd /opt/ai-agent

docker compose exec -T agent python - <<'PY'
import asyncio

checks = []


def check(name, ok, detail=""):
    checks.append((name, ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


async def main() -> int:
    from app.db.base import create_all, init_engine
    from app.tools import registry

    init_engine()
    await create_all()

    # --- registration -------------------------------------------------- #
    print("\n== tools ==")
    names = set(registry.names())
    print(f"  {len(names)} tools registered")
    for group, expected in [
        ("web", {"web_search", "web_fetch", "http_get", "http_request"}),
        ("document", {"document_read", "document_info", "document_search",
                      "spreadsheet_query"}),
        ("email", {"email_list", "email_read", "email_search", "email_send",
                   "email_accounts"}),
        ("site", {"site_login", "site_navigate", "site_download", "site_action",
                  "site_list"}),
    ]:
        missing = expected - names
        check(f"{group} tools registered", not missing,
              f"missing: {missing}" if missing else f"{len(expected)} tools")

    # --- SSRF guard: the security property that matters ---------------- #
    print("\n== ssrf guard ==")
    from app.tools.base import PermanentToolError
    from app.tools.web_tools import http_get

    for target in ("http://127.0.0.1/", "http://192.168.1.1/",
                   "http://169.254.169.254/latest/meta-data/"):
        try:
            await http_get(url=target)
            check(f"blocks {target[:38]}", False, "NOT BLOCKED")
        except PermanentToolError:
            check(f"blocks {target[:38]}", True)
        except Exception as exc:  # noqa: BLE001
            check(f"blocks {target[:38]}", False, type(exc).__name__)

    # --- real web search ------------------------------------------------ #
    print("\n== web search (real network) ==")
    from app.tools.web_tools import web_search

    try:
        found = await web_search(query="mikrotik hotspot", limit=3)
        results = found.get("results", [])
        check("web_search returns results", bool(results), f"{len(results)} hits")
        for item in results[:2]:
            print(f"        {str(item.get('title'))[:70]}")
    except Exception as exc:  # noqa: BLE001
        check("web_search returns results", False, str(exc)[:90])

    # --- documents ------------------------------------------------------ #
    print("\n== documents ==")
    from app.security import safe_path
    from app.tools.document_tools import document_read, spreadsheet_query

    csv_path = safe_path("temp/live_probe.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("name,amount\nrouter,1500\nswitch,2300\n", encoding="utf-8")
    try:
        doc = document_read(path="temp/live_probe.csv")
        check("reads a csv", "router" in doc.get("text", ""))
        sheet = spreadsheet_query(path="temp/live_probe.csv")
        check("spreadsheet_query parses rows", sheet.get("row_count") == 2,
              f"headers={sheet.get('headers')}")
    except Exception as exc:  # noqa: BLE001
        check("document tools work", False, str(exc)[:90])
    finally:
        csv_path.unlink(missing_ok=True)

    # --- traversal still blocked ---------------------------------------- #
    from app.tools.base import InvalidInput

    try:
        document_read(path="../../etc/passwd")
        check("blocks path traversal", False, "NOT BLOCKED")
    except (InvalidInput, PermanentToolError):
        check("blocks path traversal", True)
    except Exception as exc:  # noqa: BLE001
        check("blocks path traversal", False, type(exc).__name__)

    # --- website session: password must never surface ------------------- #
    print("\n== site login (secret safety) ==")
    from app.integrations.web_session import SiteProfile, get_session_manager

    manager = get_session_manager()
    secret = "sup3r-s3cret-live-probe"
    await manager.save_profile(
        SiteProfile(alias="liveprobe", login_url="https://example.com/login",
                    username="probe", password=secret)
    )
    listed = await manager.list_profiles()
    check("profile saved", "liveprobe" in listed)
    check("password absent from listing", secret not in str(listed))

    loaded = await manager.get_profile("liveprobe")
    check("password readable only from the vault", loaded.password == secret)
    await manager.delete_profile("liveprobe")
    check("profile deleted", "liveprobe" not in await manager.list_profiles())

    # --- email accounts -------------------------------------------------- #
    print("\n== email ==")
    from app.tools.email_tools import email_accounts

    try:
        accounts = await email_accounts()
        check("email_accounts callable", "accounts" in accounts,
              f"{accounts.get('count', 0)} configured")
    except Exception as exc:  # noqa: BLE001
        check("email_accounts callable", False, str(exc)[:90])

    # --- briefing --------------------------------------------------------- #
    print("\n== briefing ==")
    from app.agent.briefing import daily_briefing

    try:
        text = await daily_briefing(period_hours=24)
        check("briefing generated", bool(text.strip()), f"{len(text)} chars")
        print(f"        {text.strip().splitlines()[0][:80]}")
    except Exception as exc:  # noqa: BLE001
        check("briefing generated", False, str(exc)[:90])

    # --- transcription health --------------------------------------------- #
    print("\n== voice ==")
    from app.integrations.transcription import get_transcriber

    health = await get_transcriber().health()
    check("transcriber reports status", "ok" in health,
          f"backend={health.get('backend')} ok={health.get('ok')}")

    failed = [n for n, ok in checks if not ok]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
        return 1
    print("LIVE CAPABILITIES OK")
    return 0


raise SystemExit(asyncio.run(main()))
PY
