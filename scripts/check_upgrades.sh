#!/usr/bin/env bash
# Verify the professional upgrades against the LIVE deployed agent.
set -uo pipefail
cd /opt/ai-agent

docker compose exec -T agent python - <<'PY'
import asyncio

from app.agent.planner import make_plan, verify_completion
from app.db import repo
from app.db.base import create_all, init_engine, session_scope
from app.llm import get_llm
from app.security import Permission, autonomy

checks = []


def check(name, ok, detail=""):
    checks.append((name, ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


async def main() -> int:
    init_engine()
    await create_all()
    llm = get_llm()

    # --- autonomy: routine work no longer interrupts ------------------- #
    print("\n== autonomy ==")
    v = await autonomy.evaluate("file_delete", {"path": "temp/scratch.txt"})
    check("scratch file: acts alone", v.allow, v.reason[:60])

    v = await autonomy.evaluate(
        "file_delete", {"path": "output/report.pdf"}, user_request="summarise my week"
    )
    check("unrequested delete: still asks", not v.allow, v.reason[:60])

    v = await autonomy.evaluate(
        "file_delete",
        {"path": "uploads/my_credentials.json"},
        user_request="delete my_credentials.json",
    )
    check("credentials: never auto", not v.allow, v.reason[:60])

    # --- autonomy learns ------------------------------------------------ #
    sig_args = {"path": "output/probe-live.csv"}
    async with session_scope() as session:
        await repo.forget_approval_pattern(session, autonomy.signature("file_delete", sig_args))
    for _ in range(3):
        await autonomy.remember_decision("file_delete", sig_args, approved=True)
    v = await autonomy.evaluate("file_delete", {"path": "output/probe-other.csv"})
    check("learns after 3 approvals", v.allow and v.learned, v.reason[:60])

    await autonomy.remember_decision("file_delete", sig_args, approved=False)
    v = await autonomy.evaluate("file_delete", {"path": "output/probe-other.csv"})
    check("one rejection revokes trust", not v.allow, v.reason[:60])

    async with session_scope() as session:
        await repo.forget_approval_pattern(session, autonomy.signature("file_delete", sig_args))

    # --- planning with the REAL model ----------------------------------- #
    print("\n== planning (real model) ==")
    plan = await make_plan(
        "download the monthly sales csv, check it is not empty, and send it to me",
        permission=Permission.WRITE,
        llm=llm,
    )
    check("builds a plan", len(plan.steps) >= 2, f"{len(plan.steps)} steps")
    for i, step in enumerate(plan.steps[:5], 1):
        print(f"        {i}. {step[:90]}")
    if plan.success_criteria:
        print(f"        done when: {plan.success_criteria[:90]}")

    simple = await make_plan("hi", permission=Permission.READ, llm=llm)
    check("skips planning for trivia", simple.is_empty)

    # --- self-verification with the REAL model -------------------------- #
    print("\n== self-verification (real model) ==")
    bad = await verify_completion(
        task_id="live-probe",
        user_request="download the report and send it to me on Telegram",
        claim="I downloaded the report and sent it to you.",
        trace=['[step 1] {"tool": "http_get", "tool_result": {"ok": true}}'],
        output_files=["downloads/report.pdf"],
        llm=llm,
    )
    check("catches an unsupported claim", not bad.verified, bad.reason[:70])

    good = await verify_completion(
        task_id="live-probe",
        user_request="download the report and send it to me on Telegram",
        claim="Downloaded and sent.",
        trace=[
            '[step 1] {"tool": "http_get", "tool_result": {"ok": true}}',
            '[step 2] {"tool": "telegram_send_file", "tool_result": {"ok": true, "message_id": 55}}',
        ],
        output_files=["downloads/report.pdf"],
        llm=llm,
    )
    check("accepts a supported claim", good.verified, good.reason[:70])

    honest = await verify_completion(
        task_id="live-probe",
        user_request="download the report and send it to me",
        claim="I could not reach the server, so nothing was sent.",
        trace=['[step 1] {"tool": "http_get", "tool_result": {"ok": false}}'],
        output_files=[],
        llm=llm,
    )
    check("accepts honest failure", honest.verified, honest.reason[:70])

    failed = [n for n, ok in checks if not ok]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
        return 1
    print("LIVE UPGRADE OK")
    return 0


raise SystemExit(asyncio.run(main()))
PY
