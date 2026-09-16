"""Persistent scheduler.

Polls the ``scheduled_jobs`` table and materialises a normal Task whenever a
job is due, then computes the next fire time.  All state is in the database, so
timers survive restarts (master prompt section 17).
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import timedelta

from app.config import get_settings
from app.db import repo
from app.db.base import session_scope
from app.db.models import JobKind, utcnow
from app.logging_conf import get_logger
from app.scheduler.timeparse import TimeParseError, next_cron

log = get_logger(__name__)


class SchedulerRunner:
    def __init__(self, notifier: object | None = None) -> None:
        settings = get_settings()
        self.settings = settings
        self.interval = settings.scheduler_poll_interval_s
        self.notifier = notifier
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._next_briefing = None
        self._next_otp_check = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._loop())
        log.info("scheduler_start", extra={"interval": self.interval})

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
        self._task = None
        log.info("scheduler_stopped")

    async def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                fired = await self.tick()
                if fired:
                    log.info("scheduler_fired", extra={"count": fired})
                await self.maybe_send_briefing()
                await self.maybe_run_otp_automation()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - scheduler must never die
                log.exception("scheduler_error")
            await asyncio.sleep(self.interval)

    async def maybe_send_briefing(self) -> bool:
        """Send the daily briefing when its cron time has passed.

        Deliberately not a normal scheduled job: the briefing is a message, not
        a task, so routing it through the task queue would clutter the owner's
        task list with an entry every single day.
        """
        settings = get_settings()
        if not settings.briefing_enabled or self.notifier is None:
            return False

        now = utcnow()
        if self._next_briefing is None:
            # First tick after boot only arms the timer; it must not fire
            # immediately, or a restart would spam the owner.
            try:
                self._next_briefing = next_cron(settings.briefing_cron, base=now)
            except TimeParseError as exc:
                log.error("briefing_bad_cron", extra={"error": str(exc)[:200]})
                self._next_briefing = None
            return False

        if now < self._next_briefing:
            return False

        try:
            self._next_briefing = next_cron(settings.briefing_cron, base=now)
        except TimeParseError:
            self._next_briefing = None

        try:
            from app.agent.briefing import daily_briefing
            from app.llm import get_llm

            text = await daily_briefing(
                chat_id=settings.owner_chat_id,
                period_hours=settings.briefing_period_hours,
                llm=get_llm(),
            )
            await self.notifier.send(
                settings.owner_chat_id,
                text[:4000],
                dedupe_key=f"briefing:{now.strftime('%Y-%m-%d-%H')}",
            )
        except Exception as exc:  # noqa: BLE001 - a bad briefing must not stop the scheduler
            log.error("briefing_failed", extra={"error": str(exc)[:250]})
            return False

        log.info("briefing_sent")
        return True

    async def tick(self) -> int:
        """Fire all due jobs. Returns how many tasks were created."""
        fired = 0
        async with session_scope() as session:
            jobs = await repo.due_jobs(session, limit=20)
            for job in jobs:
                task = await repo.create_task(
                    session,
                    user_request=job.instruction,
                    title=job.name or job.instruction[:80],
                    chat_id=job.chat_id,
                    user_id=job.user_id,
                    permission="WRITE",
                    max_steps=self.settings.max_task_steps,
                    max_retries=self.settings.max_task_retries,
                    scheduled_job_id=job.id,
                    context={"source": "scheduler", "job_id": job.id},
                )
                fired += 1

                runs = job.runs + 1
                values: dict = {"runs": runs, "last_run_at": utcnow()}

                if job.max_runs is not None and runs >= job.max_runs:
                    values.update(enabled=False, next_run_at=None)
                elif job.kind == JobKind.ONCE.value:
                    values.update(enabled=False, next_run_at=None)
                elif job.kind == JobKind.INTERVAL.value and job.interval_s:
                    values["next_run_at"] = utcnow() + timedelta(seconds=job.interval_s)
                elif job.kind == JobKind.CRON.value and job.cron_expr:
                    try:
                        values["next_run_at"] = next_cron(job.cron_expr)
                    except TimeParseError as exc:
                        log.error("scheduler_bad_cron", extra={"job_id": job.id, "error": str(exc)})
                        values.update(enabled=False, next_run_at=None)
                else:
                    values.update(enabled=False, next_run_at=None)

                await repo.update_job(session, job.id, **values)
                await repo.log_event(
                    session,
                    "scheduler_job_fired",
                    task_id=task.id,
                    data={"job_id": job.id, "kind": job.kind},
                )
        return fired

    async def maybe_run_otp_automation(self) -> bool:
        """Check-and-refill the OTP-number distribution bot on its own timer.

        Deliberately LLM-free (see app/automation/otp_bot.py's module
        docstring) so it keeps working even when the configured LLM provider
        is slow, rate-limited, or down entirely - all of which have happened
        in production. Configured entirely from the web dashboard's
        Automation panel / /otpbot Telegram command; no code change needed to
        turn it on, change the interval, or point it at a different bot.
        """
        from app.automation import otp_bot

        config = await otp_bot.get_config()
        if not config.get("enabled"):
            self._next_otp_check = None
            return False

        now = utcnow()
        interval = timedelta(minutes=max(1, int(config.get("interval_minutes", 10))))
        if self._next_otp_check is None:
            # First tick after boot/enable only arms the timer - it must not
            # fire immediately on every restart.
            self._next_otp_check = now + interval
            return False
        if now < self._next_otp_check:
            return False
        self._next_otp_check = now + interval

        result = await otp_bot.run_cycle(config)
        log.info(
            "otp_automation_cycle",
            extra={"ok": result.ok, "action": result.action, "active_quota": result.active_quota},
        )

        if self.notifier is None or self.settings.owner_chat_id is None:
            return result.ok

        if not result.ok:
            await self.notifier.send(
                self.settings.owner_chat_id,
                f"\u26A0\uFE0F OTP-bot automation failed: {result.error[:300]}",
                dedupe_key=f"otp_automation_error:{result.ran_at}",
            )
        elif result.action == "added":
            await self.notifier.send(
                self.settings.owner_chat_id,
                f"\U0001F504 Quota was exhausted - cleaned up and re-added numbers to "
                f"{config['target_bot']}.\n\n{result.add_reply[:500]}",
                dedupe_key=f"otp_automation_added:{result.ran_at}",
            )
        return result.ok
