"""Persistent scheduler.

Polls the ``scheduled_jobs`` table and materialises a normal Task whenever a
job is due, then computes the next fire time.  All state is in the database, so
timers survive restarts (master prompt section 17).
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from typing import Any

from app.config import get_settings
from app.db import repo
from app.db.base import session_scope
from app.db.models import JobKind, utcnow
from app.logging_conf import get_logger
from app.scheduler.timeparse import TimeParseError, next_cron

log = get_logger(__name__)

# How often the OTP automation is LOOKED at. The one shared check time (every
# country at once) lives in otp_schedule and run_cycle() answers "not due yet"
# cheaply, so looking every minute costs nothing - whereas pacing on the
# interval made a 30-minute install wait up to 30 minutes after every boot or
# setting change.
OTP_LOOK_EVERY = timedelta(seconds=60)

# Upper bound for one run_cycle(). A cycle legitimately waits up to ~90 s per
# country file and there can be ~15 countries, so 30 minutes only ever trips on
# a call that has genuinely hung - which would otherwise freeze the whole
# scheduler (jobs, briefing and OTP alike) with nothing in the logs.
OTP_CYCLE_TIMEOUT_S = 30 * 60

# While cycles keep failing, remind the owner at most this often. Alerting on
# every failed cycle meant one message a minute while the userbot was
# unlinked, which trains the owner to mute the chat the alerts depend on.
OTP_FAILURE_REMINDER_EVERY = timedelta(minutes=30)


def _human_duration(span: timedelta) -> str:
    """'45 min', '2h', '1h 30m' - short enough for a notification header."""
    minutes = max(0, int(span.total_seconds() // 60))
    if minutes < 60:
        return f"{minutes} min"
    hours, rest = divmod(minutes, 60)
    return f"{hours}h {rest}m" if rest else f"{hours}h"


def _count(value: Any) -> str:
    """A number with thousands separators; '?' when the bot gave none."""
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "?"


def _one_line(text: str, limit: int = 300) -> str:
    return " ".join(str(text or "").split())[:limit]


def _english_reason(reason: str) -> str:
    """otp_schedule.finished_reason() for the owner, always in English.

    The reasons are produced in app/automation, which owns their wording and
    now writes them in English ("2 re-adds done", "23:30 (Dubai) stop time
    reached", "30 min run time reached") - those pass through unchanged. The
    older Banglish shapes ("2 bar re-add shesh", "23:30 (Dubai) time hoye
    geche", "30 min shomoy shesh") can still arrive from a run that finished
    before the switch, so they are rewritten into the same English shapes.
    Anything unrecognised is passed through untouched rather than guessed at.
    """
    text = str(reason or "").strip()
    match = re.fullmatch(r"(\d[\d,]*) bar re-add shesh", text)
    if match:
        n = int(match.group(1).replace(",", ""))
        return f"{n:,} re-add{'' if n == 1 else 's'} done"
    match = re.fullmatch(r"(.+?) time hoye geche", text)
    if match:
        return f"{match.group(1)} stop time reached"
    match = re.fullmatch(r"(\d[\d,]*) min shomoy shesh", text)
    if match:
        minutes = int(match.group(1).replace(",", ""))
        return f"{_human_duration(timedelta(minutes=minutes))} run time reached"
    return text


_ADDED_RE = re.compile(r"(\d[\d,]*)\s+(?:numbers?\s+)?added", re.IGNORECASE)
_DUP_RE = re.compile(r"(\d[\d,]*)\s+(?:duplicates?|dups?)\b", re.IGNORECASE)
_PREFIX_RE = re.compile(r"^([^:\n]{1,60}):\s?(.*)$")


def _refill_bullets(add_reply: str, countries: list[str]) -> list[str]:
    """One '• Country: N added' line per country from run_cycle's add_reply.

    add_reply is "<country>: <raw bot reply>" per country, and a raw reply
    can itself span lines and repeat the country ("Bangladesh: 500 added"),
    so lines are grouped under the country they name. Only known country
    names start a group when we have them, so a bot line like "Status: ok"
    is not mistaken for a country.
    """
    known = {c.lower() for c in countries}
    groups: list[tuple[str, list[str]]] = []
    for line in str(add_reply or "").splitlines():
        match = _PREFIX_RE.match(line.strip())
        name = match.group(1).strip() if match else ""
        if match and (name.lower() in known or (not known and name)):
            if groups and groups[-1][0].lower() == name.lower():
                groups[-1][1].append(match.group(2))
            else:
                groups.append((name, [match.group(2)]))
        elif groups:
            groups[-1][1].append(line)

    bullets: list[str] = []
    for name, body_lines in groups:
        body = "\n".join(body_lines)
        added = _ADDED_RE.search(body)
        if added:
            summary = f"{_count(added.group(1).replace(',', ''))} added"
            dup = _DUP_RE.search(body)
            if dup and int(dup.group(1).replace(",", "")) > 0:
                summary += f", {_count(dup.group(1).replace(',', ''))} duplicates skipped"
        else:
            first = next((ln.strip() for ln in body_lines if ln.strip()), "done")
            summary = _one_line(first, 120)
        bullets.append(f"• {name}: {summary}")
    if not bullets and known:
        # The file's country is spelled differently from the bot's stock
        # list: fall back to treating any "Name: ..." line as a country.
        return _refill_bullets(add_reply, [])
    if not bullets and str(add_reply or "").strip():
        bullets.append(f"• {_one_line(add_reply, 200)}")
    return bullets


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
        # Failure-streak bookkeeping for the OTP alerts. In memory on purpose:
        # after a restart the worst case is one fresh "failed" alert, which is
        # exactly what the owner should see if it is still broken.
        self._otp_failing_since: datetime | None = None
        self._otp_last_failure_alert: datetime | None = None

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
            # Each step is guarded on its own. They used to share one try, so
            # a recurring exception in tick() (one malformed job row) or in
            # the briefing aborted every pass before the OTP automation was
            # reached - the one job that must keep going was silently reduced
            # to a log line.
            await self._guarded("tick", self._tick_and_log)
            await self._guarded("briefing", self.maybe_send_briefing)
            await self._guarded("otp_automation", self.maybe_run_otp_automation)
            await asyncio.sleep(self.interval)

    async def _guarded(self, step: str, run: Callable[[], Awaitable[Any]]) -> None:
        try:
            await run()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - scheduler must never die
            log.exception("scheduler_error", extra={"step": step})

    async def _tick_and_log(self) -> None:
        fired = await self.tick()
        if fired:
            log.info("scheduler_fired", extra={"count": fired})

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

        The one shared check time (every country together - a single /st
        answers for all of them) lives in otp_schedule, so this only paces
        how often we LOOK: every OTP_LOOK_EVERY, starting on the first tick
        after boot. run_cycle() itself is a cheap no-op ("not due yet") until
        that time comes up, and a timer never armed before is armed rather
        than fired (otp_schedule.due_countries), so looking straight away on
        a fresh install does not trigger a burst of refills.

        Every owner-facing message here is plain English with one layout:
        an emoji header, a blank line, one "\u2022" bullet per country, and a
        single next-step line.
        """
        from app.automation import otp_bot

        config = await otp_bot.get_config()
        if not config.get("enabled"):
            self._next_otp_check = None
            # Turned off on purpose: a streak from before is not news later.
            self._otp_failing_since = None
            self._otp_last_failure_alert = None
            return False

        now = utcnow()
        if self._next_otp_check is not None and now < self._next_otp_check:
            return False
        self._next_otp_check = now + OTP_LOOK_EVERY

        result = await self._run_otp_cycle(otp_bot, config)
        if result.error == "no active files - run start() first":
            # Enabled but nothing was ever started successfully - nothing to
            # check yet, not a failure worth notifying about.
            return True
        if result.action == "not due yet":
            # Normal idle state when countries run on staggered timers. Not a
            # recovery either: between two failing cycles the countries are
            # simply not due, and that proves nothing is fixed.
            return True
        log.info(
            "otp_automation_cycle",
            extra={"ok": result.ok, "action": result.action, "active_quota": result.active_quota},
        )

        # Measured after the cycle, which can itself take many minutes.
        await self._report_otp_health(result, utcnow())

        # The per-country notices below do NOT depend on result.ok. ok is
        # False as soon as ONE country's add fails while the rest of the
        # cycle still ran - and by then exhausted_at / finished_at are
        # persisted. Returning early here meant a "file used up" or "run
        # finished" in the same cycle as any failure was never announced at
        # all: a silent, permanent stop. The failure itself was reported
        # just above; these say what else that cycle did.
        if self.notifier is None or self.settings.owner_chat_id is None:
            return result.ok

        # A country whose scheduled start time just arrived has only now sent
        # its numbers for the first time. Said plainly, because from the
        # owner's side the upload happened hours ago and silence since then
        # is indistinguishable from the schedule having been forgotten.
        if result.started_now:
            failed = [i for i in result.started_now if not i.get("added")]
            lines = [
                "\u25B6\uFE0F Scheduled start \u2014 numbers added"
                if len(failed) < len(result.started_now)
                else "\u26A0\uFE0F Scheduled start \u2014 numbers not added",
                "",
            ]
            for item in result.started_now:
                if item.get("added"):
                    lines.append(
                        f"\u2022 {item['country']}: {_count(item.get('count') or 0)} numbers "
                        f"added (tag: {item.get('tag') or 'General'})"
                    )
                else:
                    lines.append(
                        f"\u2022 \u26A0\uFE0F {item['country']}: not added \u2014 "
                        f"{_one_line(item.get('error', ''), 150)}"
                    )
            lines += [
                "",
                "I'll retry the failed ones on the next cycle."
                if failed
                else "From here each country refills on its own schedule.",
            ]
            await self.notifier.send(
                self.settings.owner_chat_id,
                "\n".join(lines),
                dedupe_key=f"otp_started:{result.ran_at}",
            )

        # The start time arrived but the bot still holds this country's
        # stock, so the file was held rather than piled on top. Said plainly:
        # from the owner's side the start time passing in silence would look
        # exactly like the schedule having been forgotten. getattr() because
        # older CycleResults do not carry the field.
        held_at_start = getattr(result, "held_at_start", None) or []
        if held_at_start:
            lines = ["\u23F8\uFE0F Scheduled start \u2014 file held", ""]
            for item in held_at_start:
                lines.append(
                    f"\u2022 {item['country']}: {_count(item.get('stock'))} numbers still in stock"
                )
            lines += ["", "It will be added automatically once stock runs low."]
            await self.notifier.send(
                self.settings.owner_chat_id,
                "\n".join(lines),
                dedupe_key=f"otp_held_at_start:{result.ran_at}",
            )

        # A country that reached its own finish line is a completed run, not
        # a problem - reported separately so it never reads as a failure.
        if result.finished:
            await self.notifier.send(
                self.settings.owner_chat_id,
                self._finished_message(result.finished),
                dedupe_key=f"otp_finished:{result.ran_at}",
            )

        if result.exhausted:
            names = ", ".join(e["country"] for e in result.exhausted)
            # A country whose file is spent needs the owner to act - nothing
            # the automation can do will produce more numbers.
            # "Stock gone" and "file spent but stock still left" are different
            # situations - with a low-stock threshold the second is the normal
            # one, and calling it "out of stock" while the bot still holds
            # numbers would be false and teach the owner to distrust the alert.
            dry = [e for e in result.exhausted if not e.get("had_stock")]
            headline = (
                "\U0001F6A8 Out of stock \u2014 new file needed"
                if dry
                else "\u26A0\uFE0F File used up \u2014 new file needed"
            )
            lines = [headline, ""]
            for item in result.exhausted:
                left = item.get("had_stock") or 0
                tail = f"{_count(left)} still in stock" if left else "stock is at 0"
                lines.append(
                    f"\u2022 {item['country']}: '{item['name']}' has nothing new left "
                    f"(all duplicates) \u2014 {tail}"
                )
            lines += [
                "",
                f"Send a new file here and I'll add it to {config['target_bot']}. "
                f"To drop it instead, say \"remove {result.exhausted[0]['country']}\".",
            ]
            await self.notifier.send(
                self.settings.owner_chat_id,
                "\n".join(lines),
                # Keyed on the countries, not the timestamp: re-alerting every
                # cycle for the same dead file would train the owner to ignore
                # it, but a different country running dry is genuinely new.
                dedupe_key=f"otp_stock_empty:{names}",
            )
        elif result.action.startswith("added"):
            countries = list(result.country_stock)
            lines = [f"\U0001F504 Stock refilled on {config['target_bot']}", ""]
            lines += _refill_bullets(result.add_reply, countries) or ["\u2022 Numbers re-added"]
            lines += ["", "Nothing to do \u2014 I'll keep watching the stock."]
            await self.notifier.send(
                self.settings.owner_chat_id,
                "\n".join(lines),
                dedupe_key=f"otp_automation_added:{result.ran_at}",
            )
        return result.ok

    @staticmethod
    def _finished_message(finished: list[dict[str, Any]]) -> str:
        """The "run finished" note, split by what happened to each file.

        A HELD country keeps its file in the active list and is only paused,
        so the way back is the resume button - telling the owner to send the
        file and say "start" again would queue a duplicate of a run that is
        merely paused. Items without "held" were removed, and for those
        sending the file again is still the only way back.
        """
        held = [i for i in finished if i.get("held")]
        removed = [i for i in finished if not i.get("held")]
        header = "\U0001F3C1 Run finished"
        if held and not removed:
            header += " \u2014 file kept"
        lines = [header, ""]
        for item in finished:
            lines.append(f"\u2022 {item['country']}: {_english_reason(item['reason'])}")
            if item.get("deleted"):
                lines.append("   Numbers were deleted from the bot.")
            elif item.get("error"):
                lines.append(f"   \u26A0\uFE0F Not deleted: {_one_line(item['error'], 200)}")
            if held and removed and item.get("held"):
                lines.append("   File kept \u2014 paused, nothing more will be added.")

        lines.append("")
        if held:
            example = held[0]["country"]
            # A button, not a phrase to type: nothing in the chat path
            # un-pauses a country, so "say X" would have been a promise the
            # bot could not keep.
            resume = (
                f"open /otpbot \u2192 \U0001F30D Per country \u2192 {example} "
                "and tap \u25B6\uFE0F Resume."
            )
            if removed:
                names = ", ".join(i["country"] for i in held)
                lines.append(f"To run {names} again, {resume}")
            else:
                lines.append(f"Nothing more will be added. To run it again, {resume}")
        if removed:
            if held:
                names = ", ".join(i["country"] for i in removed)
                lines.append(f"To run {names} again, send the file and say \"start\".")
            else:
                lines.append("To run it again, send the file and say \"start\".")
        return "\n".join(lines)

    async def _run_otp_cycle(self, otp_bot: Any, config: dict[str, Any]) -> Any:
        """run_cycle() with an overall deadline.

        run_cycle() never raises, but nothing stops one Telegram call inside it
        from hanging forever - and the scheduler awaits it, so a hang froze
        every job, the briefing and the OTP automation together. A timeout is
        reported exactly like any other failed cycle.
        """
        try:
            return await asyncio.wait_for(otp_bot.run_cycle(config), timeout=OTP_CYCLE_TIMEOUT_S)
        except asyncio.TimeoutError:
            limit = _human_duration(timedelta(seconds=OTP_CYCLE_TIMEOUT_S))
            log.error("otp_automation_timeout", extra={"timeout_s": OTP_CYCLE_TIMEOUT_S})
            return otp_bot.CycleResult(
                ok=False,
                action="error",
                error=(
                    f"the check did not finish within {limit} and was stopped "
                    "(a Telegram call probably hung)"
                ),
            )

    async def _report_otp_health(self, result: Any, at: datetime) -> None:
        """Turn ok/failed cycles into owner alerts without crying wolf.

        First failure: alert at once. Still failing: at most one reminder per
        OTP_FAILURE_REMINDER_EVERY, saying how long it has been. First good
        cycle after a streak: one recovery note, so the owner is not left
        wondering whether the last alert still applies.
        """
        if not result.ok:
            error = _one_line(result.error, 300) or "unknown error"
            every = _human_duration(OTP_FAILURE_REMINDER_EVERY)
            if self._otp_failing_since is None:
                self._otp_failing_since = at
                self._otp_last_failure_alert = at
                text = (
                    "\u26A0\uFE0F OTP-bot automation failed\n\n"
                    f"\u2022 {error}\n\n"
                    f"I'll tell you when it recovers, and remind you every {every} until then."
                )
            elif (
                self._otp_last_failure_alert is None
                or at - self._otp_last_failure_alert >= OTP_FAILURE_REMINDER_EVERY
            ):
                self._otp_last_failure_alert = at
                span = _human_duration(at - self._otp_failing_since)
                text = (
                    f"\u26A0\uFE0F OTP-bot automation has been failing for {span}\n\n"
                    f"\u2022 Last error: {error}\n\n"
                    f"I keep retrying every minute and will remind you again in {every}."
                )
            else:
                return
            await self._notify_owner(text, f"otp_automation_error:{result.ran_at}")
            return

        if self._otp_failing_since is not None:
            span = _human_duration(at - self._otp_failing_since)
            self._otp_failing_since = None
            self._otp_last_failure_alert = None
            await self._notify_owner(
                "\u2705 OTP-bot is running normally again\n\n"
                f"\u2022 It had been failing for {span}.\n\n"
                "Nothing to do \u2014 refills continue on schedule.",
                f"otp_automation_recovered:{result.ran_at}",
            )

    async def _notify_owner(self, text: str, dedupe_key: str) -> None:
        if self.notifier is None or self.settings.owner_chat_id is None:
            return
        await self.notifier.send(self.settings.owner_chat_id, text, dedupe_key=dedupe_key)
