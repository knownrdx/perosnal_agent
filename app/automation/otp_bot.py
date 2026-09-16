"""Deterministic, LLM-free automation for OTP-number distribution bots.

Why this exists (do not route this through the normal LLM task engine): the
target bot (e.g. @PBDxbot) has a strict, undocumented-by-us FSM: a file must
be sent, then the actual command must be sent as a REPLY to that file
message, or the bot silently rejects it ("No valid phone numbers found").
Getting an LLM to reliably reproduce that exact two-step reply sequence,
every single cycle, forever, is a bad bet - and if the LLM provider is slow
or down (it has been, repeatedly), a normal Task-based job would simply never
run. Everything here calls the owner's userbot directly; no LLM in the loop.

Workflow (matches how the owner actually uses it):
    1. Upload one or more files (Telegram document / web dashboard paperclip).
       Each lands in the QUEUE, untagged.
    2. Say "start" (a command, a plain trigger phrase, or the web dashboard's
       Start button). Every file in the queue needs its own tag - different
       files often mean different countries/campaigns - so start() refuses to
       run until every queued file has one; the caller is expected to collect
       missing tags (one prompt per untagged file) before calling start().
    3. start() sends the cleanup command once, then for every queued file:
       sends the file, replies to that file's message with the add-numbers
       command carrying its own tag. Successful files move from the queue
       into ACTIVE_FILES and the periodic monitor turns on.
    4. The scheduler (see app/workers/scheduler_worker.py) checks quota on
       its own timer while enabled=True; when the bot reports quota <=
       threshold, it re-runs cleanup + re-add for every active file (same
       tags, no new prompts needed - they were already collected at start).
    5. "stop" just turns the periodic monitor off. Queue and active-files
       history are left alone, so restarting later is a fresh decision, not
       a silent resume of possibly-stale state.
"""

from __future__ import annotations

import asyncio
import re
import secrets
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.db import repo
from app.db.base import session_scope
from app.integrations.telegram_user import UserbotError, get_userbot
from app.logging_conf import get_logger
from app.security import rel_path, safe_path

log = get_logger(__name__)

SETTING_KEY = "otp_bot_automation"
QUEUE_KEY = f"{SETTING_KEY}_queue"
ACTIVE_KEY = f"{SETTING_KEY}_active"
LAST_RESULT_KEY = f"{SETTING_KEY}_last_result"
LAST_START_KEY = f"{SETTING_KEY}_last_start"
LAST_TAG_KEY = f"{SETTING_KEY}_last_tag"

DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": False,            # periodic quota-monitor/refill loop on/off
    "target_bot": "@PBDxbot",
    "add_command_template": "/fan -t {tag} -l {limit} -c {count}",
    "limit": 4,
    "count": 4,
    "quota_command": "/myquota",
    "quota_threshold": 0,        # refill when active quota <= this
    "cleanup_command": "/useddelete",
    # Force-delete (Owner-level "/frcd <country> <uid>"). /useddelete only
    # clears used/expired numbers; when a country's stock must be wiped
    # outright before re-adding, this is the command that does it.
    "force_delete_command": "/frcd {country} {uid}",
    "force_delete_before_add": False,
    "force_delete_uid": "",      # owner's own uid; blank disables force-delete
    "interval_minutes": 10,
    "default_tag": "",           # if set, every new file auto-tags with this, never asks
    "thread_id": "",             # dedicated chat thread; "" = not bound yet
    "awaiting_tag_entry_id": None,  # set while a "what tag for X" prompt is pending
}

# Seeded as the dedicated thread's first message so it gets a recognisable
# title in the thread list (list_threads titles a thread from its first user
# message) instead of showing up as a nameless "New chat".
THREAD_TITLE = "\U0001F501 OTP Bot - send number files here"

# Known tag/region tokens the agent recognises straight out of a filename, so
# it can decide the tag itself instead of asking every single time - e.g.
# "numbers_BD_batch2.txt" -> "BD" with zero owner interaction.
_KNOWN_TAG_TOKENS = {
    "BD", "IN", "PK", "NG", "KE", "ID", "US", "UK", "GENERAL", "GEN",
}

# The live /myquota reply looks like:
#     📊 Your quota
#     Active : 0
#     Limit  : unlimited (disabled)
# Comma-grouped counts ("1,234") appear once the number gets large, so they
# have to be accepted here or a healthy quota reads as an unparseable one.
_QUOTA_RE = re.compile(r"active\s*[:\-]?\s*([\d,]+)", re.IGNORECASE)
_FAILURE_PHRASES = ("no valid", "error", "failed", "invalid", "not found", "denied")


def _looks_like_failure(reply_text: str) -> bool:
    lowered = reply_text.lower()
    return any(phrase in lowered for phrase in _FAILURE_PHRASES)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _sleep(seconds: float) -> None:
    """Thin wrapper around asyncio.sleep so tests can monkeypatch it to zero -
    the real delays exist only to give the target bot time to reply before we
    read its messages back.

    NOTE: this must call asyncio.sleep, never _sleep - an earlier bulk rename
    of asyncio.sleep -> _sleep rewrote this body too and made the function
    infinitely recursive. Tests monkeypatch _sleep, so only production hit it,
    surfacing as a bare "maximum recursion depth exceeded" with no traceback.
    """
    await asyncio.sleep(seconds)


@dataclass(slots=True)
class CycleResult:
    ok: bool
    action: str                     # "skipped" | "added" | "error"
    quota_reply: str = ""
    active_quota: int | None = None
    add_reply: str = ""
    files_processed: list[str] = field(default_factory=list)
    error: str = ""
    ran_at: str = field(default_factory=_now_iso)


# --------------------------------------------------------------------------- #
# Config (settings that apply to every run: target bot, timing, templates)
# --------------------------------------------------------------------------- #
async def get_config() -> dict[str, Any]:
    async with session_scope() as session:
        stored = await repo.get_setting(session, SETTING_KEY)
    merged = dict(DEFAULT_CONFIG)
    if stored:
        merged.update(stored)
    return merged


async def save_config(patch: dict[str, Any]) -> dict[str, Any]:
    current = await get_config()
    current.update({k: v for k, v in patch.items() if k in DEFAULT_CONFIG})
    async with session_scope() as session:
        await repo.set_setting(session, SETTING_KEY, current)
    return current


async def get_last_result() -> dict[str, Any] | None:
    async with session_scope() as session:
        return await repo.get_setting(session, LAST_RESULT_KEY)


async def _save_last_result(result: CycleResult) -> None:
    async with session_scope() as session:
        await repo.set_setting(session, LAST_RESULT_KEY, asdict(result))


async def get_last_start_result() -> dict[str, Any] | None:
    async with session_scope() as session:
        return await repo.get_setting(session, LAST_START_KEY)


async def _get_last_tag(country: str | None = None) -> str | None:
    """The tag last used - for THIS country when one is given.

    Service/tag is a per-country decision in practice (Bangladesh stock is
    not sold as the same service as Nigeria stock), so asking for a specific
    country must NOT fall back to the global last tag: doing so silently
    files a new country's numbers under whatever service happened to be used
    last, which is exactly the mix-up this split exists to prevent.
    """
    async with session_scope() as session:
        stored = await repo.get_setting(session, LAST_TAG_KEY)
    if not stored:
        return None
    if country:
        return (stored.get("by_country") or {}).get(country)
    return stored.get("tag")


async def _set_last_tag(tag: str, country: str | None = None) -> None:
    async with session_scope() as session:
        stored = await repo.get_setting(session, LAST_TAG_KEY) or {}
        by_country = dict(stored.get("by_country") or {})
        if country:
            by_country[country] = tag
        await repo.set_setting(
            session, LAST_TAG_KEY, {"tag": tag, "by_country": by_country}
        )


def _infer_tag_from_filename(name: str) -> str | None:
    """Read the tag straight off the filename when it's obvious, e.g.
    "numbers_BD_batch2.txt" -> "BD". Lets the agent decide on its own
    instead of asking every time a file with a self-describing name arrives.
    """
    stem = name.rsplit(".", 1)[0]
    for token in re.split(r"[ _\-]+", stem):
        if token.upper() in _KNOWN_TAG_TOKENS:
            return token.upper()
    return None


async def _decide_tag(name: str, country: str | None = None) -> str | None:
    """Best-effort autonomous tag decision for a newly queued file, in order:
    1. the tag last used for THIS country (strongest signal - same country,
       same service, new batch is the normal case)
    2. filename says it explicitly (the owner named it that way)
    3. an owner-configured default_tag applies to everything
    Returns None only when none of the above give anything - that's the one
    case worth actually asking about.

    Deliberately does NOT fall back to "the last tag used for anything" when
    the country is known: service is a per-country decision, so inheriting
    Nigeria's answer for a Bangladesh file would silently file the numbers
    under the wrong service - worse than asking one short question.
    """
    if country:
        remembered = await _get_last_tag(country)
        if remembered:
            return remembered

    inferred = _infer_tag_from_filename(name)
    if inferred:
        return inferred
    config = await get_config()
    if config.get("default_tag"):
        return config["default_tag"]
    if country:
        return None
    return await _get_last_tag()


# --------------------------------------------------------------------------- #
# Dedicated chat thread: one conversation that IS the OTP-bot workspace.
#
# Without this the owner has to remember which of several chat threads a
# number file belongs in, and a file dropped into a general conversation
# looks identical to one meant for this automation. Binding a specific
# thread makes "where do I send the files?" answer itself - anything sent
# in that thread is for the OTP bot, anything elsewhere is not.
# --------------------------------------------------------------------------- #
async def get_thread_id() -> str:
    return (await get_config()).get("thread_id") or ""


async def is_otp_thread(chat_id: int) -> bool:
    """True when the given chat's CURRENT thread is the dedicated OTP one."""
    bound = await get_thread_id()
    if not bound:
        return False
    async with session_scope() as session:
        row = await repo.ensure_session(session, chat_id)
        return row.current_thread_id == bound


async def ensure_thread(chat_id: int) -> str:
    """Create (once) and switch to the dedicated OTP-bot thread, returning it.

    Reuses the existing thread if one is already bound and still present, so
    calling this repeatedly never spawns duplicate near-identical threads.
    """
    bound = await get_thread_id()
    async with session_scope() as session:
        # A chat that has never been seen has no chat_sessions row yet, and
        # reset_session is an UPDATE - without this it would silently affect
        # zero rows and leave the chat on the default thread.
        await repo.ensure_session(session, chat_id)

        if bound:
            existing = {
                t["thread_id"]: t
                for t in await repo.list_threads(session, chat_id, limit=200)
            }
            if bound in existing:
                await repo.switch_thread(session, chat_id, bound)
                # Backfill the title for a thread that predates the titling
                # fix (or lost its seed some other way) - otherwise it stays
                # a nameless "New chat" forever, which is the one thing this
                # thread is supposed to prevent.
                if existing[bound]["title"] in ("New chat", ""):
                    await repo.add_message(
                        session, chat_id=chat_id, role="user",
                        content=THREAD_TITLE, thread_id=bound,
                    )
                return bound

        new_thread = await repo.reset_session(session, chat_id)
        # Seeded with role="user" deliberately: list_threads titles a thread
        # from its first USER message, so an assistant-role seed would leave
        # this showing as a nameless "New chat" in the thread list - exactly
        # the "which chat was it again?" problem this thread exists to solve.
        await repo.add_message(
            session, chat_id=chat_id, role="user",
            content=THREAD_TITLE, thread_id=new_thread,
        )
        await repo.add_message(
            session, chat_id=chat_id, role="assistant",
            content=(
                "Send number files here (one at a time is fine), then say "
                "\"start\" when you're done. Anything sent in this thread goes "
                "straight into the OTP-bot queue - files sent in other threads "
                "are left alone."
            ),
            thread_id=new_thread,
        )
    await save_config({"thread_id": new_thread})
    return new_thread


# --------------------------------------------------------------------------- #
# File queue: uploaded, waiting for a tag + the start() call
# --------------------------------------------------------------------------- #
async def _load_files(key: str) -> list[dict[str, Any]]:
    async with session_scope() as session:
        stored = await repo.get_setting(session, key)
    return list(stored.get("files", [])) if stored else []


async def _save_files(key: str, files: list[dict[str, Any]]) -> None:
    async with session_scope() as session:
        await repo.set_setting(session, key, {"files": files})


async def get_queue() -> list[dict[str, Any]]:
    return await _load_files(QUEUE_KEY)


async def get_active_files() -> list[dict[str, Any]]:
    return await _load_files(ACTIVE_KEY)


async def enqueue_file(path: str, name: str) -> dict[str, Any]:
    """Analyse an uploaded file and queue one entry PER COUNTRY found in it.

    A single upload routinely mixes countries (a "numbers" export can hold
    Dominican Republic and Bangladesh side by side) and the target bot keeps
    stock per country, so adding the file whole would file every number under
    one country's tag. Each country therefore becomes its own queue entry
    with its own split-out file, and each carries the service/tag last used
    for that country so the common case needs no questions at all.
    """
    from app.automation import phone_countries

    source = safe_path(path, must_exist=True)
    lines = source.read_text(encoding="utf-8", errors="ignore").splitlines()
    grouped = phone_countries.split_by_country(lines)

    queue = await get_queue()
    created: list[dict[str, Any]] = []
    batch_id = secrets.token_hex(4)

    for country, numbers in grouped.items():
        split_path = path
        if len(grouped) > 1:
            # Write the country's own file so the bot only ever receives
            # numbers that match the country/tag the command declares.
            safe_country = re.sub(r"[^A-Za-z0-9]+", "_", country).strip("_") or "unknown"
            stem = name.rsplit(".", 1)[0]
            split_name = f"{stem}__{safe_country}.txt"
            target = safe_path(f"uploads/{split_name}")
            target.write_text("\n".join(numbers) + "\n", encoding="utf-8")
            split_path = rel_path(target)

        entry = {
            "id": secrets.token_hex(4),
            "batch_id": batch_id,
            "path": split_path,
            "name": name if len(grouped) == 1 else f"{name} [{country}]",
            "source_name": name,
            "country": country,
            "count": len(numbers),
            "tag": await _decide_tag(name, country),
            "uploaded_at": _now_iso(),
        }
        queue.append(entry)
        created.append(entry)

    await _save_files(QUEUE_KEY, queue)
    return {
        "batch_id": batch_id,
        "entries": created,
        "countries": {c: len(n) for c, n in grouped.items()},
    }


async def set_queue_tag(entry_id: str, tag: str) -> dict[str, Any] | None:
    queue = await get_queue()
    for entry in queue:
        if entry["id"] == entry_id:
            entry["tag"] = tag.strip()[:60]
            await _save_files(QUEUE_KEY, queue)
            # Remembered per country, so the next Bangladesh file does not
            # inherit the tag that happened to be used for a Nigeria file.
            await _set_last_tag(entry["tag"], entry.get("country"))
            return entry
    return None


async def remove_from_queue(entry_id: str) -> bool:
    queue = await get_queue()
    remaining = [e for e in queue if e["id"] != entry_id]
    if len(remaining) == len(queue):
        return False
    await _save_files(QUEUE_KEY, remaining)
    # A prompt pointing at a now-deleted entry would hang the conversation:
    # get_awaiting_tag_entry() returns None (the id no longer resolves) while
    # the config still claims one is pending, so clear it explicitly.
    config = await get_config()
    if config.get("awaiting_tag_entry_id") == entry_id:
        await set_awaiting_tag_entry(None)
    return True


async def remove_active_file(entry_id: str) -> dict[str, Any] | None:
    """Stop monitoring one already-started file.

    Removing it here does not un-add the numbers the bot already has - it
    only means future refill cycles stop re-adding this file. Emptying the
    active set also turns the monitor off, since there is then nothing left
    for it to do.
    """
    active = await get_active_files()
    removed = next((e for e in active if e["id"] == entry_id), None)
    if removed is None:
        return None
    remaining = [e for e in active if e["id"] != entry_id]
    await _save_files(ACTIVE_KEY, remaining)
    if not remaining:
        await save_config({"enabled": False})
    return removed


async def remove_by_country(country: str) -> list[dict[str, Any]]:
    """Drop every queued AND active entry for one country.

    Country is how the owner actually thinks about this ("bad the Bangladesh
    one"), and after splitting, one country can span several entries from
    different uploads - so matching by country removes all of them rather
    than leaving stragglers behind.
    """
    wanted = country.strip().casefold()
    removed: list[dict[str, Any]] = []

    queue = await get_queue()
    keep_queue = [e for e in queue if (e.get("country") or "").casefold() != wanted]
    removed += [e for e in queue if (e.get("country") or "").casefold() == wanted]
    if len(keep_queue) != len(queue):
        await _save_files(QUEUE_KEY, keep_queue)

    active = await get_active_files()
    keep_active = [e for e in active if (e.get("country") or "").casefold() != wanted]
    removed += [e for e in active if (e.get("country") or "").casefold() == wanted]
    if len(keep_active) != len(active):
        await _save_files(ACTIVE_KEY, keep_active)
        if not keep_active:
            await save_config({"enabled": False})

    if removed:
        config = await get_config()
        if config.get("awaiting_tag_entry_id") in {e["id"] for e in removed}:
            await set_awaiting_tag_entry(None)
    return removed


async def clear_queue() -> int:
    """Throw away everything waiting to be started. Active files are left
    alone - "clear the queue" should not silently stop running work.
    """
    queue = await get_queue()
    if queue:
        await _save_files(QUEUE_KEY, [])
        await set_awaiting_tag_entry(None)
    return len(queue)


async def next_untagged_entry() -> dict[str, Any] | None:
    for entry in await get_queue():
        if not entry.get("tag"):
            return entry
    return None


async def get_awaiting_tag_entry() -> dict[str, Any] | None:
    """The queue entry currently being asked about, if a tag prompt is open."""
    config = await get_config()
    awaiting_id = config.get("awaiting_tag_entry_id")
    if not awaiting_id:
        return None
    for entry in await get_queue():
        if entry["id"] == awaiting_id:
            return entry
    return None


async def set_awaiting_tag_entry(entry_id: str | None) -> None:
    await save_config({"awaiting_tag_entry_id": entry_id})


class AddRejected(Exception):
    """The target bot's own reply says the add did not work.

    A distinct type rather than RuntimeError: RecursionError, ValueError and
    friends all subclass Exception too, and catching a broad built-in here
    once caused a real crash (infinite recursion in _sleep) to be reported to
    the owner as "the bot rejected your file", sending the investigation in
    completely the wrong direction.
    """


# --------------------------------------------------------------------------- #
# Helpers shared by start() and the periodic refill cycle
# --------------------------------------------------------------------------- #
_QUOTA_RE_MATCH = _QUOTA_RE


def _parse_quota(text: str) -> int | None:
    match = _QUOTA_RE_MATCH.search(text)
    if not match:
        return None
    return int(match.group(1).replace(",", ""))


async def _last_bot_message(target: str, *, limit: int = 5) -> dict[str, Any] | None:
    """The most recent message FROM the bot (not from us).

    read_messages returns newest-first, so the newest bot message is the
    first non-outgoing entry. Iterating in reverse here once returned the
    OLDEST message in the window instead, which is why a quota check could
    sit at "no reply" while the answer was already sitting in the chat.
    """
    messages = await get_userbot().read_messages(target, limit)
    for message in messages:
        if not message.get("out"):
            return message
    return None


async def _wait_for_new_bot_reply(
    target: str,
    *,
    after_id: int | None,
    timeout_s: float = 90.0,
    poll_s: float = 3.0,
    matches: Any | None = None,
) -> dict[str, Any] | None:
    """Wait for a bot message NEWER than ``after_id`` (optionally one whose
    text satisfies ``matches``).

    A fixed sleep-then-read is not enough: a large file can take the target
    bot well over a minute to ingest, and reading too early returns its
    PREVIOUS message - which is how an add once got reported with the
    /useddelete reply ("Deleted 4,796 used numbers") as its result.

    Recency alone is not enough either: the bot emits progress/completion
    messages of its own, so a quota check can land on the tail end of an
    earlier add ("Fast Add Complete!") and fail to parse. ``matches`` lets
    the caller say what the answer should look like, and unmatched messages
    are skipped rather than mistaken for the reply.
    """
    waited = 0.0
    newest_seen = after_id
    while waited < timeout_s:
        await _sleep(poll_s)
        waited += poll_s
        message = await _last_bot_message(target)
        if message is None:
            continue
        message_id = int(message.get("id", 0))
        if newest_seen is not None and message_id <= newest_seen:
            continue
        if matches is not None and not matches(message.get("text") or ""):
            # Something new, but not the answer we asked for - remember it so
            # we keep moving forward instead of re-examining it every poll.
            newest_seen = message_id
            continue
        return message
    return None


async def _force_delete_country(target: str, country: str, cfg: dict[str, Any]) -> str:
    """Owner-level /frcd for one country, when configured.

    /useddelete only removes used/expired numbers. Re-adding a country whose
    stock is still live would just pile duplicates on top (the bot reports
    them as "dup"), so when the owner wants a genuine replace rather than a
    top-up, this wipes that country's numbers first.
    """
    uid = str(cfg.get("force_delete_uid") or "").strip()
    if not uid:
        return ""
    command = cfg["force_delete_command"].format(country=country, uid=uid)

    previous = await _last_bot_message(target)
    previous_id = int(previous.get("id", 0)) if previous else None
    await get_userbot().send_message(target, command)
    reply = await _wait_for_new_bot_reply(target, after_id=previous_id, timeout_s=45.0)
    return (reply.get("text") or "") if reply else ""


async def _add_one_file(target: str, entry: dict[str, Any], cfg: dict[str, Any]) -> str:
    """Send one file as a reply-based add; returns the bot's reply text.
    Raises AddRejected if the bot's own reply indicates the add failed
    (e.g. "No valid phone numbers found") - a delivered message is not the
    same as a successful add, and silently reporting success on a rejected
    file would hide the exact bug this automation exists to avoid.
    """
    target_path = safe_path(entry["path"], must_exist=True)

    if cfg.get("force_delete_before_add") and entry.get("country"):
        await _force_delete_country(target, entry["country"], cfg)
        await _sleep(2)

    # Remember where the conversation stood before we touch it, so we can
    # tell this file's reply apart from whatever the bot said last.
    previous = await _last_bot_message(target)
    previous_id = int(previous.get("id", 0)) if previous else None

    sent_file = await get_userbot().send_file(target, str(target_path))
    await _sleep(2)

    command_text = cfg["add_command_template"].format(
        tag=entry.get("tag") or "General",
        limit=cfg["limit"],
        count=cfg["count"],
        country=entry.get("country") or "",
    )
    await get_userbot().send_message(target, command_text, reply_to=sent_file.get("message_id"))

    reply = await _wait_for_new_bot_reply(target, after_id=previous_id)
    if reply is None:
        raise AddRejected(
            f"{entry['name']}: no reply from {target} within the wait window - "
            "the add may or may not have gone through, check the chat"
        )
    text = reply.get("text") or ""
    if text and _looks_like_failure(text):
        raise AddRejected(f"{entry['name']}: bot rejected the add ({text[:200]})")
    return text


# --------------------------------------------------------------------------- #
# start() / stop(): owner-triggered, not on the scheduler's timer
# --------------------------------------------------------------------------- #
async def start_automation() -> dict[str, Any]:
    """Consume every queued (and tagged) file: cleanup once, then add each
    file as its own reply-based command. On success the queue becomes the
    new active_files set (replacing whatever was active before) and the
    periodic monitor (enabled) turns on.
    """
    cfg = await get_config()
    queue = await get_queue()

    if not queue:
        return {"ok": False, "error": "no files queued - upload one first", "missing_tags": []}

    missing = [e for e in queue if not e.get("tag")]
    if missing:
        return {
            "ok": False,
            "error": "every queued file needs a tag before starting",
            "missing_tags": [
                {
                    "id": e["id"],
                    "name": e["name"],
                    "country": e.get("country"),
                    "count": e.get("count"),
                }
                for e in missing
            ],
        }

    target = cfg["target_bot"]
    result: dict[str, Any] = {"ok": False, "target_bot": target, "files": [], "error": "", "ran_at": _now_iso()}

    try:
        await get_userbot().send_message(target, cfg["cleanup_command"])
        await _sleep(2)

        for entry in queue:
            reply = await _add_one_file(target, entry, cfg)
            result["files"].append({
                "name": entry["name"],
                "country": entry.get("country"),
                "count": entry.get("count"),
                "tag": entry["tag"],
                "reply": reply,
            })

        await save_config({"enabled": True, "awaiting_tag_entry_id": None})
        await _save_files(ACTIVE_KEY, queue)
        await _save_files(QUEUE_KEY, [])
        result["ok"] = True

    except UserbotError as exc:
        result["error"] = f"telegram account not linked or errored: {exc}"
    except FileNotFoundError as exc:
        result["error"] = str(exc)
    except AddRejected as exc:
        result["error"] = str(exc)
    except Exception as exc:  # noqa: BLE001 - start() must never crash the caller
        # log.exception keeps the real traceback in the container logs; the
        # owner-facing string alone is not enough to debug a crash like the
        # recursive-_sleep one, which read simply as "maximum recursion depth
        # exceeded" with no indication of where.
        log.exception("otp_bot_start_error")
        result["error"] = f"unexpected error ({type(exc).__name__}): {exc}"

    async with session_scope() as session:
        await repo.set_setting(session, LAST_START_KEY, result)
    return result


async def set_cleanup_mode(mode: str) -> dict[str, Any]:
    """Pick between "tidy up" and "wipe and replace" as one decision.

    Exposed as a single choice because that is how the owner thinks about it;
    the two config flags underneath (which command, and whether /frcd runs at
    all) always have to move together, and letting them drift apart is how
    you end up with force-delete "enabled" but silently inert.
    """
    if mode == "force":
        return await save_config({"force_delete_before_add": True})
    return await save_config({"force_delete_before_add": False})


async def stop_automation() -> dict[str, Any]:
    """Turn the periodic monitor off. Queue and active_files are untouched -
    starting again later is always a fresh, explicit decision (the owner
    asked for this specifically: a new upload after stop must not silently
    reuse old files).
    """
    cfg = await save_config({"enabled": False})
    return {"ok": True, "enabled": cfg["enabled"]}


# --------------------------------------------------------------------------- #
# Deterministic chat triggers - plain words, no LLM, work identically from
# Telegram or the web dashboard chat (both call app.agent.conversation.
# handle_message with the same text). This is the whole point: the owner
# must be able to run this even while every LLM provider is down.
# --------------------------------------------------------------------------- #
_START_WORDS = {
    "start", "shuru", "shuru koro", "shuru korbo", "start koro", "cholo",
    "run", "go", "begin", "done", "shesh", "sesh", "shesh hoise", "sesh hoise",
    "finish", "finished",
}
_STOP_WORDS = {
    "stop", "off", "bondho", "bondho koro", "bondho kore dao", "thamao",
    "pause", "stop koro",
}
_RESUME_PHRASES = ("age-r file", "agerfile", "purono file", "same file", "old file", "agerta")

# Removing things. "skip"/"bad dao" while a tag question is open means "not
# this one", which is the moment the owner most often realises a file should
# not go in at all.
_SKIP_WORDS = {
    "skip", "bad", "bad dao", "baddao", "bad de", "baddo", "cancel", "no",
    "na", "eta na", "eta bad", "remove", "delete", "bad koro", "skip koro",
}

# Offered as buttons so the common answers are one tap instead of typing.
# The owner can still type anything else - these are shortcuts, not a
# closed list.
SERVICE_CHOICES = ("WhatsApp", "Telegram", "Facebook", "Google", "Instagram", "Signal")
INTERVAL_CHOICES = (5, 10, 15, 30, 60)
# Cleanup style, phrased as the decision rather than the command: "just tidy
# up" vs "wipe and replace". The second needs /frcd, which needs the uid.
CLEANUP_CHOICES = (
    ("used", "\U0001F9F9 Used/expired only (/useddelete)"),
    ("force", "\U0001F5D1 Wipe country first (/frcd)"),
)
_CLEAR_PHRASES = ("sob bad", "sob remove", "clear queue", "queue clear",
                  "sob cancel", "clear all", "sob delete")
# "<country> bad dao" / "remove <country>" - the country is whatever is left
# once the removal word is taken out.
_REMOVE_PREFIXES = ("remove ", "delete ", "bad dao ", "bad koro ", "bad ")
_REMOVE_SUFFIXES = (" bad dao", " bad koro", " bad", " remove koro", " remove",
                    " delete koro", " delete", " cancel")


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def is_skip_trigger(text: str) -> bool:
    return _normalize(text) in _SKIP_WORDS


def is_clear_trigger(text: str) -> bool:
    return any(phrase in _normalize(text) for phrase in _CLEAR_PHRASES)


def country_to_remove(text: str) -> str | None:
    """Extract the country from "<country> bad dao" / "remove <country>".

    Returns None when the message is not a removal request, so an ordinary
    sentence mentioning a country never deletes anything.
    """
    norm = _normalize(text)
    if is_clear_trigger(norm) or is_skip_trigger(norm):
        return None

    for prefix in _REMOVE_PREFIXES:
        if norm.startswith(prefix):
            rest = norm[len(prefix):].strip()
            return rest or None
    for suffix in _REMOVE_SUFFIXES:
        if norm.endswith(suffix):
            rest = norm[: -len(suffix)].strip()
            return rest or None
    return None


def is_start_trigger(text: str) -> bool:
    return _normalize(text) in _START_WORDS


def is_stop_trigger(text: str) -> bool:
    return _normalize(text) in _STOP_WORDS


def is_resume_trigger(text: str) -> bool:
    norm = _normalize(text)
    return any(phrase in norm for phrase in _RESUME_PHRASES)


async def _notify_owner(text: str) -> None:
    """Best-effort Telegram ping regardless of which channel (Telegram or the
    web dashboard) actually triggered the change - the owner asked to be
    told on Telegram either way.
    """
    try:
        from app.config import get_settings
        from app.telegram.notifier import Notifier

        settings = get_settings()
        if settings.owner_chat_id:
            await Notifier().send(settings.owner_chat_id, text)
    except Exception:  # noqa: BLE001 - a notify failure must never break the flow
        log.exception("otp_bot_notify_failed")


def _format_start_success(result: dict[str, Any]) -> str:
    lines = [
        f"\u2705 Shuru hoye geche - {len(result['files'])} file(s) add kora hoyeche "
        f"{result['target_bot']}-e."
    ]
    for f in result["files"]:
        country = f.get("country") or "?"
        lines.append(f"  {country} - {f['count']} number (tag: {f['tag']})")
        lines.append(f"     {f['reply'][:150]}")
    lines.append("")
    lines.append(
        "Ekhon periodically quota check hobe, quota shesh hole nijei cleanup+re-add "
        "korbe. 'off' bolle bondho hobe."
    )
    return "\n".join(lines)


def _tag_question(entry: dict[str, Any]) -> str:
    """Ask for one country's service/tag, showing what was actually detected.

    The owner needs to see WHICH country and how many numbers before naming a
    service - the same upload can contain several countries, and the answer
    differs per country.
    """
    country = entry.get("country") or "?"
    count = entry.get("count")
    detail = f"{count} number" if count else "numbers"
    return (
        f"\U0001F30D {country} - {detail}\n"
        f"(file: {entry.get('source_name') or entry['name']})\n\n"
        "Ei country-r jonno kon service/tag e add korbo? (jemon: WhatsApp, Telegram)"
    )


async def handle_start_trigger() -> str:
    """Owner said "start"/"done"/etc. Ask for any missing tag first (one
    question at a time), otherwise actually start.
    """
    entry = await next_untagged_entry()
    if entry is not None:
        await set_awaiting_tag_entry(entry["id"])
        return _tag_question(entry)

    queue = await get_queue()
    if not queue:
        active = await get_active_files()
        if active:
            # Nothing new was queued, but files are already active - "start"
            # after "stop" with no new upload almost always means "resume
            # what I had going". Low-risk and fully reversible (stop undoes
            # it), so just do it instead of asking.
            return await handle_resume_trigger()
        return "Kono file queue-e nai. Age file upload koro, tarpor 'start' bolo."

    result = await start_automation()
    if result["ok"]:
        text = _format_start_success(result)
        await _notify_owner(f"\U0001F7E2 OTP-bot automation started.\n\n{text}")
        return text
    return f"\u274C Start korte parlam na: {result['error']}"


async def _continue_after_tagging(prefix: str) -> str:
    """Either ask the next question or, if nothing is left to ask, start."""
    next_entry = await next_untagged_entry()
    if next_entry is not None:
        await set_awaiting_tag_entry(next_entry["id"])
        return f"{prefix}\n\n{_tag_question(next_entry)}"

    if not await get_queue():
        return f"{prefix}\n\nQueue ekhon khali. Notun file dao, tarpor 'start' bolo."

    result = await start_automation()
    if result["ok"]:
        text_out = _format_start_success(result)
        await _notify_owner(f"\U0001F7E2 OTP-bot automation started.\n\n{text_out}")
        return f"{prefix}\n\n{text_out}"
    return f"{prefix}\n\n\u274C Start korte parlam na: {result['error']}"


async def handle_tag_answer(text: str) -> str:
    """Owner just answered a "what tag for X" question."""
    entry = await get_awaiting_tag_entry()
    if entry is None:
        return "Kono file tag-er jonno wait kortese na. 'start' bolle shuru hobe."

    label = entry.get("country") or entry["name"]

    # "skip" / "bad dao" here means "not this one" - the tag question is
    # exactly when the owner notices a country they did not mean to send.
    if is_skip_trigger(text):
        await remove_from_queue(entry["id"])
        return await _continue_after_tagging(f"\U0001F5D1 {label} bad deoa holo.")

    await set_queue_tag(entry["id"], text)
    await set_awaiting_tag_entry(None)
    return await _continue_after_tagging(f"\u2705 {label} -> {text.strip()[:60]}")


async def handle_remove_country(country: str) -> str:
    """Owner said "<country> bad dao" - drop it from queue and active alike."""
    removed = await remove_by_country(country)
    if not removed:
        queue = await get_queue()
        active = await get_active_files()
        known = sorted({e.get("country") or "?" for e in queue + active})
        available = ", ".join(known) if known else "(kichu nai)"
        return f"'{country}' khuje pelam na.\nEkhon ache: {available}"

    total = sum(e.get("count") or 0 for e in removed)
    label = removed[0].get("country") or country
    text = f"\U0001F5D1 {label} bad deoa holo ({len(removed)} entry, {total} number)."

    # If that removal answered the open question, keep the flow moving.
    if await get_awaiting_tag_entry() is None and await next_untagged_entry() is not None:
        return await _continue_after_tagging(text)
    return text


async def handle_clear_queue() -> str:
    """Owner said "sob bad dao" - empty the not-yet-started queue."""
    count = await clear_queue()
    if not count:
        return "Queue emnitei khali."
    return (
        f"\U0001F5D1 Queue clear kora holo ({count} entry bad).\n"
        "Cholte thaka file gulo (active) thik-i ache - oigula bondho korte 'stop' bolo."
    )


async def handle_stop_trigger() -> str:
    cfg = await get_config()
    if not cfg["enabled"]:
        return "Automation already off."
    await stop_automation()
    await _notify_owner("\U0001F534 OTP-bot automation bondho kora holo.")
    return (
        "\u23F8\uFE0F Automation bondho kora holo. Notun file dile ba 'start' bolle "
        "abar jiggesh korbo kon file diye shuru korbo."
    )


async def handle_resume_trigger() -> str:
    """Owner said "age-r file diye shuru koro" (resume with previously active
    files, already tagged, without needing a fresh upload).
    """
    active = await get_active_files()
    if not active:
        return "Kono age-r active file nai - notun file upload koro, tarpor 'start' bolo."
    await _save_files(QUEUE_KEY, active)
    result = await start_automation()
    if result["ok"]:
        text = _format_start_success(result)
        await _notify_owner(f"\U0001F7E2 OTP-bot automation resumed (age-r file diye).\n\n{text}")
        return text
    return f"\u274C Resume korte parlam na: {result['error']}"


# --------------------------------------------------------------------------- #
# The periodic cycle (called by the scheduler on its own timer, and by the
# dashboard's manual "Run now" button)
# --------------------------------------------------------------------------- #
async def run_cycle(config: dict[str, Any] | None = None) -> CycleResult:
    """One quota-check-and-refill pass over the ACTIVE files. Never raises -
    always returns a result, even on failure.
    """
    cfg = config or await get_config()
    target = cfg["target_bot"]
    result = CycleResult(ok=False, action="error")

    active_files = await get_active_files()
    if not active_files:
        result.error = "no active files - run start() first"
        await _save_last_result(result)
        return result

    try:
        # Anchor on the last bot message before asking, so a slow answer is
        # waited for rather than the previous one being mistaken for it.
        before_quota = await _last_bot_message(target)
        before_quota_id = int(before_quota.get("id", 0)) if before_quota else None

        await get_userbot().send_message(target, cfg["quota_command"])
        # Only a message that actually parses as a quota counts as the
        # answer: the bot also posts its own add/progress notices, and one of
        # those arriving first previously produced "could not parse an
        # 'Active' count" even though the quota reply was on its way.
        quota_msg = await _wait_for_new_bot_reply(
            target,
            after_id=before_quota_id,
            timeout_s=45.0,
            matches=lambda text: _parse_quota(text) is not None,
        )
        if quota_msg is None:
            result.error = "no reply from the bot to the quota command"
            await _save_last_result(result)
            return result
        result.quota_reply = quota_msg["text"]
        active = _parse_quota(quota_msg["text"])
        result.active_quota = active

        if active is None:
            result.error = "could not parse an 'Active' count from the quota reply"
            await _save_last_result(result)
            return result

        if active > cfg["quota_threshold"]:
            result.ok = True
            result.action = "skipped"
            await _save_last_result(result)
            return result

        # Quota exhausted: clean up once, then re-add every active file.
        await get_userbot().send_message(target, cfg["cleanup_command"])
        await _sleep(2)

        replies: list[str] = []
        for entry in active_files:
            reply = await _add_one_file(target, entry, cfg)
            result.files_processed.append(entry["name"])
            if reply:
                replies.append(f"{entry['name']}: {reply}")

        result.add_reply = "\n".join(replies)
        result.ok = True
        result.action = "added"
        await _save_last_result(result)
        return result

    except UserbotError as exc:
        result.error = f"telegram account not linked or errored: {exc}"
        await _save_last_result(result)
        return result
    except FileNotFoundError as exc:
        result.error = str(exc)
        await _save_last_result(result)
        return result
    except Exception as exc:  # noqa: BLE001 - a cycle must never crash the runner
        log.exception("otp_bot_cycle_error")
        result.error = f"unexpected error ({type(exc).__name__}): {exc}"
        await _save_last_result(result)
        return result
