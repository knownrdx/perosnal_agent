"""Autonomy policy: decide when the agent may act without asking.

The old rule was blunt - every HIGH_RISK tool stopped and waited for the owner.
That made routine work (tidying a temp file, cancelling a task the owner just
asked to cancel) as noisy as genuinely dangerous work, so the owner ends up
rubber-stamping everything and stops reading the prompts. That is worse than
useless: it trains the human to click yes.

This module narrows the question to the one that matters:

    can this action be undone, and how much would it cost if it were wrong?

Actions that are reversible, scoped to scratch space, or that the owner just
explicitly asked for are performed directly. Everything else still stops.

It also LEARNS. Every approval decision is remembered as a pattern; once the
owner has approved the same shape of action ``AUTO_APPROVE_AFTER`` times, that
shape stops being asked about. A single rejection wipes that trust immediately -
trust is slow to earn and instant to lose.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from typing import Any

from app.config import get_settings
from app.db import repo
from app.db.base import session_scope
from app.logging_conf import get_logger

log = get_logger(__name__)

# How many times the owner must approve one shape before it stops being asked.
AUTO_APPROVE_AFTER = 3

# Scratch areas: work here is assumed disposable by design.
# NOTE: output/ is deliberately NOT scratch. It holds the artefacts the agent
# produced for the owner, so deleting one destroys real work and still asks.
SCRATCH_GLOBS = ("temp/*", "temp/**", "downloads/*", "downloads/**")

# Paths that are never auto-approved regardless of learned trust.
PROTECTED_GLOBS = ("uploads/**", "tasks/**", "*.env", "**/.env", "**/*credential*", "**/*secret*")


@dataclass(slots=True)
class Verdict:
    """Outcome of the policy check."""

    allow: bool
    reason: str
    learned: bool = False   # allowed because of remembered owner decisions


def _norm(path: str) -> str:
    return str(path or "").replace("\\", "/").lstrip("./")


def _matches(path: str, globs: tuple[str, ...]) -> bool:
    candidate = _norm(path)
    return any(fnmatch.fnmatch(candidate, pattern) for pattern in globs)


def signature(tool: str, args: dict[str, Any]) -> str:
    """A stable 'shape' for an action, so similar actions share a decision.

    Concrete values are generalised: ``temp/report-2026-01.pdf`` and
    ``temp/report-2026-02.pdf`` collapse to ``file_delete:temp/*.pdf`` and are
    therefore judged as the same kind of action - which is what a human means
    by "yes, you can clean up temp files".
    """
    path = args.get("path") or args.get("file") or args.get("target") or ""
    if isinstance(path, str) and path:
        norm = _norm(path)
        directory, _, name = norm.rpartition("/")
        suffix = ("." + name.rsplit(".", 1)[1]) if "." in name else ""
        return f"{tool}:{directory}/*{suffix}" if directory else f"{tool}:*{suffix}"

    for field in ("command", "cmd"):
        value = args.get(field)
        if isinstance(value, str) and value.strip():
            return f"{tool}:{value.strip().split()[0]}"

    for field in ("chat_id", "to", "recipient"):
        if field in args:
            return f"{tool}:{args[field]}"

    return tool


def _explicitly_requested(tool: str, args: dict[str, Any], request: str) -> bool:
    """True when the owner's own words already asked for this action.

    Asking "shall I cancel the task?" right after they said "cancel the task"
    is not a safety check, it is friction.
    """
    text = (request or "").lower()
    if not text:
        return False

    verbs = {
        "file_delete": ("delete", "remove", "erase", "clean up", "clear", "get rid of"),
        "task_cancel": ("cancel", "stop", "abort", "kill"),
        "tg_bot_admin": ("botfather", "bot settings", "rename the bot", "bot description"),
    }.get(tool, ())
    if not any(verb in text for verb in verbs):
        return False

    # The target must be named too, so "delete the temp file" does not license
    # deleting something entirely different.
    target = args.get("path") or args.get("task_id") or ""
    if isinstance(target, str) and target:
        stem = _norm(target).rsplit("/", 1)[-1]
        base = stem.rsplit(".", 1)[0]
        if base and base.lower() in text:
            return True
        # A directory-level instruction ("clear the temp folder") counts.
        folder = _norm(target).rsplit("/", 1)[0]
        return bool(folder) and folder.lower() in text
    return False


async def _learned_verdict(sig: str) -> Verdict | None:
    """Consult remembered owner decisions for this shape of action."""
    async with session_scope() as session:
        stats = await repo.approval_stats(session, sig)

    if stats["rejected"]:
        return Verdict(False, "the owner has previously refused this kind of action")
    if stats["approved"] >= AUTO_APPROVE_AFTER:
        return Verdict(
            True,
            f"the owner approved this kind of action {stats['approved']} times before",
            learned=True,
        )
    return None


async def evaluate(
    tool: str,
    args: dict[str, Any],
    *,
    user_request: str = "",
    side_effect: bool = True,
) -> Verdict:
    """Decide whether ``tool`` may run without stopping to ask the owner."""
    settings = get_settings()

    if not settings.require_approval_high_risk:
        return Verdict(True, "approval gate disabled by configuration")

    if settings.autonomy_level == "paranoid":
        return Verdict(False, "paranoid mode: every risky action is confirmed")

    path = args.get("path") or args.get("file") or ""
    if isinstance(path, str) and path and _matches(path, PROTECTED_GLOBS):
        return Verdict(False, "this path holds credentials or task state")

    if settings.autonomy_level == "high":
        return Verdict(True, "autonomy set to high: acting without confirmation")

    # --- balanced (the default) ------------------------------------------ #
    if _explicitly_requested(tool, args, user_request):
        return Verdict(True, "the owner asked for exactly this in their request")

    if isinstance(path, str) and path and _matches(path, SCRATCH_GLOBS):
        return Verdict(True, "scratch space: this file is disposable by design")

    learned = await _learned_verdict(signature(tool, args))
    if learned is not None:
        return learned

    return Verdict(False, "irreversible and not previously approved")


async def remember_decision(tool: str, args: dict[str, Any], approved: bool) -> None:
    """Record what the owner decided, so the same question is not asked forever."""
    sig = signature(tool, args)
    async with session_scope() as session:
        await repo.record_approval_pattern(session, sig, approved=approved)
    log.info(
        "approval_learned",
        extra={"signature": sig, "approved": approved},
    )


def describe(tool: str, args: dict[str, Any]) -> str:
    """A one-line, human-readable description of what is about to happen."""
    path = args.get("path") or args.get("file") or ""
    if tool == "file_delete" and path:
        return f"delete {_norm(path)}"
    if tool == "task_cancel":
        return f"cancel task {args.get('task_id', '?')}"
    if tool == "tg_bot_admin":
        return f"send {args.get('command', '?')} to BotFather"
    if tool in {"shell_exec", "shell"}:
        command = str(args.get("command", ""))[:80]
        return f"run: {command}"
    rendered = ", ".join(f"{k}={str(v)[:40]}" for k, v in list(args.items())[:3])
    return f"{tool}({rendered})"
