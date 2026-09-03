"""Keeping long tasks coherent.

A task that runs 20 steps produces more history than fits in a prompt. The old
behaviour was ``history[-12:]`` - a hard cut that silently threw away step 1,
which is usually where the important context lives (what was downloaded, which
id was created). The agent would then repeat work it had already done.

This module keeps the recent steps verbatim and compresses the rest into a
short factual summary, so nothing is lost outright.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.llm import LLMClient, LLMError, Message
from app.logging_conf import get_logger

log = get_logger(__name__)

# How many steps stay verbatim before compaction kicks in.
KEEP_VERBATIM = 8
COMPACT_AFTER = 12

SUMMARY_PROMPT = """You compress an agent's execution history.

Write a factual summary of what has happened so far, for an agent that must
continue the work. Preserve concrete details: file paths, ids, URLs, counts,
error messages, and anything already completed. Drop reasoning and repetition.

At most 10 short lines. No preamble, no commentary, just the facts.
"""


@dataclass(slots=True)
class Compacted:
    summary: str
    recent: list[str]

    @property
    def was_compacted(self) -> bool:
        return bool(self.summary)


def _extract_facts(entries: list[str]) -> str:
    """Deterministic fallback summary when no model is available.

    Pulls out the concrete nouns an agent needs - which tools ran, which files
    appeared, what failed - without inventing anything.
    """
    tools: list[str] = []
    paths: set[str] = set()
    errors: list[str] = []

    for entry in entries:
        for name in re.findall(r'"tool":\s*"([\w.]+)"', entry):
            if name not in tools:
                tools.append(name)
        for path in re.findall(r'"path":\s*"([^"]{1,120})"', entry):
            paths.add(path)
        for err in re.findall(r'"error":\s*"([^"]{1,160})"', entry):
            if err and err not in errors:
                errors.append(err)

    lines: list[str] = [f"Completed {len(entries)} earlier steps."]
    if tools:
        lines.append("Tools used: " + ", ".join(tools[:12]))
    if paths:
        lines.append("Files touched: " + ", ".join(sorted(paths)[:8]))
    if errors:
        lines.append("Errors seen: " + "; ".join(errors[:3]))
    return "\n".join(lines)


async def compact(
    history: list[str],
    *,
    llm: LLMClient | None = None,
    keep: int = KEEP_VERBATIM,
    threshold: int = COMPACT_AFTER,
) -> Compacted:
    """Split history into (summary of the old, verbatim recent).

    Never raises: if summarisation fails, the deterministic fallback is used, so
    a long task degrades gracefully instead of losing its past.
    """
    if len(history) <= threshold:
        return Compacted("", history)

    older, recent = history[:-keep], history[-keep:]

    if llm is not None:
        try:
            body = "\n\n".join(entry[:1200] for entry in older)[:12000]
            reply = await llm.chat(
                [
                    Message("system", SUMMARY_PROMPT),
                    Message("user", f"EXECUTION HISTORY TO COMPRESS:\n\n{body}"),
                ]
            )
            text = (reply.content or "").strip()
            if text:
                return Compacted(text[:2000], recent)
        except LLMError as exc:
            log.warning("compaction_failed", extra={"error": str(exc)[:200]})

    return Compacted(_extract_facts(older), recent)


def attention_notes(history: list[str], user_request: str) -> list[str]:
    """Short reminders that counter the failure modes of long runs.

    Long contexts make models drift: they forget the original ask and repeat
    calls that already failed. These notes are cheap and specific.
    """
    notes: list[str] = []

    if len(history) >= 10:
        notes.append(f'The owner originally asked: "{user_request[:200]}"')

    # Repeated identical failures: name them so the model stops retrying blindly.
    failures: dict[str, int] = {}
    for entry in history[-10:]:
        if '"ok": false' in entry.lower():
            for name in re.findall(r'"tool":\s*"([\w.]+)"', entry):
                failures[name] = failures.get(name, 0) + 1
    for tool, count in failures.items():
        if count >= 2:
            notes.append(
                f"{tool} has failed {count} times - change the approach "
                f"or ask the owner instead of repeating it"
            )

    return notes[:4]
