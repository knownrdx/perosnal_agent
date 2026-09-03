"""Conversation router.

Every incoming message is classified before anything happens:

    CHAT       - a question or remark -> answer inline, no task, no worker
    TASK       - real work -> create a background Task
    FOLLOW_UP  - about work already running/just finished -> attach to that task
    CONTROL    - status/cancel style requests -> answer from state

This is what makes the bot feel like a chat instead of a ticket machine.
Cheap deterministic rules decide the obvious cases; the LLM is only consulted
when they are ambiguous, so ordinary chatting stays fast.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from app.db.models import ACTIVE_STATUSES
from app.llm import LLMError, Message
from app.logging_conf import get_logger

log = get_logger(__name__)


class Intent(str, Enum):
    CHAT = "CHAT"
    TASK = "TASK"
    FOLLOW_UP = "FOLLOW_UP"
    CONTROL = "CONTROL"


@dataclass(slots=True)
class Decision:
    intent: Intent
    reason: str = ""
    confidence: float = 1.0
    target_task_id: str | None = None


# --------------------------------------------------------------------------- #
# Deterministic signals
# --------------------------------------------------------------------------- #

# Clear "do something for me" verbs.
_TASK_PATTERNS = [
    r"\b(download|upload|fetch|scrape|crawl)\b",
    r"\b(send|forward|deliver|email|message)\s+(me|it|this|that|the|a|to)\b",
    r"\b(create|make|build|generate|write|draft|produce)\s+(a|an|the|me)\b",
    r"\b(schedule|remind|every\s+(day|morning|hour|week)|tomorrow at|at \d{1,2}(:\d{2})?\s*(am|pm)?)\b",
    r"\b(check|monitor|watch|track)\s+(the|this|that|my|for)\b",
    r"\b(run|execute|start)\s+(the|this|a|my)\b",
    r"\b(convert|compress|resize|rename|move|copy|delete)\b",
    r"\b(find|search)\s+.{0,30}\b(and|then)\b",
]

# Small talk / questions about the agent itself.
_CHAT_PATTERNS = [
    r"^\s*(hi|hey|hello|salam|assalam|yo|sup|thanks|thank you|thx|ok|okay|good|nice|great|cool)\b",
    r"^\s*(who|what|why|how|when|where|which)\s+(are|is|do|does|can|should|would|did)\b",
    r"\b(what can you do|who are you|how do you work|are you (there|ok|alive))\b",
    r"^\s*(kemon|kemon acho|ki khobor|ki koro|acho|thanks a lot)\b",
]

# Asking about existing work.
_CONTROL_PATTERNS = [
    r"\b(status|progress|how('?s| is) it going|are you done|finished yet|any update)\b",
    r"\b(cancel|stop|abort|kill)\s+(it|that|the task|this)\b",
    r"\b(what are you (doing|working on)|current task)\b",
]

# Continuations of a previous instruction.
_FOLLOW_UP_PATTERNS = [
    r"^\s*(and|also|then|plus|but|no,|actually|instead|wait)\b",
    r"\b(that one|the same|like before|as well|too)\b",
    r"^\s*(do it|go ahead|yes|yeah|yep|sure|continue|proceed|ok do it)\s*[.!]?\s*$",
    r"\b(change it|make it|use|try)\s+.{0,40}\b(instead|rather)\b",
]


def _matches(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


ROUTER_PROMPT = """You classify one message sent to a personal AI agent.

Reply with ONE JSON object, nothing else:

{"intent": "CHAT" | "TASK" | "FOLLOW_UP" | "CONTROL", "reason": "<5 words>"}

Definitions:
- TASK: the owner wants real work done (files, downloads, sending, scheduling,
  research, running something). It would take multiple steps or tools.
- CHAT: a question, greeting, opinion, or something you can answer in one
  message from knowledge or memory. No tools needed.
- FOLLOW_UP: it modifies, corrects, or continues the work described in the
  RECENT CONTEXT, rather than starting something new.
- CONTROL: asking about status/progress of existing work, or to stop it.

If the message could be answered in one sentence, prefer CHAT.
If unsure between TASK and FOLLOW_UP and there is active work, choose FOLLOW_UP.
"""


async def classify(
    text: str,
    *,
    active_task_id: str | None = None,
    active_task_request: str = "",
    recent_turns: list[str] | None = None,
    mode: str = "auto",
    llm=None,
) -> Decision:
    """Decide what to do with ``text``."""
    stripped = text.strip()

    # --- forced modes -------------------------------------------------- #
    if mode == "chat":
        return Decision(Intent.CHAT, "chat mode is pinned")
    if mode == "task":
        return Decision(Intent.TASK, "task mode is pinned")

    if not stripped:
        return Decision(Intent.CHAT, "empty message")

    # --- cheap, high-confidence rules ---------------------------------- #
    if _matches(_CONTROL_PATTERNS, stripped):
        return Decision(Intent.CONTROL, "asks about existing work",
                        target_task_id=active_task_id)

    if active_task_id and _matches(_FOLLOW_UP_PATTERNS, stripped):
        return Decision(Intent.FOLLOW_UP, "continues the active task",
                        target_task_id=active_task_id)

    # Very short messages are almost never new jobs.
    words = stripped.split()
    if len(words) <= 3 and not _matches(_TASK_PATTERNS, stripped):
        if active_task_id and _matches(_FOLLOW_UP_PATTERNS, stripped):
            return Decision(Intent.FOLLOW_UP, "short continuation",
                            target_task_id=active_task_id)
        return Decision(Intent.CHAT, "short remark")

    if _matches(_CHAT_PATTERNS, stripped) and not _matches(_TASK_PATTERNS, stripped):
        return Decision(Intent.CHAT, "greeting or question")

    if _matches(_TASK_PATTERNS, stripped):
        # An action phrased against running work is still a follow-up.
        if active_task_id and _matches(_FOLLOW_UP_PATTERNS, stripped):
            return Decision(Intent.FOLLOW_UP, "modifies the active task",
                            target_task_id=active_task_id)
        return Decision(Intent.TASK, "action verb detected")

    # --- ambiguous: ask the model -------------------------------------- #
    if llm is None:
        from app.llm import get_llm

        llm = get_llm()

    context_lines: list[str] = []
    if active_task_request:
        context_lines.append(f"ACTIVE WORK: {active_task_request[:300]}")
    for turn in (recent_turns or [])[-4:]:
        context_lines.append(turn[:200])
    context = "\n".join(context_lines) or "(no recent context)"

    try:
        data = await llm.chat_json(
            [
                Message("system", ROUTER_PROMPT),
                Message("user", f"RECENT CONTEXT:\n{context}\n\nMESSAGE:\n{stripped}"),
            ]
        )
    except (LLMError, Exception) as exc:  # noqa: BLE001 - never block on the router
        log.warning("router_llm_failed", extra={"error": str(exc)[:200]})
        # Safe default: treat it as a task so real work is never silently dropped.
        return Decision(Intent.TASK, "router unavailable, defaulting to task", 0.4)

    raw = str(data.get("intent", "")).upper().strip()
    reason = str(data.get("reason", ""))[:80]
    try:
        intent = Intent(raw)
    except ValueError:
        return Decision(Intent.TASK, "unrecognised intent, defaulting to task", 0.4)

    if intent is Intent.FOLLOW_UP and not active_task_id:
        return Decision(Intent.TASK, "no active work to follow up on", 0.6)

    return Decision(
        intent, reason, 0.8,
        target_task_id=active_task_id if intent in {Intent.FOLLOW_UP, Intent.CONTROL} else None,
    )
