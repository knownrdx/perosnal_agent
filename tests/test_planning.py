"""Planning, self-verification, and long-run context handling.

These are what separate an agent that looks capable from one that is: it thinks
before acting, checks its own claims, and does not forget step 1 by step 20.
"""

from __future__ import annotations

import pytest

from app.agent.context import KEEP_VERBATIM, attention_notes, compact
from app.agent.planner import (
    Plan,
    looks_multi_step,
    make_plan,
    verify_completion,
)
from app.llm import LLMError
from app.llm.base import LLMClient, LLMResponse
from app.security import Permission


class ScriptedLLM(LLMClient):
    """Returns queued payloads; records what it was asked."""

    name = "scripted"

    def __init__(self, *payloads) -> None:
        import json

        self.queue = [json.dumps(p) if not isinstance(p, str) else p for p in payloads]
        self.prompts: list[str] = []

    async def chat(self, messages, *, temperature=None) -> LLMResponse:
        self.prompts.append("\n".join(m.content for m in messages))
        content = self.queue.pop(0) if self.queue else "{}"
        return LLMResponse(content=content, model=self.name)

    async def health(self):
        return {"ok": True}


class DeadLLM(LLMClient):
    name = "dead"

    async def chat(self, messages, *, temperature=None):
        raise LLMError("model unavailable")

    async def health(self):
        return {"ok": False}


# --------------------------------------------------------------------------- #
# Deciding what is worth planning
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", ["hi", "list files", "status"])
def test_trivial_requests_are_not_planned(text):
    assert not looks_multi_step(text)


@pytest.mark.parametrize(
    "text",
    [
        "download the report and send it to me",
        "check each invoice then summarise the totals",
        "fetch the logs, filter errors, and write a report",
    ],
)
def test_multi_step_requests_are_planned(text):
    assert looks_multi_step(text)


async def test_plan_is_built_and_stored(environment):
    llm = ScriptedLLM({
        "complexity": "standard",
        "steps": ["Download the file", "Validate it", "Send it to Telegram"],
        "success_criteria": "a message_id came back from Telegram",
        "risks": ["the URL may be dead"],
    })

    plan = await make_plan(
        "download the report and send it to me", permission=Permission.WRITE, llm=llm
    )

    assert plan.complexity == "standard"
    assert len(plan.steps) == 3
    assert "message_id" in plan.success_criteria
    # The planner must see the tool list, or it will invent capabilities.
    assert "AVAILABLE TOOLS" in llm.prompts[0]


async def test_simple_requests_skip_the_model_entirely(environment):
    llm = ScriptedLLM()
    plan = await make_plan("hi", permission=Permission.READ, llm=llm)
    assert plan.is_empty
    assert llm.prompts == [], "no model call should be made for trivial work"


async def test_a_broken_planner_never_blocks_the_task(environment):
    plan = await make_plan(
        "download the report and send it to me",
        permission=Permission.WRITE,
        llm=DeadLLM(),
    )
    assert plan.is_empty, "no plan, but the task must still be allowed to run"


async def test_plan_is_capped(environment):
    llm = ScriptedLLM({"complexity": "complex", "steps": [f"step {i}" for i in range(20)]})
    plan = await make_plan(
        "do a very long multi part job with many stages",
        permission=Permission.WRITE,
        llm=llm,
    )
    assert len(plan.steps) <= 6, "an unbounded plan would crowd out the real context"


# --------------------------------------------------------------------------- #
# Real models do not always use the key we asked for
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", ["steps", "plan", "actions", "tasks"])
async def test_alternate_step_keys_are_accepted(environment, key):
    """A good plan under a different key must not be silently thrown away."""
    llm = ScriptedLLM({"complexity": "standard", key: ["Download it", "Send it"]})
    plan = await make_plan(
        "download the report and send it to me", permission=Permission.WRITE, llm=llm
    )
    assert plan.steps == ["Download it", "Send it"]


async def test_steps_given_as_objects_are_accepted(environment):
    llm = ScriptedLLM({
        "complexity": "standard",
        "steps": [
            {"step": "Download the csv"},
            {"action": "Check it is not empty"},
            {"description": "Send it to Telegram"},
        ],
    })
    plan = await make_plan(
        "download the csv, check it, and send it", permission=Permission.WRITE, llm=llm
    )
    assert plan.steps == ["Download the csv", "Check it is not empty", "Send it to Telegram"]


async def test_empty_step_entries_are_dropped(environment):
    llm = ScriptedLLM({"complexity": "standard", "steps": ["Do it", "", "   ", {}]})
    plan = await make_plan(
        "download the report and send it to me", permission=Permission.WRITE, llm=llm
    )
    assert plan.steps == ["Do it"]


# --------------------------------------------------------------------------- #
# Self-verification
# --------------------------------------------------------------------------- #
async def test_unsupported_claim_is_rejected(environment):
    llm = ScriptedLLM({
        "verified": False,
        "reason": "no telegram_send_file call appears in the trace",
        "missing": ["deliver the file"],
    })

    result = await verify_completion(
        task_id="t1",
        user_request="send me the report",
        claim="I sent you the report.",
        trace=['[step 1] {"tool": "file_write", "tool_result": {"ok": true}}'],
        output_files=["output/report.pdf"],
        llm=llm,
    )

    assert not result.verified
    assert result.missing == ["deliver the file"]


async def test_supported_claim_passes(environment):
    llm = ScriptedLLM({"verified": True, "reason": "the trace shows the delivery"})
    result = await verify_completion(
        task_id="t1",
        user_request="send me the report",
        claim="Sent.",
        trace=['[step 1] {"tool": "telegram_send_file", "tool_result": {"ok": true}}'],
        output_files=[],
        llm=llm,
    )
    assert result.verified


async def test_honest_failure_is_accepted(environment):
    """Admitting something could not be done is a success, not a lie."""
    llm = ScriptedLLM({
        "verified": True, "reason": "the agent honestly reported it could not connect",
    })
    result = await verify_completion(
        task_id="t1",
        user_request="send me the report",
        claim="I could not reach the server, so nothing was sent.",
        trace=['[step 1] {"tool": "http_get", "tool_result": {"ok": false}}'],
        output_files=[],
        llm=llm,
    )
    assert result.verified


async def test_no_tool_calls_needs_no_verification(environment):
    """Answering a question from knowledge is legitimate."""
    llm = ScriptedLLM()
    result = await verify_completion(
        task_id="t1", user_request="what is 2+2", claim="4",
        trace=[], output_files=[], llm=llm,
    )
    assert result.verified
    assert llm.prompts == []


async def test_a_broken_verifier_trusts_the_agent(environment):
    """A down checker must not strand genuinely completed work."""
    result = await verify_completion(
        task_id="t1",
        user_request="send the report",
        claim="Sent.",
        trace=['[step 1] {"tool": "telegram_send_file"}'],
        output_files=[],
        llm=DeadLLM(),
    )
    assert result.verified


# --------------------------------------------------------------------------- #
# Long runs keep their early context
# --------------------------------------------------------------------------- #
async def test_short_history_is_left_alone(environment):
    history = [f"[step {i}] did a thing" for i in range(5)]
    packed = await compact(history)
    assert not packed.was_compacted
    assert packed.recent == history


async def test_long_history_is_summarised_not_truncated(environment):
    history = [
        '[step 1] {"tool": "http_get", "args": {"path": "downloads/data.csv"}}',
        *[f'[step {i}] {{"tool": "file_read"}}' for i in range(2, 25)],
    ]
    llm = ScriptedLLM("Downloaded downloads/data.csv, then read it 23 times.")

    packed = await compact(history, llm=llm)

    assert packed.was_compacted
    assert len(packed.recent) == KEEP_VERBATIM
    assert "data.csv" in packed.summary, "early facts must survive"


async def test_compaction_falls_back_without_a_model(environment):
    """Even offline, concrete facts must be preserved rather than dropped."""
    history = [
        '[step 1] {"tool": "http_get", "args": {"path": "downloads/report.pdf"}}',
        '[step 2] {"tool": "telegram_send_file", "error": "chat not found"}',
        *[f'[step {i}] {{"tool": "file_read"}}' for i in range(3, 22)],
    ]

    packed = await compact(history, llm=DeadLLM())

    assert packed.was_compacted
    assert "http_get" in packed.summary
    assert "downloads/report.pdf" in packed.summary
    # An early error must survive compaction; repeating a known-bad call is
    # exactly the failure mode this guards against.
    assert "chat not found" in packed.summary


# --------------------------------------------------------------------------- #
# Attention notes counter long-run drift
# --------------------------------------------------------------------------- #
def test_original_request_is_restated_on_long_runs():
    notes = attention_notes([f"[step {i}] x" for i in range(12)], "send me the invoice")
    assert any("send me the invoice" in n for n in notes)


def test_short_runs_get_no_reminders():
    assert attention_notes(["[step 1] x"], "do a thing") == []


def test_repeated_failures_are_called_out():
    history = [
        '[step 1] {"tool": "http_get", "tool_result": {"ok": false}}',
        '[step 2] {"tool": "http_get", "tool_result": {"ok": false}}',
        '[step 3] {"tool": "http_get", "tool_result": {"ok": false}}',
    ]
    notes = attention_notes(history, "fetch it")
    assert any("http_get has failed" in n for n in notes)


def test_successful_tools_are_not_called_out():
    history = [f'[step {i}] {{"tool": "file_write", "tool_result": {{"ok": true}}}}' for i in range(4)]
    assert not any("failed" in n for n in attention_notes(history, "write files"))
