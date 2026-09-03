"""Autonomy policy: when the agent may act without interrupting the owner.

The point of this module is to remove pointless prompts WITHOUT removing the
ones that matter, so both halves are tested: what it stops asking about, and
what it must never stop asking about.
"""

from __future__ import annotations

import pytest

from app.db import repo
from app.db.base import session_scope
from app.security import autonomy


# --------------------------------------------------------------------------- #
# Signatures: similar actions must collapse to one decision
# --------------------------------------------------------------------------- #
def test_same_kind_of_file_shares_a_signature():
    a = autonomy.signature("file_delete", {"path": "temp/report-jan.pdf"})
    b = autonomy.signature("file_delete", {"path": "temp/report-feb.pdf"})
    assert a == b == "file_delete:temp/*.pdf"


def test_different_folders_do_not_share_a_signature():
    assert autonomy.signature("file_delete", {"path": "temp/a.pdf"}) != autonomy.signature(
        "file_delete", {"path": "uploads/a.pdf"}
    )


def test_different_extensions_do_not_share_a_signature():
    assert autonomy.signature("file_delete", {"path": "temp/a.pdf"}) != autonomy.signature(
        "file_delete", {"path": "temp/a.db"}
    )


def test_shell_signature_is_the_command_not_its_arguments():
    a = autonomy.signature("shell_exec", {"command": "ls -la /data"})
    b = autonomy.signature("shell_exec", {"command": "ls /tmp"})
    assert a == b == "shell_exec:ls"


# --------------------------------------------------------------------------- #
# Acting without asking
# --------------------------------------------------------------------------- #
async def test_scratch_files_do_not_need_approval(environment):
    verdict = await autonomy.evaluate("file_delete", {"path": "temp/scratch.txt"})
    assert verdict.allow
    assert "scratch" in verdict.reason


async def test_explicit_request_does_not_need_approval(environment):
    """Asking permission for what the owner literally just asked for is noise."""
    verdict = await autonomy.evaluate(
        "file_delete",
        {"path": "output/old_report.pdf"},
        user_request="delete the old_report file, it is wrong",
    )
    assert verdict.allow
    assert "asked" in verdict.reason


async def test_folder_level_instruction_counts_as_explicit(environment):
    verdict = await autonomy.evaluate(
        "file_delete",
        {"path": "output/a.txt"},
        user_request="clear the output folder please",
    )
    assert verdict.allow


async def test_cancel_is_allowed_when_the_owner_said_cancel(environment):
    verdict = await autonomy.evaluate(
        "task_cancel", {"task_id": "abc123"}, user_request="cancel task abc123"
    )
    assert verdict.allow


# --------------------------------------------------------------------------- #
# Still asking - the cases that must never be silently automated
# --------------------------------------------------------------------------- #
async def test_unrequested_deletion_outside_scratch_still_asks(environment):
    verdict = await autonomy.evaluate(
        "file_delete", {"path": "output/report.pdf"}, user_request="summarise my week"
    )
    assert not verdict.allow


async def test_credentials_are_never_auto_approved(environment):
    """Even an explicit instruction must not silently touch secrets."""
    verdict = await autonomy.evaluate(
        "file_delete",
        {"path": "uploads/my_credentials.json"},
        user_request="delete my_credentials.json",
    )
    assert not verdict.allow
    assert "credential" in verdict.reason


async def test_env_files_are_protected(environment):
    verdict = await autonomy.evaluate(
        "file_delete", {"path": "temp/.env"}, user_request="delete the .env"
    )
    assert not verdict.allow


async def test_a_similar_word_does_not_authorise_a_different_file(environment):
    """'delete the draft' must not license deleting the invoice."""
    verdict = await autonomy.evaluate(
        "file_delete", {"path": "output/invoice.pdf"}, user_request="delete the draft"
    )
    assert not verdict.allow


# --------------------------------------------------------------------------- #
# Learning from the owner's decisions
# --------------------------------------------------------------------------- #
async def test_repeated_approvals_stop_the_question(environment):
    args = {"path": "output/weekly.csv"}
    for _ in range(autonomy.AUTO_APPROVE_AFTER):
        await autonomy.remember_decision("file_delete", args, approved=True)

    verdict = await autonomy.evaluate("file_delete", args)
    assert verdict.allow and verdict.learned


async def test_two_approvals_are_not_yet_enough(environment):
    args = {"path": "output/weekly.csv"}
    for _ in range(autonomy.AUTO_APPROVE_AFTER - 1):
        await autonomy.remember_decision("file_delete", args, approved=True)

    assert not (await autonomy.evaluate("file_delete", args)).allow


async def test_one_rejection_revokes_earned_trust(environment):
    """Trust is slow to earn and instant to lose."""
    args = {"path": "output/weekly.csv"}
    for _ in range(5):
        await autonomy.remember_decision("file_delete", args, approved=True)
    assert (await autonomy.evaluate("file_delete", args)).allow

    await autonomy.remember_decision("file_delete", args, approved=False)

    verdict = await autonomy.evaluate("file_delete", args)
    assert not verdict.allow
    assert "refused" in verdict.reason


async def test_learning_generalises_to_similar_files(environment):
    """Approving three monthly reports covers the fourth."""
    for month in ("jan", "feb", "mar"):
        await autonomy.remember_decision(
            "file_delete", {"path": f"output/report-{month}.pdf"}, approved=True
        )

    verdict = await autonomy.evaluate("file_delete", {"path": "output/report-apr.pdf"})
    assert verdict.allow and verdict.learned


async def test_learning_does_not_leak_across_folders(environment):
    for month in ("jan", "feb", "mar"):
        await autonomy.remember_decision(
            "file_delete", {"path": f"output/report-{month}.pdf"}, approved=True
        )

    assert not (
        await autonomy.evaluate("file_delete", {"path": "uploads/report-apr.pdf"})
    ).allow


# --------------------------------------------------------------------------- #
# Autonomy levels
# --------------------------------------------------------------------------- #
async def test_paranoid_mode_asks_about_everything(environment, monkeypatch):
    monkeypatch.setattr(environment, "autonomy_level", "paranoid", raising=False)
    verdict = await autonomy.evaluate(
        "file_delete", {"path": "temp/scratch.txt"}, user_request="delete scratch.txt"
    )
    assert not verdict.allow


async def test_high_mode_acts_but_still_guards_secrets(environment, monkeypatch):
    monkeypatch.setattr(environment, "autonomy_level", "high", raising=False)

    assert (await autonomy.evaluate("file_delete", {"path": "output/x.pdf"})).allow
    assert not (
        await autonomy.evaluate("file_delete", {"path": "uploads/secret_key.txt"})
    ).allow


# --------------------------------------------------------------------------- #
# Human-readable descriptions
# --------------------------------------------------------------------------- #
def test_descriptions_are_plain_language():
    assert autonomy.describe("file_delete", {"path": "temp/a.txt"}) == "delete temp/a.txt"
    assert autonomy.describe("task_cancel", {"task_id": "x1"}) == "cancel task x1"
