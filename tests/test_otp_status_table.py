"""The OTP status panel as a Telegram-HTML table.

Pins what makes a table in a chat message work or break: columns that line
up in the monospace block, every dynamic value escaped (one stray "<" and
Telegram rejects the whole message), only tags Telegram accepts, and a hard
cap so a big run still fits in one message.
"""

from __future__ import annotations

import html
import re
from datetime import timedelta

import pytest

from app.automation import otp_bot, otp_schedule
from app.db import repo
from app.db.base import session_scope
from app.telegram import otp_panel

pytestmark = pytest.mark.asyncio

# Column boundaries of the <pre> table: state(4) country(14) file(7)
# stock(7) every(6) ends(6) next. A space sits at each of these offsets on
# every line when the columns are aligned.
GAPS = (4, 19, 27, 35, 42, 49)


def _entry(i: int, country: str, **extra) -> dict:
    return {
        "id": f"{i:08x}",
        "batch_id": "b0",
        "path": "uploads/none.txt",
        "name": f"n{i}.txt",
        "country": country,
        "count": 7625 + i,
        "tag": "WhatsApp",
        "uploaded_at": "2026-10-05T00:00:00+00:00",
        **extra,
    }


async def _run(entries: list[dict]) -> None:
    await otp_bot._save_files(otp_bot.ACTIVE_KEY, entries)
    # Start immediately, so a country with no start time of its own is
    # running rather than waiting for the 04:00 default.
    await otp_bot.save_config({"enabled": True, "start_at": otp_bot.START_NOW})


async def _last_stock(stock: dict[str, int]) -> None:
    async with session_scope() as session:
        await repo.set_setting(session, otp_bot.LAST_RESULT_KEY, {
            "ok": True, "action": "skipped", "country_stock": stock,
            "ran_at": "2026-10-05T10:00:00+00:00", "error": "",
        })


def _table(text: str) -> list[str]:
    match = re.search(r"<pre>(.*?)</pre>", text, re.S)
    assert match, text
    return html.unescape(match.group(1)).split("\n")


def _row(text: str, country_prefix: str) -> str:
    return next(line for line in _table(text) if line[5:].startswith(country_prefix))


async def test_columns_line_up_for_every_row(environment):
    await _run([
        _entry(1, "Sudan"),
        _entry(2, "Saint Vincent And The Grenadines"),
        _entry(3, "Bonaire, Sint Eustatius and Saba"),
        _entry(4, "Sint Maarten (Dutch part)"),
    ])
    await _last_stock({"Sudan": 676, "Sint Maarten (Dutch part)": 12345})

    lines = _table(await otp_panel.status_text())

    assert lines[0].split() == ["St", "Country", "File", "Stock", "Every", "Ends", "Next"]
    assert len(lines) == 5
    for line in lines:
        for gap in GAPS:
            assert line[gap] == " ", (gap, line)
    # Long names are cut to the column, thousands get separators.
    assert "Saint Vincent." in lines[2]
    sudan = _row("<pre>" + "\n".join(lines) + "</pre>", "Sudan")
    assert sudan.startswith("RUN ")
    assert "7,626" in sudan and "676" in sudan
    assert "12,345" in _row("<pre>" + "\n".join(lines) + "</pre>", "Sint Maarten")


async def test_each_state_gets_its_own_code(environment):
    await _run([
        _entry(1, "Running Land"),
        _entry(2, "Paused Land"),
        _entry(3, "Finished Land", finished_at="2026-10-05T01:00:00+00:00"),
        _entry(4, "Spent Land", exhausted_at="2026-10-05T01:00:00+00:00"),
        _entry(5, "Held Land", held=True),
        _entry(6, "Waiting Land"),
    ])
    await otp_schedule.set_paused("Paused Land", True)
    # A finished country is also paused - DONE must still win over OFF.
    await otp_schedule.set_paused("Finished Land", True)
    later = (otp_schedule._now() + timedelta(hours=3)).astimezone(otp_schedule.DUBAI_TZ)
    start = later.strftime("%H:%M")
    await otp_schedule.set_country_settings("Waiting Land", {"start_at": start})

    text = await otp_panel.status_text()

    assert _row(text, "Running Land").startswith("RUN ")
    assert _row(text, "Paused Land").startswith("OFF ")
    assert _row(text, "Finished Lan").startswith("DONE")
    assert _row(text, "Spent Land").startswith("USED")
    assert _row(text, "Held Land").startswith("HELD")
    waiting = _row(text, "Waiting Land")
    assert waiting.startswith("WAIT") and waiting.endswith(start)
    # The legend explains exactly the codes in use.
    for code in ("RUN", "OFF", "DONE", "USED", "HELD", "WAIT"):
        assert f"{code} {otp_panel.STATE_LEGEND[code]}" in html.unescape(text)


async def test_dynamic_values_are_escaped(environment):
    await _run([_entry(1, "A&B <Tobago>")])
    await otp_bot.save_config({"target_bot": "@x<y>&z"})
    from app.config import get_settings
    from app.security import rel_path

    uploads = get_settings().workspace / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    path = uploads / "q.txt"
    path.write_text("+2341711000001\n", encoding="utf-8")
    await otp_bot.enqueue_file(rel_path(path), "q.txt", "nigeria <script> 20h")

    text = await otp_panel.status_text()

    assert "<Tobago>" not in text and "<y>" not in text
    assert "A&amp;B &lt;Tob" in text
    assert "@x&lt;y&gt;&amp;z" in text
    # Only tags Telegram's HTML mode accepts, and every one closed.
    tags = re.findall(r"</?([a-zA-Z]+)[^>]*>", text)
    assert set(tags) <= {"b", "i", "pre"}
    for tag in set(tags):
        assert text.count(f"<{tag}>") == text.count(f"</{tag}>")
    # The plain rendering (web chat, history) gets the raw characters back.
    assert "A&B <Tob" in otp_panel.html_to_plain(text)


async def test_a_huge_run_still_fits_one_message(environment):
    await _run([_entry(i, f"Country number {i}") for i in range(250)])
    from app.config import get_settings
    from app.security import rel_path

    uploads = get_settings().workspace / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    for n in range(40):
        path = uploads / f"q{n}.txt"
        path.write_text("+2341711000001\n", encoding="utf-8")
        await otp_bot.enqueue_file(rel_path(path), f"q{n}.txt", "20h")

    text = await otp_panel.status_text()

    assert len(text) <= otp_panel.MESSAGE_LIMIT
    assert text.count("<pre>") == 1 and text.count("</pre>") == 1
    assert re.search(r"\+\d+ more", text)
    # The running table is what the owner opened the panel for: it keeps
    # its rows before the waiting list does.
    assert "Running now (250)" in text


async def test_the_summary_lines_are_there(environment):
    await otp_bot.set_cleanup_mode("force")
    await otp_bot.save_config({"quota_threshold": 200})
    await _last_stock({"Sudan": 1})

    text = otp_panel.html_to_plain(await otp_panel.status_text())

    for label in ("Running:", "Target bot:", "Restock point: 200 left",
                  "Cleanup mode: wipe country first (/frcd)", "Last check: OK"):
        assert label in text
    assert "Nothing queued or running" in text
