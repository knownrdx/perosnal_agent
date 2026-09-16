"""Telegram inline-keyboard panel for the OTP automation.

These assert the two things that actually break in a button UI: callback
payloads that exceed Telegram's 64-byte limit (silently rejected at send
time, so it only shows up in production) and keyboards whose buttons point
at entries that no longer exist.
"""

from __future__ import annotations

import pytest

from app.automation import otp_bot
from app.telegram import otp_panel

# Applied per-test rather than module-wide: half of these are pure keyboard
# builders with no I/O, and marking a sync test asyncio just warns.
asyncio_test = pytest.mark.asyncio


async def _queue_one(environment, name: str = "bd.txt", prefix: str = "+880") -> dict:
    from app.config import get_settings
    from app.security import rel_path

    uploads = get_settings().workspace / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    path = uploads / name
    path.write_text("\n".join(f"{prefix}1711{i:06d}" for i in range(5)), encoding="utf-8")
    analysis = await otp_bot.enqueue_file(rel_path(path), name)
    return analysis["entries"][0]


def _all_callback_data(markup) -> list[str]:
    return [
        button.callback_data
        for row in markup.inline_keyboard
        for button in row
        if button.callback_data
    ]


def test_every_callback_payload_fits_telegrams_64_byte_limit():
    """Telegram rejects callback_data over 64 bytes - and the failure is at
    send time, not build time, so only a test catches it before the owner
    sees a button that does nothing.
    """
    entry_id = "a1b2c3d4"  # ids are short hex, but assert against the real shape
    markups = [
        otp_panel.service_keyboard(entry_id),
        otp_panel.interval_keyboard(10),
        otp_panel.cleanup_keyboard(False),
        otp_panel.control_keyboard(True),
        otp_panel.control_keyboard(False),
    ]
    for markup in markups:
        for data in _all_callback_data(markup):
            assert len(data.encode("utf-8")) <= 64, data


def test_service_keyboard_offers_every_choice_plus_a_way_out():
    markup = otp_panel.service_keyboard("abc123")
    labels = [b.text for row in markup.inline_keyboard for b in row]
    for service in otp_bot.SERVICE_CHOICES:
        assert service in labels
    # Without this the owner has to type to get rid of a wrong country.
    assert any("Bad dao" in label for label in labels)


def test_interval_keyboard_ticks_the_current_value():
    markup = otp_panel.interval_keyboard(15)
    labels = [b.text for row in markup.inline_keyboard for b in row]
    assert any(label.startswith("\u2705") and "15" in label for label in labels)
    assert sum(1 for label in labels if label.startswith("\u2705")) == 1


def test_control_keyboard_toggles_between_start_and_stop():
    running = [b.text for row in otp_panel.control_keyboard(True).inline_keyboard for b in row]
    stopped = [b.text for row in otp_panel.control_keyboard(False).inline_keyboard for b in row]
    assert any("Stop" in t for t in running)
    assert not any("Start" in t for t in running)
    assert any("Start" in t for t in stopped)


@asyncio_test
async def test_removal_keyboard_lists_each_country_once(environment):
    from app.config import get_settings
    from app.security import rel_path

    uploads = get_settings().workspace / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    # Two uploads, same country - the owner should see one button, not two.
    for name in ("a.txt", "b.txt"):
        path = uploads / name
        path.write_text("\n".join(f"+8801711{i:06d}" for i in range(3)), encoding="utf-8")
        await otp_bot.enqueue_file(rel_path(path), name)

    markup = otp_panel.removal_keyboard(await otp_bot.get_queue())
    assert markup is not None
    labels = [b.text for row in markup.inline_keyboard for b in row]
    assert len(labels) == 1
    assert "Bangladesh" in labels[0]


def test_removal_keyboard_is_none_when_there_is_nothing_to_remove():
    assert otp_panel.removal_keyboard([]) is None


@asyncio_test
async def test_status_text_reports_the_settings_the_buttons_change(environment):
    await _queue_one(environment)
    text = await otp_panel.status_text()

    # Intervals are per-country now and shown against RUNNING countries, so
    # the global header carries what is still global.
    assert "/st" in text
    assert "Bangladesh" in text
    assert "used/expired only" in text

    await otp_bot.set_cleanup_mode("force")
    assert "/frcd" in await otp_panel.status_text()


@asyncio_test
async def test_status_text_shows_each_running_country_on_its_own_schedule(environment):
    """The whole point of per-country timers is being able to SEE them."""
    from app.automation import otp_schedule
    from app.integrations.telegram_user import set_userbot

    entry = await _queue_one(environment)
    await otp_bot.set_queue_tag(entry["id"], "WhatsApp")

    class _Bot:
        def __init__(self) -> None:
            self._id = 100

        async def send_message(self, *a, **k):
            self._id += 1
            return {"message_id": self._id}

        async def send_file(self, *a, **k):
            self._id += 1
            return {"message_id": self._id}

        async def read_messages(self, *a, **k):
            self._id += 1
            return [{"id": self._id, "text": "Added.", "out": False}]

    set_userbot(_Bot())
    await otp_bot.start_automation()
    await otp_schedule.set_country_settings("Bangladesh", {"interval_minutes": 5})

    text = await otp_panel.status_text()
    assert "Bangladesh" in text
    assert "every 5m" in text
    assert "*" in text  # marked as customised


@asyncio_test
async def test_status_text_says_so_when_there_is_nothing_queued(environment):
    text = await otp_panel.status_text()
    assert "Nothing queued" in text


@asyncio_test
async def test_cleanup_mode_flips_the_force_delete_flag(environment):
    await otp_bot.set_cleanup_mode("force")
    assert (await otp_bot.get_config())["force_delete_before_add"] is True

    await otp_bot.set_cleanup_mode("used")
    assert (await otp_bot.get_config())["force_delete_before_add"] is False
