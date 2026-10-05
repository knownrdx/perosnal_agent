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
    assert any("Remove" in label for label in labels)


def test_interval_keyboard_ticks_the_current_value():
    markup = otp_panel.interval_keyboard(15)
    labels = [b.text for row in markup.inline_keyboard for b in row]
    assert any(label.startswith("\u2705") and "15" in label for label in labels)
    assert sum(1 for label in labels if label.startswith("\u2705")) == 1


def test_interval_keyboard_offers_every_global_choice_plus_custom():
    markup = otp_panel.interval_keyboard(1)
    data = _all_callback_data(markup)
    for minutes in otp_bot.INTERVAL_CHOICES:
        assert f"otp:int:{minutes}" in data
    assert "otp:intc:" in data
    # Global only: nothing in it names a country.
    assert not any(d.startswith(("otp:cint", "otp:pickc")) for d in data)


def test_a_typed_interval_shows_on_the_custom_button():
    """7 has no button of its own - without this the current value is
    nowhere on the keyboard."""
    labels = [b.text for row in otp_panel.interval_keyboard(7).inline_keyboard for b in row]
    assert any(label.startswith("\u2705") and "7 min" in label and "Custom" in label
               for label in labels)
    assert sum(1 for label in labels if label.startswith("\u2705")) == 1


def test_the_country_panel_has_no_interval_button():
    markup = otp_panel.country_field_keyboard(otp_panel.country_key("Bangladesh"), "Bangladesh")
    labels = [b.text for row in markup.inline_keyboard for b in row]
    assert not any("interval" in label.lower() for label in labels)
    assert not any(d.split(":")[1] in {"pickc", "cint", "cintc"}
                   for d in _all_callback_data(markup))
    assert not hasattr(otp_panel, "country_interval_keyboard")


def test_the_main_panel_keeps_a_global_interval_button():
    data = _all_callback_data(otp_panel.control_keyboard(True))
    assert "otp:ask_int:" in data


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
    text = otp_panel.html_to_plain(await otp_panel.status_text())

    assert "/st" in text
    assert "Bangladesh" in text
    # The wipe choice is shown in the same words as the Cleanup buttons.
    labels = dict(otp_bot.CLEANUP_CHOICES)
    assert labels["used"] in text

    await otp_bot.set_cleanup_mode("force")
    text = otp_panel.html_to_plain(await otp_panel.status_text())
    assert labels["force"] in text and "/frcd" in text


@asyncio_test
async def test_no_panel_text_mentions_useddelete(environment):
    """The cleanup command is gone entirely - nothing the panel shows may
    still offer or promise it."""
    await _queue_one(environment)
    texts = [await otp_panel.status_text()]
    for force in (False, True):
        texts += [b.text for row in otp_panel.cleanup_keyboard(force).inline_keyboard
                  for b in row]
    texts += [b.text for row in otp_panel.control_keyboard(True).inline_keyboard for b in row]
    await otp_bot.set_cleanup_mode("used")
    texts.append(await otp_panel.status_text())

    assert not any("useddelete" in text.lower() for text in texts)
    assert not any("cleanup_command" in text for text in texts)


@asyncio_test
async def test_status_text_shows_one_interval_for_every_country(environment):
    """One /st checks every country: the interval is said once, for all of
    them, and a country with settings of its own is still marked."""
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
    await otp_bot.save_config({"interval_minutes": 2})
    await otp_schedule.set_country_settings("Bangladesh", {"quota_threshold": 500})

    text = otp_panel.html_to_plain(await otp_panel.status_text())
    assert "Bangladesh*" in text  # marked as customised
    assert "* custom settings" in text
    assert text.count("every 2 min (all countries)") == 1
    assert "Every" not in text


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
