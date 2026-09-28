"""The owner writes in whatever language suits him - the bot keeps up.

Two separate promises are tested here:

1. Messages in another script reach the DETERMINISTIC paths - the router's
   intent rules, the OTP thread's triggers, the upload-caption parser. Those
   exist so the bot works when the model is down, and until now a message in
   Bengali script matched none of them and was handed to the very model they
   are meant to bypass.
2. The model is TOLD which language to answer in, rather than left to guess.
"""

import pytest

from app.agent import bangla, language
from app.agent.router import Intent, classify
from app.automation import otp_bot
from app.automation.otp_caption import parse_caption

# Only the routing tests are async. A blanket pytestmark would mark the
# synchronous ones too, and pytest-asyncio warns on every one of them.
pytestmark = pytest.mark.filterwarnings("ignore::pytest.PytestWarning")


# --------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------- #

def test_the_language_is_named_so_a_model_recognises_it():
    assert language.detect("বট টা কি চলছে") == "Bengali"
    assert language.detect("is the bot running?") == "English"
    assert language.detect("قم بإنشاء ملف") == "Arabic"
    assert language.detect("एक रिपोर्ट बनाओ").startswith("Devanagari")
    assert language.detect("привет как дела").startswith("Cyrillic")
    assert language.detect("你好，机器人在运行吗") == "Chinese"
    assert language.detect("ファイルを作成して") == "Japanese"
    assert language.detect("สวัสดี") == "Thai"


def test_banglish_is_recognised_even_though_it_is_latin():
    """Script cannot separate it from English, so the words have to."""
    assert language.detect("amar file gulo kothay geche").startswith("Banglish")
    assert language.detect("bot ta ki ekhon cholche").startswith("Banglish")
    assert language.detect("please send me the report") == "English"


def test_urdu_is_told_apart_from_arabic():
    """Same script, different language - the reply must not switch."""
    assert language.detect("ٹھیک ہے فائل بھیجیں") == "Urdu"
    assert language.detect("قم بإنشاء ملف") == "Arabic"


def test_a_message_with_no_words_forces_no_language():
    """A bare number is not a language - answering it needs no instruction."""
    assert language.detect("20h") == ""
    assert language.detect("") == ""
    assert language.detect("   ") == ""
    assert language.directive("20h") == ""


def test_a_few_bengali_words_beat_many_english_ones():
    """Mixed messages are common; the non-Latin script decides."""
    assert language.detect("please restart the server এখন করো") == "Bengali"


def test_the_directive_names_the_language_and_forbids_a_third():
    """Naming it is the point; a Bengali question must not come back in Hindi."""
    assert "Bengali" in language.directive("বট টা কি চলছে")
    assert "Arabic" in language.directive("قم بإنشاء ملف")
    assert "Banglish" in language.directive("bot ta ki cholche")
    # English needs no warning - there is no third language to drift into
    # when the reply language is the model's default.
    for text in ("বট টা কি চলছে", "قم بإنشاء ملف", "bot ta ki cholche"):
        assert "third language" in language.directive(text)


# --------------------------------------------------------------------- #
# transliteration
# --------------------------------------------------------------------- #

def test_bengali_becomes_the_banglish_the_patterns_already_match():
    assert bangla.to_banglish("বন্ধ করো") == "bondho koro"
    assert bangla.to_banglish("শুরু করো") == "shuru koro"
    assert bangla.to_banglish("কত দূর হয়েছে") == "koto dur hoyeche"


def test_latin_text_is_returned_unchanged():
    """It runs on every message, so it must not damage what already works."""
    for text in ("start now", "bangladesh whatsapp 20h", "/status", ""):
        assert language.normalise(text) == text


def test_mixed_scripts_transliterate_token_by_token():
    assert language.normalise("bot ta ki চলছে") == "bot ta ki cholche"


def test_native_digits_become_ascii():
    """Times, durations and counts are all parsed as ASCII downstream."""
    assert language.normalise("২০ ঘন্টা") == "20 ghonta"
    assert language.normalise("٢٠") == "20"       # Arabic-Indic


def test_a_number_glued_to_a_word_is_split():
    """"৯টা" is one token but two values - "9" and "o'clock"."""
    assert bangla.to_banglish("৯টা") == "9 ta"
    assert bangla.to_banglish("২০ঘন্টা") == "20 ghonta"


# --------------------------------------------------------------------- #
# routing
# --------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_bengali_routes_without_calling_the_model():
    """The whole point: these rules must not need the LLM."""
    cases = [
        ("বট টা কি এখন চলছে?", None, Intent.CONTROL),
        ("কত দূর হয়েছে?", None, Intent.CONTROL),
        ("একটা রিপোর্ট বানাও", None, Intent.TASK),
        ("সব বাদ দাও", None, Intent.TASK),
        ("আর একটা বানাও", "task-1", Intent.FOLLOW_UP),
        ("কেমন আছো", None, Intent.CHAT),
    ]
    for text, active, expected in cases:
        decision = await classify(text, active_task_id=active, llm=_NoLLM())
        assert decision.intent is expected, f"{text} -> {decision.intent}"


@pytest.mark.asyncio
async def test_is_it_running_asks_about_work_and_does_not_start_any():
    """"cholche?" is a question; "chalu koro" is an instruction."""
    asking = await classify("bot ta ki ekhon cholche?", llm=_NoLLM())
    assert asking.intent is Intent.CONTROL

    telling = await classify("bot ta chalu koro ekhon", llm=_NoLLM())
    assert telling.intent is Intent.TASK


class _NoLLM:
    """Fails loudly if the deterministic rules did not settle it."""

    async def chat_json(self, *args, **kwargs):
        raise AssertionError("the router should not have needed the model")


# --------------------------------------------------------------------- #
# OTP triggers
# --------------------------------------------------------------------- #

def test_the_otp_triggers_answer_to_bengali():
    assert otp_bot.is_stop_trigger("বন্ধ করো")
    assert otp_bot.is_start_trigger("শুরু করো")
    assert otp_bot.is_status_trigger("স্ট্যাটাস")
    assert otp_bot.is_skip_trigger("বাদ")


def test_the_english_and_banglish_triggers_still_work():
    assert otp_bot.is_stop_trigger("stop")
    assert otp_bot.is_stop_trigger("bondho koro")
    assert otp_bot.is_start_trigger("start")


def test_an_ordinary_sentence_still_triggers_nothing():
    """Normalising must not make the triggers fire more loosely."""
    assert not otp_bot.is_stop_trigger("kalke bondho korbo naki bhabchi")
    assert not otp_bot.is_skip_trigger("bad dile ki hobe bolo to")


# --------------------------------------------------------------------- #
# captions
# --------------------------------------------------------------------- #

def test_a_bengali_caption_states_as_much_as_a_banglish_one():
    assert parse_caption("বাংলাদেশ হোয়াটসঅ্যাপ ২০ ঘন্টা") == {
        "tag": "WhatsApp", "country": "Bangladesh", "run_minutes": 1200,
    }


def test_bengali_start_and_stop_times():
    assert parse_caption("শুরু ২১:০০ বন্ধ ০৬:০০") == {
        "start_at": "21:00", "stop_at": "06:00",
    }


def test_the_time_of_day_word_supplies_the_am_pm():
    """"rat 9 ta" is how a time is said - there is no "pm" to find."""
    assert parse_caption("rat 9 ta porjonto")["stop_at"] == "21:00"
    assert parse_caption("shokal 6 ta theke")["start_at"] == "06:00"
    assert parse_caption("bikal 5 ta")["stop_at"] == "17:00"


def test_night_wraps_instead_of_adding_twelve():
    """"rat 2 ta" is 02:00. Nobody means 14:00 by it."""
    assert parse_caption("rat 2 ta porjonto")["stop_at"] == "02:00"
    assert parse_caption("rat 11 ta porjonto")["stop_at"] == "23:00"


def test_a_postposition_labels_the_time_before_it():
    """Bengali puts the label after the value, English before it.

    "sokal 6 ta theke rat 11 ta porjonto" used to come back with only a
    start: "theke" sat one character before the second time and won the
    tie-break meant for English labels like "start 21:00".
    """
    assert parse_caption("সকাল ৬টা থেকে রাত ১১টা পর্যন্ত") == {
        "start_at": "06:00", "stop_at": "23:00",
    }


def test_bengali_no_stop_is_understood():
    parsed = parse_caption("ফেসবুক কোন স্টপ নাই")
    assert parsed["tag"] == "Facebook"
    assert parsed["no_stop"] is True
    assert parsed["run_minutes"] == 0


def test_the_existing_banglish_captions_are_unchanged():
    """Everything above is additive - none of it may shift these."""
    assert parse_caption("bangladesh whatsapp 20h") == {
        "tag": "WhatsApp", "country": "Bangladesh", "run_minutes": 1200,
    }
    assert parse_caption("start 21:00 stop 06:00") == {
        "start_at": "21:00", "stop_at": "06:00",
    }
    assert parse_caption("uganda imo 12h") == {
        "tag": "IMO", "country": "Uganda", "run_minutes": 720,
    }
    assert parse_caption("") == {}
