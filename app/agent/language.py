"""Understand any language the owner writes in; always answer in English.

Two different problems live here, and conflating them is why the bot only
ever worked in English:

1. UNDERSTANDING a message well enough to act on it deterministically.
   The trigger words ("start", "stop", "status") and the caption parser are
   Latin-text patterns. A message in another script matches nothing and
   falls through to the model - which is the opposite of the point, since
   those triggers exist precisely so the bot keeps working when the model is
   down. Handled for Bengali by app/agent/bangla.py (via :func:`normalise`),
   which transliterates into the Banglish the patterns already match.

2. ANSWERING. The owner writes English, Banglish and Bengali script, but
   wants every reply in English - one language he can skim on a phone,
   whatever he typed. The standing rule lives in app/llm/prompts.py
   (REPLY_STYLE). This module adds the per-message nudge: models mirror the
   language of the message in front of them very strongly, so for a
   non-English message the model is told what it is reading AND that the
   answer is still English. Left unprompted, a Bengali question comes back in
   Bengali (or Hindi) no matter what the system prompt said earlier.

Detection is script-first because that is unambiguous and free. Romanised
languages that share the Latin alphabet cannot be told apart by script, so
Banglish is recognised by its function words instead.
"""

from __future__ import annotations

import re
import unicodedata

# The one language every reply is written in, whatever the owner typed.
REPLY_LANGUAGE = "English"

# Unicode block -> the name a model will recognise. Ordered by how likely the
# owner is to use them; ties do not matter because detection counts characters.
_SCRIPT_RANGES: tuple[tuple[str, str, tuple[tuple[int, int], ...]], ...] = (
    ("Bengali", "bn", ((0x0980, 0x09FF),)),
    ("Arabic", "ar", ((0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF))),
    ("Devanagari (Hindi/Nepali/Marathi)", "hi", ((0x0900, 0x097F),)),
    ("Urdu", "ur", ()),                      # Arabic script; disambiguated below
    ("Cyrillic (Russian/Ukrainian)", "ru", ((0x0400, 0x04FF),)),
    ("Chinese", "zh", ((0x4E00, 0x9FFF), (0x3400, 0x4DBF))),
    ("Japanese", "ja", ((0x3040, 0x309F), (0x30A0, 0x30FF))),
    ("Korean", "ko", ((0xAC00, 0xD7AF), (0x1100, 0x11FF))),
    ("Thai", "th", ((0x0E00, 0x0E7F),)),
    ("Hebrew", "he", ((0x0590, 0x05FF),)),
    ("Greek", "el", ((0x0370, 0x03FF),)),
    ("Tamil", "ta", ((0x0B80, 0x0BFF),)),
    ("Telugu", "te", ((0x0C00, 0x0C7F),)),
    ("Gujarati", "gu", ((0x0A80, 0x0AFF),)),
    ("Punjabi (Gurmukhi)", "pa", ((0x0A00, 0x0A7F),)),
    ("Sinhala", "si", ((0x0D80, 0x0DFF),)),
    ("Myanmar", "my", ((0x1000, 0x109F),)),
    ("Amharic (Ethiopic)", "am", ((0x1200, 0x137F),)),
)

# Banglish: Bengali spoken language typed in Latin letters. Not detectable by
# script, so it is recognised by the function words that carry almost every
# sentence the owner writes. Deliberately common words only - a rare word
# would make detection depend on the topic.
_BANGLISH_MARKERS = (
    "ache", "nai", "koro", "korbo", "korbe", "kore", "kori", "hobe", "holo",
    "hoyeche", "cholche", "cholbe", "shuru", "bondho", "koto", "kothay",
    "kokhon", "keno", "kemon", "amar", "tumar", "tomar", "ami", "tumi",
    "eta", "oita", "ekta", "shob", "sob", "valo", "bhalo", "thik", "bad",
    "dao", "diye", "theke", "porjonto", "tarpor", "abar", "jabe", "jay",
    "lagbe", "lage", "pari", "parbo", "parbe", "bolo", "bolbe", "dekho",
    "jani", "bujhi", "bujhlam", "ki", "na", "haa",
)
_BANGLISH_RE = re.compile(
    r"(?<![a-z])(" + "|".join(_BANGLISH_MARKERS) + r")(?![a-z])", re.IGNORECASE
)

# Urdu-specific letters, to tell it apart from Arabic (same script block).
_URDU_CHARS = set("ٹڈڑںھےۓ")


def _script_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for char in text:
        if char.isspace() or not char.isalpha():
            continue
        code = ord(char)
        for name, _code, ranges in _SCRIPT_RANGES:
            if any(low <= code <= high for low, high in ranges):
                counts[name] = counts.get(name, 0) + 1
                break
        else:
            if code < 0x0250:                       # Latin + Latin-1
                counts["Latin"] = counts.get("Latin", 0) + 1
    return counts


def detect(text: str) -> str:
    """The language/script the owner wrote in, named so a model recognises it.

    Returns "" when there is nothing to go on (numbers, emoji, an empty
    string) - callers treat that as "no instruction needed".
    """
    if not text or not text.strip():
        return ""

    counts = _script_counts(text)
    if not counts:
        return ""

    # A non-Latin script wins even when Latin characters outnumber it: a
    # message mixing English technical words with Bengali is a Bengali
    # message, and that is exactly when the model needs the reminder that
    # the answer is still English.
    non_latin = {name: n for name, n in counts.items() if name != "Latin"}
    if non_latin:
        name = max(non_latin, key=lambda key: non_latin[key])
        if name == "Arabic" and any(ch in _URDU_CHARS for ch in text):
            return "Urdu"
        return name

    # All-Latin: script cannot separate English from romanised languages.
    hits = len(set(m.group(1).lower() for m in _BANGLISH_RE.finditer(text)))
    words = re.findall(r"[a-zA-Z]{2,}", text)
    if hits >= 2 or (hits == 1 and len(words) <= 4):
        return "Banglish (Bengali written in Latin letters)"
    if not words:
        # Digits, punctuation, emoji or unit suffixes ("20h", "06:00", "ok?"):
        # there is no language here to name. Single letters do not count as
        # words for exactly this reason - "20h" is a duration.
        return ""
    return "English"


def directive(text: str) -> str:
    """A per-message system line: what the owner wrote in, answered in English.

    The standing English-only rule is REPLY_STYLE in app/llm/prompts.py. This
    is the reminder placed right beside the message, because models mirror
    the language in front of them and a rule stated once at the top loses to
    a Bengali question at the bottom. Naming the language also tells the model
    that "bot ta ki cholche" is Bengali to be understood, not noise.

    Empty for English (nothing to steer away from) and for a message with no
    words in it, like "20h".
    """
    language = detect(text)
    if not language or language == REPLY_LANGUAGE:
        return ""

    if language.startswith("Banglish"):
        language = "Banglish (Bengali typed in Latin letters)"
    return (
        f"The owner's message is in {language}. Understand it fully, but "
        f"reply in {REPLY_LANGUAGE} only - clear, friendly {REPLY_LANGUAGE}. "
        "Do not answer in Bengali script, Banglish, Hindi or any other "
        "language, even though the message is not in English."
    )


def normalise(text: str) -> str:
    """Text prepared for deterministic pattern matching.

    Unicode-normalised, native digits folded to ASCII, and Bengali script
    transliterated into the Banglish the trigger patterns already match.
    Latin input comes back effectively unchanged, so this is safe to call on
    every message.

    Returns ONE string, not the original plus a transliteration. Several
    triggers match exactly (`_normalize(text) in _SKIP_WORDS`) or are anchored
    to the start of the string, and both are broken by concatenating two
    forms: "বাদ" would become "bad বাদ", which equals neither. Transliterating
    token by token is enough for mixed input anyway - "bot ta ki চলছে" comes
    out "bot ta ki cholche", with the Latin words untouched.
    """
    if not text:
        return text

    text = unicodedata.normalize("NFC", text)
    text = _fold_digits(text)

    from app.agent import bangla

    if bangla.has_bengali(text):
        return bangla.to_banglish(text)
    return text


def _fold_digits(text: str) -> str:
    """Any Unicode decimal digit -> ASCII, e.g. '২০' -> '20', '٢٠' -> '20'.

    Times, durations and phone numbers all arrive in native digits when the
    owner's keyboard is set to his own language, and every parser downstream
    expects ASCII.
    """
    if text.isascii():
        return text
    out = []
    for char in text:
        if char.isdigit() and not char.isascii():
            value = unicodedata.digit(char, None)
            out.append(str(value) if value is not None else char)
        else:
            out.append(char)
    return "".join(out)
