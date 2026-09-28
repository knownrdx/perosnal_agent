"""Reply in whatever language the owner wrote in.

Two different problems live here, and conflating them is why the bot only
ever worked in English:

1. UNDERSTANDING a message well enough to act on it deterministically.
   The trigger words ("start", "stop", "status") and the caption parser are
   Latin-text patterns. A message in another script matches nothing and
   falls through to the model - which is the opposite of the point, since
   those triggers exist precisely so the bot keeps working when the model is
   down. Handled for Bengali by app/agent/bangla.py, which transliterates
   into the Banglish the patterns already match.

2. ANSWERING in the same language the owner used. That is this module. It
   does NOT try to translate anything - it tells the model what it is
   looking at and to match it. Modern models are good at this when asked
   explicitly and bad at it when left to guess, and the failure mode when
   they guess (a Bengali question answered in Hindi, or an English question
   answered in Bengali) is exactly what the owner saw.

Detection is script-first because that is unambiguous and free. Romanised
languages that share the Latin alphabet cannot be told apart by script, so
for those the model is simply told "match the owner's language" and left to
recognise it - which it does reliably, being the one thing it is good at.
"""

from __future__ import annotations

import re
import unicodedata

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
    # message, and answering it in English is the complaint, not the fix.
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
        # there is no language here to match, and forcing one onto a reply to
        # a bare number is worse than saying nothing. Single letters do not
        # count as words for exactly this reason - "20h" is a duration.
        return ""
    return "English"


def directive(text: str) -> str:
    """A system-prompt line telling the model which language to answer in.

    Empty when the message gives no signal, so nothing is forced onto a bare
    "ok" or a number.
    """
    language = detect(text)
    if not language:
        return ""

    if language.startswith("Banglish"):
        return (
            "The owner wrote in Banglish - Bengali speech typed in Latin "
            "letters. Reply in Banglish too, Latin letters only (never "
            "Bengali script, never Hindi or Urdu words). Keep technical "
            "words in English: file, server, bot, restart, error. "
            "English is acceptable if you cannot phrase it naturally in "
            "Banglish - but never switch to a third language."
        )
    if language == "English":
        return "The owner wrote in English. Reply in English."
    return (
        f"The owner wrote in {language}. Reply in {language}, in the same "
        "script they used. Keep technical terms in English where that is "
        "how they are normally written. English is acceptable if you cannot "
        "phrase it naturally - but never switch to a third language."
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
