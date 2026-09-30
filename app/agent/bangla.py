"""Bengali-script input: normalise it so the rest of the code sees one thing.

The owner types three ways, often in the same message:

    Banglish   "bot ta ki ekhon cholche"      (Bengali speech, Latin letters)
    Bangla     "বট টা কি এখন চলছে"             (Bengali script)
    English    "is the bot running"

Every deterministic trigger in this codebase - the OTP thread's start/stop
words, the router's intent patterns, the upload-caption parser - was written
against Latin text only, so a message in Bengali script matched nothing and
fell through to the LLM. That is exactly backwards: those triggers exist so
the bot keeps working WITHOUT the model, and the model is worse at Bengali
script than at anything else.

So: transliterate Bengali script into the Banglish the patterns already
match, and map Bengali digits to ASCII. One function, called at the edge,
and every existing pattern keeps working unchanged.

This is deliberately NOT a general transliterator. It only has to be good
enough that "চালু করো" comes out close enough to "chalu koro" for the
existing word lists to hit, and it must never mangle Latin text that is
already fine.
"""

from __future__ import annotations

import re

# Bengali-Arabic digits -> ASCII. Phone numbers, times, counts and "20h" all
# arrive this way when the owner's keyboard is in Bengali.
_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

# Whole words first: the ones that carry meaning for a trigger, spelled the
# way the Latin patterns already expect. Checked before letter-by-letter
# transliteration, because a word-level match is always more accurate.
_WORDS: dict[str, str] = {
    # start / stop
    "শুরু": "shuru", "চালু": "chalu", "চালাও": "chalao", "শুরুকরো": "shuru koro",
    "বন্ধ": "bondho", "থামাও": "thamao", "অফ": "off", "বন্ধকরো": "bondho koro",
    # status / progress
    "অবস্থা": "obostha", "কতদূর": "koto dur", "শেষ": "shesh", "হয়েছে": "hoyeche",
    "স্ট্যাটাস": "status", "খবর": "khobor", "চলছে": "cholche",
    # doing things
    "বানাও": "banao", "বানিয়ে": "baniye", "তৈরি": "toiri", "পাঠাও": "pathao",
    "পাঠিয়ে": "pathiye", "নামাও": "namao", "খুঁজে": "khuje", "খোঁজ": "khoj",
    "দেখাও": "dekhao", "দেখো": "dekho", "লেখো": "likho", "লিখে": "likhe",
    "করো": "koro", "কর": "koro", "দাও": "dao", "নাও": "nao",
    "মুছে": "muche", "মুছো": "mucho", "ডিলিট": "delete", "সরাও": "sorao",
    "যোগ": "jog", "সেট": "set", "আপডেট": "update", "ঠিক": "thik",
    "বাদ": "bad", "বাতিল": "batil",
    # questions / chat
    "কি": "ki", "কী": "ki", "কেন": "keno", "কীভাবে": "kivabe", "কিভাবে": "kivabe",
    "কখন": "kokhon", "কোথায়": "kothay", "কোন": "kon", "কত": "koto",
    "কেমন": "kemon", "আছো": "acho", "আছে": "ache", "নাই": "nai", "নেই": "nai",
    "ধন্যবাদ": "dhonnobad", "আচ্ছা": "accha", "হ্যাঁ": "haa", "হ্যা": "haa",
    "না": "na", "বুঝলাম": "bujhlam", "ভালো": "valo", "ভাল": "valo",
    # continuation
    "আর": "ar", "আরো": "aro", "আরেকটা": "arekta", "একটা": "ekta",
    "তারপর": "tarpor", "আবার": "abar", "ওটা": "oita", "ওটাও": "oitao",
    "সব": "sob", "এই": "ei", "সেই": "sei",
    # time
    "ঘণ্টা": "ghonta", "ঘন্টা": "ghonta", "মিনিট": "min", "দিন": "din",
    "সময়": "somoy", "থেকে": "theke", "পর্যন্ত": "porjonto", "পরে": "pore",
    "এখন": "ekhon", "এখনই": "ekhoni", "এখুনি": "ekhoni",
    "আজ": "aj", "কাল": "kal", "রাত": "rat", "সারারাত": "sararat",
    "সারাদিন": "sara din", "লিমিট": "limit", "ডিফল্ট": "default",
    "টা": "ta", "স্টপ": "stop", "কোন": "kono", "কোনো": "kono",
    "সকাল": "sokal", "দুপুর": "dupur", "বিকাল": "bikal", "সন্ধ্যা": "sondha",
    # OTP domain
    "নাম্বার": "number", "নম্বর": "number", "ফাইল": "file", "দেশ": "desh",
    "স্টক": "stock", "সার্ভিস": "service", "ট্যাগ": "tag", "বট": "bot",
    "কিউ": "queue", "লিস্ট": "list", "হেল্প": "help", "সাহায্য": "help",
    # Service names, as the owner actually types them. Letter-level
    # transliteration mangles these ("হোয়াটসঅ্যাপ" -> "hoj়atsojap"), and a
    # caption naming a service is the single most useful thing it can do.
    "হোয়াটসঅ্যাপ": "whatsapp", "হোয়াটসএপ": "whatsapp", "ওয়াটসঅ্যাপ": "whatsapp",
    "টেলিগ্রাম": "telegram", "ফেসবুক": "facebook", "গুগল": "google",
    "ইনস্টাগ্রাম": "instagram", "ইন্সটাগ্রাম": "instagram",
    "সিগন্যাল": "signal", "ইমো": "imo", "ভাইবার": "viber",
    "টিকটক": "tiktok", "টুইটার": "twitter",
    # Countries the owner ships to most. Same reasoning: these decide which
    # stock a file joins, and a letter-level guess would not match the bot's
    # own spelling.
    "বাংলাদেশ": "bangladesh", "ভারত": "india", "ইন্ডিয়া": "india",
    "পাকিস্তান": "pakistan", "ইন্দোনেশিয়া": "indonesia",
    "নাইজেরিয়া": "nigeria", "কেনিয়া": "kenya", "উগান্ডা": "uganda",
    "ফিলিপাইন": "philippines", "ভিয়েতনাম": "vietnam", "মিশর": "egypt",
    "নেপাল": "nepal", "শ্রীলঙ্কা": "sri lanka", "মায়ানমার": "myanmar",
    "আমেরিকা": "united states", "যুক্তরাষ্ট্র": "united states",
    "ব্রাজিল": "brazil", "রাশিয়া": "russia", "তুরস্ক": "turkey",
}

# Letter-level fallback for everything not in the word list. Conservative and
# lossy on purpose - it only has to land close enough for a word-boundary
# pattern match, and being readable matters more than being reversible.
_CHARS: dict[str, str] = {
    "অ": "o", "আ": "a", "ই": "i", "ঈ": "i", "উ": "u", "ঊ": "u", "ঋ": "ri",
    "এ": "e", "ঐ": "oi", "ও": "o", "ঔ": "ou",
    "ক": "k", "খ": "kh", "গ": "g", "ঘ": "gh", "ঙ": "ng",
    "চ": "ch", "ছ": "chh", "জ": "j", "ঝ": "jh", "ঞ": "n",
    "ট": "t", "ঠ": "th", "ড": "d", "ঢ": "dh", "ণ": "n",
    "ত": "t", "থ": "th", "দ": "d", "ধ": "dh", "ন": "n",
    "প": "p", "ফ": "ph", "ব": "b", "ভ": "bh", "ম": "m",
    "য": "j", "র": "r", "ল": "l", "শ": "sh", "ষ": "sh", "স": "s", "হ": "h",
    "ড়": "r", "ঢ়": "rh", "য়": "y", "ৎ": "t", "ং": "ng", "ঃ": "h", "ঁ": "",
    # vowel signs
    "া": "a", "ি": "i", "ী": "i", "ু": "u", "ূ": "u", "ৃ": "ri",
    "ে": "e", "ৈ": "oi", "ো": "o", "ৌ": "ou",
    "্": "",          # hasant: joins consonants, drop it
    "\u200c": "", "\u200d": "",   # zero-width joiners
    "।": ".", "॥": ".",
}

_BENGALI_RE = re.compile(r"[\u0980-\u09FF]")


def has_bengali(text: str) -> bool:
    """True if the text contains any Bengali-script character."""
    return bool(_BENGALI_RE.search(text or ""))


def normalise_digits(text: str) -> str:
    """Bengali digits -> ASCII, e.g. '২০ঘন্টা' -> '20ghonta'."""
    return (text or "").translate(_DIGITS)


def _transliterate_word(word: str) -> str:
    if word in _WORDS:
        return _WORDS[word]
    # Digits and Bengali text arrive glued together - "৯টা" (9 o'clock),
    # "২০ঘন্টা" (20 hours) - and the caption parser reads times and durations
    # with patterns like r"(\d+)\s*(h|ghonta)". Splitting on the boundary lets
    # the word list resolve the text half instead of letter-mangling it.
    # The tail must be NON-digit: "(\d+)(.+)" is not anchored enough and
    # backtracks to split "20" into "2 0", which then parses as two numbers.
    match = re.match(r"^(\d+)(\D.*)$", word)
    if match:
        return match.group(1) + " " + _transliterate_word(match.group(2))
    return "".join(_CHARS.get(ch, ch) for ch in word)


def to_banglish(text: str) -> str:
    """Bengali script -> Banglish, leaving Latin text untouched.

    Word-level mapping first (accurate for the words that drive a trigger),
    then letter-level for the rest. Latin/ASCII input is returned unchanged
    apart from digit normalisation, so this is safe to call on everything,
    and mixed-script input works token by token: "bot ta ki চলছে" comes out
    "bot ta ki cholche".
    """
    if not text:
        return text
    text = normalise_digits(text)
    if not has_bengali(text):
        return text

    out: list[str] = []
    for token in re.split(r"(\s+)", text):
        if not token.strip():
            out.append(token)
            continue
        # Strip trailing punctuation so "চলছে?" still matches the word list.
        head = token.rstrip("?!.,;:")
        tail = token[len(head):]
        out.append(_transliterate_word(head) + tail)
    return "".join(out)
