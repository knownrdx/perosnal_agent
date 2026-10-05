"""Deciding that two country names mean the same country.

The name we hold came from the phone-prefix table; the one in the bot's
reply came from the bot. They describe the same place in different words:

    ours              theirs
    Congo (DRC)       DR Congo
    Ivory Coast       Côte d'Ivoire
    Myanmar           Myanmar (Burma)
    Turkey            Türkiye

The old rule compared letters-only and asked whether either was a PREFIX of
the other. "congodrc" and "drcongo" are not, so the lookup missed, an empty
result read as ZERO STOCK, and the automation cleaned up and re-added the
whole file - every cycle, forever. The owner watched 16,499 duplicates get
skipped every five minutes.

Prefix matching is also too loose the other way: "Niger" is a prefix of
"Nigeria", and merging two countries' stock is worse than missing a match.

The approach here is therefore in three parts:

1. Both names are reduced to plain lower-case words first: accents folded
   off ("Côte" -> "cote", where it used to become "cte"), "&" read as
   "and", apostrophes closed up.
2. A table of the names that genuinely collide - the two Congos, the four
   Guineas, the Sudans, the Koreas, China and the places written "...,
   China". Inside those families a name is identified EXPLICITLY, by
   listing every spelling of each country; two names match only if they
   resolve to the same entry, and anything in a family that cannot be
   resolved matches nothing.
3. For everything else, compare the significant words. This handles the
   long tail - "Central African Rep." vs "Central African Republic",
   "Viet Nam" vs "Vietnam" - without a table of 245 countries. A short
   alias table covers the names whose forms share no word at all
   ("Laos" / "Lao PDR", "Vatican" / "Holy See").
"""

from __future__ import annotations

import re
import unicodedata

# Words that carry no identity - they appear in dozens of official names and
# matching on them alone would merge unrelated countries. "sar" (Special
# Administrative Region), "pdr" (People's Democratic Republic) and "fed" are
# abbreviations of words already in here.
_NOISE = frozenset({
    "the", "of", "and", "republic", "rep", "democratic", "dem", "people",
    "peoples", "state", "states", "kingdom", "united", "federal",
    "federation", "fed", "islamic", "arab", "new", "isle", "island",
    "islands", "territory", "territories", "province", "saint", "st",
    "sar", "pdr",
})

# Names the same place is genuinely known by, where the two forms share no
# significant word. Anything else the word comparison already handles.
#
# A name is looked up here by all its letters AND by its significant words
# alone, so "Kingdom of Eswatini", "Lao People's Democratic Republic" and
# "Russian Federation" reach "eswatini", "lao" and "russian" without every
# official form being listed. Two names that are BOTH found here match only
# if they are in the same entry - which is what keeps "Ireland" away from
# "United Kingdom of Great Britain and Northern Ireland". A spelling must
# therefore appear in one entry only.
_ALIASES: tuple[frozenset[str], ...] = (
    frozenset({"cotedivoire", "ivorycoast"}),
    frozenset({"myanmar", "burma"}),
    frozenset({"netherlands", "holland"}),
    frozenset({"czechia", "czech", "czechrepublic"}),
    frozenset({"eswatini", "swaziland"}),
    frozenset({"timorleste", "easttimor"}),
    frozenset({"capeverde", "caboverde"}),
    frozenset({"unitedarabemirates", "uae", "emirates"}),
    frozenset({"unitedstates", "unitedstatesofamerica", "usa", "us", "america"}),
    frozenset({"unitedkingdom", "uk", "britain", "greatbritain",
               "unitedkingdomofgreatbritainandnorthernireland",
               "northernireland"}),
    frozenset({"ireland", "eire", "republicofireland"}),
    frozenset({"turkey", "turkiye"}),
    frozenset({"laos", "lao"}),
    frozenset({"kyrgyzstan", "kyrgyz", "kirghizia"}),
    frozenset({"macao", "macau"}),
    frozenset({"vatican", "vaticancity", "holysee"}),
    frozenset({"palestine", "palestinian", "occupiedpalestinian"}),
    frozenset({"northmacedonia", "macedonia", "fyrom",
               "macedoniaformeryugoslav", "formeryugoslavmacedonia"}),
    frozenset({"russia", "russian"}),
    frozenset({"syria", "syrian"}),
    frozenset({"taiwan", "chinesetaipei"}),
    # Entries for places whose names CONTAIN another country's name. They
    # are here to be known, not to be renamed: otherwise the word test files
    # "Caribbean Netherlands" under Netherlands and "South Georgia and the
    # South Sandwich Islands" under Georgia.
    frozenset({"bonairesinteustatiusandsaba", "caribbeannetherlands", "bes"}),
    frozenset({"netherlandsantilles"}),
    frozenset({"georgia"}),
    frozenset({"southgeorgia", "southgeorgiaandthesouthsandwichislands",
               "southgeorgiasouthsandwich"}),
)

# Countries whose names contain each other. Word comparison cannot separate
# these, so every spelling is listed against its country and a name inside a
# family matches ONLY a name mapped to the same country. A spelling in one
# of these families that is not listed matches nothing rather than guessing,
# because the cost of guessing here is numbers filed under the wrong stock.
#
# Spellings are matched on all their letters, never with noise words
# removed: "Korea, Democratic People's Republic of" without its noise is
# just "korea", which is the OTHER Korea.
_FAMILIES: tuple[dict[str, str], ...] = (
    {   # Congo
        "congodrc": "cd", "drcongo": "cd", "drccongo": "cd", "drc": "cd",
        "democraticrepublicofthecongo": "cd", "democraticrepublicofcongo": "cd",
        "congokinshasa": "cd", "congodemrep": "cd", "congodemocraticrepublic": "cd",
        "congothedemocraticrepublicofthe": "cd",
        "congodemocraticrepublicofthe": "cd", "congodr": "cd",
        "thedemocraticrepublicofcongo": "cd",       # libphonenumber's own name
        "thedemocraticrepublicofthecongo": "cd",
        "zaire": "cd",
        "congo": "cg", "republicofthecongo": "cg", "republicofcongo": "cg",
        "congobrazzaville": "cg", "congorep": "cg", "congorepublic": "cg",
        "congorepublicofthe": "cg",
    },
    {   # Guinea
        "guinea": "gn", "republicofguinea": "gn",
        "guineabissau": "gw",
        "equatorialguinea": "gq",
        "papuanewguinea": "pg",
    },
    {   # Sudan
        "sudan": "sd", "southsudan": "ss",
    },
    {   # Korea
        "southkorea": "kr", "republicofkorea": "kr", "koreasouth": "kr",
        "korea": "kr", "korearepublic": "kr", "korearepublicof": "kr",
        "korearep": "kr", "skorea": "kr",
        "northkorea": "kp", "koreanorth": "kp", "nkorea": "kp",
        "democraticpeoplesrepublicofkorea": "kp",
        "koreademocraticpeoplesrepublicof": "kp", "koreadempeoplesrep": "kp",
        "dprk": "kp", "dprkorea": "kp", "koreadpr": "kp", "koreadprk": "kp",
    },
    {   # Niger
        "niger": "ne", "nigeria": "ng",
    },
    {   # Dominica
        "dominica": "dm", "dominicanrepublic": "do",
    },
    {   # Samoa
        "samoa": "ws", "americansamoa": "as",
    },
    {   # Virgin Islands
        "virginislands": "vi", "usvirginislands": "vi",
        "unitedstatesvirginislands": "vi", "virginislandsus": "vi",
        "virginislandsusa": "vi", "usvi": "vi",
        "britishvirginislands": "vg", "virginislandsbritish": "vg",
        "virginislandsuk": "vg", "bvi": "vg",
    },
    {   # China - "Taiwan, Province of China" and "Hong Kong SAR China"
        # contain all of "China", so the word test would hand China's stock
        # to whichever of them the bot listed first.
        "china": "cn", "prc": "cn", "peoplesrepublicofchina": "cn",
        "chinapeoplesrepublicof": "cn", "chinaprc": "cn",
        "mainlandchina": "cn", "chinamainland": "cn",
        "taiwanprovinceofchina": "tw", "taiwanchina": "tw", "chinataiwan": "tw",
        "taiwanrepublicofchina": "tw", "republicofchinataiwan": "tw",
        "hongkongsarchina": "hk", "hongkongchina": "hk", "chinahongkong": "hk",
        "chinahongkongsar": "hk", "hongkongsarofchina": "hk",
        "hongkongspecialadministrativeregionofchina": "hk",
        "macaosarchina": "mo", "macausarchina": "mo", "macaochina": "mo",
        "macauchina": "mo", "chinamacao": "mo", "chinamacau": "mo",
        "chinamacaosar": "mo", "chinamacausar": "mo", "macaosarofchina": "mo",
        "macaospecialadministrativeregionofchina": "mo",
    },
)

# The short name of a family member that HAS a name without the family word.
# "Taiwan", "Hong Kong" and "Macao" do not contain "china", so they stay
# outside the family and meet "Taiwan (ROC)" or "Hong Kong SAR" through the
# ordinary word test. When the bot prints the long form instead ("Hong Kong
# SAR China") that resolves inside the family, and is compared to an
# outside name through the short name here. "China" itself is inside the
# family, so this path can never reach it.
_FAMILY_PLAIN: dict[str, str] = {"tw": "Taiwan", "hk": "Hong Kong", "mo": "Macao"}

_WORD_RE = re.compile(r"[a-z0-9]+")
_APOSTROPHE_RE = re.compile(r"['‘’ʼ`]")

# Letters that are not a base letter plus an accent, so NFKD leaves them
# whole and they would be dropped like the accents used to be.
_UNDECOMPOSABLE = str.maketrans({
    "ø": "o", "æ": "ae", "œ": "oe", "ł": "l", "đ": "d", "ı": "i",
})

# The one word that puts a name in each family. Checked against unlisted
# spellings so "congo brazzaville rep" is recognised as a Congo without
# being mistaken for either specific one.
_FAMILY_KEYWORDS: tuple[tuple[dict[str, str], str], ...] = tuple(
    (family, keyword)
    for family, keyword in zip(
        _FAMILIES,
        ("congo", "guinea", "sudan", "korea", "niger", "dominica",
         "samoa", "virginislands", "china"),
        strict=True,
    )
)


def _tokens(value: str) -> list[str]:
    """The words of a name, folded to plain lower-case ASCII.

    Matching works on [a-z0-9], and anything else used to be DROPPED: "Côte
    d'Ivoire" became "ctedivoire" and never met its alias "cotedivoire";
    "Türkiye", "São Tomé" and "Curaçao" broke the same way. Accents are now
    folded off instead (NFKD splits "ô" into "o" plus a combining mark, and
    the mark is discarded).

    "&" is read as "and" so "Bosnia & Herzegovina" is letter-for-letter the
    same name as "Bosnia and Herzegovina", and apostrophes are closed up so
    "People's" is the noise word "peoples" rather than "people" plus a stray
    "s".
    """
    folded = unicodedata.normalize(
        "NFKD", value.casefold().translate(_UNDECOMPOSABLE)
    )
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    folded = _APOSTROPHE_RE.sub("", folded.replace("&", " and "))
    return _WORD_RE.findall(folded)


def _significant(tokens: list[str]) -> set[str]:
    """Significant words, noise removed.

    If a name is nothing BUT noise words ("United States"), the noise is
    kept - a name has to be made of something.
    """
    significant = {word for word in tokens if word not in _NOISE}
    return significant or set(tokens)


def _family_of(flat: str) -> tuple[dict[str, str], str | None] | None:
    """The colliding-names family this spelling belongs to, if any.

    Returns the family and the country it resolves to - or None for the
    country when the family clearly applies but the spelling is unlisted,
    which must NOT be treated as a match.
    """
    for family in _FAMILIES:
        if flat in family:
            return family, family[flat]

    # An unlisted spelling still belongs to a family if it contains that
    # family's DISTINGUISHING word: "congo brazzaville rep", "guinea
    # conakry". Matched against the family key word only - testing against
    # whole entries put "unitedstates" inside the Virgin Islands family via
    # "unitedstatesvirginislands", which then refused to match "United
    # States of America".
    for family, keyword in _FAMILY_KEYWORDS:
        if keyword in flat:
            return family, None
    return None


def _alias_of(tokens: list[str]) -> int | None:
    """Which alias entry a name belongs to, if any.

    All the letters are tried before the significant words alone, so a name
    listed in full is never re-filed by what is left of it without noise.
    """
    flat = "".join(tokens)
    core = "".join(word for word in tokens if word not in _NOISE)
    for candidate in (flat, core):
        for index, alias in enumerate(_ALIASES):
            if candidate in alias:
                return index
    return None


def _is_name_plus_noise(short: list[str], long: list[str]) -> bool:
    """True when ``long`` is ``short`` followed by nothing but noise words.

    The letters are compared across word breaks, so "Hongkong" still meets
    "Hong Kong SAR". But the cut has to fall between two words and what is
    left has to be noise: a bare letter-prefix test matched "India" to
    "Indian Ocean Territory" and "United States" to "United States Minor
    Outlying Islands".
    """
    target = "".join(short)
    built = ""
    for index, word in enumerate(long):
        built += word
        if built == target:
            return all(rest in _NOISE for rest in long[index + 1:])
        if not target.startswith(built):
            return False
    return False


def _same_outside_families(left: list[str], right: list[str]) -> bool:
    """Compare two names that belong to no colliding family."""
    if "".join(left) == "".join(right):
        return True

    left_alias, right_alias = _alias_of(left), _alias_of(right)
    if left_alias is not None and right_alias is not None:
        # Both are known spellings, so the table decides - in BOTH
        # directions. Falling through to the word test here would let
        # "Ireland" match "...and Northern Ireland" on the shared word.
        return left_alias == right_alias

    left_words, right_words = _significant(left), _significant(right)
    if not left_words & right_words:
        # No shared significant word, but one name may simply BE the other
        # plus noise, or the same words run together: "Hongkong" vs "Hong
        # Kong SAR". The word test cannot see either.
        return _is_name_plus_noise(left, right) or _is_name_plus_noise(right, left)

    # Every significant word of the shorter name appears in the longer one:
    # "Central African Rep." vs "Central African Republic" -> {central,
    # african} <= {central, african}; "Myanmar" vs "Myanmar (Burma)" ->
    # {myanmar} <= {myanmar, burma}.
    shorter, longer = sorted((left_words, right_words), key=len)
    return shorter <= longer


def same_country(left: str, right: str) -> bool:
    """True when two spellings name the same country."""
    if not left or not right:
        return False

    left_tokens, right_tokens = _tokens(left), _tokens(right)
    left_flat, right_flat = "".join(left_tokens), "".join(right_tokens)
    if not left_flat or not right_flat:
        return False
    if left_flat == right_flat:
        return True

    left_family = _family_of(left_flat)
    right_family = _family_of(right_flat)
    if left_family or right_family:
        # One side names a country that collides with another. Only an
        # explicit match counts; a guess here files numbers under the wrong
        # stock, which is worse than asking again.
        if not (left_family and right_family):
            # The one exception: a family member with a short name of its
            # own ("Hong Kong SAR China" -> "Hong Kong") may still meet a
            # name outside every family through that short name.
            known, outside = (
                (left_family, right_tokens) if left_family
                else (right_family, left_tokens)
            )
            plain = _FAMILY_PLAIN.get(known[1] or "")
            return plain is not None and _same_outside_families(
                _tokens(plain), outside
            )
        if left_family[0] is not right_family[0]:
            return False
        return left_family[1] is not None and left_family[1] == right_family[1]

    return _same_outside_families(left_tokens, right_tokens)


def find(stock: dict[str, int], country: str) -> int | None:
    """The stock figure for ``country``, however the bot spelled it."""
    wanted = " ".join(country.strip().casefold().split())
    lowered = {" ".join(k.strip().casefold().split()): v for k, v in stock.items()}
    if wanted in lowered:
        return lowered[wanted]
    for name, value in lowered.items():
        if same_country(name, wanted):
            return value
    return None
