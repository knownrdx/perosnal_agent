"""Deciding that two country names mean the same country.

The name we hold came from the phone-prefix table; the one in the bot's
reply came from the bot. They describe the same place in different words:

    ours              theirs
    Congo (DRC)       DR Congo
    Cote d'Ivoire     Ivory Coast
    Myanmar           Myanmar (Burma)

The old rule compared letters-only and asked whether either was a PREFIX of
the other. "congodrc" and "drcongo" are not, so the lookup missed, an empty
result read as ZERO STOCK, and the automation cleaned up and re-added the
whole file - every cycle, forever. The owner watched 16,499 duplicates get
skipped every five minutes.

Prefix matching is also too loose the other way: "Niger" is a prefix of
"Nigeria", and merging two countries' stock is worse than missing a match.

The approach here is therefore in two parts:

1. A table of the names that genuinely collide - the two Congos, the four
   Guineas, the Sudans, the Koreas. Inside those families a name is
   identified EXPLICITLY, by listing every spelling of each country; two
   names match only if they resolve to the same entry, and anything in a
   family that cannot be resolved matches nothing.
2. For everything else, compare the significant words. This handles the
   long tail - "Central African Rep." vs "Central African Republic",
   "Viet Nam" vs "Vietnam" - without a table of 245 countries.
"""

from __future__ import annotations

import re

# Words that carry no identity - they appear in dozens of official names and
# matching on them alone would merge unrelated countries.
_NOISE = frozenset({
    "the", "of", "and", "republic", "rep", "democratic", "dem", "people",
    "peoples", "state", "states", "kingdom", "united", "federal",
    "federation", "islamic", "arab", "new", "isle", "island", "islands",
    "territory", "territories", "province", "saint", "st",
})

# Names the same place is genuinely known by, where the two forms share no
# significant word. Anything else the word comparison already handles.
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
    frozenset({"unitedkingdom", "uk", "britain", "greatbritain"}),
)

# Countries whose names contain each other. Word comparison cannot separate
# these, so every spelling is listed against its country and a name inside a
# family matches ONLY a name mapped to the same country. A spelling in one
# of these families that is not listed matches nothing rather than guessing,
# because the cost of guessing here is numbers filed under the wrong stock.
_FAMILIES: tuple[dict[str, str], ...] = (
    {   # Congo
        "congodrc": "cd", "drcongo": "cd", "drccongo": "cd", "drc": "cd",
        "democraticrepublicofthecongo": "cd", "democraticrepublicofcongo": "cd",
        "congokinshasa": "cd", "congodemrep": "cd", "congodemocraticrepublic": "cd",
        "zaire": "cd",
        "congo": "cg", "republicofthecongo": "cg", "republicofcongo": "cg",
        "congobrazzaville": "cg", "congorep": "cg",
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
        "korea": "kr",
        "northkorea": "kp", "koreanorth": "kp",
        "democraticpeoplesrepublicofkorea": "kp",
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
        "unitedstatesvirginislands": "vi",
        "britishvirginislands": "vg",
    },
)

_WORD_RE = re.compile(r"[a-z0-9]+")

# The one word that puts a name in each family. Checked against unlisted
# spellings so "congo brazzaville rep" is recognised as a Congo without
# being mistaken for either specific one.
_FAMILY_KEYWORDS: tuple[tuple[dict[str, str], str], ...] = tuple(
    (family, keyword)
    for family, keyword in zip(
        _FAMILIES,
        ("congo", "guinea", "sudan", "korea", "niger", "dominica",
         "samoa", "virginislands"),
    )
)


def _squash(value: str) -> str:
    return "".join(_WORD_RE.findall(value.casefold()))


def _words(value: str) -> set[str]:
    """Significant words, lower-cased, noise removed.

    If a name is nothing BUT noise words ("United States"), the noise is
    kept - a name has to be made of something.
    """
    found = _WORD_RE.findall(value.casefold())
    significant = {word for word in found if word not in _NOISE}
    return significant or set(found)


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


def same_country(left: str, right: str) -> bool:
    """True when two spellings name the same country."""
    if not left or not right:
        return False

    left_flat, right_flat = _squash(left), _squash(right)
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
            return False
        if left_family[0] is not right_family[0]:
            return False
        return left_family[1] is not None and left_family[1] == right_family[1]

    for alias in _ALIASES:
        if left_flat in alias and right_flat in alias:
            return True

    left_words, right_words = _words(left), _words(right)
    if not left_words & right_words:
        # No shared significant word, but one name may simply BE the other
        # plus noise: "United States" vs "United States of America" reduce
        # to nothing significant on either side, so the word test cannot
        # see them. Compare the letters directly in that case.
        if left_flat.startswith(right_flat) or right_flat.startswith(left_flat):
            return True
        return False

    # Every significant word of the shorter name appears in the longer one:
    # "Central African Rep." vs "Central African Republic" -> {central,
    # african} <= {central, african}; "Myanmar" vs "Myanmar (Burma)" ->
    # {myanmar} <= {myanmar, burma}.
    shorter, longer = sorted((left_words, right_words), key=len)
    return shorter <= longer


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
