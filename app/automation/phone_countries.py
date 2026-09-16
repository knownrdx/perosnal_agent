"""E.164 dialing-prefix -> country resolution.

The number files are bare phone numbers, one per line, with no country column,
so the only way to know what is in a file is to read the numbers themselves.
Matching is longest-prefix-first because dialing codes are not fixed width and
overlap badly: +1 is North America, but +1242 is the Bahamas and +1809 the
Dominican Republic, so a naive "first two digits" split silently merges
countries that the target bot treats as completely separate stock.
"""

from __future__ import annotations

import re
from collections import Counter

# Country name -> dialing prefixes (without "+").
#
# Deliberately not the full ITU list: this covers the regions the owner
# actually uploads, plus the NANP carve-outs that would otherwise all collapse
# into "United States". Unknown prefixes are reported as-is rather than
# guessed at, so a missing entry shows up as an obvious "+998" bucket instead
# of quietly landing in the wrong country's tag.
_PREFIXES: dict[str, tuple[str, ...]] = {
    "Bangladesh": ("880",),
    "India": ("91",),
    "Pakistan": ("92",),
    "Indonesia": ("62",),
    "Nigeria": ("234",),
    "Kenya": ("254",),
    "Central African Republic": ("236",),
    "Zimbabwe": ("263",),
    "South Africa": ("27",),
    "Egypt": ("20",),
    "Morocco": ("212",),
    "Algeria": ("213",),
    "Tunisia": ("216",),
    "Ghana": ("233",),
    "Ivory Coast": ("225",),
    "Cameroon": ("237",),
    "Senegal": ("221",),
    "Uganda": ("256",),
    "Tanzania": ("255",),
    "Ethiopia": ("251",),
    "Zambia": ("260",),
    "Mozambique": ("258",),
    "Angola": ("244",),
    "Congo (DRC)": ("243",),
    "Congo (Brazzaville)": ("242",),
    "Chad": ("235",),
    "Niger": ("227",),
    "Mali": ("223",),
    "Burkina Faso": ("226",),
    "Benin": ("229",),
    "Togo": ("228",),
    "Guinea": ("224",),
    "Rwanda": ("250",),
    "Burundi": ("257",),
    "Somalia": ("252",),
    "Sudan": ("249",),
    "Libya": ("218",),
    "Madagascar": ("261",),
    "Malawi": ("265",),
    "Botswana": ("267",),
    "Namibia": ("264",),
    "Liberia": ("231",),
    "Sierra Leone": ("232",),
    "Gambia": ("220",),
    "Mauritania": ("222",),
    "Gabon": ("241",),
    "Philippines": ("63",),
    "Vietnam": ("84",),
    "Thailand": ("66",),
    "Malaysia": ("60",),
    "Myanmar": ("95",),
    "Cambodia": ("855",),
    "Laos": ("856",),
    "Nepal": ("977",),
    "Sri Lanka": ("94",),
    "Afghanistan": ("93",),
    "Uzbekistan": ("998",),
    "Kazakhstan": ("7",),
    "Turkey": ("90",),
    "Iraq": ("964",),
    "Iran": ("98",),
    "Saudi Arabia": ("966",),
    "UAE": ("971",),
    "Qatar": ("974",),
    "Kuwait": ("965",),
    "Oman": ("968",),
    "Bahrain": ("973",),
    "Jordan": ("962",),
    "Lebanon": ("961",),
    "Syria": ("963",),
    "Yemen": ("967",),
    "Israel": ("972",),
    "China": ("86",),
    "Hong Kong": ("852",),
    "United Kingdom": ("44",),
    "Germany": ("49",),
    "France": ("33",),
    "Italy": ("39",),
    "Spain": ("34",),
    "Portugal": ("351",),
    "Netherlands": ("31",),
    "Belgium": ("32",),
    "Poland": ("48",),
    "Romania": ("40",),
    "Ukraine": ("380",),
    "Russia": ("79",),
    "Brazil": ("55",),
    "Mexico": ("52",),
    "Argentina": ("54",),
    "Colombia": ("57",),
    "Peru": ("51",),
    "Chile": ("56",),
    "Venezuela": ("58",),
    "Australia": ("61",),
    # NANP: +1 is shared, so the specific area codes must win over plain "1".
    "Dominican Republic": ("1809", "1829", "1849"),
    "Jamaica": ("1876", "1658"),
    "Trinidad and Tobago": ("1868",),
    "Bahamas": ("1242",),
    "Barbados": ("1246",),
    "Haiti": ("509",),
    "Puerto Rico": ("1787", "1939"),
    "Canada": ("1204", "1236", "1249", "1250", "1289", "1306", "1343", "1365",
               "1387", "1403", "1416", "1418", "1431", "1437", "1438", "1450",
               "1506", "1514", "1519", "1548", "1579", "1581", "1587", "1604",
               "1613", "1639", "1647", "1672", "1705", "1709", "1742", "1778",
               "1780", "1782", "1807", "1819", "1825", "1867", "1873", "1902",
               "1905"),
    "United States": ("1",),
}

# Flattened prefix -> country, longest first so lookup is a simple scan.
_LOOKUP: list[tuple[str, str]] = sorted(
    (
        (prefix, country)
        for country, prefixes in _PREFIXES.items()
        # A bare string (not a tuple) would iterate character by character and
        # register nonsense single-digit prefixes, so normalise defensively.
        for prefix in ((prefixes,) if isinstance(prefixes, str) else prefixes)
    ),
    key=lambda item: len(item[0]),
    reverse=True,
)

UNKNOWN = "Unknown"

_DIGITS_RE = re.compile(r"\d+")


def normalise_number(raw: str) -> str:
    """Digits only, e.g. '+236 77-000' -> '23677000'. Empty if there are none."""
    digits = "".join(_DIGITS_RE.findall(raw))
    return digits


def country_of(raw_number: str) -> str:
    """Country for one number, or a '+<prefix>' placeholder when unrecognised."""
    digits = normalise_number(raw_number)
    if not digits:
        return UNKNOWN
    for prefix, country in _LOOKUP:
        if digits.startswith(prefix):
            return country
    # Unrecognised: surface the leading digits so a missing table entry is
    # visible and fixable, instead of being silently bucketed as Unknown.
    return f"+{digits[:3]}"


def split_by_country(lines: list[str]) -> dict[str, list[str]]:
    """Group raw lines by country, preserving each line exactly as given.

    Blank lines are dropped; everything else is kept verbatim so the file
    handed to the target bot is byte-identical to what the owner uploaded,
    minus the numbers that belong to another country.
    """
    grouped: dict[str, list[str]] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        grouped.setdefault(country_of(stripped), []).append(stripped)
    return grouped


def count_by_country(lines: list[str]) -> dict[str, int]:
    """How many numbers of each country, most numerous first."""
    counter: Counter[str] = Counter()
    for line in lines:
        stripped = line.strip()
        if stripped:
            counter[country_of(stripped)] += 1
    return dict(counter.most_common())
