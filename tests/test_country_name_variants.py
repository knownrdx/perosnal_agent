"""The spellings of a country the target bot might use, and the pairs that must stay apart.

``country_names.same_country`` decides whether the name in our phone-prefix
table and the name in the bot's /st reply are one country. Both kinds of
mistake are expensive:

- A MISSED match reads as zero stock. The automation then cleans up and
  re-adds the whole file every cycle, forever. "Congo (DRC)" vs "DR Congo"
  did exactly this in production.
- A WRONG match puts one country's numbers under another country's stock,
  which is worse.

These tests cover what the first fix did not: accented letters, "&" written
for "and", official long names that share no word with the short name, and
an exhaustive check that no two entries in our own table merge.
"""

import pytest

from app.automation import country_names, phone_countries
from app.automation.country_names import find, same_country


@pytest.fixture(autouse=True)
def environment():
    """Replaces conftest's per-test SQLite workspace.

    These are pure string comparisons. Building a database for each of the
    ~370 cases added close to a minute to the suite for nothing.
    """
    yield None


def _assert_same(left: str, right: str) -> None:
    # Both directions, plus the lookup production actually performs.
    assert same_country(left, right), f"{left!r} should match {right!r}"
    assert same_country(right, left), f"{right!r} should match {left!r}"
    assert find({right: 7}, left) == 7


def _assert_different(left: str, right: str) -> None:
    assert not same_country(left, right), f"{left!r} merged with {right!r}"
    assert not same_country(right, left), f"{right!r} merged with {left!r}"
    assert find({right: 7}, left) is None


# --------------------------------------------------------------------- #
# accented letters
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("left,right", [
    # Matching used to keep only [a-z0-9], which DROPPED accented letters:
    # "Côte" became "cte" and "Türkiye" became "trkiye".
    ("Côte d'Ivoire", "Cote d'Ivoire"),
    ("Côte d’Ivoire", "Ivory Coast"),           # curly apostrophe too
    ("Türkiye", "Turkiye"),
    ("São Tomé and Príncipe", "Sao Tome And Principe"),
    ("Curaçao", "Curacao"),
    ("Réunion", "Reunion"),
    ("Åland Islands", "Aland Islands"),
    ("Åland", "Åland Islands"),
    ("Saint Barthélemy", "Saint Barthelemy"),
])
def test_accented_letters_are_folded_not_dropped(left, right):
    _assert_same(left, right)


# --------------------------------------------------------------------- #
# "&" for "and"
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("left,right", [
    ("Bosnia & Herzegovina", "Bosnia and Herzegovina"),
    ("Trinidad & Tobago", "Trinidad and Tobago"),
    ("Antigua & Barbuda", "Antigua and Barbuda"),
    ("St. Kitts & Nevis", "Saint Kitts And Nevis"),
    ("São Tomé & Príncipe", "Sao Tome And Principe"),
    ("Turks & Caicos Islands", "Turks And Caicos Islands"),
    ("St Vincent & the Grenadines", "Saint Vincent And The Grenadines"),
    ("St. Pierre & Miquelon", "Saint Pierre And Miquelon"),
    ("Wallis & Futuna", "Wallis And Futuna"),
    ("Svalbard & Jan Mayen", "Svalbard And Jan Mayen"),
])
def test_ampersand_means_and(left, right):
    _assert_same(left, right)


# --------------------------------------------------------------------- #
# official, old and local names
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("left,right", [
    ("Turkey", "Türkiye"),
    ("Turkey", "Turkiye"),
    ("Turkey", "Republic of Türkiye"),
    ("Laos", "Lao PDR"),
    ("Laos", "Lao"),
    ("Laos", "Lao People's Democratic Republic"),
    ("Kyrgyzstan", "Kyrgyz Republic"),
    ("Macao", "Macau"),
    ("Macao", "Macau SAR"),
    ("Macao", "Macao SAR China"),
    ("Macao", "China, Macao SAR"),
    ("Vatican", "Holy See"),
    ("Vatican", "Vatican City"),
    ("Vatican", "Holy See (Vatican City State)"),
    ("Palestine", "Palestinian Territory"),
    ("Palestine", "Palestinian Territories"),
    ("Palestine", "State of Palestine"),
    ("Palestine", "Occupied Palestinian Territory"),
    ("Swaziland", "Eswatini"),
    ("Swaziland", "Kingdom of Eswatini"),
    ("North Macedonia", "Macedonia"),
    ("North Macedonia", "Macedonia, the former Yugoslav Republic of"),
    ("Czech Republic", "Czechia"),
    ("Ivory Coast", "Côte d'Ivoire"),
    ("Russia", "Russian Federation"),
    ("Syria", "Syrian Arab Republic"),
    ("Vietnam", "Viet Nam"),
    ("Brunei", "Brunei Darussalam"),
    ("Cape Verde", "Cabo Verde"),
    ("East Timor", "Timor-Leste"),
    ("Micronesia", "Federated States of Micronesia"),
    ("Micronesia", "Micronesia, Federated States of"),
    ("UAE", "United Arab Emirates"),
    ("United States", "USA"),
    ("United States", "U.S.A."),
    ("United States", "United States of America"),
    ("United Kingdom", "UK"),
    ("United Kingdom", "Great Britain"),
    ("United Kingdom", "United Kingdom of Great Britain and Northern Ireland"),
    ("Hong Kong", "Hong Kong SAR"),
    ("Hong Kong", "Hongkong"),
    ("Hong Kong", "Hong Kong SAR China"),
    ("Hong Kong", "China, Hong Kong SAR"),
    ("Taiwan", "Taiwan Province of China"),
    ("Taiwan", "Taiwan, Province of China"),
    ("Taiwan", "Taiwan (ROC)"),
    ("China", "People's Republic of China"),
    ("Moldova", "Republic of Moldova"),
    ("Moldova", "Moldova, Republic of"),
    ("Tanzania", "United Republic of Tanzania"),
    ("Tanzania", "Tanzania, United Republic of"),
    ("Bolivia", "Bolivia (Plurinational State of)"),
    ("Iran", "Islamic Republic of Iran"),
    ("Iran", "Iran, Islamic Republic of"),
    ("South Korea", "Korea Republic"),
    ("South Korea", "Korea, Republic of"),
    ("South Korea", "Republic of Korea"),
    ("North Korea", "DPRK"),
    ("North Korea", "Democratic People's Republic of Korea"),
    ("North Korea", "Korea, Democratic People's Republic of"),
    ("U.S. Virgin Islands", "Virgin Islands, U.S."),
    ("British Virgin Islands", "Virgin Islands, British"),
    ("Congo (DRC)", "Congo, The Democratic Republic of the"),
    ("Congo (DRC)", "The Democratic Republic Of Congo"),
    ("Congo (Brazzaville)", "Congo, Republic of the"),
    ("Netherlands", "The Netherlands"),
    ("Netherlands", "Holland"),
    ("Bonaire, Sint Eustatius and Saba", "Caribbean Netherlands"),
])
def test_official_and_former_names_match(left, right):
    _assert_same(left, right)


# --------------------------------------------------------------------- #
# different countries with similar names
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("left,right", [
    ("Niger", "Nigeria"),
    ("Guinea", "Guinea-Bissau"),
    ("Guinea", "Equatorial Guinea"),
    ("Guinea", "Papua New Guinea"),
    ("Guinea-Bissau", "Equatorial Guinea"),
    ("Guinea-Bissau", "Papua New Guinea"),
    ("Equatorial Guinea", "Papua New Guinea"),
    ("Sudan", "South Sudan"),
    ("Dominica", "Dominican Republic"),
    ("Congo", "DR Congo"),
    ("Congo (Brazzaville)", "Congo (DRC)"),
    ("Samoa", "American Samoa"),
    ("China", "Taiwan"),
    ("China", "Taiwan Province of China"),
    ("China", "Chinese Taipei"),
    ("China", "Hong Kong"),
    ("China", "Hong Kong SAR China"),
    ("China", "Macao SAR China"),
    ("Hong Kong", "Macao"),
    ("Hong Kong SAR China", "Macao SAR China"),
    ("India", "British Indian Ocean Territory"),
    ("India", "Indian Ocean Territory"),
    ("United States", "U.S. Virgin Islands"),
    ("United States", "United States Virgin Islands"),
    ("USA", "US Virgin Islands"),
    ("United States", "United States Minor Outlying Islands"),
    ("Saint Martin", "Sint Maarten"),
    ("Saint Martin", "Sint Maarten (Dutch part)"),
    ("Australia", "Austria"),
    ("Iran", "Iraq"),
    ("Mali", "Malawi"),
    ("Mali", "Somalia"),
    ("Oman", "Romania"),
    ("Slovakia", "Slovenia"),
    ("Gambia", "Zambia"),
    ("South Korea", "North Korea"),
    ("Korea Republic", "DPRK"),
    ("Ireland", "Northern Ireland"),
    ("Ireland", "United Kingdom of Great Britain and Northern Ireland"),
    ("Netherlands", "Caribbean Netherlands"),
    ("Holland", "Caribbean Netherlands"),
    ("Netherlands", "Netherlands Antilles"),
    ("Georgia", "South Georgia"),
    ("Georgia", "South Georgia and the South Sandwich Islands"),
])
def test_neighbouring_names_stay_separate(left, right):
    _assert_different(left, right)


# --------------------------------------------------------------------- #
# a whole /st reply, as the lookup sees it
# --------------------------------------------------------------------- #

# None of the bot's spellings below is ours letter for letter, so find()
# cannot take its exact-match shortcut and every lookup goes through
# same_country, in the bot's order.

def test_each_china_name_finds_its_own_stock():
    """Each of these contains "China". A plain word-subset test gave China
    the stock of whichever one the bot listed first."""
    stock = {
        "Taiwan, Province of China": 1,
        "Hong Kong SAR China": 2,
        "Macao SAR China": 3,
        "People's Republic of China": 4,
    }
    assert find(stock, "Taiwan") == 1
    assert find(stock, "Hong Kong") == 2
    assert find(stock, "Macao") == 3
    assert find(stock, "China") == 4


def test_each_guinea_finds_its_own_stock():
    stock = {
        "Papua-New-Guinea": 1,
        "Equatorial-Guinea": 2,
        "Guinea Bissau": 3,
        "Republic of Guinea": 4,
    }
    assert find(stock, "Papua New Guinea") == 1
    assert find(stock, "Equatorial Guinea") == 2
    assert find(stock, "Guinea-Bissau") == 3
    assert find(stock, "Guinea") == 4


# --------------------------------------------------------------------- #
# our own table, exhaustively
# --------------------------------------------------------------------- #

def test_no_two_countries_in_our_own_table_match():
    """Every pair of different ISO codes in the phone-prefix table must stay
    apart. This covers all ~30,000 pairs, so a new alias or noise word
    cannot quietly merge two of the countries we actually upload."""
    entries = sorted(phone_countries._ISO_NAMES.items())
    merged = [
        (code_a, name_a, code_b, name_b)
        for index, (code_a, name_a) in enumerate(entries)
        for code_b, name_b in entries[index + 1:]
        if same_country(name_a, name_b) or same_country(name_b, name_a)
    ]
    assert not merged, f"{len(merged)} merged pairs: {merged[:10]}"


def test_libphonenumber_s_names_resolve_to_the_same_region():
    """A second, independent naming of the same regions.

    libphonenumber's geocoder has its own English name for each region
    ("The Democratic Republic Of Congo", "Côte d'Ivoire", "Timor-Leste"). A
    bot built on it would print those. Each one must match our name for that
    region and no other. The geocoder leaves regions inside a shared
    calling code unnamed, so those are skipped.
    """
    import phonenumbers
    from phonenumbers import PhoneNumberType, geocoder

    ours = phone_countries._ISO_NAMES
    missed, merged = [], []
    for region in sorted(ours):
        number = (
            phonenumbers.example_number_for_type(region, PhoneNumberType.MOBILE)
            or phonenumbers.example_number(region)
        )
        theirs = geocoder.country_name_for_number(number, "en") if number else ""
        if not theirs:
            continue
        if not same_country(ours[region], theirs):
            missed.append((region, ours[region], theirs))
        merged.extend(
            (code, name, region, theirs)
            for code, name in ours.items()
            if code != region and same_country(name, theirs)
        )
    assert not missed, f"missed: {missed}"
    assert not merged, f"merged: {merged}"


@pytest.mark.parametrize("name", sorted(set(phone_countries._ISO_NAMES.values())))
def test_every_name_in_our_table_finds_itself(name):
    assert same_country(name, name)
    assert find({name: 5}, name) == 5


def test_no_spelling_is_listed_for_two_places():
    """Two known spellings match only when they share an alias entry, so a
    spelling listed twice would be filed under whichever entry came first
    and silently stop matching the other."""
    seen: dict[str, int] = {}
    for index, alias in enumerate(country_names._ALIASES):
        for spelling in alias:
            assert spelling not in seen, f"{spelling!r} is in two alias entries"
            seen[spelling] = index
            # A family spelling is decided by the family before aliases are
            # consulted, so listing it here as well would be dead weight.
            assert country_names._family_of(spelling) is None, spelling
