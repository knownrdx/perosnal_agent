"""E.164 dialing-prefix -> country resolution.

The number files are bare phone numbers, one per line, with no country column,
so the only way to know what is in a file is to read the numbers themselves.

Coverage is the whole ITU list, not a hand-written shortlist. The previous
table held ~100 entries picked by hand, so anything outside it landed in a
"+380"-style bucket the target bot has never heard of - which is what the
owner saw as "country detect kore na" for perfectly ordinary numbers.
`phonenumbers` (Google's libphonenumber port, pure Python, no compiler) knows
every region and, crucially, resolves the ranges a flat prefix table CANNOT:
+1 splits across 25 NANP countries by area code, +7 across Russia and
Kazakhstan, +44 across the UK and its crown dependencies.

The library is optional. Without it everything still works off the generated
calling-code table below - less precise inside shared codes, never worse than
the old hand-written behaviour.
"""

from __future__ import annotations

import re
from collections import Counter

try:  # optional: precision comes from the library, correctness does not
    import phonenumbers as _pn

    _HAS_PN = True
except Exception:  # noqa: BLE001 - a missing optional dep is not an error
    _pn = None
    _HAS_PN = False

UNKNOWN = "Unknown"

_DIGITS_RE = re.compile(r"\d+")

# ISO-3166 alpha-2 -> the name shown to the owner and matched against the
# target bot's own stock lines. Generated from libphonenumber's region list,
# with the handful of long CLDR forms replaced by the short names the bot
# actually prints ("Congo (DRC)", not "Congo - Kinshasa").
_ISO_NAMES: dict[str, str] = {
    'AC': 'Ascension Island', 'AD': 'Andorra', 'AE': 'UAE',
    'AF': 'Afghanistan', 'AG': 'Antigua and Barbuda', 'AI': 'Anguilla',
    'AL': 'Albania', 'AM': 'Armenia', 'AO': 'Angola',
    'AR': 'Argentina', 'AS': 'American Samoa', 'AT': 'Austria',
    'AU': 'Australia', 'AW': 'Aruba', 'AX': 'Åland Islands',
    'AZ': 'Azerbaijan', 'BA': 'Bosnia and Herzegovina', 'BB': 'Barbados',
    'BD': 'Bangladesh', 'BE': 'Belgium', 'BF': 'Burkina Faso',
    'BG': 'Bulgaria', 'BH': 'Bahrain', 'BI': 'Burundi',
    'BJ': 'Benin', 'BL': 'Saint Barthelemy', 'BM': 'Bermuda',
    'BN': 'Brunei', 'BO': 'Bolivia', 'BQ': 'Bonaire, Sint Eustatius and Saba',
    'BR': 'Brazil', 'BS': 'Bahamas', 'BT': 'Bhutan',
    'BW': 'Botswana', 'BY': 'Belarus', 'BZ': 'Belize',
    'CA': 'Canada', 'CC': 'Cocos Islands', 'CD': 'Congo (DRC)',
    'CF': 'Central African Republic', 'CG': 'Congo (Brazzaville)', 'CH': 'Switzerland',
    'CI': 'Ivory Coast', 'CK': 'Cook Islands', 'CL': 'Chile',
    'CM': 'Cameroon', 'CN': 'China', 'CO': 'Colombia',
    'CR': 'Costa Rica', 'CU': 'Cuba', 'CV': 'Cape Verde',
    'CW': 'Curaçao', 'CX': 'Christmas Island', 'CY': 'Cyprus',
    'CZ': 'Czech Republic', 'DE': 'Germany', 'DJ': 'Djibouti',
    'DK': 'Denmark', 'DM': 'Dominica', 'DO': 'Dominican Republic',
    'DZ': 'Algeria', 'EC': 'Ecuador', 'EE': 'Estonia',
    'EG': 'Egypt', 'EH': 'Western Sahara', 'ER': 'Eritrea',
    'ES': 'Spain', 'ET': 'Ethiopia', 'FI': 'Finland',
    'FJ': 'Fiji', 'FK': 'Falkland Islands', 'FM': 'Micronesia',
    'FO': 'Faroe Islands', 'FR': 'France', 'GA': 'Gabon',
    'GB': 'United Kingdom', 'GD': 'Grenada', 'GE': 'Georgia',
    'GF': 'French Guiana', 'GG': 'Guernsey', 'GH': 'Ghana',
    'GI': 'Gibraltar', 'GL': 'Greenland', 'GM': 'Gambia',
    'GN': 'Guinea', 'GP': 'Guadeloupe', 'GQ': 'Equatorial Guinea',
    'GR': 'Greece', 'GT': 'Guatemala', 'GU': 'Guam',
    'GW': 'Guinea-Bissau', 'GY': 'Guyana', 'HK': 'Hong Kong',
    'HN': 'Honduras', 'HR': 'Croatia', 'HT': 'Haiti',
    'HU': 'Hungary', 'ID': 'Indonesia', 'IE': 'Ireland',
    'IL': 'Israel', 'IM': 'Isle Of Man', 'IN': 'India',
    'IO': 'British Indian Ocean Territory', 'IQ': 'Iraq', 'IR': 'Iran',
    'IS': 'Iceland', 'IT': 'Italy', 'JE': 'Jersey',
    'JM': 'Jamaica', 'JO': 'Jordan', 'JP': 'Japan',
    'KE': 'Kenya', 'KG': 'Kyrgyzstan', 'KH': 'Cambodia',
    'KI': 'Kiribati', 'KM': 'Comoros', 'KN': 'Saint Kitts And Nevis',
    'KP': 'North Korea', 'KR': 'South Korea', 'KW': 'Kuwait',
    'KY': 'Cayman Islands', 'KZ': 'Kazakhstan', 'LA': 'Laos',
    'LB': 'Lebanon', 'LC': 'Saint Lucia', 'LI': 'Liechtenstein',
    'LK': 'Sri Lanka', 'LR': 'Liberia', 'LS': 'Lesotho',
    'LT': 'Lithuania', 'LU': 'Luxembourg', 'LV': 'Latvia',
    'LY': 'Libya', 'MA': 'Morocco', 'MC': 'Monaco',
    'MD': 'Moldova', 'ME': 'Montenegro', 'MF': 'Saint Martin',
    'MG': 'Madagascar', 'MH': 'Marshall Islands', 'MK': 'North Macedonia',
    'ML': 'Mali', 'MM': 'Myanmar', 'MN': 'Mongolia',
    'MO': 'Macao', 'MP': 'Northern Mariana Islands', 'MQ': 'Martinique',
    'MR': 'Mauritania', 'MS': 'Montserrat', 'MT': 'Malta',
    'MU': 'Mauritius', 'MV': 'Maldives', 'MW': 'Malawi',
    'MX': 'Mexico', 'MY': 'Malaysia', 'MZ': 'Mozambique',
    'NA': 'Namibia', 'NC': 'New Caledonia', 'NE': 'Niger',
    'NF': 'Norfolk Island', 'NG': 'Nigeria', 'NI': 'Nicaragua',
    'NL': 'Netherlands', 'NO': 'Norway', 'NP': 'Nepal',
    'NR': 'Nauru', 'NU': 'Niue', 'NZ': 'New Zealand',
    'OM': 'Oman', 'PA': 'Panama', 'PE': 'Peru',
    'PF': 'French Polynesia', 'PG': 'Papua New Guinea', 'PH': 'Philippines',
    'PK': 'Pakistan', 'PL': 'Poland', 'PM': 'Saint Pierre And Miquelon',
    'PR': 'Puerto Rico', 'PS': 'Palestine', 'PT': 'Portugal',
    'PW': 'Palau', 'PY': 'Paraguay', 'QA': 'Qatar',
    'RE': 'Reunion', 'RO': 'Romania', 'RS': 'Serbia',
    'RU': 'Russia', 'RW': 'Rwanda', 'SA': 'Saudi Arabia',
    'SB': 'Solomon Islands', 'SC': 'Seychelles', 'SD': 'Sudan',
    'SE': 'Sweden', 'SG': 'Singapore', 'SH': 'Saint Helena',
    'SI': 'Slovenia', 'SJ': 'Svalbard And Jan Mayen', 'SK': 'Slovakia',
    'SL': 'Sierra Leone', 'SM': 'San Marino', 'SN': 'Senegal',
    'SO': 'Somalia', 'SR': 'Suriname', 'SS': 'South Sudan',
    'ST': 'Sao Tome And Principe', 'SV': 'El Salvador', 'SX': 'Sint Maarten (Dutch part)',
    'SY': 'Syria', 'SZ': 'Swaziland', 'TA': 'Tristan da Cunha',
    'TC': 'Turks And Caicos Islands', 'TD': 'Chad', 'TG': 'Togo',
    'TH': 'Thailand', 'TJ': 'Tajikistan', 'TK': 'Tokelau',
    'TL': 'East Timor', 'TM': 'Turkmenistan', 'TN': 'Tunisia',
    'TO': 'Tonga', 'TR': 'Turkey', 'TT': 'Trinidad and Tobago',
    'TV': 'Tuvalu', 'TW': 'Taiwan', 'TZ': 'Tanzania',
    'UA': 'Ukraine', 'UG': 'Uganda', 'US': 'United States',
    'UY': 'Uruguay', 'UZ': 'Uzbekistan', 'VA': 'Vatican',
    'VC': 'Saint Vincent And The Grenadines', 'VE': 'Venezuela', 'VG': 'British Virgin Islands',
    'VI': 'U.S. Virgin Islands', 'VN': 'Vietnam', 'VU': 'Vanuatu',
    'WF': 'Wallis And Futuna', 'WS': 'Samoa', 'XK': 'Kosovo',
    'YE': 'Yemen', 'YT': 'Mayotte', 'ZA': 'South Africa',
    'ZM': 'Zambia', 'ZW': 'Zimbabwe',
}

# Calling code (and, for shared codes, calling code + leading digits) -> ISO
# region. Used when the library is absent, and for numbers it cannot place.
_CC_TO_ISO: dict[str, str] = {
    '1': 'US', '1242': 'BS', '1246': 'BB',
    '1264': 'AI', '1268': 'AG', '1284': 'VG',
    '1340': 'VI', '1345': 'KY', '1441': 'BM',
    '1473': 'GD', '1649': 'TC', '1658': 'JM',
    '1664': 'MS', '1670': 'MP', '1671': 'GU',
    '1684': 'AS', '1721': 'SX', '1758': 'LC',
    '1767': 'DM', '1784': 'VC', '1787': 'PR',
    '18001': 'DO', '1868': 'TT', '1869': 'KN',
    '1876': 'JM', '1939': 'PR', '20': 'EG',
    '211': 'SS', '212': 'MA', '213': 'DZ',
    '216': 'TN', '218': 'LY', '220': 'GM',
    '221': 'SN', '222': 'MR', '223': 'ML',
    '224': 'GN', '225': 'CI', '226': 'BF',
    '227': 'NE', '228': 'TG', '229': 'BJ',
    '230': 'MU', '231': 'LR', '232': 'SL',
    '233': 'GH', '234': 'NG', '235': 'TD',
    '236': 'CF', '237': 'CM', '238': 'CV',
    '239': 'ST', '240': 'GQ', '241': 'GA',
    '242': 'CG', '243': 'CD', '244': 'AO',
    '245': 'GW', '246': 'IO', '247': 'AC',
    '248': 'SC', '249': 'SD', '250': 'RW',
    '251': 'ET', '252': 'SO', '253': 'DJ',
    '254': 'KE', '255': 'TZ', '256': 'UG',
    '257': 'BI', '258': 'MZ', '260': 'ZM',
    '261': 'MG', '262': 'RE', '263': 'ZW',
    '264': 'NA', '265': 'MW', '266': 'LS',
    '267': 'BW', '268': 'SZ', '269': 'KM',
    '27': 'ZA', '290': 'SH', '2908': 'TA',
    '291': 'ER', '297': 'AW', '298': 'FO',
    '299': 'GL', '30': 'GR', '31': 'NL',
    '32': 'BE', '33': 'FR', '34': 'ES',
    '350': 'GI', '351': 'PT', '352': 'LU',
    '353': 'IE', '354': 'IS', '355': 'AL',
    '356': 'MT', '357': 'CY', '358': 'FI',
    '35818': 'AX', '359': 'BG', '36': 'HU',
    '370': 'LT', '371': 'LV', '372': 'EE',
    '373': 'MD', '374': 'AM', '375': 'BY',
    '376': 'AD', '377': 'MC', '378': 'SM',
    '380': 'UA', '381': 'RS', '382': 'ME',
    '383': 'XK', '385': 'HR', '386': 'SI',
    '387': 'BA', '389': 'MK', '39': 'IT',
    '3906698': 'VA', '40': 'RO', '41': 'CH',
    '420': 'CZ', '421': 'SK', '423': 'LI',
    '43': 'AT', '44': 'GB', '4474576': 'IM',
    '45': 'DK', '46': 'SE', '47': 'NO',
    '4779': 'SJ', '48': 'PL', '49': 'DE',
    '500': 'FK', '501': 'BZ', '502': 'GT',
    '503': 'SV', '504': 'HN', '505': 'NI',
    '506': 'CR', '507': 'PA', '508': 'PM',
    '509': 'HT', '51': 'PE', '52': 'MX',
    '53': 'CU', '54': 'AR', '55': 'BR',
    '56': 'CL', '57': 'CO', '58': 'VE',
    '590': 'GP', '591': 'BO', '592': 'GY',
    '593': 'EC', '594': 'GF', '595': 'PY',
    '596': 'MQ', '597': 'SR', '598': 'UY',
    '599': 'CW', '60': 'MY', '61': 'AU',
    '62': 'ID', '63': 'PH', '64': 'NZ',
    '65': 'SG', '66': 'TH', '670': 'TL',
    '672': 'NF', '673': 'BN', '674': 'NR',
    '675': 'PG', '676': 'TO', '677': 'SB',
    '678': 'VU', '679': 'FJ', '680': 'PW',
    '681': 'WF', '682': 'CK', '683': 'NU',
    '685': 'WS', '686': 'KI', '687': 'NC',
    '688': 'TV', '689': 'PF', '690': 'TK',
    '691': 'FM', '692': 'MH', '7': 'RU',
    '77': 'KZ', '800': '001', '808': '001',
    '81': 'JP', '82': 'KR', '84': 'VN',
    '850': 'KP', '852': 'HK', '853': 'MO',
    '855': 'KH', '856': 'LA', '86': 'CN',
    '870': '001', '878': '001', '880': 'BD',
    '881': '001', '882': '001', '883': '001',
    '886': 'TW', '888': '001', '90': 'TR',
    '91': 'IN', '92': 'PK', '93': 'AF',
    '94': 'LK', '95': 'MM', '960': 'MV',
    '961': 'LB', '962': 'JO', '963': 'SY',
    '964': 'IQ', '965': 'KW', '966': 'SA',
    '967': 'YE', '968': 'OM', '970': 'PS',
    '971': 'AE', '972': 'IL', '973': 'BH',
    '974': 'QA', '975': 'BT', '976': 'MN',
    '977': 'NP', '979': '001', '98': 'IR',
    '992': 'TJ', '993': 'TM', '994': 'AZ',
    '995': 'GE', '996': 'KG', '998': 'UZ',
}

# Longest-first, so "1876" (Jamaica) wins over "1" (United States) - a naive
# shortest-match split silently merges countries the target bot keeps as
# completely separate stock.
_CC_LOOKUP: list[tuple[str, str]] = sorted(
    _CC_TO_ISO.items(), key=lambda item: len(item[0]), reverse=True
)


def normalise_number(raw: str) -> str:
    """Digits only, e.g. '+236 77-000' -> '23677000'. Empty if there are none."""
    return "".join(_DIGITS_RE.findall(raw))


def _iso_of(digits: str) -> str | None:
    """ISO region for a digits-only number, or None if it cannot be placed."""
    if not digits:
        return None

    if _HAS_PN:
        try:
            parsed = _pn.parse("+" + digits, None)
        except Exception:  # noqa: BLE001 - malformed input is data, not a bug
            parsed = None
        if parsed is not None:
            # region_code_for_number() is the precise, area-code-aware answer.
            region = _pn.region_code_for_number(parsed)
            if region:
                # Inside a shared calling code the library will happily return
                # a micro-territory for a number equally valid in the main
                # region (+44 7911... reads as Guernsey). NANP is excluded
                # because there the area code genuinely IS the country.
                main = _pn.region_code_for_country_code(parsed.country_code)
                if (
                    main
                    and main != region
                    and parsed.country_code != 1
                    and _pn.is_valid_number_for_region(parsed, main)
                ):
                    return main
                return region
            # Valid country code, unrecognised national part (a test range, a
            # truncated line): the calling code alone is still real information.
            main = _pn.region_code_for_country_code(parsed.country_code)
            if main and main != "ZZ":
                return main

    for prefix, iso in _CC_LOOKUP:
        if digits.startswith(prefix):
            return iso
    return None


def country_of(raw_number: str) -> str:
    """Country for one number, or a '+<prefix>' placeholder when unrecognised."""
    digits = normalise_number(raw_number)
    if not digits:
        return UNKNOWN
    iso = _iso_of(digits)
    if iso:
        return _ISO_NAMES.get(iso, iso)
    # Unrecognised: surface the leading digits so a genuinely unknown code is
    # visible and fixable, instead of being quietly bucketed as Unknown.
    return f"+{digits[:3]}"


def iso_of_number(raw_number: str) -> str | None:
    """ISO-3166 alpha-2 for one number, or None.

    Exposed for callers that would rather match on a stable code than on a
    display name.
    """
    return _iso_of(normalise_number(raw_number))


def country_of_iso(iso: str) -> str:
    """Display name for an ISO region code."""
    code = iso.strip().upper()
    return _ISO_NAMES.get(code, code)


def known_country_names() -> list[str]:
    """Every country name this module can produce, for name matching."""
    return sorted(set(_ISO_NAMES.values()))


def find_country_name(text: str) -> str | None:
    """Longest country name mentioned in free text, or None.

    Used to read a country out of an upload caption. Longest-first so
    "Central African Republic" is not truncated to a shorter name that
    happens to be a substring, and word-boundary anchored so "Chad" does not
    match inside "Chadian numbers ready".
    """
    lowered = f" {' '.join(text.casefold().split())} "
    for name in sorted(set(_ISO_NAMES.values()), key=len, reverse=True):
        needle = name.casefold()
        if re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", lowered):
            return name
    return None


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
