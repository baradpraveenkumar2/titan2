"""
Month-level holiday / event calendar used by both the dataset builder and the
forecasting engine (so future months get a known holiday flag).

Holidays are assigned per COUNTRY and per MONTH because the data is monthly.
Islamic dates shift ~11 days earlier each year, so they are listed explicitly.
"""
from __future__ import annotations

NONE = "No Holiday"

MENA_RAMADAN = {
    "United Arab Emirates", "Saudi Arabia", "Qatar", "Oman",
    "Bahrain", "Kuwait", "Egypt", "Morocco",
}
EID_ADHA_COUNTRIES = MENA_RAMADAN | {"Kenya", "Ghana", "South Africa"}

# month -> holiday name (applies to MENA_RAMADAN countries)
RAMADAN_MONTHS = {
    "2025-03": "Ramadan & Eid al-Fitr",
    "2026-02": "Ramadan",
    "2026-03": "Ramadan & Eid al-Fitr",
    "2027-02": "Ramadan",
    "2027-03": "Ramadan & Eid al-Fitr",
}
EID_ADHA_MONTHS = {"2025-06", "2026-05", "2027-05"}

# Country specific national days (month in which the national day falls)
NATIONAL_DAY_MONTHS = {
    "United Arab Emirates": {"12"},   # 2 Dec
    "Qatar": {"12"},                  # 18 Dec
    "Bahrain": {"12"},                # 16 Dec
    "Saudi Arabia": {"09"},           # 23 Sep
    "Kuwait": {"02"},                 # 25 Feb
}
CHRISTMAS_COUNTRIES = {"South Africa", "Kenya", "Ghana", "Egypt"}

# Display order / list of all holiday names (useful for the app and for dummies)
ALL_HOLIDAYS = [
    "Ramadan", "Ramadan & Eid al-Fitr", "Eid al-Adha",
    "White Friday", "National Day", "Christmas / New Year",
]


def holiday_for(country: str, month: str) -> str:
    """Return the holiday name for a country in a 'YYYY-MM' month, or 'No Holiday'."""
    mm = month[5:7]
    if country in MENA_RAMADAN and month in RAMADAN_MONTHS:
        return RAMADAN_MONTHS[month]
    if country in EID_ADHA_COUNTRIES and month in EID_ADHA_MONTHS:
        return "Eid al-Adha"
    if mm == "11":
        return "White Friday"
    if mm in NATIONAL_DAY_MONTHS.get(country, set()):
        return "National Day"
    if mm == "12" and country in CHRISTMAS_COUNTRIES:
        return "Christmas / New Year"
    return NONE
