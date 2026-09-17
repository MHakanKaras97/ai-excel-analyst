import difflib
import re

# Conservative similarity cutoff for difflib.get_close_matches (0.0-1.0).
# The stdlib default (0.6) is loose enough to match weak references like
# "Rev" -> "Revenue"; 0.8 only accepts near-identical spellings (e.g. a
# single missing/typo'd character) and rejects short/partial references,
# so ambiguity is never silently guessed away. Adjust here if it proves
# too strict or too loose in practice.
FUZZY_CUTOFF = 0.8

MONTH_NAMES = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

PERIOD_LABEL_PATTERN = re.compile(r"^\d{4}-\d{2}$")


def _column_result(found: bool, reason, column=None, candidates=None) -> dict:
    return {
        "found": found,
        "reason": reason,
        "column": column,
        "candidates": candidates or [],
    }


def resolve_column(column_hint: str, column_names: list[str]) -> dict:
    """Resolve a free-text column hint against known column names.

    Resolution order: exact case-insensitive match, then (only if no exact
    match exists) a conservative fuzzy match via difflib. Ambiguity at
    either stage is reported, never silently guessed.
    """
    names = list(column_names)

    if not column_hint or not column_hint.strip():
        return _column_result(False, "column_not_found")

    hint = column_hint.strip()

    exact_matches = [name for name in names if name.lower() == hint.lower()]
    if len(exact_matches) == 1:
        return _column_result(True, None, column=exact_matches[0])
    if len(exact_matches) > 1:
        return _column_result(False, "ambiguous_column", candidates=exact_matches)

    if not names:
        return _column_result(False, "column_not_found")

    lower_names = [name.lower() for name in names]
    close = difflib.get_close_matches(hint.lower(), lower_names, n=len(lower_names), cutoff=FUZZY_CUTOFF)

    candidates = []
    for lower_match in close:
        for name in names:
            if name.lower() == lower_match and name not in candidates:
                candidates.append(name)

    if len(candidates) == 1:
        return _column_result(True, None, column=candidates[0])
    if len(candidates) > 1:
        return _column_result(False, "ambiguous_column", candidates=candidates)

    return _column_result(False, "column_not_found")


def _period_result(found: bool, reason, period=None, candidates=None) -> dict:
    return {
        "found": found,
        "reason": reason,
        "period": period,
        "candidates": candidates or [],
    }


def _month_number(token: str):
    token_lower = token.lower()
    if token_lower in MONTH_NAMES:
        return MONTH_NAMES[token_lower]
    if token.isdigit() and 1 <= int(token) <= 12:
        return int(token)
    return None


def _parse_month_and_year(hint: str):
    """Parse a two-token "<month> <year>" hint (e.g. "March 2024", "Mar 2024",
    "03 2024") into (month, year) ints. Returns None for any other shape."""
    parts = hint.split()
    if len(parts) != 2:
        return None

    month_token, year_token = parts
    if not (year_token.isdigit() and len(year_token) == 4):
        return None

    month_num = _month_number(month_token)
    if month_num is None:
        return None

    return month_num, int(year_token)


def _month_of_label(label: str) -> int:
    return int(label[5:7])


def resolve_period(period_hint: str, period_labels: list[str]) -> dict:
    """Resolve a free-text period hint against known "YYYY-MM" period labels.

    Resolution order: exact known label, then a fully-specified month+year
    normalized to "YYYY-MM", then a bare month name/number matched across
    all known years. Ambiguity is always reported, never defaulted to the
    latest/first candidate.
    """
    labels = list(period_labels)

    if not period_hint or not period_hint.strip():
        return _period_result(False, "period_not_found")

    hint = period_hint.strip()

    if hint in labels:
        return _period_result(True, None, period=hint)

    month_year = _parse_month_and_year(hint)
    if month_year is not None:
        month_num, year = month_year
        candidate = f"{year:04d}-{month_num:02d}"
        if candidate in labels:
            return _period_result(True, None, period=candidate)
        return _period_result(False, "period_not_found")

    if len(hint.split()) == 1:
        month_num = _month_number(hint)
        if month_num is not None:
            matches = sorted(label for label in labels if PERIOD_LABEL_PATTERN.match(label) and _month_of_label(label) == month_num)
            if len(matches) == 1:
                return _period_result(True, None, period=matches[0])
            if len(matches) > 1:
                return _period_result(False, "ambiguous_period", candidates=matches)
            return _period_result(False, "period_not_found")

    return _period_result(False, "period_not_found")
