import re
import unicodedata
from datetime import date as _date
from datetime import datetime as _datetime

import numpy as np
import pandas as pd

RECOGNIZED_CURRENCY_SYMBOLS = {"$", "€", "£", "¥"}

PLAIN_DECIMAL_RE = re.compile(r"^-?\d+(\.\d+)?$")
SINGLE_DOT_3DIGIT_RE = re.compile(r"^-?\d+\.\d{3}$")
COMMA_THOUSANDS_RE = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")
MALFORMED_COMMA_RE = re.compile(r"^-?\d+(,\d+)+(\.\d+)?$")
AMBIGUOUS_NUMERIC_DATE_RE = re.compile(r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$")

DATE_FORMATS = ["%Y-%m-%d", "%Y/%m/%d", "%B %d, %Y", "%d %B %Y"]


def _changed(raw, normalized_value) -> bool:
    if pd.isna(raw):
        return False
    return raw != normalized_value


def _result(raw, normalized_value, normalized_type, is_ambiguous, warning) -> dict:
    return {
        "raw": raw,
        "normalized_value": normalized_value,
        "normalized_type": normalized_type,
        "was_changed": _changed(raw, normalized_value),
        "is_ambiguous": is_ambiguous,
        "warning": warning,
    }


def _parse_numeric_string(s: str):
    if s == "":
        return "not_numeric", None, None

    if PLAIN_DECIMAL_RE.match(s):
        if SINGLE_DOT_3DIGIT_RE.match(s):
            return "ambiguous", None, "ambiguous_decimal_or_thousands"
        return "resolved", float(s), None

    if COMMA_THOUSANDS_RE.match(s):
        return "resolved", float(s.replace(",", "")), None

    if MALFORMED_COMMA_RE.match(s):
        return "ambiguous", None, "ambiguous_thousands_separator"

    return "not_numeric", None, None


def _parse_date_string(s: str):
    for fmt in DATE_FORMATS:
        try:
            return "resolved", _datetime.strptime(s, fmt).date(), None
        except ValueError:
            continue

    if AMBIGUOUS_NUMERIC_DATE_RE.match(s):
        return "ambiguous", None, "ambiguous_date_format"

    return "not_date", None, None


def _normalize_string(raw: str) -> dict:
    s = raw.strip()

    if s == "":
        return _result(raw, raw, None, False, None)

    if "%" in s:
        if s.count("%") == 1 and s.endswith("%"):
            inner = s[:-1].strip()
            status, number, warning = _parse_numeric_string(inner)
            if status == "resolved":
                return _result(raw, number / 100.0, "percentage", False, None)
            return _result(raw, raw, None, True, warning or "invalid_percentage_format")
        return _result(raw, raw, None, True, "invalid_percentage_format")

    currency_positions = [i for i, ch in enumerate(s) if unicodedata.category(ch) == "Sc"]
    if currency_positions:
        if len(currency_positions) > 1:
            return _result(raw, raw, None, True, "multiple_currency_symbols")

        idx = currency_positions[0]
        if idx not in (0, len(s) - 1):
            return _result(raw, raw, None, True, "unsupported_currency_symbol_position")

        symbol = s[idx]
        if symbol not in RECOGNIZED_CURRENCY_SYMBOLS:
            return _result(raw, raw, None, True, "unrecognized_currency_symbol")

        remainder = (s[:idx] + s[idx + 1:]).strip()
        status, number, warning = _parse_numeric_string(remainder)
        if status == "resolved":
            return _result(raw, number, "currency", False, None)
        return _result(raw, raw, None, True, warning or "invalid_currency_format")

    status, number, warning = _parse_numeric_string(s)
    if status == "resolved":
        return _result(raw, number, "numeric", False, None)
    if status == "ambiguous":
        return _result(raw, raw, None, True, warning)

    date_status, parsed_date, date_warning = _parse_date_string(s)
    if date_status == "resolved":
        return _result(raw, parsed_date.isoformat(), "date", False, None)
    if date_status == "ambiguous":
        return _result(raw, raw, None, True, date_warning)

    return _result(raw, raw, None, False, None)


def normalize_value(value) -> dict:
    if pd.isna(value):
        return _result(value, value, None, False, None)

    if isinstance(value, bool):
        return _result(value, value, None, False, None)

    if isinstance(value, (int, float, np.integer, np.floating)):
        return _result(value, value, "numeric", False, None)

    if isinstance(value, (_date, np.datetime64)):
        return _result(value, value, "date", False, None)

    if isinstance(value, str):
        return _normalize_string(value)

    return _result(value, value, None, False, None)


MAX_UNRESOLVED_EXAMPLES = 5


def normalize_series(series: pd.Series) -> dict:
    total = len(series)

    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
        return {
            "normalized_series": series.copy(),
            "report": {
                "total": total,
                "changed_count": 0,
                "unresolved_count": 0,
                "missing_count": int(series.isna().sum()),
                "examples": {"unresolved": []},
            },
        }

    normalized_values = []
    changed_count = 0
    unresolved_count = 0
    missing_count = 0
    unresolved_examples = []

    for value in series:
        result = normalize_value(value)
        normalized_values.append(result["normalized_value"])

        if pd.isna(value):
            missing_count += 1
        if result["was_changed"]:
            changed_count += 1
        if result["is_ambiguous"]:
            unresolved_count += 1
            if len(unresolved_examples) < MAX_UNRESOLVED_EXAMPLES:
                unresolved_examples.append({"raw": result["raw"], "warning": result["warning"]})

    normalized_series = pd.Series(normalized_values, index=series.index, name=series.name)

    return {
        "normalized_series": normalized_series,
        "report": {
            "total": total,
            "changed_count": changed_count,
            "unresolved_count": unresolved_count,
            "missing_count": missing_count,
            "examples": {"unresolved": unresolved_examples},
        },
    }


def normalize_dataframe(df: pd.DataFrame) -> dict:
    total_columns = len(df.columns)
    columns_report = []

    normalized_df = pd.DataFrame(index=df.index.copy())

    for i in range(total_columns):
        series = df.iloc[:, i]
        result = normalize_series(series)
        normalized_df.insert(i, df.columns[i], result["normalized_series"].to_numpy(), allow_duplicates=True)

        columns_report.append({
            "name": str(df.columns[i]),
            "position": i,
            **result["report"],
        })

    normalized_df.columns = df.columns.copy()

    return {
        "normalized_df": normalized_df,
        "report": {
            "total_columns": total_columns,
            "columns": columns_report,
        },
    }
