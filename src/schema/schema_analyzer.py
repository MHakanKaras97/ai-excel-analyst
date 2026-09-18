"""Deterministic semantic schema candidate builder (V0.8.5).

Builds a ColumnSchema for every column of a raw DataFrame using only
already-existing deterministic signals — dtype, data_profiler's inferred
type, and data_normalizer's per-value normalized_type detections (currency/
percentage/numeric/date). No new detection heuristics are invented here for
type/date/numeric detection; this module only *classifies* what those
modules already found into the small role/semantic_type/time_role
vocabulary. This candidate schema is authoritative on its own — an LLM
semantic proposal (V0.8.13) may only override it after Python validation.
"""
import pandas as pd

from src.data_normalizer import normalize_value
from src.data_profiler import profile_dataframe
from src.schema.models import make_column_schema, make_dataset_schema

CURRENCY_UNIT_CODES = {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY"}

MAX_SAMPLE_VALUES = 3
HIGH_UNIQUENESS_RATIO = 0.98
DETERMINISTIC_CONFIDENCE = 0.9
HEURISTIC_CONFIDENCE = 0.65
UNKNOWN_CONFIDENCE = 0.3


def _sample_values(series: pd.Series) -> list:
    non_null = series.dropna()
    samples = []
    for value in non_null.head(MAX_SAMPLE_VALUES):
        if hasattr(value, "item"):
            value = value.item()
        samples.append(str(value))
    return samples


def _dominant_normalized_type(column_report: dict, series: pd.Series) -> str | None:
    """Return "currency"/"percentage" if a clear majority of this text
    column's non-null values normalized to that type, else None. Reuses
    normalize_value's own per-value classification — no re-parsing here."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return None

    counts = {"currency": 0, "percentage": 0}
    for value in non_null:
        result = normalize_value(value)
        normalized_type = result["normalized_type"]
        if normalized_type in counts:
            counts[normalized_type] += 1

    total = len(non_null)
    for normalized_type, count in counts.items():
        if count / total >= 0.5:
            return normalized_type
    return None


def _detect_currency_unit(name: str, series: pd.Series) -> str | None:
    haystacks = [name] + [str(v) for v in series.dropna().head(MAX_SAMPLE_VALUES)]
    for symbol, code in CURRENCY_UNIT_CODES.items():
        if any(symbol in text for text in haystacks):
            return code
    return None


def _has_time_component(series: pd.Series) -> bool:
    non_null = series.dropna()
    if len(non_null) == 0 or not pd.api.types.is_datetime64_any_dtype(non_null):
        return False
    times = pd.DatetimeIndex(non_null)
    return bool(((times.hour != 0) | (times.minute != 0) | (times.second != 0)).any())


def _classify_column(name: str, series: pd.Series, profile_column: dict, row_count: int) -> dict:
    inferred_type = profile_column["inferred_type"]
    unique_count = profile_column["unique_count"]

    if inferred_type == "date-like":
        time_role = "timestamp" if _has_time_component(series) else "period"
        return make_column_schema(
            name, role="date", semantic_type="date", unit=None, time_role=time_role,
            confidence=DETERMINISTIC_CONFIDENCE,
            evidence={"header": name, "sample_values": _sample_values(series), "dtype": profile_column["dtype"]},
        )

    if inferred_type == "numeric":
        return make_column_schema(
            name, role="measure", semantic_type="numeric", unit=None, time_role="none",
            confidence=DETERMINISTIC_CONFIDENCE,
            evidence={"header": name, "sample_values": _sample_values(series), "dtype": profile_column["dtype"]},
        )

    # inferred_type == "text": look for a dominant currency/percentage shape
    # among the raw values before falling back to dimension/identifier text.
    dominant = _dominant_normalized_type(profile_column, series)
    if dominant == "currency":
        unit = _detect_currency_unit(name, series)
        return make_column_schema(
            name, role="measure", semantic_type="currency", unit=unit, time_role="none",
            confidence=HEURISTIC_CONFIDENCE,
            evidence={"header": name, "sample_values": _sample_values(series), "dtype": profile_column["dtype"]},
        )
    if dominant == "percentage":
        return make_column_schema(
            name, role="measure", semantic_type="percentage", unit="%", time_role="none",
            confidence=HEURISTIC_CONFIDENCE,
            evidence={"header": name, "sample_values": _sample_values(series), "dtype": profile_column["dtype"]},
        )

    if row_count > 0 and unique_count / row_count >= HIGH_UNIQUENESS_RATIO and row_count > 1:
        return make_column_schema(
            name, role="identifier", semantic_type="identifier", unit=None, time_role="none",
            confidence=HEURISTIC_CONFIDENCE,
            evidence={"header": name, "sample_values": _sample_values(series), "dtype": profile_column["dtype"]},
        )

    if row_count == 0:
        return make_column_schema(
            name, role="unknown", semantic_type="unknown", unit=None, time_role="none",
            confidence=UNKNOWN_CONFIDENCE,
            evidence={"header": name, "sample_values": [], "dtype": profile_column["dtype"]},
        )

    return make_column_schema(
        name, role="dimension", semantic_type="text", unit=None, time_role="none",
        confidence=HEURISTIC_CONFIDENCE,
        evidence={"header": name, "sample_values": _sample_values(series), "dtype": profile_column["dtype"]},
    )


def analyze_schema(raw_df: pd.DataFrame, profile: dict | None = None) -> dict:
    """Build a deterministic candidate DatasetSchema for `raw_df`.

    `profile` may be passed in (e.g. reusing an already-computed
    FileAnalysis["profile"]) to avoid recomputing it; otherwise it is
    computed here. Column normalization detection is always recomputed
    since analyze_file() does not retain normalize_dataframe()'s report.
    """
    if profile is None:
        profile = profile_dataframe(raw_df)

    row_count = profile["row_count"]
    columns = [
        _classify_column(str(raw_df.columns[i]), raw_df.iloc[:, i], profile["columns"][i], row_count)
        for i in range(len(raw_df.columns))
    ]
    return make_dataset_schema(columns)
