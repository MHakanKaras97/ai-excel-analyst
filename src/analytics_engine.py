from datetime import date

import numpy as np
import pandas as pd


def _to_native(value):
    if pd.isna(value):
        return None
    if isinstance(value, (date, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def _numeric_column_stats(series: pd.Series) -> dict:
    non_null = series.dropna()
    count = len(non_null)
    missing_count = len(series) - count

    if count == 0:
        return {
            "count": count,
            "missing_count": missing_count,
            "sum": None,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "std": None,
        }

    std = non_null.std()

    return {
        "count": count,
        "missing_count": missing_count,
        "sum": float(non_null.sum()),
        "mean": float(non_null.mean()),
        "median": float(non_null.median()),
        "min": float(non_null.min()),
        "max": float(non_null.max()),
        "std": None if pd.isna(std) else float(std),
    }


def analyze_numeric(df: pd.DataFrame) -> dict:
    columns = []
    for i in range(len(df.columns)):
        series = df.iloc[:, i]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        stats = _numeric_column_stats(series)
        columns.append({"name": str(df.columns[i]), **stats})

    return {"columns": columns}


def _date_column_stats(series: pd.Series) -> dict:
    non_null = series.dropna()
    count = len(non_null)
    missing_count = len(series) - count

    if count == 0:
        return {
            "count": count,
            "missing_count": missing_count,
            "min": None,
            "max": None,
            "duration_days": None,
        }

    min_date = non_null.min()
    max_date = non_null.max()

    return {
        "count": count,
        "missing_count": missing_count,
        "min": min_date.isoformat(),
        "max": max_date.isoformat(),
        "duration_days": (max_date - min_date).days,
    }


def analyze_dates(df: pd.DataFrame) -> dict:
    columns = []
    for i in range(len(df.columns)):
        series = df.iloc[:, i]
        if not pd.api.types.is_datetime64_any_dtype(series):
            continue
        stats = _date_column_stats(series)
        columns.append({"name": str(df.columns[i]), **stats})

    return {"columns": columns}


def compare_values(previous, current) -> dict:
    if pd.isna(previous) or pd.isna(current):
        return {
            "previous": _to_native(previous),
            "current": _to_native(current),
            "absolute_change": None,
            "percentage_change": None,
            "is_valid": False,
            "reason": "missing_value",
        }

    absolute_change = current - previous

    if previous == 0:
        if current == 0:
            return {
                "previous": _to_native(previous),
                "current": _to_native(current),
                "absolute_change": _to_native(absolute_change),
                "percentage_change": 0.0,
                "is_valid": True,
                "reason": None,
            }
        return {
            "previous": _to_native(previous),
            "current": _to_native(current),
            "absolute_change": _to_native(absolute_change),
            "percentage_change": None,
            "is_valid": False,
            "reason": "division_by_zero",
        }

    percentage_change = (current - previous) / previous * 100

    return {
        "previous": _to_native(previous),
        "current": _to_native(current),
        "absolute_change": _to_native(absolute_change),
        "percentage_change": _to_native(percentage_change),
        "is_valid": True,
        "reason": None,
    }


def compare_periods(values: pd.Series) -> dict:
    periods = [_to_native(label) for label in values.index]
    n = len(values)

    if n < 2 or not pd.api.types.is_numeric_dtype(values):
        return {
            "periods": periods,
            "comparisons": [],
            "valid_comparison_count": 0,
            "invalid_comparison_count": 0,
            "insufficient_data": True,
        }

    comparisons = []
    valid_comparison_count = 0
    invalid_comparison_count = 0

    for i in range(n - 1):
        result = compare_values(values.iloc[i], values.iloc[i + 1])
        comparisons.append({
            "from_period": periods[i],
            "to_period": periods[i + 1],
            **result,
        })
        if result["is_valid"]:
            valid_comparison_count += 1
        else:
            invalid_comparison_count += 1

    return {
        "periods": periods,
        "comparisons": comparisons,
        "valid_comparison_count": valid_comparison_count,
        "invalid_comparison_count": invalid_comparison_count,
        "insufficient_data": False,
    }


def detect_trend(values: pd.Series) -> dict:
    periods_result = compare_periods(values)

    increase_count = 0
    decrease_count = 0
    no_change_count = 0
    direction_comparison_count = 0

    for comparison in periods_result["comparisons"]:
        absolute_change = comparison["absolute_change"]
        if absolute_change is None:
            continue
        direction_comparison_count += 1
        if absolute_change > 0:
            increase_count += 1
        elif absolute_change < 0:
            decrease_count += 1
        else:
            no_change_count += 1

    if direction_comparison_count == 0:
        trend = "insufficient_data"
    elif increase_count == direction_comparison_count:
        trend = "increasing"
    elif decrease_count == direction_comparison_count:
        trend = "decreasing"
    elif no_change_count == direction_comparison_count:
        trend = "stable"
    else:
        trend = "volatile"

    return {
        "trend": trend,
        "valid_comparison_count": periods_result["valid_comparison_count"],
        "invalid_comparison_count": periods_result["invalid_comparison_count"],
        "direction_comparison_count": direction_comparison_count,
        "increase_count": increase_count,
        "decrease_count": decrease_count,
        "no_change_count": no_change_count,
    }
