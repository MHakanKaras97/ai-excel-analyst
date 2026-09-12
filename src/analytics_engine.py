import pandas as pd


def _to_native(value):
    if pd.isna(value):
        return None
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
