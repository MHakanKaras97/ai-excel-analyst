from datetime import date

import numpy as np
import pandas as pd

METHOD = "iqr"
MIN_OBSERVATIONS = 4
IQR_MULTIPLIER = 1.5


def _to_native(value):
    if pd.isna(value):
        return None
    if isinstance(value, (date, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def _insufficient_data_result(sample_size: int) -> dict:
    return {
        "method": METHOD,
        "insufficient_data": True,
        "sample_size": sample_size,
        "lower_bound": None,
        "upper_bound": None,
        "anomaly_count": 0,
        "anomalies": [],
    }


def detect_iqr_anomalies(values: pd.Series) -> dict:
    """Flag values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] using only non-null,
    numeric observations. Requires at least MIN_OBSERVATIONS non-null values;
    otherwise reports insufficient_data with no anomalies. Input order and
    the original index/position are preserved for each reported anomaly.
    """
    if not pd.api.types.is_numeric_dtype(values):
        return _insufficient_data_result(0)

    non_null = values.dropna()
    if len(non_null) < MIN_OBSERVATIONS:
        return _insufficient_data_result(len(non_null))

    q1 = non_null.quantile(0.25)
    q3 = non_null.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - IQR_MULTIPLIER * iqr
    upper_bound = q3 + IQR_MULTIPLIER * iqr

    anomalies = []
    for position, (label, value) in enumerate(values.items()):
        if pd.isna(value):
            continue
        if value > upper_bound:
            direction = "high"
        elif value < lower_bound:
            direction = "low"
        else:
            continue
        anomalies.append({
            "position": position,
            "period": _to_native(label),
            "value": _to_native(value),
            "direction": direction,
        })

    return {
        "method": METHOD,
        "insufficient_data": False,
        "sample_size": len(non_null),
        "lower_bound": _to_native(lower_bound),
        "upper_bound": _to_native(upper_bound),
        "anomaly_count": len(anomalies),
        "anomalies": anomalies,
    }
