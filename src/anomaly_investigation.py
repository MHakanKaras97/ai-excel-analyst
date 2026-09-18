"""Anomaly investigation (V0.8.12).

V0.7's detect_iqr_anomalies() answers "is this value anomalous?". This
module answers "what deterministic evidence surrounds it?" — for one
already-detected anomaly, it gathers the period's own value, the previous
period's value, the period-over-period change (delegated entirely to
analytics_engine.compare_values(), never recomputed here), and, for any
other already-analyzed numeric column sharing the same period axis, that
column's own value/change so a *coincident* movement can be surfaced.

This module never claims causality. It only reports "X also changed
during the same period" — phrasing and any causal-language guarding is
the AI-facing layer's job (src/evidence/validator.py's
contains_unsupported_causal_claim), not this deterministic layer's.
"""
from src.analytics_engine import compare_values


def _related_measure_change(column_name: str, monthly_series, period: str) -> dict | None:
    if monthly_series is None or period not in monthly_series.index:
        return None

    position = list(monthly_series.index).index(period)
    if position == 0:
        previous_value, current_value = None, monthly_series.iloc[position]
        comparison = {"previous": None, "current": float(current_value), "absolute_change": None,
                      "percentage_change": None, "is_valid": False, "reason": "no_previous_period"}
    else:
        previous_value = monthly_series.iloc[position - 1]
        current_value = monthly_series.iloc[position]
        comparison = compare_values(previous_value, current_value)

    return {"column": column_name, **comparison}


def investigate_anomaly(anomaly: dict, column: str, monthly_series, related_series: dict | None = None) -> dict:
    """Build a deterministic evidence bundle around one already-detected
    anomaly record (an item from detect_iqr_anomalies()'s "anomalies" list).

    `related_series` is an optional {column_name: monthly_series} mapping
    for OTHER already-computed numeric columns sharing the same period
    axis — used only to report whether they also moved during the same
    period, never to imply why the anomaly occurred.
    """
    period = anomaly.get("period")

    own_change = _related_measure_change(column, monthly_series, period)

    coincident_changes = []
    for other_column, other_series in (related_series or {}).items():
        if other_column == column:
            continue
        change = _related_measure_change(other_column, other_series, period)
        if change is not None:
            coincident_changes.append(change)

    return {
        "period": period,
        "column": column,
        "value": anomaly.get("value"),
        "direction": anomaly.get("direction"),
        "own_change": own_change,
        "coincident_changes": coincident_changes,
    }
