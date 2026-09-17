import copy
import json

import pandas as pd

import src.qa_engine as qa_engine
from src.qa_chart_engine import build_chart_spec


def _intent(intent_name, column_hint=None, metric=None, period_hint=None,
            from_period_hint=None, to_period_hint=None):
    return {
        "intent": intent_name,
        "column_hint": column_hint,
        "metric": metric,
        "period_hint": period_hint,
        "from_period_hint": from_period_hint,
        "to_period_hint": to_period_hint,
    }


def _numeric_summary():
    return {
        "columns": [
            {"name": "TotalPrice", "count": 3, "missing_count": 0, "sum": 450.0,
             "mean": 150.0, "median": 150.0, "min": 100.0, "max": 200.0, "std": 50.0},
            {"name": "Revenue", "count": 3, "missing_count": 0, "sum": 900.0,
             "mean": 300.0, "median": 300.0, "min": 200.0, "max": 400.0, "std": 100.0},
        ],
    }


def _monthly_series():
    return pd.Series([100.0, 200.0, 150.0], index=["2024-01", "2024-02", "2024-03"], name="TotalPrice")


def _period_comparison():
    return {
        "periods": ["2024-01", "2024-02", "2024-03"],
        "comparisons": [
            {
                "from_period": "2024-01", "to_period": "2024-02",
                "previous": 100.0, "current": 200.0,
                "absolute_change": 100.0, "percentage_change": 100.0,
                "is_valid": True, "reason": None,
            },
            {
                "from_period": "2024-02", "to_period": "2024-03",
                "previous": 200.0, "current": 150.0,
                "absolute_change": -50.0, "percentage_change": -25.0,
                "is_valid": True, "reason": None,
            },
        ],
        "valid_comparison_count": 2,
        "invalid_comparison_count": 0,
        "insufficient_data": False,
    }


def _anomalies(with_anomaly=True):
    if with_anomaly:
        return {
            "method": "iqr", "insufficient_data": False, "sample_size": 3,
            "lower_bound": 10.0, "upper_bound": 190.0, "anomaly_count": 1,
            "anomalies": [{"position": 1, "period": "2024-02", "value": 200.0, "direction": "high"}],
        }
    return {
        "method": "iqr", "insufficient_data": False, "sample_size": 3,
        "lower_bound": 10.0, "upper_bound": 300.0, "anomaly_count": 0, "anomalies": [],
    }


def _profile(with_missing=True):
    region_missing = 2 if with_missing else 0
    return {
        "row_count": 10, "column_count": 2, "is_empty": False,
        "duplicate_row_count": 0, "duplicate_column_names": [],
        "columns": [
            {"name": "TotalPrice", "dtype": "float64", "inferred_type": "numeric",
             "unique_count": 3, "missing_count": 0, "missing_percentage": 0.0},
            {"name": "Region", "dtype": "object", "inferred_type": "text",
             "unique_count": 3, "missing_count": region_missing, "missing_percentage": region_missing * 10.0},
        ],
    }


def _payload(**overrides):
    payload = {
        "numeric_summary": _numeric_summary(),
        "period_comparison": _period_comparison(),
        "profile": _profile(),
    }
    payload.update(overrides)
    return payload


# ==================================================
# GENERAL
# ==================================================


def test_valid_trend_intent_succeeds():
    result = build_chart_spec(_intent("trend_chart", column_hint="TotalPrice"), _payload(), _monthly_series())

    assert result["found"] is True
    assert result["reason"] is None
    assert result["intent"] == "trend_chart"
    assert result["chart_type"] == "trend"


def test_unsupported_intent_fails_cleanly():
    result = build_chart_spec(_intent("unsupported"), _payload(), _monthly_series())

    assert result == {
        "found": False, "reason": "unsupported", "intent": "unsupported",
        "chart_type": None, "column": None, "metric": None, "period": None,
        "from_period": None, "to_period": None, "data": {}, "extra": {},
    }


def test_unknown_intent_name_is_treated_as_unsupported():
    result = build_chart_spec(_intent("forecast_chart"), _payload(), _monthly_series())

    assert result["found"] is False
    assert result["reason"] == "unsupported"


def test_no_mutation_of_intent_payload_or_series():
    intent = _intent("trend_chart", column_hint="TotalPrice")
    payload = _payload()
    series = _monthly_series()
    intent_before = copy.deepcopy(intent)
    payload_before = copy.deepcopy(payload)
    series_before = series.copy()

    build_chart_spec(intent, payload, series, _anomalies())

    assert intent == intent_before
    assert payload == payload_before
    assert series.equals(series_before)


def test_all_intent_results_are_json_serializable():
    cases = [
        build_chart_spec(_intent("trend_chart"), _payload(), _monthly_series(), _anomalies()),
        build_chart_spec(_intent("numeric_summary_chart", column_hint="TotalPrice"), _payload(), None),
        build_chart_spec(_intent("period_change_chart"), _payload(), _monthly_series()),
        build_chart_spec(_intent("missing_values_chart"), _payload(), None),
        build_chart_spec(_intent("unsupported"), _payload(), None),
    ]
    for result in cases:
        json.dumps(result)


# ==================================================
# TREND CHART
# ==================================================


def test_trend_chart_success_with_no_column_hint_uses_series_name():
    result = build_chart_spec(_intent("trend_chart"), _payload(), _monthly_series())

    assert result["found"] is True
    assert result["column"] == "TotalPrice"
    assert result["data"]["periods"] == ["2024-01", "2024-02", "2024-03"]
    assert result["data"]["values"] == [100.0, 200.0, 150.0]


def test_trend_chart_success_with_matching_column_hint():
    result = build_chart_spec(_intent("trend_chart", column_hint="TotalPrice"), _payload(), _monthly_series())

    assert result["found"] is True
    assert result["column"] == "TotalPrice"


def test_trend_chart_column_not_found():
    result = build_chart_spec(_intent("trend_chart", column_hint="Nonexistent"), _payload(), _monthly_series())

    assert result["found"] is False
    assert result["reason"] == "column_not_found"


def test_trend_chart_ambiguous_column():
    payload = _payload(numeric_summary={"columns": [
        {"name": "Revenue", "sum": 1.0}, {"name": "REVENUE", "sum": 1.0},
    ]})

    result = build_chart_spec(_intent("trend_chart", column_hint="revenue"), payload, _monthly_series())

    assert result["found"] is False
    assert result["reason"] == "ambiguous_column"
    assert set(result["extra"]["candidates"]) == {"Revenue", "REVENUE"}


def test_trend_chart_column_series_mismatch():
    result = build_chart_spec(_intent("trend_chart", column_hint="Revenue"), _payload(), _monthly_series())

    assert result["found"] is False
    assert result["reason"] == "column_series_mismatch"


def test_trend_chart_monthly_series_missing():
    result = build_chart_spec(_intent("trend_chart"), _payload(), None)

    assert result["found"] is False
    assert result["reason"] == "trend_not_computed"


def test_trend_chart_monthly_series_empty():
    empty_series = pd.Series([], dtype="float64", name="TotalPrice")

    result = build_chart_spec(_intent("trend_chart"), _payload(), empty_series)

    assert result["found"] is False
    assert result["reason"] == "trend_not_computed"


def test_trend_chart_includes_anomalies_when_available():
    anomalies = _anomalies(with_anomaly=True)

    result = build_chart_spec(_intent("trend_chart"), _payload(), _monthly_series(), anomalies)

    assert result["extra"]["anomalies"] == anomalies


def test_trend_chart_anomalies_absent_is_none_not_recalculated():
    result = build_chart_spec(_intent("trend_chart"), _payload(), _monthly_series(), None)

    assert result["extra"]["anomalies"] is None


# ==================================================
# NUMERIC SUMMARY CHART
# ==================================================


def test_numeric_summary_chart_success_with_explicit_column_and_metric():
    result = build_chart_spec(
        _intent("numeric_summary_chart", column_hint="TotalPrice", metric="mean"), _payload(), None,
    )

    assert result["found"] is True
    assert result["chart_type"] == "numeric_summary"
    assert result["column"] == "TotalPrice"
    assert result["metric"] == "mean"
    assert result["data"] == {"columns": [_numeric_summary()["columns"][0]]}


def test_numeric_summary_chart_default_metric_is_sum_when_null():
    result = build_chart_spec(_intent("numeric_summary_chart", column_hint="TotalPrice"), _payload(), None)

    assert result["found"] is True
    assert result["metric"] == "sum"


def test_numeric_summary_chart_every_supported_metric_succeeds():
    for metric in ("sum", "mean", "median", "min", "max", "std", "count"):
        result = build_chart_spec(
            _intent("numeric_summary_chart", column_hint="TotalPrice", metric=metric), _payload(), None,
        )
        assert result["found"] is True
        assert result["metric"] == metric


def test_numeric_summary_chart_column_required_when_hint_is_null():
    result = build_chart_spec(_intent("numeric_summary_chart"), _payload(), None)

    assert result["found"] is False
    assert result["reason"] == "column_required"


def test_numeric_summary_chart_column_not_found():
    result = build_chart_spec(
        _intent("numeric_summary_chart", column_hint="Nonexistent"), _payload(), None,
    )

    assert result["found"] is False
    assert result["reason"] == "column_not_found"


def test_numeric_summary_chart_ambiguous_column():
    payload = _payload(numeric_summary={"columns": [
        {"name": "Revenue", "sum": 1.0}, {"name": "REVENUE", "sum": 1.0},
    ]})

    result = build_chart_spec(_intent("numeric_summary_chart", column_hint="revenue"), payload, None)

    assert result["found"] is False
    assert result["reason"] == "ambiguous_column"


def test_numeric_summary_chart_numeric_summary_missing_is_column_not_found():
    payload = _payload(numeric_summary=None)

    result = build_chart_spec(
        _intent("numeric_summary_chart", column_hint="TotalPrice"), payload, None,
    )

    assert result["found"] is False
    assert result["reason"] == "column_not_found"


def test_numeric_summary_chart_unsupported_metric():
    result = build_chart_spec(
        _intent("numeric_summary_chart", column_hint="TotalPrice", metric="variance"), _payload(), None,
    )

    assert result["found"] is False
    assert result["reason"] == "unsupported_metric"


def test_numeric_summary_chart_filtered_summary_contains_only_requested_column():
    result = build_chart_spec(
        _intent("numeric_summary_chart", column_hint="Revenue", metric="sum"), _payload(), None,
    )

    assert len(result["data"]["columns"]) == 1
    assert result["data"]["columns"][0]["name"] == "Revenue"


def test_numeric_summary_chart_does_not_recalculate_statistics():
    numeric_summary = _numeric_summary()
    payload = _payload(numeric_summary=numeric_summary)

    result = build_chart_spec(
        _intent("numeric_summary_chart", column_hint="TotalPrice", metric="sum"), payload, None,
    )

    # Value must be read verbatim from the already-computed summary, not derived.
    assert result["data"]["columns"][0]["sum"] == numeric_summary["columns"][0]["sum"]


# ==================================================
# PERIOD CHANGE CHART
# ==================================================


def test_period_change_chart_full_comparison_when_no_period_hints():
    period_comparison = _period_comparison()
    payload = _payload(period_comparison=period_comparison)

    result = build_chart_spec(_intent("period_change_chart"), payload, _monthly_series())

    assert result["found"] is True
    assert result["chart_type"] == "period_change"
    assert result["data"] == period_comparison


def test_period_change_chart_general_request_returns_isolated_deep_copy():
    period_comparison = _period_comparison()
    payload = _payload(period_comparison=period_comparison)

    result = build_chart_spec(_intent("period_change_chart"), payload, _monthly_series())

    assert result["found"] is True
    assert result["data"] == period_comparison
    assert result["data"] is not period_comparison
    assert result["data"]["comparisons"] is not period_comparison["comparisons"]

    result["data"]["comparisons"][0]["absolute_change"] = 999999.0
    result["data"]["periods"].append("2099-01")

    assert payload["period_comparison"] == period_comparison
    assert payload["period_comparison"]["comparisons"][0]["absolute_change"] != 999999.0
    assert "2099-01" not in payload["period_comparison"]["periods"]


def test_period_change_chart_specific_january_to_march_filtering():
    payload = _payload(period_comparison={
        "periods": ["2024-01", "2024-02", "2024-03"],
        "comparisons": [
            {"from_period": "2024-01", "to_period": "2024-03", "previous": 100.0, "current": 150.0,
             "absolute_change": 50.0, "percentage_change": 50.0, "is_valid": True, "reason": None},
        ],
        "valid_comparison_count": 1, "invalid_comparison_count": 0, "insufficient_data": False,
    })

    result = build_chart_spec(
        _intent("period_change_chart", from_period_hint="January", to_period_hint="March"), payload, None,
    )

    assert result["found"] is True
    assert result["from_period"] == "2024-01"
    assert result["to_period"] == "2024-03"


def test_period_change_chart_returns_only_the_exact_matched_comparison():
    result = build_chart_spec(
        _intent("period_change_chart", from_period_hint="2024-01", to_period_hint="2024-02"), _payload(), None,
    )

    assert len(result["data"]["comparisons"]) == 1
    assert result["data"]["comparisons"][0]["from_period"] == "2024-01"
    assert result["data"]["comparisons"][0]["to_period"] == "2024-02"


def test_period_change_chart_from_period_only_is_incomplete():
    result = build_chart_spec(
        _intent("period_change_chart", from_period_hint="2024-01"), _payload(), None,
    )

    assert result["found"] is False
    assert result["reason"] == "period_range_incomplete"


def test_period_change_chart_to_period_only_is_incomplete():
    result = build_chart_spec(
        _intent("period_change_chart", to_period_hint="2024-02"), _payload(), None,
    )

    assert result["found"] is False
    assert result["reason"] == "period_range_incomplete"


def test_period_change_chart_period_not_found():
    result = build_chart_spec(
        _intent("period_change_chart", from_period_hint="2024-01", to_period_hint="December"), _payload(), None,
    )

    assert result["found"] is False
    assert result["reason"] == "period_not_found"


def test_period_change_chart_ambiguous_period():
    payload = _payload(period_comparison={
        "periods": ["2023-03", "2024-03", "2024-05"],
        "comparisons": [],
        "valid_comparison_count": 0, "invalid_comparison_count": 0, "insufficient_data": False,
    })

    result = build_chart_spec(
        _intent("period_change_chart", from_period_hint="March", to_period_hint="2024-05"), payload, None,
    )

    assert result["found"] is False
    assert result["reason"] == "ambiguous_period"
    assert result["extra"]["candidates"] == ["2023-03", "2024-03"]


def test_period_change_chart_change_not_found_when_transition_does_not_exist():
    result = build_chart_spec(
        _intent("period_change_chart", from_period_hint="2024-01", to_period_hint="2024-03"), _payload(), None,
    )

    assert result["found"] is False
    assert result["reason"] == "period_change_not_found"


def test_period_change_chart_missing_period_comparison():
    payload = _payload(period_comparison=None)

    result = build_chart_spec(_intent("period_change_chart"), payload, None)

    assert result["found"] is False
    assert result["reason"] == "period_comparison_not_computed"


def test_period_change_chart_matching_column_hint_succeeds():
    result = build_chart_spec(
        _intent("period_change_chart", column_hint="TotalPrice"), _payload(), _monthly_series(),
    )

    assert result["found"] is True
    assert result["column"] == "TotalPrice"


def test_period_change_chart_column_series_mismatch():
    result = build_chart_spec(
        _intent("period_change_chart", column_hint="Revenue"), _payload(), _monthly_series(),
    )

    assert result["found"] is False
    assert result["reason"] == "column_series_mismatch"


def test_period_change_chart_does_not_infer_reverse_transition():
    # Only 2024-01 -> 2024-02 exists; asking for the reverse must not match it.
    result = build_chart_spec(
        _intent("period_change_chart", from_period_hint="2024-02", to_period_hint="2024-01"), _payload(), None,
    )

    assert result["found"] is False
    assert result["reason"] == "period_change_not_found"


def test_period_change_chart_filtered_counts_are_correct():
    payload = _payload(period_comparison={
        "periods": ["2024-01", "2024-02"],
        "comparisons": [
            {"from_period": "2024-01", "to_period": "2024-02", "previous": 100.0, "current": None,
             "absolute_change": None, "percentage_change": None, "is_valid": False, "reason": "missing_value"},
        ],
        "valid_comparison_count": 0, "invalid_comparison_count": 1, "insufficient_data": False,
    })

    result = build_chart_spec(
        _intent("period_change_chart", from_period_hint="2024-01", to_period_hint="2024-02"), payload, None,
    )

    assert result["data"]["valid_comparison_count"] == 0
    assert result["data"]["invalid_comparison_count"] == 1


def test_period_change_chart_original_period_comparison_unchanged():
    period_comparison = _period_comparison()
    payload = _payload(period_comparison=period_comparison)
    before = copy.deepcopy(period_comparison)

    build_chart_spec(
        _intent("period_change_chart", from_period_hint="2024-01", to_period_hint="2024-02"), payload, None,
    )

    assert period_comparison == before


# ==================================================
# MISSING VALUES CHART
# ==================================================


def test_missing_values_chart_success_with_profile():
    profile = _profile(with_missing=True)
    payload = _payload(profile=profile)

    result = build_chart_spec(_intent("missing_values_chart"), payload, None)

    assert result["found"] is True
    assert result["chart_type"] == "missing_values"
    assert result["data"] == profile


def test_missing_values_chart_profile_missing():
    payload = _payload(profile=None)

    result = build_chart_spec(_intent("missing_values_chart"), payload, None)

    assert result["found"] is False
    assert result["reason"] == "profile_not_available"


def test_missing_values_chart_preserves_full_profile():
    profile = _profile(with_missing=True)
    payload = _payload(profile=profile)

    result = build_chart_spec(_intent("missing_values_chart"), payload, None)

    assert len(result["data"]["columns"]) == len(profile["columns"])


def test_missing_values_chart_column_hint_does_not_silently_filter_data():
    profile = _profile(with_missing=True)
    payload = _payload(profile=profile)

    result = build_chart_spec(
        _intent("missing_values_chart", column_hint="Region"), payload, None,
    )

    assert result["found"] is True
    assert result["column"] is None
    assert len(result["data"]["columns"]) == len(profile["columns"])


# ==================================================
# RESOLUTION REUSE
# ==================================================


def test_resolve_column_from_qa_engine_is_reused_not_duplicated(monkeypatch):
    calls = []
    original = qa_engine.resolve_column

    def spy(hint, names):
        calls.append((hint, list(names)))
        return original(hint, names)

    monkeypatch.setattr("src.qa_chart_engine.resolve_column", spy)

    build_chart_spec(_intent("trend_chart", column_hint="TotalPrice"), _payload(), _monthly_series())

    assert calls == [("TotalPrice", ["TotalPrice", "Revenue"])]


def test_resolve_period_from_qa_engine_is_reused_not_duplicated(monkeypatch):
    calls = []
    original = qa_engine.resolve_period

    def spy(hint, labels):
        calls.append((hint, list(labels)))
        return original(hint, labels)

    monkeypatch.setattr("src.qa_chart_engine.resolve_period", spy)

    build_chart_spec(
        _intent("period_change_chart", from_period_hint="2024-01", to_period_hint="2024-02"), _payload(), None,
    )

    assert calls == [
        ("2024-01", ["2024-01", "2024-02", "2024-03"]),
        ("2024-02", ["2024-01", "2024-02", "2024-03"]),
    ]
