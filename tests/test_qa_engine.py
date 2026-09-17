import copy
import json

import pandas as pd

from src.qa_engine import dispatch_intent, resolve_column, resolve_period

# ==================================================
# COLUMN RESOLUTION
# ==================================================


def test_resolve_column_exact_match():
    result = resolve_column("Revenue", ["Revenue", "Region"])

    assert result == {"found": True, "reason": None, "column": "Revenue", "candidates": []}


def test_resolve_column_case_insensitive_exact_match():
    result = resolve_column("revenue", ["Revenue", "Region"])

    assert result["found"] is True
    assert result["reason"] is None
    assert result["column"] == "Revenue"
    assert result["candidates"] == []


def test_resolve_column_duplicate_exact_matches_is_ambiguous():
    result = resolve_column("revenue", ["Revenue", "REVENUE"])

    assert result["found"] is False
    assert result["reason"] == "ambiguous_column"
    assert result["column"] is None
    assert set(result["candidates"]) == {"Revenue", "REVENUE"}


def test_resolve_column_obvious_fuzzy_single_match_resolves():
    # "Revenu" is a one-character-short typo of "Revenue" — close enough to
    # resolve automatically, with no other column close enough to compete.
    result = resolve_column("Revenu", ["Revenue", "Region"])

    assert result["found"] is True
    assert result["reason"] is None
    assert result["column"] == "Revenue"
    assert result["candidates"] == []


def test_resolve_column_weak_fuzzy_reference_does_not_resolve():
    # "Rev" is a short, weak partial reference — below the conservative
    # cutoff, so it must not be silently resolved to "Revenue".
    result = resolve_column("Rev", ["Revenue", "Region"])

    assert result["found"] is False
    assert result["reason"] == "column_not_found"
    assert result["column"] is None
    assert result["candidates"] == []


def test_resolve_column_multiple_plausible_fuzzy_candidates_is_ambiguous():
    result = resolve_column("Revenue1", ["RevenueA", "RevenueB"])

    assert result["found"] is False
    assert result["reason"] == "ambiguous_column"
    assert result["column"] is None
    assert set(result["candidates"]) == {"RevenueA", "RevenueB"}


def test_resolve_column_no_fuzzy_candidates_is_not_found():
    result = resolve_column("xyz123", ["Revenue", "Region"])

    assert result == {"found": False, "reason": "column_not_found", "column": None, "candidates": []}


def test_resolve_column_empty_column_list():
    result = resolve_column("Revenue", [])

    assert result == {"found": False, "reason": "column_not_found", "column": None, "candidates": []}


def test_resolve_column_does_not_mutate_input_list():
    columns = ["Revenue", "Region"]
    before = list(columns)

    resolve_column("Revenue", columns)

    assert columns == before


# ==================================================
# PERIOD RESOLUTION
# ==================================================


def test_resolve_period_exact_yyyy_mm():
    result = resolve_period("2024-03", ["2024-01", "2024-02", "2024-03"])

    assert result == {"found": True, "reason": None, "period": "2024-03", "candidates": []}


def test_resolve_period_bare_month_with_exactly_one_year():
    result = resolve_period("March", ["2024-01", "2024-02", "2024-03"])

    assert result["found"] is True
    assert result["reason"] is None
    assert result["period"] == "2024-03"
    assert result["candidates"] == []


def test_resolve_period_bare_month_across_multiple_years_is_ambiguous():
    result = resolve_period("March", ["2023-03", "2024-03", "2024-05"])

    assert result["found"] is False
    assert result["reason"] == "ambiguous_period"
    assert result["period"] is None
    assert result["candidates"] == ["2023-03", "2024-03"]


def test_resolve_period_bare_month_with_no_match():
    result = resolve_period("December", ["2024-01", "2024-02", "2024-03"])

    assert result == {"found": False, "reason": "period_not_found", "period": None, "candidates": []}


def test_resolve_period_month_name_and_year():
    result = resolve_period("March 2024", ["2024-01", "2024-02", "2024-03"])

    assert result["found"] is True
    assert result["period"] == "2024-03"


def test_resolve_period_abbreviated_month_name_and_year():
    result = resolve_period("Mar 2024", ["2024-01", "2024-02", "2024-03"])

    assert result["found"] is True
    assert result["period"] == "2024-03"


def test_resolve_period_numeric_month_and_year():
    result = resolve_period("03 2024", ["2024-01", "2024-02", "2024-03"])

    assert result["found"] is True
    assert result["period"] == "2024-03"


def test_resolve_period_month_and_year_not_present_is_not_found():
    result = resolve_period("March 2025", ["2024-01", "2024-02", "2024-03"])

    assert result == {"found": False, "reason": "period_not_found", "period": None, "candidates": []}


def test_resolve_period_ambiguity_never_defaults_to_latest_or_first_year():
    result = resolve_period("March", ["2022-03", "2023-03", "2024-03"])

    assert result["found"] is False
    assert result["period"] is None
    # All candidate years must be surfaced, not silently narrowed to one.
    assert result["candidates"] == ["2022-03", "2023-03", "2024-03"]


def test_resolve_period_does_not_mutate_input_list():
    periods = ["2024-01", "2024-02", "2024-03"]
    before = list(periods)

    resolve_period("March", periods)

    assert periods == before


def test_resolve_period_empty_period_list():
    result = resolve_period("March", [])

    assert result == {"found": False, "reason": "period_not_found", "period": None, "candidates": []}


# ==================================================
# INTENT DISPATCH (V0.5.3b)
# ==================================================


def _intent(intent_name, metric=None, column_hint=None, period_hint=None, from_period_hint=None, to_period_hint=None):
    return {
        "intent": intent_name,
        "metric": metric,
        "column_hint": column_hint,
        "period_hint": period_hint,
        "from_period_hint": from_period_hint,
        "to_period_hint": to_period_hint,
    }


def _numeric_summary():
    return {
        "columns": [
            {
                "name": "TotalPrice",
                "count": 3,
                "missing_count": 0,
                "sum": 450.0,
                "mean": 150.0,
                "median": 150.0,
                "min": 100.0,
                "max": 200.0,
                "std": 50.0,
            },
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
            "method": "iqr",
            "insufficient_data": False,
            "sample_size": 3,
            "lower_bound": 10.0,
            "upper_bound": 190.0,
            "anomaly_count": 1,
            "anomalies": [{"position": 1, "period": "2024-02", "value": 200.0, "direction": "high"}],
        }
    return {
        "method": "iqr",
        "insufficient_data": False,
        "sample_size": 3,
        "lower_bound": 10.0,
        "upper_bound": 300.0,
        "anomaly_count": 0,
        "anomalies": [],
    }


def _profile(with_missing=True):
    region_missing = 2 if with_missing else 0
    return {
        "row_count": 10,
        "column_count": 2,
        "is_empty": False,
        "duplicate_row_count": 0,
        "duplicate_column_names": [],
        "columns": [
            {"name": "TotalPrice", "dtype": "float64", "inferred_type": "numeric", "unique_count": 3, "missing_count": 0, "missing_percentage": 0.0},
            {"name": "Region", "dtype": "object", "inferred_type": "text", "unique_count": 3, "missing_count": region_missing, "missing_percentage": region_missing * 10.0},
        ],
    }


def _payload(**overrides):
    payload = {
        "numeric_summary": _numeric_summary(),
        "period_comparison": _period_comparison(),
        "anomalies": _anomalies(),
        "profile": _profile(),
    }
    payload.update(overrides)
    return payload


# --- period_value ---


def test_dispatch_period_value_success():
    result = dispatch_intent(
        _intent("period_value", column_hint="TotalPrice", period_hint="2024-03"),
        _payload(),
        _monthly_series(),
    )

    assert result["found"] is True
    assert result["reason"] is None
    assert result["intent"] == "period_value"
    assert result["column"] == "TotalPrice"
    assert result["period"] == "2024-03"
    assert result["value"] == 150.0


def test_dispatch_period_value_column_not_found():
    result = dispatch_intent(
        _intent("period_value", column_hint="Nonexistent", period_hint="2024-03"),
        _payload(),
        _monthly_series(),
    )

    assert result["found"] is False
    assert result["reason"] == "column_not_found"
    assert result["column"] is None
    assert result["period"] is None
    assert result["value"] is None


def test_dispatch_period_value_ambiguous_column():
    payload = _payload(numeric_summary={"columns": [{"name": "Revenue", **{k: 1.0 for k in ("sum", "mean", "median", "min", "max")}, "count": 1, "missing_count": 0, "std": 0.0}, {"name": "REVENUE", **{k: 1.0 for k in ("sum", "mean", "median", "min", "max")}, "count": 1, "missing_count": 0, "std": 0.0}]})

    result = dispatch_intent(
        _intent("period_value", column_hint="revenue", period_hint="2024-03"),
        payload,
        _monthly_series(),
    )

    assert result["found"] is False
    assert result["reason"] == "ambiguous_column"
    assert set(result["extra"]["candidates"]) == {"Revenue", "REVENUE"}


def test_dispatch_period_value_period_not_found():
    result = dispatch_intent(
        _intent("period_value", column_hint="TotalPrice", period_hint="December"),
        _payload(),
        _monthly_series(),
    )

    assert result["found"] is False
    assert result["reason"] == "period_not_found"
    assert result["period"] is None


def test_dispatch_period_value_ambiguous_period():
    series = pd.Series([1.0, 2.0], index=["2023-03", "2024-03"], name="TotalPrice")

    result = dispatch_intent(
        _intent("period_value", column_hint="TotalPrice", period_hint="March"),
        _payload(),
        series,
    )

    assert result["found"] is False
    assert result["reason"] == "ambiguous_period"
    assert result["extra"]["candidates"] == ["2023-03", "2024-03"]


# --- column_hint vs. monthly_series safety (column_series_mismatch) ---


def _two_column_numeric_summary():
    return {
        "columns": [
            {"name": "TotalPrice", "count": 3, "missing_count": 0, "sum": 450.0, "mean": 150.0, "median": 150.0, "min": 100.0, "max": 200.0, "std": 50.0},
            {"name": "Revenue", "count": 3, "missing_count": 0, "sum": 900.0, "mean": 300.0, "median": 300.0, "min": 200.0, "max": 400.0, "std": 100.0},
        ],
    }


def test_dispatch_period_value_matching_column_hint_and_series_name_succeeds():
    payload = _payload(numeric_summary=_two_column_numeric_summary())

    result = dispatch_intent(
        _intent("period_value", column_hint="TotalPrice", period_hint="2024-03"),
        payload,
        _monthly_series(),  # name="TotalPrice"
    )

    assert result["found"] is True
    assert result["reason"] is None
    assert result["column"] == "TotalPrice"
    assert result["value"] == 150.0


def test_dispatch_period_value_column_hint_mismatched_with_series_name_fails():
    payload = _payload(numeric_summary=_two_column_numeric_summary())

    result = dispatch_intent(
        _intent("period_value", column_hint="Revenue", period_hint="2024-03"),
        payload,
        _monthly_series(),  # name="TotalPrice" — hint resolves fine but to the wrong series
    )

    assert result["found"] is False
    assert result["reason"] == "column_series_mismatch"
    assert result["column"] is None
    assert result["period"] is None
    assert result["value"] is None


def test_dispatch_anomaly_check_column_hint_mismatched_with_series_name_fails():
    payload = _payload(numeric_summary=_two_column_numeric_summary())

    result = dispatch_intent(
        _intent("anomaly_check", column_hint="Revenue"),
        payload,
        _monthly_series(),  # name="TotalPrice"
    )

    assert result["found"] is False
    assert result["reason"] == "column_series_mismatch"
    assert result["value"] is None


def test_dispatch_period_value_no_column_hint_behavior_is_unchanged():
    # No column_hint at all: falls back to the series' own name, exactly as
    # before this safety fix — the mismatch check never applies here.
    payload = _payload(numeric_summary=_two_column_numeric_summary())

    result = dispatch_intent(
        _intent("period_value", period_hint="2024-03"),
        payload,
        _monthly_series(),  # name="TotalPrice"
    )

    assert result["found"] is True
    assert result["reason"] is None
    assert result["column"] == "TotalPrice"
    assert result["value"] == 150.0


def test_dispatch_period_value_mismatch_result_is_json_serializable():
    payload = _payload(numeric_summary=_two_column_numeric_summary())

    result = dispatch_intent(
        _intent("period_value", column_hint="Revenue", period_hint="2024-03"),
        payload,
        _monthly_series(),
    )

    json.dumps(result)


def test_dispatch_intent_mismatch_check_does_not_mutate_inputs():
    intent = _intent("period_value", column_hint="Revenue", period_hint="2024-03")
    payload = _payload(numeric_summary=_two_column_numeric_summary())
    series = _monthly_series()

    intent_before = copy.deepcopy(intent)
    payload_before = copy.deepcopy(payload)
    series_before = series.copy()

    dispatch_intent(intent, payload, series)

    assert intent == intent_before
    assert payload == payload_before
    assert series.equals(series_before)


# --- period_extremum ---


def test_dispatch_period_extremum_min():
    result = dispatch_intent(
        _intent("period_extremum", metric="min"),
        _payload(),
        _monthly_series(),
    )

    assert result["found"] is True
    assert result["intent"] == "period_extremum"
    assert result["period"] == "2024-01"
    assert result["value"] == 100.0


def test_dispatch_period_extremum_max():
    result = dispatch_intent(
        _intent("period_extremum", metric="max"),
        _payload(),
        _monthly_series(),
    )

    assert result["found"] is True
    assert result["period"] == "2024-02"
    assert result["value"] == 200.0


def test_dispatch_period_extremum_trend_not_computed():
    result = dispatch_intent(
        _intent("period_extremum", metric="max"),
        _payload(),
        None,
    )

    assert result["found"] is False
    assert result["reason"] == "trend_not_computed"


# --- period_change ---


def test_dispatch_period_change_success():
    result = dispatch_intent(
        _intent("period_change", from_period_hint="2024-01", to_period_hint="2024-02"),
        _payload(),
        _monthly_series(),
    )

    assert result["found"] is True
    assert result["reason"] is None
    assert result["intent"] == "period_change"
    assert result["value"] == 100.0
    assert result["extra"]["previous"] == 100.0
    assert result["extra"]["current"] == 200.0
    assert result["extra"]["absolute_change"] == 100.0
    assert result["extra"]["percentage_change"] == 100.0
    assert result["extra"]["is_valid"] is True
    assert result["extra"]["from_period"] == "2024-01"
    assert result["extra"]["to_period"] == "2024-02"


def test_dispatch_period_change_missing_transition():
    result = dispatch_intent(
        _intent("period_change", from_period_hint="2024-01", to_period_hint="2024-03"),
        _payload(),
        _monthly_series(),
    )

    assert result["found"] is False
    assert result["reason"] == "period_not_found"
    assert result["value"] is None


# --- column_stat ---


def test_dispatch_column_stat_success_for_each_metric():
    expected = {"sum": 450.0, "mean": 150.0, "median": 150.0, "min": 100.0, "max": 200.0, "count": 3}

    for metric, expected_value in expected.items():
        result = dispatch_intent(
            _intent("column_stat", metric=metric, column_hint="TotalPrice"),
            _payload(),
            None,
        )

        assert result["found"] is True
        assert result["reason"] is None
        assert result["column"] == "TotalPrice"
        assert result["value"] == expected_value


def test_dispatch_column_stat_column_not_found():
    result = dispatch_intent(
        _intent("column_stat", metric="sum", column_hint="Nonexistent"),
        _payload(),
        None,
    )

    assert result["found"] is False
    assert result["reason"] == "column_not_found"
    assert result["value"] is None


# --- anomaly_check ---


def test_dispatch_anomaly_check_with_anomalies():
    result = dispatch_intent(
        _intent("anomaly_check"),
        _payload(anomalies=_anomalies(with_anomaly=True)),
        _monthly_series(),
    )

    assert result["found"] is True
    assert result["reason"] is None
    assert result["value"] == 1
    assert len(result["extra"]["anomalies"]) == 1


def test_dispatch_anomaly_check_with_no_anomalies():
    result = dispatch_intent(
        _intent("anomaly_check"),
        _payload(anomalies=_anomalies(with_anomaly=False)),
        _monthly_series(),
    )

    assert result["found"] is True
    assert result["reason"] is None
    assert result["value"] == 0
    assert result["extra"]["anomalies"] == []


def test_dispatch_anomaly_check_missing_payload():
    result = dispatch_intent(
        _intent("anomaly_check"),
        _payload(anomalies=None),
        _monthly_series(),
    )

    assert result["found"] is False
    assert result["reason"] == "trend_not_computed"
    assert result["value"] is None


# --- missing_values ---


def test_dispatch_missing_values_with_missing_columns():
    result = dispatch_intent(
        _intent("missing_values"),
        _payload(profile=_profile(with_missing=True)),
        None,
    )

    assert result["found"] is True
    assert result["reason"] is None
    assert result["value"] == 1
    assert result["extra"]["columns"] == [{"name": "Region", "missing_count": 2, "missing_percentage": 20.0}]


def test_dispatch_missing_values_with_no_missing_columns():
    result = dispatch_intent(
        _intent("missing_values"),
        _payload(profile=_profile(with_missing=False)),
        None,
    )

    assert result["found"] is True
    assert result["reason"] is None
    assert result["value"] == 0
    assert result["extra"]["columns"] == []


# --- unsupported ---


def test_dispatch_unsupported_intent():
    result = dispatch_intent(_intent("unsupported"), _payload(), _monthly_series())

    assert result == {
        "found": False,
        "reason": "unsupported",
        "intent": "unsupported",
        "column": None,
        "period": None,
        "value": None,
        "extra": {},
    }


# --- cross-cutting guarantees ---


def test_dispatch_intent_does_not_mutate_inputs():
    intent = _intent("period_value", column_hint="TotalPrice", period_hint="2024-03")
    payload = _payload()
    series = _monthly_series()

    intent_before = copy.deepcopy(intent)
    payload_before = copy.deepcopy(payload)
    series_before = series.copy()

    dispatch_intent(intent, payload, series)

    assert intent == intent_before
    assert payload == payload_before
    assert series.equals(series_before)


def test_dispatch_intent_results_are_json_serializable():
    cases = [
        (_intent("period_value", column_hint="TotalPrice", period_hint="2024-03"), _payload(), _monthly_series()),
        (_intent("period_change", from_period_hint="2024-01", to_period_hint="2024-02"), _payload(), _monthly_series()),
        (_intent("anomaly_check"), _payload(), _monthly_series()),
        (_intent("missing_values"), _payload(), None),
    ]

    for intent, payload, series in cases:
        result = dispatch_intent(intent, payload, series)
        json.dumps(result)


def test_resolve_column_and_resolve_period_still_behave_as_documented():
    # Regression guard: dispatch_intent must reuse the existing resolvers
    # unchanged, not reimplement resolution logic.
    column_result = resolve_column("Revenu", ["Revenue", "Region"])
    period_result = resolve_period("March 2024", ["2024-01", "2024-02", "2024-03"])

    assert column_result["found"] is True and column_result["column"] == "Revenue"
    assert period_result["found"] is True and period_result["period"] == "2024-03"
