import copy
import json

import pandas as pd

import src.analytics_engine as analytics_engine
from src.multi_file_comparison import (
    compare_files,
    dispatch_comparison_intent,
    resolve_file,
)

# ==================================================
# Fixtures — plain dicts matching DataFile / FileAnalysis
# ==================================================


def _numeric_summary(name="TotalPrice", **overrides):
    entry = {
        "name": name, "count": 3, "missing_count": 0, "sum": 100.0,
        "mean": 33.33, "median": 30.0, "min": 20.0, "max": 50.0, "std": 10.0,
    }
    entry.update(overrides)
    return {"columns": [entry]}


def _monthly_series(values=(100.0, 200.0, 150.0), index=("2024-01", "2024-02", "2024-03"), name="TotalPrice"):
    return pd.Series(list(values), index=list(index), name=name)


def _trend(label="increasing"):
    return {
        "trend": label, "valid_comparison_count": 2, "invalid_comparison_count": 0,
        "direction_comparison_count": 2, "increase_count": 2, "decrease_count": 0, "no_change_count": 0,
    }


def _anomalies(count=1):
    return {
        "method": "iqr", "insufficient_data": False, "sample_size": 3,
        "lower_bound": 10.0, "upper_bound": 190.0, "anomaly_count": count,
        "anomalies": [{"position": 1, "period": "2024-02", "value": 200.0, "direction": "high"}] if count else [],
    }


_DEFAULT = object()


def _file_analysis(numeric_summary=None, monthly_series=_DEFAULT, trend=_DEFAULT, anomalies=_DEFAULT):
    return {
        "profile": {"row_count": 3, "column_count": 2},
        "normalized_df": None,
        "numeric_summary": numeric_summary if numeric_summary is not None else _numeric_summary(),
        "date_summary": {"columns": []},
        "monthly_series": _monthly_series() if monthly_series is _DEFAULT else monthly_series,
        "trend": _trend() if trend is _DEFAULT else trend,
        "period_comparison": None,
        "anomalies": _anomalies() if anomalies is _DEFAULT else anomalies,
    }


def _data_file(file_id, display_name, role=None, analysis="default", sum_value=100.0):
    return {
        "file_id": file_id,
        "filename": f"{display_name}.xlsx",
        "display_name": display_name,
        "role": role,
        "raw_df": None,
        "value_column": (1, "TotalPrice"),
        "period_column": (0, "PurchaseDate"),
        "analysis": _file_analysis(numeric_summary=_numeric_summary(sum=sum_value)) if analysis == "default" else analysis,
    }


def _two_files(role_a="previous", role_b="current", sum_a=100.0, sum_b=130.0):
    file_a = _data_file("file-a-id", "January", role=role_a, sum_value=sum_a)
    file_b = _data_file("file-b-id", "February", role=role_b, sum_value=sum_b)
    return {file_a["file_id"]: file_a, file_b["file_id"]: file_b}


def _intent(intent_name, column_hint=None, metric=None, period_hint=None,
            from_file_hint=None, to_file_hint=None):
    return {
        "intent": intent_name, "column_hint": column_hint, "metric": metric,
        "period_hint": period_hint, "from_file_hint": from_file_hint, "to_file_hint": to_file_hint,
    }


# ==================================================
# resolve_file
# ==================================================


def test_resolve_file_exact_role_match():
    data_files = _two_files()

    result = resolve_file("previous", data_files)

    assert result["found"] is True
    assert result["file_id"] == "file-a-id"


def test_resolve_file_exact_display_name_match():
    data_files = _two_files()

    result = resolve_file("February", data_files)

    assert result["found"] is True
    assert result["file_id"] == "file-b-id"


def test_resolve_file_case_insensitive_exact_match():
    data_files = _two_files()

    result = resolve_file("PREVIOUS", data_files)

    assert result["found"] is True
    assert result["file_id"] == "file-a-id"


def test_resolve_file_no_fuzzy_matching():
    data_files = _two_files()

    # "Januar" is a one-character-short typo of "January" — resolve_column
    # would resolve this fuzzily, resolve_file must not.
    result = resolve_file("Januar", data_files)

    assert result["found"] is False
    assert result["reason"] == "file_not_found"


def test_resolve_file_role_not_assigned():
    data_files = {"f1": _data_file("f1", "January", role=None)}

    result = resolve_file("previous", data_files)

    assert result["found"] is False
    assert result["reason"] == "role_not_assigned"


def test_resolve_file_duplicate_role():
    data_files = {
        "f1": _data_file("f1", "January", role="current"),
        "f2": _data_file("f2", "February", role="current"),
    }

    result = resolve_file("current", data_files)

    assert result["found"] is False
    assert result["reason"] == "duplicate_role"
    assert set(result["candidates"]) == {"f1", "f2"}


def test_resolve_file_not_found():
    data_files = _two_files()

    result = resolve_file("Nonexistent File", data_files)

    assert result["found"] is False
    assert result["reason"] == "file_not_found"


def test_resolve_file_empty_hint_is_not_found():
    result = resolve_file("", _two_files())

    assert result["found"] is False
    assert result["reason"] == "file_not_found"


def test_resolve_file_never_infers_from_upload_order():
    # Neither file has a role or a display_name matching "previous"/"first";
    # resolve_file must not silently pick the first-inserted entry.
    data_files = {
        "f1": _data_file("f1", "Alpha", role=None),
        "f2": _data_file("f2", "Beta", role=None),
    }

    result = resolve_file("previous", data_files)

    assert result["found"] is False
    assert result["reason"] == "role_not_assigned"


# ==================================================
# compare_files — file_value_comparison
# ==================================================


def test_compare_files_value_comparison_success():
    file_a = _file_analysis(numeric_summary=_numeric_summary(sum=100.0))
    file_b = _file_analysis(numeric_summary=_numeric_summary(sum=130.0))

    result = compare_files("file_value_comparison", file_a, file_b, column_hint="TotalPrice", metric="sum")

    assert result["found"] is True
    assert result["reason"] is None
    assert result["column"] == "TotalPrice"
    assert result["metric"] == "sum"
    assert result["previous"] == 100.0
    assert result["current"] == 130.0
    assert result["absolute_change"] == 30.0
    assert result["percentage_change"] == 30.0
    assert result["is_valid"] is True


def test_compare_files_value_comparison_column_not_found_in_file_a():
    file_a = _file_analysis(numeric_summary=_numeric_summary(name="Revenue"))
    file_b = _file_analysis(numeric_summary=_numeric_summary(name="TotalPrice"))

    result = compare_files("file_value_comparison", file_a, file_b, column_hint="TotalPrice", metric="sum")

    assert result["found"] is False
    assert result["reason"] == "column_not_found_in_file_a"


def test_compare_files_value_comparison_column_not_found_in_file_b():
    file_a = _file_analysis(numeric_summary=_numeric_summary(name="TotalPrice"))
    file_b = _file_analysis(numeric_summary=_numeric_summary(name="Revenue"))

    result = compare_files("file_value_comparison", file_a, file_b, column_hint="TotalPrice", metric="sum")

    assert result["found"] is False
    assert result["reason"] == "column_not_found_in_file_b"


def test_compare_files_value_comparison_ambiguous_column_in_file_a():
    file_a = _file_analysis(numeric_summary={"columns": [
        {"name": "Revenue", "sum": 1.0}, {"name": "REVENUE", "sum": 1.0},
    ]})
    file_b = _file_analysis()

    result = compare_files("file_value_comparison", file_a, file_b, column_hint="revenue", metric="sum")

    assert result["found"] is False
    assert result["reason"] == "ambiguous_column_in_file_a"
    assert set(result["extra"]["candidates"]) == {"Revenue", "REVENUE"}


def test_compare_files_value_comparison_ambiguous_column_in_file_b():
    file_a = _file_analysis(numeric_summary=_numeric_summary(name="Revenue"))
    file_b = _file_analysis(numeric_summary={"columns": [
        {"name": "Revenue", "sum": 1.0}, {"name": "REVENUE", "sum": 1.0},
    ]})

    result = compare_files("file_value_comparison", file_a, file_b, column_hint="revenue", metric="sum")

    assert result["found"] is False
    assert result["reason"] == "ambiguous_column_in_file_b"


def test_compare_files_value_comparison_reuses_compare_values(monkeypatch):
    calls = []
    original = analytics_engine.compare_values

    def spy(previous, current):
        calls.append((previous, current))
        return original(previous, current)

    monkeypatch.setattr("src.multi_file_comparison.compare_values", spy)

    file_a = _file_analysis(numeric_summary=_numeric_summary(sum=100.0))
    file_b = _file_analysis(numeric_summary=_numeric_summary(sum=130.0))

    compare_files("file_value_comparison", file_a, file_b, column_hint="TotalPrice", metric="sum")

    assert calls == [(100.0, 130.0)]


# ==================================================
# compare_files — file_period_comparison
# ==================================================


def test_compare_files_period_comparison_success():
    file_a = _file_analysis(monthly_series=_monthly_series((100.0, 200.0), ("2024-01", "2024-02")))
    file_b = _file_analysis(monthly_series=_monthly_series((110.0, 220.0), ("2024-01", "2024-02")))

    result = compare_files("file_period_comparison", file_a, file_b, column_hint="TotalPrice", period_hint="January")

    assert result["found"] is True
    assert result["period"] == "2024-01"
    assert result["previous"] == 100.0
    assert result["current"] == 110.0
    assert result["absolute_change"] == 10.0


def test_compare_files_period_comparison_period_not_found_in_file_a():
    file_a = _file_analysis(monthly_series=_monthly_series((100.0,), ("2024-01",)))
    file_b = _file_analysis(monthly_series=_monthly_series((110.0,), ("2024-02",)))

    result = compare_files("file_period_comparison", file_a, file_b, period_hint="February")

    assert result["found"] is False
    assert result["reason"] == "period_not_found_in_file_a"


def test_compare_files_period_comparison_period_not_found_in_file_b():
    file_a = _file_analysis(monthly_series=_monthly_series((100.0,), ("2024-02",)))
    file_b = _file_analysis(monthly_series=_monthly_series((110.0,), ("2024-01",)))

    result = compare_files("file_period_comparison", file_a, file_b, period_hint="February")

    assert result["found"] is False
    assert result["reason"] == "period_not_found_in_file_b"


def test_compare_files_period_comparison_ambiguous_period_in_file_a():
    file_a = _file_analysis(monthly_series=_monthly_series((1.0, 2.0), ("2023-03", "2024-03")))
    file_b = _file_analysis(monthly_series=_monthly_series((1.0,), ("2024-03",)))

    result = compare_files("file_period_comparison", file_a, file_b, period_hint="March")

    assert result["found"] is False
    assert result["reason"] == "ambiguous_period_in_file_a"
    assert result["extra"]["candidates"] == ["2023-03", "2024-03"]


def test_compare_files_period_comparison_ambiguous_period_in_file_b():
    file_a = _file_analysis(monthly_series=_monthly_series((1.0,), ("2024-03",)))
    file_b = _file_analysis(monthly_series=_monthly_series((1.0, 2.0), ("2023-03", "2024-03")))

    result = compare_files("file_period_comparison", file_a, file_b, period_hint="March")

    assert result["found"] is False
    assert result["reason"] == "ambiguous_period_in_file_b"


def test_compare_files_period_comparison_trend_not_computed_in_file_a():
    file_a = _file_analysis(monthly_series=None)
    file_b = _file_analysis()

    result = compare_files("file_period_comparison", file_a, file_b, period_hint="January")

    assert result["found"] is False
    assert result["reason"] == "trend_not_computed_in_file_a"


def test_compare_files_period_comparison_trend_not_computed_in_file_b():
    file_a = _file_analysis()
    file_b = _file_analysis(monthly_series=None)

    result = compare_files("file_period_comparison", file_a, file_b, period_hint="January")

    assert result["found"] is False
    assert result["reason"] == "trend_not_computed_in_file_b"


def test_compare_files_period_comparison_column_series_mismatch():
    numeric_summary = {"columns": [
        {"name": "TotalPrice", "sum": 1.0}, {"name": "Revenue", "sum": 1.0},
    ]}
    file_a = _file_analysis(
        numeric_summary=numeric_summary,
        monthly_series=_monthly_series(name="TotalPrice"),
    )
    file_b = _file_analysis(numeric_summary=numeric_summary, monthly_series=_monthly_series(name="TotalPrice"))

    result = compare_files("file_period_comparison", file_a, file_b, column_hint="Revenue", period_hint="January")

    assert result["found"] is False
    assert result["reason"] == "column_series_mismatch"


def test_compare_files_period_comparison_reuses_compare_values(monkeypatch):
    calls = []
    original = analytics_engine.compare_values

    def spy(previous, current):
        calls.append((previous, current))
        return original(previous, current)

    monkeypatch.setattr("src.multi_file_comparison.compare_values", spy)

    file_a = _file_analysis(monthly_series=_monthly_series((100.0,), ("2024-01",)))
    file_b = _file_analysis(monthly_series=_monthly_series((150.0,), ("2024-01",)))

    compare_files("file_period_comparison", file_a, file_b, period_hint="January")

    assert calls == [(100.0, 150.0)]


# ==================================================
# compare_files — file_trend_comparison
# ==================================================


def test_compare_files_trend_comparison_success():
    file_a = _file_analysis(trend=_trend("increasing"))
    file_b = _file_analysis(trend=_trend("decreasing"))

    result = compare_files("file_trend_comparison", file_a, file_b)

    assert result["found"] is True
    assert result["previous"] == "increasing"
    assert result["current"] == "decreasing"
    assert result["absolute_change"] is None
    assert result["is_valid"] is True
    assert result["extra"]["file_a_trend"]["trend"] == "increasing"
    assert result["extra"]["file_b_trend"]["trend"] == "decreasing"


def test_compare_files_trend_comparison_trend_not_computed_in_file_a():
    file_a = _file_analysis(trend=None)
    file_b = _file_analysis()

    result = compare_files("file_trend_comparison", file_a, file_b)

    assert result["found"] is False
    assert result["reason"] == "trend_not_computed_in_file_a"


def test_compare_files_trend_comparison_trend_not_computed_in_file_b():
    file_a = _file_analysis()
    file_b = _file_analysis(trend=None)

    result = compare_files("file_trend_comparison", file_a, file_b)

    assert result["found"] is False
    assert result["reason"] == "trend_not_computed_in_file_b"


def test_compare_files_trend_comparison_does_not_recalculate_trend():
    from src.analytics_engine import detect_trend

    called = []
    original_detect_trend = detect_trend

    def spy(*args, **kwargs):
        called.append(1)
        return original_detect_trend(*args, **kwargs)

    import src.multi_file_comparison as module
    assert not hasattr(module, "detect_trend")  # trend detection is not even imported here

    file_a = _file_analysis(trend=_trend("increasing"))
    file_b = _file_analysis(trend=_trend("decreasing"))
    compare_files("file_trend_comparison", file_a, file_b)

    assert called == []


# ==================================================
# compare_files — file_anomaly_comparison
# ==================================================


def test_compare_files_anomaly_comparison_success():
    file_a = _file_analysis(anomalies=_anomalies(count=1))
    file_b = _file_analysis(anomalies=_anomalies(count=3))

    result = compare_files("file_anomaly_comparison", file_a, file_b)

    assert result["found"] is True
    assert result["previous"] == 1
    assert result["current"] == 3
    assert result["absolute_change"] == 2
    assert result["extra"]["file_a_anomalies"]["anomaly_count"] == 1
    assert result["extra"]["file_b_anomalies"]["anomaly_count"] == 3


def test_compare_files_anomaly_comparison_trend_not_computed_in_file_a():
    file_a = _file_analysis(anomalies=None)
    file_b = _file_analysis()

    result = compare_files("file_anomaly_comparison", file_a, file_b)

    assert result["found"] is False
    assert result["reason"] == "trend_not_computed_in_file_a"


def test_compare_files_anomaly_comparison_trend_not_computed_in_file_b():
    file_a = _file_analysis()
    file_b = _file_analysis(anomalies=None)

    result = compare_files("file_anomaly_comparison", file_a, file_b)

    assert result["found"] is False
    assert result["reason"] == "trend_not_computed_in_file_b"


def test_compare_files_anomaly_comparison_does_not_recalculate_anomalies():
    import src.multi_file_comparison as module
    assert not hasattr(module, "detect_iqr_anomalies")  # not even imported here

    file_a = _file_analysis(anomalies=_anomalies(count=1))
    file_b = _file_analysis(anomalies=_anomalies(count=1))
    result = compare_files("file_anomaly_comparison", file_a, file_b)

    assert result["found"] is True


# ==================================================
# compare_files — unsupported
# ==================================================


def test_compare_files_unsupported_comparison_type():
    result = compare_files("file_forecast_comparison", _file_analysis(), _file_analysis())

    assert result["found"] is False
    assert result["reason"] == "unsupported_comparison"


# ==================================================
# dispatch_comparison_intent — pairwise / files
# ==================================================


def test_dispatch_comparison_intent_success_with_full_contract_shape():
    data_files = _two_files(sum_a=100.0, sum_b=130.0)

    result = dispatch_comparison_intent(
        _intent("file_value_comparison", column_hint="TotalPrice", metric="sum",
                from_file_hint="previous", to_file_hint="current"),
        data_files,
    )

    assert result["found"] is True
    assert result["comparison_type"] == "file_value_comparison"
    assert result["file_a"] == {"file_id": "file-a-id", "display_name": "January", "role": "previous"}
    assert result["file_b"] == {"file_id": "file-b-id", "display_name": "February", "role": "current"}
    assert result["previous"] == 100.0
    assert result["current"] == 130.0
    assert set(result.keys()) == {
        "found", "reason", "comparison_type", "metric", "column", "period",
        "file_a", "file_b", "previous", "current", "absolute_change",
        "percentage_change", "is_valid", "extra",
    }


def test_dispatch_comparison_intent_insufficient_files():
    data_files = {"f1": _data_file("f1", "January", role="previous")}

    result = dispatch_comparison_intent(
        _intent("file_value_comparison", column_hint="TotalPrice", metric="sum",
                from_file_hint="previous", to_file_hint="current"),
        data_files,
    )

    assert result["found"] is False
    assert result["reason"] == "insufficient_files"


def test_dispatch_comparison_intent_file_not_found():
    data_files = _two_files()

    result = dispatch_comparison_intent(
        _intent("file_value_comparison", from_file_hint="Nonexistent", to_file_hint="current"),
        data_files,
    )

    assert result["found"] is False
    assert result["reason"] == "file_not_found"


def test_dispatch_comparison_intent_role_not_assigned():
    data_files = _two_files(role_a=None, role_b=None)

    result = dispatch_comparison_intent(
        _intent("file_value_comparison", from_file_hint="previous", to_file_hint="current"),
        data_files,
    )

    assert result["found"] is False
    assert result["reason"] == "role_not_assigned"


def test_dispatch_comparison_intent_duplicate_role():
    data_files = _two_files(role_a="current", role_b="current")

    result = dispatch_comparison_intent(
        _intent("file_value_comparison", from_file_hint="current", to_file_hint="current"),
        data_files,
    )

    assert result["found"] is False
    assert result["reason"] == "duplicate_role"


def test_dispatch_comparison_intent_unsupported_comparison():
    data_files = _two_files()

    result = dispatch_comparison_intent(
        _intent("file_forecast_comparison", from_file_hint="previous", to_file_hint="current"),
        data_files,
    )

    assert result["found"] is False
    assert result["reason"] == "unsupported_comparison"


def test_dispatch_comparison_intent_rejects_same_file_resolved_twice():
    data_files = _two_files()

    result = dispatch_comparison_intent(
        _intent("file_value_comparison", from_file_hint="previous", to_file_hint="January"),
        data_files,
    )

    assert result["found"] is False
    assert result["reason"] == "insufficient_files"


def test_dispatch_comparison_intent_never_selects_a_third_file_from_a_larger_set():
    data_files = _two_files()
    data_files["f3"] = _data_file("f3", "March", role="reference", sum_value=999.0)

    result = dispatch_comparison_intent(
        _intent("file_value_comparison", column_hint="TotalPrice", metric="sum",
                from_file_hint="previous", to_file_hint="current"),
        data_files,
    )

    assert result["found"] is True
    assert result["file_a"]["file_id"] == "file-a-id"
    assert result["file_b"]["file_id"] == "file-b-id"
    # The third file's value (999.0) must never leak into the comparison.
    assert result["current"] != 999.0


def test_dispatch_comparison_intent_pairwise_only_no_from_to_third_field_exists():
    # The structured intent schema itself has only two file-reference
    # fields (from_file_hint/to_file_hint) — there is no way to even
    # request a 3+ file comparison through this contract.
    intent = _intent("file_value_comparison", from_file_hint="previous", to_file_hint="current")
    assert set(intent.keys()) == {"intent", "column_hint", "metric", "period_hint", "from_file_hint", "to_file_hint"}


def test_dispatch_comparison_intent_column_not_found_uses_dispatch_flow():
    data_files = _two_files()

    result = dispatch_comparison_intent(
        _intent("file_value_comparison", column_hint="Nonexistent", metric="sum",
                from_file_hint="previous", to_file_hint="current"),
        data_files,
    )

    assert result["found"] is False
    assert result["reason"] == "column_not_found_in_file_a"
    assert result["file_a"]["file_id"] == "file-a-id"
    assert result["file_b"]["file_id"] == "file-b-id"


# ==================================================
# JSON safety
# ==================================================


def test_dispatch_comparison_intent_result_is_json_serializable():
    data_files = _two_files()

    result = dispatch_comparison_intent(
        _intent("file_value_comparison", column_hint="TotalPrice", metric="sum",
                from_file_hint="previous", to_file_hint="current"),
        data_files,
    )

    json.dumps(result)


def test_compare_files_period_comparison_result_has_no_numpy_or_nat_leakage():
    file_a = _file_analysis(monthly_series=_monthly_series((100.0,), ("2024-01",)))
    file_b = _file_analysis(monthly_series=_monthly_series((150.0,), ("2024-01",)))

    result = compare_files("file_period_comparison", file_a, file_b, period_hint="January")

    json.dumps(result)
    assert isinstance(result["previous"], float)
    assert isinstance(result["current"], float)


def test_compare_files_anomaly_comparison_result_is_json_serializable():
    file_a = _file_analysis(anomalies=_anomalies(count=1))
    file_b = _file_analysis(anomalies=_anomalies(count=2))

    result = compare_files("file_anomaly_comparison", file_a, file_b)

    json.dumps(result)


# ==================================================
# No mutation
# ==================================================


def test_dispatch_comparison_intent_does_not_mutate_data_files():
    data_files = _two_files()
    before = copy.deepcopy({
        fid: {k: v for k, v in df.items() if k not in ("raw_df", "analysis")}
        for fid, df in data_files.items()
    })
    analysis_before = {
        fid: copy.deepcopy({k: v for k, v in df["analysis"].items() if k != "monthly_series"})
        for fid, df in data_files.items()
    }
    series_before = {fid: df["analysis"]["monthly_series"].copy() for fid, df in data_files.items()}

    dispatch_comparison_intent(
        _intent("file_value_comparison", column_hint="TotalPrice", metric="sum",
                from_file_hint="previous", to_file_hint="current"),
        data_files,
    )

    after = {
        fid: {k: v for k, v in df.items() if k not in ("raw_df", "analysis")}
        for fid, df in data_files.items()
    }
    analysis_after = {
        fid: {k: v for k, v in df["analysis"].items() if k != "monthly_series"}
        for fid, df in data_files.items()
    }
    assert after == before
    assert analysis_after == analysis_before
    for fid, df in data_files.items():
        assert df["analysis"]["monthly_series"].equals(series_before[fid])


def test_compare_files_does_not_mutate_file_analysis_dicts():
    file_a = _file_analysis()
    file_b = _file_analysis()
    file_a_before = copy.deepcopy({k: v for k, v in file_a.items() if k != "monthly_series"})
    file_b_before = copy.deepcopy({k: v for k, v in file_b.items() if k != "monthly_series"})
    series_a_before = file_a["monthly_series"].copy()
    series_b_before = file_b["monthly_series"].copy()

    compare_files("file_value_comparison", file_a, file_b, column_hint="TotalPrice", metric="sum")

    assert {k: v for k, v in file_a.items() if k != "monthly_series"} == file_a_before
    assert {k: v for k, v in file_b.items() if k != "monthly_series"} == file_b_before
    assert file_a["monthly_series"].equals(series_a_before)
    assert file_b["monthly_series"].equals(series_b_before)


# ==================================================
# Resolution reuse verification
# ==================================================


def test_column_resolution_is_independent_per_file_and_reuses_resolve_column():
    import src.qa_engine as qa_engine

    calls = []
    original = qa_engine.resolve_column

    def spy(hint, names):
        calls.append((hint, list(names)))
        return original(hint, names)

    import src.multi_file_comparison as module
    module.resolve_column = spy
    try:
        file_a = _file_analysis(numeric_summary=_numeric_summary(name="TotalPrice"))
        file_b = _file_analysis(numeric_summary=_numeric_summary(name="Total Price"))

        compare_files("file_value_comparison", file_a, file_b, column_hint="TotalPrice", metric="sum")
    finally:
        module.resolve_column = original

    assert calls == [
        ("TotalPrice", ["TotalPrice"]),
        ("TotalPrice", ["Total Price"]),
    ]


def test_period_resolution_is_independent_per_file_and_reuses_resolve_period():
    import src.qa_engine as qa_engine

    calls = []
    original = qa_engine.resolve_period

    def spy(hint, labels):
        calls.append((hint, list(labels)))
        return original(hint, labels)

    import src.multi_file_comparison as module
    module.resolve_period = spy
    try:
        file_a = _file_analysis(monthly_series=_monthly_series((1.0,), ("2024-01",)))
        file_b = _file_analysis(monthly_series=_monthly_series((2.0,), ("2024-01", ), name="TotalPrice"))

        compare_files("file_period_comparison", file_a, file_b, period_hint="January")
    finally:
        module.resolve_period = original

    assert calls == [
        ("January", ["2024-01"]),
        ("January", ["2024-01"]),
    ]
