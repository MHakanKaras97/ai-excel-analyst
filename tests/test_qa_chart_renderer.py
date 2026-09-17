import copy

import plotly.graph_objects as go
import pytest

import src.qa_chart_renderer as qa_chart_renderer
from src.qa_chart_renderer import render_chart_spec


def _base_spec(**overrides):
    spec = {
        "found": True, "reason": None, "intent": "trend_chart", "chart_type": "trend",
        "column": None, "metric": None, "period": None, "from_period": None, "to_period": None,
        "data": {}, "extra": {},
    }
    spec.update(overrides)
    return spec


def _trend_spec():
    return _base_spec(
        chart_type="trend", column="TotalPrice",
        data={"periods": ["2024-01", "2024-02", "2024-03"], "values": [100.0, 200.0, 150.0]},
        extra={"anomalies": None},
    )


def _numeric_summary_spec():
    return _base_spec(
        intent="numeric_summary_chart", chart_type="numeric_summary",
        column="TotalPrice", metric="sum",
        data={"columns": [{"name": "TotalPrice", "sum": 450.0, "mean": 150.0, "median": 150.0,
                            "min": 100.0, "max": 200.0, "std": 50.0, "count": 3}]},
    )


def _period_comparison_data():
    return {
        "periods": ["2024-01", "2024-02", "2024-03"],
        "comparisons": [
            {"from_period": "2024-01", "to_period": "2024-02", "previous": 100.0, "current": 200.0,
             "absolute_change": 100.0, "percentage_change": 100.0, "is_valid": True, "reason": None},
            {"from_period": "2024-02", "to_period": "2024-03", "previous": 200.0, "current": 150.0,
             "absolute_change": -50.0, "percentage_change": -25.0, "is_valid": True, "reason": None},
        ],
        "valid_comparison_count": 2, "invalid_comparison_count": 0, "insufficient_data": False,
    }


def _period_change_spec(filtered=False):
    data = _period_comparison_data()
    if filtered:
        data = {
            "periods": data["periods"], "comparisons": [data["comparisons"][0]],
            "valid_comparison_count": 1, "invalid_comparison_count": 0, "insufficient_data": False,
        }
    return _base_spec(
        intent="period_change_chart", chart_type="period_change", column="TotalPrice",
        from_period="2024-01" if filtered else None, to_period="2024-02" if filtered else None,
        data=data,
    )


def _profile_data():
    return {
        "row_count": 10, "column_count": 2, "is_empty": False,
        "duplicate_row_count": 0, "duplicate_column_names": [],
        "columns": [
            {"name": "TotalPrice", "dtype": "float64", "inferred_type": "numeric",
             "unique_count": 3, "missing_count": 0, "missing_percentage": 0.0},
            {"name": "Region", "dtype": "object", "inferred_type": "text",
             "unique_count": 3, "missing_count": 2, "missing_percentage": 20.0},
        ],
    }


def _missing_values_spec():
    return _base_spec(
        intent="missing_values_chart", chart_type="missing_values", data=_profile_data(),
    )


def _failure_spec(reason="column_not_found"):
    return _base_spec(found=False, reason=reason, chart_type=None, data={}, extra={})


# ==================================================
# FAILURE / UNSUPPORTED
# ==================================================


def test_found_false_returns_none():
    result = render_chart_spec(_failure_spec())

    assert result is None


def test_found_false_does_not_call_any_builder(monkeypatch):
    calls = []
    for name in ("build_trend_chart", "build_numeric_summary_chart", "build_period_change_chart", "build_missing_values_chart"):
        monkeypatch.setattr(qa_chart_renderer, name, lambda *a, **k: calls.append(name) or go.Figure())

    render_chart_spec(_failure_spec())

    assert calls == []


def test_unknown_chart_type_raises_value_error():
    spec = _base_spec(chart_type="forecast")

    with pytest.raises(ValueError):
        render_chart_spec(spec)


def test_unsupported_intent_spec_returns_none():
    spec = _base_spec(found=False, reason="unsupported", intent="unsupported", chart_type=None)

    assert render_chart_spec(spec) is None


# ==================================================
# TREND
# ==================================================


def test_trend_rendering_produces_figure_with_expected_data():
    spec = _trend_spec()

    fig = render_chart_spec(spec)

    assert isinstance(fig, go.Figure)
    assert list(fig.data[0].x) == ["2024-01", "2024-02", "2024-03"]
    assert list(fig.data[0].y) == [100.0, 200.0, 150.0]


def test_trend_with_anomalies_adds_marker_trace():
    spec = _trend_spec()
    spec["extra"]["anomalies"] = {
        "method": "iqr", "insufficient_data": False, "sample_size": 3,
        "lower_bound": 10.0, "upper_bound": 190.0, "anomaly_count": 1,
        "anomalies": [{"position": 1, "period": "2024-02", "value": 200.0, "direction": "high"}],
    }

    fig = render_chart_spec(spec)

    assert len(fig.data) == 2
    assert list(fig.data[1].x) == ["2024-02"]
    assert list(fig.data[1].marker.color) == ["red"]


def test_trend_without_anomalies_has_single_trace():
    spec = _trend_spec()
    spec["extra"]["anomalies"] = None

    fig = render_chart_spec(spec)

    assert len(fig.data) == 1


def test_trend_series_name_matches_spec_column():
    spec = _trend_spec()

    fig = render_chart_spec(spec)

    # build_trend_chart doesn't expose the series name on the figure directly,
    # so this is verified via the spy test below; here we just confirm the
    # figure still renders correctly when column is set.
    assert isinstance(fig, go.Figure)


# ==================================================
# NUMERIC SUMMARY
# ==================================================


@pytest.mark.parametrize("metric", ["sum", "mean", "median", "min", "max", "std", "count"])
def test_numeric_summary_rendering_for_each_supported_metric(metric):
    spec = _numeric_summary_spec()
    spec["metric"] = metric

    fig = render_chart_spec(spec)

    expected_value = spec["data"]["columns"][0][metric]
    assert isinstance(fig, go.Figure)
    assert list(fig.data[0].x) == ["TotalPrice"]
    assert list(fig.data[0].y) == [expected_value]


# ==================================================
# PERIOD CHANGE
# ==================================================


def test_period_change_general_spec_renders_all_comparisons():
    spec = _period_change_spec(filtered=False)

    fig = render_chart_spec(spec)

    assert isinstance(fig, go.Figure)
    assert list(fig.data[0].x) == ["2024-01 → 2024-02", "2024-02 → 2024-03"]
    assert list(fig.data[0].y) == [100.0, -50.0]


def test_period_change_filtered_spec_renders_only_matched_comparison():
    spec = _period_change_spec(filtered=True)

    fig = render_chart_spec(spec)

    assert list(fig.data[0].x) == ["2024-01 → 2024-02"]
    assert list(fig.data[0].y) == [100.0]


# ==================================================
# MISSING VALUES
# ==================================================


def test_missing_values_rendering():
    spec = _missing_values_spec()

    fig = render_chart_spec(spec)

    assert isinstance(fig, go.Figure)
    assert list(fig.data[0].x) == ["TotalPrice", "Region"]
    assert list(fig.data[0].y) == [0.0, 20.0]


# ==================================================
# NO MUTATION
# ==================================================


@pytest.mark.parametrize("spec_factory", [
    _trend_spec, _numeric_summary_spec,
    lambda: _period_change_spec(filtered=False),
    lambda: _period_change_spec(filtered=True),
    _missing_values_spec,
])
def test_render_chart_spec_does_not_mutate_input_spec(spec_factory):
    spec = spec_factory()
    before = copy.deepcopy(spec)

    render_chart_spec(spec)

    assert spec == before


def test_render_chart_spec_does_not_mutate_failure_spec():
    spec = _failure_spec()
    before = copy.deepcopy(spec)

    render_chart_spec(spec)

    assert spec == before


# ==================================================
# BUILDER REUSE (SPY VERIFICATION)
# ==================================================


def test_trend_calls_build_trend_chart_exactly_once_with_expected_arguments(monkeypatch):
    calls = []

    def fake_build_trend_chart(series, title=None, anomalies=None):
        calls.append((series, anomalies))
        return go.Figure()

    monkeypatch.setattr(qa_chart_renderer, "build_trend_chart", fake_build_trend_chart)

    spec = _trend_spec()
    spec["extra"]["anomalies"] = {"anomaly_count": 0, "anomalies": []}

    render_chart_spec(spec)

    assert len(calls) == 1
    series, anomalies = calls[0]
    assert list(series.index) == ["2024-01", "2024-02", "2024-03"]
    assert list(series.to_numpy()) == [100.0, 200.0, 150.0]
    assert series.name == "TotalPrice"
    assert anomalies == {"anomaly_count": 0, "anomalies": []}


def test_numeric_summary_calls_build_numeric_summary_chart_exactly_once_with_expected_arguments(monkeypatch):
    calls = []

    def fake_build_numeric_summary_chart(numeric_summary, metric="sum"):
        calls.append((numeric_summary, metric))
        return go.Figure()

    monkeypatch.setattr(qa_chart_renderer, "build_numeric_summary_chart", fake_build_numeric_summary_chart)

    spec = _numeric_summary_spec()
    render_chart_spec(spec)

    assert len(calls) == 1
    numeric_summary, metric = calls[0]
    assert numeric_summary == spec["data"]
    assert metric == "sum"


def test_period_change_calls_build_period_change_chart_exactly_once_with_expected_arguments(monkeypatch):
    calls = []

    def fake_build_period_change_chart(period_comparison):
        calls.append(period_comparison)
        return go.Figure()

    monkeypatch.setattr(qa_chart_renderer, "build_period_change_chart", fake_build_period_change_chart)

    spec = _period_change_spec(filtered=True)
    render_chart_spec(spec)

    assert len(calls) == 1
    assert calls[0] == spec["data"]


def test_missing_values_calls_build_missing_values_chart_exactly_once_with_expected_arguments(monkeypatch):
    calls = []

    def fake_build_missing_values_chart(profile):
        calls.append(profile)
        return go.Figure()

    monkeypatch.setattr(qa_chart_renderer, "build_missing_values_chart", fake_build_missing_values_chart)

    spec = _missing_values_spec()
    render_chart_spec(spec)

    assert len(calls) == 1
    assert calls[0] == spec["data"]
