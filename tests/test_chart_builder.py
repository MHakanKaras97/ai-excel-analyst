import copy

import pandas as pd
import plotly.graph_objects as go
import pytest

from src.chart_builder import (
    build_missing_values_chart,
    build_numeric_summary_chart,
    build_period_change_chart,
    build_trend_chart,
)


def test_build_trend_chart_uses_scatter_trace():
    series = pd.Series([100.0, 200.0], index=["2023-01", "2023-02"])

    fig = build_trend_chart(series)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "scatter"


def test_build_trend_chart_uses_lines_and_markers_mode():
    series = pd.Series([100.0, 200.0], index=["2023-01", "2023-02"])

    fig = build_trend_chart(series)

    assert fig.data[0].mode == "lines+markers"


def test_build_trend_chart_x_values_match_index_order():
    series = pd.Series([100.0, 200.0, 150.0], index=["2023-01", "2023-02", "2023-03"])

    fig = build_trend_chart(series)

    assert list(fig.data[0].x) == ["2023-01", "2023-02", "2023-03"]


def test_build_trend_chart_y_values_match_series_values_order():
    series = pd.Series([100.0, 200.0, 150.0], index=["2023-01", "2023-02", "2023-03"])

    fig = build_trend_chart(series)

    assert list(fig.data[0].y) == [100.0, 200.0, 150.0]


def test_build_trend_chart_respects_supplied_title():
    series = pd.Series([100.0], index=["2023-01"])

    fig = build_trend_chart(series, title="Custom Title")

    assert fig.layout.title.text == "Custom Title"


def test_build_trend_chart_has_default_title_when_none_supplied():
    series = pd.Series([100.0], index=["2023-01"])

    fig = build_trend_chart(series)

    assert fig.layout.title.text is not None
    assert len(fig.layout.title.text) > 0


def test_build_trend_chart_sets_axis_titles():
    series = pd.Series([100.0], index=["2023-01"])

    fig = build_trend_chart(series)

    assert fig.layout.xaxis.title.text == "Period"
    assert fig.layout.yaxis.title.text == "Value"


def test_build_trend_chart_empty_series_returns_valid_figure():
    series = pd.Series([], dtype="float64")

    fig = build_trend_chart(series)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert list(fig.data[0].x) == []
    assert list(fig.data[0].y) == []


def test_build_trend_chart_single_point_series_returns_valid_figure():
    series = pd.Series([42.0], index=["2023-01"])

    fig = build_trend_chart(series)

    assert isinstance(fig, go.Figure)
    assert list(fig.data[0].x) == ["2023-01"]
    assert list(fig.data[0].y) == [42.0]


def test_build_trend_chart_does_not_mutate_input_series():
    series = pd.Series([100.0, 200.0], index=["2023-01", "2023-02"])
    before = series.copy()

    build_trend_chart(series)

    pd.testing.assert_series_equal(series, before)


def _numeric_summary(columns):
    return {"columns": columns}


def test_build_numeric_summary_chart_uses_bar_trace():
    numeric_summary = _numeric_summary([
        {"name": "revenue", "sum": 100.0, "mean": 50.0, "median": 50.0, "min": 10.0, "max": 90.0, "std": 5.0, "count": 2},
    ])

    fig = build_numeric_summary_chart(numeric_summary)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "bar"


def test_build_numeric_summary_chart_preserves_column_order():
    numeric_summary = _numeric_summary([
        {"name": "zeta", "sum": 3.0},
        {"name": "alpha", "sum": 1.0},
        {"name": "middle", "sum": 2.0},
    ])

    fig = build_numeric_summary_chart(numeric_summary)

    assert list(fig.data[0].x) == ["zeta", "alpha", "middle"]


def test_build_numeric_summary_chart_uses_selected_metric_values():
    numeric_summary = _numeric_summary([
        {"name": "revenue", "sum": 100.0, "mean": 25.0},
        {"name": "cost", "sum": 60.0, "mean": 15.0},
    ])

    fig = build_numeric_summary_chart(numeric_summary, metric="mean")

    assert list(fig.data[0].y) == [25.0, 15.0]


def test_build_numeric_summary_chart_default_metric_is_sum():
    numeric_summary = _numeric_summary([
        {"name": "revenue", "sum": 100.0, "mean": 25.0},
    ])

    fig = build_numeric_summary_chart(numeric_summary)

    assert list(fig.data[0].y) == [100.0]


def test_build_numeric_summary_chart_supports_min_and_max_metrics():
    numeric_summary = _numeric_summary([
        {"name": "revenue", "min": 5.0, "max": 90.0},
    ])

    fig_min = build_numeric_summary_chart(numeric_summary, metric="min")
    fig_max = build_numeric_summary_chart(numeric_summary, metric="max")

    assert list(fig_min.data[0].y) == [5.0]
    assert list(fig_max.data[0].y) == [90.0]


def test_build_numeric_summary_chart_unsupported_metric_raises_value_error():
    numeric_summary = _numeric_summary([{"name": "revenue", "sum": 100.0}])

    with pytest.raises(ValueError):
        build_numeric_summary_chart(numeric_summary, metric="variance")


def test_build_numeric_summary_chart_empty_columns_returns_valid_figure():
    numeric_summary = _numeric_summary([])

    fig = build_numeric_summary_chart(numeric_summary)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert list(fig.data[0].x) == []
    assert list(fig.data[0].y) == []


def test_build_numeric_summary_chart_preserves_duplicate_column_name_labels():
    # numeric_summary carries only names, not positions, so duplicate-named
    # columns intentionally produce duplicate x-axis labels here — this test
    # documents that behavior rather than silently disambiguating it.
    numeric_summary = _numeric_summary([
        {"name": "revenue", "sum": 100.0},
        {"name": "revenue", "sum": 200.0},
    ])

    fig = build_numeric_summary_chart(numeric_summary)

    assert list(fig.data[0].x) == ["revenue", "revenue"]
    assert list(fig.data[0].y) == [100.0, 200.0]


def test_build_numeric_summary_chart_does_not_mutate_input_dict():
    numeric_summary = _numeric_summary([
        {"name": "revenue", "sum": 100.0},
    ])
    before = copy.deepcopy(numeric_summary)

    build_numeric_summary_chart(numeric_summary)

    assert numeric_summary == before


def test_build_numeric_summary_chart_preserves_none_values():
    numeric_summary = _numeric_summary([
        {"name": "revenue", "std": None},
        {"name": "cost", "std": 3.5},
    ])

    fig = build_numeric_summary_chart(numeric_summary, metric="std")

    assert list(fig.data[0].y) == [None, 3.5]


def _comparison(from_period, to_period, previous, current, absolute_change, percentage_change, is_valid=True):
    return {
        "from_period": from_period,
        "to_period": to_period,
        "previous": previous,
        "current": current,
        "absolute_change": absolute_change,
        "percentage_change": percentage_change,
        "is_valid": is_valid,
    }


def _period_comparison(comparisons):
    return {
        "periods": [],
        "comparisons": comparisons,
        "valid_comparison_count": sum(1 for c in comparisons if c["is_valid"]),
        "invalid_comparison_count": sum(1 for c in comparisons if not c["is_valid"]),
        "insufficient_data": False,
    }


def test_build_period_change_chart_returns_figure_with_bar_trace():
    period_comparison = _period_comparison([
        _comparison("2025-01", "2025-02", 100.0, 150.0, 50.0, 50.0),
    ])

    fig = build_period_change_chart(period_comparison)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "bar"


def test_build_period_change_chart_uses_transition_labels():
    period_comparison = _period_comparison([
        _comparison("2025-01", "2025-02", 100.0, 150.0, 50.0, 50.0),
    ])

    fig = build_period_change_chart(period_comparison)

    assert list(fig.data[0].x) == ["2025-01 → 2025-02"]


def test_build_period_change_chart_preserves_comparison_order():
    period_comparison = _period_comparison([
        _comparison("2025-03", "2025-04", 300.0, 250.0, -50.0, -16.7),
        _comparison("2025-01", "2025-02", 100.0, 150.0, 50.0, 50.0),
    ])

    fig = build_period_change_chart(period_comparison)

    assert list(fig.data[0].x) == ["2025-03 → 2025-04", "2025-01 → 2025-02"]
    assert list(fig.data[0].y) == [-50.0, 50.0]


def test_build_period_change_chart_uses_absolute_change_values_for_multiple_comparisons():
    period_comparison = _period_comparison([
        _comparison("2025-01", "2025-02", 100.0, 150.0, 50.0, 50.0),
        _comparison("2025-02", "2025-03", 150.0, 120.0, -30.0, -20.0),
        _comparison("2025-03", "2025-04", 120.0, 120.0, 0.0, 0.0),
    ])

    fig = build_period_change_chart(period_comparison)

    assert list(fig.data[0].y) == [50.0, -30.0, 0.0]


def test_build_period_change_chart_preserves_none_absolute_change():
    period_comparison = _period_comparison([
        _comparison("2025-01", "2025-02", None, 150.0, None, None, is_valid=False),
    ])

    fig = build_period_change_chart(period_comparison)

    assert list(fig.data[0].y) == [None]


def test_build_period_change_chart_empty_comparisons_returns_valid_figure():
    period_comparison = _period_comparison([])

    fig = build_period_change_chart(period_comparison)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert list(fig.data[0].x) == []
    assert list(fig.data[0].y) == []


def test_build_period_change_chart_does_not_mutate_input_dict():
    period_comparison = _period_comparison([
        _comparison("2025-01", "2025-02", 100.0, 150.0, 50.0, 50.0),
    ])
    before = copy.deepcopy(period_comparison)

    build_period_change_chart(period_comparison)

    assert period_comparison == before


def test_build_period_change_chart_does_not_use_percentage_change():
    # absolute_change and percentage_change deliberately diverge here so that
    # accidentally reading percentage_change would be caught by this test.
    period_comparison = _period_comparison([
        _comparison("2025-01", "2025-02", 100.0, 150.0, 50.0, 999.0),
    ])

    fig = build_period_change_chart(period_comparison)

    assert list(fig.data[0].y) == [50.0]


def test_build_period_change_chart_sets_expected_layout():
    period_comparison = _period_comparison([
        _comparison("2025-01", "2025-02", 100.0, 150.0, 50.0, 50.0),
    ])

    fig = build_period_change_chart(period_comparison)

    assert fig.layout.title.text == "Period Change"
    assert fig.layout.xaxis.title.text == "Period"
    assert fig.layout.yaxis.title.text == "Absolute Change"


def _profile_column(name, missing_percentage, dtype="float64", inferred_type="numeric",
                     unique_count=1, missing_count=0):
    return {
        "name": name,
        "dtype": dtype,
        "inferred_type": inferred_type,
        "unique_count": unique_count,
        "missing_count": missing_count,
        "missing_percentage": missing_percentage,
    }


def _profile(columns):
    return {
        "row_count": 0,
        "column_count": len(columns),
        "is_empty": False,
        "duplicate_row_count": 0,
        "duplicate_column_names": [],
        "columns": columns,
    }


def test_build_missing_values_chart_returns_figure_with_bar_trace():
    profile = _profile([_profile_column("revenue", 0.0)])

    fig = build_missing_values_chart(profile)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "bar"


def test_build_missing_values_chart_uses_column_labels():
    profile = _profile([_profile_column("revenue", 0.0), _profile_column("notes", 100.0)])

    fig = build_missing_values_chart(profile)

    assert list(fig.data[0].x) == ["revenue", "notes"]


def test_build_missing_values_chart_preserves_column_order():
    profile = _profile([
        _profile_column("zeta", 10.0),
        _profile_column("alpha", 20.0),
        _profile_column("middle", 30.0),
    ])

    fig = build_missing_values_chart(profile)

    assert list(fig.data[0].x) == ["zeta", "alpha", "middle"]
    assert list(fig.data[0].y) == [10.0, 20.0, 30.0]


def test_build_missing_values_chart_uses_missing_percentage_values_directly():
    profile = _profile([_profile_column("revenue", 12.5), _profile_column("cost", 87.5)])

    fig = build_missing_values_chart(profile)

    assert list(fig.data[0].y) == [12.5, 87.5]


def test_build_missing_values_chart_preserves_none_values():
    profile = _profile([_profile_column("revenue", None), _profile_column("cost", 5.0)])

    fig = build_missing_values_chart(profile)

    assert list(fig.data[0].y) == [None, 5.0]


def test_build_missing_values_chart_preserves_duplicate_column_name_labels():
    profile = _profile([_profile_column("revenue", 0.0), _profile_column("revenue", 100.0)])

    fig = build_missing_values_chart(profile)

    assert list(fig.data[0].x) == ["revenue", "revenue"]
    assert list(fig.data[0].y) == [0.0, 100.0]


def test_build_missing_values_chart_empty_columns_returns_valid_figure():
    profile = _profile([])

    fig = build_missing_values_chart(profile)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert list(fig.data[0].x) == []
    assert list(fig.data[0].y) == []


def test_build_missing_values_chart_does_not_mutate_input_profile():
    profile = _profile([_profile_column("revenue", 0.0)])
    before = copy.deepcopy(profile)

    build_missing_values_chart(profile)

    assert profile == before


def test_build_missing_values_chart_sets_expected_layout():
    profile = _profile([_profile_column("revenue", 0.0)])

    fig = build_missing_values_chart(profile)

    assert fig.layout.title.text == "Missing Values"
    assert fig.layout.xaxis.title.text == "Column"
    assert fig.layout.yaxis.title.text == "Missing Percentage"
