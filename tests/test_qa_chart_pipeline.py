"""End-to-end chart pipeline test (V0.6.5).

Exercises the full chain by locally composing the three existing, already
independently-tested stages — no new production code is introduced:

    question -> chart_intent_interpreter.interpret_chart_question()
             -> Structured Chart Intent
             -> qa_chart_engine.build_chart_spec()
             -> Grounded Chart Specification
             -> qa_chart_renderer.render_chart_spec()
             -> Plotly Figure (or None)

A FakeProvider stands in for the LLM boundary — no live Gemini call is made.
"""
import copy
import json
import sys

import pandas as pd
import plotly.graph_objects as go

from src.ai_provider import AIProviderError
from src.chart_intent_interpreter import interpret_chart_question
from src.qa_chart_engine import build_chart_spec
from src.qa_chart_renderer import render_chart_spec

_THIS_MODULE = sys.modules[__name__]


class FakeProvider:
    def __init__(self, response=None, error: AIProviderError | None = None):
        self._response = response
        self._error = error
        self.last_prompt = None
        self.call_count = 0

    def generate(self, prompt: str) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        if self._error is not None:
            raise self._error
        return self._response


def _run_chart_pipeline(question, column_names, provider, analysis_payload, monthly_series=None, anomalies=None):
    """Mirrors exactly the sequence a future caller (e.g. app.py) would run."""
    interpretation = interpret_chart_question(question, column_names, provider)
    if not interpretation["is_valid"]:
        return None

    spec = build_chart_spec(interpretation["intent"], analysis_payload, monthly_series, anomalies)
    return render_chart_spec(spec)


def _intent_response(**overrides):
    payload = {
        "intent": "trend_chart", "column_hint": None, "metric": None,
        "period_hint": None, "from_period_hint": None, "to_period_hint": None,
    }
    payload.update(overrides)
    return json.dumps(payload)


def _numeric_summary():
    return {
        "columns": [
            {"name": "TotalPrice", "count": 3, "missing_count": 0, "sum": 450.0,
             "mean": 150.0, "median": 150.0, "min": 100.0, "max": 200.0, "std": 50.0},
        ],
    }


def _monthly_series():
    return pd.Series([100.0, 200.0, 150.0], index=["2024-01", "2024-02", "2024-03"], name="TotalPrice")


def _period_comparison():
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


def _profile():
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


def _analysis_payload(**overrides):
    payload = {
        "numeric_summary": _numeric_summary(),
        "period_comparison": _period_comparison(),
        "profile": _profile(),
    }
    payload.update(overrides)
    return payload


# ==================================================
# FULL SUCCESSFUL PIPELINES
# ==================================================


def test_full_successful_trend_pipeline():
    provider = FakeProvider(response=_intent_response(intent="trend_chart", column_hint="TotalPrice"))

    figure = _run_chart_pipeline(
        "Show the TotalPrice trend.", ["TotalPrice"], provider, _analysis_payload(), _monthly_series(),
    )

    assert isinstance(figure, go.Figure)
    assert list(figure.data[0].x) == ["2024-01", "2024-02", "2024-03"]
    assert list(figure.data[0].y) == [100.0, 200.0, 150.0]
    assert provider.call_count == 1


def test_full_successful_numeric_summary_pipeline():
    provider = FakeProvider(response=_intent_response(intent="numeric_summary_chart", column_hint="TotalPrice", metric="mean"))

    figure = _run_chart_pipeline(
        "Show the average TotalPrice.", ["TotalPrice"], provider, _analysis_payload(),
    )

    assert isinstance(figure, go.Figure)
    assert list(figure.data[0].x) == ["TotalPrice"]
    assert list(figure.data[0].y) == [150.0]
    assert provider.call_count == 1


def test_full_successful_period_change_pipeline():
    provider = FakeProvider(response=_intent_response(
        intent="period_change_chart", from_period_hint="January", to_period_hint="February",
    ))

    figure = _run_chart_pipeline(
        "Plot the change from January to February.", ["TotalPrice"], provider,
        _analysis_payload(), _monthly_series(),
    )

    assert isinstance(figure, go.Figure)
    assert list(figure.data[0].x) == ["2024-01 → 2024-02"]
    assert list(figure.data[0].y) == [100.0]
    assert provider.call_count == 1


def test_full_successful_missing_values_pipeline():
    provider = FakeProvider(response=_intent_response(intent="missing_values_chart", column_hint=None))

    figure = _run_chart_pipeline(
        "Show missing values.", ["TotalPrice", "Region"], provider, _analysis_payload(),
    )

    assert isinstance(figure, go.Figure)
    assert list(figure.data[0].x) == ["TotalPrice", "Region"]
    assert list(figure.data[0].y) == [0.0, 20.0]
    assert provider.call_count == 1


def _quantity_monthly_series():
    return pd.Series(
        [500.0, 600.0, 700.0, 800.0, 900.0, 6260.0, 800.0, 400.0, 560.0],
        index=["2023-06", "2023-12", "2024-01", "2024-02", "2024-03",
               "2024-04", "2024-05", "2025-01", "2025-06"],
        name="Quantity",
    )


def test_full_reported_bug_trend_chart_restricted_to_year():
    # Exact reported request: "Show monthly quantity trend for 2024"
    # previously plotted the full range (~Jan 2023 -> Jul 2025) instead of
    # only the 2024 periods.
    provider = FakeProvider(response=_intent_response(
        intent="trend_chart", column_hint="Quantity", period_hint="2024",
    ))

    figure = _run_chart_pipeline(
        "Show monthly quantity trend for 2024", ["Quantity"], provider,
        _analysis_payload(numeric_summary={"columns": [
            {"name": "Quantity", "count": 9, "missing_count": 0, "sum": 11520.0,
             "mean": 1280.0, "median": 700.0, "min": 400.0, "max": 6260.0, "std": 1856.0},
        ]}),
        _quantity_monthly_series(),
    )

    assert isinstance(figure, go.Figure)
    assert list(figure.data[0].x) == ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05"]
    assert list(figure.data[0].y) == [700.0, 800.0, 900.0, 6260.0, 800.0]
    # Neither the 2023 nor 2025 data must appear on the chart.
    assert "2023-06" not in figure.data[0].x
    assert "2025-01" not in figure.data[0].x


# ==================================================
# INTERPRETER FAILURES
# ==================================================


def test_interpreter_invalid_json_stops_before_engine_and_renderer(monkeypatch):
    build_spec_calls = []
    render_calls = []
    monkeypatch.setattr(_THIS_MODULE, "build_chart_spec", lambda *a, **k: build_spec_calls.append(1) or {})
    monkeypatch.setattr(_THIS_MODULE, "render_chart_spec", lambda *a, **k: render_calls.append(1) or go.Figure())

    provider = FakeProvider(response="this is not json")

    figure = _run_chart_pipeline("Show the trend.", ["TotalPrice"], provider, _analysis_payload(), _monthly_series())

    assert figure is None
    assert build_spec_calls == []
    assert render_calls == []


def test_interpreter_invalid_schema_returns_none():
    provider = FakeProvider(response=_intent_response(intent="forecast_chart"))

    figure = _run_chart_pipeline("Predict next month.", ["TotalPrice"], provider, _analysis_payload(), _monthly_series())

    assert figure is None


def test_interpreter_provider_error_returns_none_with_single_call():
    provider = FakeProvider(error=AIProviderError("provider_error"))

    figure = _run_chart_pipeline("Show the trend.", ["TotalPrice"], provider, _analysis_payload(), _monthly_series())

    assert figure is None
    assert provider.call_count == 1


# ==================================================
# ENGINE / RENDERER FAILURES
# ==================================================


def test_engine_failure_produces_no_figure():
    # column_hint doesn't exist anywhere in numeric_summary -> column_not_found.
    provider = FakeProvider(response=_intent_response(intent="trend_chart", column_hint="Nonexistent"))

    figure = _run_chart_pipeline("Show the Nonexistent trend.", ["TotalPrice"], provider, _analysis_payload(), _monthly_series())

    assert figure is None


def test_unsupported_chart_intent_stops_safely_with_no_figure():
    provider = FakeProvider(response=_intent_response(intent="unsupported", column_hint=None))

    figure = _run_chart_pipeline("Why did sales decrease?", ["TotalPrice"], provider, _analysis_payload(), _monthly_series())

    assert figure is None


def test_column_series_mismatch_produces_no_figure():
    payload = _analysis_payload(numeric_summary={
        "columns": [
            {"name": "TotalPrice", "sum": 450.0, "mean": 150.0, "median": 150.0, "min": 100.0, "max": 200.0, "std": 50.0, "count": 3},
            {"name": "Revenue", "sum": 900.0, "mean": 300.0, "median": 300.0, "min": 200.0, "max": 400.0, "std": 100.0, "count": 3},
        ],
    })
    provider = FakeProvider(response=_intent_response(intent="trend_chart", column_hint="Revenue"))

    figure = _run_chart_pipeline(
        "Show the Revenue trend.", ["TotalPrice", "Revenue"], provider, payload, _monthly_series(),
    )

    assert figure is None


def test_no_hallucination_wrong_column_hint_does_not_render_against_another_column():
    # A column_hint that is a weak/decoy reference must not silently resolve
    # to *any* column (neither the intended one nor a substitute) — the
    # pipeline must fail closed, never invent or substitute data.
    provider = FakeProvider(response=_intent_response(
        intent="trend_chart", column_hint="an unrelated decoy reference 999999.99",
    ))

    figure = _run_chart_pipeline(
        "Show the trend.", ["TotalPrice"], provider, _analysis_payload(), _monthly_series(),
    )

    assert figure is None


# ==================================================
# NO MUTATION
# ==================================================


def test_pipeline_does_not_mutate_inputs():
    provider = FakeProvider(response=_intent_response(intent="trend_chart", column_hint="TotalPrice"))
    payload = _analysis_payload()
    series = _monthly_series()
    column_names = ["TotalPrice"]

    payload_before = copy.deepcopy(payload)
    series_before = series.copy()
    column_names_before = list(column_names)

    _run_chart_pipeline("Show the TotalPrice trend.", column_names, provider, payload, series)

    assert payload == payload_before
    assert series.equals(series_before)
    assert column_names == column_names_before


# ==================================================
# PROVIDER CALL COUNT / RENDER-ONLY-ON-SUCCESS GUARANTEES
# ==================================================


def test_provider_called_exactly_once_on_every_interpreter_success_case():
    cases = [
        _intent_response(intent="trend_chart", column_hint="TotalPrice"),
        _intent_response(intent="numeric_summary_chart", column_hint="TotalPrice", metric="sum"),
        _intent_response(intent="period_change_chart"),
        _intent_response(intent="missing_values_chart"),
        _intent_response(intent="unsupported"),
    ]
    for response in cases:
        provider = FakeProvider(response=response)
        _run_chart_pipeline("A question.", ["TotalPrice"], provider, _analysis_payload(), _monthly_series())
        assert provider.call_count == 1


def test_render_chart_spec_is_only_called_with_a_found_true_spec(monkeypatch):
    seen_specs = []
    real_render = render_chart_spec

    def spy_render(spec):
        seen_specs.append(spec)
        return real_render(spec)

    monkeypatch.setattr(_THIS_MODULE, "render_chart_spec", spy_render)

    provider = FakeProvider(response=_intent_response(intent="trend_chart", column_hint="Nonexistent"))
    figure = _run_chart_pipeline("Show the trend.", ["TotalPrice"], provider, _analysis_payload(), _monthly_series())

    assert figure is None
    assert len(seen_specs) == 1
    assert seen_specs[0]["found"] is False


def test_rendering_occurs_only_after_a_found_true_grounded_spec(monkeypatch):
    build_spec_calls = []
    render_calls = []
    real_build_spec = build_chart_spec
    real_render = render_chart_spec

    def spy_build_spec(*args, **kwargs):
        result = real_build_spec(*args, **kwargs)
        build_spec_calls.append(result)
        return result

    def spy_render(spec):
        render_calls.append(spec)
        return real_render(spec)

    monkeypatch.setattr(_THIS_MODULE, "build_chart_spec", spy_build_spec)
    monkeypatch.setattr(_THIS_MODULE, "render_chart_spec", spy_render)

    provider = FakeProvider(response=_intent_response(intent="trend_chart", column_hint="TotalPrice"))
    figure = _run_chart_pipeline("Show the trend.", ["TotalPrice"], provider, _analysis_payload(), _monthly_series())

    assert isinstance(figure, go.Figure)
    assert len(build_spec_calls) == 1
    assert build_spec_calls[0]["found"] is True
    assert len(render_calls) == 1
    assert render_calls[0]["found"] is True
