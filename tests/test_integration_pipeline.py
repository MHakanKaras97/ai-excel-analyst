import json

import pandas as pd

from src.ai_interpreter import interpret
from src.analytics_engine import analyze_dates, analyze_numeric, compare_periods, detect_trend
from src.data_normalizer import normalize_dataframe
from src.data_profiler import profile_dataframe
from src.excel_loader import load_excel

VALID_INSIGHT_RESPONSE = json.dumps({
    "summary": "Revenue increased steadily across all quarters.",
    "key_insights": ["Revenue grew from 100000.0 to 130000.0."],
    "trend_interpretation": "increasing",
    "warnings": [],
    "recommendations": [],
})

SENTINEL_NOTE = "CONFIDENTIAL_ROW_LEVEL_NOTE_998877"


class FakeProvider:
    def __init__(self, response: str):
        self._response = response
        self.last_prompt = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self._response


def _build_synthetic_excel(tmp_path):
    df = pd.DataFrame({
        "period": pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01"]),
        "revenue": ["$100,000", "$110,000", "$120,000", "$130,000"],
        "notes": [SENTINEL_NOTE, "n/a", "n/a", "n/a"],
    })
    path = tmp_path / "quarterly_revenue.xlsx"
    df.to_excel(path, index=False)
    return path


def test_full_pipeline_from_excel_to_ai_interpretation(tmp_path):
    excel_path = _build_synthetic_excel(tmp_path)

    # 1. ExcelLoader
    loaded_df = load_excel(excel_path)
    original_snapshot = loaded_df.copy(deep=True)
    assert pd.api.types.is_datetime64_any_dtype(loaded_df["period"])

    # 2. DataProfiler (on raw, pre-normalization data)
    profile = profile_dataframe(loaded_df)
    revenue_profile = next(c for c in profile["columns"] if c["name"] == "revenue")
    assert revenue_profile["inferred_type"] == "text"  # currency strings, not yet normalized

    # 3. DataNormalizer
    normalization_result = normalize_dataframe(loaded_df)
    normalized_df = normalization_result["normalized_df"]
    revenue_report = next(c for c in normalization_result["report"]["columns"] if c["name"] == "revenue")
    assert revenue_report["changed_count"] == 4
    assert revenue_report["unresolved_count"] == 0
    assert pd.api.types.is_numeric_dtype(normalized_df["revenue"])

    # 4. Analytics Engine
    numeric_summary = analyze_numeric(normalized_df)
    date_summary = analyze_dates(normalized_df)

    revenue_by_period = pd.Series(
        normalized_df["revenue"].to_numpy(),
        index=normalized_df["period"],
    )
    trend = detect_trend(revenue_by_period)
    period_comparison = compare_periods(revenue_by_period)

    assert numeric_summary["columns"][0]["sum"] == 460000.0
    assert date_summary["columns"][0]["count"] == 4
    assert trend["trend"] == "increasing"
    assert period_comparison["valid_comparison_count"] == 3

    # 5. Assemble the payload the AI layer is allowed to see
    analytics_payload = {
        "numeric_summary": numeric_summary,
        "date_summary": date_summary,
        "trend": trend,
        "period_comparison": period_comparison,
    }

    # 6 & 7. AI Interpreter with a FakeProvider
    provider = FakeProvider(response=VALID_INSIGHT_RESPONSE)
    result = interpret(analytics_payload, provider)

    assert result["is_valid"] is True
    assert result["reason"] is None
    assert result["insight"]["trend_interpretation"] == "increasing"

    assert "460000.0" in provider.last_prompt
    assert "increasing" in provider.last_prompt

    # 8. No raw row-level data reached the AI layer
    assert SENTINEL_NOTE not in provider.last_prompt
    assert "notes" not in provider.last_prompt
    assert "$100,000" not in provider.last_prompt

    # 9. Original DataFrame returned by ExcelLoader is untouched throughout
    pd.testing.assert_frame_equal(loaded_df, original_snapshot)
