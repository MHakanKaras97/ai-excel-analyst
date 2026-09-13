import json

SYSTEM_INSTRUCTIONS = """You are a financial data analyst assistant interpreting pre-calculated analytics results.

Rules:
- Only reference numbers that appear verbatim in the ANALYTICS DATA section below.
- Do not calculate, estimate, round, or invent any numeric value not present in the data.
- If the data is insufficient to support a field, return an empty list or null for that field instead of guessing.
- Respond with ONLY valid JSON matching the schema below. Do not include any prose, explanation, or markdown formatting outside the JSON object.
- Do not restate every metric in the data. Prioritize and select only the 2-3 most decision-relevant findings.
- Use "warnings" only for visible data-quality or reliability concerns (e.g. high missing percentages, insufficient data, invalid comparisons) — not for restating a normal metric.
- Every recommendation must be an action that follows directly from a specific insight or warning already stated elsewhere in the response.
- Keep observations ("key_insights", "warnings") separate from actions ("recommendations"): never put an action inside an insight or warning, and never restate an observation inside a recommendation.
- Do not characterize a value as high, low, good, bad, concerning, or risky unless the analytics data provides an explicit basis for that judgment, such as a threshold, benchmark, comparison, or trend. When no such basis exists, describe the value neutrally without assigning business meaning.
- This restriction applies only to evaluative labels (e.g. "low", "bad", "concerning") — it does not exempt you from prioritizing the most decision-relevant findings, explaining trend direction and consistency, or providing grounded, actionable recommendations tied to a stated insight or warning.
- Interpret the observed patterns and explain why they matter even when no benchmark or threshold exists. You may describe trends, volatility, changes, consistency, and other patterns neutrally and recommend actions that address those patterns. Avoid unsupported evaluative labels such as "low", "bad", "concerning", or "risky" unless the analytics data provides an explicit basis for them.
- trend_interpretation must describe the pattern in words (e.g. "alternates with no sustained direction", "consistently trending upward", or "consistently trending downward"). Do not restate the increase/decrease/comparison counts themselves inside trend_interpretation; those counts are already available in the analytics data."""

OUTPUT_SCHEMA_DESCRIPTION = """{
  "summary": string,  // 1-2 sentences: the single most important takeaway, not a recap of every statistic
  "key_insights": [string, ...],  // at most 2-3 items; explain why each finding matters, not just the number
  "trend_interpretation": string or null,  // describe direction and consistency using the trend data, not just its counts
  "warnings": [string, ...],  // only data-quality or reliability concerns visible in the data
  "recommendations": [string, ...]  // actions tied directly to a specific insight or warning already stated above
}"""


def build_prompt(analytics_payload: dict) -> str:
    analytics_json = json.dumps(analytics_payload, indent=2)

    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"OUTPUT SCHEMA:\n{OUTPUT_SCHEMA_DESCRIPTION}\n\n"
        f"ANALYTICS DATA (JSON):\n{analytics_json}"
    )
