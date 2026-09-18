import json

SYSTEM_INSTRUCTIONS = """You are a multi-file chart-request interpreter for a financial data analyst tool.

Your ONLY job is to:
1. classify the user's chart request into exactly one of the supported intents below
2. extract free-text hints, including which two files are being compared

You must NOT:
- calculate anything
- resolve column names, periods, or file identities
- perform fuzzy matching
- generate a chart
- generate Plotly code

NEVER invent a value for any hint.
NEVER resolve a hint to an actual uploaded file or dataset column. All resolution and chart generation is performed later by Python, not by you.

Supported intents:
- "file_comparison_chart": The user wants a chart comparing a numeric statistic (metric) of a column between two files (e.g. "compare total sales between the two files").
- "unsupported": Use when the request does not clearly correspond to a two-file value comparison chart (e.g. a trend chart across files, an anomaly chart, or anything unrelated to comparing two files' values). Do not force an ambiguous request into the supported intent.

Hint extraction rules:
- column_hint: the column name or phrase the user used, exactly/as closely as practical, or null if none is mentioned.
- metric: only when the user explicitly names one — "average"/"mean" -> mean, "total"/"sum" -> sum, "median" -> median, "minimum" -> min, "maximum" -> max, "standard deviation" -> std, "count" -> count. Use null if no metric is explicitly requested — never infer one.
- from_file_hint: the user's own wording for the first ("previous"/baseline) file — a role such as "previous" or a file's display name. Use null if not mentioned.
- to_file_hint: the user's own wording for the second ("current"/comparison) file — a role such as "current" or a file's display name. Use null if not mentioned."""

OUTPUT_SCHEMA_DESCRIPTION = """{
  "intent": "file_comparison_chart | unsupported",
  "column_hint": "string | null",
  "metric": "sum | mean | median | min | max | std | count | null",
  "period_hint": "string | null",
  "from_file_hint": "string | null",
  "to_file_hint": "string | null"
}"""

OUTPUT_RULES = """Output rules:
- Return JSON only. Do not include any prose, explanation, or markdown formatting outside the JSON object.
- Use exactly the intent and metric values listed above, spelled exactly as shown.
- Use null where a field does not apply. "period_hint" is not used by the supported intent and should always be null.
- Return exactly these six keys — no extra keys, no missing keys.
- Preserve hints as plain strings, exactly reflecting the user's wording."""


def build_multi_file_chart_intent_prompt(question: str, column_names: list[str], file_labels: list[str]) -> str:
    columns_json = json.dumps(list(column_names))
    files_json = json.dumps(list(file_labels))

    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"OUTPUT SCHEMA:\n{OUTPUT_SCHEMA_DESCRIPTION}\n\n"
        f"{OUTPUT_RULES}\n\n"
        f"KNOWN DATASET COLUMNS (JSON, context only — do not resolve one):\n{columns_json}\n\n"
        f"KNOWN FILE ROLES/DISPLAY NAMES (JSON, context only — do not resolve one):\n{files_json}\n\n"
        f"USER QUESTION:\n{question}"
    )
