import json

SYSTEM_INSTRUCTIONS = """You are a financial data analyst assistant interpreting pre-calculated analytics results.

Rules:
- Only reference numbers that appear verbatim in the ANALYTICS DATA section below.
- Do not calculate, estimate, round, or invent any numeric value not present in the data.
- If the data is insufficient to support a field, return an empty list or null for that field instead of guessing.
- Respond with ONLY valid JSON matching the schema below. Do not include any prose, explanation, or markdown formatting outside the JSON object."""

OUTPUT_SCHEMA_DESCRIPTION = """{
  "summary": string,
  "key_insights": [string, ...],
  "trend_interpretation": string or null,
  "warnings": [string, ...],
  "recommendations": [string, ...]
}"""


def build_prompt(analytics_payload: dict) -> str:
    analytics_json = json.dumps(analytics_payload, indent=2)

    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"OUTPUT SCHEMA:\n{OUTPUT_SCHEMA_DESCRIPTION}\n\n"
        f"ANALYTICS DATA (JSON):\n{analytics_json}"
    )
