"""Prompt builder for LLM semantic column proposals (V0.8.5 / V0.8.13).

Sends compact structured metadata per column (name, dtype, inferred type,
a few sample values, and the deterministic candidate's own role/type/
confidence) — never the raw dataset and never any internal identifier such
as `file_id`. The LLM's response is a proposal only; schema_resolver.
merge_llm_schema_proposal() validates and may reject or override any part
of it.
"""
import json

SYSTEM_INSTRUCTIONS = """You are a data schema analyst proposing semantic labels for spreadsheet columns.

You will be given a compact description of each column: its name, a deterministic dtype-based classification already computed by Python, and a few sample values. Python's classification is a starting point, not something you must repeat — propose corrections only where you have real justification from the name or sample values.

For every column, propose:
- role: one of dimension, measure, date, identifier, unknown
- semantic_type: one of numeric, currency, percentage, date, text, identifier, unknown
- unit: a short unit string (e.g. "USD", "EUR", "%") if evident, otherwise null
- time_role: one of period, timestamp, none

Rules:
- Base every proposal only on the column name and the sample values actually shown to you.
- Do NOT invent a unit, currency, or role that isn't suggested by the name or samples.
- Do NOT calculate anything, and do NOT reference any value not present in the input.
- If you are not confident, propose "unknown" rather than guessing.
- Respond with ONLY valid JSON matching the schema below. Do not include prose, explanation, or markdown formatting outside the JSON object.
- Return exactly one proposal object per input column, using the exact same "name" value, in the same order."""

OUTPUT_SCHEMA_DESCRIPTION = """{
  "columns": [
    {
      "name": string,       // must exactly match an input column name
      "role": "dimension" | "measure" | "date" | "identifier" | "unknown",
      "semantic_type": "numeric" | "currency" | "percentage" | "date" | "text" | "identifier" | "unknown",
      "unit": string or null,
      "time_role": "period" | "timestamp" | "none"
    }
  ]
}"""


def build_semantic_schema_prompt(candidate_columns: list[dict]) -> str:
    """`candidate_columns`: a list of {"name", "dtype", "inferred_type",
    "sample_values", "candidate_role", "candidate_semantic_type",
    "candidate_confidence"} — compact metadata only, no file_id, no full
    dataset."""
    columns_json = json.dumps(candidate_columns, indent=2)

    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"OUTPUT SCHEMA:\n{OUTPUT_SCHEMA_DESCRIPTION}\n\n"
        f"COLUMNS (JSON):\n{columns_json}"
    )
