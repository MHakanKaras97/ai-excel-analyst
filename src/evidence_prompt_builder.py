"""Prompt builder for the evidence-aware AI insight/Q&A layer (V0.8.14).

Sends only a deterministic Evidence list (see src/evidence/builder.py) plus
the user's question (when there is one) — never a raw DataFrame, never a
`file_id` (Evidence objects never carry one; see evidence/models.py).
"""
import json

SYSTEM_INSTRUCTIONS = """You are a financial data analyst assistant explaining a set of pre-computed, deterministic evidence facts.

Each item in the EVIDENCE list below is an already-verified fact — a value, a change, an anomaly count, or a trend label — computed by Python, not by you. Each item has an "evidence_id".

Rules:
- Only reference numbers, values, or labels that appear verbatim in the EVIDENCE list below.
- Do NOT calculate, estimate, round, derive, or invent any number not already present in a piece of evidence.
- Do NOT invent new evidence, and do NOT reference an evidence_id that is not in the list.
- List, in "evidence_ids_used", the evidence_id of every evidence item your response actually relies on. Do not include an id you did not use.
- If two or more evidence items changed during the same period, you may note that they changed together (e.g. "X also decreased during the same period"). NEVER state or imply that one caused the other, and never use words like "caused", "led to", "resulted in", or "due to" to connect two different evidence items.
- If a forecast evidence item is present, you MUST describe it as a prediction/baseline forecast (e.g. "the baseline forecast is..."), never as a stated fact about the future (never "revenue will be...").
- Respond with ONLY valid JSON matching the schema below. Do not include prose, explanation, or markdown formatting outside the JSON object.
- If the evidence is insufficient to say something meaningful, return an empty list or null for that field instead of guessing."""

OUTPUT_SCHEMA_DESCRIPTION = """{
  "summary": string,  // 1-2 sentences: the single most important takeaway from the evidence
  "key_insights": [string, ...],  // at most 2-3 items, each grounded in one or more evidence items
  "trend_interpretation": string or null,  // how the evidence values relate to each other, in words
  "warnings": [string, ...],  // only data-quality/reliability concerns visible in the evidence
  "recommendations": [string, ...],  // actions tied directly to a specific insight or warning above
  "evidence_ids_used": [string, ...]  // every evidence_id actually relied upon above
}"""


def build_evidence_prompt(payload: dict) -> str:
    """`payload`: {"evidence": [...], "question": str or None}."""
    evidence_json = json.dumps(payload.get("evidence") or [], indent=2)
    question = payload.get("question")
    question_section = f"\n\nQUESTION:\n{question}" if question else ""

    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"OUTPUT SCHEMA:\n{OUTPUT_SCHEMA_DESCRIPTION}\n\n"
        f"EVIDENCE (JSON):\n{evidence_json}{question_section}"
    )
