"""Semantic column resolution and LLM proposal validation (V0.8.5/V0.8.6).

Two independent responsibilities live here:

1. `resolve_semantic_column()` — resolve a free-text business term (e.g.
   "total sales") against a dataset's columns when the existing exact/
   case-insensitive/fuzzy resolution in qa_engine.resolve_column() finds
   nothing. This is a fallback layered ON TOP of resolve_column(), not a
   replacement — resolve_column() is tried first and its result (found, or
   an ambiguity) is always honored as-is, so all existing V0.7 resolution
   behavior is unchanged. Semantic scoring is deliberately conservative
   (small keyword-expansion table, no embeddings/ML) and only ever resolves
   when exactly one candidate clears the high-confidence band with no other
   candidate also clearing it — anything less is reported ambiguous rather
   than guessed.

2. `merge_llm_schema_proposal()` — validate an LLM's proposed role/
   semantic_type/unit for one column against the deterministic candidate
   schema (V0.8.5). The LLM proposal is never authoritative: an invalid
   enum value, a proposal for a column that doesn't exist, or a proposal
   that contradicts strong deterministic evidence is rejected/overridden by
   Python rather than accepted verbatim.
"""
import re

from src.qa_engine import resolve_column
from src.schema.models import COLUMN_ROLES, SEMANTIC_TYPES, TIME_ROLES

HIGH_CONFIDENCE = 0.6
MEDIUM_CONFIDENCE = 0.3

# Deliberately small, explicit synonym groups — not a general ontology.
# Each hint token is expanded to its group (including itself) before being
# compared against a column's own tokens.
SYNONYM_GROUPS = [
    {"sales", "revenue"},
    {"total", "sum", "amount"},
    {"net"},
    {"gross"},
    {"units", "quantity", "qty", "count"},
    {"price", "cost"},
    {"profit", "margin"},
]

STOPWORDS = {"the", "a", "an", "of", "for", "in", "on"}

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

# Only strong deterministic evidence (dtype-certain classifications) blocks
# an LLM override; heuristic text-based classifications (currency/
# percentage/identifier/dimension guesses) may be corrected by a proposal.
_STRONG_CONFIDENCE_FLOOR = 0.85


def _tokenize(text: str) -> set:
    tokens = set(_TOKEN_PATTERN.findall(text.lower()))
    return tokens - STOPWORDS


def _expand_synonyms(tokens: set) -> set:
    expanded = set(tokens)
    for token in tokens:
        for group in SYNONYM_GROUPS:
            if token in group:
                expanded |= group
    return expanded


def _score_column(hint_tokens: set, hint_tokens_expanded: set, column_tokens: set) -> tuple[float, bool]:
    """Return (dice_score, has_literal_overlap).

    The Dice denominator uses the *original* (unexpanded) hint token count
    so synonym expansion only ever widens which columns are found, never
    dilutes the score of a column that already shares a literal word with
    the hint. `has_literal_overlap` is True only when at least one token is
    shared without needing synonym expansion — a purely synonym-derived
    match (e.g. "amount" -> "Total Price" via the total/sum/amount group)
    is never confident enough to auto-resolve on its own, since it is one
    inference step removed from what the user actually typed.
    """
    if not hint_tokens or not column_tokens:
        return 0.0, False
    expanded_overlap = hint_tokens_expanded & column_tokens
    if not expanded_overlap:
        return 0.0, False
    # Capped at 1.0: using the *original* hint token count in the
    # denominator (see the docstring above) means synonym expansion can
    # occasionally match more distinct tokens than the hint itself had,
    # which would otherwise push this above the well-formed Dice range.
    dice = min(1.0, 2 * len(expanded_overlap) / (len(hint_tokens) + len(column_tokens)))
    has_literal_overlap = bool(hint_tokens & column_tokens)
    return dice, has_literal_overlap


def _semantic_result(found: bool, reason, column=None, confidence=None, candidates=None) -> dict:
    return {
        "found": found,
        "reason": reason,
        "column": column,
        "confidence": confidence,
        "candidates": candidates or [],
    }


def resolve_semantic_column(hint: str, schema_columns: list[dict]) -> dict:
    """Resolve `hint` against `schema_columns` (a list of ColumnSchema-shaped
    dicts, at minimum each needing a "name"). Tries exact/case-insensitive/
    fuzzy resolution first (qa_engine.resolve_column) and only falls back to
    conservative keyword-overlap scoring if that finds nothing at all.
    """
    column_names = [c["name"] for c in schema_columns]
    exact = resolve_column(hint, column_names)
    if exact["found"]:
        return _semantic_result(True, None, column=exact["column"], confidence=1.0)
    if exact["reason"] == "ambiguous_column":
        return _semantic_result(False, "ambiguous_column", candidates=exact["candidates"])

    if not hint or not hint.strip():
        return _semantic_result(False, "column_not_found")

    raw_hint_tokens = _tokenize(hint)
    if not raw_hint_tokens:
        return _semantic_result(False, "column_not_found")
    expanded_hint_tokens = _expand_synonyms(raw_hint_tokens)

    scored = []
    for name in column_names:
        score, has_literal_overlap = _score_column(raw_hint_tokens, expanded_hint_tokens, _tokenize(name))
        if score > 0:
            scored.append((name, round(score, 4), has_literal_overlap))
    scored.sort(key=lambda item: item[1], reverse=True)

    # Deliberate "resolve don't guess" trade-off: this is a count check
    # (exactly one candidate), never a margin check. Two candidates both
    # clearing HIGH_CONFIDENCE remain ambiguous even when one clearly
    # outscores the other (e.g. 1.0 vs. 0.667) — a large score gap between
    # two lexically plausible columns is not proof the higher one is
    # actually what the user meant, so it is surfaced for confirmation
    # rather than auto-picked. See
    # test_high_margin_between_two_high_confidence_candidates_is_still_ambiguous_by_design.
    high_with_literal = [item for item in scored if item[1] >= HIGH_CONFIDENCE and item[2]]
    plausible = [item for item in scored if item[1] >= MEDIUM_CONFIDENCE]

    if len(high_with_literal) == 1:
        name, score, _ = high_with_literal[0]
        return _semantic_result(True, None, column=name, confidence=score)

    if plausible:
        return _semantic_result(
            False, "ambiguous_column",
            candidates=[{"column": name, "confidence": score} for name, score, _ in plausible],
        )

    return _semantic_result(False, "column_not_found")


def _rejected(field_name: str, reason: str) -> dict:
    return {"accepted": False, "field": field_name, "reason": reason}


def _accepted(field_name: str, value) -> dict:
    return {"accepted": True, "field": field_name, "value": value}


def merge_llm_schema_proposal(candidate_column: dict, proposal: dict) -> dict:
    """Validate an LLM proposal `{role, semantic_type, unit, time_role}` for
    one column against its deterministic candidate ColumnSchema and return a
    merged ColumnSchema plus a per-field decision log.

    Every field is gated by the same two independent checks, applied
    identically to all four fields (including `unit` — it is a free string
    rather than a closed enum like the other three, so its own check is a
    type check instead of a vocabulary check, but the strong-evidence gate
    below applies to it exactly the same way):

    1. The proposed value must pass its own validity check (closed
       vocabulary for role/semantic_type/time_role; a plain string for
       unit).
    2. If the deterministic candidate's own confidence is at or above
       `_STRONG_CONFIDENCE_FLOOR` (i.e. it is a dtype-certain fact, not a
       heuristic guess) AND the proposal disagrees with the candidate's
       current value for that field, the proposal is rejected as
       "contradicts_strong_evidence" — a strong deterministic classification
       is never overridden by the LLM, for any field.

    Never mutates `candidate_column`.
    """
    decisions = []
    merged = dict(candidate_column)
    is_strong = candidate_column["confidence"] >= _STRONG_CONFIDENCE_FLOOR

    proposed_role = proposal.get("role")
    if proposed_role is not None:
        if proposed_role not in COLUMN_ROLES:
            decisions.append(_rejected("role", "invalid_value"))
        elif is_strong and proposed_role != candidate_column["role"]:
            decisions.append(_rejected("role", "contradicts_strong_evidence"))
        else:
            merged["role"] = proposed_role
            decisions.append(_accepted("role", proposed_role))

    proposed_semantic_type = proposal.get("semantic_type")
    if proposed_semantic_type is not None:
        if proposed_semantic_type not in SEMANTIC_TYPES:
            decisions.append(_rejected("semantic_type", "invalid_value"))
        elif is_strong and proposed_semantic_type != candidate_column["semantic_type"]:
            decisions.append(_rejected("semantic_type", "contradicts_strong_evidence"))
        else:
            merged["semantic_type"] = proposed_semantic_type
            decisions.append(_accepted("semantic_type", proposed_semantic_type))

    proposed_unit = proposal.get("unit")
    if proposed_unit is not None:
        if not isinstance(proposed_unit, str):
            decisions.append(_rejected("unit", "invalid_value"))
        elif is_strong and proposed_unit != candidate_column["unit"]:
            decisions.append(_rejected("unit", "contradicts_strong_evidence"))
        else:
            merged["unit"] = proposed_unit
            decisions.append(_accepted("unit", proposed_unit))

    proposed_time_role = proposal.get("time_role")
    if proposed_time_role is not None:
        if proposed_time_role not in TIME_ROLES:
            decisions.append(_rejected("time_role", "invalid_value"))
        elif is_strong and proposed_time_role != candidate_column["time_role"]:
            decisions.append(_rejected("time_role", "contradicts_strong_evidence"))
        else:
            merged["time_role"] = proposed_time_role
            decisions.append(_accepted("time_role", proposed_time_role))

    return {"schema": merged, "decisions": decisions}


def build_candidate_column_metadata(dataset_schema: dict) -> list[dict]:
    """Compact, LLM-facing metadata for each column of a deterministic
    candidate DatasetSchema (V0.8.5) — name, sample values, and the
    candidate's own classification, so the LLM can propose corrections
    without ever seeing the raw dataset or any internal identifier."""
    metadata = []
    for column in dataset_schema["columns"]:
        evidence = column.get("evidence") or {}
        metadata.append({
            "name": column["name"],
            "dtype": evidence.get("dtype"),
            "sample_values": evidence.get("sample_values") or [],
            "candidate_role": column["role"],
            "candidate_semantic_type": column["semantic_type"],
            "candidate_confidence": column["confidence"],
        })
    return metadata


def apply_semantic_proposals(dataset_schema: dict, proposals_by_name: dict) -> dict:
    """Apply validated LLM proposals (keyed by column name, as returned by
    semantic_interpreter.interpret_semantic_schema) across a whole
    DatasetSchema via merge_llm_schema_proposal(), column by column. A
    column with no proposal is passed through unchanged. Does not mutate
    `dataset_schema`.
    """
    merged_columns = []
    all_decisions = {}
    for column in dataset_schema["columns"]:
        proposal = proposals_by_name.get(column["name"])
        if proposal is None:
            merged_columns.append(column)
            continue
        result = merge_llm_schema_proposal(column, proposal)
        merged_columns.append(result["schema"])
        all_decisions[column["name"]] = result["decisions"]
    return {"schema": {"columns": merged_columns}, "decisions": all_decisions}
