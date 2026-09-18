"""Deterministic evidence validation (V0.8.8).

Nothing here is authored by an LLM — this module only checks that: (a) an
Evidence list is internally well-formed (validate_evidence_list), (b) any
evidence_id an AI response claims to have used actually exists in that list
(validate_referenced_evidence_ids — an existence check only; it does not
itself compare cited numeric values against the evidence), and (c)
AI-generated text does not smuggle in an unsupported causal claim between
two evidenced facts (contains_unsupported_causal_claim). All three checks
are pure and deterministic.

Cited *values* are grounded separately: src/evidence_ai_interpreter.py
reuses ai_interpreter.has_hallucinated_numbers() (the same numeric-
grounding check used by the single-file/comparison AI insight) against the
Evidence payload, so a response citing a number not present in any
evidence item is rejected there, not in this module.
"""
import re

from src.evidence.models import EVIDENCE_TYPES

CAUSAL_PHRASES = (
    "caused", "causes", "causing",
    "because of the increase", "because of the decrease",
    "led to", "leads to",
    "resulted in", "results in",
    "due to the increase", "due to the decrease",
    "driving the", "drove the",
    "is responsible for", "are responsible for",
)


def _structural_error(reason: str, evidence_id=None) -> dict:
    return {"valid": False, "reason": reason, "evidence_id": evidence_id}


def validate_evidence_list(evidence_list: list[dict]) -> dict:
    """Structural validation of a whole Evidence list: unique ids, known
    types, no leaked `file_id`, and no contradictory duplicate id."""
    seen_ids = set()
    for evidence in evidence_list:
        if not isinstance(evidence, dict) or "evidence_id" not in evidence:
            return _structural_error("malformed_evidence")

        evidence_id = evidence["evidence_id"]
        if evidence_id in seen_ids:
            return _structural_error("duplicate_evidence_id", evidence_id)
        seen_ids.add(evidence_id)

        if evidence.get("type") not in EVIDENCE_TYPES:
            return _structural_error("unknown_evidence_type", evidence_id)

        if _contains_file_id(evidence):
            return _structural_error("file_id_leaked", evidence_id)

    return {"valid": True, "reason": None, "evidence_id": None}


def _contains_file_id(value) -> bool:
    if isinstance(value, dict):
        if "file_id" in value:
            return True
        return any(_contains_file_id(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_file_id(v) for v in value)
    return False


def validate_referenced_evidence_ids(referenced_ids: list[str], evidence_list: list[dict]) -> dict:
    """Check that every id in `referenced_ids` (as an AI response claims to
    have used) exists in `evidence_list`. Returns the set of unsupported
    ids — empty means every reference is backed by real evidence."""
    known_ids = {e["evidence_id"] for e in evidence_list if isinstance(e, dict) and "evidence_id" in e}
    unsupported = [ref for ref in referenced_ids if ref not in known_ids]
    return {"valid": len(unsupported) == 0, "unsupported_ids": unsupported}


_CAUSAL_PATTERN = re.compile("|".join(re.escape(phrase) for phrase in CAUSAL_PHRASES), re.IGNORECASE)


def contains_unsupported_causal_claim(text: str) -> bool:
    """Deterministic, conservative causal-language guard: flags common
    causal connective phrases (see CAUSAL_PHRASES) between two facts.
    Heuristic by nature (like the existing numeric-hallucination regex
    checks in ai_interpreter.py) — a phrase-based guard, not a semantic
    understanding of causality, and deliberately errs toward flagging
    borderline phrasing rather than letting an unsupported causal claim
    through silently.
    """
    if not text:
        return False
    return bool(_CAUSAL_PATTERN.search(text))
