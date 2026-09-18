"""Evidence-aware Q&A orchestration (V0.8.15).

A thin composition layer over the existing, unmodified Q&A pipeline
(qa_interpreter -> qa_engine -> qa_answer) and the multi-file comparison
pipeline (multi_file_qa_interpreter -> multi_file_comparison -> qa_answer):
it adds an Evidence bundle and an optional AI explanation on top, without
changing how the deterministic answer itself is produced.

The deterministic answer is always computed first and returned regardless
of what happens next — a failed or skipped AI explanation never blanks out
or replaces it. This is what keeps deterministic Q&A independent of AI
prose generation, per the V0.8 architecture rule.
"""
from src.evidence.builder import build_evidence_from_comparison_result, build_evidence_from_grounded_result
from src.evidence_ai_interpreter import interpret_with_evidence
from src.multi_file_comparison import dispatch_comparison_intent
from src.multi_file_qa_interpreter import interpret_multi_file_question
from src.qa_answer import answer_comparison_result, answer_grounded_result
from src.qa_engine import dispatch_intent
from src.qa_interpreter import interpret_question


def answer_question_with_evidence(question: str, column_names: list[str], provider,
                                   analysis_payload: dict, monthly_series=None, file_label: str | None = None) -> dict:
    """Single-file Q&A with an added Evidence bundle and AI explanation.

    Returns {"deterministic_answer", "grounded_result", "evidence",
    "ai_explanation"}. `ai_explanation` is None whenever there is no
    evidence to explain (interpretation/resolution failed) or explaining it
    was not attempted; a failed AI explanation is a dict with
    `is_valid: False`, never an exception, and never touches
    `deterministic_answer`.
    """
    interpretation = interpret_question(question, column_names, provider)
    if not interpretation["is_valid"]:
        return {
            "deterministic_answer": answer_grounded_result(interpretation),
            "grounded_result": interpretation,
            "evidence": [],
            "ai_explanation": None,
        }

    grounded_result = dispatch_intent(interpretation["intent"], analysis_payload, monthly_series)
    deterministic_answer = answer_grounded_result(grounded_result)
    evidence = build_evidence_from_grounded_result(grounded_result, file_label=file_label)

    ai_explanation = interpret_with_evidence(evidence, provider, question=question) if evidence else None

    return {
        "deterministic_answer": deterministic_answer,
        "grounded_result": grounded_result,
        "evidence": evidence,
        "ai_explanation": ai_explanation,
    }


def answer_comparison_with_evidence(question: str, column_names: list[str], file_labels: list[str],
                                     provider, data_files: dict) -> dict:
    """Multi-file comparison Q&A with an added Evidence bundle and AI
    explanation, mirroring answer_question_with_evidence()'s shape and
    independence guarantee.
    """
    interpretation = interpret_multi_file_question(question, column_names, file_labels, provider)
    if not interpretation["is_valid"]:
        return {
            "deterministic_answer": answer_comparison_result(interpretation),
            "grounded_result": interpretation,
            "evidence": [],
            "ai_explanation": None,
        }

    comparison_result = dispatch_comparison_intent(interpretation["intent"], data_files)
    deterministic_answer = answer_comparison_result(comparison_result)
    evidence = build_evidence_from_comparison_result(comparison_result)

    ai_explanation = interpret_with_evidence(evidence, provider, question=question) if evidence else None

    return {
        "deterministic_answer": deterministic_answer,
        "grounded_result": comparison_result,
        "evidence": evidence,
        "ai_explanation": ai_explanation,
    }
