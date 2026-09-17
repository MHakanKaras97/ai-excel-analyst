"""Shapes a multi_file_comparison Comparison Result into the payload the AI
Comparison Insight layer (V0.7.4) is allowed to see.

The only transformation applied is removing each file's internal `file_id`
(an opaque identifier with no narrative value, and a possible source of
spurious digit runs) from `file_a`/`file_b` — everything else needed to
usefully interpret the comparison (display_name, role, and every
comparison fact itself) is preserved verbatim. No calculation, no derived
values, no invented metadata.
"""


def _public_file_identity(file_identity):
    if not file_identity:
        return None
    return {
        "display_name": file_identity.get("display_name"),
        "role": file_identity.get("role"),
    }


def build_comparison_ai_payload(comparison_result: dict) -> dict:
    """Return a copy of `comparison_result` with `file_id` removed from
    `file_a`/`file_b`. Does not mutate the input `comparison_result`.

    Safe for both a successful (`found: True`) and a failure
    (`found: False`) result — `file_a`/`file_b` may legitimately be `None`
    in a failure result (e.g. the file hints never resolved), which is
    preserved as `None` rather than raising.
    """
    payload = dict(comparison_result)
    payload["file_a"] = _public_file_identity(comparison_result.get("file_a"))
    payload["file_b"] = _public_file_identity(comparison_result.get("file_b"))
    return payload
