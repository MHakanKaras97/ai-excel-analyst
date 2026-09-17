from src.qa_engine import resolve_column, resolve_period

# ==================================================
# COLUMN RESOLUTION
# ==================================================


def test_resolve_column_exact_match():
    result = resolve_column("Revenue", ["Revenue", "Region"])

    assert result == {"found": True, "reason": None, "column": "Revenue", "candidates": []}


def test_resolve_column_case_insensitive_exact_match():
    result = resolve_column("revenue", ["Revenue", "Region"])

    assert result["found"] is True
    assert result["reason"] is None
    assert result["column"] == "Revenue"
    assert result["candidates"] == []


def test_resolve_column_duplicate_exact_matches_is_ambiguous():
    result = resolve_column("revenue", ["Revenue", "REVENUE"])

    assert result["found"] is False
    assert result["reason"] == "ambiguous_column"
    assert result["column"] is None
    assert set(result["candidates"]) == {"Revenue", "REVENUE"}


def test_resolve_column_obvious_fuzzy_single_match_resolves():
    # "Revenu" is a one-character-short typo of "Revenue" — close enough to
    # resolve automatically, with no other column close enough to compete.
    result = resolve_column("Revenu", ["Revenue", "Region"])

    assert result["found"] is True
    assert result["reason"] is None
    assert result["column"] == "Revenue"
    assert result["candidates"] == []


def test_resolve_column_weak_fuzzy_reference_does_not_resolve():
    # "Rev" is a short, weak partial reference — below the conservative
    # cutoff, so it must not be silently resolved to "Revenue".
    result = resolve_column("Rev", ["Revenue", "Region"])

    assert result["found"] is False
    assert result["reason"] == "column_not_found"
    assert result["column"] is None
    assert result["candidates"] == []


def test_resolve_column_multiple_plausible_fuzzy_candidates_is_ambiguous():
    result = resolve_column("Revenue1", ["RevenueA", "RevenueB"])

    assert result["found"] is False
    assert result["reason"] == "ambiguous_column"
    assert result["column"] is None
    assert set(result["candidates"]) == {"RevenueA", "RevenueB"}


def test_resolve_column_no_fuzzy_candidates_is_not_found():
    result = resolve_column("xyz123", ["Revenue", "Region"])

    assert result == {"found": False, "reason": "column_not_found", "column": None, "candidates": []}


def test_resolve_column_empty_column_list():
    result = resolve_column("Revenue", [])

    assert result == {"found": False, "reason": "column_not_found", "column": None, "candidates": []}


def test_resolve_column_does_not_mutate_input_list():
    columns = ["Revenue", "Region"]
    before = list(columns)

    resolve_column("Revenue", columns)

    assert columns == before


# ==================================================
# PERIOD RESOLUTION
# ==================================================


def test_resolve_period_exact_yyyy_mm():
    result = resolve_period("2024-03", ["2024-01", "2024-02", "2024-03"])

    assert result == {"found": True, "reason": None, "period": "2024-03", "candidates": []}


def test_resolve_period_bare_month_with_exactly_one_year():
    result = resolve_period("March", ["2024-01", "2024-02", "2024-03"])

    assert result["found"] is True
    assert result["reason"] is None
    assert result["period"] == "2024-03"
    assert result["candidates"] == []


def test_resolve_period_bare_month_across_multiple_years_is_ambiguous():
    result = resolve_period("March", ["2023-03", "2024-03", "2024-05"])

    assert result["found"] is False
    assert result["reason"] == "ambiguous_period"
    assert result["period"] is None
    assert result["candidates"] == ["2023-03", "2024-03"]


def test_resolve_period_bare_month_with_no_match():
    result = resolve_period("December", ["2024-01", "2024-02", "2024-03"])

    assert result == {"found": False, "reason": "period_not_found", "period": None, "candidates": []}


def test_resolve_period_month_name_and_year():
    result = resolve_period("March 2024", ["2024-01", "2024-02", "2024-03"])

    assert result["found"] is True
    assert result["period"] == "2024-03"


def test_resolve_period_abbreviated_month_name_and_year():
    result = resolve_period("Mar 2024", ["2024-01", "2024-02", "2024-03"])

    assert result["found"] is True
    assert result["period"] == "2024-03"


def test_resolve_period_numeric_month_and_year():
    result = resolve_period("03 2024", ["2024-01", "2024-02", "2024-03"])

    assert result["found"] is True
    assert result["period"] == "2024-03"


def test_resolve_period_month_and_year_not_present_is_not_found():
    result = resolve_period("March 2025", ["2024-01", "2024-02", "2024-03"])

    assert result == {"found": False, "reason": "period_not_found", "period": None, "candidates": []}


def test_resolve_period_ambiguity_never_defaults_to_latest_or_first_year():
    result = resolve_period("March", ["2022-03", "2023-03", "2024-03"])

    assert result["found"] is False
    assert result["period"] is None
    # All candidate years must be surfaced, not silently narrowed to one.
    assert result["candidates"] == ["2022-03", "2023-03", "2024-03"]


def test_resolve_period_does_not_mutate_input_list():
    periods = ["2024-01", "2024-02", "2024-03"]
    before = list(periods)

    resolve_period("March", periods)

    assert periods == before


def test_resolve_period_empty_period_list():
    result = resolve_period("March", [])

    assert result == {"found": False, "reason": "period_not_found", "period": None, "candidates": []}
