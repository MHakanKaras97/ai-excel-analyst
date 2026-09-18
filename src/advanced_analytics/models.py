"""Shared constants/contract helpers for the advanced analytics metrics
(V0.8.9). Every metric function returns a fully-keyed dict with an explicit
"insufficient_data" flag rather than a silently wrong or fabricated value
when the input history is too short.
"""

DEFAULT_ROLLING_WINDOW = 3
MIN_VOLATILITY_OBSERVATIONS = 2
MIN_SEASONALITY_OBSERVATIONS = 24  # at least two full years of monthly data


def insufficient_result(**fields) -> dict:
    """Build a result dict with insufficient_data=True and every other
    field explicitly set to None/empty, so callers never have to guess
    which fields are meaningful when data was insufficient."""
    return {"insufficient_data": True, **fields}
