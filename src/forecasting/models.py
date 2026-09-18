"""Forecast result contract (V0.8.10).

Every forecast entry is explicitly marked `is_forecast: True` — a
prediction, never a historical fact — and carries a `bounds_available`
flag: when the available history is too short to defensibly estimate a
residual spread, `lower_bound`/`upper_bound` are `None` rather than a
fabricated interval.
"""

MIN_RESIDUALS_FOR_BOUNDS = 2
CONFIDENCE_Z = 1.96  # ~95% interval under a normal residual approximation


def make_forecast_entry(period: str, forecast: float, method: str,
                         bounds_available: bool, lower_bound=None, upper_bound=None) -> dict:
    return {
        "period": period,
        "forecast": forecast,
        "method": method,
        "is_forecast": True,
        "bounds_available": bounds_available,
        "lower_bound": lower_bound if bounds_available else None,
        "upper_bound": upper_bound if bounds_available else None,
    }
