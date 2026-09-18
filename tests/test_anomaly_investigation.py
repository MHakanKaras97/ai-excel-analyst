import pandas as pd

from src.anomaly_investigation import investigate_anomaly


def _series(values, periods=("2024-01", "2024-02", "2024-03", "2024-04")):
    return pd.Series(values, index=list(periods))


def test_investigate_anomaly_reports_own_change():
    series = _series([100.0, 100.0, 250.0, 100.0])
    anomaly = {"position": 2, "period": "2024-03", "value": 250.0, "direction": "high"}

    result = investigate_anomaly(anomaly, "Revenue", series)

    assert result["period"] == "2024-03"
    assert result["column"] == "Revenue"
    assert result["value"] == 250.0
    assert result["direction"] == "high"
    assert result["own_change"]["previous"] == 100.0
    assert result["own_change"]["current"] == 250.0
    assert result["own_change"]["absolute_change"] == 150.0


def test_investigate_anomaly_at_first_period_has_no_previous():
    series = _series([300.0, 100.0, 100.0, 100.0])
    anomaly = {"position": 0, "period": "2024-01", "value": 300.0, "direction": "high"}

    result = investigate_anomaly(anomaly, "Revenue", series)

    assert result["own_change"]["previous"] is None
    assert result["own_change"]["is_valid"] is False
    assert result["own_change"]["reason"] == "no_previous_period"


def test_investigate_anomaly_reports_coincident_changes_without_claiming_causality():
    revenue = _series([100.0, 100.0, 60.0, 100.0])
    units = _series([50.0, 50.0, 30.0, 50.0])
    anomaly = {"position": 2, "period": "2024-03", "value": 60.0, "direction": "low"}

    result = investigate_anomaly(anomaly, "Revenue", revenue, related_series={"Units": units})

    assert len(result["coincident_changes"]) == 1
    coincident = result["coincident_changes"][0]
    assert coincident["column"] == "Units"
    assert coincident["previous"] == 50.0
    assert coincident["current"] == 30.0
    # No causal claim anywhere in the deterministic result — it is only numbers.
    assert "cause" not in str(result).lower()


def test_investigate_anomaly_excludes_the_anomalous_column_itself_from_coincident_changes():
    revenue = _series([100.0, 100.0, 250.0, 100.0])
    anomaly = {"position": 2, "period": "2024-03", "value": 250.0, "direction": "high"}

    result = investigate_anomaly(anomaly, "Revenue", revenue, related_series={"Revenue": revenue})

    assert result["coincident_changes"] == []


def test_investigate_anomaly_handles_missing_related_series_gracefully():
    revenue = _series([100.0, 100.0, 250.0, 100.0])
    anomaly = {"position": 2, "period": "2024-03", "value": 250.0, "direction": "high"}

    result = investigate_anomaly(anomaly, "Revenue", revenue, related_series=None)

    assert result["coincident_changes"] == []


def test_investigate_anomaly_handles_related_series_missing_the_period():
    revenue = _series([100.0, 100.0, 250.0, 100.0])
    shorter = pd.Series([1.0, 2.0], index=["2024-01", "2024-02"])
    anomaly = {"position": 2, "period": "2024-03", "value": 250.0, "direction": "high"}

    result = investigate_anomaly(anomaly, "Revenue", revenue, related_series={"Other": shorter})

    assert result["coincident_changes"] == []


def test_investigate_anomaly_does_not_mutate_series():
    revenue = _series([100.0, 100.0, 250.0, 100.0])
    before = revenue.copy()
    anomaly = {"position": 2, "period": "2024-03", "value": 250.0, "direction": "high"}

    investigate_anomaly(anomaly, "Revenue", revenue)

    assert revenue.equals(before)
