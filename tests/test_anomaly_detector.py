import json
from datetime import date

import numpy as np
import pandas as pd

from src.anomaly_detector import detect_iqr_anomalies


def test_detect_iqr_anomalies_normal_data_has_no_anomalies():
    values = pd.Series([8, 9, 10, 11, 12, 10, 9])

    result = detect_iqr_anomalies(values)

    assert result["method"] == "iqr"
    assert result["insufficient_data"] is False
    assert result["anomaly_count"] == 0
    assert result["anomalies"] == []


def test_detect_iqr_anomalies_detects_high_anomaly():
    values = pd.Series([10, 11, 12, 13, 100])

    result = detect_iqr_anomalies(values)

    assert result["anomaly_count"] == 1
    assert result["anomalies"][0]["value"] == 100
    assert result["anomalies"][0]["direction"] == "high"
    assert result["anomalies"][0]["position"] == 4


def test_detect_iqr_anomalies_detects_low_anomaly():
    values = pd.Series([10, 11, 12, 13, -100])

    result = detect_iqr_anomalies(values)

    assert result["anomaly_count"] == 1
    assert result["anomalies"][0]["value"] == -100
    assert result["anomalies"][0]["direction"] == "low"
    assert result["anomalies"][0]["position"] == 4


def test_detect_iqr_anomalies_detects_multiple_anomalies():
    values = pd.Series([10, 11, 12, 13, 100, -100])

    result = detect_iqr_anomalies(values)

    assert result["anomaly_count"] == 2
    directions = {a["direction"] for a in result["anomalies"]}
    assert directions == {"high", "low"}


def test_detect_iqr_anomalies_ignores_missing_values():
    values = pd.Series([10, None, 11, 12, 13, 100])

    result = detect_iqr_anomalies(values)

    assert result["sample_size"] == 5
    assert all(a["value"] is not None for a in result["anomalies"])
    assert result["anomaly_count"] == 1
    assert result["anomalies"][0]["value"] == 100


def test_detect_iqr_anomalies_insufficient_data_returns_empty_result():
    values = pd.Series([10, 20, 30])

    result = detect_iqr_anomalies(values)

    assert result["insufficient_data"] is True
    assert result["anomalies"] == []
    assert result["anomaly_count"] == 0
    assert result["lower_bound"] is None
    assert result["upper_bound"] is None
    assert result["sample_size"] == 3


def test_detect_iqr_anomalies_constant_series_has_no_anomalies():
    values = pd.Series([5, 5, 5, 5, 5])

    result = detect_iqr_anomalies(values)

    assert result["insufficient_data"] is False
    assert result["lower_bound"] == 5.0
    assert result["upper_bound"] == 5.0
    assert result["anomaly_count"] == 0


def test_detect_iqr_anomalies_supports_duplicate_index_values():
    values = pd.Series([10, 11, 12, 13, 100], index=["Q1", "Q1", "Q1", "Q1", "Q1"])

    result = detect_iqr_anomalies(values)

    assert result["anomaly_count"] == 1
    anomaly = result["anomalies"][0]
    assert anomaly["period"] == "Q1"
    assert anomaly["position"] == 4


def test_detect_iqr_anomalies_preserves_input_order():
    values = pd.Series([100, 10, 11, 12, 13, -100])

    result = detect_iqr_anomalies(values)

    positions = [a["position"] for a in result["anomalies"]]
    assert positions == sorted(positions)
    assert positions == [0, 5]


def test_detect_iqr_anomalies_does_not_mutate_input_series():
    values = pd.Series([10, None, 11, 12, 13, 100])
    before = values.copy(deep=True)

    detect_iqr_anomalies(values)

    pd.testing.assert_series_equal(values, before)


def test_detect_iqr_anomalies_output_is_json_friendly():
    values = pd.Series([10, 11, 12, 13, 100])

    result = detect_iqr_anomalies(values)

    json.dumps(result)  # must not raise
    assert isinstance(result["anomalies"][0]["value"], (int, float))
    assert isinstance(result["anomalies"][0]["position"], int)
    assert isinstance(result["lower_bound"], float)
    assert isinstance(result["upper_bound"], float)
    assert isinstance(result["sample_size"], int)


def test_detect_iqr_anomalies_non_numeric_input_returns_insufficient_data():
    values = pd.Series(["a", "b", "c", "d", "e"])

    result = detect_iqr_anomalies(values)

    assert result["insufficient_data"] is True
    assert result["anomalies"] == []
    assert result["anomaly_count"] == 0
    json.dumps(result)  # must not raise


def test_detect_iqr_anomalies_sample_size_counts_only_non_null_numeric_values():
    values = pd.Series([10, 20, None, 30, 40])

    result = detect_iqr_anomalies(values)

    assert result["sample_size"] == 4


def test_detect_iqr_anomalies_sample_size_is_zero_for_non_numeric_series():
    values = pd.Series(["a", "b", "c"])

    result = detect_iqr_anomalies(values)

    assert result["sample_size"] == 0
    assert result["insufficient_data"] is True


def test_detect_iqr_anomalies_sample_size_reflects_usable_count_when_insufficient():
    values = pd.Series([10, None, None, 20])

    result = detect_iqr_anomalies(values)

    assert result["insufficient_data"] is True
    assert result["sample_size"] == 2


def test_detect_iqr_anomalies_converts_timestamp_period_labels_to_iso_strings():
    values = pd.Series(
        [10, 11, 12, 13, 100],
        index=pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01"]),
    )

    result = detect_iqr_anomalies(values)

    assert result["anomalies"][0]["period"] == "2023-05-01T00:00:00"
    assert isinstance(result["anomalies"][0]["period"], str)
    json.dumps(result)  # must not raise


def test_detect_iqr_anomalies_converts_python_date_period_labels_to_iso_strings():
    values = pd.Series(
        [10, 11, 12, 13, 100],
        index=[date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1), date(2023, 4, 1), date(2023, 5, 1)],
    )

    result = detect_iqr_anomalies(values)

    assert result["anomalies"][0]["period"] == "2023-05-01T00:00:00"
    json.dumps(result)  # must not raise


def test_detect_iqr_anomalies_converts_numpy_datetime64_period_labels_to_iso_strings():
    values = pd.Series(
        [10, 11, 12, 13, 100],
        index=pd.Index([
            np.datetime64("2023-01-01"),
            np.datetime64("2023-02-01"),
            np.datetime64("2023-03-01"),
            np.datetime64("2023-04-01"),
            np.datetime64("2023-05-01"),
        ]),
    )

    result = detect_iqr_anomalies(values)

    assert result["anomalies"][0]["period"] == "2023-05-01T00:00:00"
    json.dumps(result)  # must not raise
