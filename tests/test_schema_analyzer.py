import pandas as pd

from src.schema.schema_analyzer import analyze_schema


def _column_by_name(schema, name):
    return next(c for c in schema["columns"] if c["name"] == name)


def test_numeric_column_classified_as_measure():
    df = pd.DataFrame({"Revenue": [100.0, 200.0, 300.0]})

    schema = analyze_schema(df)

    column = _column_by_name(schema, "Revenue")
    assert column["role"] == "measure"
    assert column["semantic_type"] == "numeric"
    assert column["confidence"] >= 0.9


def test_date_column_classified_with_period_time_role():
    df = pd.DataFrame({"period": pd.to_datetime(["2024-01-01", "2024-02-01"])})

    schema = analyze_schema(df)

    column = _column_by_name(schema, "period")
    assert column["role"] == "date"
    assert column["semantic_type"] == "date"
    assert column["time_role"] == "period"


def test_date_column_with_time_component_classified_as_timestamp():
    df = pd.DataFrame({"logged_at": pd.to_datetime(["2024-01-01 08:30:00", "2024-01-02 14:00:00"])})

    schema = analyze_schema(df)

    column = _column_by_name(schema, "logged_at")
    assert column["time_role"] == "timestamp"


def test_currency_text_column_classified_as_currency_measure():
    df = pd.DataFrame({"Net Revenue": ["$100,000", "$110,000", "$120,000"]})

    schema = analyze_schema(df)

    column = _column_by_name(schema, "Net Revenue")
    assert column["role"] == "measure"
    assert column["semantic_type"] == "currency"
    assert column["unit"] == "USD"


def test_euro_currency_column_detects_eur_unit():
    df = pd.DataFrame({"Revenue (€)": ["€100", "€200", "€300"]})

    schema = analyze_schema(df)

    column = _column_by_name(schema, "Revenue (€)")
    assert column["unit"] == "EUR"


def test_percentage_text_column_classified_as_percentage_measure():
    df = pd.DataFrame({"Margin": ["10%", "12%", "15%"]})

    schema = analyze_schema(df)

    column = _column_by_name(schema, "Margin")
    assert column["semantic_type"] == "percentage"
    assert column["unit"] == "%"


def test_high_cardinality_text_column_classified_as_identifier():
    df = pd.DataFrame({"OrderID": [f"ORD-{i}" for i in range(20)]})

    schema = analyze_schema(df)

    column = _column_by_name(schema, "OrderID")
    assert column["role"] == "identifier"


def test_low_cardinality_text_column_classified_as_dimension():
    df = pd.DataFrame({"Region": ["North", "South", "North", "South", "East"]})

    schema = analyze_schema(df)

    column = _column_by_name(schema, "Region")
    assert column["role"] == "dimension"
    assert column["semantic_type"] == "text"


def test_empty_dataframe_columns_classified_as_unknown_without_raising():
    df = pd.DataFrame({"Empty": pd.Series([], dtype="object")})

    schema = analyze_schema(df)

    column = _column_by_name(schema, "Empty")
    assert column["role"] == "unknown"


def test_reuses_precomputed_profile_when_given():
    from src.data_profiler import profile_dataframe

    df = pd.DataFrame({"Revenue": [100.0, 200.0]})
    profile = profile_dataframe(df)

    schema = analyze_schema(df, profile=profile)

    assert _column_by_name(schema, "Revenue")["role"] == "measure"


def test_analyze_schema_covers_every_column_exactly_once():
    df = pd.DataFrame({
        "Revenue": [100.0, 200.0],
        "Region": ["North", "South"],
        "period": pd.to_datetime(["2024-01-01", "2024-02-01"]),
    })

    schema = analyze_schema(df)

    assert [c["name"] for c in schema["columns"]] == ["Revenue", "Region", "period"]


def test_analyze_schema_does_not_mutate_input_dataframe():
    df = pd.DataFrame({"Revenue": [100.0, 200.0]})
    before = df.copy()

    analyze_schema(df)

    assert df.equals(before)
