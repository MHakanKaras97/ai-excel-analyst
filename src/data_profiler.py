import pandas as pd

DATE_PARSE_THRESHOLD = 0.8


def _infer_type(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "date-like"

    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        non_null = series.dropna()
        if len(non_null) > 0:
            parsed = pd.to_datetime(non_null, errors="coerce")
            if parsed.notna().sum() / len(non_null) >= DATE_PARSE_THRESHOLD:
                return "date-like"

    return "text"


def _column_profile(df: pd.DataFrame, position: int, row_count: int) -> dict:
    series = df.iloc[:, position]
    missing_count = int(series.isna().sum())
    missing_percentage = 0.0 if row_count == 0 else (missing_count / row_count) * 100

    return {
        "name": str(df.columns[position]),
        "dtype": str(series.dtype),
        "inferred_type": _infer_type(series),
        "unique_count": int(series.nunique()),
        "missing_count": missing_count,
        "missing_percentage": missing_percentage,
    }


def profile_dataframe(df: pd.DataFrame) -> dict:
    row_count = len(df)
    column_count = len(df.columns)

    dup_column_mask = df.columns.duplicated(keep=False)
    duplicate_column_names = sorted(set(df.columns[dup_column_mask].astype(str)))

    return {
        "row_count": row_count,
        "column_count": column_count,
        "is_empty": row_count == 0,
        "duplicate_row_count": int(df.duplicated().sum()),
        "duplicate_column_names": duplicate_column_names,
        "columns": [
            _column_profile(df, i, row_count) for i in range(column_count)
        ],
    }
