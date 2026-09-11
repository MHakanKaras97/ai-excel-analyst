from pathlib import Path

import pandas as pd

SUPPORTED_EXTENSIONS = {".xlsx", ".xls"}


def load_excel(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Excel file not found: {path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {path.suffix}")

    return pd.read_excel(path)
