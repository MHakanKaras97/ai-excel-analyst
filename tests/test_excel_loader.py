from pathlib import Path

import pandas as pd
import pytest

from src.excel_loader import load_excel


def test_load_excel_reads_xlsx_into_dataframe(tmp_path: Path) -> None:
    file_path = tmp_path / "data.xlsx"
    expected = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    expected.to_excel(file_path, index=False)

    result = load_excel(file_path)

    pd.testing.assert_frame_equal(result, expected)


def test_load_excel_raises_filenotfounderror_for_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.xlsx"

    with pytest.raises(FileNotFoundError):
        load_excel(missing_path)


def test_load_excel_raises_valueerror_for_unsupported_extension(tmp_path: Path) -> None:
    file_path = tmp_path / "data.txt"
    file_path.write_text("not an excel file")

    with pytest.raises(ValueError):
        load_excel(file_path)


def test_load_excel_dispatches_xls_files_to_pandas(tmp_path, monkeypatch) -> None:
    file_path = tmp_path / "data.xls"
    file_path.touch()
    expected = pd.DataFrame({"a": [1]})

    def fake_read_excel(path):
        assert Path(path) == file_path
        return expected

    monkeypatch.setattr(pd, "read_excel", fake_read_excel)

    result = load_excel(file_path)

    pd.testing.assert_frame_equal(result, expected)
