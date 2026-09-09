import pandas as pd
import pytest

from complaints.ingest import REQUIRED_COLUMNS, audit, load_raw

ROWS = [
    [
        "C1",
        "1/1/2023",
        "Mortgage",
        "I_1",
        "My mortgage payment was lost.",
        "Texas",
        "75001",
        "Closed",
    ],
    ["C2", "2/5/2023", "Credit card", "I_2", "  Charged twice XXXX {$12.00} ", "", "nan", "Closed"],
    ["C3", "3/9/2023", "Credit card", "I_2", "Please see attached", "N/A", "  ", "Closed"],
]


def _write_csv(tmp_path):
    path = tmp_path / "raw.csv"
    pd.DataFrame(ROWS, columns=list(REQUIRED_COLUMNS)).to_csv(path, index=False)
    return path


def test_load_raw_strips_and_normalises_blanks_to_nan(tmp_path) -> None:
    df = load_raw(_write_csv(tmp_path))

    assert df.shape == (3, 8)
    assert df["Complaint Description"].iloc[1] == "Charged twice XXXX {$12.00}"
    assert df["State"].isna().tolist() == [False, True, True]
    assert df["ZIP"].isna().tolist() == [False, True, True]
    assert str(df["Date Received"].dtype).startswith("datetime64")


def test_load_raw_requires_columns(tmp_path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame({"Complaint Description": ["a"]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        load_raw(path)


def test_load_raw_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_raw("does/not/exist.csv")


def test_audit_reports_shape_nulls_and_text_stats(tmp_path) -> None:
    report = audit(load_raw(_write_csv(tmp_path)))

    assert report["shape"] == {"rows": 3, "columns": 8}
    assert report["nulls"]["State"] == 2
    assert report["duplicates"]["complaint_id"] == 0
    assert report["labels"] == {"Credit card": 2, "Mortgage": 1}
    assert report["text"]["with_redaction"] == 1
    assert report["text"]["with_money_mask"] == 1
    assert report["text"]["under_10_words"] == 3
    assert report["date_range"] == {"min": "2023-01-01", "max": "2023-03-09", "unparsed": 0}
