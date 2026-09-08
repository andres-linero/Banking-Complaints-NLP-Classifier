"""Stage 1: ingest the raw complaints CSV into a validated pandas DataFrame.

This module only loads, types, and audits the raw data. It makes no modeling
decisions: no label mapping, no rare-class grouping, no text cleaning. Those
belong to later stages so that every decision downstream is made on top of an
honest picture of what the raw data actually contains.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from complaints.config import RAW_DATA_PATH, TARGET_COLUMN, TEXT_COLUMN

ID_COLUMN = "Complaint ID"
DATE_COLUMN = "Date Received"
ISSUE_COLUMN = "Issue ID"
STATE_COLUMN = "State"
ZIP_COLUMN = "ZIP"
RESPONSE_COLUMN = "Bank Response"

REQUIRED_COLUMNS = (
    ID_COLUMN,
    DATE_COLUMN,
    TARGET_COLUMN,
    ISSUE_COLUMN,
    TEXT_COLUMN,
    STATE_COLUMN,
    ZIP_COLUMN,
    RESPONSE_COLUMN,
)

# Placeholders some exports use for missing values, on top of the pandas defaults
# (empty cell, nan, NULL, None, N/A, ...).
EXTRA_NA_VALUES = ["-", "?"]

# Pattern for the anonymized tokens the source uses: XXXX, XX/XX/XXXX, etc.
REDACTION_PATTERN = r"X{2,}"


def load_raw(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV and enforce column presence and types.

    Every column is read as a string. pandas already maps empty cells and the
    usual NA spellings to NaN; on top of that, whitespace is stripped and cells
    that were only whitespace become NaN too. The date column is parsed to
    datetime. Nothing is dropped here.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find data file: {path}")

    df = pd.read_csv(path, dtype=str, na_values=EXTRA_NA_VALUES)

    missing = set(REQUIRED_COLUMNS).difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    for column in REQUIRED_COLUMNS:
        stripped = df[column].str.strip()
        df[column] = stripped.mask(stripped == "")

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce", format="%m/%d/%Y")
    return df


def audit(df: pd.DataFrame) -> dict:
    """Describe what the raw data contains so cleaning decisions are evidence-based.

    Returns a plain dict of shape, nulls, duplicates, label counts, and text
    statistics. It is JSON-serialisable so it can be saved next to the model.
    """
    text = df[TEXT_COLUMN].fillna("")
    words = text.str.split().str.len()
    chars = text.str.len()

    return {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": list(df.columns),
        "nulls": {col: int(df[col].isna().sum()) for col in df.columns},
        "unique_values": {col: int(df[col].nunique(dropna=True)) for col in df.columns},
        "duplicates": {
            "full_rows": int(df.duplicated().sum()),
            "complaint_id": int(df[ID_COLUMN].duplicated().sum()),
            "description": int(df[TEXT_COLUMN].duplicated().sum()),
        },
        "date_range": {
            "min": _iso(df[DATE_COLUMN].min()),
            "max": _iso(df[DATE_COLUMN].max()),
            "unparsed": int(df[DATE_COLUMN].isna().sum()),
        },
        "labels": {str(k): int(v) for k, v in df[TARGET_COLUMN].value_counts(dropna=False).items()},
        "responses": {
            str(k): int(v) for k, v in df[RESPONSE_COLUMN].value_counts(dropna=False).items()
        },
        "text": {
            "words": _quantiles(words),
            "chars": _quantiles(chars),
            "under_10_words": int((words < 10).sum()),
            "over_512_words": int((words > 512).sum()),
            "uppercase_only": int((text.str.isupper() & (chars > 20)).sum()),
            "non_ascii": int(text.str.contains(r"[^\x00-\x7F]", regex=True).sum()),
            "with_redaction": int(text.str.contains(REDACTION_PATTERN, regex=True).sum()),
            "redaction_tokens_per_row": float(text.str.count(rf"\b{REDACTION_PATTERN}\b").mean()),
            "with_money_mask": int(text.str.contains(r"\{\$", regex=True).sum()),
        },
    }


def _quantiles(series: pd.Series) -> dict:
    stats = series.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {str(k): float(v) for k, v in stats.items()}


def _iso(value) -> str | None:
    return None if pd.isna(value) else value.date().isoformat()
