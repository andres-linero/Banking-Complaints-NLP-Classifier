"""Stage 2: turn the validated raw data into a clean training table.

Run with: uv run python -m complaints.clean

Decisions applied here, each one backed by reports/data_study/study.md:
    1. map the 17 raw labels onto the classes in configs/labels.yaml
    2. drop rows whose raw label is in the yaml `drop` list
    3. collapse redaction tokens (XXXX, XX/XX/XXXX) and money masks ({$12.00})
    4. drop complaints shorter than `min_words`
    5. drop repeated complaint texts, keeping the first

No lowercasing, no stopwords, no lemmatization. Those are feature choices that
belong to each model, not to the data.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from complaints.config import (
    RUNTIME_CONFIG_PATH,
    TARGET_COLUMN,
    TEXT_COLUMN,
    load_label_config,
    load_runtime_config,
)
from complaints.ingest import DATE_COLUMN, ID_COLUMN, ISSUE_COLUMN, load_raw

CLASS_COLUMN = "product"
CLEAN_TEXT_COLUMN = "text"
OUTPUT_COLUMNS = [
    ID_COLUMN,
    DATE_COLUMN,
    ISSUE_COLUMN,
    TARGET_COLUMN,
    CLASS_COLUMN,
    CLEAN_TEXT_COLUMN,
]

REDACTION_TOKEN = "redacted"
MONEY_TOKEN = "money"

# XXXX, XX/XX/XXXX, XX00, XXXXXXXX, and glued forms like ABCXXXX or HoldXX/XX/XXXX.
_REDACTION = re.compile(r"[\dX/]*X{2,}[\dX/]*")
_MONEY = re.compile(r"\{\$[^}]*\}")
_REPEATED_REDACTION = re.compile(rf"(?:{REDACTION_TOKEN}\W*){{2,}}")
_WHITESPACE = re.compile(r"\s+")


def map_labels(df: pd.DataFrame, mapping: dict[str, str], drop: set[str]) -> pd.DataFrame:
    """Add the class column and remove dropped labels. Unknown raw labels are an error."""
    known = set(mapping) | drop
    unknown = set(df[TARGET_COLUMN].dropna().unique()).difference(known)
    if unknown:
        raise ValueError(f"Raw labels missing from labels.yaml: {sorted(unknown)}")

    result = df[~df[TARGET_COLUMN].isin(drop)].copy()
    result[CLASS_COLUMN] = result[TARGET_COLUMN].map(mapping)
    return result


def normalize_text(text: str) -> str:
    """Replace source anonymisation with stable placeholder words and tidy whitespace."""
    text = _MONEY.sub(f" {MONEY_TOKEN} ", str(text))
    text = _REDACTION.sub(f" {REDACTION_TOKEN} ", text)
    text = _REPEATED_REDACTION.sub(f"{REDACTION_TOKEN} ", text)
    return _WHITESPACE.sub(" ", text).strip()


def clean(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    """Apply the cleaning steps in order and return the table plus a count per step."""
    steps: dict[str, int] = {"raw_rows": int(len(df))}

    df = map_labels(df, config["mapping"], config["drop"])
    steps["after_drop_labels"] = int(len(df))

    df = df.dropna(subset=[TEXT_COLUMN, CLASS_COLUMN])
    steps["after_drop_null_text"] = int(len(df))

    df[CLEAN_TEXT_COLUMN] = df[TEXT_COLUMN].map(normalize_text)

    words = df[CLEAN_TEXT_COLUMN].str.split().str.len()
    df = df[words >= config["min_words"]]
    steps["after_drop_short"] = int(len(df))

    df = df.drop_duplicates(subset=[CLEAN_TEXT_COLUMN], keep="first")
    steps["after_drop_duplicates"] = int(len(df))

    df = df[OUTPUT_COLUMNS].reset_index(drop=True)
    report = {
        "steps": steps,
        "removed": steps["raw_rows"] - steps["after_drop_duplicates"],
        "classes": {str(k): int(v) for k, v in df[CLASS_COLUMN].value_counts().items()},
    }
    return df, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean the raw complaints into a training table.")
    parser.add_argument("--config", default=str(RUNTIME_CONFIG_PATH))
    parser.add_argument("--data-path")
    parser.add_argument("--labels-config")
    parser.add_argument("--out-path")
    parser.add_argument("--reports-dir")
    args = parser.parse_args()

    runtime = load_runtime_config(args.config)
    df, report = clean(
        load_raw(args.data_path or runtime.raw_data_path),
        load_label_config(args.labels_config or runtime.labels_config_path),
    )

    out_path = Path(args.out_path or runtime.processed_data_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    report_path = Path(args.reports_dir or runtime.reports_dir) / "cleaning" / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote {len(df)} rows to {out_path}")
    for step, count in report["steps"].items():
        print(f"  {step:<24} {count:>6}")
    print("Classes:")
    for name, count in report["classes"].items():
        print(f"  {name:<20} {count:>6}")


if __name__ == "__main__":
    main()
