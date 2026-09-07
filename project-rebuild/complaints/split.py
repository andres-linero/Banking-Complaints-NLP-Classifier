"""Stage 3: freeze one stratified train/test split.

Run with: uv run python -m complaints.split

Every model is trained on train.parquet and scored on test.parquet, so their
results are comparable. The split is deterministic (fixed seed) and the
Complaint ID -> split assignment is written to reports/split/assignment.csv so
the exact rows are on record even if the parquet files are regenerated.
The test file must not be read by anything except the final evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from complaints.clean import CLASS_COLUMN
from complaints.config import RUNTIME_CONFIG_PATH, load_runtime_config
from complaints.ingest import ID_COLUMN

TEST_SIZE = 0.2
RANDOM_SEED = 42


def split(
    df: pd.DataFrame, test_size: float = TEST_SIZE, seed: int = RANDOM_SEED
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Stratified split on the class column. Returns train, test, and a report."""
    if df[ID_COLUMN].duplicated().any():
        raise ValueError("Complaint IDs must be unique before splitting")

    train, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df[CLASS_COLUMN]
    )
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    counts = pd.DataFrame(
        {
            "train": train[CLASS_COLUMN].value_counts(),
            "test": test[CLASS_COLUMN].value_counts(),
        }
    ).fillna(0)
    report = {
        "test_size": test_size,
        "seed": seed,
        "rows": {"train": int(len(train)), "test": int(len(test))},
        "classes": {
            str(name): {"train": int(row["train"]), "test": int(row["test"])}
            for name, row in counts.sort_values("train", ascending=False).iterrows()
        },
    }
    return train, test, report


def assignment(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """One row per complaint: its ID and which side of the split it landed on."""
    return (
        pd.concat(
            [
                pd.DataFrame({ID_COLUMN: train[ID_COLUMN], "split": "train"}),
                pd.DataFrame({ID_COLUMN: test[ID_COLUMN], "split": "test"}),
            ]
        )
        .sort_values(ID_COLUMN)
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the train/test split.")
    parser.add_argument("--config", default=str(RUNTIME_CONFIG_PATH))
    parser.add_argument("--in-path")
    parser.add_argument("--train-path")
    parser.add_argument("--test-path")
    parser.add_argument("--reports-dir")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    runtime = load_runtime_config(args.config)
    df = pd.read_parquet(args.in_path or runtime.processed_data_path)
    train, test, report = split(df, test_size=args.test_size, seed=args.seed)

    train_path = Path(args.train_path or runtime.train_data_path)
    test_path = Path(args.test_path or runtime.test_data_path)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    train.to_parquet(train_path, index=False)
    test.to_parquet(test_path, index=False)

    report_dir = Path(args.reports_dir or runtime.reports_dir) / "split"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    assignment(train, test).to_csv(report_dir / "assignment.csv", index=False)

    print(f"train {report['rows']['train']} rows -> {train_path}")
    print(f"test  {report['rows']['test']} rows -> {test_path}")
    print(f"{'class':<20} {'train':>6} {'test':>6}")
    for name, row in report["classes"].items():
        print(f"{name:<20} {row['train']:>6} {row['test']:>6}")


if __name__ == "__main__":
    main()
