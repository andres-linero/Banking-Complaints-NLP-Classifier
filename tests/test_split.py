import json
import sys

import pandas as pd
import pytest
import yaml

from complaints import split as split_stage
from complaints.split import assignment, split


def _frame(n_a: int = 80, n_b: int = 20) -> pd.DataFrame:
    n = n_a + n_b
    return pd.DataFrame(
        {
            "Complaint ID": [f"C{i}" for i in range(n)],
            "product": ["A"] * n_a + ["B"] * n_b,
            "text": [f"complaint number {i}" for i in range(n)],
        }
    )


def test_split_is_stratified_and_disjoint() -> None:
    train, test, report = split(_frame(), test_size=0.25, seed=1)

    assert len(train) == 75 and len(test) == 25
    assert report["classes"] == {"A": {"train": 60, "test": 20}, "B": {"train": 15, "test": 5}}
    assert not set(train["Complaint ID"]).intersection(test["Complaint ID"])


def test_split_is_deterministic() -> None:
    first, _, _ = split(_frame(), seed=7)
    second, _, _ = split(_frame(), seed=7)
    other, _, _ = split(_frame(), seed=8)

    assert first["Complaint ID"].tolist() == second["Complaint ID"].tolist()
    assert first["Complaint ID"].tolist() != other["Complaint ID"].tolist()


def test_split_rejects_duplicate_ids() -> None:
    df = _frame()
    df.loc[1, "Complaint ID"] = "C0"

    with pytest.raises(ValueError, match="unique"):
        split(df)


def test_assignment_lists_every_id_once() -> None:
    train, test, _ = split(_frame())
    table = assignment(train, test)

    assert len(table) == 100
    assert table["Complaint ID"].is_unique
    assert table["split"].value_counts().to_dict() == {"train": 80, "test": 20}


def test_cli_writes_parquets_and_reports(tmp_path, monkeypatch) -> None:
    in_path = tmp_path / "clean.parquet"
    _frame().to_parquet(in_path, index=False)
    runtime_path = tmp_path / "runtime.yaml"
    runtime_path.write_text(
        yaml.safe_dump(
            {
                "raw_data_path": str(tmp_path / "raw.csv"),
                "labels_config_path": str(tmp_path / "labels.yaml"),
                "processed_data_path": str(in_path),
                "train_data_path": str(tmp_path / "train.parquet"),
                "test_data_path": str(tmp_path / "test.parquet"),
                "reports_dir": str(tmp_path / "reports"),
                "models_dir": str(tmp_path / "models"),
                "mlflow_dir": str(tmp_path / "mlruns"),
            }
        )
    )
    monkeypatch.setattr(sys, "argv", ["split", "--config", str(runtime_path)])

    split_stage.main()

    assert len(pd.read_parquet(tmp_path / "train.parquet")) == 80
    assert len(pd.read_parquet(tmp_path / "test.parquet")) == 20
    report = json.loads((tmp_path / "reports/split/report.json").read_text())
    assert report["rows"] == {"train": 80, "test": 20}
    assert len(pd.read_csv(tmp_path / "reports/split/assignment.csv")) == 100
