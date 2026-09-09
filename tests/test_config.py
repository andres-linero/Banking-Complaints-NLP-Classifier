import json
import sys

import pandas as pd
import pytest
import yaml

from complaints import clean, data_study
from complaints.config import PROJECT_ROOT, load_runtime_config
from complaints.ingest import REQUIRED_COLUMNS


def test_runtime_paths_are_relative_to_repo_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runtime = load_runtime_config()
    assert runtime.labels_config_path == PROJECT_ROOT / "configs" / "labels.yaml"
    assert runtime.processed_data_path == PROJECT_ROOT / "data" / "processed" / "clean.parquet"


@pytest.mark.parametrize("content", ["[]", "raw_data_path: sample.csv", "null"])
def test_invalid_runtime_config(tmp_path, content):
    path = tmp_path / "runtime.yaml"
    path.write_text(content)
    with pytest.raises(ValueError, match="Runtime config must contain"):
        load_runtime_config(path)


def test_both_stages_use_runtime_config_and_cli_override(tmp_path, monkeypatch):
    raw_path = tmp_path / "raw.csv"
    pd.DataFrame(
        [["C1", "1/1/2023", "Credit card", "I1", "Charged twice", "TX", "1", "Closed"]],
        columns=REQUIRED_COLUMNS,
    ).to_csv(raw_path, index=False)
    labels_path = tmp_path / "labels.yaml"
    labels_path.write_text("classes:\n  Card: [Credit card]\nmin_words: 1\n")
    runtime_path = tmp_path / "runtime.yaml"
    output = tmp_path / "clean.parquet"
    reports = tmp_path / "reports"
    runtime_path.write_text(
        yaml.safe_dump(
            {
                "raw_data_path": str(tmp_path / "missing.csv"),
                "labels_config_path": str(labels_path),
                "processed_data_path": str(output),
                "train_data_path": str(tmp_path / "train.parquet"),
                "test_data_path": str(tmp_path / "test.parquet"),
                "reports_dir": str(reports),
                "models_dir": str(tmp_path / "models"),
                "mlflow_dir": str(tmp_path / "mlruns"),
            }
        )
    )
    for stage in (clean, data_study):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                stage.__name__,
                "--config",
                str(runtime_path),
                "--data-path",
                str(raw_path),
            ],
        )
        stage.main()
    assert pd.read_parquet(output)["product"].tolist() == ["Card"]
    assert json.loads((reports / "cleaning/report.json").read_text())["steps"]["raw_rows"] == 1
    assert json.loads((reports / "data_study/audit.json").read_text())["shape"]["rows"] == 1
    assert (reports / "data_study/study.md").exists()
