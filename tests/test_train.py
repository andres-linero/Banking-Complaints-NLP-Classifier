import json
import sys

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.pipeline import Pipeline

from complaints import train as train_stage
from complaints.train import build_pipeline, train

CONFIG = {
    "features": {"ngram_range": [1, 2], "min_df": 1},
    "classifier": {"C": 1.0, "class_weight": "balanced", "max_iter": 500},
}


def _train_frame() -> pd.DataFrame:
    mortgage = [
        f"my mortgage escrow payment and home loan servicer problem number {i}" for i in range(12)
    ]
    card = [f"the credit card charge was disputed and the card was frozen {i}" for i in range(12)]
    return pd.DataFrame(
        {
            "Complaint ID": [f"C{i}" for i in range(24)],
            "product": ["Mortgage"] * 12 + ["Credit card"] * 12,
            "text": mortgage + card,
        }
    )


def test_build_pipeline_has_vectorizer_then_classifier() -> None:
    pipeline = build_pipeline(CONFIG)

    assert isinstance(pipeline, Pipeline)
    assert [name for name, _ in pipeline.steps] == ["vectorize", "classify"]
    assert pipeline.named_steps["classify"].C == 1.0


def test_train_fits_on_raw_text_and_reports_cv() -> None:
    pipeline, report = train(_train_frame(), CONFIG, folds=3)

    assert report["classes"] == ["Credit card", "Mortgage"]
    assert report["train_rows"] == 24
    assert report["cross_validation"]["folds"] == 3
    assert 0 <= report["cross_validation"]["f1_macro_mean"] <= 1

    proba = pipeline.predict_proba(["mortgage escrow home loan"])
    assert proba.shape == (1, 2)
    assert np.isclose(proba.sum(), 1.0)
    assert pipeline.predict(["mortgage escrow home loan"])[0] == "Mortgage"
    assert pipeline.predict(["credit card charge disputed"])[0] == "Credit card"


def test_cli_saves_model_and_report(tmp_path, monkeypatch) -> None:
    train_path = tmp_path / "train.parquet"
    _train_frame().to_parquet(train_path, index=False)
    model_config = tmp_path / "model.yaml"
    model_config.write_text(yaml.safe_dump(CONFIG))
    runtime_path = tmp_path / "runtime.yaml"
    runtime_path.write_text(
        yaml.safe_dump(
            {
                "raw_data_path": str(tmp_path / "raw.csv"),
                "labels_config_path": str(tmp_path / "labels.yaml"),
                "processed_data_path": str(tmp_path / "clean.parquet"),
                "train_data_path": str(train_path),
                "test_data_path": str(tmp_path / "test.parquet"),
                "reports_dir": str(tmp_path / "reports"),
                "models_dir": str(tmp_path / "models"),
                "mlflow_dir": str(tmp_path / "mlruns"),
            }
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--config",
            str(runtime_path),
            "--model-config",
            str(model_config),
            "--no-mlflow",
        ],
    )

    train_stage.main()

    model = joblib.load(tmp_path / "models" / "baseline.joblib")
    assert model.predict(["mortgage escrow"])[0] == "Mortgage"
    report = json.loads((tmp_path / "reports" / "train" / "baseline.json").read_text())
    assert report["train_rows"] == 24
    assert "mlflow_run_id" not in report
