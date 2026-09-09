import json
import sys

import joblib
import numpy as np
import pandas as pd
import yaml

from complaints import evaluate as evaluate_stage
from complaints.evaluate import calibration_bins, score, threshold_sweep, worst_mistakes
from complaints.train import build_pipeline

CONFIG = {
    "features": {"ngram_range": [1, 2], "min_df": 1},
    "classifier": {"C": 1.0, "class_weight": "balanced", "max_iter": 500},
}


def _frame(n: int = 12, prefix: str = "T") -> pd.DataFrame:
    mortgage = [f"my mortgage escrow payment and home loan servicer problem {i}" for i in range(n)]
    card = [f"the credit card charge was disputed and the card was frozen {i}" for i in range(n)]
    return pd.DataFrame(
        {
            "Complaint ID": [f"{prefix}{i}" for i in range(2 * n)],
            "product": ["Mortgage"] * n + ["Credit card"] * n,
            "text": mortgage + card,
        }
    )


def test_score_reports_accuracy_per_class_and_confusion() -> None:
    pipeline = build_pipeline(CONFIG).fit(_frame()["text"], _frame()["product"])

    result = score(pipeline, _frame(6, "X"))

    assert result["classes"] == ["Credit card", "Mortgage"]
    assert result["accuracy"] == 1.0
    assert np.array(result["confusion_matrix"]).sum() == 12
    assert set(result["per_class"]["Mortgage"]) >= {"precision", "recall", "f1-score", "support"}


def test_threshold_sweep_coverage_falls_as_threshold_rises() -> None:
    truth = np.array(["a", "a", "b", "b"])
    predicted = np.array(["a", "b", "b", "b"])
    confidence = np.array([0.9, 0.55, 0.7, 0.95])

    sweep = threshold_sweep(truth, predicted, confidence)
    coverage = [r["coverage"] for r in sweep["sweep"]]

    assert coverage == sorted(coverage, reverse=True)
    assert sweep["sweep"][0]["routed_accuracy"] == 0.75
    assert sweep["recommended"] == 0.6  # first cutoff that drops the wrong 0.55 prediction


def test_calibration_bins_count_every_row() -> None:
    truth = np.array(["a"] * 4)
    predicted = np.array(["a", "a", "b", "a"])
    confidence = np.array([0.95, 0.85, 0.35, 0.99])

    bins = calibration_bins(truth, predicted, confidence)

    assert sum(b["count"] for b in bins) == 4
    assert bins[0]["bin"] == "0.3-0.4" and bins[0]["accuracy"] == 0.0


def test_worst_mistakes_lists_only_wrong_rows_most_confident_first() -> None:
    df = _frame(2)
    predicted = np.array(["Mortgage", "Credit card", "Mortgage", "Credit card"])
    confidence = np.array([0.9, 0.6, 0.8, 0.7])

    table = worst_mistakes(df, predicted, confidence)

    assert table["Complaint ID"].tolist() == ["T2", "T1"]
    assert table["confidence"].tolist() == [0.8, 0.6]


def test_cli_writes_report_figures_and_mistakes(tmp_path, monkeypatch) -> None:
    train, test = _frame(12), _frame(5, "V")
    train.to_parquet(tmp_path / "train.parquet", index=False)
    test.to_parquet(tmp_path / "test.parquet", index=False)
    model_path = tmp_path / "models" / "baseline.joblib"
    model_path.parent.mkdir()
    joblib.dump(build_pipeline(CONFIG).fit(train["text"], train["product"]), model_path)
    runtime_path = tmp_path / "runtime.yaml"
    runtime_path.write_text(
        yaml.safe_dump(
            {
                "raw_data_path": str(tmp_path / "raw.csv"),
                "labels_config_path": str(tmp_path / "labels.yaml"),
                "processed_data_path": str(tmp_path / "clean.parquet"),
                "train_data_path": str(tmp_path / "train.parquet"),
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
        ["evaluate", "--config", str(runtime_path), "--no-learning-curve", "--no-mlflow"],
    )

    evaluate_stage.main()

    out = tmp_path / "reports" / "evaluate"
    report = json.loads((out / "baseline.json").read_text())
    assert report["test_rows"] == 10
    assert "learning_curve" not in report
    assert (out / "worst_mistakes.csv").exists()
    for name in ("confusion_matrix", "per_class_f1", "threshold_curve", "calibration"):
        assert (out / "figures" / f"{name}.png").stat().st_size > 0
