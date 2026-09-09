"""Stage 5: score the saved model on the frozen test set, once.

Run with: uv run python -m complaints.evaluate

Loads the fitted pipeline, predicts every row of test.parquet, and writes:
    reports/evaluate/baseline.json        accuracy, per-class scores, confusion
                                          matrix, threshold sweep, calibration
    reports/evaluate/worst_mistakes.csv   confident wrong predictions to read
    reports/evaluate/figures/*.png        confusion matrix, per-class F1,
                                          learning curve, threshold curve,
                                          calibration curve
The test metrics are also attached to the model's MLflow run.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, learning_curve

from complaints.clean import CLASS_COLUMN, CLEAN_TEXT_COLUMN
from complaints.config import RUNTIME_CONFIG_PATH, load_runtime_config
from complaints.ingest import ID_COLUMN
from complaints.train import MODEL_NAME, build_pipeline
from complaints.vectorize import BASELINE_CONFIG_PATH, load_model_config

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

THRESHOLDS = [round(t, 2) for t in np.arange(0.0, 0.96, 0.05)]
TARGET_ROUTED_ACCURACY = 0.90
LEARNING_CURVE_SIZES = [0.1, 0.25, 0.5, 0.75, 1.0]

# Chart tokens: light surface, ink for text, one blue ramp for magnitude,
# blue + orange for the two-series learning and threshold curves.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e6e5e1"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
BLUE_RAMP = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def score(pipeline, test_df: pd.DataFrame) -> dict:
    """Everything that can be computed from predictions on the test rows."""
    texts, truth = test_df[CLEAN_TEXT_COLUMN], test_df[CLASS_COLUMN]
    classes = [str(c) for c in pipeline.classes_]
    proba = pipeline.predict_proba(texts)
    predicted = np.array(classes)[proba.argmax(axis=1)]
    confidence = proba.max(axis=1)

    report = classification_report(
        truth, predicted, labels=classes, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(truth, predicted, labels=classes)

    return {
        "model": MODEL_NAME,
        "test_rows": int(len(test_df)),
        "classes": classes,
        "accuracy": float(accuracy_score(truth, predicted)),
        "f1_macro": float(f1_score(truth, predicted, average="macro")),
        "f1_weighted": float(f1_score(truth, predicted, average="weighted")),
        "per_class": {c: {k: float(v) for k, v in report[c].items()} for c in classes},
        "confusion_matrix": matrix.tolist(),
        "thresholds": threshold_sweep(truth.to_numpy(), predicted, confidence),
        "calibration": calibration_bins(truth.to_numpy(), predicted, confidence),
        "_predicted": predicted,
        "_confidence": confidence,
    }


def threshold_sweep(truth: np.ndarray, predicted: np.ndarray, confidence: np.ndarray) -> dict:
    """For each confidence cutoff: how many complaints route automatically, and how accurately."""
    rows = []
    correct = truth == predicted
    for t in THRESHOLDS:
        routed = confidence >= t
        rows.append(
            {
                "threshold": t,
                "coverage": float(routed.mean()),
                "routed_accuracy": float(correct[routed].mean()) if routed.any() else None,
            }
        )
    recommended = next(
        (
            r["threshold"]
            for r in rows
            if r["routed_accuracy"] and r["routed_accuracy"] >= TARGET_ROUTED_ACCURACY
        ),
        None,
    )
    return {
        "target_routed_accuracy": TARGET_ROUTED_ACCURACY,
        "recommended": recommended,
        "sweep": rows,
    }


def calibration_bins(
    truth: np.ndarray, predicted: np.ndarray, confidence: np.ndarray, bins: int = 10
) -> list:
    """Does a 0.8 confidence mean right 80% of the time? One row per confidence bin."""
    edges = np.linspace(0, 1, bins + 1)
    correct = truth == predicted
    out = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence >= lo) & (confidence < hi if hi < 1 else confidence <= hi)
        if mask.any():
            out.append(
                {
                    "bin": f"{lo:.1f}-{hi:.1f}",
                    "count": int(mask.sum()),
                    "mean_confidence": float(confidence[mask].mean()),
                    "accuracy": float(correct[mask].mean()),
                }
            )
    return out


def worst_mistakes(
    test_df: pd.DataFrame, predicted: np.ndarray, confidence: np.ndarray, n: int = 25
) -> pd.DataFrame:
    """The wrong predictions the model was most sure about. These are the ones to read."""
    wrong = test_df[CLASS_COLUMN].to_numpy() != predicted
    table = pd.DataFrame(
        {
            ID_COLUMN: test_df[ID_COLUMN].to_numpy(),
            "truth": test_df[CLASS_COLUMN].to_numpy(),
            "predicted": predicted,
            "confidence": confidence,
            "text": test_df[CLEAN_TEXT_COLUMN].str.slice(0, 300).to_numpy(),
        }
    )[wrong]
    return table.sort_values("confidence", ascending=False).head(n).reset_index(drop=True)


def compute_learning_curve(config: dict, train_df: pd.DataFrame, folds: int = 5) -> dict:
    """Macro-F1 on train folds and validation folds as the training set grows."""
    sizes, train_scores, val_scores = learning_curve(
        build_pipeline(config),
        train_df[CLEAN_TEXT_COLUMN],
        train_df[CLASS_COLUMN],
        train_sizes=LEARNING_CURVE_SIZES,
        cv=StratifiedKFold(folds, shuffle=True, random_state=42),
        scoring="f1_macro",
        n_jobs=-1,
    )
    return {
        "train_sizes": [int(s) for s in sizes],
        "train_f1_macro": train_scores.mean(axis=1).tolist(),
        "validation_f1_macro": val_scores.mean(axis=1).tolist(),
        "validation_f1_macro_std": val_scores.std(axis=1).tolist(),
    }


# ----- figures -------------------------------------------------------------


def _axes(width: float, height: float):
    fig, ax = plt.subplots(figsize=(width, height), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=9, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    return fig, ax


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return str(path)


def plot_confusion_matrix(result: dict, path: Path) -> str:
    matrix = np.array(result["confusion_matrix"])
    share = matrix / matrix.sum(axis=1, keepdims=True)
    classes = result["classes"]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", BLUE_RAMP)

    fig, ax = _axes(7.5, 6.5)
    ax.yaxis.grid(False)
    ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=35, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted", color=INK_SOFT)
    ax.set_ylabel("True", color=INK_SOFT)
    ax.set_title("Confusion matrix, share of each true class", loc="left", color=INK, fontsize=11)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if matrix[i, j]:
                ax.text(
                    j,
                    i,
                    f"{share[i, j]:.0%}\n{matrix[i, j]}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if share[i, j] > 0.5 else INK,
                )
    return _save(fig, path)


def plot_per_class_f1(result: dict, path: Path) -> str:
    rows = sorted(result["per_class"].items(), key=lambda kv: kv[1]["f1-score"])
    names = [k for k, _ in rows]
    f1 = [v["f1-score"] for _, v in rows]
    support = [int(v["support"]) for _, v in rows]

    fig, ax = _axes(7.5, 4.2)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    ax.barh(names, f1, color=BLUE, height=0.55)
    for i, (value, n) in enumerate(zip(f1, support, strict=True)):
        ax.text(value + 0.01, i, f"{value:.2f}   n={n}", va="center", fontsize=8.5, color=INK_SOFT)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("F1 on the test set", color=INK_SOFT)
    ax.set_title(
        f"Per-class F1, macro average {result['f1_macro']:.2f}", loc="left", color=INK, fontsize=11
    )
    return _save(fig, path)


def plot_learning_curve(curve: dict, path: Path) -> str:
    x = curve["train_sizes"]
    fig, ax = _axes(7.5, 4.2)
    ax.plot(
        x,
        curve["train_f1_macro"],
        color=ORANGE,
        linewidth=2,
        marker="o",
        markersize=6,
        label="Training folds",
    )
    ax.plot(
        x,
        curve["validation_f1_macro"],
        color=BLUE,
        linewidth=2,
        marker="o",
        markersize=6,
        label="Validation folds",
    )
    lo = np.array(curve["validation_f1_macro"]) - np.array(curve["validation_f1_macro_std"])
    hi = np.array(curve["validation_f1_macro"]) + np.array(curve["validation_f1_macro_std"])
    ax.fill_between(x, lo, hi, color=BLUE, alpha=0.10, linewidth=0)
    ax.set_xlabel("Training complaints", color=INK_SOFT)
    ax.set_ylabel("Macro F1", color=INK_SOFT)
    ax.set_ylim(0.5, 1.02)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title(
        "Learning curve, 5-fold cross-validation on the train split",
        loc="left",
        color=INK,
        fontsize=11,
    )
    return _save(fig, path)


def plot_threshold_curve(sweep: dict, path: Path) -> str:
    rows = [r for r in sweep["sweep"] if r["routed_accuracy"] is not None]
    t = [r["threshold"] for r in rows]
    fig, ax = _axes(7.5, 4.2)
    ax.plot(
        t,
        [r["coverage"] for r in rows],
        color=ORANGE,
        linewidth=2,
        marker="o",
        markersize=5,
        label="Share routed automatically",
    )
    ax.plot(
        t,
        [r["routed_accuracy"] for r in rows],
        color=BLUE,
        linewidth=2,
        marker="o",
        markersize=5,
        label="Accuracy of what was routed",
    )
    ax.axhline(sweep["target_routed_accuracy"], color=GRID, linewidth=1)
    if sweep["recommended"] is not None:
        ax.axvline(sweep["recommended"], color=INK_SOFT, linewidth=1, linestyle=(0, (4, 4)))
        ax.text(
            sweep["recommended"] + 0.01,
            0.02,
            f"threshold {sweep['recommended']:.2f}",
            fontsize=8.5,
            color=INK_SOFT,
        )
    ax.set_xlabel("Confidence threshold", color=INK_SOFT)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.set_title(
        "Human-review threshold: coverage against accuracy", loc="left", color=INK, fontsize=11
    )
    return _save(fig, path)


def plot_calibration(bins: list, path: Path) -> str:
    fig, ax = _axes(5.2, 5.0)
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1)
    ax.plot(
        [b["mean_confidence"] for b in bins],
        [b["accuracy"] for b in bins],
        color=BLUE,
        linewidth=2,
        marker="o",
        markersize=6,
    )
    for b in bins:
        ax.text(
            b["mean_confidence"] + 0.025,
            b["accuracy"] - 0.01,
            str(b["count"]),
            fontsize=7.5,
            ha="left",
            va="top",
            color=INK_SOFT,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Model confidence", color=INK_SOFT)
    ax.set_ylabel("Share actually correct", color=INK_SOFT)
    ax.set_title("Calibration, counts per bin", loc="left", color=INK, fontsize=11)
    return _save(fig, path)


# ----- orchestration -------------------------------------------------------


def log_to_mlflow(result: dict, figures: list[str], tracking_dir: Path, run_id: str | None) -> None:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{tracking_dir.resolve() / 'mlflow.db'}")
    mlflow.set_experiment("complaints-classifier")
    with mlflow.start_run(run_id=run_id, run_name=None if run_id else f"{MODEL_NAME}-evaluate"):
        mlflow.log_metrics(
            {
                "test_accuracy": result["accuracy"],
                "test_f1_macro": result["f1_macro"],
                "test_f1_weighted": result["f1_weighted"],
            }
        )
        if result["thresholds"]["recommended"] is not None:
            mlflow.log_metric("recommended_threshold", result["thresholds"]["recommended"])
        for name, scores in result["per_class"].items():
            mlflow.log_metric(f"test_f1_{name.lower().replace(' ', '_')}", scores["f1-score"])
        for figure in figures:
            mlflow.log_artifact(figure, artifact_path="figures")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the saved model on the frozen test set.")
    parser.add_argument("--config", default=str(RUNTIME_CONFIG_PATH))
    parser.add_argument("--model-config", default=str(BASELINE_CONFIG_PATH))
    parser.add_argument("--model-path")
    parser.add_argument("--test-path")
    parser.add_argument("--train-path")
    parser.add_argument("--reports-dir")
    parser.add_argument("--mlflow-dir")
    parser.add_argument(
        "--no-learning-curve", action="store_true", help="skip the slow learning curve"
    )
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    runtime = load_runtime_config(args.config)
    pipeline = joblib.load(args.model_path or runtime.models_dir / f"{MODEL_NAME}.joblib")
    test_df = pd.read_parquet(args.test_path or runtime.test_data_path)

    result = score(pipeline, test_df)
    predicted, confidence = result.pop("_predicted"), result.pop("_confidence")

    out_dir = Path(args.reports_dir or runtime.reports_dir) / "evaluate"
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    figures = [
        plot_confusion_matrix(result, figures_dir / "confusion_matrix.png"),
        plot_per_class_f1(result, figures_dir / "per_class_f1.png"),
        plot_threshold_curve(result["thresholds"], figures_dir / "threshold_curve.png"),
        plot_calibration(result["calibration"], figures_dir / "calibration.png"),
    ]
    if not args.no_learning_curve:
        train_df = pd.read_parquet(args.train_path or runtime.train_data_path)
        result["learning_curve"] = compute_learning_curve(
            load_model_config(args.model_config), train_df
        )
        figures.append(
            plot_learning_curve(result["learning_curve"], figures_dir / "learning_curve.png")
        )

    worst_mistakes(test_df, predicted, confidence).to_csv(
        out_dir / "worst_mistakes.csv", index=False
    )
    (out_dir / f"{MODEL_NAME}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    if not args.no_mlflow:
        train_report = (
            Path(args.reports_dir or runtime.reports_dir) / "train" / f"{MODEL_NAME}.json"
        )
        run_id = (
            json.loads(train_report.read_text()).get("mlflow_run_id")
            if train_report.exists()
            else None
        )
        log_to_mlflow(result, figures, Path(args.mlflow_dir or runtime.mlflow_dir), run_id)

    print(
        f"test rows {result['test_rows']}   accuracy {result['accuracy']:.3f}   "
        f"macro-F1 {result['f1_macro']:.3f}"
    )
    print(f"{'class':<18} {'precision':>9} {'recall':>7} {'f1':>6} {'n':>5}")
    for name, s in result["per_class"].items():
        print(
            f"{name:<18} {s['precision']:>9.2f} {s['recall']:>7.2f} "
            f"{s['f1-score']:>6.2f} {int(s['support']):>5}"
        )
    rec = result["thresholds"]["recommended"]
    if rec is not None:
        row = next(r for r in result["thresholds"]["sweep"] if r["threshold"] == rec)
        print(
            f"recommended threshold {rec:.2f}: routes {row['coverage']:.0%} automatically "
            f"at {row['routed_accuracy']:.1%} accuracy, the rest go to a person"
        )
    print(f"figures in {figures_dir}")


if __name__ == "__main__":
    main()
