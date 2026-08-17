"""Training-report figures saved alongside the metrics JSON.

Generates the three graphs the training workflow was missing:
- class distribution (the data-imbalance picture),
- confusion matrix on the held-out split,
- per-class F1 versus support (how imbalance affects quality).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix

from banking_complaints.labels import friendly_label

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES_BLUE = "#2a78d6"

SEQUENTIAL_BLUES = LinearSegmentedColormap.from_list(
    "seq_blue",
    ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)

AVERAGE_LABELS = {"macro avg", "weighted avg"}


def _new_axes(width: float, height: float):
    fig, ax = plt.subplots(figsize=(width, height), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    return fig, ax


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_class_distribution(labels: pd.Series, path: str | Path) -> Path:
    """Horizontal bar chart of class counts — the imbalance picture."""
    counts = labels.value_counts().sort_values()
    names = [friendly_label(label) for label in counts.index]

    fig, ax = _new_axes(7.5, 0.42 * len(counts) + 1.2)
    bars = ax.barh(names, counts.values, height=0.62, color=SERIES_BLUE)
    ax.bar_label(bars, padding=4, color=INK_SECONDARY, fontsize=8)
    ax.spines["bottom"].set_visible(False)
    ax.xaxis.set_visible(False)
    ax.tick_params(axis="y", labelcolor=INK, labelsize=9, length=0)
    ax.set_title(
        f"Class distribution — largest/smallest ratio {counts.max() / counts.min():.0f}:1",
        color=INK,
        fontsize=11,
        loc="left",
        pad=12,
    )
    return _save(fig, Path(path))


def plot_confusion_matrix(y_true, y_pred, path: str | Path) -> Path:
    """Row-normalized confusion matrix on the held-out test split."""
    class_names = sorted(pd.unique(pd.concat([pd.Series(y_true), pd.Series(y_pred)])))
    matrix = confusion_matrix(y_true, y_pred, labels=class_names, normalize="true")
    display_names = [friendly_label(name) for name in class_names]

    size = 0.75 * len(class_names) + 2.5
    fig, ax = _new_axes(size + 1.5, size)
    ax.imshow(matrix, cmap=SEQUENTIAL_BLUES, vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), display_names, rotation=45, ha="right", color=INK)
    ax.set_yticks(range(len(class_names)), display_names, color=INK)
    ax.tick_params(length=0, labelsize=8)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            value = matrix[i, j]
            if value < 0.01:
                continue
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color=SURFACE if value > 0.55 else INK_SECONDARY,
            )
    ax.set_xlabel("Predicted", color=INK_MUTED, fontsize=9)
    ax.set_ylabel("Actual", color=INK_MUTED, fontsize=9)
    ax.set_title(
        "Confusion matrix (row-normalized, test split)",
        color=INK,
        fontsize=11,
        loc="left",
        pad=12,
    )
    return _save(fig, Path(path))


def _nudge_overlapping_labels(fig, annotations, step: int = 11, max_nudges: int = 10) -> None:
    """Shift point labels upward until none of their rendered boxes overlap.

    Labels moved off their point get a hairline leader back to it.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    placed = []
    for annotation in annotations:
        bbox = annotation.get_window_extent(renderer)
        nudges = 0
        while nudges < max_nudges and any(bbox.overlaps(other) for other in placed):
            dx, dy = annotation.xyann
            annotation.xyann = (dx, dy + step)
            bbox = annotation.get_window_extent(renderer)
            nudges += 1
        placed.append(bbox)
        if nudges:
            ax = annotation.axes
            ax.annotate(
                annotation.get_text(),
                annotation.xy,
                textcoords="offset points",
                xytext=annotation.xyann,
                ha=annotation.get_ha(),
                fontsize=annotation.get_fontsize(),
                color=annotation.get_color(),
                arrowprops={
                    "arrowstyle": "-",
                    "color": BASELINE,
                    "linewidth": 0.75,
                    "shrinkA": 2,
                    "shrinkB": 4,
                },
            )
            annotation.remove()


def plot_f1_vs_support(classification_report: dict, path: str | Path) -> Path:
    """Scatter of per-class F1 against test support — where imbalance bites."""
    rows = [
        {"label": friendly_label(label), "f1": values["f1-score"], "support": values["support"]}
        for label, values in classification_report.items()
        if isinstance(values, dict) and "f1-score" in values and label not in AVERAGE_LABELS
    ]
    frame = pd.DataFrame(rows)
    macro_f1 = frame["f1"].mean()

    fig, ax = _new_axes(7.5, 5)
    ax.grid(axis="y", color=GRID, linewidth=0.75)
    ax.set_axisbelow(True)
    ax.scatter(frame["support"], frame["f1"], s=64, color=SERIES_BLUE, zorder=3)
    ax.axhline(macro_f1, color=BASELINE, linewidth=1, linestyle=(0, (4, 4)))
    ax.annotate(
        f"macro F1 {macro_f1:.2f}",
        (frame["support"].max(), macro_f1),
        textcoords="offset points",
        xytext=(0, 5),
        ha="right",
        fontsize=8,
        color=INK_MUTED,
    )
    # Right-edge points get left-side labels so names never run off the axes.
    label_flip_x = frame["support"].max() * 0.75
    annotations = []
    for row in frame.itertuples():
        on_right = row.support >= label_flip_x
        annotations.append(
            ax.annotate(
                row.label,
                (row.support, row.f1),
                textcoords="offset points",
                xytext=(-7, -3) if on_right else (7, -3),
                ha="right" if on_right else "left",
                fontsize=8,
                color=INK_SECONDARY,
            )
        )
    _nudge_overlapping_labels(fig, annotations)
    ax.set_xlabel("Test support (examples)", color=INK_MUTED, fontsize=9)
    ax.set_ylabel("F1", color=INK_MUTED, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_xlim(left=0)
    ax.set_title(
        "Per-class F1 versus support — small classes drag macro F1",
        color=INK,
        fontsize=11,
        loc="left",
        pad=12,
    )
    return _save(fig, Path(path))


def save_training_figures(
    labels: pd.Series,
    y_test,
    predictions,
    classification_report: dict,
    figures_dir: str | Path,
) -> dict:
    """Save the full training-report figure set; returns name -> file path."""
    figures_dir = Path(figures_dir)
    return {
        "class_distribution": str(
            plot_class_distribution(labels, figures_dir / "class_distribution.png")
        ),
        "confusion_matrix": str(
            plot_confusion_matrix(y_test, predictions, figures_dir / "confusion_matrix.png")
        ),
        "f1_vs_support": str(
            plot_f1_vs_support(classification_report, figures_dir / "f1_vs_support.png")
        ),
    }
