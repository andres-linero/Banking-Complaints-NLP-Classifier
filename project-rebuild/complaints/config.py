"""Shared runtime paths and label configuration for every rebuild stage."""

from dataclasses import dataclass
from pathlib import Path

import yaml

REBUILD_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REBUILD_ROOT.parent
RUNTIME_CONFIG_PATH = REBUILD_ROOT / "configs" / "runtime.yaml"

TEXT_COLUMN = "Complaint Description"
TARGET_COLUMN = "Banking Product"


@dataclass(frozen=True)
class RuntimeConfig:
    raw_data_path: Path
    labels_config_path: Path
    processed_data_path: Path
    reports_dir: Path


def load_runtime_config(path: str | Path = RUNTIME_CONFIG_PATH) -> RuntimeConfig:
    """Load runtime paths; relative YAML values resolve against project-rebuild."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    fields = tuple(RuntimeConfig.__dataclass_fields__)
    if not isinstance(raw, dict) or set(raw) != set(fields):
        raise ValueError(f"Runtime config must contain exactly: {', '.join(fields)}")
    paths = {}
    for name in fields:
        value = raw[name]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Runtime config {name} must be a nonempty path string")
        value = Path(value).expanduser()
        paths[name] = (REBUILD_ROOT / value).resolve()
    return RuntimeConfig(**paths)


_DEFAULT_RUNTIME = load_runtime_config()
RAW_DATA_PATH = _DEFAULT_RUNTIME.raw_data_path
LABELS_CONFIG_PATH = _DEFAULT_RUNTIME.labels_config_path
PROCESSED_DATA_PATH = _DEFAULT_RUNTIME.processed_data_path
REPORTS_DIR = _DEFAULT_RUNTIME.reports_dir


def load_label_config(path: str | Path = LABELS_CONFIG_PATH) -> dict:
    """Read labels.yaml and flatten it into {raw label: class}, a drop set, and min_words."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    mapping: dict[str, str] = {}
    for class_name, raw_labels in raw["classes"].items():
        for raw_label in raw_labels:
            if raw_label in mapping:
                raise ValueError(f"Raw label listed twice in labels.yaml: {raw_label!r}")
            mapping[raw_label] = class_name
    drop = set(raw.get("drop", []) or [])
    if overlap := drop.intersection(mapping):
        raise ValueError(f"Raw labels both mapped and dropped: {sorted(overlap)}")
    return {"mapping": mapping, "drop": drop, "min_words": int(raw.get("min_words", 0))}
