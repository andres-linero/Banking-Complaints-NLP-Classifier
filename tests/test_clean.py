import pandas as pd
import pytest

from complaints.clean import clean, load_label_config, map_labels, normalize_text
from complaints.ingest import REQUIRED_COLUMNS


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("on XX/XX/XX22 I opened", "on redacted I opened"),
        ("XXXX XXXX charged me", "redacted charged me"),
        ("fee of {$12.00} on XX00", "fee of money on redacted"),
        ("Debit/HoldXX/XX/XXXXXXXX", "Debit/Hold redacted"),
        ("  too   many   spaces  ", "too many spaces"),
        ("no masks here", "no masks here"),
    ],
)
def test_normalize_text(raw: str, expected: str) -> None:
    assert normalize_text(raw) == expected


def test_load_label_config_flattens_yaml(tmp_path) -> None:
    path = tmp_path / "labels.yaml"
    path.write_text(
        "classes:\n  Card:\n    - Credit card\n    - Prepaid card\ndrop:\n  - Other\nmin_words: 3\n"
    )

    config = load_label_config(path)

    assert config["mapping"] == {"Credit card": "Card", "Prepaid card": "Card"}
    assert config["drop"] == {"Other"}
    assert config["min_words"] == 3


def test_load_label_config_rejects_duplicate_raw_label(tmp_path) -> None:
    path = tmp_path / "labels.yaml"
    path.write_text("classes:\n  A:\n    - x\n  B:\n    - x\n")

    with pytest.raises(ValueError, match="listed twice"):
        load_label_config(path)


def test_map_labels_fails_on_unknown_raw_label() -> None:
    df = pd.DataFrame({"Banking Product": ["Mortgage", "Mystery"]})

    with pytest.raises(ValueError, match="missing from labels.yaml"):
        map_labels(df, {"Mortgage": "Mortgage"}, set())


def test_clean_applies_every_step_in_order() -> None:
    long_text = "I was charged twice for the same purchase and nobody will help me"
    rows = [
        ["C1", "1/1/2023", "Credit card", "I_1", long_text, "TX", "1", "Closed"],
        ["C2", "1/2/2023", "Prepaid card", "I_1", long_text + " XXXX", "TX", "1", "Closed"],
        ["C3", "1/3/2023", "Credit card", "I_1", "see attachment", "TX", "1", "Closed"],
        ["C4", "1/4/2023", "Other", "I_2", long_text + " again", "TX", "1", "Closed"],
        [
            "C5",
            "1/5/2023",
            "Mortgage",
            "I_3",
            "My mortgage escrow was miscalculated by the bank",
            "TX",
            "1",
            "Closed",
        ],
    ]
    df = pd.DataFrame(rows, columns=list(REQUIRED_COLUMNS))
    config = {
        "mapping": {"Credit card": "Card", "Prepaid card": "Card", "Mortgage": "Mortgage"},
        "drop": {"Other"},
        "min_words": 5,
    }

    result, report = clean(df, config)

    assert result["Complaint ID"].tolist() == ["C1", "C2", "C5"]
    assert result["product"].tolist() == ["Card", "Card", "Mortgage"]
    assert result["text"].iloc[1] == long_text + " redacted"
    assert report["steps"] == {
        "raw_rows": 5,
        "after_drop_labels": 4,
        "after_drop_null_text": 4,
        "after_drop_short": 3,
        "after_drop_duplicates": 3,
    }
    assert report["classes"] == {"Card": 2, "Mortgage": 1}


def test_labels_yaml_covers_every_raw_label_in_the_real_data() -> None:
    """The real labels.yaml must account for every product in the real CSV."""
    from complaints.config import LABELS_CONFIG_PATH, RAW_DATA_PATH
    from complaints.ingest import load_raw

    if not RAW_DATA_PATH.exists():
        pytest.skip("raw CSV not available")

    df = load_raw(RAW_DATA_PATH)
    config = load_label_config(LABELS_CONFIG_PATH)

    raw_labels = set(df["Banking Product"].dropna().unique())
    covered = set(config["mapping"]) | config["drop"]

    assert raw_labels == covered, (
        f"uncovered: {raw_labels - covered}, stale: {covered - raw_labels}"
    )
    assert set(config["mapping"].values()) == {
        "Bank account",
        "Credit card",
        "Credit reporting",
        "Mortgage",
        "Debt collection",
        "Student loan",
        "Loan",
    }
