import pandas as pd
import pytest

from banking_complaints.data import group_rare_classes, normalize_product_labels


def test_group_rare_classes_groups_low_support_values() -> None:
    df = pd.DataFrame(
        {
            "Banking Product": ["Mortgage", "Mortgage", "Credit card"],
            "Complaint Description": ["a", "b", "c"],
        }
    )

    result = group_rare_classes(df, min_count=2)

    assert result["Product Grouped"].tolist() == ["Mortgage", "Mortgage", "Other"]


def test_group_rare_classes_requires_target_column() -> None:
    with pytest.raises(ValueError, match="Missing target column"):
        group_rare_classes(pd.DataFrame({"Complaint Description": ["a"]}))


def test_normalize_product_labels_merges_taxonomy_versions() -> None:
    df = pd.DataFrame(
        {
            "Banking Product": [
                "Credit card",
                "Credit card or prepaid card",
                "Checking or savings account",
            ]
        }
    )

    result = normalize_product_labels(df)

    assert result["Normalized Product"].tolist() == [
        "Credit card / prepaid card",
        "Credit card / prepaid card",
        "Bank account / checking / savings",
    ]
