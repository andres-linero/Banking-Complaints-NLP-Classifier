from pathlib import Path

import pandas as pd

from banking_complaints.config import (
    DATA_PATH,
    GROUPED_TARGET_COLUMN,
    NORMALIZED_TARGET_COLUMN,
    TARGET_COLUMN,
    TEXT_COLUMN,
)

PRODUCT_LABEL_NORMALIZATION = {
    "Credit reporting, credit repair services, or other personal consumer reports": (
        "Credit reporting"
    ),
    "Credit reporting": "Credit reporting",
    "Credit card or prepaid card": "Credit card / prepaid card",
    "Credit card": "Credit card / prepaid card",
    "Prepaid card": "Credit card / prepaid card",
    "Checking or savings account": "Bank account / checking / savings",
    "Bank account or service": "Bank account / checking / savings",
    "Money transfer, virtual currency, or money service": "Money transfer / money service",
    "Money transfers": "Money transfer / money service",
    "Payday loan": "Payday / title / personal loan",
    "Payday loan, title loan, or personal loan": "Payday / title / personal loan",
    "Consumer Loan": "Consumer / vehicle loan",
    "Vehicle loan or lease": "Consumer / vehicle loan",
}


def load_complaints(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load the complaints CSV and validate the columns the model needs."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find data file: {path}")

    df = pd.read_csv(path)
    required = {TEXT_COLUMN, TARGET_COLUMN}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.dropna(subset=[TEXT_COLUMN, TARGET_COLUMN]).copy()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)
    return df


def normalize_product_labels(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    output_column: str = NORMALIZED_TARGET_COLUMN,
) -> pd.DataFrame:
    """Merge equivalent product labels from different complaint taxonomy versions."""
    if target_column not in df.columns:
        raise ValueError(f"Missing target column: {target_column}")

    result = df.copy()
    result[output_column] = result[target_column].map(
        lambda value: PRODUCT_LABEL_NORMALIZATION.get(value, value)
    )
    return result


def group_rare_classes(
    df: pd.DataFrame,
    min_count: int = 50,
    target_column: str = TARGET_COLUMN,
    output_column: str = GROUPED_TARGET_COLUMN,
) -> pd.DataFrame:
    """Group low-support classes into Other to make training more stable."""
    if target_column not in df.columns:
        raise ValueError(f"Missing target column: {target_column}")

    result = df.copy()
    counts = result[target_column].value_counts()
    rare_classes = set(counts[counts < min_count].index)
    result[output_column] = result[target_column].apply(
        lambda value: "Other" if value in rare_classes else value
    )
    return result
