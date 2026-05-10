from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "complaints_banking_2023.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
NLTK_DATA_DIR = PROJECT_ROOT / "nltk_data"

TEXT_COLUMN = "Complaint Description"
TARGET_COLUMN = "Banking Product"
NORMALIZED_TARGET_COLUMN = "Normalized Product"
GROUPED_TARGET_COLUMN = "Product Grouped"
CLEAN_TEXT_COLUMN = "Cleaned Complaint"
LABEL_COLUMN = "label"
