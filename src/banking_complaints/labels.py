DISPLAY_LABELS = {
    "Bank account / checking / savings": "Bank account issue",
    "Consumer / vehicle loan": "Loan issue",
    "Credit card / prepaid card": "Credit card issue",
    "Credit reporting": "Credit report issue",
    "Debt collection": "Debt collection issue",
    "Money transfer / money service": "Money transfer issue",
    "Mortgage": "Mortgage issue",
    "Other": "Other financial issue",
    "Student loan": "Student loan issue",
}


def friendly_label(label: str) -> str:
    return DISPLAY_LABELS.get(label, label)
