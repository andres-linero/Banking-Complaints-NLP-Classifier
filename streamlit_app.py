import json
import time
from html import escape
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from banking_complaints.config import (
    DATA_PATH,
    GROUPED_TARGET_COLUMN,
    MODELS_DIR,
    NORMALIZED_TARGET_COLUMN,
    REPORTS_DIR,
    TARGET_COLUMN,
    TEXT_COLUMN,
)
from banking_complaints.data import group_rare_classes, load_complaints, normalize_product_labels
from banking_complaints.predict import DEFAULT_MODEL_PATH, predict_product

MODEL_PATH = MODELS_DIR / "complaint_classifier.joblib"
METRICS_PATH = REPORTS_DIR / "sklearn_metrics.json"
PREDICTION_HISTORY_LIMIT = 10
MODEL_HISTORY = [
    {
        "version": "Notebook baseline",
        "accuracy": 0.7142,
        "model": "Cleaned TF-IDF + LogisticRegression",
    },
    {
        "version": "Normalized labels",
        "accuracy": 0.7691,
        "model": "Word/char TF-IDF + LinearSVC",
    },
    {
        "version": "spaCy lemmatized",
        "accuracy": 0.7890,
        "model": "spaCy lemma + TF-IDF + LinearSVC",
    },
]
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

EXAMPLE_COMPLAINTS = {
    "Unexpected checking account fees": (
        "My checking account was charged several overdraft fees even though I had enough "
        "money in the account. I contacted the bank but they have not reversed the charges."
    ),
    "Mortgage payment reporting issue": (
        "My mortgage servicer reported my payment as late to the credit bureaus even though "
        "I paid before the due date. They refuse to correct the record."
    ),
    "Debt collector calls": (
        "A debt collector keeps calling my phone about an account I do not recognize. I asked "
        "for proof of the debt and they continue to threaten legal action."
    ),
    "Credit card dispute": (
        "I disputed a fraudulent credit card charge, but the company closed the dispute and "
        "put the amount back on my statement without explaining the decision."
    ),
    "Money transfer delay": (
        "I sent a money transfer to a family member and the funds never arrived. The company "
        "keeps saying it is under review and will not return my money."
    ),
}


st.set_page_config(
    page_title="Banking Complaints NLP",
    page_icon="",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: #f8fafc;
        color: #0f172a;
    }
    [data-testid="stHeader"] {
        background: transparent;
        border-bottom: 0;
        box-shadow: none;
    }
    [data-testid="stToolbar"],
    [data-testid="stDecoration"] {
        display: none;
        background: transparent;
    }
    [data-testid="stDeployButton"] {
        display: none;
    }
    [data-testid="stAppViewContainer"] {
        background: #f8fafc;
    }
    .block-container {
        max-width: 1160px;
        padding-top: 0.85rem;
        padding-bottom: 1.6rem;
    }
    section[data-testid="stSidebar"] {
        background: #ffffff;
    }
    div[data-testid="stTabs"] button {
        font-size: 0.95rem;
        font-weight: 650;
        color: #475569;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #2563eb;
    }
    div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
        background-color: #2563eb;
    }
    div[data-testid="stTextArea"] textarea {
        color: #0f172a;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        background: #ffffff;
        min-height: 170px;
    }
    div[data-testid="stTextArea"] textarea:focus {
        border-color: #2563eb;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.14);
    }
    div[data-testid="stSelectbox"] {
        margin-bottom: 0.5rem;
    }
    div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        color: #0f172a;
    }
    div[data-testid="stSelectbox"] [data-baseweb="select"] span,
    div[data-testid="stSelectbox"] [data-baseweb="select"] svg {
        color: #0f172a;
        fill: #475569;
    }
    div[data-baseweb="popover"] ul,
    div[data-baseweb="popover"] li {
        background: #ffffff;
        color: #0f172a;
    }
    .stButton > button {
        border-radius: 8px;
        font-weight: 750;
        min-height: 2.6rem;
    }
    h1, h2, h3, h4, h5, h6, p, label, span {
        color: inherit;
    }
    h3 {
        color: #0f172a;
        letter-spacing: 0;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid #d8e0ea;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_dashboard_data() -> pd.DataFrame:
    df = load_complaints(DATA_PATH)
    df = normalize_product_labels(df)
    return group_rare_classes(
        df,
        min_count=50,
        target_column=NORMALIZED_TARGET_COLUMN,
        output_column=GROUPED_TARGET_COLUMN,
    )


@st.cache_resource
def load_model(path: Path = DEFAULT_MODEL_PATH):
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_data
def load_metrics(path: Path = METRICS_PATH) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner="Classifying training examples...")
def classify_training_sample(texts: tuple[str, ...]) -> list[str]:
    model = load_model(MODEL_PATH)
    if model is None:
        return []
    return model.predict(list(texts)).tolist()


def metric_value(metrics: dict, label: str) -> str:
    value = metrics.get(label)
    if isinstance(value, float):
        return f"{value:.3f}"
    return "n/a"


def class_report_frame(metrics: dict) -> pd.DataFrame:
    report = metrics.get("classification_report", {})
    rows = []
    for label, values in report.items():
        if not isinstance(values, dict) or "f1-score" not in values:
            continue
        rows.append(
            {
                "class": friendly_label(label),
                "model_label": label,
                "precision": values["precision"],
                "recall": values["recall"],
                "f1": values["f1-score"],
                "support": values["support"],
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("support", ascending=False)


def model_history_frame(metrics: dict) -> pd.DataFrame:
    history = pd.DataFrame(MODEL_HISTORY)
    if isinstance(metrics.get("accuracy"), float):
        history.loc[history.index[-1], "accuracy"] = metrics["accuracy"]
    return history


def summary_metric(metrics: dict, average_name: str, metric_name: str) -> str:
    report = metrics.get("classification_report", {})
    value = report.get(average_name, {}).get(metric_name)
    if isinstance(value, float):
        return f"{value:.3f}"
    return "n/a"


def friendly_label(label: str) -> str:
    return DISPLAY_LABELS.get(label, label)


def render_html_card_grid(cards: list[dict], height: int = 150) -> None:
    card_html = ""
    for card in cards:
        card_html += f"""
        <div class="metric-card {escape(card.get("tone", "blue"))}">
            <div class="metric-label">{escape(str(card["label"]))}</div>
            <div class="metric-value">{escape(str(card["value"]))}</div>
            <div class="metric-note">{escape(str(card.get("note", "")))}</div>
        </div>
        """

    grid_columns = min(max(len(cards), 1), 5)
    components.html(
        f"""
        <style>
            html {{
                background: #f8fafc;
            }}
            body {{
                margin: 0;
                background: #f8fafc;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            }}
            .metric-grid {{
                display: grid;
                grid-template-columns: repeat({grid_columns}, minmax(0, 1fr));
                gap: 12px;
            }}
            .metric-card {{
                box-sizing: border-box;
                min-height: 112px;
                border-radius: 8px;
                padding: 15px 16px 14px;
                background: #ffffff;
                border: 1px solid #d8e0ea;
                border-top: 4px solid var(--accent);
                box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
            }}
            .metric-card.blue {{ --accent: #2563eb; }}
            .metric-card.green {{ --accent: #059669; }}
            .metric-card.amber {{ --accent: #d97706; }}
            .metric-card.slate {{ --accent: #64748b; }}
            .metric-card.rose {{ --accent: #dc2626; }}
            .metric-label {{
                color: #64748b;
                font-size: 12px;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.04em;
            }}
            .metric-value {{
                margin-top: 7px;
                font-size: 24px;
                line-height: 1.15;
                font-weight: 800;
                overflow-wrap: anywhere;
                color: #0f172a;
            }}
            .metric-note {{
                margin-top: 7px;
                font-size: 13px;
                line-height: 1.25;
                color: #64748b;
            }}
            @media (max-width: 900px) {{
                .metric-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
            }}
            @media (max-width: 520px) {{
                .metric-grid {{ grid-template-columns: 1fr; }}
            }}
        </style>
        <div class="metric-grid">{card_html}</div>
        """,
        height=max(height, 152),
    )


def render_section_intro(title: str, text: str) -> None:
    st.markdown(
        f"""
        <div style="
            border: 1px solid #d8e0ea;
            border-left: 4px solid #2563eb;
            border-radius: 8px;
            background: #ffffff;
            padding: 0.72rem 0.9rem;
            margin: 0.25rem 0 0.9rem;
        ">
            <div style="font-size: 1.04rem; font-weight: 750; color: #0f172a;">
                {escape(title)}
            </div>
            <div style="font-size: 0.9rem; color: #64748b; margin-top: 0.12rem;">
                {escape(text)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_result(prediction: dict, latency_ms: float, text: str) -> None:
    display_product = friendly_label(prediction["product"])
    components.html(
        f"""
        <style>
            html {{
                background: #f8fafc;
            }}
            body {{
                margin: 0;
                background: #f8fafc;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            }}
            .result-card {{
                box-sizing: border-box;
                border-radius: 8px;
                border: 1px solid #d8e0ea;
                border-top: 5px solid #2563eb;
                background: #ffffff;
                padding: 18px;
                box-shadow: 0 14px 32px rgba(15, 23, 42, 0.08);
            }}
            .result-title {{
                font-size: 12px;
                font-weight: 800;
                text-transform: uppercase;
                letter-spacing: 0.04em;
                color: #2563eb;
            }}
            .result-product {{
                margin-top: 8px;
                font-size: 28px;
                line-height: 1.1;
                font-weight: 800;
                color: #0f172a;
                overflow-wrap: anywhere;
            }}
            .result-subtitle {{
                margin-top: 8px;
                color: #475569;
                font-size: 14px;
                line-height: 1.45;
            }}
            .result-grid {{
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 8px;
                margin-top: 12px;
            }}
            .pill {{
                border-radius: 8px;
                padding: 10px 11px;
                background: #f8fafc;
                border: 1px solid #e2e8f0;
            }}
            .pill-label {{
                font-size: 11px;
                font-weight: 700;
                text-transform: uppercase;
                color: #64748b;
            }}
            .pill-value {{
                margin-top: 4px;
                font-size: 15px;
                font-weight: 750;
                color: #0f172a;
            }}
            @media (max-width: 640px) {{
                .result-grid {{ grid-template-columns: 1fr; }}
                .result-product {{ font-size: 26px; }}
            }}
        </style>
        <div class="result-card">
            <div class="result-title">Predicted category</div>
            <div class="result-product">{escape(display_product)}</div>
            <div class="result-subtitle">
                This complaint most closely matches the
                {escape(display_product.lower())} product group.
            </div>
            <div class="result-grid">
                <div class="pill">
                    <div class="pill-label">Sentiment</div>
                    <div class="pill-value">{escape(prediction["sentiment"])}</div>
                </div>
                <div class="pill">
                    <div class="pill-label">Response time</div>
                    <div class="pill-value">{latency_ms:.0f} ms</div>
                </div>
                <div class="pill">
                    <div class="pill-label">Input length</div>
                    <div class="pill-value">{len(text.split())} words</div>
                </div>
            </div>
        </div>
        """,
        height=282,
    )


def ensure_prediction_state() -> None:
    if "prediction_events" not in st.session_state:
        st.session_state.prediction_events = []
    if "latest_prediction" not in st.session_state:
        st.session_state.latest_prediction = None


def record_prediction_event(text: str, prediction: dict, latency_ms: float) -> None:
    ensure_prediction_state()
    event = {
        "prediction": friendly_label(prediction["product"]),
        "model_label": prediction["product"],
        "sentiment": prediction["sentiment"],
        "latency_ms": latency_ms,
        "words": len(text.split()),
        "characters": len(text),
        "preview": text.strip().replace("\n", " ")[:120],
    }
    st.session_state.prediction_events = (
        [event] + st.session_state.prediction_events
    )[:PREDICTION_HISTORY_LIMIT]
    st.session_state.latest_prediction = {
        "text": text,
        "prediction": prediction,
        "latency_ms": latency_ms,
    }


def render_prediction_monitor(metrics: dict) -> None:
    ensure_prediction_state()
    events = st.session_state.prediction_events
    last_event = events[0] if events else {}

    latency = f"{last_event['latency_ms']:.0f} ms" if last_event else "n/a"
    render_html_card_grid(
        [
            {
                "label": "Saved Accuracy",
                "value": metric_value(metrics, "accuracy"),
                "note": "Held-out test split",
                "tone": "green",
            },
            {
                "label": "Session Calls",
                "value": len(events),
                "note": "Predictions made here",
                "tone": "blue",
            },
            {
                "label": "Last Category",
                "value": last_event.get("prediction", "none"),
                "note": "User-facing category",
                "tone": "amber",
            },
            {
                "label": "Last Latency",
                "value": latency,
                "note": "Response time",
                "tone": "slate",
            },
        ]
    )

def render_prediction_history() -> None:
    ensure_prediction_state()
    events = st.session_state.prediction_events
    if not events:
        return
    with st.expander("Recent prediction calls", expanded=False):
        st.dataframe(pd.DataFrame(events), width="stretch", hide_index=True)


def render_data_overview(df: pd.DataFrame) -> None:
    render_section_intro(
        "Training Data",
        "How many complaints the model learned from and how the product labels are shaped.",
    )

    total_rows = len(df)
    original_classes = df[TARGET_COLUMN].nunique()
    model_classes = df[GROUPED_TARGET_COLUMN].nunique()
    avg_words = df[TEXT_COLUMN].str.split().str.len().mean()

    render_html_card_grid(
        [
            {
                "label": "Complaints",
                "value": f"{total_rows:,}",
                "note": "Rows used",
                "tone": "blue",
            },
            {
                "label": "Original Classes",
                "value": original_classes,
                "note": "Raw dataset labels",
                "tone": "amber",
            },
            {
                "label": "Model Classes",
                "value": model_classes,
                "note": "After normalization",
                "tone": "green",
            },
            {
                "label": "Avg Words",
                "value": f"{avg_words:.0f}",
                "note": "Per complaint",
                "tone": "slate",
            },
            {
                "label": "Text Source",
                "value": "Complaint",
                "note": "Description column",
                "tone": "rose",
            },
        ]
    )

    left, right = st.columns([1, 1])
    with left:
        st.caption("Original product labels")
        original_counts = (
            df[TARGET_COLUMN].value_counts().rename_axis("product").reset_index(name="count")
        )
        st.bar_chart(original_counts, x="product", y="count", height=360)
    with right:
        st.caption("Normalized model labels")
        grouped_counts = (
            df[GROUPED_TARGET_COLUMN].value_counts().rename_axis("product").reset_index(name="count")
        )
        st.bar_chart(grouped_counts, x="product", y="count", height=360)


def render_model_section(metrics: dict) -> None:
    render_section_intro(
        "Model Performance",
        "The saved classifier is evaluated on a held-out test split from the training data.",
    )

    render_html_card_grid(
        [
            {
                "label": "Accuracy",
                "value": metric_value(metrics, "accuracy"),
                "note": "Overall test accuracy",
                "tone": "green",
            },
            {
                "label": "Model",
                "value": "spaCy TF-IDF SVC",
                "note": "Saved artifact type",
                "tone": "blue",
            },
            {
                "label": "Macro F1",
                "value": summary_metric(metrics, "macro avg", "f1-score"),
                "note": "Average by class",
                "tone": "amber",
            },
            {
                "label": "Weighted F1",
                "value": summary_metric(metrics, "weighted avg", "f1-score"),
                "note": "Weighted by support",
                "tone": "slate",
            },
            {
                "label": "Labels",
                "value": "Normalized",
                "note": "Merged categories",
                "tone": "rose",
            },
        ]
    )

    report_df = class_report_frame(metrics)
    if report_df.empty:
        st.info(
            "Train the model to generate class metrics: "
            "python -m banking_complaints.train_sklearn"
        )
        return

    history_df = model_history_frame(metrics)
    left, right = st.columns([0.95, 1.05], gap="large")
    with left:
        st.markdown("#### Model improvement curve")
        st.line_chart(history_df, x="version", y="accuracy", height=260)
    with right:
        st.markdown("#### Training versions")
        st.dataframe(
            history_df,
            width="stretch",
            hide_index=True,
            column_config={
                "accuracy": st.column_config.NumberColumn(format="%.4f"),
            },
        )

    st.markdown("#### Class metrics")
    st.dataframe(
        report_df,
        width="stretch",
        hide_index=True,
        column_config={
            "precision": st.column_config.NumberColumn(format="%.3f"),
            "recall": st.column_config.NumberColumn(format="%.3f"),
            "f1": st.column_config.NumberColumn(format="%.3f"),
            "support": st.column_config.NumberColumn(format="%d"),
        },
    )


def render_classifier(df: pd.DataFrame) -> None:
    render_section_intro(
        "Prediction Lab",
        "Choose an example or write a complaint, then send it through the saved model.",
    )

    training_examples = df[[TEXT_COLUMN, GROUPED_TARGET_COLUMN]].sample(
        n=min(5, len(df)),
        random_state=7,
    )
    training_options = {
        f"Training row: {text[:100]}": text
        for text in training_examples[TEXT_COLUMN].tolist()
    }
    example_options = {
        "Write my own complaint": (
            "The bank charged me fees I do not recognize and nobody has resolved my complaint."
        ),
        **EXAMPLE_COMPLAINTS,
        **training_options,
    }
    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        st.markdown("### Complaint Input")
        selected = st.selectbox(
            "Example",
            list(example_options.keys()),
        )
        text = st.text_area("Complaint narrative", value=example_options[selected], height=172)

        if st.button("Classify complaint", type="primary", width="stretch"):
            if not text.strip():
                st.warning("Add complaint text first.")
                return
            try:
                started_at = time.perf_counter()
                prediction = predict_product(text)
                latency_ms = (time.perf_counter() - started_at) * 1000
            except FileNotFoundError as exc:
                st.error(str(exc))
                return

            record_prediction_event(text, prediction, latency_ms)

    with right:
        st.markdown("### Prediction Output")
        latest = st.session_state.latest_prediction
        if latest:
            render_prediction_result(
                latest["prediction"],
                latest["latency_ms"],
                latest["text"],
            )
        else:
            components.html(
                """
                <style>
                    body {
                        margin: 0;
                        background: #f8fafc;
                        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                    }
                    .empty-card {
                        border: 1px dashed #cbd5e1;
                        border-top: 5px solid #94a3b8;
                        border-radius: 8px;
                        background: #ffffff;
                        color: #64748b;
                        padding: 28px;
                        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
                    }
                    .empty-title {
                        color: #0f172a;
                        font-size: 20px;
                        font-weight: 760;
                        margin-bottom: 8px;
                    }
                </style>
                <div class="empty-card">
                    <div class="empty-title">No prediction yet</div>
                    Choose an example or paste a complaint, then run the classifier.
                </div>
                """,
                height=176,
            )

    render_prediction_history()


def render_training_predictions(df: pd.DataFrame) -> None:
    render_section_intro(
        "Sample Predictions",
        "A sampled view comparing normalized training labels with model predictions.",
    )
    sample_size = st.slider("Sample rows", min_value=10, max_value=200, value=50, step=10)
    sample = df[[TEXT_COLUMN, TARGET_COLUMN, GROUPED_TARGET_COLUMN]].sample(
        n=min(sample_size, len(df)),
        random_state=11,
    )
    predictions = classify_training_sample(tuple(sample[TEXT_COLUMN].tolist()))
    if not predictions:
        st.info("Train the model first: python -m banking_complaints.train_sklearn")
        return

    result = sample.copy()
    result["Model Prediction"] = [friendly_label(prediction) for prediction in predictions]
    result["Model Label"] = predictions
    result["Expected Routing"] = result[GROUPED_TARGET_COLUMN].apply(friendly_label)
    result["Correct"] = result[GROUPED_TARGET_COLUMN] == result["Model Label"]
    st.dataframe(
        result[
            [
                TEXT_COLUMN,
                TARGET_COLUMN,
                "Expected Routing",
                "Model Prediction",
                "Correct",
            ]
        ],
        width="stretch",
        hide_index=True,
    )
    accuracy = result["Correct"].mean()
    st.caption(f"Sample accuracy on displayed rows: {accuracy:.1%}")


def main() -> None:
    st.markdown(
        """
        <div style="margin-bottom: 0.65rem;">
            <div style="
                font-size: 1.85rem;
                line-height: 1.1;
                font-weight: 820;
                color: #0f172a;
            ">
                Banking Complaints Classifier
            </div>
            <div style="
                max-width: 760px;
                margin-top: 0.3rem;
                font-size: 0.94rem;
                color: #64748b;
            ">
                Classify complaint narratives into likely financial product categories.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not DATA_PATH.exists():
        st.error(f"Dataset not found: {DATA_PATH}")
        return

    df = load_dashboard_data()
    metrics = load_metrics()

    render_prediction_monitor(metrics)

    tabs = st.tabs(["Prediction Lab", "Model Performance", "Training Data", "Sample Predictions"])
    with tabs[0]:
        render_classifier(df)
    with tabs[1]:
        render_model_section(metrics)
    with tabs[2]:
        render_data_overview(df)
    with tabs[3]:
        render_training_predictions(df)


if __name__ == "__main__":
    main()
