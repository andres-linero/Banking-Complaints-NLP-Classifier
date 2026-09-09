"""Door for people: a small Streamlit page around the predictor.

Run with: uv run streamlit run src/complaints/app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from complaints.predictor import Prediction, get_predictor


@st.cache_resource
def predictor():
    return get_predictor()


def render_prediction(result: Prediction, threshold: float) -> None:
    left, right = st.columns([2, 1])
    left.metric("Product team", result.product)
    right.metric("Confidence", f"{result.confidence:.0%}")

    if result.needs_review:
        st.warning(
            f"Confidence is under {threshold:.0%}, so this complaint goes to a person "
            "instead of being routed automatically."
        )
    else:
        st.success("Confident enough to route automatically.")

    table = (
        pd.DataFrame(
            {
                "product": list(result.probabilities),
                "probability": list(result.probabilities.values()),
            }
        )
        .sort_values("probability", ascending=False)
        .reset_index(drop=True)
    )
    st.bar_chart(table.set_index("product"), horizontal=True)


def main() -> None:
    st.set_page_config(page_title="Banking complaints classifier", page_icon="🏦")
    st.title("Banking complaints classifier")
    st.caption(
        "Paste a complaint. The model returns the product team it belongs to and how sure it is."
    )

    model = predictor()
    text = st.text_area("Complaint text", height=220, placeholder="I was charged twice for ...")
    if st.button("Classify", type="primary") and text.strip():
        render_prediction(model.predict(text), model.review_threshold)

    with st.sidebar:
        st.subheader("Model")
        st.write(f"**{model.model_name}** · review threshold {model.review_threshold:.2f}")
        st.write("Classes:")
        st.write("\n".join(f"- {c}" for c in model.classes))


if __name__ == "__main__":
    main()
