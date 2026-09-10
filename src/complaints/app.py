"""Door for people: the Streamlit app, two pages.

Run with: uv run streamlit run src/complaints/app.py
    Classify   paste one complaint, see where it goes and how sure the model is
    Inbox      a synthetic inbox of real held-out complaints, routed into folders
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from complaints.inbox import REVIEW_FOLDER, make_inbox, route_inbox, summarize
from complaints.predictor import Prediction, get_predictor


@st.cache_resource
def predictor():
    return get_predictor()


def render_prediction(result: Prediction, threshold: float) -> None:
    left, middle, right = st.columns([2, 2, 1])
    left.metric("Product team", result.product)
    middle.metric("Forward to", result.destination)
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


def classify_page() -> None:
    st.title("Classify one complaint")
    st.caption(
        "Paste a complaint. The model returns the product team it belongs to and how sure it is."
    )
    model = predictor()
    text = st.text_area("Complaint text", height=220, placeholder="I was charged twice for ...")
    if st.button("Classify", type="primary") and text.strip():
        render_prediction(model.predict(text), model.review_threshold)


def inbox_page() -> None:
    st.title("Inbox")
    st.caption(
        "A demo mailbox. Every email is a real banking complaint the model never trained on, "
        "wrapped in a synthetic sender, subject, and date. The model routes each one to a "
        "department folder, or to a person when it is not sure."
    )
    model = predictor()

    left, right = st.columns([1, 3])
    n = left.slider("Emails", min_value=5, max_value=100, value=20, step=5)
    seed = left.number_input("Shuffle seed", min_value=0, value=42, step=1)
    routed = route_inbox(make_inbox(n=n, seed=int(seed)), model)
    stats = summarize(routed)

    c1, c2, c3 = right.columns(3)
    c1.metric("Routed automatically", f"{stats['automatic']} of {stats['total']}")
    c2.metric(
        "Correct among those",
        f"{stats['automatic_correct']} of {stats['automatic']}" if stats["automatic"] else "-",
    )
    c3.metric("Sent to a person", stats["review"])

    folders = [REVIEW_FOLDER] + [f for f in stats["folders"] if f != REVIEW_FOLDER]
    tabs = st.tabs([f"{f} ({stats['folders'].get(f, 0)})" for f in folders])
    for tab, folder in zip(tabs, folders, strict=True):
        with tab:
            for r in (x for x in routed if x.folder == folder):
                mark = "correct" if r.correct else f"wrong, should be {r.email.true_product}"
                header = f"{r.email.subject}  ·  {r.prediction.confidence:.0%}  ·  {mark}"
                with st.expander(header):
                    st.write(
                        f"**From** {r.email.sender}  ·  **Received** {r.email.received_at}  ·  "
                        f"**Forward to** {r.prediction.destination}"
                    )
                    st.write(r.email.body)
                    st.caption(
                        f"Predicted {r.prediction.product} at {r.prediction.confidence:.0%}. "
                        f"True label {r.email.true_product}. Complaint {r.email.id}."
                    )

    st.caption(
        "The words 'redacted' and 'money' stand in for details the source anonymised. "
        "Senders, subjects, and dates are invented; the complaint text is real."
    )


def main() -> None:
    st.set_page_config(page_title="Banking complaints classifier", page_icon="🏦", layout="wide")
    model = predictor()
    with st.sidebar:
        st.subheader("Model")
        st.write(f"**{model.model_name}** · review threshold {model.review_threshold:.2f}")
        st.write("Routing:")
        st.write(
            "\n".join(f"- {c}: {model.routing['destinations'][c]}" for c in model.classes)
            + f"\n- Needs a person: {model.routing['review_queue']}"
        )

    pages = [
        st.Page(classify_page, title="Classify", default=True),
        st.Page(inbox_page, title="Inbox"),
    ]
    st.navigation(pages).run()


if __name__ == "__main__":
    main()
