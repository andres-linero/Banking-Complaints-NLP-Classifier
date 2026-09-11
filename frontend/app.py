"""Door for people: the Streamlit app, two pages.

Run with: uv run streamlit run frontend/app.py
    Bank email workflow demo   send test-set emails or your own and watch them get routed
    Inbox      the whole test set as a mailbox, routed into team folders
"""

from __future__ import annotations

import time
from datetime import datetime

import pandas as pd
import streamlit as st

from complaints.inbox import REVIEW_FOLDER, Email, make_inbox, make_subject, route_inbox, summarize
from complaints.live import RunRecord, handle_email, handle_text, scoreboard
from complaints.predictor import get_predictor

STEP_DELAY = 0.5  # seconds between the lines of a live status box

# One colour per team, fixed order, plus the review queue. Used for the sidebar cards and tiles.
TEAM_COLORS = {
    "Bank account": "#2a78d6",
    "Credit card": "#eb6834",
    "Credit reporting": "#1baf7a",
    "Debt collection": "#eda100",
    "Loan": "#e87ba4",
    "Mortgage": "#008300",
    "Student loan": "#4a3aa7",
    REVIEW_FOLDER: "#e34948",
}
INBOX_SEED = 42
INBOX_SIZE = 100
PAGE_SIZE = 25  # emails per page in the Inbox table


@st.cache_resource
def predictor():
    return get_predictor()


@st.cache_data(show_spinner=False)
def inbox_emails():
    return make_inbox(n=INBOX_SIZE, seed=INBOX_SEED)


def _department_card_html(name: str, address: str) -> str:
    """A small card for one destination: department name, mailbox underneath in a light colour."""
    color = TEAM_COLORS.get(name, "#64748b")
    return (
        '<div style="padding:8px 10px;border:1px solid #e2e8f0;border-radius:8px;'
        f'background:#ffffff;border-left:5px solid {color};">'
        f'<div style="font-size:13px;font-weight:600;color:#0f172a;">{name}</div>'
        f'<div style="font-size:11px;color:#94a3b8;font-family:ui-monospace,Menlo,monospace;'
        f'word-break:break-all;margin-top:2px;">{address}</div>'
        "</div>"
    )


def _tile_html(label: str, value: str, color: str, note: str = "") -> str:
    """A boxed stat with a coloured top edge."""
    return (
        '<div style="flex:1 1 0;padding:10px 14px;border:1px solid #e2e8f0;border-radius:8px;'
        f'background:#ffffff;border-top:4px solid {color};">'
        f'<div style="font-size:12px;color:#64748b;">{label}</div>'
        f'<div style="font-size:26px;font-weight:600;color:#0f172a;line-height:1.2;">{value}</div>'
        f'<div style="font-size:11px;color:#94a3b8;">{note}</div></div>'
    )


def _tiles_html(tiles: list[tuple[str, str, str, str]]) -> str:
    return (
        '<div style="display:flex;gap:12px;">' + "".join(_tile_html(*t) for t in tiles) + "</div>"
    )


def _class_tag_html(name: str) -> str:
    """The class a test complaint belongs to, as a coloured tag under the picker."""
    color = TEAM_COLORS.get(name, "#64748b")
    return (
        '<div style="display:flex;align-items:center;gap:8px;margin:4px 0 10px 0;'
        'font-size:12px;color:#64748b;">Class in the test set:'
        f'<span style="display:inline-block;padding:2px 10px;border-radius:12px;border:1px solid '
        f'{color};color:{color};font-weight:600;">{name}</span></div>'
    )


def _record_header_html(record: RunRecord) -> str:
    """One row per routed email: who and what on the left, the prediction on the right."""
    p = record.prediction
    department = REVIEW_FOLDER if p.needs_review else p.product
    color = TEAM_COLORS.get(department, "#64748b")
    if record.correct is None:
        verdict = ""
    else:
        ok = record.correct
        if p.needs_review:
            text = "guess was right" if ok else "guess was wrong"
        else:
            text = "correct" if ok else "wrong"
        verdict = (
            f'<span style="padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600;'
            f'background:{"#dcfce7" if ok else "#fee2e2"};color:{"#166534" if ok else "#991b1b"};">'
            f"{text}</span>"
        )
    guess = ""
    if p.needs_review:
        guess_color = TEAM_COLORS.get(p.product, "#64748b")
        guess = (
            f'<span style="font-size:12px;color:#64748b;">model guess</span>'
            f'<span style="padding:2px 8px;border-radius:10px;border:1px dashed {guess_color};'
            f'color:{guess_color};font-size:11px;">{p.product}</span>'
        )
    return (
        '<div style="display:flex;justify-content:space-between;align-items:center;gap:16px;'
        f'padding:8px 12px;border-left:5px solid {color};background:#ffffff;">'
        '<div style="min-width:0;">'
        f'<div style="font-size:13px;font-weight:600;color:#0f172a;">'
        f"Email from {record.sender}</div>"
        "</div>"
        '<div style="display:flex;align-items:center;gap:10px;white-space:nowrap;">'
        f'<span style="font-size:12px;color:#64748b;">Routed to</span>'
        f'<span style="padding:3px 10px;border-radius:12px;border:1.5px solid {color};'
        f'color:{color};font-weight:600;font-size:12px;">{department}</span>'
        f"{guess}"
        f'<span style="font-size:12px;color:#64748b;">Confidence</span>'
        f'<span style="font-size:14px;font-weight:600;color:#0f172a;">{p.confidence:.0%}</span>'
        f"{verdict}</div></div>"
    )


def _render_card(record: RunRecord, live: bool) -> None:
    """One card per email: the header, then a fold that spins while the email is routed."""
    with st.container(border=True):
        st.html(_record_header_html(record))
        label = f"Routing email from {record.sender}..." if live else "Read more"
        with st.status(label, state="running" if live else "complete", expanded=live) as box:
            st.markdown("> " + record.body.replace("\n", " "))
            steps_tab, call_tab = st.tabs(["Routing steps", "Model call"])
            with steps_tab:
                for line in record.steps():
                    if live:
                        time.sleep(STEP_DELAY)
                    st.write(line)
            with call_tab:
                st.code(record.call_snippet(), language="python")
            if live:
                box.update(label="Read more", state="complete", expanded=True)


def _clear_run() -> None:
    """Button callback: runs before the page redraws, so widget-backed keys may be reset here."""
    state = st.session_state
    state.records = []
    state.cursor = 0
    for key in ("compose_from", "compose_subject", "compose_body"):
        state[key] = ""
    state.compose_label = state.compose_id = None


def live_page() -> None:
    st.title("Bank email workflow demo")
    st.markdown(
        "Real complaints from the frozen test set, dressed as emails. Send one and the saved "
        "model classifies it, routes it to the team mailbox on the left, or flags it for a "
        "person when unsure. Each test email carries its true label, so every routing is checked."
    )
    model = predictor()
    state = st.session_state
    state.setdefault("records", [])
    state.setdefault("cursor", 0)
    pending: list[RunRecord] = []

    top = st.container()
    right = st.container()
    left = st.container()

    with right:
        emails = inbox_emails()
        for key, default in (
            ("compose_from", ""),
            ("compose_subject", ""),
            ("compose_body", ""),
            ("compose_label", None),
            ("compose_id", None),
        ):
            state.setdefault(key, default)

        # Top bar: pick a real complaint from the test set and paste it into the email below.
        st.html(
            '<div style="font-size:12px;font-weight:600;letter-spacing:.06em;'
            'text-transform:uppercase;color:#64748b;margin-bottom:4px;">From the test set</div>'
        )
        pick_col, paste_col = st.columns([4, 1], vertical_alignment="bottom")
        choice = pick_col.selectbox(
            "Test complaints",
            options=list(range(len(emails))),
            index=state.cursor % len(emails),
            format_func=lambda i: f"{emails[i].sender} · {emails[i].subject}",
            label_visibility="collapsed",
        )

        def _paste(i: int = choice) -> None:
            e = emails[i]
            state.compose_from, state.compose_subject, state.compose_body = (
                e.sender,
                e.subject,
                e.body,
            )
            state.compose_label, state.compose_id = e.true_product, e.id
            state.cursor = i

        paste_col.button("Paste", on_click=_paste, width="stretch")
        picked = emails[choice]
        st.html(_class_tag_html(picked.true_product))

        # The email itself, laid out like a mail client's compose window.
        with st.expander("New message", expanded=True):
            st.html(
                '<div style="display:flex;justify-content:space-between;align-items:center;">'
                '<div style="font-size:15px;font-weight:600;color:#0f172a;">New message</div>'
                '<div style="font-size:12px;color:#64748b;">To: complaints@bank.example</div></div>'
                '<div style="font-size:11px;color:#94a3b8;margin-top:2px;">The model decides which '
                "team mailbox this is forwarded to.</div>"
            )
            st.text_input("From", key="compose_from", placeholder="you@example.com")
            st.text_input("Subject", key="compose_subject", placeholder="What is this about?")
            st.text_area(
                "Message",
                key="compose_body",
                height=180,
                placeholder="Describe the complaint, or paste one from the test set above.",
            )
            send_col, clear_col = st.columns([3, 1], vertical_alignment="bottom")
            if send_col.button("Send", type="primary", width="stretch"):
                body = state.compose_body.strip()
                if body:
                    if state.compose_label:
                        email = Email(
                            id=state.compose_id or "typed",
                            sender=state.compose_from or "you",
                            subject=state.compose_subject or make_subject(body),
                            received_at=datetime.now().replace(microsecond=0).isoformat(),
                            body=body,
                            true_product=state.compose_label,
                        )
                        pending.append(handle_email(email, model))
                    else:
                        pending.append(handle_text(body, model))
            clear_col.button("Clear", on_click=_clear_run, width="stretch")

    with top:
        board = scoreboard(state.records + pending)
        st.html(
            _tiles_html(
                [
                    ("Arrived", str(board["arrived"]), "#2a78d6", "emails sent so far"),
                    (
                        "Routed automatically",
                        str(board["automatic"]),
                        "#008300",
                        f"confidence {model.review_threshold:.0%} or more",
                    ),
                    (
                        "Correct among checked",
                        f"{board['correct']} of {board['checked']}",
                        "#4a3aa7",
                        "test emails only, they carry a label",
                    ),
                    (
                        "Sent to a person",
                        str(board["review"]),
                        TEAM_COLORS[REVIEW_FOLDER],
                        f"confidence under {model.review_threshold:.0%}",
                    ),
                ]
            )
        )

    with left:
        for record in state.records:
            _render_card(record, live=False)
        for record in pending:
            _render_card(record, live=True)
            state.records.append(record)

    st.caption(
        "The words 'redacted' and 'money' stand in for details the source anonymised. "
        "Senders, subjects, and dates are invented; the complaint text is real."
    )


@st.cache_data(show_spinner="Routing the whole test set...")
def routed_test_set():
    """Every held-out complaint as an email, routed once and cached for the session."""
    return route_inbox(make_inbox(n=None, seed=INBOX_SEED), get_predictor())


def inbox_v2_page() -> None:
    model = predictor()
    threshold = model.review_threshold
    st.title("Inbox")
    st.caption(
        "The whole test set as a mailbox: every email is a real banking complaint the model never "
        "trained on, with an invented sender, subject, and date. Flagged emails went to a person."
    )
    routed = routed_test_set()
    stats = summarize(routed)
    total = stats["total"]
    routed_ok = stats["automatic_correct"]
    guess_ok = sum(
        1 for r in routed if r.correct
    )  # routed right + flagged where the guess was right
    st.html(
        _tiles_html(
            [
                (
                    "Total emails",
                    f"{total:,}",
                    "#2a78d6",
                    "held-out complaints, the frozen test set",
                ),
                (
                    "Model accuracy, all emails",
                    f"{guess_ok / total:.1%}",
                    "#4a3aa7",
                    f"{guess_ok:,} of {total:,} guesses right, flagged ones included",
                ),
                (
                    "Accuracy when routed",
                    f"{routed_ok / stats['automatic']:.1%}",
                    "#008300",
                    f"{routed_ok} of {stats['automatic']} routed automatically",
                ),
                (
                    "Flagged for a person",
                    f"{stats['review']:,}",
                    TEAM_COLORS[REVIEW_FOLDER],
                    f"{stats['review'] / total:.0%} of all, confidence under {threshold:.0%}",
                ),
            ]
        )
    )

    folders = ["All", REVIEW_FOLDER] + [f for f in stats["folders"] if f != REVIEW_FOLDER]
    pick = st.segmented_control("Folder", folders, default="All") or "All"
    state = st.session_state
    if state.get("inbox_folder") != pick:
        state.inbox_folder, state.inbox_page = pick, 1
    shown = [r for r in routed if pick == "All" or r.folder == pick]
    pages = max(1, -(-len(shown) // PAGE_SIZE))
    state.inbox_page = min(max(1, state.get("inbox_page", 1)), pages)

    lo = (state.inbox_page - 1) * PAGE_SIZE
    hi = min(lo + PAGE_SIZE, len(shown))
    view = shown[lo:hi]

    table = pd.DataFrame(
        {
            "Flag": ["⚑" if r.prediction.needs_review else "" for r in view],
            "Received": [r.email.received_at.replace("T", " ") for r in view],
            "From": [r.email.sender for r in view],
            "Subject": [r.email.subject for r in view],
            "Routed to": [r.folder for r in view],
            "Confidence": [r.prediction.confidence for r in view],
        }
    )
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        height=38 + 35 * max(len(view), 1),
        column_config={
            "Flag": st.column_config.TextColumn(
                width="small",
                help=(
                    f"Flagged when confidence is under {threshold:.0%}: the email goes to a "
                    "person instead of a team."
                ),
            ),
            "Received": st.column_config.TextColumn(help="Invented timestamp for the demo."),
            "From": st.column_config.TextColumn(help="Invented sender for the demo."),
            "Subject": st.column_config.TextColumn(help="First words of the real complaint."),
            "Routed to": st.column_config.TextColumn(
                help="The team mailbox the model forwarded to, or Needs a person."
            ),
            "Confidence": st.column_config.ProgressColumn(
                format="percent",
                min_value=0,
                max_value=1,
                help=(
                    f"The model's probability for its top class. Under {threshold:.0%} means "
                    "flagged."
                ),
            ),
        },
    )

    prev_col, label_col, next_col = st.columns([1, 6, 1], vertical_alignment="center")
    if prev_col.button("Previous", disabled=state.inbox_page <= 1, width="stretch"):
        state.inbox_page -= 1
        st.rerun()
    if next_col.button("Next", disabled=state.inbox_page >= pages, width="stretch"):
        state.inbox_page += 1
        st.rerun()
    label_col.markdown(
        f'<div style="text-align:center;color:#64748b;font-size:13px;">'
        f"Emails {lo + 1 if shown else 0}–{hi} of {len(shown):,} · page {state.inbox_page} of "
        f"{pages}</div>",
        unsafe_allow_html=True,
    )

    choice = st.selectbox(
        "Open an email from this page",
        options=list(range(len(view))),
        index=None,
        placeholder="Pick an email to read it in full",
        format_func=lambda i: f"{view[i].email.sender} · {view[i].email.subject}",
    )
    if choice is not None:
        r = view[choice]
        color = TEAM_COLORS.get(r.folder, "#64748b")
        st.html(
            f'<div style="padding:12px 16px;border:1px solid #e2e8f0;border-left:5px solid {color};'
            'border-radius:8px;background:#ffffff;">'
            f'<div style="font-size:16px;font-weight:600;color:#0f172a;">{r.email.subject}</div>'
            f'<div style="font-size:12px;color:#64748b;margin-top:2px;">From {r.email.sender} · '
            f"{r.email.received_at.replace('T', ' ')} · complaint {r.email.id}</div>"
            f'<div style="font-size:13px;color:#0f172a;margin-top:10px;">{r.email.body}</div>'
            f'<div style="font-size:12px;color:#475569;margin-top:10px;">Predicted '
            f"<b>{r.prediction.product}</b> at {r.prediction.confidence:.0%} · forwarded to "
            f"{r.prediction.destination} · true label <b>{r.email.true_product}</b> · "
            f"{'correct' if r.correct else 'wrong'}</div></div>"
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
        st.subheader("Routing email")
        rows = [(c, model.routing["destinations"][c]) for c in model.classes]
        rows.append(("Needs a person", model.routing["review_queue"]))
        st.html(
            '<div style="display:flex;flex-direction:column;gap:6px;">'
            + "".join(_department_card_html(name, addr) for name, addr in rows)
            + "</div>"
        )

    pages = [
        st.Page(live_page, title="Bank email workflow demo", default=True),
        st.Page(inbox_v2_page, title="Inbox"),
    ]
    st.navigation(pages).run()


if __name__ == "__main__":
    main()
