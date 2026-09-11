# Use case: routing a bank complaints inbox

[Back to training and evaluation](../README.md)

A bank's shared complaints inbox receives messages about seven product areas. This demo uses
the trained classifier to select a team mailbox from each complaint's text, or send uncertain
predictions to a review queue for a person. It shows how the saved model could support an email
workflow, with true test labels available to check its decisions.

## Before you start

Run every command in this guide from the **repo root**, not from `docs/`. Follow the
[training quick start](../README.md#quick-start) first to install dependencies and produce
`models/baseline.joblib` and `data/processed/test.parquet`. The app loads the saved model;
it does not train a new one.

**Contents:** [Demo](#demo) · [Routing configuration](#routing-configuration) ·
[Serving](#serving) · [API](#api) · [Limits](#limits)

## Demo

Launch the two-page app from the repo root:

```bash
uv run streamlit run frontend/app.py
```

Senders, subjects, and dates are invented; the test-set complaint text is real and held out
from training. Subjects are generated from the complaint's opening words. Complaints you write
in a cleared compose window are your own input and have no known test label. The held-out text
has been normalised: source redactions and money masks appear as `redacted` and `money`.

### Bank email workflow demo

Pick one of 100 sampled test complaints, see its class tag, and use **Paste** to load the compose
window, or paste or write your own complaint. Send it to watch the model choose a team mailbox
or flag it for a person. A boxed scoreboard at the top tracks arrivals, automatic routes,
correct predictions among checked automatically routed test emails, and emails sent to a person.

**Clear** resets the run, compose fields, and attached test label. Use it before entering your
own complaint after a test example: editing a pasted test complaint currently retains its
original label, so the resulting correctness badge would not validate the new text.

Each handled email gets a card with its sender, a **Routed to** pill in the team's colour,
confidence, and a correct or wrong badge when a test label is available. When sent to a person,
the card still shows the model's guess and whether that guess was right. Expand **Read more**
for the complaint, routing steps, and a `predictor.predict` call preview with its returned fields.
The preview abbreviates long input; the actual prediction uses the full message. The displayed
time measures the predictor call, excluding the animation delays between routing steps.

### Inbox

The whole frozen test set—**1,388 emails**—is routed in one batch and cached
until the cache is cleared or the app restarts. The top tiles show:

| Tile | Saved baseline result at threshold 0.75 |
| --- | --- |
| Total emails | 1,388 |
| Model accuracy, all emails | 82.5%, including guesses on flagged emails |
| Accuracy when routed | 95.0%: 574 correct out of 604 automatically routed |
| Flagged for a person | 784 |

Filter by team folder or **Needs a person**. The table shows 25 emails per page, with **Previous**
and **Next** controls. Flag and confidence tooltips explain the 0.75 threshold. Use the picker
below the table to read one email in full, including its prediction, destination, and true label.
The API's `/inbox` and `/inbox/next` endpoints provide the fake mail source
described in the [API section](#api). The Streamlit app calls the Python modules directly and
does not require the FastAPI server.

<!-- Screenshots pending from Andres. Add these files under docs/, then uncomment this block.
<p>
  <img src="demo-workflow.png" width="48%" alt="Bank email workflow demo with scoreboard, compose window, and routed email cards">
  <img src="demo-inbox.png" width="48%" alt="Inbox with test-set metrics, folder filter, and paginated emails">
</p>
-->

## Routing configuration

The predictor loads `configs/routing.yaml` with the saved model. It defines one team mailbox
per class under `destinations` and a `review_queue` for complaints below the confidence threshold.
Every model class must have a mailbox. Each prediction includes a `destination` field containing
the selected team mailbox or review queue. These are demo addresses, not live integrations.

The five configuration files work together: `runtime.yaml` sets paths, `labels.yaml` defines
the label map, `baseline.yaml` sets model options, `serving.yaml` selects the saved model and
review threshold and input-length limit, and `routing.yaml` maps classes to mailboxes. All live under `configs/`.

## Serving

One predictor, three doors. `predictor.py` loads the saved model once, applies the same text
normalisation as training, and returns the product, confidence, class probabilities, whether
the complaint needs a person, and its destination mailbox. The routing model powers all three
doors; the Streamlit demo shows it handling individual emails and a complete inbox.

| Audience | Door | Command |
| --- | --- | --- |
| Applications | FastAPI `POST /predict` | `uv run uvicorn complaints.api:app --reload` |
| People | Streamlit app, workflow demo and Inbox pages | `uv run streamlit run frontend/app.py` |
| AI agents | MCP tools over stdio | `uv run --group mcp python -m complaints.mcp_server` |

### API

Start the FastAPI server with the command above, then run the request below in another terminal.

| Endpoint | Returns |
| --- | --- |
| `POST /predict` | Product, confidence, review flag, class probabilities, and `destination` mailbox |
| `GET /health` | Model name, threshold, classes, and the routing table as `destinations` plus `review_queue` |
| `GET /inbox?n=20&seed=42` | Fake mail source: a seeded sample of held-out complaints wrapped as emails; `n` accepts 1–200 |
| `GET /inbox/next` | One email per call, cycling through the default 20-email sample with seed 42 |

The mail-source endpoints return email fields and the true class, without predictions. A fixed
seed reproduces the sampled complaints and generated senders; dates are relative to request
time. The `/inbox/next` cursor is shared within each server process and resets on restart.

`POST /predict` requires nonblank text and accepts at most 20,000 characters by default, set by
`max_text_chars` in `configs/serving.yaml`.

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The bank charged me fees I do not recognize and nobody has resolved my complaint."}'
```

```json
{
  "product": "Bank account",
  "confidence": 0.3496,
  "needs_review": true,
  "probabilities": {
    "Bank account": 0.3496,
    "Credit card": 0.1787,
    "Mortgage": 0.1433,
    "...": "..."
  }
}
```

The review threshold lives in `configs/serving.yaml`, currently 0.75. Confidence below the
threshold selects the review queue; confidence at or above it selects the predicted team.
Restart running serving processes after changing the threshold, mailboxes, or saved model so
the predictor and cached inbox results reload. The response above preserves the real saved-model
output for that text (with abbreviated probabilities); the current API also includes
`"destination": "complaints-review@bank.example"` for this reviewed complaint.

### Limits

The API has no authentication and is for a local demo only. The model is trained to read banking
complaints; other input should go to a person. It flags confidence below 0.75 for review, but
has no separate out-of-domain detector, so unrelated text is not guaranteed to be flagged.
Routing selects a destination mailbox; the demo does not send real email.

