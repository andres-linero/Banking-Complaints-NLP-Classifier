# Bank email workflow demo

A two-page Streamlit app that shows the routing model at work on a bank's complaint inbox.
Senders, subjects, and dates are invented. The complaint text is real: it comes from the
1,388 held-out test complaints the model never trained on, so every routing decision can be
checked against the true label.

```bash
uv run streamlit run frontend/app.py
```

## Page 1: routing one email at a time

<img src="demo-workflow.png" width="100%" alt="Bank email workflow demo: scoreboard tiles, compose window, and routed email cards">

1. Pick a complaint from the test-set picker, or write your own in the compose window.
2. Press **Send**. The model reads the text and returns a product class and a confidence.
3. Confidence at or above 0.75: the email is routed to that team's mailbox.
   Below 0.75: it goes to the review queue for a person, with the model's guess shown for reference.
4. Each email gets a card. Open **Read more** to see the message, the routing steps, and the
   timed `predictor.predict` call with its raw result.

The scoreboard at the top counts what has arrived, what was routed automatically, how many of
those were correct, and how many went to a person.

## Page 2: the whole inbox

<img src="demo-inbox.png" width="100%" alt="Inbox page: test-set metrics, folder filter, and paginated emails">

All 1,388 test emails are routed once and shown as an inbox. The tiles at the top are the
saved evaluation numbers: overall accuracy, accuracy of the automatically routed emails, and
how many were flagged for a person. Filter by team folder or **Needs a person**, page through
the list, and open any email to read it with its prediction, destination, and true label.

## What is behind it

| Piece | Where |
| --- | --- |
| The model | `models/baseline.joblib`, TF-IDF plus logistic regression |
| Review threshold | `configs/serving.yaml`, `review_threshold: 0.75` |
| Team mailboxes | `configs/routing.yaml`, one address per class plus a review queue |
| Fake mail | `src/complaints/inbox.py` wraps test complaints as emails |
| Live handling | `src/complaints/live.py` calls the predictor and records each step |
| The pages | `frontend/app.py` |

The same predictor serves the FastAPI endpoint and the MCP tools, so what the demo shows is
exactly what an application or an agent would get.
