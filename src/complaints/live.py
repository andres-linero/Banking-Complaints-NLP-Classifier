"""Demo: the live run. One record per handled message, and the running scoreboard.

Two kinds of input go through the same predictor: an email from the synthetic
inbox (real held-out complaint, so the routing can be checked against its true
label) and a message the viewer types (no label, so it is only routed). This
module holds the logic; the Streamlit page only draws it.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime

from complaints.inbox import Email
from complaints.predictor import Prediction, Predictor


@dataclass(frozen=True)
class RunRecord:
    source: str  # "inbox" or "typed"
    sender: str
    subject: str
    body: str
    handled_at: str
    prediction: Prediction
    true_product: str | None  # None for typed messages
    correct: bool | None  # None when there is no label to check
    elapsed_ms: float = 0.0  # wall time of the predictor call

    def steps(self) -> list[str]:
        """The lines the status box shows, in order, once the record is complete."""
        p = self.prediction
        lines = [
            f"Received from {self.sender}",
            f"Classified as {p.product} at {p.confidence:.0%} confidence",
        ]
        if p.needs_review:
            lines.append(f"Under the threshold, sent to a person: {p.destination}")
        else:
            lines.append(f"Forwarded to {p.destination}")
        if self.true_product is None:
            lines.append("No label to check")
        elif self.correct:
            lines.append(f"Matches the true label: {self.true_product}")
        else:
            lines.append(f"Wrong, the true label is {self.true_product}")
        return lines

    def call_snippet(self) -> str:
        """The exact call the page made and what came back, for people who want to see it."""
        text = self.body if len(self.body) <= 80 else self.body[:77] + "..."
        text = text.replace('"', "'")
        return f'predictor.predict("{text}")\n# {self.elapsed_ms:.1f} ms\n' + json.dumps(
            self.prediction.to_dict(), indent=2
        )


def _timed_predict(predictor: Predictor, text: str) -> tuple[Prediction, float]:
    start = time.perf_counter()
    prediction = predictor.predict(text)
    return prediction, (time.perf_counter() - start) * 1000


def handle_email(email: Email, predictor: Predictor, now: datetime | None = None) -> RunRecord:
    prediction, elapsed = _timed_predict(predictor, email.body)
    return RunRecord(
        source="inbox",
        sender=email.sender,
        subject=email.subject,
        body=email.body,
        handled_at=(now or datetime.now()).replace(microsecond=0).isoformat(),
        prediction=prediction,
        true_product=email.true_product,
        correct=prediction.product == email.true_product,
        elapsed_ms=elapsed,
    )


def handle_text(text: str, predictor: Predictor, now: datetime | None = None) -> RunRecord:
    from complaints.inbox import make_subject

    prediction, elapsed = _timed_predict(predictor, text)
    return RunRecord(
        source="typed",
        sender="you",
        subject=make_subject(text),
        body=text,
        handled_at=(now or datetime.now()).replace(microsecond=0).isoformat(),
        prediction=prediction,
        true_product=None,
        correct=None,
        elapsed_ms=elapsed,
    )


def scoreboard(records: list[RunRecord]) -> dict:
    """Arrived, routed automatically, correct among the labelled ones, sent to a person."""
    automatic = [r for r in records if not r.prediction.needs_review]
    checked = [r for r in automatic if r.correct is not None]
    return {
        "arrived": len(records),
        "automatic": len(automatic),
        "checked": len(checked),
        "correct": sum(1 for r in checked if r.correct),
        "review": len(records) - len(automatic),
    }
