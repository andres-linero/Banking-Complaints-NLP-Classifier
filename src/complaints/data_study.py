"""Study the raw complaints before designing the label map and text cleaning.

Run with: uv run python -m complaints.data_study

Writes reports/data_study/audit.json and reports/data_study/study.md, and
prints the markdown so the findings can be read straight from the terminal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from complaints.config import (
    RUNTIME_CONFIG_PATH,
    TARGET_COLUMN,
    TEXT_COLUMN,
    load_runtime_config,
)
from complaints.ingest import (
    ISSUE_COLUMN,
    REDACTION_PATTERN,
    audit,
    load_raw,
)


def build_study(df: pd.DataFrame) -> str:
    """Render the study as markdown: audit tables plus the label and text checks."""
    report = audit(df)
    text = df[TEXT_COLUMN].fillna("")
    words = text.str.split().str.len()
    sections: list[str] = ["# Raw data study\n"]

    sections.append(f"Rows: {report['shape']['rows']} · Columns: {report['shape']['columns']}\n")

    sections.append("## Nulls, empties and junk literals (after normalisation)\n")
    sections.append(
        _table(
            ["Column", "Nulls", "Unique"],
            [[col, report["nulls"][col], report["unique_values"][col]] for col in df.columns],
        )
    )

    dup = report["duplicates"]
    sections.append("## Duplicates\n")
    sections.append(
        _table(
            ["Check", "Count"],
            [
                ["Identical rows", dup["full_rows"]],
                ["Repeated Complaint ID", dup["complaint_id"]],
                ["Repeated description", dup["description"]],
            ],
        )
    )

    sections.append("## Raw labels\n")
    sections.append(
        _table(
            ["Banking Product", "Rows", "Median words"],
            [
                [label, count, int(words[df[TARGET_COLUMN] == label].median())]
                for label, count in report["labels"].items()
            ],
        )
    )

    sections.append("## Issue ID against product\n")
    sections.append(
        "Does each Issue ID belong to a single product? If so the issue is a nested "
        "sub-label and is a strong feature or a second target.\n"
    )
    per_issue = df.groupby(ISSUE_COLUMN)[TARGET_COLUMN].nunique()
    sections.append(
        _table(
            ["Check", "Count"],
            [
                ["Issue IDs", int(per_issue.shape[0])],
                ["Issue IDs tied to exactly one product", int((per_issue == 1).sum())],
                ["Issue IDs spanning two or more products", int((per_issue > 1).sum())],
            ],
        )
    )

    t = report["text"]
    sections.append("## Complaint text\n")
    sections.append(
        _table(
            ["Statistic", "Words", "Chars"],
            [
                [k, round(t["words"][k]), round(t["chars"][k])]
                for k in ("min", "1%", "25%", "50%", "75%", "99%", "max")
            ],
        )
    )
    sections.append(
        _table(
            ["Check", "Rows"],
            [
                ["Under 10 words", t["under_10_words"]],
                ["Over 512 words (BERT truncation)", t["over_512_words"]],
                ["Uppercase only", t["uppercase_only"]],
                ["Non-ASCII characters", t["non_ascii"]],
                ["Contain redaction tokens (XXXX)", t["with_redaction"]],
                ["Contain money masks ({$...})", t["with_money_mask"]],
            ],
        )
    )
    sections.append(
        f"Average redaction tokens per complaint: {t['redaction_tokens_per_row']:.1f}\n"
    )

    sections.append("## Most common redaction forms\n")
    forms = text.str.findall(rf"\b{REDACTION_PATTERN}[\w/]*\b").explode().value_counts().head(8)
    sections.append(_table(["Token", "Occurrences"], [[k, int(v)] for k, v in forms.items()]))

    sections.append("## Shortest complaints\n")
    shortest = df.loc[words.nsmallest(12).index, [TARGET_COLUMN, TEXT_COLUMN]]
    sections.append(_table(["Product", "Text"], shortest.values.tolist()))

    sections.append("## Repeated descriptions\n")
    repeated = text[text.duplicated(keep=False)].value_counts().head(5)
    sections.append(
        _table(
            ["Times", "Text (first 120 chars)"],
            [[int(v), k[:120].replace("\n", " ")] for k, v in repeated.items()],
        )
    )

    return "\n".join(sections)


def _table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        lines.append("| " + " | ".join(str(cell).replace("|", "\\|") for cell in row) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Study the raw complaints CSV.")
    parser.add_argument("--config", default=str(RUNTIME_CONFIG_PATH))
    parser.add_argument("--data-path")
    parser.add_argument("--out-dir")
    args = parser.parse_args()

    runtime = load_runtime_config(args.config)
    df = load_raw(args.data_path or runtime.raw_data_path)
    out_dir = Path(args.out_dir or runtime.reports_dir / "data_study")
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "audit.json").write_text(json.dumps(audit(df), indent=2), encoding="utf-8")
    study = build_study(df)
    (out_dir / "study.md").write_text(study, encoding="utf-8")
    print(study)


if __name__ == "__main__":
    main()
