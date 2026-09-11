# Banking Complaints Classifier

![Model](https://img.shields.io/badge/model-TF--IDF%20%2B%20logistic%20regression-2563eb)
![Accuracy](https://img.shields.io/badge/test%20accuracy-0.825-059669)
![Macro F1](https://img.shields.io/badge/macro%20F1-0.806-059669)
![Tracking](https://img.shields.io/badge/tracking-MLflow-d97706)
![Tests](https://img.shields.io/badge/tests-pytest-7c3aed)

Train and evaluate a model that classifies consumer banking complaints into seven product
classes using TF-IDF and logistic regression. The pipeline freezes the data split, trains on
the training rows, and measures accuracy and routing confidence on held-out complaints.

This project ships the routing model and a demo showing it at work on an inbox. This README
covers the training pipeline, results, and reproducibility. The
[bank inbox use-case guide](docs/inbox-use-case.md) covers the demo and serving interfaces.

**Contents:**
[How it works](#how-it-works) ·
[Results](#results) ·
[Stages and files](#stages-and-files) ·
[Quick start](#quick-start) ·
[Inbox use case](docs/inbox-use-case.md) ·
[Development](#development) ·
[Project layout](#project-layout)

## How it works

The data path cleans the complaints and writes a reproducible train/test split. The model path
cross-validates on the training rows, fits the final model on those rows, and evaluates it on
the held-out test set. The saved model is then available to the serving interfaces.

<img src="docs/pipeline.svg" alt="Training map: data path from the raw CSV through ingest, clean, and split; model path through vectorize, classifier, evaluate, and serve; training and evaluation tracked in MLflow" width="100%">

**How the model is trained**

- **Clean before anything learns.** Anonymised tokens and money masks are collapsed, very short and
  duplicate complaints are dropped. Of 17 raw labels, 16 map onto 7 classes in `labels.yaml`;
  the single row labelled Other financial service is dropped.
- **Keep test rows out of training.** 80 / 20, stratified by class, seed 42: 5,551 training rows
  and 1,388 test rows. The Complaint ID assignment is written to disk. Regenerating the split
  reproduces it when the cleaned input, row order, and settings are unchanged.
- **TF-IDF turns text into numbers.** Each complaint becomes a vector over up to 50,000 words and word
  pairs, weighted so common words count less and rare ones count more. Fitted on train rows only.
- **Logistic regression draws the boundaries.** One weight per term per class. It outputs a
  probability for each of the 7 classes, and the highest one is the prediction.
- **Class weights are balanced.** Bank account has 14 times more rows than Loan, so Loan mistakes
  cost more during training to stop the model ignoring it.
- **5-fold cross-validation before the final fit.** The train rows are scored five ways to check
  the settings, then the model is fitted once on all of them and saved.
- **One confidence threshold.** Evaluate sweeps it on the test set and reports the trade between
  how many complaints route automatically and how often they are right.

Training and evaluation log settings, scores, and artifacts to MLflow by default;
`--no-mlflow` disables tracking. Data study, cleaning, and splitting write local reports.

## Results

Saved baseline results on 1,388 complaints held out from model fitting.

| Metric | Value | Note |
| --- | --- | --- |
| Accuracy | 0.825 | Share of all test complaints classified correctly, including those flagged for review |
| Macro F1 | 0.806 | Every class counts the same, big or small |
| Threshold 0.55 | 71% routed at 90.7% accuracy | First tested cutoff reaching at least 90% routed accuracy |
| Threshold 0.75 | 44% routed at 95.0% accuracy | The setting in `serving.yaml`; the rest go to a person |
| Weakest class | Loan, F1 0.68 | 32 test rows, errors spread across five other classes |

<p>
  <img src="reports/evaluate/figures/confusion_matrix.png" width="48%" alt="Confusion matrix on the test set">
  <img src="reports/evaluate/figures/threshold_curve.png" width="48%" alt="Coverage and routed accuracy against the confidence threshold">
</p>

Rows of the confusion matrix are the true class. The two largest error counts are Credit card
classified as Bank account (44) and Bank account classified as Credit card (30). The threshold
curve shows the observed trade: higher cutoffs route fewer complaints, with higher accuracy
among those routed in this evaluation.

The threshold sweep uses this same test set, so its routed-accuracy figures describe the observed
tradeoff rather than an independent test of a threshold chosen on separate validation data.

Per-class scores, the full confusion matrix, and the most confident wrong predictions are in
`reports/evaluate/`.

## Stages and files

Data study, cleaning, splitting, training, and evaluation are runnable modules under
`src/complaints/`. Each accepts `--config`, defaulting to `configs/runtime.yaml`; explicit
command-line paths override the YAML. Ingest and predictor are library modules used by these
commands and the serving interfaces.

| Stage | Module | Reads | Writes |
| --- | --- | --- | --- |
| 1 · Ingest | `ingest.py` | `data/raw/complaints_banking_2023.csv` | Nothing. Library used by the next two stages |
| 1 · Data study | `data_study.py` | Raw CSV | `reports/data_study/audit.json`, `study.md` |
| 2 · Clean | `clean.py` | Raw CSV, `configs/labels.yaml` | `data/processed/clean.parquet`, `reports/cleaning/report.json` |
| 3 · Split | `split.py` | `clean.parquet` | `train.parquet`, `test.parquet`, `reports/split/` |
| 4 · Train | `train.py` | `train.parquet`, `configs/baseline.yaml` | `models/baseline.joblib`, `reports/train/baseline.json`, MLflow run |
| 5 · Evaluate | `evaluate.py` | `test.parquet`, `models/baseline.joblib`; training data and model config for the learning curve | `reports/evaluate/baseline.json`, `worst_mistakes.csv`, `figures/` |
| 6 · Serve | `predictor.py` | `models/baseline.joblib`, `configs/serving.yaml`, `configs/routing.yaml` | Returns product, confidence, needs_review, probabilities, destination |

Two more modules support the stages. `config.py` resolves paths and loads the label map, and
`vectorize.py` builds the TF-IDF step from `configs/baseline.yaml`.

Five YAML files under `configs/` hold the settings: `runtime.yaml` for paths, `labels.yaml`
for the label map, `baseline.yaml` for model settings, `serving.yaml` for the model name and
review threshold and input-length limit, and `routing.yaml` for one team mailbox per class plus a review queue.
See the [routing configuration](docs/inbox-use-case.md#routing-configuration) for how serving
selects a destination mailbox.

## Quick start

Requirements: Python 3.10 to 3.12 and [uv](https://docs.astral.sh/uv/). Run all commands from
the repo root. The training commands require `data/raw/complaints_banking_2023.csv`; serving
requires the saved model; the demo and fake-mail endpoints also require the frozen test set.

```bash
uv sync                                  # create .venv from uv.lock
uv run python -m complaints.data_study   # audit the raw CSV, optional
uv run python -m complaints.clean        # labels and text cleaning
uv run python -m complaints.split        # freeze train and test
uv run python -m complaints.train        # fit the baseline, log to MLflow
uv run python -m complaints.evaluate     # score the held-out test set
```

The commands print their outputs or summaries. Add `--no-learning-curve` to evaluate to skip
the extra training fits used for that figure; the other evaluation outputs are still generated.

Note:

- Reports are committed, so the numbers above can be checked without retraining. Parquet files,
  the saved model, and the MLflow folder are git-ignored and can be regenerated with the commands
  above. Runtime depends on the machine and whether the learning curve is included.
- The test set is held out from training and read by evaluate and the demo. The split is frozen
  to disk with the Complaint ID assignment, so every model is scored on identical rows.
- Cleaning is not modelling. Lowercasing and n-grams are model choices in `baseline.yaml`, not
  cleaning steps.
- Confidence is the largest probability returned by the classifier. It drives the review
  threshold but is not a guarantee that an individual prediction is correct.

To browse the runs:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

## Use case: bank email routing

The saved model also powers a two-page Streamlit demo at `frontend/app.py`: send individual
complaints through an email workflow, or explore the frozen test set as a routed inbox.

See the **[bank inbox use-case guide](docs/inbox-use-case.md)** for the demo walkthrough,
launch command, mailbox configuration, API example, and local-demo limits.

## Development

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q
uv run ruff check src tests frontend
uv run ruff format --check src tests frontend
```

GitHub Actions runs the same three checks with uv on Python 3.11 for every push to `main` and
every pull request.

Reproducibility notes:

- The dataset is a local CSV with redacted narratives. Never commit credentials or unredacted
  personal data.
- The split is deterministic, seed 42, and `reports/split/assignment.csv` records which Complaint
  ID went where.
- Trained models, parquet files, MLflow runs, and the virtual environment are git-ignored.
- Optional dependency groups: `mcp` for the MCP SDK, `notebook` for Jupyter and historical
  notebook libraries, and `dev` for tests and linting.
- Transformer experiments are future work tracked in issue #9.

## Project layout

```text
src/complaints/              one module per stage, plus the serving doors
src/complaints/live.py       handles individual demo messages, times predictions, tracks scoreboard
src/complaints/inbox.py      builds fake mail from held-out complaints, batches routing and counts
frontend/app.py              Streamlit Bank email workflow demo and Inbox pages
configs/                     runtime.yaml, labels.yaml, baseline.yaml, serving.yaml, routing.yaml
configs/routing.yaml         one team mailbox per class and the review queue
tests/                       tests for every stage
reports/                     committed outputs of every stage
data/raw/                    complaints_banking_2023.csv, the dataset
data/processed/              parquet files, git-ignored
models/  mlruns/             saved model and MLflow runs, git-ignored
notebooks/                   original exploration notebook
docs/pipeline.svg            the training map above
docs/inbox-use-case.md       demo walkthrough, routing configuration, API, and limits
pyproject.toml               metadata and dependency groups
uv.lock                      pinned lockfile used by uv sync
.github/workflows/ci.yml     lint and test workflow
```

## Notebook role

The notebook is the exploration record: EDA, preprocessing experiments, model trials, and the
original write-up. The reusable implementation lives in `src/complaints`. Opening the notebook needs its own
dependencies: `uv sync --group notebook`.
