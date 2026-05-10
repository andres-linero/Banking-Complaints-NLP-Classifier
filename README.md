# Banking Complaints Classifier

![Model](https://img.shields.io/badge/model-spaCy%20%2B%20TF--IDF%20%2B%20LinearSVC-2563eb)
![Accuracy](https://img.shields.io/badge/accuracy-0.7890-059669)
![Dashboard](https://img.shields.io/badge/dashboard-Streamlit-d97706)
![API](https://img.shields.io/badge/api-FastAPI-64748b)
![Tests](https://img.shields.io/badge/tests-pytest-7c3aed)

Classify consumer banking complaint narratives into likely financial product categories.

`banking-complaints-classifier` turns raw complaint text into a routing suggestion such as
`Bank account issue`, `Credit card issue`, `Credit report issue`, or `Mortgage issue`.
The project started in a notebook, then was moved into a reproducible Python package with
training scripts, tests, a prediction API, and a Streamlit dashboard.

[Dashboard](#streamlit-dashboard) · [Train](#model-training) · [API](#api-serving) · [Notebook](#notebook-role)

## Current Model

The default production model is:

```text
Complaint Description
-> spaCy lemmatization
-> TF-IDF vectorization
-> LinearSVC classifier
-> normalized product category
```

Current held-out test accuracy:

```text
0.7890
```

The model predicts the normalized internal labels used for training, while the dashboard displays
friendlier labels for users.

| Internal model label | Dashboard label |
| --- | --- |
| Bank account / checking / savings | Bank account issue |
| Credit card / prepaid card | Credit card issue |
| Credit reporting | Credit report issue |
| Debt collection | Debt collection issue |
| Money transfer / money service | Money transfer issue |
| Mortgage | Mortgage issue |
| Student loan | Student loan issue |

## Streamlit Dashboard

The Streamlit dashboard is the main demo interface.

It shows:

- live complaint classification
- user-friendly prediction labels
- session prediction count and response time
- saved model accuracy
- model improvement curve
- class-level precision, recall, F1, and support
- original vs normalized training labels
- sampled training-row predictions

Run it with:

```bash
streamlit run streamlit_app.py
```

Then open:

```text
http://localhost:8501
```

## Local Environment

Create and activate the virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m spacy download en_core_web_sm
python -m pip install -e .
python -m ipykernel install --user --name banking-complaints-nlp --display-name "Python (banking complaints NLP)"
```

In VS Code, select the notebook kernel named `Python (banking complaints NLP)`.

## Model Training

Train the default sklearn model:

```bash
source .venv/bin/activate
python -m banking_complaints.train_sklearn
```

Training writes:

```text
models/complaint_classifier.joblib
reports/sklearn_metrics.json
```

The project also includes a BERT fine-tuning path for comparison:

```bash
python -m banking_complaints.train_bert \
  --data-path complaints_banking_2023.csv \
  --model-name distilbert-base-uncased \
  --output-dir models/bert_classifier \
  --report-path reports/bert_metrics.json
```

## API Serving

Serve predictions with FastAPI:

```bash
uvicorn banking_complaints.api:app --reload
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The bank charged me fees I do not recognize and nobody has resolved my complaint."}'
```

Example response:

```json
{
  "product": "Bank account / checking / savings",
  "sentiment": "negative"
}
```

## Tests

Run the automated tests:

```bash
pytest
```

Run lint checks:

```bash
ruff check src tests streamlit_app.py
```

## Project Layout

```text
src/banking_complaints/      production Python package
tests/                       automated tests
models/                      trained model artifacts
reports/                     evaluation metrics
nltk_data/                   local NLTK assets, ignored by git
streamlit_app.py             dashboard UI
NLP_Project_Andres_RL.ipynb  original exploration notebook
complaints_banking_2023.csv  local dataset
```

## Notebook Role

The notebook is the exploration record: initial EDA, preprocessing experiments, model trials,
and project explanation. The reusable implementation now lives in `src/banking_complaints`
so it can be trained, tested, served, and demonstrated outside the notebook.
