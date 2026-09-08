# Project rebuild

Ground-up rebuild of the classifier as separately runnable stages. Run these
commands from `project-rebuild` with the repository's dependencies installed:

```bash
python -m complaints.data_study   # audit the raw CSV
python -m complaints.clean        # apply labels and cleaning; write the training table
python -m complaints.split        # freeze the stratified train/test split
python -m pytest tests            # tests for the rebuild only
```

`complaints/config.py` loads the shared paths from `configs/runtime.yaml` and the
label mapping, dropped labels, and minimum word count from `configs/labels.yaml`.
The cleaning stage uses both configurations; ingestion and the data study use the
configured raw CSV path. YAML paths resolve relative to `project-rebuild`.
The default raw CSV remains in the parent repository; outputs go inside the rebuild.

All three runnable stages accept `--config path/to/runtime.yaml`. Explicit command-line
paths override the YAML settings:

```bash
python -m complaints.clean --config configs/runtime.yaml --out-path data/processed/clean.parquet
python -m complaints.data_study --config configs/runtime.yaml --out-dir reports/data_study
```

Cleaning also accepts `--data-path`, `--labels-config`, and `--reports-dir`.
The data study also accepts `--data-path`.

The data study writes `reports/data_study/audit.json` and `study.md`.
Cleaning writes `data/processed/clean.parquet` and `reports/cleaning/report.json`.
The split writes `data/processed/train.parquet`, `data/processed/test.parquet`, and
`reports/split/report.json` plus `assignment.csv`, the Complaint ID to split record.

## Data stages

```mermaid
flowchart LR
    classDef file fill:#dbeafe,stroke:#2563eb,stroke-width:1.5px,color:#0f172a
    classDef code fill:#fef3c7,stroke:#d97706,stroke-width:1.5px,color:#0f172a

    CSV["complaints_banking_2023.csv"] --> ING["ingest.py<br/>load and type the raw rows"]
    ING --> STUDY["data_study.py<br/>audit the raw data"] --> REP["reports/data_study/"]
    ING --> CLEAN["clean.py<br/>map labels, clean text"] --> PQ["data/processed/clean.parquet"]
    PQ --> SPLIT["split.py<br/>stratified 80/20"] --> TRAIN["train.parquet"]
    SPLIT --> TEST["test.parquet"]

    class CSV,REP,PQ,TRAIN,TEST file
    class ING,STUDY,CLEAN,SPLIT code
```
