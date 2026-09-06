# Data Science Experiments with an AI Coding Assistant

Sixteen independent, reproducible projects built from Vijay Eranti's prompt catalog. Each project owns its application, data-generation or acquisition path, model code, tests, documentation, and startup command.

> **Video walkthrough:** Recording pending. The final YouTube URL and chapter timestamps will be added here after all applications pass the release audit.

Individual code-and-UX recordings are stored in their respective project folders and linked from each project README. Recordings for Projects 00–05 are currently uploaded; Projects 06–15 are marked pending until their files are added. The final combined YouTube link is still required for submission.

## Portfolio

| # | Project | Focus | Status |
|---:|---|---|---|
| 00 | [Dynamic Todo Workspace](00_dynamic_todo_workspace/) | Full-stack task analytics | Core tested |
| 01 | [NYC Taxi Trip Prediction](01_nyc_taxi_trip_prediction/) | Regression, geospatial features | Core tested |
| 02 | [Nano LLM Transformer](02_nano_llm_transformer/) | Small causal transformer | Core tested |
| 03 | [Customer Segmentation](03_customer_segmentation_clustering/) | Unsupervised clustering | Core tested |
| 04 | [Associative Pattern Mining](04_associative_pattern_mining/) | Apriori association rules | Core tested |
| 05 | [Data Science Skills Lab](05_data_science_skills_lab/) | Interactive analytical curriculum | Core tested |
| 06 | [Anomaly Detection](06_anomaly_detection/) | Isolation-based detection | Core tested |
| 07 | [AutoML Model Tournament](07_automl_autogluon/) | Model selection and ensembling | Core tested |
| 08 | [Data Science Visual Mastery](08_datascience_visual_mastery/) | Visual intuition and quizzes | Core tested |
| 09 | [FlowForge DAG Engine](09_flowforge_dag_engine/) | Typed analytical workflows | Core tested |
| 10 | [CRISP-DM Master's Curriculum](10_crispdm_masters_curriculum/) | End-to-end methodology | Core tested |
| 11 | [Enterprise DS Audit](11_enterprise_ds_audit/) | Leakage and quality governance | Core tested |
| 12 | [Time-Series Forecasting](12_timeseries_forecasting/) | Backtesting and uncertainty | Core tested |
| 13 | [NYC Taxi Audit Platform](13_crispdm_nyc_taxi_audit_platform/) | Explainable, governed ML | Core tested |
| 14 | [Multimodal AutoML Suite](14_autogluon_multimodal_automl_suite/) | Tabular/text/image fusion | Core tested |
| 15 | [SPY Forecasting](15_spy_timeseries_sota_forecasting/) | Leakage-safe financial ML | Core tested |

## Repository contract

Every numbered directory is independently runnable and must contain:

- its own `README.md` and preserved prompt;
- a deterministic experiment with an explicit random seed;
- an interactive application and JSON API;
- tests for the scientific core;
- documented data provenance, metrics, limitations, and CRISP-DM decisions;
- no dependency on another numbered project.

Generated demonstration data is clearly labeled. Reported metrics must be computed by the checked-in code; no result is copied from the reference implementation.

## Quick start

Use Python 3.10 or newer. Enter any project and follow its README. Python/FastAPI projects use the same command shape:

```bash
cd 01_nyc_taxi_trip_prediction
python -m pip install -r requirements.txt
python -m uvicorn app:app --reload --port 8001
```

Then open `http://127.0.0.1:8001`. Ports run from `8000` through `8015`.

Run the repository audit from the root:

```bash
python scripts/audit_repository.py
python scripts/run_all_tests.py
python scripts/smoke_apps.py
```

Verification currently covers 40 Python unit tests, 3 strict-TypeScript tests, a TypeScript production build, structural checks, and live HTTP startup probes for all 16 dashboards.

## Reproduction record

- [Prompt catalog](PROMPTS.md)
- [Implementation plan](IMPLEMENTATION_PLAN.md)
- [YouTube recording script and chapters](VIDEO_WALKTHROUGH.md)
- [Reference repository](https://github.com/dlmastery/data_science_examples)

## Academic integrity

This repository is an independent reproduction inspired by the provided prompts. It credits the source prompt catalog, distinguishes synthetic demonstrations from downloaded datasets, and reports only locally reproducible results.
