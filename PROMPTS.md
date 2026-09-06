# Prompt and reproduction catalog

Source: [Vijay Eranti's data science examples](https://github.com/dlmastery/data_science_examples/blob/main/PROMPTS.md). The short source prompts below are preserved for attribution; the implementation notes record how this reproduction makes each result independently verifiable.

## 00 — Dynamic Todo Workspace

> Build a modern end-to-end dynamic todo application with an excellent UX and reasonable assumptions.

Reproduction direction: add task persistence, priority and status controls, productivity summaries, health checks, and browser-testable interactions.

## 01 — NYC Taxi Trip Prediction

> Build an end-to-end data science project using the Kaggle NYC taxi challenge, including data, training, deployment, CRISP-DM, an interactive map, and trip estimation.

Reproduction direction: use a deterministic NYC-like demo dataset by default, haversine and temporal features, train-only preprocessing, regression baselines, and transparent provenance.

## 02 — Nano LLM Transformer

> Build a simple laptop-size LLM and chatbot with modern primitives, CRISP-DM, research iteration, and a data-science admin dashboard.

Reproduction direction: implement a genuinely small causal transformer, fixed vocabulary/corpus, training diagnostics, guarded generation, and CPU-safe defaults.

## 03 — Customer Segmentation

> Build a clustering project using a popular Kaggle-style dataset, CRISP-DM, research iteration, and a detailed dashboard.

Reproduction direction: synthesize RFM data with declared segments, scale features inside the pipeline, compare cluster counts with silhouette score, and expose actionable personas.

## 04 — Associative Pattern Mining

> Build an association-pattern-mining project using a popular Kaggle-style dataset and popular methods with CRISP-DM and a detailed dashboard.

Reproduction direction: implement Apriori transparently, rank rules by support/confidence/lift, and let users change thresholds.

## 05 — Data Science Skills Lab

> Demonstrate data-science and analytics skills on appropriate popular datasets with CRISP-DM, and render live execution as a friendly interactive dashboard.

Reproduction direction: provide safe built-in laboratories for profiling, visualization, preprocessing, modeling, evaluation, and interpretation.

## 06 — Anomaly Detection

> Build anomaly detection using a popular dataset and popular methods with CRISP-DM, research iteration, and a detailed dashboard.

Reproduction direction: compare statistical and isolation-based detectors, preserve class imbalance, and emphasize precision-recall metrics.

## 07 — AutoML with AutoGluon

> Illustrate AutoML with AutoGluon across data-science tasks, CRISP-DM, research iteration, and a detailed dashboard.

Reproduction direction: provide a lightweight offline model tournament by default and an optional AutoGluon adapter, with identical folds and honest timing.

## 08 — Data Science Visual Mastery

> Teach Naive Bayes, model evaluation, differential calculus/gradient descent, and the chain rule/backpropagation using mathematical and visual intuition, simulations, quizzes, and interview questions.

Reproduction direction: create a static-host-ready interactive curriculum with manipulable decision thresholds and gradient steps.

## 09 — FlowForge DAG Engine

> Demonstrate strong TypeScript practices with a complicated end-to-end full-stack project.

Reproduction direction: implement typed DAG validation, topological execution, failure states, run history, and an interactive workflow canvas.

## 10 — CRISP-DM Master's Curriculum

> Build a textbook-quality end-to-end CRISP-DM project with quizzes, EDA, preprocessing, clustering, anomalies, supervised learning, association rules, LSH, and synthesis.

Reproduction direction: make every phase traceable to executable evidence in one curriculum application.

## 11 — Enterprise Data Science Audit

> Perform an advanced data-science audit for all projects and provide the detailed report in a website.

Reproduction direction: inspect repository structure and Python ASTs for seeds, leakage risks, unsafe metrics, temporal splits, provenance, and tests without claiming formal certification.

## 12 — Time-Series Forecasting

> Build a time-series forecasting website with detailed CRISP-DM steps and admin dashboards.

Reproduction direction: use a seasonal demo series, chronological validation, naive and learned baselines, residual checks, and empirical prediction intervals.

## 13 — CRISP-DM NYC Taxi Audit Platform

> Build a transparent NYC taxi CRISP-DM platform with EDA, explainability, clustering, model research, ablations, APIs, MLOps, code auditing, and a detailed report.

Reproduction direction: integrate a governed taxi pipeline, experiment registry, permutation explanations, drift checks, model card, and load-test script.

## 14 — Multimodal AutoML Suite

> Build a spectacular AutoML demonstration using current AutoGluon capabilities across several data-science modalities.

Reproduction direction: create deterministic tabular, text, and image-summary tasks with late fusion; keep AutoGluon optional because its installation is heavyweight.

## 15 — SPY Time-Series Forecasting

> Design an end-to-end SPY forecasting platform using state-of-the-art machine learning, strict leakage prevention, a data-science auditor, detailed specifications, UML, and rendered diagrams.

Reproduction direction: accept user-supplied SPY history or generate explicitly synthetic market data, use lagged returns, purged walk-forward evaluation, trading costs, uncertainty, and no performance promises.

## Repository-level instructions

The instructor additionally requires all artifacts in a public GitHub repository, extensive screenshots, preserved prompts and implementation plans, browser testing, a top-level project tour, and a thorough YouTube code-and-UX walkthrough linked from `README.md`.
