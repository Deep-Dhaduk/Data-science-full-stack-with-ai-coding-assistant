"""Fair offline AutoML tournament with an optional path to AutoGluon."""

from __future__ import annotations

from functools import lru_cache
from time import perf_counter

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SEED = 77
app = FastAPI(title="AutoML Model Tournament", version="1.0.0")


@lru_cache(maxsize=1)
def tournament() -> dict[str, object]:
    data = load_breast_cancer()
    models = {
        "scaled_logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED)),
        "random_forest": RandomForestClassifier(n_estimators=140, min_samples_leaf=2, random_state=SEED, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=90, random_state=SEED),
    }
    folds = StratifiedKFold(5, shuffle=True, random_state=SEED)
    leaderboard = []
    for name, model in models.items():
        started = perf_counter()
        probabilities = cross_val_predict(model, data.data, data.target, cv=folds, method="predict_proba", n_jobs=None)[:, 1]
        leaderboard.append({"model": name, "roc_auc": round(roc_auc_score(data.target, probabilities), 4), "elapsed_seconds": round(perf_counter() - started, 3)})
    leaderboard.sort(key=lambda row: row["roc_auc"], reverse=True)
    return {"task": "binary classification", "dataset": "scikit-learn Breast Cancer Wisconsin", "rows": len(data.data), "folds": 5, "metric": "out-of-fold ROC AUC", "leaderboard": leaderboard, "winner": leaderboard[0]["model"], "autogluon": "optional extension; not required for offline default"}


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>AutoML Tournament</title><style>
:root{font-family:Inter,system-ui;background:#07101d;color:#eff6ff}body{margin:0}.shell{max-width:1030px;margin:auto;padding:45px 20px}.eyebrow{color:#60a5fa;letter-spacing:.14em;text-transform:uppercase}h1{font-size:clamp(2.3rem,6vw,4.8rem);margin:.1em 0}.card{background:#101d33;border:1px solid #284872;border-radius:20px;padding:22px;margin-top:20px}.row{display:grid;grid-template-columns:2fr 1fr 1fr;gap:14px;padding:16px;border-bottom:1px solid #284872}.winner{background:#132b4c}.row b{color:#93c5fd}.tag{display:inline-block;background:#1d4ed8;border-radius:99px;padding:6px 10px;margin-right:6px}@media(max-width:600px){.row{grid-template-columns:1fr}}</style></head><body><main class=shell><p class=eyebrow>Project 07 · reproducible automl</p><h1>AutoML Model Tournament</h1><p>Same dataset, same five folds, same metric. Selection evidence before spectacle.</p><section class=card><p><span class=tag id=dataset>Loading</span><span class=tag id=method></span></p><div id=board></div></section><section class=card><h2>Why an offline default?</h2><p>AutoGluon is a valuable optional extension, but a fair scikit-learn tournament keeps this project runnable on ordinary machines and makes its selection protocol easy to inspect.</p></section></main><script>fetch('/api/leaderboard').then(r=>r.json()).then(x=>{dataset.textContent=x.dataset;method.textContent=x.folds+' folds · '+x.metric;board.innerHTML=x.leaderboard.map((m,i)=>`<div class='row ${i===0?'winner':''}'><b>${i+1}. ${m.model}</b><span>AUC ${m.roc_auc}</span><span>${m.elapsed_seconds}s</span></div>`).join('')})</script></body></html>"""


@app.get("/api/leaderboard")
def leaderboard() -> dict[str, object]:
    return tournament()
