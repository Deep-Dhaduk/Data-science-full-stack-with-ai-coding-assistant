"""One executable report spanning all required CRISP-DM learning stages."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SEED = 110
app = FastAPI(title="CRISP-DM Master's Curriculum", version="1.0.0")


@lru_cache(maxsize=1)
def lifecycle_report() -> dict[str, object]:
    data = load_wine()
    x, y = data.data, data.target
    scaled = StandardScaler().fit_transform(x)  # Descriptive unsupervised branch only.
    clusters = KMeans(3, n_init=15, random_state=SEED).fit_predict(scaled)
    anomalies = IsolationForest(contamination=.05, random_state=SEED).fit_predict(scaled)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, stratify=y, random_state=SEED)
    supervised = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=120, random_state=SEED))
    supervised.fit(x_train, y_train)
    balanced = balanced_accuracy_score(y_test, supervised.predict(x_test))
    # Lightweight random-hyperplane LSH signature for a demonstrable sub-linear index primitive.
    rng = np.random.default_rng(SEED)
    planes = rng.normal(size=(10, scaled.shape[1]))
    signatures = (scaled @ planes.T > 0).astype(int)
    buckets = len({tuple(row) for row in signatures})
    high_alcohol = x[:, 0] > np.median(x[:, 0])
    high_color = x[:, 9] > np.median(x[:, 9])
    joint_support = float(np.mean(high_alcohol & high_color))
    confidence = joint_support / float(np.mean(high_alcohol))
    return {
        "business_understanding": {"goal": "teach traceable decisions across a complete analytical lifecycle", "success_measure": "every phase links to executable evidence"},
        "data_understanding": {"dataset": "scikit-learn Wine", "rows": len(x), "features": x.shape[1], "classes": len(np.unique(y)), "missing": int(np.isnan(x).sum())},
        "preparation": {"supervised_pipeline": "split first; scaler and classifier fit on training rows only", "unsupervised_branch": "standardized full data for descriptive learning"},
        "clustering": {"k": 3, "silhouette": round(float(silhouette_score(scaled, clusters)), 3)},
        "anomaly_detection": {"flagged": int(np.sum(anomalies == -1)), "method": "Isolation Forest"},
        "supervised_learning": {"metric": "holdout balanced accuracy", "value": round(float(balanced), 3), "test_rows": len(y_test)},
        "association_rule": {"rule": "high alcohol → high color intensity", "support": round(joint_support, 3), "confidence": round(confidence, 3)},
        "lsh": {"hyperplanes": 10, "occupied_buckets": buckets, "purpose": "approximate candidate retrieval"},
        "deployment": {"interface": "FastAPI report and responsive evidence dashboard", "limitations": "teaching demo, not a production wine decision system"},
    }


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>CRISP-DM Mastery</title><style>
:root{font-family:Inter,system-ui;background:#0c1017;color:#f8fafc}body{margin:0}.shell{max-width:1100px;margin:auto;padding:45px 20px}.eyebrow{color:#facc15;letter-spacing:.14em;text-transform:uppercase}h1{font-size:clamp(2.3rem,6vw,4.8rem);margin:.1em 0}.phase{background:#171e29;border:1px solid #384352;border-radius:18px;padding:20px;margin-top:14px}.phase h2{color:#fde047;text-transform:capitalize}.evidence{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px}.fact{background:#0e141d;border-radius:10px;padding:13px}.fact b{display:block;color:#fde68a}</style></head><body><main class=shell><p class=eyebrow>Project 10 · executable methodology</p><h1>CRISP-DM Master's Curriculum</h1><p>Each phase is backed by code-generated evidence rather than static methodology prose.</p><div id=phases></div></main><script>fetch('/api/lifecycle').then(r=>r.json()).then(x=>phases.innerHTML=Object.entries(x).map(([phase,facts],i)=>`<section class=phase><h2>${i+1}. ${phase.replaceAll('_',' ')}</h2><div class=evidence>${Object.entries(facts).map(([k,v])=>`<div class=fact><b>${k.replaceAll('_',' ')}</b>${v}</div>`).join('')}</div></section>`).join(''))</script></body></html>"""


@app.get("/api/lifecycle")
def lifecycle() -> dict[str, object]:
    return lifecycle_report()
