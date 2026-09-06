"""Imbalanced anomaly-detection comparison with labeled synthetic telemetry."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler


SEED = 66
app = FastAPI(title="Anomaly Threat Intelligence", version="1.0.0")


class EventRequest(BaseModel):
    requests_per_second: float = Field(default=85, ge=0, le=10000)
    error_rate: float = Field(default=.04, ge=0, le=1)
    latency_ms: float = Field(default=240, ge=0, le=100000)


def make_telemetry(n_normal: int = 950, n_anomaly: int = 50, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    normal = rng.multivariate_normal([55, .018, 115], np.diag([12**2, .009**2, 25**2]), n_normal)
    anomaly = rng.multivariate_normal([130, .16, 420], np.diag([45**2, .06**2, 120**2]), n_anomaly)
    x = np.maximum(np.vstack([normal, anomaly]), 0)
    y = np.r_[np.zeros(n_normal, dtype=int), np.ones(n_anomaly, dtype=int)]
    order = rng.permutation(len(y))
    return x[order], y[order]


@lru_cache(maxsize=1)
def detector() -> tuple[StandardScaler, IsolationForest, dict[str, float]]:
    x, y = make_telemetry()
    # Unsupervised fit: labels are reserved strictly for evaluation.
    scaler = StandardScaler().fit(x)
    transformed = scaler.transform(x)
    model = IsolationForest(n_estimators=160, contamination=.05, random_state=SEED).fit(transformed)
    scores = -model.decision_function(transformed)
    prediction = (model.predict(transformed) == -1).astype(int)
    report = {
        "anomaly_rate": round(float(y.mean()), 3),
        "average_precision": round(average_precision_score(y, scores), 3),
        "precision": round(precision_score(y, prediction), 3),
        "recall": round(recall_score(y, prediction), 3),
    }
    return scaler, model, report


def score_event(event: EventRequest) -> dict[str, object]:
    scaler, model, _ = detector()
    row = np.array([[event.requests_per_second, event.error_rate, event.latency_ms]])
    score = float(-model.decision_function(scaler.transform(row))[0])
    return {"anomaly_score": round(score, 4), "flagged": bool(model.predict(scaler.transform(row))[0] == -1), "interpretation": "higher scores are more unusual relative to the demo telemetry"}


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>Anomaly Lab</title><style>
:root{font-family:Inter,system-ui;background:#120a0d;color:#fff1f2}body{margin:0}.shell{max-width:1020px;margin:auto;padding:45px 20px}.eyebrow{color:#fb7185;letter-spacing:.14em;text-transform:uppercase}h1{font-size:clamp(2.3rem,6vw,4.8rem);margin:.1em 0}.card{background:#281117;border:1px solid #642c39;border-radius:20px;padding:22px;margin-top:20px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:13px}.metric{background:#170b0f;border-radius:12px;padding:15px}.metric b{display:block;font-size:1.8rem;color:#fb7185}label{display:grid;gap:7px}input,button{padding:12px;border-radius:9px;border:1px solid #824051;background:#16090d;color:white}button{background:#fb7185;color:#310914;font-weight:900;cursor:pointer}.flag{border-left:5px solid #fb7185}</style></head><body><main class=shell><p class=eyebrow>Project 06 · imbalanced detection</p><h1>Anomaly Threat Intelligence</h1><p>Isolation scoring evaluated with precision, recall, and average precision—not misleading accuracy.</p><section class='card grid' id=metrics></section><form class='card grid' id=form><label>Requests / second<input id=rps type=number value=85></label><label>Error rate<input id=errors type=number step=.01 min=0 max=1 value=.04></label><label>Latency (ms)<input id=latency type=number value=240></label><button>Score event</button></form><section class=card id=result>Submit an event to inspect its score.</section></main><script>
fetch('/api/metrics').then(r=>r.json()).then(x=>metrics.innerHTML=Object.entries(x).map(([k,v])=>`<div class=metric><b>${v}</b>${k.replaceAll('_',' ')}</div>`).join(''));form.onsubmit=async e=>{e.preventDefault();const x=await fetch('/api/score',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({requests_per_second:+rps.value,error_rate:+errors.value,latency_ms:+latency.value})}).then(r=>r.json());result.className='card '+(x.flagged?'flag':'');result.innerHTML=`<h2>${x.flagged?'Review required':'Within learned range'}</h2><p>Anomaly score: <b>${x.anomaly_score}</b></p><p>${x.interpretation}</p>`}</script></body></html>"""


@app.get("/api/metrics")
def metrics() -> dict[str, float]:
    return detector()[2]


@app.post("/api/score")
def score(event: EventRequest) -> dict[str, object]:
    return score_event(event)
