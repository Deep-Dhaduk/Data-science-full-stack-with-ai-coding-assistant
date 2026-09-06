"""RFM customer segmentation with transparent cluster selection."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 33
FEATURES = ["recency_days", "frequency", "monetary"]
app = FastAPI(title="Customer Segmentation Studio", version="1.0.0")


class CustomerRequest(BaseModel):
    recency_days: float = Field(default=14, ge=0, le=730)
    frequency: float = Field(default=18, ge=1, le=500)
    monetary: float = Field(default=1200, ge=1, le=100000)


def make_rfm_data(n_per_segment: int = 180, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = np.array([[12, 25, 1800], [65, 10, 650], [210, 3, 160], [30, 7, 260]])
    scales = np.array([[7, 6, 350], [22, 3, 180], [55, 1.5, 60], [12, 2, 75]])
    groups = [rng.normal(center, scale, size=(n_per_segment, 3)) for center, scale in zip(centers, scales)]
    data = np.vstack(groups)
    return np.maximum(data, [0, 1, 1])


def choose_k(data: np.ndarray, candidates: range = range(2, 7)) -> tuple[int, list[dict[str, float]]]:
    scaled = StandardScaler().fit_transform(data)
    scores = []
    for k in candidates:
        labels = KMeans(k, n_init=12, random_state=SEED).fit_predict(scaled)
        scores.append({"k": k, "silhouette": round(float(silhouette_score(scaled, labels)), 4)})
    best = max(scores, key=lambda item: item["silhouette"])
    return int(best["k"]), scores


@lru_cache(maxsize=1)
def segmentation() -> tuple[Pipeline, dict[str, object]]:
    data = make_rfm_data()
    best_k, scores = choose_k(data)
    pipe = Pipeline([("scale", StandardScaler()), ("cluster", KMeans(best_k, n_init=20, random_state=SEED))])
    labels = pipe.fit_predict(data)
    profiles = []
    names = {}
    for cluster_id in range(best_k):
        rows = data[labels == cluster_id]
        recency, frequency, monetary = rows.mean(axis=0)
        if monetary > 1000 and frequency > 15:
            persona = "Champions"
        elif recency > 120:
            persona = "At risk"
        elif frequency < 6:
            persona = "Occasional"
        else:
            persona = "Growing loyalists"
        names[cluster_id] = persona
        profiles.append({
            "cluster": cluster_id,
            "persona": persona,
            "customers": int(len(rows)),
            "recency_days": round(float(recency), 1),
            "frequency": round(float(frequency), 1),
            "monetary": round(float(monetary), 2),
        })
    return pipe, {"selected_k": best_k, "selection_scores": scores, "profiles": profiles, "names": names, "rows": len(data)}


def assign(customer: CustomerRequest) -> dict[str, object]:
    pipe, report = segmentation()
    row = np.array([[customer.recency_days, customer.frequency, customer.monetary]])
    cluster = int(pipe.predict(row)[0])
    return {"cluster": cluster, "persona": report["names"][cluster]}


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>Segmentation Studio</title><style>
:root{font-family:Inter,system-ui;background:#071711;color:#e9fff5}body{margin:0}.shell{max-width:1050px;margin:auto;padding:45px 20px}.eyebrow{color:#4ade80;letter-spacing:.15em;text-transform:uppercase}h1{font-size:clamp(2.3rem,6vw,4.8rem);margin:.1em 0}.card{background:#10271d;border:1px solid #27583e;border-radius:20px;padding:22px;margin-top:20px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:13px}.persona{background:#0a1c14;border-radius:13px;padding:16px}.persona b{display:block;font-size:1.35rem;color:#4ade80}.persona small{display:block;margin-top:8px;color:#a7d9b9}label{display:grid;gap:7px;color:#a7d9b9}input,button{padding:12px;border-radius:9px;border:1px solid #397653;background:#081c13;color:white}button{background:#22c55e;color:#04210e;font-weight:900;cursor:pointer}</style></head><body><main class=shell><p class=eyebrow>Project 03 · unsupervised learning</p><h1>Customer Segmentation Studio</h1><p>RFM personas backed by scaling, reproducible K-means, and silhouette-based model selection.</p><section class='card grid' id=profiles></section><section class=card><p id=selection>Loading selection evidence…</p></section><form class='card grid' id=form><label>Days since purchase<input id=recency type=number min=0 value=14></label><label>Purchase frequency<input id=frequency type=number min=1 value=18></label><label>Lifetime value ($)<input id=monetary type=number min=1 value=1200></label><button>Assign persona</button></form><section class=card id=result>Enter a customer profile.</section></main><script>
fetch('/api/report').then(r=>r.json()).then(x=>{profiles.innerHTML=x.profiles.map(p=>`<div class=persona><b>${p.persona}</b>${p.customers} customers<small>R ${p.recency_days}d · F ${p.frequency} · M $${p.monetary}</small></div>`).join('');selection.textContent=`Selected k=${x.selected_k}. Silhouette evidence: `+x.selection_scores.map(s=>`k${s.k}: ${s.silhouette}`).join(' · ')});form.onsubmit=async e=>{e.preventDefault();const x=await fetch('/api/assign',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({recency_days:+recency.value,frequency:+frequency.value,monetary:+monetary.value})}).then(r=>r.json());result.innerHTML=`<h2>${x.persona}</h2><p>Model cluster ${x.cluster}. Treat this as a decision-support label, not an intrinsic customer identity.</p>`}</script></body></html>"""


@app.get("/api/report")
def report() -> dict[str, object]:
    result = dict(segmentation()[1])
    result.pop("names")
    result["data_provenance"] = "deterministic synthetic RFM demonstration"
    return result


@app.post("/api/assign")
def assign_customer(customer: CustomerRequest) -> dict[str, object]:
    return assign(customer)
