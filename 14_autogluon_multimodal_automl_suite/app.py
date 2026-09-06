"""Lightweight multimodal AutoML benchmark with transparent late fusion."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SEED = 141
app = FastAPI(title="Multimodal AutoML Suite", version="1.0.0")


class ProductRequest(BaseModel):
    price: float = Field(default=48, ge=1, le=1000)
    rating: float = Field(default=4.3, ge=1, le=5)
    review: str = Field(default="excellent quality and fast delivery", max_length=300)
    image_brightness: float = Field(default=.7, ge=0, le=1)
    image_contrast: float = Field(default=.55, ge=0, le=1)


POSITIVE = {"excellent", "great", "love", "fast", "quality", "perfect"}
NEGATIVE = {"bad", "broken", "slow", "poor", "return", "hate"}


def text_features(text: str) -> list[float]:
    words = [word.strip(".,!?").lower() for word in text.split()]
    return [sum(word in POSITIVE for word in words), sum(word in NEGATIVE for word in words), len(words)]


def make_multimodal(n: int = 1000, seed: int = SEED) -> tuple[dict[str, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    price = np.clip(rng.lognormal(3.8, .55, n), 5, 600)
    rating = rng.uniform(1, 5, n)
    positive_words = rng.poisson(np.maximum(rating - 2.2, .1))
    negative_words = rng.poisson(np.maximum(3.8 - rating, .1))
    word_count = rng.integers(4, 45, n)
    brightness = rng.beta(4, 2, n)
    contrast = rng.beta(3, 3, n)
    latent = 1.25 * (rating - 3) + .7 * positive_words - .85 * negative_words + 1.2 * (brightness - .5) + .5 * (contrast - .5) - .002 * (price - 70) + rng.normal(0, .8, n)
    y = (latent > 0).astype(int)
    return {
        "tabular": np.c_[price, rating],
        "text": np.c_[positive_words, negative_words, word_count],
        "image_summary": np.c_[brightness, contrast],
        "late_fusion": np.c_[price, rating, positive_words, negative_words, word_count, brightness, contrast],
    }, y


@lru_cache(maxsize=1)
def benchmark() -> tuple[object, dict[str, object]]:
    modalities, y = make_multimodal()
    indices = np.arange(len(y))
    train, test = train_test_split(indices, test_size=.25, stratify=y, random_state=SEED)
    leaderboard = []
    fusion_model = None
    for name, x in modalities.items():
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=800, random_state=SEED))
        model.fit(x[train], y[train])
        probabilities = model.predict_proba(x[test])[:, 1]
        predictions = (probabilities >= .5).astype(int)
        leaderboard.append({"modality": name, "roc_auc": round(roc_auc_score(y[test], probabilities), 3), "balanced_accuracy": round(balanced_accuracy_score(y[test], predictions), 3), "features": x.shape[1]})
        if name == "late_fusion":
            fusion_model = model
    leaderboard.sort(key=lambda row: row["roc_auc"], reverse=True)
    return fusion_model, {"dataset": "deterministic synthetic product multimodal data", "rows": len(y), "positive_rate": round(float(y.mean()), 3), "leaderboard": leaderboard, "selection_protocol": "single shared stratified 75/25 split", "autogluon_status": "optional extension, not installed by default"}


def product_row(request: ProductRequest) -> np.ndarray:
    positive, negative, length = text_features(request.review)
    return np.array([[request.price, request.rating, positive, negative, length, request.image_brightness, request.image_contrast]])


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>Multimodal AutoML</title><style>
:root{font-family:Inter,system-ui;background:#0f0b18;color:#faf5ff}body{margin:0}.shell{max-width:1080px;margin:auto;padding:45px 20px}.eyebrow{color:#e879f9;letter-spacing:.14em;text-transform:uppercase}h1{font-size:clamp(2.3rem,6vw,4.8rem);margin:.1em 0}.card{background:#21152d;border:1px solid #553668;border-radius:19px;padding:21px;margin-top:17px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}.row{display:grid;grid-template-columns:2fr repeat(3,1fr);gap:12px;padding:12px;border-bottom:1px solid #553668}.row b{color:#f0abfc}label{display:grid;gap:6px}input,textarea,button{padding:11px;border-radius:9px;border:1px solid #754889;background:#150c1e;color:white}button{background:#d946ef;color:#29042e;font-weight:900}@media(max-width:650px){.row{grid-template-columns:1fr}}</style></head><body><main class=shell><p class=eyebrow>Project 14 · multimodal comparison</p><h1>Multimodal AutoML Suite</h1><p>Tabular, text, image-summary, and late-fusion models evaluated on the exact same split.</p><section class=card><div id=board></div></section><form class='card grid' id=form><label>Price<input id=price type=number value=48></label><label>Rating<input id=rating type=number step=.1 min=1 max=5 value=4.3></label><label>Brightness<input id=brightness type=number step=.05 min=0 max=1 value=.7></label><label>Contrast<input id=contrast type=number step=.05 min=0 max=1 value=.55></label><label>Review<textarea id=review>excellent quality and fast delivery</textarea></label><button>Run fusion model</button></form><section class=card id=result>Submit a product example.</section></main><script>fetch('/api/benchmark').then(r=>r.json()).then(x=>board.innerHTML=x.leaderboard.map((m,i)=>`<div class=row><b>${i+1}. ${m.modality}</b><span>AUC ${m.roc_auc}</span><span>Balanced acc. ${m.balanced_accuracy}</span><span>${m.features} features</span></div>`).join(''));form.onsubmit=async e=>{e.preventDefault();const x=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({price:+price.value,rating:+rating.value,review:review.value,image_brightness:+brightness.value,image_contrast:+contrast.value})}).then(r=>r.json());result.innerHTML=`<h2>${Math.round(x.positive_probability*100)}% positive probability</h2><p>${x.extracted_text_features.positive_words} positive and ${x.extracted_text_features.negative_words} negative words detected.</p>`}</script></body></html>"""


@app.get("/api/benchmark")
def report() -> dict[str, object]:
    return benchmark()[1]


@app.post("/api/predict")
def predict(request: ProductRequest) -> dict[str, object]:
    model, _ = benchmark()
    row = product_row(request)
    positive, negative, length = text_features(request.review)
    return {"positive_probability": round(float(model.predict_proba(row)[0, 1]), 4), "extracted_text_features": {"positive_words": positive, "negative_words": negative, "word_count": length}, "warning": "educational synthetic-data model"}
