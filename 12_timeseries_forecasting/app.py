"""Chronological forecasting with naive baseline and empirical intervals."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error


SEED = 121
LAGS = 14
app = FastAPI(title="TimePulse Forecasting", version="1.0.0")


class ForecastRequest(BaseModel):
    horizon: int = Field(default=14, ge=1, le=60)


def make_series(n: int = 420, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return 120 + .08 * t + 18 * np.sin(2 * np.pi * t / 7) + 9 * np.sin(2 * np.pi * t / 30) + rng.normal(0, 4, n)


def supervised(series: np.ndarray, lags: int = LAGS) -> tuple[np.ndarray, np.ndarray]:
    return np.array([series[i - lags:i] for i in range(lags, len(series))]), series[lags:]


@lru_cache(maxsize=1)
def fitted() -> tuple[Ridge, np.ndarray, dict[str, object]]:
    series = make_series()
    x, y = supervised(series)
    split = int(len(x) * .8)
    model = Ridge(alpha=5).fit(x[:split], y[:split])
    prediction = model.predict(x[split:])
    naive = x[split:, -7]  # seasonal naive: same weekday one week ago
    residuals = y[split:] - prediction
    report = {
        "data": "deterministic seasonal synthetic demand",
        "train_rows": split,
        "test_rows": len(y) - split,
        "ridge_mae": round(mean_absolute_error(y[split:], prediction), 3),
        "seasonal_naive_mae": round(mean_absolute_error(y[split:], naive), 3),
        "interval_90_half_width": round(float(np.quantile(np.abs(residuals), .9)), 3),
        "split": "first 80% train → final 20% test; no shuffle",
    }
    return model, residuals, report


def forecast(horizon: int) -> list[dict[str, float | int]]:
    model, residuals, _ = fitted()
    history = list(make_series())
    width = float(np.quantile(np.abs(residuals), .9))
    result = []
    for step in range(1, horizon + 1):
        point = float(model.predict(np.array(history[-LAGS:]).reshape(1, -1))[0])
        history.append(point)
        result.append({"step": step, "point": round(point, 2), "lower_90": round(point - width, 2), "upper_90": round(point + width, 2)})
    return result


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>TimePulse</title><style>
:root{font-family:Inter,system-ui;background:#07131a;color:#ecfeff}body{margin:0}.shell{max-width:1070px;margin:auto;padding:45px 20px}.eyebrow{color:#34d399;letter-spacing:.14em;text-transform:uppercase}h1{font-size:clamp(2.3rem,6vw,4.8rem);margin:.1em 0}.card{background:#10242a;border:1px solid #28525c;border-radius:19px;padding:21px;margin-top:18px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}.metric{background:#091a20;border-radius:11px;padding:14px}.metric b{display:block;color:#6ee7b7;font-size:1.7rem}input,button{padding:12px;border-radius:9px;border:1px solid #39717c;background:#08191e;color:white}button{background:#34d399;color:#03271b;font-weight:900}.point{display:grid;grid-template-columns:1fr 2fr;gap:10px;padding:8px;border-bottom:1px solid #28525c}.range{background:#164e63;border-radius:99px;padding:4px 10px;text-align:center}</style></head><body><main class=shell><p class=eyebrow>Project 12 · temporal validation</p><h1>TimePulse Forecasting</h1><p>Chronological holdout, a seasonal naive challenger, and residual-based uncertainty.</p><section class='card grid' id=metrics></section><form class=card id=form><label>Forecast horizon <input id=horizon type=number min=1 max=60 value=14></label> <button>Forecast</button></form><section class=card id=forecast></section></main><script>fetch('/api/report').then(r=>r.json()).then(x=>metrics.innerHTML=Object.entries(x).filter(([k,v])=>typeof v==='number').map(([k,v])=>`<div class=metric><b>${v}</b>${k.replaceAll('_',' ')}</div>`).join(''));form.onsubmit=async e=>{e.preventDefault();const x=await fetch('/api/forecast',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({horizon:+horizon.value})}).then(r=>r.json());forecast.innerHTML=x.forecast.map(p=>`<div class=point><b>t + ${p.step}: ${p.point}</b><span class=range>${p.lower_90} — ${p.upper_90}</span></div>`).join('')};form.requestSubmit()</script></body></html>"""


@app.get("/api/report")
def report() -> dict[str, object]:
    return fitted()[2]


@app.post("/api/forecast")
def forecast_api(request: ForecastRequest) -> dict[str, object]:
    return {"forecast": forecast(request.horizon), "interval": "empirical absolute-residual 90% band; not guaranteed conditional coverage"}
