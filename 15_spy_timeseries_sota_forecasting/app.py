"""Leakage-safe, cost-aware market forecasting demonstration."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler


SEED = 151
LAGS = 12
PURGE = 5
app = FastAPI(title="SPY Leakage-Safe Forecasting Lab", version="1.0.0")


class ScenarioRequest(BaseModel):
    volatility_multiplier: float = Field(default=1, ge=.25, le=4)
    transaction_cost_bps: float = Field(default=2, ge=0, le=100)


def make_market(n: int = 900, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic prices and log returns; never presented as historical SPY."""
    rng = np.random.default_rng(seed)
    returns = np.zeros(n)
    volatility = .009
    for t in range(1, n):
        volatility = .88 * volatility + .12 * abs(returns[t - 1]) + .0007
        returns[t] = .00025 + .10 * returns[t - 1] - .06 * (returns[t - 2] if t > 1 else 0) + rng.normal(0, min(volatility, .025))
    prices = 300 * np.exp(np.cumsum(returns))
    return prices, returns


def lagged_dataset(returns: np.ndarray, lags: int = LAGS) -> tuple[np.ndarray, np.ndarray]:
    rows, targets = [], []
    for t in range(lags, len(returns)):
        past = returns[t - lags:t]
        rows.append(np.r_[past, np.mean(past[-5:]), np.std(past[-10:])])
        targets.append(returns[t])
    return np.asarray(rows), np.asarray(targets)


def drawdown(strategy_returns: np.ndarray) -> float:
    equity = np.cumprod(1 + strategy_returns)
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak - 1))


@lru_cache(maxsize=1)
def fitted() -> tuple[object, dict[str, object], np.ndarray, np.ndarray]:
    prices, returns = make_market()
    x, y = lagged_dataset(returns)
    boundary = int(len(x) * .75)
    train_end = boundary - PURGE
    x_train, y_train = x[:train_end], y[:train_end]
    x_test, y_test = x[boundary:], y[boundary:]
    model = make_pipeline(RobustScaler(), Ridge(alpha=18)).fit(x_train, y_train)
    prediction = model.predict(x_test)
    cost = 2 / 10_000
    position = np.sign(prediction)
    turnover = np.abs(np.diff(np.r_[0, position]))
    strategy = position * y_test - turnover * cost
    std = float(np.std(strategy, ddof=1))
    sharpe = float(np.sqrt(252) * np.mean(strategy) / std) if std else 0
    report = {
        "provenance": "deterministic synthetic market process; not downloaded SPY history",
        "train_rows": len(x_train), "purged_rows": PURGE, "test_rows": len(x_test),
        "split": "chronological 75/25 with five-row purge; no shuffle",
        "model": "RobustScaler(train only) + Ridge",
        "forecast_mae": round(mean_absolute_error(y_test, prediction), 6),
        "zero_return_baseline_mae": round(mean_absolute_error(y_test, np.zeros_like(y_test)), 6),
        "directional_accuracy": round(float(np.mean(np.sign(prediction) == np.sign(y_test))), 3),
        "strategy_sharpe_after_2bps": round(sharpe, 3),
        "max_drawdown": round(drawdown(strategy), 4),
        "audit_controls": ["lagged inputs end before target", "chronological split", "purge gap", "scaler fit in training pipeline", "transaction cost charged on turnover", "synthetic provenance visible"],
        "disclaimer": "Educational synthetic backtest. Not investment advice and not evidence of future SPY performance.",
    }
    return model, report, x_test, y_test


def scenario(request: ScenarioRequest) -> dict[str, object]:
    model, _, x_test, y_test = fitted()
    prediction = model.predict(x_test) * request.volatility_multiplier
    position = np.sign(prediction)
    turnover = np.abs(np.diff(np.r_[0, position]))
    returns = position * y_test * request.volatility_multiplier - turnover * request.transaction_cost_bps / 10_000
    std = float(np.std(returns, ddof=1))
    return {
        "annualized_sharpe": round(float(np.sqrt(252) * np.mean(returns) / std), 3) if std else 0,
        "max_drawdown": round(drawdown(returns), 4),
        "total_return": round(float(np.prod(1 + returns) - 1), 4),
        "turnover_events": int(np.count_nonzero(turnover)),
        "warning": "synthetic stress scenario, not a forecast or recommendation",
    }


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>SPY Forecast Lab</title><style>
:root{font-family:Inter,system-ui;background:#07110f;color:#ecfdf5}body{margin:0}.shell{max-width:1090px;margin:auto;padding:45px 20px}.eyebrow{color:#4ade80;letter-spacing:.14em;text-transform:uppercase}h1{font-size:clamp(2.3rem,6vw,4.8rem);margin:.1em 0}.warning{border-left:5px solid #facc15;padding-left:14px;color:#fef08a}.card{background:#10231e;border:1px solid #2b5548;border-radius:19px;padding:21px;margin-top:17px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}.metric{background:#091914;border-radius:11px;padding:14px}.metric b{display:block;color:#86efac;font-size:1.6rem}label{display:grid;gap:6px}input,button{padding:12px;border-radius:9px;border:1px solid #3f7463;background:#081712;color:white}button{background:#4ade80;color:#052516;font-weight:900}</style></head><body><main class=shell><p class=eyebrow>Project 15 · leakage-safe financial ml</p><h1>SPY Forecasting Lab</h1><p class=warning>Uses synthetic market data. Nothing here is investment advice or a claim about historical SPY performance.</p><section class='card grid' id=metrics></section><section class=card><h2>Audit controls</h2><ul id=controls></ul></section><form class='card grid' id=form><label>Volatility multiplier<input id=vol type=number step=.25 min=.25 max=4 value=1></label><label>Transaction cost (bps)<input id=cost type=number min=0 max=100 value=2></label><button>Run stress scenario</button></form><section class='card grid' id=result></section></main><script>fetch('/api/report').then(r=>r.json()).then(x=>{const keys=['forecast_mae','zero_return_baseline_mae','directional_accuracy','strategy_sharpe_after_2bps','max_drawdown'];metrics.innerHTML=keys.map(k=>`<div class=metric><b>${x[k]}</b>${k.replaceAll('_',' ')}</div>`).join('');controls.innerHTML=x.audit_controls.map(c=>`<li>${c}</li>`).join('')});form.onsubmit=async e=>{e.preventDefault();const x=await fetch('/api/scenario',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({volatility_multiplier:+vol.value,transaction_cost_bps:+cost.value})}).then(r=>r.json());result.innerHTML=Object.entries(x).filter(([k,v])=>typeof v==='number').map(([k,v])=>`<div class=metric><b>${v}</b>${k.replaceAll('_',' ')}</div>`).join('')};form.requestSubmit()</script></body></html>"""


@app.get("/api/report")
def report() -> dict[str, object]:
    return fitted()[1]


@app.post("/api/scenario")
def run_scenario(request: ScenarioRequest) -> dict[str, object]:
    return scenario(request)
