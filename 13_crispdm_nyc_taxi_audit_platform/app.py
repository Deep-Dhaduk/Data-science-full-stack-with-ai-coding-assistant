"""Governed NYC-like taxi model tournament with drift and explanations."""

from __future__ import annotations

from functools import lru_cache
from math import pi

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SEED = 131
FEATURES = ["distance_km", "hour_sin", "hour_cos", "rain", "airport"]
app = FastAPI(title="Governed NYC Taxi Audit Platform", version="1.0.0")


class InferenceRequest(BaseModel):
    distance_km: float = Field(default=8.5, ge=.1, le=60)
    hour: int = Field(default=18, ge=0, le=23)
    rain: bool = False
    airport: bool = False


def make_data(n: int = 1800, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    distance = np.clip(rng.gamma(2.1, 3, n), .2, 40)
    hour = rng.integers(0, 24, n)
    rain = rng.binomial(1, .16, n)
    airport = rng.binomial(1, .13, n)
    rush = (((hour >= 7) & (hour <= 9)) | ((hour >= 16) & (hour <= 19))).astype(int)
    y = 4 + distance * 3.0 + rush * 7 + rain * 3.5 + airport * 5 + .045 * distance**2 + rng.normal(0, 2.8, n)
    x = np.c_[distance, np.sin(2 * pi * hour / 24), np.cos(2 * pi * hour / 24), rain, airport]
    return x, np.maximum(y, 2)


def population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 8) -> float:
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.histogram(reference, bins=edges)[0] / len(reference)
    cur = np.histogram(current, bins=edges)[0] / len(current)
    ref, cur = np.clip(ref, 1e-5, None), np.clip(cur, 1e-5, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


@lru_cache(maxsize=1)
def experiment() -> tuple[object, dict[str, object]]:
    x, y = make_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=SEED)
    candidates = {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=2)),
        "random_forest": RandomForestRegressor(n_estimators=130, min_samples_leaf=3, random_state=SEED, n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100, max_depth=2, random_state=SEED),
    }
    leaderboard = []
    fitted_models = {}
    for name, model in candidates.items():
        model.fit(x_train, y_train)
        fitted_models[name] = model
        prediction = model.predict(x_test)
        leaderboard.append({"model": name, "mae_minutes": round(mean_absolute_error(y_test, prediction), 3), "r2": round(r2_score(y_test, prediction), 3)})
    leaderboard.sort(key=lambda row: row["mae_minutes"])
    winner_name = leaderboard[0]["model"]
    winner = fitted_models[winner_name]
    importance = permutation_importance(winner, x_test, y_test, scoring="neg_mean_absolute_error", n_repeats=5, random_state=SEED)
    explanations = sorted([{"feature": name, "importance": round(float(value), 4)} for name, value in zip(FEATURES, importance.importances_mean)], key=lambda item: item["importance"], reverse=True)
    clusters = KMeans(4, n_init=12, random_state=SEED).fit_predict(StandardScaler().fit_transform(x[:, [0, 1, 2]]))
    shifted_distance = x_test[:, 0] * 1.18
    report = {
        "provenance": "deterministic synthetic NYC-like trips",
        "rows": len(x), "train_rows": len(x_train), "test_rows": len(x_test),
        "leaderboard": leaderboard, "winner": winner_name, "permutation_importance": explanations,
        "mobility_clusters": {"k": 4, "counts": np.bincount(clusters).tolist()},
        "drift_scenario": {"distance_psi": round(population_stability_index(x_train[:, 0], shifted_distance), 4), "scenario": "test trip distances increased by 18%"},
        "audit_controls": ["seed pinned", "split before fit", "test rows excluded from training", "baseline included", "MAE and R² reported", "synthetic provenance displayed"],
        "model_card": {"intended_use": "educational trip-duration estimation", "excluded_use": "dispatch, billing, safety, or employment decisions", "known_gaps": "no street graph, live traffic, toll, borough, or event features"},
    }
    return winner, report


def inference(request: InferenceRequest) -> dict[str, object]:
    model, report = experiment()
    row = np.array([[request.distance_km, np.sin(2 * pi * request.hour / 24), np.cos(2 * pi * request.hour / 24), int(request.rain), int(request.airport)]])
    return {"duration_minutes": round(float(model.predict(row)[0]), 2), "model": report["winner"], "warning": "educational estimate from synthetic training data"}


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>Taxi Governance Lab</title><style>
:root{font-family:Inter,system-ui;background:#0c1015;color:#f7fafc}body{margin:0}.shell{max-width:1120px;margin:auto;padding:45px 20px}.eyebrow{color:#fbbf24;letter-spacing:.14em;text-transform:uppercase}h1{font-size:clamp(2.3rem,6vw,4.8rem);margin:.1em 0}.card{background:#181e26;border:1px solid #3d4653;border-radius:19px;padding:21px;margin-top:16px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}.fact{background:#0e141b;border-radius:11px;padding:14px}.fact b{display:block;color:#fcd34d;font-size:1.35rem}label{display:grid;gap:6px}input,button{padding:12px;border-radius:9px;border:1px solid #5b6674;background:#0d1319;color:white}button{background:#fbbf24;color:#2a1b02;font-weight:900}.row{display:grid;grid-template-columns:2fr 1fr 1fr;gap:12px;padding:11px;border-bottom:1px solid #3d4653}</style></head><body><main class=shell><p class=eyebrow>Project 13 · governed ml capstone</p><h1>NYC Taxi Audit Platform</h1><p>Inference, model research, explainability, drift, clustering, and control evidence in one auditable system.</p><form class='card grid' id=form><label>Distance (km)<input id=distance type=number step=.1 value=8.5></label><label>Hour<input id=hour type=number min=0 max=23 value=18></label><label><span>Conditions</span><span><input id=rain type=checkbox> Rain</span></label><label><span>Route</span><span><input id=airport type=checkbox> Airport</span></label><button>Estimate</button></form><section class=card id=result>Submit a trip.</section><section class=card><h2>Model tournament</h2><div id=board></div></section><section class='card grid' id=governance></section></main><script>fetch('/api/report').then(r=>r.json()).then(x=>{board.innerHTML=x.leaderboard.map(m=>`<div class=row><b>${m.model}</b><span>MAE ${m.mae_minutes}</span><span>R² ${m.r2}</span></div>`).join('');governance.innerHTML=`<div class=fact><b>${x.winner}</b>selected model</div><div class=fact><b>${x.drift_scenario.distance_psi}</b>distance PSI scenario</div><div class=fact><b>${x.mobility_clusters.k}</b>mobility clusters</div><div class=fact><b>${x.audit_controls.length}</b>audit controls</div>`});form.onsubmit=async e=>{e.preventDefault();const x=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({distance_km:+distance.value,hour:+hour.value,rain:rain.checked,airport:airport.checked})}).then(r=>r.json());result.innerHTML=`<h2>${x.duration_minutes} minutes</h2><p>${x.model} · ${x.warning}</p>`}</script></body></html>"""


@app.get("/api/report")
def report() -> dict[str, object]:
    return experiment()[1]


@app.post("/api/predict")
def predict(request: InferenceRequest) -> dict[str, object]:
    return inference(request)
