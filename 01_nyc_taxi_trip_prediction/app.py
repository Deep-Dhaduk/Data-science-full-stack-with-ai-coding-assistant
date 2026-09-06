"""Leakage-safe NYC-like taxi duration and fare regression demonstration."""

from __future__ import annotations

from functools import lru_cache
from math import asin, cos, radians, sin, sqrt

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


SEED = 41
FEATURES = ["distance_km", "hour_sin", "hour_cos", "passengers", "rain", "airport"]
app = FastAPI(title="NYC Taxi Trip Lab", version="1.0.0")


class TripRequest(BaseModel):
    pickup_lat: float = Field(default=40.7580, ge=40.45, le=40.95)
    pickup_lon: float = Field(default=-73.9855, ge=-74.3, le=-73.65)
    dropoff_lat: float = Field(default=40.6413, ge=40.45, le=40.95)
    dropoff_lon: float = Field(default=-73.7781, ge=-74.3, le=-73.65)
    hour: int = Field(default=17, ge=0, le=23)
    passengers: int = Field(default=1, ge=1, le=6)
    rain: bool = False
    airport: bool = True


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance without leaking either target into the features."""
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * 6371.0088 * asin(sqrt(a))


def make_demo_data(n: int = 1600, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Return explicitly synthetic NYC-like trips for offline reproduction."""
    rng = np.random.default_rng(seed)
    distance = np.clip(rng.gamma(2.2, 2.4, n), 0.35, 32)
    hour = rng.integers(0, 24, n)
    passengers = rng.integers(1, 7, n)
    rain = rng.binomial(1, 0.18, n)
    airport = rng.binomial(1, 0.12, n)
    rush = ((hour >= 7) & (hour <= 9)) | ((hour >= 16) & (hour <= 19))
    duration = 4.5 + distance * 3.15 + rush * 7.5 + rain * 3.0 + airport * 5.0
    duration += rng.normal(0, 2.4, n)
    fare = 3.0 + distance * 2.65 + duration * 0.18 + airport * 5.0 + rng.normal(0, 1.1, n)
    x = np.column_stack(
        [distance, np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24), passengers, rain, airport]
    )
    return x, np.column_stack([np.maximum(duration, 2), np.maximum(fare, 3)])


@lru_cache(maxsize=1)
def trained_models() -> tuple[RandomForestRegressor, RandomForestRegressor, dict[str, float]]:
    x, y = make_demo_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=SEED)
    duration_model = RandomForestRegressor(n_estimators=90, min_samples_leaf=3, random_state=SEED, n_jobs=-1)
    fare_model = RandomForestRegressor(n_estimators=90, min_samples_leaf=3, random_state=SEED + 1, n_jobs=-1)
    duration_model.fit(x_train, y_train[:, 0])
    fare_model.fit(x_train, y_train[:, 1])
    pred_duration = duration_model.predict(x_test)
    pred_fare = fare_model.predict(x_test)
    metrics = {
        "duration_mae_minutes": round(mean_absolute_error(y_test[:, 0], pred_duration), 3),
        "duration_r2": round(r2_score(y_test[:, 0], pred_duration), 3),
        "fare_mae_usd": round(mean_absolute_error(y_test[:, 1], pred_fare), 3),
        "fare_r2": round(r2_score(y_test[:, 1], pred_fare), 3),
        "test_rows": len(x_test),
    }
    return duration_model, fare_model, metrics


def feature_row(trip: TripRequest) -> np.ndarray:
    distance = haversine_km(trip.pickup_lat, trip.pickup_lon, trip.dropoff_lat, trip.dropoff_lon)
    return np.array([[distance, sin(2 * np.pi * trip.hour / 24), cos(2 * np.pi * trip.hour / 24), trip.passengers, int(trip.rain), int(trip.airport)]])


def estimate(trip: TripRequest) -> dict[str, float | str]:
    duration_model, fare_model, _ = trained_models()
    x = feature_row(trip)
    return {
        "distance_km": round(float(x[0, 0]), 2),
        "duration_minutes": round(float(duration_model.predict(x)[0]), 1),
        "fare_usd": round(float(fare_model.predict(x)[0]), 2),
        "data_provenance": "deterministic synthetic NYC-like training data",
    }


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>NYC Taxi Lab</title><style>
:root{font-family:Inter,system-ui;background:#07131b;color:#eaf7f7}body{margin:0}.shell{max-width:1050px;margin:auto;padding:44px 20px}.eyebrow{color:#fbbf24;letter-spacing:.14em;text-transform:uppercase}h1{font-size:clamp(2.2rem,6vw,4.8rem);margin:.1em 0}.card{background:#10242d;border:1px solid #24505e;border-radius:20px;padding:22px;margin-top:20px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:13px}label{display:grid;gap:6px;color:#a8cbd0}input,button{padding:12px;border-radius:9px;border:1px solid #376976;background:#081a22;color:white}button{background:#fbbf24;color:#17202a;font-weight:800;cursor:pointer}.metric{padding:18px;background:#091c24;border-radius:12px}.metric b{display:block;color:#fbbf24;font-size:2rem}.notice{border-left:4px solid #fbbf24;padding-left:14px;color:#b9d7da}</style></head><body><main class=shell><p class=eyebrow>Project 01 · geospatial regression</p><h1>NYC Taxi Trip Lab</h1><p class=notice>Offline demo trained on clearly labeled synthetic NYC-like trips. No copied benchmark claims.</p><section class='card grid' id=metrics></section><form class='card grid' id=form><label>Pickup latitude<input id=plat type=number step=.0001 value=40.7580></label><label>Pickup longitude<input id=plon type=number step=.0001 value=-73.9855></label><label>Drop-off latitude<input id=dlat type=number step=.0001 value=40.6413></label><label>Drop-off longitude<input id=dlon type=number step=.0001 value=-73.7781></label><label>Hour<input id=hour type=number min=0 max=23 value=17></label><label>Passengers<input id=pax type=number min=1 max=6 value=1></label><label><span>Weather</span><span><input id=rain type=checkbox> Rain</span></label><label><span>Route type</span><span><input id=airport type=checkbox checked> Airport trip</span></label><button>Estimate trip</button></form><section class='card grid' id=result><p>Submit a route to calculate an estimate.</p></section></main><script>
fetch('/api/metrics').then(r=>r.json()).then(m=>metrics.innerHTML=Object.entries(m).map(([k,v])=>`<div class=metric><b>${v}</b>${k.replaceAll('_',' ')}</div>`).join(''));
form.onsubmit=async e=>{e.preventDefault();const body={pickup_lat:+plat.value,pickup_lon:+plon.value,dropoff_lat:+dlat.value,dropoff_lon:+dlon.value,hour:+hour.value,passengers:+pax.value,rain:rain.checked,airport:airport.checked};const x=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}).then(r=>r.json());result.innerHTML=`<div class=metric><b>${x.distance_km} km</b>Great-circle distance</div><div class=metric><b>${x.duration_minutes} min</b>Estimated duration</div><div class=metric><b>$${x.fare_usd}</b>Estimated fare</div>`}</script></body></html>"""


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": "random_forest"}


@app.get("/api/metrics")
def metrics() -> dict[str, float]:
    return trained_models()[2]


@app.post("/api/predict")
def predict(trip: TripRequest) -> dict[str, float | str]:
    return estimate(trip)
