"""Safe, executable mini-labs for core data-science skills."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


SEED = 55
app = FastAPI(title="Data Science Skills Lab", version="1.0.0")


@lru_cache(maxsize=1)
def dataset():
    return load_wine()


def execute_skill(skill: str) -> dict[str, object]:
    bunch = dataset()
    x, y = bunch.data, bunch.target
    if skill == "profile":
        return {"rows": len(x), "features": x.shape[1], "classes": len(np.unique(y)), "missing": int(np.isnan(x).sum()), "target_counts": np.bincount(y).tolist()}
    if skill == "pca":
        transformed = PCA(2, random_state=SEED).fit_transform(StandardScaler().fit_transform(x))
        return {"dimensions": 2, "preview": np.round(transformed[:12], 3).tolist(), "note": "Scaler and PCA are descriptive here; supervised evaluation uses a train-only fit."}
    if skill == "classify":
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, stratify=y, random_state=SEED)
        model = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, random_state=SEED).fit(x_train, y_train)
        prediction = model.predict(x_test)
        return {"accuracy": round(accuracy_score(y_test, prediction), 3), "balanced_accuracy": round(balanced_accuracy_score(y_test, prediction), 3), "confusion_matrix": confusion_matrix(y_test, prediction).tolist(), "test_rows": len(y_test)}
    raise KeyError(skill)


LABS = {
    "profile": ("Data profiling", "Inspect shape, class balance, and missingness before modeling."),
    "pca": ("Visual projection", "Scale thirteen chemical measurements into a two-dimensional PCA preview."),
    "classify": ("Supervised evaluation", "Use a stratified holdout and show both accuracy and balanced accuracy."),
}


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    cards = "".join(f"<button class='lab' onclick=run('{key}')><b>{title}</b><span>{description}</span></button>" for key, (title, description) in LABS.items())
    return f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>DS Skills Lab</title><style>
:root{{font-family:Inter,system-ui;background:#071523;color:#eaf5ff}}body{{margin:0}}.shell{{max-width:1050px;margin:auto;padding:45px 20px}}.eyebrow{{color:#38bdf8;letter-spacing:.14em;text-transform:uppercase}}h1{{font-size:clamp(2.3rem,6vw,4.8rem);margin:.1em 0}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px;margin-top:22px}}.lab,.card{{background:#0e2538;border:1px solid #28506b;border-radius:18px;padding:20px;color:white;text-align:left}}.lab{{cursor:pointer}}.lab:hover{{border-color:#38bdf8;transform:translateY(-2px)}}.lab b{{display:block;font-size:1.25rem;color:#7dd3fc;margin-bottom:8px}}.lab span{{color:#b8d5e7}}pre{{white-space:pre-wrap;line-height:1.6}}</style></head><body><main class=shell><p class=eyebrow>Project 05 · executable curriculum</p><h1>Data Science Skills Lab</h1><p>Run a skill and receive a designed explanation—not an unfriendly raw JSON wall.</p><section class=grid>{cards}</section><section class=card style='margin-top:20px'><h2 id=heading>Choose a laboratory</h2><pre id=output>Results and interpretation will appear here.</pre></section></main><script>
async function run(skill){{
  const heading=document.getElementById('heading');
  const output=document.getElementById('output');
  heading.textContent='Running…';
  try {{
    const response=await fetch('/api/skills/'+skill);
    if(!response.ok) throw new Error('Skill request failed with status '+response.status);
    const data=await response.json();
    heading.textContent=data.title;
    output.textContent=Object.entries(data.result)
      .map(([key,value])=>`${{key.replaceAll('_',' ')}}: ${{Array.isArray(value)?JSON.stringify(value):value}}`)
      .join(String.fromCharCode(10));
  }} catch(error) {{
    heading.textContent='Unable to run skill';
    output.textContent=error instanceof Error?error.message:String(error);
  }}
}}</script></body></html>"""


@app.get("/api/skills")
def list_skills() -> dict[str, object]:
    return {key: {"title": value[0], "description": value[1]} for key, value in LABS.items()}


@app.get("/api/skills/{skill}")
def run_skill(skill: str) -> dict[str, object]:
    if skill not in LABS:
        raise HTTPException(404, "Unknown skill")
    return {"skill": skill, "title": LABS[skill][0], "result": execute_skill(skill), "dataset": "scikit-learn Wine"}
