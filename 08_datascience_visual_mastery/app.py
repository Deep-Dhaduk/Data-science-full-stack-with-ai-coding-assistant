"""Interactive numerical intuition for classification metrics and gradients."""

from __future__ import annotations

from math import exp

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


app = FastAPI(title="Data Science Visual Mastery", version="1.0.0")


class MatrixRequest(BaseModel):
    true_positive: int = Field(default=72, ge=0)
    false_positive: int = Field(default=12, ge=0)
    false_negative: int = Field(default=8, ge=0)
    true_negative: int = Field(default=108, ge=0)


class GradientRequest(BaseModel):
    start: float = Field(default=7, ge=-100, le=100)
    learning_rate: float = Field(default=.15, gt=0, le=1)
    steps: int = Field(default=12, ge=1, le=100)


def classification_metrics(matrix: MatrixRequest) -> dict[str, float]:
    tp, fp, fn, tn = matrix.true_positive, matrix.false_positive, matrix.false_negative, matrix.true_negative
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    specificity = tn / (tn + fp) if tn + fp else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {"precision": round(precision, 3), "recall": round(recall, 3), "specificity": round(specificity, 3), "f1": round(f1, 3), "type_i_errors": fp, "type_ii_errors": fn}


def descend(request: GradientRequest) -> list[dict[str, float]]:
    # Minimize f(x)=(x-2)^2; derivative f'(x)=2(x-2).
    x = request.start
    path = []
    for step in range(request.steps + 1):
        path.append({"step": step, "x": round(x, 5), "loss": round((x - 2) ** 2, 5), "gradient": round(2 * (x - 2), 5)})
        x -= request.learning_rate * 2 * (x - 2)
    return path


def bayes_posterior(prior: float, sensitivity: float, false_positive_rate: float) -> float:
    evidence = sensitivity * prior + false_positive_rate * (1 - prior)
    return sensitivity * prior / evidence if evidence else 0


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>Visual Mastery</title><style>
:root{font-family:Inter,system-ui;background:#08101a;color:#ecfeff}body{margin:0}.shell{max-width:1100px;margin:auto;padding:45px 20px}.eyebrow{color:#22d3ee;letter-spacing:.14em;text-transform:uppercase}h1{font-size:clamp(2.3rem,6vw,4.8rem);margin:.1em 0}.card{background:#10202e;border:1px solid #285069;border-radius:20px;padding:22px;margin-top:20px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:13px}label{display:grid;gap:6px}input,button{padding:11px;border-radius:9px;border:1px solid #39718d;background:#081823;color:white}button{background:#22d3ee;color:#06202a;font-weight:900;cursor:pointer}.metric{background:#081823;border-radius:12px;padding:14px}.metric b{color:#67e8f9;display:block;font-size:1.6rem}.bar{height:14px;background:#083344;border-radius:99px;overflow:hidden}.bar span{display:block;height:100%;background:#22d3ee}</style></head><body><main class=shell><p class=eyebrow>Project 08 · learn by manipulating</p><h1>Data Science Visual Mastery</h1><p>Move the numbers. Watch definitions become behavior.</p><section class=card><h2>Confusion matrix laboratory</h2><form class=grid id=matrix><label>True positive<input id=tp type=number value=72></label><label>False positive (Type I)<input id=fp type=number value=12></label><label>False negative (Type II)<input id=fn type=number value=8></label><label>True negative<input id=tn type=number value=108></label><button>Recalculate</button></form><div class=grid id=metrics></div></section><section class=card><h2>Gradient descent laboratory</h2><p>Minimize <b>f(x) = (x − 2)²</b>.</p><form class=grid id=gradient><label>Starting x<input id=start type=number value=7></label><label>Learning rate<input id=rate type=number step=.05 value=.15></label><label>Steps<input id=steps type=number value=12></label><button>Run descent</button></form><div id=path></div></section><section class=card><h2>Quick check</h2><p>If false negatives are extremely costly, which metric deserves special attention?</p><button onclick="this.nextElementSibling.textContent='Recall: it measures the share of actual positives that the model finds.'">Reveal answer</button><p></p></section></main><script>
matrix.onsubmit=async e=>{e.preventDefault();const x=await fetch('/api/confusion',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({true_positive:+tp.value,false_positive:+fp.value,false_negative:+fn.value,true_negative:+tn.value})}).then(r=>r.json());metrics.innerHTML=Object.entries(x).map(([k,v])=>`<div class=metric><b>${v}</b>${k.replaceAll('_',' ')}</div>`).join('')};gradient.onsubmit=async e=>{e.preventDefault();const x=await fetch('/api/gradient',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({start:+start.value,learning_rate:+rate.value,steps:+steps.value})}).then(r=>r.json());path.innerHTML=x.path.map(p=>`<p>Step ${p.step}: x=${p.x}, loss=${p.loss}</p><div class=bar><span style='width:${Math.min(100,p.loss*4)}%'></span></div>`).join('')};matrix.requestSubmit();gradient.requestSubmit()</script></body></html>"""


@app.post("/api/confusion")
def confusion(request: MatrixRequest) -> dict[str, float]:
    return classification_metrics(request)


@app.post("/api/gradient")
def gradient(request: GradientRequest) -> dict[str, object]:
    return {"objective": "(x - 2)^2", "path": descend(request)}


@app.get("/api/bayes")
def bayes(prior: float = .01, sensitivity: float = .9, false_positive_rate: float = .05) -> dict[str, float]:
    return {"posterior": round(bayes_posterior(prior, sensitivity, false_positive_rate), 4)}
