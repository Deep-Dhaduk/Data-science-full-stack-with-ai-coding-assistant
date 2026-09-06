"""Transparent Apriori-style market-basket mining without hidden libraries."""

from __future__ import annotations

from itertools import combinations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


BASKETS = [
    {"bread", "milk", "eggs"}, {"bread", "butter"}, {"milk", "diapers", "beer", "bread"},
    {"bread", "milk", "diapers", "beer"}, {"milk", "diapers", "beer"}, {"bread", "milk", "butter"},
    {"coffee", "milk", "bread"}, {"coffee", "cookies"}, {"bread", "eggs", "butter"},
    {"milk", "eggs"}, {"bread", "diapers", "beer"}, {"coffee", "milk", "cookies"},
] * 8
app = FastAPI(title="Market Basket Pattern Lab", version="1.0.0")


class MiningRequest(BaseModel):
    min_support: float = Field(default=0.2, gt=0, le=1)
    min_confidence: float = Field(default=0.55, gt=0, le=1)


def support(items: frozenset[str], baskets: list[set[str]] = BASKETS) -> float:
    return sum(items.issubset(basket) for basket in baskets) / len(baskets)


def mine_rules(min_support: float = 0.2, min_confidence: float = 0.55) -> list[dict[str, object]]:
    products = sorted(set().union(*BASKETS))
    frequent: dict[frozenset[str], float] = {}
    for size in range(1, min(4, len(products) + 1)):
        for combo in combinations(products, size):
            itemset = frozenset(combo)
            value = support(itemset)
            if value >= min_support:
                frequent[itemset] = value
    rules = []
    for itemset, item_support in frequent.items():
        if len(itemset) < 2:
            continue
        for left_size in range(1, len(itemset)):
            for left_tuple in combinations(sorted(itemset), left_size):
                left = frozenset(left_tuple)
                right = itemset - left
                confidence = item_support / frequent.get(left, support(left))
                right_support = frequent.get(right, support(right))
                lift = confidence / right_support
                if confidence >= min_confidence:
                    rules.append({
                        "antecedent": sorted(left), "consequent": sorted(right),
                        "support": round(item_support, 3), "confidence": round(confidence, 3), "lift": round(lift, 3),
                    })
    return sorted(rules, key=lambda rule: (rule["lift"], rule["confidence"]), reverse=True)


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>Basket Lab</title><style>
:root{font-family:Inter,system-ui;background:#171006;color:#fff7ed}body{margin:0}.shell{max-width:1050px;margin:auto;padding:45px 20px}.eyebrow{color:#fb923c;letter-spacing:.14em;text-transform:uppercase}h1{font-size:clamp(2.3rem,6vw,4.8rem);margin:.1em 0}.card{background:#29190b;border:1px solid #633b1c;border-radius:20px;padding:22px;margin-top:20px}.controls{display:grid;grid-template-columns:1fr 1fr auto;gap:15px;align-items:end}label{display:grid;gap:7px}input,button{padding:12px;border-radius:9px;border:1px solid #88542a;background:#160d05;color:white}button{background:#fb923c;color:#271203;font-weight:900;cursor:pointer}.rule{display:grid;grid-template-columns:2fr repeat(3,1fr);gap:12px;padding:14px;border-bottom:1px solid #57361e}.rule b{color:#fdba74}@media(max-width:650px){.controls,.rule{grid-template-columns:1fr}}</style></head><body><main class=shell><p class=eyebrow>Project 04 · association rules</p><h1>Market Basket Lab</h1><p>Inspect how support, confidence, and lift change the merchandising evidence.</p><form class='card controls' id=form><label>Minimum support<input id=support type=number step=.05 min=.05 max=1 value=.2></label><label>Minimum confidence<input id=confidence type=number step=.05 min=.05 max=1 value=.55></label><button>Mine rules</button></form><section class=card><p id=summary></p><div id=rules></div></section></main><script>
async function mine(){const x=await fetch('/api/mine',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({min_support:+support.value,min_confidence:+confidence.value})}).then(r=>r.json());summary.textContent=`${x.rule_count} rules from ${x.transactions} transactions`;rules.innerHTML=x.rules.slice(0,30).map(r=>`<div class=rule><b>${r.antecedent.join(' + ')} → ${r.consequent.join(' + ')}</b><span>support ${r.support}</span><span>confidence ${r.confidence}</span><span>lift ${r.lift}</span></div>`).join('')||'<p>No rule clears both thresholds.</p>'}form.onsubmit=e=>{e.preventDefault();mine()};mine()</script></body></html>"""


@app.post("/api/mine")
def mine(request: MiningRequest) -> dict[str, object]:
    rules = mine_rules(request.min_support, request.min_confidence)
    return {"transactions": len(BASKETS), "rule_count": len(rules), "rules": rules, "provenance": "authored demo baskets"}
