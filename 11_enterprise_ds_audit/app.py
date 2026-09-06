"""Repository-aware scientific controls audit; advisory, not certification."""

from __future__ import annotations

import ast
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse


PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parent
app = FastAPI(title="Enterprise Data Science Audit", version="1.0.0")


def inspect_python(path: Path) -> list[dict[str, str]]:
    source = path.read_text(encoding="utf-8")
    findings: list[dict[str, str]] = []
    try:
        ast.parse(source)
    except SyntaxError as error:
        return [{"severity": "high", "control": "syntax", "message": str(error)}]
    seed_markers = ("random_state", "default_rng", "manual_seed")
    if not any(marker in source for marker in seed_markers) and ("sklearn" in source or "torch" in source):
        findings.append({"severity": "medium", "control": "reproducibility", "message": "No random_state token found; manually verify deterministic execution."})
    if "train_test_split" in source and source.find(".fit(") < source.find("train_test_split"):
        findings.append({"severity": "high", "control": "leakage", "message": "A fit call appears before the split; inspect whether it learns from held-out rows."})
    if "accuracy_score" in source and not any(metric in source for metric in ("balanced_accuracy", "average_precision", "roc_auc", "f1")):
        findings.append({"severity": "medium", "control": "metrics", "message": "Accuracy appears without a class-imbalance companion metric."})
    return findings


def audit_repository(root: Path = ROOT) -> dict[str, object]:
    projects = sorted(path for path in root.iterdir() if path.is_dir() and path.name[:2].isdigit())
    results = []
    for project in projects:
        findings = []
        for required in ("README.md", "PROMPT.md"):
            if not (project / required).exists():
                findings.append({"severity": "high", "control": "documentation", "message": f"Missing {required}"})
        runtimes = list(project.glob("*.py")) + list(project.glob("src/*.ts"))
        if not runtimes:
            findings.append({"severity": "high", "control": "implementation", "message": "No Python or TypeScript runtime found."})
        if not list(project.glob("test*.py")) and not list(project.glob("**/*.test.ts")):
            findings.append({"severity": "medium", "control": "verification", "message": "No unit tests discovered."})
        for source_file in project.glob("*.py"):
            if not source_file.name.startswith("test"):
                findings.extend(inspect_python(source_file))
        results.append({"project": project.name, "findings": findings, "passed_controls": max(0, 5 - len(findings))})
    totals = {level: sum(1 for row in results for finding in row["findings"] if finding["severity"] == level) for level in ("high", "medium", "low")}
    return {"scope": str(root), "projects_scanned": len(projects), "totals": totals, "results": results, "disclaimer": "Automated heuristics support review; they do not certify scientific correctness."}


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>DS Audit</title><style>
:root{font-family:Inter,system-ui;background:#0c1116;color:#edf6ff}body{margin:0}.shell{max-width:1100px;margin:auto;padding:45px 20px}.eyebrow{color:#2dd4bf;letter-spacing:.14em;text-transform:uppercase}h1{font-size:clamp(2.3rem,6vw,4.8rem);margin:.1em 0}.card{background:#151f27;border:1px solid #354955;border-radius:18px;padding:20px;margin-top:14px}.summary{display:flex;gap:12px;flex-wrap:wrap}.metric{flex:1;background:#0b171c;border-radius:12px;padding:15px}.metric b{display:block;font-size:2rem;color:#5eead4}.finding{padding:10px;border-left:4px solid #f59e0b;margin:8px 0;background:#201c14}.high{border-color:#fb7185}.clear{color:#86efac}</style></head><body><main class=shell><p class=eyebrow>Project 11 · scientific governance</p><h1>Enterprise DS Audit</h1><p>Repository evidence and actionable heuristics—never a decorative “A+” certification.</p><section class='card summary' id=summary></section><div id=projects></div></main><script>fetch('/api/audit').then(r=>r.json()).then(x=>{summary.innerHTML=`<div class=metric><b>${x.projects_scanned}</b>projects scanned</div><div class=metric><b>${x.totals.high}</b>high findings</div><div class=metric><b>${x.totals.medium}</b>medium findings</div>`;projects.innerHTML=x.results.map(p=>`<section class=card><h2>${p.project}</h2>${p.findings.length?p.findings.map(f=>`<div class='finding ${f.severity}'><b>${f.severity} · ${f.control}</b><br>${f.message}</div>`).join(''):'<p class=clear>No heuristic findings. Manual review is still required.</p>'}</section>`).join('')})</script></body></html>"""


@app.get("/api/audit")
def audit() -> dict[str, object]:
    return audit_repository()
