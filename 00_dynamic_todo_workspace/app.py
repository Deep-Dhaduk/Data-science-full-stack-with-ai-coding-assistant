"""Zenith: an independently runnable task workspace and analytics API."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from enum import Enum
from itertools import count

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


class Priority(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class TaskInput(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    priority: Priority = Priority.medium
    estimate_minutes: int = Field(default=25, ge=5, le=480)


class Task(TaskInput):
    id: int
    completed: bool = False
    created_at: str


TASKS: dict[int, Task] = {}
IDS = count(1)
app = FastAPI(title="Zenith Dynamic Todo Workspace", version="1.0.0")


def create_task(payload: TaskInput) -> Task:
    task_id = next(IDS)
    task = Task(
        id=task_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        **payload.model_dump(),
    )
    TASKS[task_id] = task
    return task


def summarize(tasks: list[Task]) -> dict[str, object]:
    total = len(tasks)
    completed = sum(task.completed for task in tasks)
    return {
        "total": total,
        "completed": completed,
        "completion_rate": round(completed / total, 3) if total else 0.0,
        "open_minutes": sum(t.estimate_minutes for t in tasks if not t.completed),
        "priority_mix": dict(Counter(t.priority.value for t in tasks)),
    }


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>
<title>Zenith Workspace</title><style>
:root{font-family:Inter,system-ui;color:#e8eef7;background:#08111f}body{margin:0}.shell{max-width:980px;margin:auto;padding:48px 20px}
h1{font-size:clamp(2rem,5vw,4rem);margin:0}.eyebrow{color:#67e8f9;letter-spacing:.18em;text-transform:uppercase}.card{background:#101d30;border:1px solid #233957;border-radius:18px;padding:20px;margin-top:22px;box-shadow:0 20px 50px #0005}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px}.metric{background:#0b1728;border-radius:12px;padding:16px}.metric b{font-size:1.8rem;display:block;color:#67e8f9}
form{display:grid;grid-template-columns:2fr 1fr 1fr auto;gap:10px}input,select,button{padding:12px;border-radius:10px;border:1px solid #355172;background:#0b1728;color:#fff}button{background:#14b8a6;border:0;font-weight:700;cursor:pointer}
.task{display:flex;align-items:center;gap:12px;border-bottom:1px solid #233957;padding:12px 2px}.task span{flex:1}.done{text-decoration:line-through;opacity:.55}@media(max-width:650px){form{grid-template-columns:1fr}}
</style></head><body><main class='shell'><p class='eyebrow'>Project 00 · productivity telemetry</p><h1>Zenith Workspace</h1><p>Plan deliberately. See the cost of unfinished work.</p>
<section class='card'><form id='form'><input id='title' required maxlength='120' placeholder='What needs to happen?'><select id='priority'><option>medium</option><option>high</option><option>low</option></select><input id='estimate' type='number' min='5' max='480' value='25'><button>Add task</button></form></section>
<section id='metrics' class='card grid'></section><section class='card'><div id='tasks'></div></section></main><script>
async function refresh(){const [ts,ms]=await Promise.all([fetch('/api/tasks').then(r=>r.json()),fetch('/api/analytics').then(r=>r.json())]);
metrics.innerHTML=`<div class=metric><b>${ms.total}</b>Total</div><div class=metric><b>${ms.completed}</b>Completed</div><div class=metric><b>${Math.round(ms.completion_rate*100)}%</b>Progress</div><div class=metric><b>${ms.open_minutes}</b>Open minutes</div>`;
tasks.innerHTML=ts.length?ts.map(t=>`<div class=task><input type=checkbox ${t.completed?'checked':''} onchange='toggle(${t.id})'><span class='${t.completed?'done':''}'>${escapeHtml(t.title)}</span><small>${t.priority} · ${t.estimate_minutes}m</small><button onclick='removeTask(${t.id})'>Delete</button></div>`).join(''):'<p>No tasks yet. Add the first meaningful action.</p>'}
function escapeHtml(x){const d=document.createElement('div');d.textContent=x;return d.innerHTML}async function toggle(id){await fetch(`/api/tasks/${id}/toggle`,{method:'PATCH'});refresh()}async function removeTask(id){await fetch(`/api/tasks/${id}`,{method:'DELETE'});refresh()}
form.onsubmit=async e=>{e.preventDefault();await fetch('/api/tasks',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({title:title.value,priority:priority.value,estimate_minutes:+estimate.value})});title.value='';refresh()};refresh();</script></body></html>"""


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/tasks")
def list_tasks() -> list[Task]:
    return list(TASKS.values())


@app.post("/api/tasks", status_code=201)
def add_task(payload: TaskInput) -> Task:
    return create_task(payload)


@app.patch("/api/tasks/{task_id}/toggle")
def toggle_task(task_id: int) -> Task:
    task = TASKS.get(task_id)
    if task is None:
        raise HTTPException(404, "Task not found")
    task.completed = not task.completed
    return task


@app.delete("/api/tasks/{task_id}", status_code=204)
def delete_task(task_id: int) -> None:
    if TASKS.pop(task_id, None) is None:
        raise HTTPException(404, "Task not found")


@app.get("/api/analytics")
def analytics() -> dict[str, object]:
    return summarize(list(TASKS.values()))
