"""A CPU-size character transformer with transparent training diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from torch import nn


SEED = 22
CORPUS = """data science begins with a question. inspect the data before fitting a model.
split training and evaluation data before learning preprocessing parameters.
compare every model with a simple baseline. measure uncertainty and document limitations.
a useful assistant explains evidence, refuses to invent metrics, and makes work reproducible.
crisp dm connects business understanding, data understanding, preparation, modeling, evaluation, and deployment.
"""
CHARS = sorted(set(CORPUS))
STOI = {char: index for index, char in enumerate(CHARS)}
ITOS = {index: char for char, index in STOI.items()}
app = FastAPI(title="Nano LLM Lab", version="1.0.0")


def encode(text: str) -> list[int]:
    fallback = STOI[" "]
    return [STOI.get(char.lower(), fallback) for char in text]


def decode(tokens: list[int]) -> str:
    return "".join(ITOS.get(token, " ") for token in tokens)


class TinyCausalTransformer(nn.Module):
    def __init__(self, vocab_size: int, width: int = 48, heads: int = 4, layers: int = 2, context: int = 96):
        super().__init__()
        self.context = context
        self.token = nn.Embedding(vocab_size, width)
        self.position = nn.Embedding(context, width)
        block = nn.TransformerEncoderLayer(width, heads, width * 4, dropout=0.0, batch_first=True, activation="gelu")
        self.blocks = nn.TransformerEncoder(block, layers)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, length = tokens.shape
        positions = torch.arange(length, device=tokens.device)
        hidden = self.token(tokens) + self.position(positions)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(length, device=tokens.device)
        return self.head(self.norm(self.blocks(hidden, mask=causal_mask, is_causal=True)))


@dataclass
class ModelState:
    model: TinyCausalTransformer
    losses: list[float]


def new_state() -> ModelState:
    torch.manual_seed(SEED)
    return ModelState(TinyCausalTransformer(len(CHARS)), [])


STATE = new_state()


def train_steps(state: ModelState, steps: int = 25, block: int = 48) -> list[float]:
    torch.manual_seed(SEED + len(state.losses))
    data = torch.tensor(encode(CORPUS), dtype=torch.long)
    optimizer = torch.optim.AdamW(state.model.parameters(), lr=3e-3)
    state.model.train()
    for _ in range(steps):
        starts = torch.randint(0, len(data) - block - 1, (8,))
        x = torch.stack([data[s : s + block] for s in starts])
        y = torch.stack([data[s + 1 : s + block + 1] for s in starts])
        logits = state.model(x)
        loss = nn.functional.cross_entropy(logits.reshape(-1, len(CHARS)), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(state.model.parameters(), 1.0)
        optimizer.step()
        state.losses.append(round(float(loss.detach()), 4))
    return state.losses[-steps:]


@torch.inference_mode()
def generate(state: ModelState, prompt: str, max_new_tokens: int = 90, temperature: float = 0.75) -> str:
    state.model.eval()
    tokens = torch.tensor([encode(prompt or "data")], dtype=torch.long)
    for _ in range(max_new_tokens):
        context = tokens[:, -state.model.context :]
        logits = state.model(context)[:, -1, :] / max(temperature, 0.1)
        next_token = torch.multinomial(torch.softmax(logits, dim=-1), 1)
        tokens = torch.cat([tokens, next_token], dim=1)
    return decode(tokens[0].tolist())


class GenerateRequest(BaseModel):
    prompt: str = Field(default="data science", max_length=200)
    max_new_tokens: int = Field(default=80, ge=10, le=180)
    temperature: float = Field(default=0.75, ge=0.1, le=1.5)


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>Nano LLM Lab</title><style>
:root{font-family:Inter,system-ui;background:#110b1e;color:#f3e8ff}body{margin:0}.shell{max-width:980px;margin:auto;padding:45px 20px}.eyebrow{color:#c084fc;letter-spacing:.15em;text-transform:uppercase}h1{font-size:clamp(2.4rem,6vw,5rem);margin:.1em 0}.card{background:#1d1231;border:1px solid #49306b;border-radius:20px;padding:22px;margin-top:20px}.grid{display:grid;grid-template-columns:2fr 1fr;gap:14px}textarea,input,button{box-sizing:border-box;width:100%;padding:13px;border-radius:10px;border:1px solid #67468c;background:#120b20;color:white}button{background:#a855f7;border:0;font-weight:800;cursor:pointer}pre{white-space:pre-wrap;min-height:110px;color:#e9d5ff}.metrics{display:flex;gap:12px;flex-wrap:wrap}.metric{background:#120b20;border-radius:12px;padding:14px;flex:1}.metric b{display:block;font-size:1.7rem;color:#c084fc}@media(max-width:650px){.grid{grid-template-columns:1fr}}</style></head><body><main class=shell><p class=eyebrow>Project 02 · tiny causal transformer</p><h1>Nano LLM Lab</h1><p>A real, inspectable character transformer small enough for a laptop. Its tiny teaching corpus limits fluency by design.</p><section class='card metrics' id=metrics></section><section class='card'><button id=train>Train 25 steps</button><p id=status>Train before generating; progress is kept in this server process.</p></section><form class='card grid' id=form><textarea id=prompt>data science</textarea><div><label>Temperature<input id=temp type=number step=.05 min=.1 max=1.5 value=.75></label><button>Generate</button></div></form><section class=card><pre id=output>Model output will appear here.</pre></section></main><script>
async function stats(){const s=await fetch('/api/stats').then(r=>r.json());metrics.innerHTML=`<div class=metric><b>${s.parameters.toLocaleString()}</b>parameters</div><div class=metric><b>${s.training_steps}</b>steps</div><div class=metric><b>${s.latest_loss??'—'}</b>latest loss</div><div class=metric><b>${s.vocabulary_size}</b>characters</div>`}train.onclick=async()=>{status.textContent='Training…';const r=await fetch('/api/train',{method:'POST'}).then(r=>r.json());status.textContent=`Finished. Loss ${r.first_loss} → ${r.last_loss}`;stats()};form.onsubmit=async e=>{e.preventDefault();output.textContent='Generating…';const r=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt:prompt.value,temperature:+temp.value,max_new_tokens:80})}).then(r=>r.json());output.textContent=r.text};stats()</script></body></html>"""


@app.get("/api/stats")
def stats() -> dict[str, int | float | None | str]:
    return {
        "architecture": "2-layer causal character transformer",
        "parameters": sum(p.numel() for p in STATE.model.parameters()),
        "training_steps": len(STATE.losses),
        "latest_loss": STATE.losses[-1] if STATE.losses else None,
        "vocabulary_size": len(CHARS),
        "corpus_characters": len(CORPUS),
    }


@app.post("/api/train")
def train() -> dict[str, float | int]:
    losses = train_steps(STATE)
    return {"steps": len(losses), "first_loss": losses[0], "last_loss": losses[-1]}


@app.post("/api/generate")
def complete(request: GenerateRequest) -> dict[str, str]:
    return {"text": generate(STATE, request.prompt, request.max_new_tokens, request.temperature)}
