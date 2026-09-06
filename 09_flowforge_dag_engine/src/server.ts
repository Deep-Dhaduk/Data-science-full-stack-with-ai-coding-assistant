import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { executeFlow, sampleFlow, topologicalOrder } from "./engine.js";

const port = Number(process.env.PORT ?? 8009);
const page = `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>FlowForge</title><style>
:root{font-family:Inter,system-ui;background:#0c0f16;color:#f8fafc}body{margin:0}.shell{max-width:1050px;margin:auto;padding:45px 20px}.eyebrow{color:#a78bfa;letter-spacing:.14em;text-transform:uppercase}h1{font-size:clamp(2.4rem,6vw,5rem);margin:.1em 0}.card{background:#171b27;border:1px solid #393e54;border-radius:20px;padding:22px;margin-top:20px}.flow{display:flex;gap:12px;flex-wrap:wrap}.node{flex:1;min-width:150px;background:#211a35;border:1px solid #6d52a0;border-radius:14px;padding:16px}.node b{display:block;color:#c4b5fd}.ok{border-color:#34d399;background:#112a25}button{padding:13px 20px;border:0;border-radius:10px;background:#8b5cf6;color:white;font-weight:900;cursor:pointer}</style></head><body><main class=shell><p class=eyebrow>Project 09 · typed orchestration</p><h1>FlowForge DAG Engine</h1><p>A strict TypeScript workflow runtime with graph validation and observable ordered execution.</p><section class="card"><div class="flow" id="flow"></div></section><section class="card"><button id="run">Execute pipeline</button><p id="status">Validated order is ready.</p></section></main><script>
fetch('/api/flow').then(r=>r.json()).then(x=>flow.innerHTML=x.nodes.map(n=>'<div class="node" id="n-'+n.id+'"><b>'+n.label+'</b><small>'+n.operation+' · after '+(n.dependencies.join(', ')||'start')+'</small></div>').join(''));run.onclick=async()=>{status.textContent='Running…';const x=await fetch('/api/run',{method:'POST'}).then(r=>r.json());for(const [i,id] of x.order.entries()){await new Promise(r=>setTimeout(r,250));document.querySelector('#n-'+id).classList.add('ok');status.textContent='Completed '+(i+1)+' / '+x.order.length+': '+id}status.textContent='Pipeline succeeded in dependency order.'}</script></body></html>`;

function json(response: ServerResponse, value: unknown, status = 200): void {
  response.writeHead(status, { "content-type": "application/json; charset=utf-8" });
  response.end(JSON.stringify(value));
}

createServer(async (request: IncomingMessage, response: ServerResponse) => {
  if (request.url === "/") {
    response.writeHead(200, { "content-type": "text/html; charset=utf-8" });
    response.end(page);
  } else if (request.url === "/api/flow" && request.method === "GET") {
    json(response, { nodes: sampleFlow, order: topologicalOrder(sampleFlow) });
  } else if (request.url === "/api/run" && request.method === "POST") {
    json(response, await executeFlow(sampleFlow));
  } else if (request.url === "/api/health") {
    json(response, { status: "ok" });
  } else {
    json(response, { error: "Not found" }, 404);
  }
}).listen(port, "127.0.0.1", () => console.log(`FlowForge listening on http://127.0.0.1:${port}`));
