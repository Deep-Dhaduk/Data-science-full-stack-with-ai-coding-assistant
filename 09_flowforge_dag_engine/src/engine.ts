export type NodeStatus = "pending" | "running" | "succeeded" | "failed";

export interface FlowNode {
  readonly id: string;
  readonly label: string;
  readonly dependencies: readonly string[];
  readonly operation: "extract" | "validate" | "transform" | "train" | "publish";
}

export interface NodeResult {
  readonly id: string;
  readonly status: NodeStatus;
  readonly message: string;
  readonly durationMs: number;
}

export interface FlowRun {
  readonly order: readonly string[];
  readonly results: readonly NodeResult[];
}

export class DagValidationError extends Error {}

export function topologicalOrder(nodes: readonly FlowNode[]): string[] {
  const byId = new Map(nodes.map((node) => [node.id, node]));
  if (byId.size !== nodes.length) throw new DagValidationError("Node IDs must be unique");
  for (const node of nodes) {
    for (const dependency of node.dependencies) {
      if (!byId.has(dependency)) throw new DagValidationError(`Unknown dependency: ${dependency}`);
    }
  }
  const inDegree = new Map(nodes.map((node) => [node.id, node.dependencies.length]));
  const dependents = new Map<string, string[]>();
  for (const node of nodes) {
    for (const dependency of node.dependencies) {
      const current = dependents.get(dependency) ?? [];
      current.push(node.id);
      dependents.set(dependency, current);
    }
  }
  const ready = nodes.filter((node) => inDegree.get(node.id) === 0).map((node) => node.id).sort();
  const order: string[] = [];
  while (ready.length) {
    const id = ready.shift();
    if (id === undefined) break;
    order.push(id);
    for (const dependent of dependents.get(id) ?? []) {
      const remaining = (inDegree.get(dependent) ?? 0) - 1;
      inDegree.set(dependent, remaining);
      if (remaining === 0) ready.push(dependent);
    }
    ready.sort();
  }
  if (order.length !== nodes.length) throw new DagValidationError("Cycle detected");
  return order;
}

export async function executeFlow(nodes: readonly FlowNode[]): Promise<FlowRun> {
  const order = topologicalOrder(nodes);
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const results: NodeResult[] = [];
  for (const id of order) {
    const node = byId.get(id);
    if (node === undefined) throw new DagValidationError(`Missing node during execution: ${id}`);
    const started = performance.now();
    await Promise.resolve();
    results.push({ id, status: "succeeded", message: `${node.operation}: ${node.label}`, durationMs: Math.max(0, Math.round(performance.now() - started)) });
  }
  return { order, results };
}

export const sampleFlow: readonly FlowNode[] = [
  { id: "extract", label: "Load customer events", dependencies: [], operation: "extract" },
  { id: "validate", label: "Check schema and nulls", dependencies: ["extract"], operation: "validate" },
  { id: "features", label: "Build RFM features", dependencies: ["validate"], operation: "transform" },
  { id: "train", label: "Fit segmentation model", dependencies: ["features"], operation: "train" },
  { id: "publish", label: "Publish model card", dependencies: ["train"], operation: "publish" }
] as const;
