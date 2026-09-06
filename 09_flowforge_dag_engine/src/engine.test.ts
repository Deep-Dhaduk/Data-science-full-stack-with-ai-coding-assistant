import assert from "node:assert/strict";
import test from "node:test";
import { DagValidationError, executeFlow, sampleFlow, topologicalOrder, type FlowNode } from "./engine.js";

test("orders every dependency before its consumer", () => {
  const order = topologicalOrder(sampleFlow);
  assert.ok(order.indexOf("extract") < order.indexOf("validate"));
  assert.ok(order.indexOf("features") < order.indexOf("train"));
});

test("rejects a cycle", () => {
  const cyclic: FlowNode[] = [
    { id: "a", label: "A", dependencies: ["b"], operation: "extract" },
    { id: "b", label: "B", dependencies: ["a"], operation: "validate" }
  ];
  assert.throws(() => topologicalOrder(cyclic), DagValidationError);
});

test("executes all nodes successfully", async () => {
  const run = await executeFlow(sampleFlow);
  assert.equal(run.results.length, sampleFlow.length);
  assert.ok(run.results.every((result) => result.status === "succeeded"));
});
