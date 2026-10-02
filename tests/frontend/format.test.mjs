// Unit tests for the UI's pure helpers:  node --test tests/frontend/
import assert from "node:assert/strict";
import test from "node:test";

import {
  METRIC_LABELS,
  METRIC_ORDER,
  confidence,
  extensionFor,
  maskAccount,
  money,
  percent,
  pickRecordingType,
  ratio,
  runLabel,
  seconds,
  traceLine,
} from "../../frontend/js/format.js";

test("percent and ratio", () => {
  assert.equal(percent(85), "85.0%");
  assert.equal(percent(66.6666), "66.7%");
  assert.equal(percent(null), "n/a");
  assert.equal(ratio(209, 246).toFixed(3), "84.959");
  assert.equal(ratio(1, 0), null);
});

test("latency reads in ms below a second", () => {
  assert.equal(seconds(5.4), "5 ms");
  assert.equal(seconds(708.2), "708 ms");
  assert.equal(seconds(2410), "2.41 s");
});

test("money, masked accounts and confidence", () => {
  assert.equal(money("8388.75", "GEL"), "8,388.75 GEL");
  assert.equal(money(5, "USD"), "5.00 USD");
  assert.equal(maskAccount("GE00MOCK000001000000001"), "****0001");
  assert.equal(confidence(0.859), "86%");
});

test("the recorder picks the first format the browser supports", () => {
  assert.equal(pickRecordingType((t) => t.startsWith("audio/mp4")), "audio/mp4"); // Safari
  assert.equal(pickRecordingType(() => true), "audio/webm;codecs=opus"); // Chrome
  assert.equal(pickRecordingType(() => false), "");
  assert.equal(extensionFor("audio/webm;codecs=opus"), "webm");
  assert.equal(extensionFor("audio/ogg;codecs=opus"), "ogg");
  assert.equal(extensionFor("audio/mp4"), "m4a");
  assert.equal(extensionFor("audio/wav"), "wav");
});

test("run labels and the trace line", () => {
  const run = { model_name: null, status: "completed", passed_cases: 182, total_cases: 246, started_at: "2026-10-02T10:00:00Z" };
  assert.match(runLabel(run), /^offline \(rules \+ templates\) · \d{4}-\d{2}-\d{2} \d{2}:\d{2} · 182\/246 passed$/);
  assert.match(runLabel({ ...run, model_name: "qwen2.5:3b", status: "running" }), /^qwen2\.5:3b · .* · running$/);
  const trace = { intent: "spending_analysis", tool: "analyze_spending", response_source: "llm" };
  assert.equal(traceLine(trace, 3177.4), "spending analysis · analyze_spending · written by the LLM · 3.18 s");
  assert.equal(traceLine({ ...trace, tool: null, response_source: "template" }, 4), "spending analysis · template · 4 ms");
});

test("every metric shown on the dashboard has a label", () => {
  for (const name of METRIC_ORDER) assert.ok(METRIC_LABELS[name], name);
});
