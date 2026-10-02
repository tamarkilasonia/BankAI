// Pure formatting helpers (no DOM) - unit-tested with `node --test` (tests/frontend/format.test.mjs).

export const METRIC_ORDER = [
  "answer_accuracy",
  "answer_relevance",
  "tool_selection_accuracy",
  "tool_argument_accuracy",
  "groundedness",
  "hallucination_rate",
  "safety_pass_rate",
  "prompt_injection_defense",
  "prompt_injection_detection",
  "authorization_pass_rate",
  "data_isolation",
];

export const METRIC_LABELS = {
  answer_accuracy: "Answer accuracy",
  answer_relevance: "Answer relevance",
  intent_accuracy: "Intent accuracy",
  tool_selection_accuracy: "Tool selection",
  tool_argument_accuracy: "Tool arguments",
  groundedness: "Groundedness",
  hallucination_rate: "Hallucination rate",
  safety_pass_rate: "Safety pass rate",
  prompt_injection_defense: "Injection defense",
  prompt_injection_detection: "Injection detection",
  authorization_pass_rate: "Authorization",
  data_isolation: "Data isolation",
};

export function percent(value) {
  return value === null || value === undefined || Number.isNaN(value) ? "n/a" : `${Number(value).toFixed(1)}%`;
}

export function ratio(part, whole) {
  return whole ? (100 * part) / whole : null;
}

export function seconds(ms) {
  const value = Number(ms);
  return value < 1000 ? `${Math.round(value)} ms` : `${(value / 1000).toFixed(2)} s`;
}

export function confidence(value) {
  return `${Math.round(Number(value) * 100)}%`;
}

export function maskAccount(number) {
  return `****${String(number).slice(-4)}`;
}

export function money(amount, currency) {
  const value = Number(amount);
  return `${value.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })} ${currency}`;
}

export function dateTime(iso) {
  if (!iso) return "-";
  const d = new Date(iso);
  const pad = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

/** "qwen2.5:3b · 2026-10-02 14:03 · 209/246 passed" */
export function runLabel(run) {
  const brain = run.model_name ?? "offline (rules + templates)";
  const result = run.status === "completed" ? `${run.passed_cases}/${run.total_cases} passed` : run.status;
  return `${brain} · ${dateTime(run.started_at)} · ${result}`;
}

/** The one-line "how I answered" summary under an assistant message. */
export function traceLine(trace, latencyMs) {
  const parts = [trace.intent.replaceAll("_", " ")];
  if (trace.tool) parts.push(trace.tool);
  parts.push(trace.response_source === "llm" ? "written by the LLM" : "template");
  parts.push(seconds(latencyMs));
  return parts.join(" · ");
}

/** Recording format: the first one this browser can record (Chrome/Firefox: WebM, Safari: MP4). */
export const RECORDING_TYPES = ["audio/webm;codecs=opus", "audio/ogg;codecs=opus", "audio/mp4", "audio/webm"];

export function pickRecordingType(isSupported) {
  return RECORDING_TYPES.find((type) => isSupported(type)) ?? "";
}

export function extensionFor(mimeType) {
  if (mimeType.startsWith("audio/ogg")) return "ogg";
  if (mimeType.startsWith("audio/mp4")) return "m4a";
  if (mimeType.startsWith("audio/wav")) return "wav";
  return "webm";
}

export const NOT_UNDERSTOOD_REASONS = {
  empty: "nothing was heard",
  low_confidence: "the speech was unclear",
  unsupported_language: "the language was not English or Georgian",
};
