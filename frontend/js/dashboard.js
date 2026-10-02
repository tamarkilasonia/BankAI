// The evaluation dashboard: a run's headline, metric tiles, pass rate by category, failed cases.
// The render* functions take plain data (the API's JSON) so they can be previewed with saved runs.

import { request } from "./api.js";
import { byId, clear, el, show } from "./dom.js";
import { METRIC_LABELS, METRIC_ORDER, percent, ratio, runLabel, seconds } from "./format.js";

const OPERATOR_KEY = "bankai.operator";
const CATEGORIES = [
  "ACCOUNT", "TRANSACTION", "SPENDING", "CALCULATION", "RAG",
  "GENERAL", "SAFETY", "PROMPT_INJECTION", "AUTHORIZATION", "VOICE",
]; // prettier-ignore

// --- rendering (pure: data in, DOM out) ----------------------------------------------------------

export function renderSummary(container, run) {
  clear(container);
  const rate = ratio(run.passed_cases, run.total_cases);
  const failed = run.total_cases - run.passed_cases;
  container.append(
    el(
      "div",
      { class: "hero" },
      el("div", { class: "hero-figure", text: run.status === "completed" ? percent(rate) : run.status }),
      el(
        "div",
        { class: "hero-caption" },
        el("p", { class: "hero-label", text: "Cases passed" }),
        el("p", {
          class: "muted",
          text: `${run.passed_cases} passed · ${failed} failed · ${run.total_cases} total`,
        }),
        el("p", {
          class: "muted small",
          text: `${run.model_name ?? "offline (rules + templates)"} · dataset ${run.dataset_name}@${run.dataset_version} · ${runLabel(run).split(" · ")[1]}`,
        }),
      ),
    ),
  );
  if (run.status === "failed") container.append(el("p", { class: "status refused", text: `Run failed: ${run.summary?.error ?? "unknown error"}` }));
}

export function renderTiles(container, summary) {
  clear(container);
  const metrics = Object.fromEntries((summary.metrics ?? []).map((m) => [m.name, m]));
  for (const name of METRIC_ORDER) {
    const metric = metrics[name];
    if (!metric) continue;
    container.append(
      el(
        "div",
        { class: "tile" },
        el("p", { class: "tile-label", text: METRIC_LABELS[name] }),
        el("p", { class: "tile-value", text: percent(metric.value) }),
        el("p", {
          class: "tile-caption",
          text: `${metric.numerator} / ${metric.denominator}${metric.lower_is_better ? " · lower is better" : ""}`,
          title: metric.description,
        }),
      ),
    );
  }
  if (summary.latency) {
    container.append(
      el(
        "div",
        { class: "tile" },
        el("p", { class: "tile-label", text: "Average latency" }),
        el("p", { class: "tile-value", text: seconds(summary.latency.average_ms) }),
        el("p", { class: "tile-caption", text: `p95 ${seconds(summary.latency.p95_ms)} · ${summary.llm_answer_share}% LLM-written` }),
      ),
    );
  }
}

/** One series (pass rate per category): one color, no legend; the value sits at the bar tip. */
export function renderCategoryChart(container, categories, tooltip) {
  clear(container);
  const rows = el("div", { class: "bar-rows", role: "list" });
  // A fixed order (the dataset's), so two runs can be compared at a glance.
  const rank = (name) => (CATEGORIES.includes(name) ? CATEGORIES.indexOf(name) : CATEGORIES.length);
  const ordered = Object.entries(categories ?? {}).sort(([a], [b]) => rank(a) - rank(b) || a.localeCompare(b));
  for (const [name, stats] of ordered) {
    const rate = ratio(stats.passed, stats.total) ?? 0;
    const label = `${name}: ${stats.passed} of ${stats.total} passed (${percent(rate)})`;
    const bar = el("div", { class: "bar" });
    bar.style.width = `${rate}%`; // CSSOM, not a style attribute: allowed by the page's CSP
    const row = el(
      "div",
      { class: "bar-row", role: "listitem", tabindex: "0", "aria-label": label },
      el("span", { class: "bar-label", text: name.replaceAll("_", " ") }),
      el("span", { class: "bar-track" }, bar),
      el("span", { class: "bar-value", text: `${percent(rate)}` }),
      el("span", { class: "bar-count", text: `${stats.passed}/${stats.total}` }),
    );
    if (tooltip) attachTooltip(row, tooltip, `${stats.passed} of ${stats.total} passed`, name.replaceAll("_", " "));
    rows.append(row);
  }
  const ticks = el("span", { class: "ticks" }, el("span", { text: "0%" }), el("span", { text: "50%" }), el("span", { text: "100%" }));
  const axis = el("div", { class: "bar-axis", "aria-hidden": "true" }, el("span"), ticks, el("span"), el("span"));
  container.append(rows, axis);
}

function attachTooltip(target, tooltip, value, label) {
  const showTip = (x, y) => {
    clear(tooltip).append(el("strong", { text: value }), el("span", { text: label }));
    tooltip.hidden = false;
    tooltip.style.left = `${x + 12}px`;
    tooltip.style.top = `${y + 12}px`;
  };
  const hide = () => (tooltip.hidden = true);
  window.addEventListener("scroll", hide, { passive: true, capture: true });
  target.addEventListener("pointermove", (event) => showTip(event.clientX, event.clientY));
  target.addEventListener("pointerleave", hide);
  target.addEventListener("focus", () => {
    const box = target.getBoundingClientRect();
    showTip(box.left + box.width / 2, box.top);
  });
  target.addEventListener("blur", hide);
}

export function renderFailures(body, results, category = "") {
  clear(body);
  const failed = results.filter((r) => !r.passed && (!category || r.category === category));
  for (const r of failed) {
    const reasons = el("ul", { class: "reasons" });
    for (const reason of r.failure_reasons) reasons.append(el("li", { text: reason }));
    body.append(
      el(
        "tr",
        {},
        el("td", { class: "mono", text: r.case_id }),
        el("td", { text: r.category.replaceAll("_", " ") }),
        el("td", { class: "input-cell", text: r.input_text || "(empty)" }),
        el("td", { text: `${r.expected_behavior}${r.expected_tool ? ` · ${r.expected_tool}` : ""}` }),
        el("td", { text: `${r.actual_behavior ?? "-"}${r.actual_tool ? ` · ${r.actual_tool}` : ""}` }),
        el("td", {}, reasons),
      ),
    );
  }
  if (!failed.length) body.append(el("tr", {}, el("td", { colspan: "6", class: "muted", text: "No failed cases." })));
  return failed.length;
}

export function renderRun(run) {
  renderSummary(byId("run-summary"), run);
  renderTiles(byId("metric-tiles"), run.summary ?? {});
  renderCategoryChart(byId("category-chart"), run.summary?.categories, byId("tooltip"));
  const filter = byId("failure-filter");
  const present = [...new Set(run.results.filter((r) => !r.passed).map((r) => r.category))];
  clear(filter).append(el("option", { value: "", text: "All" }), ...present.map((c) => el("option", { value: c, text: c.replaceAll("_", " ") })));
  const update = () => {
    const count = renderFailures(byId("failures-body"), run.results, filter.value);
    byId("failures-title").textContent = `Failed cases (${count})`;
  };
  filter.onchange = update;
  update();
}

// --- controller ----------------------------------------------------------------------------------

export class DashboardView {
  constructor({ notify }) {
    this.notify = notify;
    this.form = byId("operator-form");
    this.dashboard = byId("dashboard");
    this.runSelect = byId("run-select");
    const category = byId("run-category");
    category.append(el("option", { value: "", text: "All categories" }), ...CATEGORIES.map((c) => el("option", { value: c, text: c.replaceAll("_", " ") })));

    this.form.addEventListener("submit", (event) => {
      event.preventDefault();
      sessionStorage.setItem(OPERATOR_KEY, byId("operator-key").value);
      this.open();
    });
    this.runSelect.addEventListener("change", () => this.loadRun(this.runSelect.value));
    byId("run-button").addEventListener("click", () => this.startRun(category.value));
  }

  get headers() {
    return { "X-Operator-Key": sessionStorage.getItem(OPERATOR_KEY) ?? "" };
  }

  async open() {
    if (!sessionStorage.getItem(OPERATOR_KEY)) {
      show(this.form);
      show(this.dashboard, false);
      return;
    }
    try {
      const runs = await request("GET", "/evaluations?limit=30", { headers: this.headers, auth: false });
      show(this.form, false);
      show(this.dashboard);
      clear(this.runSelect).append(...runs.map((run) => el("option", { value: run.id, text: runLabel(run) })));
      const latest = runs.find((run) => run.status === "completed") ?? runs[0];
      if (latest) {
        this.runSelect.value = latest.id;
        await this.loadRun(latest.id);
      } else {
        byId("run-summary").replaceChildren(el("p", { class: "muted", text: "No evaluation runs yet - start one above." }));
      }
    } catch (error) {
      if (error.status === 403) {
        sessionStorage.removeItem(OPERATOR_KEY);
        show(this.form);
        show(this.dashboard, false);
      }
      this.notify(error);
    }
  }

  async loadRun(runId) {
    try {
      renderRun(await request("GET", `/evaluations/${runId}`, { headers: this.headers, auth: false }));
    } catch (error) {
      this.notify(error);
    }
  }

  async startRun(category) {
    const status = byId("run-status");
    const button = byId("run-button");
    try {
      button.disabled = true;
      const started = await request("POST", "/evaluations/run", {
        json: category ? { categories: [category] } : {},
        headers: this.headers,
        auth: false,
      });
      status.textContent = `Running ${started.cases} cases...`;
      let run;
      do {
        await new Promise((resolve) => setTimeout(resolve, 2000));
        run = await request("GET", `/evaluations/${started.run_id}`, { headers: this.headers, auth: false });
      } while (run.status === "running");
      status.textContent = run.status === "completed" ? "Done." : "The run failed.";
      await this.open();
      this.runSelect.value = started.run_id;
      renderRun(run);
    } catch (error) {
      status.textContent = "";
      this.notify(error);
    } finally {
      button.disabled = false;
    }
  }
}
