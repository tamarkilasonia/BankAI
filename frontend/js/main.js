// Bootstrap: health check, MOCK sign-in, tabs, error banner.

import { clearSession, getSession, request, saveSession, whenUnauthorized } from "./api.js";
import { ChatView } from "./chat.js";
import { DashboardView } from "./dashboard.js";
import { byId, el, show } from "./dom.js";

const VIEWS = ["login-view", "chat-view", "evaluation-view"];
let chat = null;
let dashboard = null;

function notify(error) {
  const id = error.requestId ? ` (request ID ${error.requestId})` : "";
  byId("banner-text").textContent = `${error.message}${id}`;
  show(byId("banner"));
}

function showView(id) {
  for (const view of VIEWS) show(byId(view), view === id);
}

function selectTab() {
  const tab = location.hash === "#evaluation" ? "evaluation" : "chat";
  for (const link of document.querySelectorAll(".tab")) link.classList.toggle("active", link.dataset.tab === tab);
  showView(`${tab}-view`);
  if (tab === "evaluation") dashboard.open();
}

function showLogin(message) {
  show(byId("tabs"), false);
  show(byId("session-box"), false);
  showView("login-view");
  if (message) notify(new Error(message));
  const select = byId("customer-select");
  if (!select.options.length) {
    for (let i = 1; i <= 10; i += 1) {
      const number = `CUST-${String(i).padStart(4, "0")}`;
      select.append(el("option", { value: number, text: number }));
    }
  }
}

function startApp(session, health) {
  show(byId("tabs"));
  show(byId("session-box"));
  byId("session-name").textContent = `${session.customer.full_name} (${session.customer.customer_number})`;
  chat = new ChatView({ customer: session.customer, health, notify });
  dashboard = new DashboardView({ notify });
  chat.loadAccounts();
  window.addEventListener("hashchange", selectTab);
  selectTab();
}

async function main() {
  byId("banner-close").addEventListener("click", () => show(byId("banner"), false));
  byId("sign-out").addEventListener("click", () => {
    clearSession();
    location.hash = "";
    location.reload();
  });
  whenUnauthorized(() => {
    clearSession();
    showLogin("Your session has ended. Please sign in again.");
  });

  let health = { components: {} };
  try {
    health = await request("GET", "/health", { auth: false });
  } catch (error) {
    if (error.status !== 503) notify(error); // 503 = degraded but answering; the body is not needed
  }

  byId("login-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = byId("login-button");
    button.disabled = true;
    try {
      const login = await request("POST", "/auth/login", {
        json: { customer_number: byId("customer-select").value },
        auth: false,
      });
      startApp(saveSession(login), health);
    } catch (error) {
      notify(error);
    } finally {
      button.disabled = false;
    }
  });

  const session = getSession();
  if (session) startApp(session, health);
  else showLogin();
}

main();
