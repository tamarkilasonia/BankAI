// Talking to the BankAI API (same origin). Every call goes through request(), which adds
// the session token and turns the API's error format into an ApiError the UI can show.

const SESSION_KEY = "bankai.session";

export class ApiError extends Error {
  constructor(status, code, message, requestId) {
    super(message);
    this.status = status;
    this.code = code;
    this.requestId = requestId;
  }
}

// sessionStorage: the session ends with the browser tab (and with the token's own expiry).
export function getSession() {
  try {
    const session = JSON.parse(sessionStorage.getItem(SESSION_KEY) ?? "null");
    return session && Date.parse(session.expiresAt) > Date.now() ? session : null;
  } catch {
    return null;
  }
}

export function saveSession(login) {
  const session = {
    token: login.access_token,
    customer: login.customer,
    expiresAt: new Date(Date.now() + login.expires_in * 1000).toISOString(),
  };
  sessionStorage.setItem(SESSION_KEY, JSON.stringify(session));
  return session;
}

export function clearSession() {
  sessionStorage.removeItem(SESSION_KEY);
}

let onUnauthorized = () => {};

export function whenUnauthorized(handler) {
  onUnauthorized = handler;
}

/**
 * request("POST", "/chat", { json: {...} })
 * Options: json (body), form (FormData), headers, auth (default true), as ("json" | "blob").
 */
export async function request(method, path, options = {}) {
  const { json, form, headers = {}, auth = true, as = "json" } = options;
  const sent = { ...headers };
  const session = getSession();
  if (auth && session) sent.Authorization = `Bearer ${session.token}`;
  let body;
  if (json !== undefined) {
    sent["Content-Type"] = "application/json";
    body = JSON.stringify(json);
  } else if (form !== undefined) {
    body = form; // the browser sets the multipart boundary
  }

  let response;
  try {
    response = await fetch(path, { method, headers: sent, body });
  } catch {
    throw new ApiError(0, "network_error", "The server cannot be reached. Is it running?", null);
  }
  if (!response.ok) {
    let error = {};
    try {
      error = (await response.json()).error ?? {};
    } catch {
      // not our JSON error format (e.g. a proxy page): keep the generic message
    }
    if (response.status === 401 && auth) onUnauthorized();
    throw new ApiError(
      response.status,
      error.code ?? "http_error",
      error.message ?? `The request failed (HTTP ${response.status}).`,
      error.request_id ?? response.headers.get("X-Request-ID"),
    );
  }
  return as === "blob" ? response.blob() : response.json();
}
