// Building DOM safely. Everything the page shows - agent answers, transcripts, evaluation
// inputs (which include prompt-injection attacks!) - is untrusted text, so it only ever
// enters the page through textContent. This module is the one place elements are made;
// no HTML-parsing DOM API is used anywhere in the frontend (a test enforces that).

export function byId(id) {
  const node = document.getElementById(id);
  if (!node) throw new Error(`missing element #${id}`);
  return node;
}

/**
 * el("p", { class: "muted", text: "Hello" }, child1, child2)
 * Props: class, text, title, and any attribute (set with setAttribute). Children: nodes or strings.
 */
export function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = String(value);
    else if (key === "dataset") Object.assign(node.dataset, value);
    else node.setAttribute(key, value === true ? "" : String(value));
  }
  for (const child of children) {
    if (child === null || child === undefined || child === false) continue;
    node.append(typeof child === "string" ? document.createTextNode(child) : child);
  }
  return node;
}

export function clear(node) {
  node.replaceChildren();
  return node;
}

export function show(node, visible = true) {
  node.hidden = !visible;
}
