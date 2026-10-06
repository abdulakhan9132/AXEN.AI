// Logic brain: exact skills that do NOT use the neural network (math, date and time).
function tryAnswer(text) {
  const t = text.trim();
  const expr = t.replace(/^(what is|what's|calculate|compute|solve)\s+/i, "")
    .replace(/[=?\s]+$/, "")
    .replace(/×/g, "*")
    .replace(/(\d+)\s*[xX]\s*(\d+)/g, "$1*$2")
    .replace(/\^/g, "**");
  if (/^[\d\s+\-*/%().]+$/.test(expr) && /\d/.test(expr) && /[+\-*/%]/.test(expr)) {
    try {
      const v = Function('"use strict";return (' + expr + ")")();
      if (Number.isFinite(v)) return `${expr.replace(/\*\*/g, "^")} = ${Math.round(v * 1e9) / 1e9}`;
    } catch {}
  }
  if (/^(what|tell).*\b(time|date|day)\b/i.test(t)) return "It is " + new Date().toString().slice(0, 33) + ".";
  return null;
}
module.exports = { tryAnswer };
