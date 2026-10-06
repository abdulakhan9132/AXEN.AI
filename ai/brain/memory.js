// Memory adapter: delegates all persistent storage to the Python Runtime single source of truth.
const { execSync } = require("child_process");
const path = require("path");

const PY_SCRIPT = path.join(__dirname, "..", "..", "app", "agent", "memory.py");
const PY_BIN = process.platform === "win32" ? ".venv\\Scripts\\python.exe" : ".venv/bin/python";

function callPython(cmd, ...args) {
  try {
    const procArgs = [PY_SCRIPT, cmd, ...args];
    // Use the virtual environment python if it exists, otherwise fallback to system python
    let pythonCmd = "python";
    if (require("fs").existsSync(path.join(__dirname, "..", "..", ".venv"))) {
      pythonCmd = path.join(__dirname, "..", "..", PY_BIN);
    }
    const result = execSync(`"${pythonCmd}" "${procArgs.join('" "')}"`, { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] });
    if (!result.trim()) return null;
    return JSON.parse(result);
  } catch (err) {
    // Return graceful defaults if Python fails (e.g. during tests without python initialized)
    return null;
  }
}

const words = (s) => new Set(s.toLowerCase().match(/[a-z0-9']+/g) || []);
function sim(a, b) {
  const A = words(a), B = words(b);
  if (!A.size || !B.size) return 0;
  let n = 0;
  for (const w of A) if (B.has(w)) n++;
  return n / Math.sqrt(A.size * B.size);
}

function handleCommand(text) {
  let r = text.match(/^\s*remember\s+(?:that\s+)?(.+)/i);
  if (r) {
    const fact = r[1].trim().replace(/[.!]+$/, "");
    callPython("remember", fact);
    return `Okay, I will remember that ${fact}.`;
  }
  r = text.match(/^\s*forget\s+(?:that\s+)?(.+)/i);
  if (r) {
    const q = r[1].trim().toLowerCase().replace(/[.!]+$/, "");
    const res = callPython("forget", q);
    return (res && res.removed) ? "Done, I forgot that." : "I did not have that saved.";
  }
  if (/\b(my|me)\b/i.test(text) && /^(what|who|where|when|do you|which)\b/i.test(text.trim())) {
    const matches = callPython("recall", text) || [];
    if (matches.length > 0) {
      return `You told me: ${matches[0]}.`;
    }
  }
  return null;
}

function recall(text, minSim = 0.3) {
  const matches = callPython("recallExchange", text) || [];
  return matches.map(axen => ({ axen, good: true }));
}

function addExchange(user, axen, good = false) {
  callPython("addExchange", user, axen, good ? "true" : "false");
}

function markGood(user, axen, good) {
  callPython("markGood", user, axen, good ? "true" : "false");
}

const stats = () => {
  const res = callPython("stats") || { facts: 0, exchanges: 0 };
  return { facts: res.facts, exchanges: res.exchanges };
};

// Deprecated in Phase 1: memory logic moved to Python
module.exports = { handleCommand, recall, addExchange, markGood, stats, sim };
