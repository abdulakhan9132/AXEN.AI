// Personality: traits that shape HOW Axen generates text. Random per Axen (first run),
// then they drift slowly from thumbs-up / thumbs-down feedback.
const fs = require("fs");
const path = require("path");
const FILE = path.join(__dirname, "..", "data", "personality.json");
const clamp = (x) => Math.min(0.95, Math.max(0.05, x));

function load() {
  try { return JSON.parse(fs.readFileSync(FILE, "utf8")); } catch {}
  const r = () => 0.2 + Math.random() * 0.6;
  const p = { curiosity: r(), caution: r(), talkativeness: r(), creativity: r(), formality: r() };
  save(p);
  return p;
}
function save(p) { fs.mkdirSync(path.dirname(FILE), { recursive: true }); fs.writeFileSync(FILE, JSON.stringify(p, null, 1)); }

function sampling() {
  const p = load();
  return {
    temperature: Math.max(0.25, 0.2 + p.creativity * 0.5 - p.caution * 0.15),
    topK: Math.round(3 + p.creativity * 15),
    maxTokens: Math.round(60 + p.talkativeness * 140),
  };
}
function feedback(up) {
  const p = load();
  p.creativity = clamp(p.creativity + (up ? 0.01 : -0.03));
  p.caution = clamp(p.caution + (up ? -0.005 : 0.03));
  p.curiosity = clamp(p.curiosity + (up ? 0.01 : 0));
  save(p);
  return p;
}
module.exports = { load, sampling, feedback };
