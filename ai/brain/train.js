// Train Axen's language brain from scratch.  Usage:  node ai/brain/train.js 2000
// Reads every .txt file in ai/data/corpus/, trains, and saves ai/data/brain.json.
// Safe to stop with Ctrl+C (it saves first). Run it again to continue training.
const fs = require("fs");
const path = require("path");
const { Model } = require("./model");
const tok = require("./tokenizer");

const DATA = path.join(__dirname, "..", "data");
const WEIGHTS = path.join(DATA, "brain.json");
const CORPUS = path.join(DATA, "corpus");

function loadCorpusIds() {
  fs.mkdirSync(CORPUS, { recursive: true });
  const text = fs.readdirSync(CORPUS).filter((f) => f.endsWith(".txt"))
    .map((f) => fs.readFileSync(path.join(CORPUS, f), "utf8")).join("\n");
  return tok.encode(text);
}

function loadOrCreate() {
  return fs.existsSync(WEIGHTS) ? Model.load(WEIGHTS) : new Model();
}

function train(model, steps, { logEvery = 50, saveEvery = 250 } = {}) {
  const ids = loadCorpusIds();
  if (ids.length < model.cfg.T + 2)
    throw new Error("Corpus too small. Add more text files to ai/data/corpus/");
  let stopping = false;
  process.once("SIGINT", () => { stopping = true; });
  const t0 = Date.now();
  for (let i = 1; i <= steps && !stopping; i++) {
    const loss = model.trainStep(ids);
    if (i % logEvery === 0)
      console.log(`step ${model.trained}  loss ${loss.toFixed(3)}  ${((Date.now() - t0) / 1000).toFixed(0)}s`);
    if (i % saveEvery === 0) model.save(WEIGHTS);
  }
  model.save(WEIGHTS);
  console.log("Saved", WEIGHTS, "- total steps:", model.trained, "loss:", model.loss?.toFixed(3));
}

module.exports = { train, loadOrCreate, WEIGHTS, DATA };

if (require.main === module) {
  const steps = parseInt(process.argv[2] || "1000", 10);
  const model = loadOrCreate();
  console.log(`Axen brain: ${model.paramCount().toLocaleString()} parameters, trained ${model.trained} steps so far`);
  train(model, steps);
}
