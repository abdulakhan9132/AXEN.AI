// Axen's orchestrator: perceive -> memory -> logic -> recall -> language brain -> reflect.
const fs = require("fs");
const tok = require("./tokenizer");
const { Model } = require("./model");
const { WEIGHTS } = require("./train");
const memory = require("./memory");
const reasoner = require("./reasoner");
const personality = require("./personality");

let model = null;
function getModel() {
  if (!model && fs.existsSync(WEIGHTS)) model = Model.load(WEIGHTS);
  return model;
}
const reload = () => { model = null; return getModel(); };

function status() {
  const m = getModel();
  return {
    trained: !!m,
    steps: m ? m.trained : 0,
    loss: m && m.loss != null ? +m.loss.toFixed(3) : null,
    params: m ? m.paramCount() : 0,
    personality: personality.load(),
    memory: memory.stats(),
  };
}

async function queryOllama(userText, emit) {
  const url = "http://localhost:11434/api/generate";
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 60000);
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "qwen2.5:1.5b",
        prompt: userText,
        system: "You are AXEN, a local AI assistant. Answer the user's current request directly and naturally in simple words.",
        stream: true,
      }),
      signal: controller.signal,
    });
    clearTimeout(timer);
    if (!res.ok) return null;

    emit("stage", "Language brain thinking (Qwen 2.5 local)");
    let fullText = "";
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          const json = JSON.parse(line);
          if (json.response) {
            fullText += json.response;
            emit("token", json.response);
          }
        } catch {}
      }
    }
    const finalAnswer = fullText.trim();
    return finalAnswer || null;
  } catch (err) {
    clearTimeout(timer);
    return null;
  }
}

// emit(type, data) types: "stage" (thinking trace), "token" (answer text), "done"
async function respond(userText, emit = () => {}) {
  emit("stage", "Perceiving your message");

  const cmd = memory.handleCommand(userText);
  if (cmd) { emit("stage", "Memory brain handled this"); emit("token", cmd); return cmd; }

  const logic = reasoner.tryAnswer(userText);
  if (logic) { emit("stage", "Logic brain solved this exactly"); emit("token", logic); memory.addExchange(userText, logic); return logic; }

  const recalled = memory.recall(userText, 0.7).filter((e) => e.good);
  if (recalled.length) {
    emit("stage", "Recalled a reply you liked before");
    emit("token", recalled[0].axen);
    memory.addExchange(userText, recalled[0].axen, true);
    return recalled[0].axen;
  }

  // Try local Ollama language brain first for high-quality answers
  const ollamaText = await queryOllama(userText, emit);
  if (ollamaText) {
    memory.addExchange(userText, ollamaText);
    return ollamaText;
  }

  const m = getModel();
  if (!m) {
    const msg = "My brain has not been trained yet. In the project folder run: node ai/brain/train.js 2000";
    emit("token", msg);
    return msg;
  }
  const p = personality.sampling();
  emit("stage", `Language brain thinking (temperature ${p.temperature.toFixed(2)}, ${m.trained} steps trained)`);
  const prompt = `User: ${userText.replace(/\n/g, " ")}\nAxen:`;
  let text = "";
  await m.generate(tok.encode(prompt), {
    ...p, stop: ["\nUser:", "\n\n"],
    onToken: (id) => { const c = tok.decode([id]); text += c; emit("token", c); },
  });
  text = text.replace(/\n+User:$/, "").replace(/\n+$/, "").trim();
  memory.addExchange(userText, text);
  return text;
}

function feedback(userText, axenText, up) {
  memory.markGood(userText, axenText, up);
  personality.feedback(up);
  if (up) fs.appendFileSync(require("path").join(__dirname, "..", "data", "corpus", "learned.txt"), `User: ${userText}\nAxen: ${axenText}\n\n`);
}

async function providerRespond(userText, emit = () => {}) {
  const m = getModel();
  if (!m) {
    const msg = "My brain has not been trained yet. In the project folder run: node ai/brain/train.js 2000";
    emit("token", msg);
    return msg;
  }
  const p = personality.sampling();
  emit("stage", `Language brain thinking (temperature ${p.temperature.toFixed(2)}, ${m.trained} steps trained)`);
  const prompt = `User: ${userText.replace(/\n/g, " ")}\nAxen:`;
  let text = "";
  await m.generate(tok.encode(prompt), {
    ...p, stop: ["\nUser:", "\n\n"],
    onToken: (id) => { const c = tok.decode([id]); text += c; emit("token", c); },
  });
  text = text.replace(/\n+User:$/, "").replace(/\n+$/, "").trim();
  return text;
}

module.exports = { respond, feedback, status, reload, providerRespond };
