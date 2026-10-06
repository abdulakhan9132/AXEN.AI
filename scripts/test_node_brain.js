// End-to-end acceptance check for the plain Node Axen brain.
// The script restores memory/personality/learned examples after the checks.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const root = path.resolve(__dirname, "..");
const dataDir = path.join(root, "ai", "data");
const files = ["memory.json", "personality.json", path.join("corpus", "learned.txt")];
const snapshots = files.map((name) => {
  const target = path.join(dataDir, name);
  return { target, existed: fs.existsSync(target), data: fs.existsSync(target) ? fs.readFileSync(target) : null };
});

const API = process.env.AXEN_TEST_URL || "http://127.0.0.1:8000";

async function readEvents(response) {
  assert.equal(response.ok, true, `Chat route returned HTTP ${response.status}`);
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const events = [];
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) >= 0) {
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const line = frame.split(/\r?\n/).find((item) => item.startsWith("data:"));
      if (line) events.push(JSON.parse(line.slice(5).trim()));
    }
    if (done) break;
  }
  return events;
}

async function ask(text) {
  return readEvents(await fetch(`${API}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages: [{ role: "user", content: text }] }),
  }));
}

async function run() {
  fs.writeFileSync(snapshots[0].target, JSON.stringify({ facts: [], exchanges: [] }));
  const statusResponse = await fetch(`${API}/api/brain`);
  assert.equal(statusResponse.ok, true, "Brain status route is reachable");
  const status = await statusResponse.json();
  assert.equal(status.trained, true, "The trained brain weights are loaded");
  assert.ok(status.steps >= 600, "The brain has completed at least 600 training steps");

  const greeting = await ask("hi");
  assert.ok(greeting.some((event) => event.type === "stage" && /Language brain/.test(event.data)), "hi uses the language brain");
  assert.ok(greeting.some((event) => event.type === "token"), "The language brain streams answer tokens");
  assert.ok(greeting.some((event) => event.type === "done" && String(event.data).trim()), "The language brain finishes with a reply");

  const math = await ask("what is 12*(3+4)");
  assert.ok(math.some((event) => event.type === "done" && String(event.data).includes("84")), "Exact math brain returns 84");

  const remembered = await ask("remember that my name is Alex");
  assert.ok(remembered.some((event) => event.type === "done" && /remember/i.test(event.data)), "Remember command is handled");
  const recall = await ask("what is my name?");
  assert.ok(recall.some((event) => event.type === "done" && /my name is Alex/i.test(event.data)), "Saved fact is recalled");

  const prompt = `Axen acceptance test ${Date.now()}`;
  const first = await ask(prompt);
  const firstReply = first.find((event) => event.type === "done")?.data;
  assert.ok(String(firstReply || "").trim(), "A unique language prompt receives a reply");
  const feedbackResponse = await fetch(`${API}/api/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user: prompt, axen: firstReply, up: true }),
  });
  assert.equal(feedbackResponse.ok, true, "Feedback route accepts thumbs-up");
  const repeated = await ask(prompt);
  assert.equal(repeated.find((event) => event.type === "done")?.data, firstReply, "The liked reply is reused for the same prompt");
  assert.ok(repeated.some((event) => event.type === "stage" && /Recalled/.test(event.data)), "Liked reply is recalled from memory");

  console.log(JSON.stringify({
    passed: true,
    steps: status.steps,
    loss: status.loss,
    params: status.params,
    greeting: greeting.find((event) => event.type === "done")?.data,
    math: math.find((event) => event.type === "done")?.data,
    memoryRecall: recall.find((event) => event.type === "done")?.data,
    streamingEvents: greeting.filter((event) => event.type === "token").length,
    feedbackRecall: "passed",
  }, null, 2));
}

run().catch((error) => {
  console.error(error);
  process.exitCode = 1;
}).finally(() => {
  for (const snapshot of snapshots) {
    if (snapshot.existed) {
      fs.mkdirSync(path.dirname(snapshot.target), { recursive: true });
      fs.writeFileSync(snapshot.target, snapshot.data);
    } else if (fs.existsSync(snapshot.target)) {
      fs.rmSync(snapshot.target);
    }
  }
});
