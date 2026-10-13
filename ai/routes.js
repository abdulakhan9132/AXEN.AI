// Plug-in routes for Axen's brain. Call handle(req, res) from your server; it returns true if it handled the request.
const axen = require("./brain/axen");
const learner = require("./brain/learner");
const reverse = require("./reverse-engineering");
const fs = require("node:fs");
const path = require("node:path");
let inspecting = false;

async function inspect(target) {
  if (inspecting) throw new Error("An inspection is already running. Try again when it finishes.");
  inspecting = true;
  try { return await reverse.analyze(target); }
  finally { inspecting = false; }
}

async function readJson(req) {
  let body = "";
  for await (const chunk of req) {
    body += chunk;
    if (Buffer.byteLength(body) > 1024 * 1024) throw new Error("Request is too large.");
  }
  return body ? JSON.parse(body) : {};
}
const json = (res, obj) => { res.writeHead(200, { "Content-Type": "application/json" }); res.end(JSON.stringify(obj)); };

async function handle(req, res) {
  const url = req.url.split("?")[0];

  if (req.method === "GET" && url === "/api/reverse-engineering/status") {
    let training = null;
    try { training = JSON.parse(fs.readFileSync(path.join(__dirname, "data", "training", "latest.json"), "utf8")); } catch {}
    json(res, { available: true, busy: inspecting, mode: "static-source-inspection", limits: reverse.LIMITS, training });
    return true;
  }
  if (req.method === "POST" && url === "/api/reverse-engineering") {
    if (req.headers.origin && req.headers.origin !== `http://${req.headers.host}`) {
      res.writeHead(403, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Open the inspector from Axen to continue." }));
      return true;
    }
    try {
      const { target = "." } = await readJson(req);
      const report = await inspect(target);
      json(res, { report, markdown: reverse.toMarkdown(report) });
    } catch (error) {
      res.writeHead(400, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: error.code === "ENOENT" ? "Folder not found inside the Axen project." : error.message }));
    }
    return true;
  }

  if (req.method === "POST" && url === "/api/chat") {
    const { messages = [] } = await readJson(req);
    const last = [...messages].reverse().find((m) => m.role === "user");
    res.writeHead(200, { "Content-Type": "text/event-stream", "Cache-Control": "no-cache", Connection: "keep-alive" });
    const send = (type, data) => res.write("data: " + JSON.stringify({ type, data }) + "\n\n");
    try {
      let answer;
      const content = last && typeof last.content === "string" ? last.content : "";
      const command = content.match(/^\s*\/(?:reverse|inspect)(?:\s+(.+))?\s*$/i);
      const simpleRequest = /^(?:please\s+)?reverse[- ]engineer\s+(?:this|my|the)\s+(?:project|codebase|app)[.!?]?$/i.test(content.trim());
      if (command || simpleRequest) {
        send("stage", "Inspecting source files and imports");
        const target = command?.[1]?.replace(/^["']|["']$/g, "") || ".";
        const report = await inspect(target);
        answer = reverse.toMarkdown(report);
        send("token", answer);
      } else {
        answer = await axen.respond(content, send);
      }
      send("done", answer);
    } catch (e) { send("error", e.message); }
    res.end();
    return true;
  }
  if (req.method === "POST" && url === "/api/provider/chat") {
    const { messages = [] } = await readJson(req);
    const last = [...messages].reverse().find((m) => m.role === "user");
    res.writeHead(200, { "Content-Type": "text/event-stream", "Cache-Control": "no-cache", Connection: "keep-alive" });
    const send = (type, data) => res.write("data: " + JSON.stringify({ type, data }) + "\n\n");
    try {
      let answer = await axen.providerRespond(last ? last.content : "", send);
      send("done", answer);
    } catch (e) { send("error", e.message); }
    res.end();
    return true;
  }
  if (req.method === "GET" && url === "/api/brain") { json(res, { ...axen.status(), sleeping: learner.isSleeping() }); return true; }
  if (req.method === "POST" && url === "/api/feedback") {
    const { user, axen: reply, up } = await readJson(req);
    axen.feedback(user, reply, !!up);
    const learning = up ? learner.learnFromFeedback(10) : null;
    json(res, { ok: true, learning });
    return true;
  }
  if (req.method === "POST" && url === "/api/sleep") { json(res, { started: learner.sleep(500) }); return true; }
  return false;
}
module.exports = { handle };
