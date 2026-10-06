// Plug-in routes for Axen's brain. Call handle(req, res) from your server; it returns true if it handled the request.
const axen = require("./brain/axen");
const learner = require("./brain/learner");

async function readJson(req) {
  let body = "";
  for await (const chunk of req) body += chunk;
  return body ? JSON.parse(body) : {};
}
const json = (res, obj) => { res.writeHead(200, { "Content-Type": "application/json" }); res.end(JSON.stringify(obj)); };

async function handle(req, res) {
  const url = req.url.split("?")[0];

  if (req.method === "POST" && url === "/api/chat") {
    const { messages = [] } = await readJson(req);
    const last = [...messages].reverse().find((m) => m.role === "user");
    res.writeHead(200, { "Content-Type": "text/event-stream", "Cache-Control": "no-cache", Connection: "keep-alive" });
    const send = (type, data) => res.write("data: " + JSON.stringify({ type, data }) + "\n\n");
    try {
      let answer;
      answer = await axen.respond(last ? last.content : "", send);
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
