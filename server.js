// Node.js server for Axen AI (Plain Node.js 18+, no external npm packages)
const http = require("http");
const fs = require("fs");
const path = require("path");
const routes = require("./ai/routes");

const PORT = process.env.PORT || 8000;
const HOST = process.env.HOST || "127.0.0.1";
const TEMPLATE_DIR = path.join(__dirname, "app", "ui", "templates");
const STATIC_DIR = path.join(__dirname, "app", "ui", "static");
const CONV_FILE = path.join(__dirname, "data", "axen_conversations.json");
const FILE_UPLOADS_DIR = path.join(__dirname, "data", "storage", "uploads");
const FILE_METADATA_DIR = path.join(__dirname, "data", "storage", "metadata");

const MIME_TYPES = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
  ".woff": "font/woff",
  ".woff2": "font/woff2",
  ".ttf": "font/ttf",
  ".txt": "text/plain; charset=utf-8",
};

function loadConversations() {
  try {
    return JSON.parse(fs.readFileSync(CONV_FILE, "utf8"));
  } catch {
    return {};
  }
}

function saveConversations(convs) {
  try {
    fs.mkdirSync(path.dirname(CONV_FILE), { recursive: true });
    fs.writeFileSync(CONV_FILE, JSON.stringify(convs, null, 2));
  } catch (err) {
    console.error("Failed to save conversations:", err);
  }
}

async function readJson(req) {
  let body = "";
  for await (const chunk of req) body += chunk;
  return body ? JSON.parse(body) : {};
}

function sendJson(res, status, data) {
  res.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
  });
  res.end(JSON.stringify(data));
}

function serveStatic(res, filePath) {
  if (!fs.existsSync(filePath) || fs.statSync(filePath).isDirectory()) {
    res.writeHead(404, { "Content-Type": "text/plain" });
    return res.end("Not Found");
  }
  const ext = path.extname(filePath).toLowerCase();
  const mime = MIME_TYPES[ext] || "application/octet-stream";
  res.writeHead(200, { "Content-Type": mime });
  fs.createReadStream(filePath).pipe(res);
}

const server = http.createServer(async (req, res) => {
  if (req.method === "OPTIONS") {
    res.writeHead(204);
    return res.end();
  }

  // 1. Handled by Axen AI routes (STEP 2: Connect it to my server)
  try {
    if (await routes.handle(req, res)) {
      return;
    }
  } catch (err) {
    console.error("AI route error:", err);
    if (!res.headersSent) {
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: err.message }));
    }
    return;
  }

  const parsedUrl = new URL(req.url, `http://${req.headers.host || "127.0.0.1"}`);
  const pathname = parsedUrl.pathname;

  // 2. Web UI root and Vision UI
  if (req.method === "GET") {
    if (pathname === "/reverse-engineering") {
      return serveStatic(res, path.join(TEMPLATE_DIR, "reverse-engineering.html"));
    }
    if (pathname === "/" || pathname === "/index.html") {
      const indexPath = path.join(TEMPLATE_DIR, "index.html");
      return serveStatic(res, indexPath);
    }
    if (pathname === "/vision" || pathname === "/vision.html") {
      const visionPath = path.join(TEMPLATE_DIR, "vision.html");
      return serveStatic(res, visionPath);
    }
  }

  // 3. Static assets: /static/*
  if (req.method === "GET" && pathname.startsWith("/static/")) {
    const rel = pathname.slice("/static/".length);
    const safeRel = path.normalize(rel).replace(/^(\.\.[\/\\])+/, "");
    const fullPath = path.join(STATIC_DIR, safeRel);
    return serveStatic(res, fullPath);
  }

  // 3b. Static generated images: /images/*
  if (req.method === "GET" && pathname.startsWith("/images/")) {
    const rel = pathname.slice("/images/".length);
    const safeRel = path.normalize(rel).replace(/^(\.\.[\/\\])+/, "");
    const fullPath = path.join(__dirname, "outputs", "images", safeRel);
    return serveStatic(res, fullPath);
  }

  // 3c. Image generation API: /api/image & /api/generate-image
  if (req.method === "POST" && (pathname === "/api/image" || pathname === "/api/generate-image")) {
    const body = await readJson(req);
    const prompt = body.prompt;
    if (!prompt) return sendJson(res, 400, { error: "Prompt is required." });

    const pyBin = process.platform === "win32" && fs.existsSync(path.join(__dirname, ".venv", "Scripts", "python.exe"))
      ? path.join(__dirname, ".venv", "Scripts", "python.exe")
      : "python";
    const scriptPath = path.join(__dirname, "scripts", "generate_image.py");

    const { spawn } = require("child_process");
    const proc = spawn(pyBin, [scriptPath], { cwd: __dirname });
    let stdout = "";
    let stderr = "";

    proc.stdout.on("data", (chunk) => { stdout += chunk; });
    proc.stderr.on("data", (chunk) => { stderr += chunk; });

    proc.on("close", (code) => {
      try {
        const out = JSON.parse(stdout || "{}");
        if (code !== 0 || out.error) {
          const errMsg = out.error || stderr.trim() || `Image generation failed with code ${code}`;
          return sendJson(res, 500, { error: errMsg, detail: errMsg });
        }
        return sendJson(res, 200, out);
      } catch (err) {
        const errMsg = stderr.trim() || stdout.trim() || "Failed to parse image generation result";
        return sendJson(res, 500, { error: errMsg, detail: errMsg });
      }
    });

    proc.stdin.write(JSON.stringify(body));
    proc.stdin.end();
    return;
  }

  // 4. Session management (keeping user's chat history and sidebar exactly as they are)
  if (req.method === "GET" && pathname === "/api/sessions") {
    const convs = loadConversations();
    const q = (parsedUrl.searchParams.get("q") || "").toLowerCase();
    let sessions = Object.values(convs);
    if (q) {
      sessions = sessions.filter(
        (s) =>
          (s.title && s.title.toLowerCase().includes(q)) ||
          (s.messages && s.messages.some((m) => m.content && m.content.toLowerCase().includes(q)))
      );
    }
    sessions.sort((a, b) => new Date(b.updated_at || 0) - new Date(a.updated_at || 0));
    return sendJson(res, 200, {
      sessions,
      active_session_id: sessions[0] ? sessions[0].id : null,
    });
  }

  if (req.method === "POST" && (pathname === "/api/sessions" || pathname === "/api/sessions/new")) {
    const body = await readJson(req);
    const convs = loadConversations();
    const id = "sess_" + Date.now() + "_" + Math.random().toString(36).substring(2, 7);
    const now = new Date().toISOString();
    const newSession = {
      id,
      title: body.title || "New Chat",
      created_at: now,
      updated_at: now,
      messages: [],
    };
    convs[id] = newSession;
    saveConversations(convs);
    return sendJson(res, 201, { session_id: id, session: newSession });
  }

  if (pathname.startsWith("/api/sessions/")) {
    const subPath = pathname.slice("/api/sessions/".length);
    const convs = loadConversations();
    
    // Check if appending messages: /api/sessions/:id/messages
    if (subPath.includes("/messages")) {
      const sessId = subPath.split("/")[0];
      const sess = convs[sessId];
      if (!sess) return sendJson(res, 404, { error: "Session not found" });
      if (req.method === "POST") {
        const body = await readJson(req);
        if (!sess.messages) sess.messages = [];
        sess.messages.push(body);
        sess.updated_at = new Date().toISOString();
        if ((!sess.title || sess.title === "New Chat") && body.role === "user" && body.content) {
          sess.title = body.content.slice(0, 32).trim() || "New Chat";
        }
        saveConversations(convs);
        return sendJson(res, 200, { success: true, session: sess });
      }
    }

    const sessId = subPath;
    if (req.method === "GET") {
      const sess = convs[sessId];
      if (!sess) return sendJson(res, 404, { error: "Session not found" });
      return sendJson(res, 200, { id: sess.id, session: sess, messages: sess.messages || [] });
    }
    if (req.method === "DELETE") {
      delete convs[sessId];
      saveConversations(convs);
      const remaining = Object.values(convs);
      return sendJson(res, 200, { success: true, active_session_id: remaining[0] ? remaining[0].id : null });
    }
  }

  // 5. General compatibility endpoints
  if (req.method === "GET" && (pathname === "/api/status" || pathname === "/api/health")) {
    return sendJson(res, 200, {
      status: "running",
      server: "Axen Native Brain",
    });
  }

  if (req.method === "GET" && pathname === "/api/projects") {
    return sendJson(res, 200, { projects: [] });
  }

  if (req.method === "GET" && pathname === "/api/files") {
    let files = [];
    try {
      files = fs.readdirSync(FILE_METADATA_DIR, { withFileTypes: true })
        .filter((entry) => entry.isFile() && /^[a-z0-9_-]+\.json$/i.test(entry.name))
        .map((entry) => {
          try { return JSON.parse(fs.readFileSync(path.join(FILE_METADATA_DIR, entry.name), "utf8")); }
          catch { return null; }
        })
        .filter((file) => file && typeof file.file_id === "string" && typeof file.filename === "string")
        .sort((a, b) => (b.created_at || 0) - (a.created_at || 0));
    } catch (error) {
      if (error.code !== "ENOENT") throw error;
    }
    return sendJson(res, 200, { files });
  }

  const deleteFileMatch = pathname.match(/^\/api\/files\/([a-z0-9_-]+)$/i);
  if (req.method === "DELETE" && deleteFileMatch) {
    const fileId = deleteFileMatch[1];
    const metadataPath = path.join(FILE_METADATA_DIR, `${fileId}.json`);
    try {
      const meta = JSON.parse(fs.readFileSync(metadataPath, "utf8"));
      const storedName = meta.stored_filename;
      if (meta.file_id !== fileId || typeof storedName !== "string" || path.basename(storedName) !== storedName || !storedName.startsWith(`${fileId}.`)) {
        return sendJson(res, 400, { error: "Stored file metadata is invalid." });
      }
      fs.unlinkSync(path.join(FILE_UPLOADS_DIR, storedName));
      fs.unlinkSync(metadataPath);
      return sendJson(res, 200, { ok: true, file_id: fileId });
    } catch (error) {
      if (error.code === "ENOENT") return sendJson(res, 404, { error: "File not found." });
      return sendJson(res, 500, { error: "Could not delete the file." });
    }
  }

  if (req.method === "GET" && pathname === "/api/memory") {
    const memBrain = require("./ai/brain/memory");
    return sendJson(res, 200, { stats: memBrain.stats() });
  }

  // 5b. Unified Model Management
  if (req.method === "GET" && pathname === "/api/models") {
    return sendJson(res, 200, {
      models: [
        {
          id: "router",
          name: "Auto Router (Intelligent)",
          provider: "hybrid",
          type: "local",
          context_length: 4096,
          status: "active",
          available: true,
          description: "Routes requests automatically between local neural brain, coding models, and solvers."
        },
        {
          id: "own_brain",
          name: "AXEN Local Brain (Neural)",
          provider: "node_transformer",
          type: "local",
          context_length: 1024,
          status: "ready",
          available: true,
          description: "Ultra-lightweight zero-GPU local neural network running on Node.js."
        },
        {
          id: "qwen2.5:1.5b",
          name: "Qwen 2.5 Coder (1.5B Q4)",
          provider: "ollama",
          type: "local",
          context_length: 8192,
          status: "ready",
          available: false,
          description: "Specialized code generation & technical reasoning model optimized for 4GB VRAM."
        },
        {
          id: "tinyllama",
          name: "TinyLlama (1.1B PyTorch CUDA)",
          provider: "pytorch_hf",
          type: "local",
          context_length: 2048,
          status: "ready",
          available: true,
          description: "Direct local PyTorch/Transformers inference with CUDA acceleration."
        },
        {
          id: "cloud_openrouter",
          name: "OpenRouter Cloud API (Optional)",
          provider: "openrouter",
          type: "cloud",
          context_length: 128000,
          status: "ready",
          available: !!(process.env.OPENROUTER_API_KEY || process.env.CLOUD_API_KEY),
          description: "Optional cloud provider for frontier models (Claude 3.5 Sonnet, GPT-4o, DeepSeek)."
        }
      ],
      active_model: "router"
    });
  }

  if (req.method === "POST" && pathname === "/api/models/select") {
    const body = await readJson(req);
    const modelId = body.model_id || "router";
    return sendJson(res, 200, {
      success: true,
      active_model: modelId,
      message: `Switched active model to ${modelId}`
    });
  }

  // Not found fallback
  res.writeHead(404, { "Content-Type": "application/json" });
  res.end(JSON.stringify({ error: "Not found", path: pathname }));
});

server.listen(PORT, HOST, () => {
  console.log(`\n======================================================`);
  console.log(`         AXEN® — Local Brain AI Assistant`);
  console.log(`======================================================`);
  console.log(`  Interface : http://${HOST}:${PORT}`);
  console.log(`  Backend   : Pure Node.js (No cloud, No Ollama, No npm)`);
  console.log(`  Brain     : Custom Transformer (Scratch GPT-style)`);
  console.log(`======================================================\n`);
});

module.exports = server;
