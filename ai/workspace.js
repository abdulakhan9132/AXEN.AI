// Local project tools for Axen's online coding agent. All paths stay under data/generated-sites.
const fs = require("fs");
const path = require("path");
const http = require("http");

const GENERATED_ROOT = path.resolve(__dirname, "..", "data", "generated-sites");
const previews = new Map();
const MIME = {
  ".html": "text/html; charset=utf-8", ".htm": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8", ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml", ".png": "image/png", ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif",
  ".ico": "image/x-icon", ".txt": "text/plain; charset=utf-8", ".md": "text/markdown; charset=utf-8",
  ".py": "text/plain; charset=utf-8", ".ts": "text/plain; charset=utf-8", ".tsx": "text/plain; charset=utf-8",
  ".jsx": "text/plain; charset=utf-8", ".java": "text/plain; charset=utf-8", ".go": "text/plain; charset=utf-8",
  ".rs": "text/plain; charset=utf-8", ".php": "text/plain; charset=utf-8", ".rb": "text/plain; charset=utf-8",
};

function ensureRoot() {
  fs.mkdirSync(GENERATED_ROOT, { recursive: true });
}

function projectPath(projectId) {
  if (typeof projectId !== "string" || !/^[a-z0-9][a-z0-9-]{2,70}$/.test(projectId)) {
    throw new Error("Project ID is invalid.");
  }
  const root = path.resolve(GENERATED_ROOT, projectId);
  if (!root.startsWith(GENERATED_ROOT + path.sep)) throw new Error("Project path is outside Axen's workspace.");
  return root;
}

function safeRelativePath(relativePath) {
  if (typeof relativePath !== "string" || !relativePath.trim() || relativePath.length > 240) {
    throw new Error("A short relative file path is required.");
  }
  const normalized = relativePath.replace(/\\/g, "/");
  const parts = normalized.split("/");
  if (normalized.startsWith("/") || /^[a-zA-Z]:/.test(normalized) || parts.some((part) => !part || part === "." || part === "..")) {
    throw new Error("Use a relative path inside the project.");
  }
  if (parts.some((part) => [".git", "node_modules", ".ssh"].includes(part.toLowerCase()) || part.toLowerCase().startsWith(".env"))) {
    throw new Error("That protected path is not available in generated projects.");
  }
  return parts;
}

function assertNoSymlinks(root, parts, includeLast = true) {
  let current = root;
  const count = includeLast ? parts.length : Math.max(0, parts.length - 1);
  for (const part of parts.slice(0, count)) {
    current = path.join(current, part);
    try {
      const stat = fs.lstatSync(current);
      if (stat.isSymbolicLink()) throw new Error("Symbolic links are not allowed in the project workspace.");
    } catch (error) {
      if (error.code !== "ENOENT") throw error;
    }
  }
}

function createProject({ name, description = "" }) {
  ensureRoot();
  const label = String(name || "Axen project").trim().slice(0, 70);
  const slug = label.toLowerCase().normalize("NFKD").replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 42) || "axen-project";
  const projectId = `${slug}-${require("crypto").randomBytes(3).toString("hex")}`;
  const root = projectPath(projectId);
  fs.mkdirSync(root, { recursive: false });
  const metadata = { id: projectId, name: label, description: String(description).slice(0, 500), created_at: new Date().toISOString() };
  fs.writeFileSync(path.join(root, ".axen-project.json"), JSON.stringify(metadata, null, 2), { flag: "wx" });
  return metadata;
}

function listProjects() {
  ensureRoot();
  return fs.readdirSync(GENERATED_ROOT, { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && !entry.isSymbolicLink())
    .slice(0, 100)
    .map((entry) => {
      const root = path.join(GENERATED_ROOT, entry.name);
      try {
        const meta = JSON.parse(fs.readFileSync(path.join(root, ".axen-project.json"), "utf8"));
        return { ...meta, file_count: listFiles({ project_id: meta.id }).length, preview_url: previews.get(meta.id)?.url || null };
      } catch { return null; }
    }).filter(Boolean).sort((a, b) => b.created_at.localeCompare(a.created_at));
}

function listFiles({ project_id }) {
  const root = projectPath(project_id);
  const files = [];
  const walk = (folder, depth = 0) => {
    if (depth > 8 || files.length >= 200) return;
    for (const entry of fs.readdirSync(folder, { withFileTypes: true })) {
      if (entry.name === ".axen-project.json" || entry.isSymbolicLink()) continue;
      const fullPath = path.join(folder, entry.name);
      if (entry.isDirectory()) walk(fullPath, depth + 1);
      else if (entry.isFile()) files.push(path.relative(root, fullPath).replace(/\\/g, "/"));
      if (files.length >= 200) break;
    }
  };
  if (!fs.existsSync(root)) throw new Error("Project not found.");
  walk(root);
  return files.sort();
}

function readFile({ project_id, path: relativePath }) {
  const root = projectPath(project_id);
  const parts = safeRelativePath(relativePath);
  assertNoSymlinks(root, parts);
  const target = path.join(root, ...parts);
  const stat = fs.statSync(target);
  if (!stat.isFile() || stat.size > 1024 * 1024) throw new Error("Only files up to 1 MB can be read by the agent.");
  return fs.readFileSync(target, "utf8");
}

function writeFile({ project_id, path: relativePath, content }) {
  const root = projectPath(project_id);
  const parts = safeRelativePath(relativePath);
  if (!fs.existsSync(root)) throw new Error("Project not found.");
  if (typeof content !== "string" || Buffer.byteLength(content, "utf8") > 1024 * 1024) throw new Error("A file must be under 1 MB.");
  if (!fs.existsSync(path.join(root, ...parts)) && listFiles({ project_id }).length >= 200) throw new Error("This project reached its 200-file limit.");
  assertNoSymlinks(root, parts, false);
  const target = path.join(root, ...parts);
  fs.mkdirSync(path.dirname(target), { recursive: true });
  assertNoSymlinks(root, parts);
  if (fs.existsSync(target) && fs.lstatSync(target).isSymbolicLink()) throw new Error("Symbolic links cannot be overwritten.");
  fs.writeFileSync(target, content, "utf8");
  return { path: parts.join("/"), bytes: Buffer.byteLength(content, "utf8") };
}

function safePreviewFile(root, pathname) {
  let decoded;
  try { decoded = decodeURIComponent(pathname); } catch { return null; }
  const parts = decoded.replace(/\\/g, "/").split("/").filter(Boolean);
  if (parts.some((part) => part === "." || part === ".." || part.toLowerCase() === ".git")) return null;
  const target = path.resolve(root, ...parts);
  if (target !== root && !target.startsWith(root + path.sep)) return null;
  assertNoSymlinks(root, parts);
  let resolved = target;
  if (fs.existsSync(resolved) && fs.statSync(resolved).isDirectory()) resolved = path.join(resolved, "index.html");
  return resolved;
}

async function startPreview({ project_id }) {
  const root = projectPath(project_id);
  if (!fs.existsSync(path.join(root, "index.html"))) throw new Error("Create an index.html file before starting this web preview.");
  const existing = previews.get(project_id);
  if (existing?.server.listening) return { project_id, url: existing.url };
  const server = http.createServer((req, res) => {
    if (!(["GET", "HEAD"].includes(req.method))) {
      res.writeHead(405, { "Content-Type": "text/plain; charset=utf-8", Allow: "GET, HEAD" });
      return res.end("Method not allowed");
    }
    const target = safePreviewFile(root, new URL(req.url, "http://127.0.0.1").pathname);
    if (!target || !fs.existsSync(target) || !fs.statSync(target).isFile()) {
      res.writeHead(404, { "Content-Type": "text/plain; charset=utf-8", "X-Content-Type-Options": "nosniff" });
      return res.end("Preview file not found");
    }
    res.writeHead(200, {
      "Content-Type": MIME[path.extname(target).toLowerCase()] || "application/octet-stream",
      "X-Content-Type-Options": "nosniff",
      "Referrer-Policy": "no-referrer",
      "Content-Security-Policy": "default-src 'self' https: data: blob:; img-src 'self' https: data: blob:; style-src 'self' 'unsafe-inline' https:; script-src 'self' 'unsafe-inline' https:; connect-src 'self' https:; font-src 'self' https: data:; object-src 'none'; base-uri 'self';",
    });
    if (req.method === "HEAD") return res.end();
    fs.createReadStream(target).pipe(res);
  });
  server.unref();
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address();
  const url = `http://127.0.0.1:${address.port}/`;
  previews.set(project_id, { server, url });
  return { project_id, url };
}

function stopPreview(project_id) {
  const preview = previews.get(project_id);
  if (!preview) return false;
  preview.server.close();
  previews.delete(project_id);
  return true;
}

module.exports = { createProject, listProjects, listFiles, readFile, writeFile, startPreview, stopPreview, GENERATED_ROOT };
