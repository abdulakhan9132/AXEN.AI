// Bounded source inspection: never import or execute the inspected code.
const fs = require("node:fs");
const path = require("node:path");
const ROOT = path.resolve(__dirname, "..");
const EXT = {".js":"JavaScript",".mjs":"JavaScript",".cjs":"JavaScript",".jsx":"JavaScript",
  ".ts":"TypeScript",".tsx":"TypeScript",".py":"Python",".java":"Java",".go":"Go",".rs":"Rust",
  ".c":"C",".h":"C",".cpp":"C++",".hpp":"C++",".cs":"C#",".php":"PHP",".rb":"Ruby",
  ".swift":"Swift",".kt":"Kotlin",".html":"HTML",".css":"CSS",".sql":"SQL",
  ".json":"JSON",".md":"Markdown",".toml":"TOML",".yaml":"YAML",".yml":"YAML"};
const SKIP = new Set(["node_modules","vendor","__pycache__","dist","build","coverage","data",
  "storage","outputs","checkpoints","models","archive","venv","env"]);
const SENSITIVE = /(^\.|(?:^|[._-])(?:secrets?|credentials?|tokens?|passwords?|private|id_rsa|id_ed25519)(?:[._-]|$)|\.min\.|(?:package-lock|yarn\.lock|pnpm-lock))/i;
const LIMITS = Object.freeze({files:600,entries:8000,depth:10,fileBytes:192*1024,bytes:6*1024*1024,findings:80});
function inside(root, candidate) {
  const rel = path.relative(root,candidate);
  return !rel || (!path.isAbsolute(rel) && rel !== ".." && !rel.startsWith(".." + path.sep));
}
function resolveTarget(target=".", allowedRoot=ROOT) {
  if (typeof target !== "string" || !target.trim() || target.length>500 || /[\0\r\n]/.test(target)) throw new Error("Choose a folder inside the Axen project.");
  const root=fs.realpathSync(allowedRoot), candidate=path.resolve(root,target.trim());
  if (!inside(root,candidate)) throw new Error("Folder is outside Axen. Copy the source into an Axen project subfolder first.");
  let current=root;
  for (const part of path.relative(root,candidate).split(path.sep).filter(Boolean)) {
    if (part.startsWith(".") || SKIP.has(part.toLowerCase()) || SENSITIVE.test(part)) throw new Error("That folder is excluded from source inspection.");
    current=path.join(current,part);
    if (fs.lstatSync(current).isSymbolicLink()) throw new Error("Linked folders cannot be inspected.");
  }
  if (!inside(root,fs.realpathSync(candidate)) || !fs.statSync(candidate).isDirectory()) throw new Error("Choose a source folder.");
  return {root,candidate};
}
function clean(value) { return String(value).replace(/[\u0000-\u001f\u007f]/g,"").slice(0,180); }
function matches(text, regex, extract) {
  const found=[];
  for (const m of text.matchAll(regex)) {
    found.push({...extract(m),line:text.slice(0,m.index).split("\n").length});
    if (found.length>=LIMITS.findings) break;
  }
  return found;
}
function inspect(file,text,language) {
  const imports=[],symbols=[],routes=[];
  if (["JavaScript","TypeScript"].includes(language)) {
    imports.push(...matches(text,/(?:\bfrom\s*|\brequire\s*\(\s*|\bimport\s*\(\s*|\bimport\s*)["']([^"'\n]+)["']/g,m=>({name:clean(m[1])})));
    symbols.push(...matches(text,/\b(?:async\s+)?(?:function|class)\s+([A-Za-z_$][\w$]*)|\b(?:const|let)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:\([^;\n]*\)|[A-Za-z_$][\w$]*)\s*=>/g,m=>({name:m[1]||m[2]})));
    routes.push(...matches(text,/\b(?:app|router|server)\.(get|post|put|patch|delete|options|head)\s*\(\s*["']([^"'\n]+)["']/gi,m=>({method:m[1].toUpperCase(),route:clean(m[2])})));
    routes.push(...matches(text,/\b(?:url|pathname)\s*===?\s*["'](\/[^"'\n]*)["']/g,m=>({method:"CHECK",route:clean(m[1])})));
  } else if (language==="Python") {
    imports.push(...matches(text,/^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))/gm,m=>({name:m[1]||m[2]})));
    symbols.push(...matches(text,/^\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_]\w*)/gm,m=>({name:m[1]})));
    routes.push(...matches(text,/@(?:app|router)\.(get|post|put|patch|delete|options|head)\s*\(\s*["']([^"'\n]+)["']/gi,m=>({method:m[1].toUpperCase(),route:clean(m[2])})));
  } else {
    imports.push(...matches(text,/^\s*(?:#include\s*[<"]([^>"]+)[>"]|(?:use|import|require|using)\s+([\w.:/]+))/gm,m=>({name:clean(m[1]||m[2])})));
    symbols.push(...matches(text,/\b(?:class|interface|struct|enum|fn|func|fun|def|function)\s+([A-Za-z_]\w*)/g,m=>({name:m[1]})));
  }
  const signals=matches(text,/\b(eval|exec|spawn|execFile|system)\s*\(|\b(child_process|subprocess)\b|\b(fetch|requests\.(?:get|post)|https?\.request)\s*\(/g,m=>({kind:m[1]||m[2]?"Execution boundary":"Network boundary"}));
  return {path:file,language,lines:text.split("\n").length,bytes:Buffer.byteLength(text),imports,symbols,routes,signals};
}
async function analyze(target=".",{allowedRoot=ROOT}={}) {
  const {root,candidate}=resolveTarget(target,allowedRoot), files=[],manifests=[],warnings=[];
  let bytes=0,entries=0,skipped=0,truncated=false;
  async function walk(folder,depth) {
    if (depth>LIMITS.depth) {truncated=true;return;}
    const dir=await fs.promises.opendir(folder);
    for await (const entry of dir) {
      if (++entries>LIMITS.entries || files.length>=LIMITS.files || bytes>=LIMITS.bytes) {truncated=true;break;}
      const full=path.join(folder,entry.name);
      if (entry.isSymbolicLink() || entry.name.startsWith(".") || SKIP.has(entry.name.toLowerCase()) || SENSITIVE.test(entry.name)) {skipped++;continue;}
      if (entry.isDirectory()) {await walk(full,depth+1);continue;}
      const ext=path.extname(entry.name).toLowerCase();
      if (!entry.isFile() || !EXT[ext]) {skipped++;continue;}
      try {
        if (fs.lstatSync(full).isSymbolicLink() || !inside(candidate,fs.realpathSync(full))) {skipped++;continue;}
        const stat=fs.statSync(full);
        if (stat.size>LIMITS.fileBytes || bytes+stat.size>LIMITS.bytes) {skipped++;truncated=true;continue;}
        const text=fs.readFileSync(full,"utf8");
        if (text.includes("\0")) {skipped++;continue;}
        bytes+=stat.size;
        const relative=path.relative(candidate,full).replace(/\\/g,"/");
        files.push(inspect(relative,text,EXT[ext]));
        if (entry.name==="package.json") {
          try {
            const pkg=JSON.parse(text);
            manifests.push({path:relative,name:clean(pkg.name||path.basename(folder)),
              entry:typeof pkg.main==="string"?clean(pkg.main):null,
              scripts:Object.keys(pkg.scripts||{}).slice(0,30).map(clean),
              dependencies:[...new Set([...Object.keys(pkg.dependencies||{}),...Object.keys(pkg.devDependencies||{})])].slice(0,100).map(clean)});
          } catch {warnings.push("Could not parse "+clean(relative));}
        }
      } catch {warnings.push("Could not read "+clean(path.relative(candidate,full)));}
      if (files.length%20===0) await new Promise(resolve=>setImmediate(resolve));
    }
  }
  await walk(candidate,0);
  files.sort((a,b)=>a.path.localeCompare(b.path));
  const names=new Set(files.map(f=>f.path)),edges=[];
  for (const file of files) for (const imp of file.imports) {
    let base;
    if (["JavaScript","TypeScript"].includes(file.language) && imp.name.startsWith(".")) {
      base=path.posix.normalize(path.posix.join(path.posix.dirname(file.path),imp.name));
    } else if (file.language==="Python") {
      const dots=imp.name.match(/^\.+/)?.[0].length||0, moduleName=imp.name.slice(dots).replace(/\./g,"/");
      base=dots?path.posix.join(path.posix.dirname(file.path),"../".repeat(dots-1),moduleName):moduleName;
    } else continue;
    const resolved=[base,...[".js",".mjs",".cjs",".ts",".tsx",".jsx",".py","/index.js","/index.ts","/__init__.py"].map(e=>base+e)].find(p=>names.has(p));
    if (resolved && edges.length<3000) edges.push({from:file.path,to:resolved,line:imp.line});
  }
  const languages={};
  for (const file of files) languages[file.language]=(languages[file.language]||0)+1;
  return {target:path.relative(root,candidate).replace(/\\/g,"/")||".",created_at:new Date().toISOString(),mode:"static-source-inspection",
    summary:{files:files.length,lines:files.reduce((n,f)=>n+f.lines,0),bytes,languages,skipped,truncated},
    files,edges,manifests,warnings:warnings.slice(0,30),limits:LIMITS,
    limitations:["Pattern-based inspection: comments and strings may cause false matches.",
      "Runtime behavior, dynamic imports and generated code need separate verification.",
      "Binary decompilation and target execution are not performed.",
      "Hidden files, credentials, weights and user data folders are excluded."]};
}
function md(value) { return clean(value).replace(/[\\\x60*_{}\[\]<>|]/g,"\\$&"); }
function toMarkdown(r) {
  const s=r.summary;
  const lines=["# AXEN Reverse Engineering","","Folder: "+md(r.target),"",
    s.files+" source files · "+s.lines+" lines · "+r.edges.length+" resolved local import links",
    "","## Scope","",...r.limitations.map(x=>"- "+x),
    s.truncated?"\nScan reached a size limit. Inspect a smaller subfolder for fuller coverage.":"",
    "","## Languages","",...Object.entries(s.languages).map(([k,v])=>"- "+k+": "+v+" files"),
    "","## Package entry points",""];
  for (const m of r.manifests) lines.push("- "+md(m.path)+": "+md(m.name)+(m.entry?"; main "+md(m.entry):""),
    "  Scripts: "+m.scripts.map(md).join(", "),"  Dependencies: "+(m.dependencies.map(md).join(", ")||"none declared"));
  if (!r.manifests.length) lines.push("No package.json entry points found in this scan.");
  lines.push("","## Source map","");
  for (const f of r.files) {
    lines.push("### "+md(f.path),"",f.language+" · "+f.lines+" lines");
    if (f.symbols.length) lines.push("Symbols: "+f.symbols.map(x=>md(x.name)+" (line "+x.line+")").join(", "));
    if (f.routes.length) lines.push("Route patterns: "+f.routes.map(x=>md(x.method+" "+x.route)+" (line "+x.line+")").join(", "));
    const links=r.edges.filter(x=>x.from===f.path);
    if (links.length) lines.push("Local imports: "+links.map(x=>md(x.to)+" (line "+x.line+")").join(", "));
    if (f.signals.length) lines.push("Review boundaries: "+f.signals.map(x=>x.kind+" at line "+x.line).join(", "));
    lines.push("");
  }
  if (r.warnings.length) lines.push("## Unread files","",...r.warnings.map(md));
  return lines.join("\n");
}
module.exports={analyze,toMarkdown,resolveTarget,LIMITS};
