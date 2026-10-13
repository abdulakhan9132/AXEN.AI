const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const http = require("node:http");
const { analyze, toMarkdown } = require("../ai/reverse-engineering");

test("memory arguments remain data when chat text contains shell characters", t => {
  const cp=require("node:child_process");
  const calls=[];
  t.mock.method(cp,"execFileSync",(exe,args,options)=>{calls.push({exe,args,options});return "null";});
  const modulePath=require.resolve("../ai/brain/memory");
  delete require.cache[modulePath];
  const memory=require("../ai/brain/memory");
  const fact='value " & echo should-not-execute';
  memory.handleCommand("remember that "+fact);
  assert.equal(calls.length,1);
  assert.equal(calls[0].args[2],fact);
  assert.equal(calls[0].options.shell,undefined);
  assert.equal(calls[0].options.windowsHide,true);
  delete require.cache[modulePath];
});

test("source inspection resolves dependencies and never executes the target",async () => {
  const root=fs.mkdtempSync(path.join(os.tmpdir(),"axen-inspection-"));
  try {
    fs.writeFileSync(path.join(root,"package.json"),JSON.stringify({name:"fixture",main:"index.js",scripts:{start:"DO NOT EXECUTE"},dependencies:{express:"1"}}));
    fs.writeFileSync(path.join(root,"index.js"),'const helper = require("./helper");\nfunction greet() {}\napp.get("/coffee", greet);\nthrow new Error("MUST NOT EXECUTE");\n');
    fs.writeFileSync(path.join(root,"helper.js"),'exports.twice = n => n * 2;\n');
    fs.mkdirSync(path.join(root,"service"));
    fs.writeFileSync(path.join(root,"service","main.py"),'from service.helper import double\n@app.post("/items")\ndef create():\n  return double(2)\n');
    fs.writeFileSync(path.join(root,"service","helper.py"),'def double(value):\n  return value * 2\n');
    fs.writeFileSync(path.join(root,".env"),"PRIVATE_VALUE=hidden");
    fs.writeFileSync(path.join(root,"secrets.json"),'{"token":"hidden"}');
    fs.mkdirSync(path.join(root,"node_modules"));
    fs.writeFileSync(path.join(root,"node_modules","ignored.js"),"function privateName(){}");
    fs.writeFileSync(path.join(root,"binary.js"),Buffer.from([0,1,2]));
    const r=await analyze(".",{allowedRoot:root});
    assert.equal(r.summary.files,5);
    assert.ok(r.edges.some(e=>e.from==="index.js" && e.to==="helper.js" && e.line===1));
    assert.ok(r.edges.some(e=>e.from==="service/main.py" && e.to==="service/helper.py"));
    assert.ok(r.files.find(f=>f.path==="index.js").routes.some(r=>r.method==="GET" && r.route==="/coffee" && r.line===3));
    assert.ok(r.files.find(f=>f.path==="service/main.py").symbols.some(s=>s.name==="create"));
    assert.deepEqual(r.manifests[0].scripts,["start"]);
    assert.ok(!JSON.stringify(r).includes("PRIVATE_VALUE"));
    assert.ok(!JSON.stringify(r).includes("privateName"));
    assert.match(toMarkdown(r),/line 3/);
    await assert.rejects(analyze("..",{allowedRoot:root}),/outside Axen/);
    await assert.rejects(analyze("node_modules",{allowedRoot:root}),/excluded/);
    await assert.rejects(analyze({bad:true},{allowedRoot:root}),/Choose/);
    // Directory junctions model the Windows escape scenario without symlink privileges.
    const outside=fs.mkdtempSync(path.join(os.tmpdir(),"axen-outside-"));
    try {
      fs.symlinkSync(outside,path.join(root,"linked"),process.platform==="win32"?"junction":"dir");
      await assert.rejects(analyze("linked",{allowedRoot:root}),/Linked/);
      assert.ok(!(await analyze(".",{allowedRoot:root})).files.some(f=>f.path.startsWith("linked")));
    } finally {
      if(fs.existsSync(path.join(root,"linked"))) fs.unlinkSync(path.join(root,"linked"));
      fs.rmdirSync(outside);
    }
  } finally {
    // Only the temporary fixture created by this test is removed.
    assert.ok(root.startsWith(path.join(os.tmpdir(),"axen-inspection-")));
    fs.rmSync(root,{recursive:true,force:true});
  }
});

test("large source and hidden target folders are excluded",async () => {
  const root=fs.mkdtempSync(path.join(os.tmpdir(),"axen-limits-"));
  try {
    fs.writeFileSync(path.join(root,"huge.js"),"x".repeat(200*1024));
    fs.mkdirSync(path.join(root,".private"));
    const r=await analyze(".",{allowedRoot:root});
    assert.equal(r.summary.files,0);assert.equal(r.summary.truncated,true);
    await assert.rejects(analyze(".private",{allowedRoot:root}),/excluded/);
  } finally {assert.ok(root.startsWith(path.join(os.tmpdir(),"axen-limits-")));fs.rmSync(root,{recursive:true,force:true});}
});

test("HTTP inspection, error responses, and chat command work without a language model",async () => {
  const routes=require("../ai/routes");
  const server=http.createServer(async (req,res)=>{try{if(!await routes.handle(req,res)){res.writeHead(404);res.end();}}catch(e){res.writeHead(500);res.end(e.message);}});
  await new Promise(resolve=>server.listen(0,"127.0.0.1",resolve));
  const url="http://127.0.0.1:"+server.address().port;
  try {
    const post=async(data,extra={})=>fetch(url+"/api/reverse-engineering",{method:"POST",headers:{"Content-Type":"application/json",...extra},body:JSON.stringify(data)});
    const good=await post({target:"ai"});
    assert.equal(good.status,200);
    const payload=await good.json();
    assert.ok(payload.report.files.some(f=>f.path==="routes.js"));
    assert.match(payload.markdown,/Source map/);
    assert.equal((await post({target:"../"})).status,400);
    assert.equal((await post({target:"ai"},{Origin:"https://unrelated.example"})).status,403);
    const bad=await fetch(url+"/api/reverse-engineering",{method:"POST",headers:{"Content-Type":"application/json"},body:"{"});
    assert.equal(bad.status,400);
    const chat=await fetch(url+"/api/chat",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({messages:[{role:"user",content:"/reverse ai"}]})});
    const stream=await chat.text();
    assert.match(stream,/"type":"done"/);assert.match(stream,/AXEN Reverse Engineering/);
  } finally {server.closeAllConnections();await new Promise(resolve=>server.close(resolve));}
});
