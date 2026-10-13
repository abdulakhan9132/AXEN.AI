// Focused training with separate validation and a replay regression gate.
const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");
const { Model } = require("./model");
const tok = require("./tokenizer");
const DATA = path.resolve(__dirname,"../data");
const WEIGHTS = path.join(DATA,"brain.json");
const OUTPUT = path.join(DATA,"training");
const digest = data => crypto.createHash("sha256").update(data).digest("hex");
function records(name) {
  return fs.readFileSync(path.join(DATA,"curriculum",name),"utf8").trim().split(/\r?\n/).map(line => {
    const entry=JSON.parse(line);
    if (typeof entry.prompt!=="string" || typeof entry.answer!=="string" || !entry.prompt || !entry.answer) throw new Error("Invalid curriculum entry.");
    return entry;
  });
}
const formatted = x => "User: "+x.prompt+"\nAxen: "+x.answer+"\n\n";
function score(model,examples) {
  let loss=0,count=0;
  for (const example of examples) {
    const prefix=tok.encode("User: "+example.prompt+"\nAxen: ");
    const ids=tok.encode(formatted(example));
    for (let start=0;start<ids.length-1;start+=model.cfg.T) {
      const input=ids.slice(start,start+model.cfg.T);
      const {logits}=model.forward(input);
      for (let i=0;i<input.length && start+i+1<ids.length;i++) {
        if (start+i+1<prefix.length) continue;
        const off=i*tok.VOCAB_SIZE;
        let max=-Infinity;
        for(let j=0;j<tok.VOCAB_SIZE;j++) max=Math.max(max,logits[off+j]);
        let sum=0;
        for(let j=0;j<tok.VOCAB_SIZE;j++) sum+=Math.exp(logits[off+j]-max);
        loss+=Math.log(sum)+max-logits[off+ids[start+i+1]];
        count++;
      }
    }
  }
  return loss/count;
}
async function samples(model,examples) {
  const result=[];
  for (const e of examples.slice(0,3)) {
    const output=await model.generate(tok.encode("User: "+e.prompt+"\nAxen:"),{topK:1,maxTokens:120,stop:["\nUser:","\n\n"]});
    result.push({prompt:e.prompt,response:tok.decode(output).trim()});
  }
  return result;
}
function answerChecks(outputs) {
  const expected = [/\b(?:source|files?|dependenc\w*)\b/i, /\b(?:route|handler|calls?|request)\b/i, /\b(?:runtime|dynamic|comments?|patterns?)\b/i];
  return outputs.filter((sample, i) => expected[i]?.test(sample.response)).length;
}
async function run({steps=300,promote=false}={}) {
  if (!Number.isInteger(steps) || steps<1 || steps>5000) throw new Error("Choose 1 to 5000 steps.");
  const training=records("reverse-train.jsonl"), evaluation=records("reverse-eval.jsonl");
  if (evaluation.some(x=>training.some(t=>x.prompt===t.prompt || x.answer===t.answer))) throw new Error("Training/evaluation overlap.");
  const starter=fs.readFileSync(path.join(DATA,"corpus","starter.txt"),"utf8");
  const replay=[...new Map(starter.trim().split(/\r?\n\r?\n/).map(block=>{
    const match=block.match(/^User: (.+)\r?\nAxen: ([\s\S]+)$/);
    return match?[match[1],{prompt:match[1],answer:match[2]}]:[null,null];
  }).filter(([key])=>key)).values()];
  const originalBytes=fs.readFileSync(WEIGHTS), originalHash=digest(originalBytes);
  const model=Model.load(WEIGHTS), originalSteps=model.trained;
  // Small learning rate and replay reduce damage to existing language patterns.
  model.cfg.lr=0.0002;
  const replayIds=tok.encode(replay.map(formatted).join(""));
  const trainIds=tok.encode(training.map(formatted).join(""));
  fs.mkdirSync(OUTPUT,{recursive:true});
  const before={domain:score(model,evaluation),replay:score(model,replay)};
  const beforeSamples=await samples(model,evaluation);
  const history=[], started=Date.now();
  const maxReplay=before.replay+Math.max(0.02,before.replay*0.10);
  let best=null;
  const bestPath=path.join(OUTPUT,"candidate-best.json"), candidatePath=path.join(OUTPUT,"candidate-last.json");
  const random=Math.random; let seed=42;
  Math.random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;};
  try {
    for(let i=1;i<=steps;i++) {
      model.trainStep(i%3===0?trainIds:replayIds);
      if(i%50===0 || i===steps) {
        const scores={step:i,domain:score(model,evaluation),replay:score(model,replay)};
        history.push(scores);
        console.log(JSON.stringify({progress:i,steps,domain:scores.domain,replay:scores.replay,elapsed_seconds:Math.round((Date.now()-started)/1000)}));
        if(scores.domain<before.domain && scores.replay<=maxReplay && (!best || scores.domain<best.domain)) {
          best=scores;model.save(bestPath);
        }
      }
    }
  } finally {Math.random=random;}
  model.save(candidatePath);
  const last=history[history.length-1], chosen=best?Model.load(bestPath):model;
  const afterSamples=await samples(chosen,evaluation);
  const taskChecks={before:answerChecks(beforeSamples),after:answerChecks(afterSamples),total:3,required:3};
  const eligible=Boolean(best && taskChecks.after>=taskChecks.required && taskChecks.after>=taskChecks.before);
  let promoted=false,backup=null;
  if(promote && eligible) {
    if(digest(fs.readFileSync(WEIGHTS))!==originalHash) throw new Error("Brain changed during training. Candidate preserved; activation cancelled.");
    backup=path.join(OUTPUT,"brain-before-"+Date.now()+".json");
    fs.writeFileSync(backup,originalBytes,{flag:"wx"});
    fs.copyFileSync(bestPath,WEIGHTS+".next");
    fs.renameSync(WEIGHTS+".next",WEIGHTS);
    promoted=true;
  }
  const report={created_at:new Date().toISOString(),steps_run:steps,original_steps:originalSteps,
    candidate_steps:chosen.trained,parameters:chosen.paramCount(),train_examples:training.length,eval_examples:evaluation.length,
    seed:42,learning_rate:model.cfg.lr,before,after:best||last,last,history,max_replay_loss:maxReplay,
    promoted,eligible,task_checks:taskChecks,backup:backup?path.basename(backup):null,candidate:best?"candidate-best.json":"candidate-last.json",
    reason:eligible?(promoted?"Candidate passed loss and basic answer checks and was activated.":"Candidate passed the limited checks; activation was not requested."):"Candidate failed the loss or basic answer checks. Existing brain retained.",
    before_samples:beforeSamples,after_samples:afterSamples,
    original_sha256:originalHash,training_sha256:digest(JSON.stringify(training)),evaluation_sha256:digest(JSON.stringify(evaluation)),
    limitations:["Evaluation measures next-character prediction on a small original curriculum.",
      "Three basic keyword checks reject obviously irrelevant samples; they are not a comprehensive task benchmark.",
      "Lower loss does not prove task competence, instruction following or frontier-model parity.",
      "Source inspection is a separate deterministic tool, not a learned model skill."]};
  fs.writeFileSync(path.join(OUTPUT,"latest.json"),JSON.stringify(report,null,2));
  console.log(JSON.stringify({complete:true,promoted,reason:report.reason,before,after:report.after,report:path.join(OUTPUT,"latest.json")}));
  return report;
}
if(require.main===module)run({steps:Number(process.argv[2]||300),promote:process.argv.includes("--promote")}).catch(e=>{console.error(e.message);process.exitCode=1;});
module.exports={run,score};
