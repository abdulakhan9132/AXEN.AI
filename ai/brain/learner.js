// Train in a separate process so the chat never freezes.
const { fork } = require("child_process");
const path = require("path");
const axen = require("./axen");

let sleeping = false;
let queuedSteps = 0;

function startTraining(steps, onDone = () => {}) {
  sleeping = true;
  let child;
  let finished = false;

  const finish = (error) => {
    if (finished) return;
    finished = true;
    sleeping = false;
    try { axen.reload(); } catch (reloadError) { console.error("Could not reload learned model:", reloadError); }
    try { onDone(error); } catch (callbackError) { console.error("Training callback failed:", callbackError); }

    const nextSteps = queuedSteps;
    queuedSteps = 0;
    if (nextSteps > 0) startTraining(nextSteps);
  };

  try {
    child = fork(path.join(__dirname, "train.js"), [String(steps)], { silent: true });
    child.once("error", finish);
    child.once("exit", (code) => finish(code === 0 ? null : new Error(`Training exited with code ${code}`)));
  } catch (error) {
    finish(error);
  }
}

function sleep(steps = 500, onDone = () => {}) {
  if (sleeping) return false;
  startTraining(steps, onDone);
  return true;
}

// Positive feedback triggers a small background update. Feedback received
// during an active training run is queued and processed immediately afterward.
function learnFromFeedback(steps = 10) {
  if (sleeping) {
    queuedSteps += steps;
    return "queued";
  }
  startTraining(steps);
  return "started";
}

const isSleeping = () => sleeping;
module.exports = { sleep, isSleeping, learnFromFeedback };
