# AXEN bounded autonomous learning — operating guide

The first complete curriculum is **Python debugging: inclusive range boundaries**.
Enabling a session automatically selects an authorized measured weakness,
retrieves approved documentation, analyzes and cross-checks a supported concept,
generates a bounded exercise repair, executes it in a capability-contained
interpreter, runs unseen tests, stores a verified lesson, and streams actual
events into the Evolution dashboard. No Evaluate clicks are required.

This is an incremental implementation, not a claim of general autonomous research
or general LLM skill acquisition. The Python arithmetic curriculum uses a
deterministic AST repair rule. Production changes and model checkpoint activation
remain separate human decisions.

## Diagnosis and architecture

See `docs/AUTONOMY_DIAGNOSIS.md` for the pre-change findings. The manual experiment
endpoint previously had neither a model nor an evaluator; no background worker
was wired to the application. Simulated training gains and host-side candidate
execution have been removed.

```text
Operator configuration + trusted controls
  -> SQLite observations / authorized goal planner / prioritized queue
  -> bounded single worker (FastAPI lifecycle; default OFF)
  -> DISCOVER -> RETRIEVE -> ANALYZE -> CROSS-CHECK -> ORGANIZE
  -> PRACTICE -> VERIFY -> STORE
  -> verified SQLite memory + provenance-bearing training examples
  -> future chat context retrieval
  -> optional isolated code candidates / staged neural checkpoints
  -> existing approval queue -> explicit human approval -> reviewed upgrade

Every step -> append-only SQLite events -> resumable SSE -> Evolution dashboard
```

`data/axen_learning.db` holds goals, step checkpoints, observations, events,
connector consent and the emergency latch. Verified lessons also go to the
existing `data/axen_memory.db`. Future assistant context retrieves them lexically
and wraps them as untrusted reference data with source and verification scope.
Memory retrieval does not change network parameters.

The queue resumes interrupted goals at their last completed step. It does not
restart after a reboot by default. A failed goal stays failed until the operator
retries it. Identical evidence is deduplicated; timings do not create new goals.
The configured session currently permits one completed/failed goal by default
and then turns OFF. Concurrency is always one worker/goal. This implementation
does not install an OS service or task scheduler entry.

Task failure, execution failure and low-confidence Python-task signals are
recorded without storing the user's message; they select the authorized
curriculum. Actual fixed benchmarks still establish the demonstrated weakness.

## Windows PowerShell installation and startup

Run from the existing repository. Do not replace your current checkpoints or
virtual environment.

```powershell
Set-Location 'C:\Users\HN089WS\OneDrive\Desktop\AEXN AI'
if (-not (Test-Path '.venv\Scripts\python.exe')) { py -3.13 -m venv .venv }
& .\.venv\Scripts\python.exe -m pip install fastapi uvicorn pydantic requests PyYAML psutil keyring pytest httpx
# For a new installation, install the project's remaining dependencies:
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
# A fresh environment also needs PyTorch. Preserve an existing CUDA build:
& .\.venv\Scripts\python.exe -c "import importlib.util,sys;sys.exit(0 if importlib.util.find_spec('torch') else 1)"
if ($LASTEXITCODE -ne 0) { & .\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu }
& .\.venv\Scripts\python.exe run_ui.py
```

The existing launcher uses Node.js for the companion backend; Node must be on
PATH. Open `http://127.0.0.1:8000`, select **Evolution**, choose **LEARNING**, then
**Start learning**. Leave the offline checkbox unchecked for public research.
Check it only to explicitly use the labeled local educational fixture.
The process and laptop must remain running. Closing the server stops learning.
The fresh-environment CPU installation follows the official
[PyTorch installation guidance](https://pytorch.org/get-started/locally/).

To run only the FastAPI application without launching the Node companion:

```powershell
& .\.venv\Scripts\python.exe -m uvicorn app.ui.app:app --host 127.0.0.1 --port 8000
```

The learning endpoint is part of the Python FastAPI app, not the independent
Node-only server. Use one Uvicorn process; multiple web workers are unsupported.
Do not expose operator endpoints to the internet. Their local same-origin
control-header protection is a CSRF boundary, not multi-user authentication.

The root CLI and npm scripts also expose the integrated system:

```powershell
& .\.venv\Scripts\python.exe main.py serve
# In another terminal after the app starts:
& .\.venv\Scripts\python.exe main.py autonomy status
& .\.venv\Scripts\python.exe main.py autonomy start --mode LEARNING
& .\.venv\Scripts\python.exe main.py autonomy pause
& .\.venv\Scripts\python.exe main.py autonomy resume
& .\.venv\Scripts\python.exe main.py autonomy emergency-stop
& .\.venv\Scripts\python.exe main.py autonomy-demo --offline
& .\.venv\Scripts\python.exe main.py autonomy-test
# npm alternatives:
npm start
npm run learning:status
npm run learning:demo
npm run learning:test
```

CLI controls contact the running app; they never create a second learning worker.

## Controls and live progress

The dashboard shows current goal/step, worker status, source links, research
findings, practice and unseen-test counts, memory retrieval, code candidates,
actual neural losses when configured, and the existing approval queue.
Its EventSource uses `/api/autonomy/stream`; reconnects use `Last-Event-ID`.
No activity or training progress is invented.

PowerShell equivalents:

```powershell
$axenControlHeaders = @{ 'X-AXEN-Control' = 'operator' }
Invoke-RestMethod 'http://127.0.0.1:8000/api/autonomy/control' -Method Post -Headers $axenControlHeaders -ContentType 'application/json' -Body '{"action":"start","mode":"LEARNING","offline_demo":false}'
Invoke-RestMethod 'http://127.0.0.1:8000/api/autonomy/status'
Invoke-RestMethod 'http://127.0.0.1:8000/api/autonomy/events?after=0'
Invoke-RestMethod 'http://127.0.0.1:8000/api/autonomy/control' -Method Post -Headers $axenControlHeaders -ContentType 'application/json' -Body '{"action":"pause"}'
Invoke-RestMethod 'http://127.0.0.1:8000/api/autonomy/control' -Method Post -Headers $axenControlHeaders -ContentType 'application/json' -Body '{"action":"resume"}'
Invoke-RestMethod 'http://127.0.0.1:8000/api/autonomy/control' -Method Post -Headers $axenControlHeaders -ContentType 'application/json' -Body '{"action":"stop"}'
```

Pause takes effect between bounded steps; an already-started HTTPS request has
an eight-second timeout. Stop/emergency cancellation is checked during research,
practice and training. Emergency stop prevents subsequent memory writes and
requires an explicit reset after the thread has exited.

```powershell
Invoke-RestMethod 'http://127.0.0.1:8000/api/autonomy/control' -Method Post -Headers $axenControlHeaders -ContentType 'application/json' -Body '{"action":"emergency-stop"}'
Invoke-RestMethod 'http://127.0.0.1:8000/api/autonomy/control' -Method Post -Headers $axenControlHeaders -ContentType 'application/json' -Body '{"action":"reset-emergency"}'
# Substitute an actual failed goal ID, after stopping and resolving the failure:
Invoke-RestMethod 'http://127.0.0.1:8000/api/autonomy/goals/ACTUAL_GOAL_ID/retry' -Method Post -Headers $axenControlHeaders
```

## Approved sources and budgets

Edit `configs/autonomy.json` as the operator, then restart the application.
The default approved source is the official Python tutorial's range section:
`https://docs.python.org/3/tutorial/controlflow.html#the-range-function`.

Add exact hostnames to `domains` and canonical HTTPS URLs with a topic to
`sources`. Both are required. Subdomains are not automatically authorized.
The source search currently ranks this operator-curated catalog; it does not
scrape a search engine or crawl GitHub/arXiv automatically. Retrieved commands,
credential requests and permission changes are never executed or obeyed.

Public research honors robots.txt, crawl delay, source/request limits, request
spacing, byte limits, access restrictions and timeouts. Redirects, authenticated
URLs, private/reserved DNS addresses and nontext downloads are refused. A robots
or source request failure produces a limitation rather than automatic bypass.
Only a bounded relevant excerpt is persisted with URL, retrieval timestamp and
content hash; it is an observation until cross-checked and applied successfully.

Initial limits: 30 minutes, one goal per session, five sources, two videos,
three code experiments, one training job, 15 public requests, two seconds between
requests. Real psutil CPU/RAM monitoring pauses sessions above 85% usage or below
128 MB available RAM. Close heavy applications before enabling learning.
Optional training additionally requires at least 1 GB available RAM and uses CPU
to preserve the laptop's small VRAM budget. Thresholds are operator-adjustable;
the worker cannot grant itself permissions.

Supported modes are OFF, LEARNING, DEVELOPMENT and SUPERVISED EVOLUTION. OFF
starts no autonomous activity. LEARNING stores verified lessons. DEVELOPMENT
also evaluates explicitly configured code targets. SUPERVISED EVOLUTION can run
explicitly configured candidate training. No mode deploys automatically.

## Educational videos and authorized accounts

There is no automatic account creation, cookie/password extraction, private
repository discovery, posting, commenting, messaging or purchasing.
GitHub public repository metadata and YouTube video metadata are the implemented
official API adapters. Other educational/cloud services are not implemented.
There is no OAuth login wizard in this version.

Provision narrowly scoped credentials directly into Windows Credential Manager
through a hidden-input CLI, then authorize the corresponding read capability:

```powershell
& .\.venv\Scripts\python.exe scripts\connector_credentials.py github
Invoke-RestMethod 'http://127.0.0.1:8000/api/autonomy/connectors/authorize' -Method Post -Headers $axenControlHeaders -ContentType 'application/json' -Body '{"service":"github","scopes":["public_read"],"consent":true}'
& .\.venv\Scripts\python.exe scripts\connector_credentials.py youtube
Invoke-RestMethod 'http://127.0.0.1:8000/api/autonomy/connectors/authorize' -Method Post -Headers $axenControlHeaders -ContentType 'application/json' -Body '{"service":"youtube","scopes":["metadata_read"],"consent":true}'
Invoke-RestMethod 'http://127.0.0.1:8000/api/autonomy/connectors'
Invoke-RestMethod 'http://127.0.0.1:8000/api/autonomy/connectors/github/revoke' -Method Post -Headers $axenControlHeaders
```

Use a GitHub fine-grained read-only token restricted to public resources and a
YouTube Data API key restricted to that API. Capability names above are AXEN's
local permissions, not OAuth scope strings. Keep credentials out of config,
prompts, chat and SQLite. OS vault availability is required; there is no plaintext
fallback. The APIs are bounded to 20 reads per service per process and at least
two seconds between reads. GitHub access is public metadata only.

For video learning, set `video_learning.enabled` to true. YouTube search currently
returns title, author and URL. Duration is recorded as unavailable by the search
adapter rather than invented. If you have an explicitly licensed/exported
transcript, put it under `knowledge/learning/videos/` and set
`video_learning.licensed_transcripts` to an object mapping video IDs to those
repository-relative paths. Only these configured transcripts are read. Missing
caption permission is reported and public documentation supplies the alternative.
No yt-dlp scraping, DRM bypass, automatic media download or speech recognition is
implemented. Summarizing a transcript alone cannot verify a skill.

YouTube caption downloads have separate authorization requirements; metadata
access does not grant them. See the official
[caption API](https://developers.google.com/youtube/v3/docs/captions/download).

## Containment and production evolution

The demonstrated exercise executes a small arithmetic Python subset through an
interpreter with no eval/exec, imports, strings, attributes, host objects, files,
network, arbitrary calls or recursion. AST size, integers, loop ranges, fuel and
elapsed time are bounded. It supports only one plain arithmetic function.
This is capability containment, not general Python execution.

For general Python snippets, `SandboxExecutor` requires a Linux Docker container
and refuses to fall back to the host. Docker was unavailable on the inspected
machine, so actual Docker isolation is not claimed as tested here. To install
and provision it explicitly:

```powershell
winget install --exact --id Docker.DockerDesktop
# Start Docker Desktop and select its Linux container engine before continuing.
docker pull python:3.13-slim
docker version
```

Containers run non-root with no network/host mounts, a read-only filesystem,
dropped capabilities, no-new-privileges, 512 MB RAM, 0.5 CPU and PID limits.
Timeout/cancellation removes the container. For stronger adversarial isolation,
use a dedicated VM. See [Docker's runtime controls](https://docs.docker.com/engine/containers/run/).

Production code evolution currently supports the same operator-defined
`inclusive_sum_nonnegative` contract only. An example `code_targets` entry is
`{"path":"path/to/inclusive_sum.py","contract":"inclusive_sum_nonnegative"}`.
It must point to an actual eligible production file implementing that contract.
In DEVELOPMENT mode, a reproducible failure is baselined, a repair is isolated,
train and held-out tests are compared, and an improvement is submitted to the
existing approval queue. With the default empty target list, the exercise repair
is recorded but no production improvement is claimed or submitted.

Candidate approval rejects paths outside the repository, protected policy/tests/
autonomous infrastructure, missing measured benchmarks and changes made after
the baseline. Backups go to `.backups/evolution`. To roll back an actually merged
experiment, substitute its real ID:

```powershell
Invoke-RestMethod 'http://127.0.0.1:8000/api/evolution/rollback/ACTUAL_EXPERIMENT_ID' -Method Post -Headers $axenControlHeaders
```

Rollback refuses to overwrite later user edits. A checkpoint activated to a new
path has no prior backup; remove that reviewed copy only after disabling it and
confirming its path, or restore an operator-held checkpoint. The default demo
needs no production rollback because it changes no production code or weights.
Verified memory can be deactivated through the existing Memory UI; preserve event
logs when removing a lesson.

## Neural candidate training

Verified practice collects provenance-bearing examples under
`data/training/verified_learning/`. Collection does not train a model. Dataset
curation deduplicates examples and rejects unverified data. Neural training
requires separate nonoverlapping train, validation and held-out splits, explicit
source/provenance, a real supplied model and optional replay examples.

To opt in, configure `neural_training.enabled=true`, `checkpoint` as an
operator-trusted file under `checkpoints/`, `dataset` as a JSON file under
`data/training/` (keys `train`, `validation`, `held_out`, optional `replay`), and
`max_steps` between 1 and 30. Set `dataset` to `verified_learning` to use the
current goal's verified arithmetic examples. Choose SUPERVISED EVOLUTION.
Do not load an untrusted downloaded checkpoint. Loading uses `weights_only=True`.

Candidate training deep-copies the base model, uses CPU and batch size one,
compares validation and held-out losses before/after, checks changed parameter
hashes, saves/reloads matching checkpoint tensors and records provenance.
Regression or unavailable dependencies produce failure, never simulated gains.
The production base remains unchanged. A measured improvement creates a separate
weight review request.

Before approving that request, set `neural_training.activation_target` to a
reviewed `.pt` path, for example `checkpoints/reviewed/approved_candidate.pt`.
Approval verifies the staged checkpoint's hash and copies it to that explicitly
configured path. It does not silently switch the live model router; selecting or
reloading the reviewed model remains an operator operation. This version does
not automate model deployment across Ollama providers.

## Demonstrations and validation

Separate demo databases keep demonstration evidence out of the assistant's live
memory. No network happens unless the explicit `--allow-network` flag is supplied:

```powershell
& .\.venv\Scripts\python.exe scripts\learning_demo.py --offline --output outputs\my-offline-demo
& .\.venv\Scripts\python.exe scripts\learning_demo.py --allow-network --output outputs\my-public-demo
& .\.venv\Scripts\python.exe -m pytest tests\test_autonomous_learning.py tests\test_autonomy_governance.py tests\test_neural_trainer_evolution.py tests\test_sandbox.py -q
```

The recorded public and offline demonstrations improved the fixed exercise from
1/4 to 4/4 practice tests and 0/3 to 3/3 unseen tests, persisted one verified
lesson and retrieved it. Their JSON reports disclose scope, resource limits and
source access. They used an explicit demo-only 99% RAM threshold with a 64 MB
available-memory reserve because this machine was above the default 85% limit.
The normal dashboard configuration remains unchanged.

See `docs/AUTONOMY_VALIDATION.md` for final results and `docs/AUTONOMY_FILES.md` for
the change manifest. Tests use mock research/API transports separately from the
actual public-document smoke. Browser tests use actual HTML/JS/CSS with mock
transport, not live account/model credentials.

Known limits: one narrow curriculum; no demonstrated general LLM reasoning gain;
no semantic embeddings; no unattended OS service; no arbitrary curriculum
generation; no OAuth wizard; no real account/caption smoke without credentials;
no real Docker smoke without Docker; no proof of full-app neural improvement.
The genuine tested achievement is a complete bounded learning cycle, persistence,
retrieval, measured practice, real events and governance around optional upgrades.
