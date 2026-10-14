# Phase 0: autonomy diagnosis (2026-10-14)

Inspected all existing `axen/evolution` modules, their API wiring, the UI,
SQLite memory, model interface, foundation trainer and evolution tests before
implementation. The repository already contains uncommitted evolution work.

| Capability | Actual implementation before this change |
| --- | --- |
| Experiment persistence | Working SQLite experiments, candidates, benchmarks, approvals and lineage. |
| Starting experiments | Manual POST `/api/evolution/start-experiment`; reads a file, constructs `CandidateGenerator()` without a model, and a controller without an evaluator. |
| Autonomous observation | Bounded callback-driven `run_observation_loop` exists, but no API/UI caller, scheduler, persistent queue, or background worker invokes it. |
| Candidate generation | Can accept a model, but the API does not supply one. Offline fallback adds a comment to unchanged source. `ModelRequest(prompt=...)` does not match the actual context-based interface. Missing `uuid` import on one path. |
| Weakness detection | Parses supplied pytest output and benchmark metrics; does not collect these itself. Counts failures, rather than confirming repeatability across runs. |
| Code search / selection | Baseline and held-out partitioning exist. Controller rejects absent evaluations. Evaluator executes candidate source with `exec` in the application process, even when a sandbox is supplied. |
| Task/meta agents | Optional model/loop wrappers. Missing TaskAgent model returns `Processed: ...` and can report success. MetaAgent fallback projected gains are hypotheses, not measurements. |
| Research / video / connectors | Absent. No source provenance, allowlist, authorized captions or account credential storage. |
| Learning memory | General persistent memory/retrieval works; no verified learning workflow or goal-linked provenance schema. |
| Neural training | FoundationTrainer really calls backward/optimizer.step and saves checkpoints. Evolution wrapper trains the supplied base model in place, compares validation loss with training loss, invents a gain and checkpoint path when model is absent, and can proceed without validation samples. Existing test expects that invented gain. |
| Containment | Subprocess timeout and proxy variables only. Memory limit and direct socket blocking are not enforced. Not a security sandbox. Docker is not installed in the inspected environment. |
| Approval | Pending requests and backups exist. Merge path lacks repository containment, protected-path enforcement, current-source checks and evaluated-candidate checks. Reviewer names are labels, not authentication. |
| Dashboard | Fetches experiment/approval counts on refresh; no live worker activity or lifecycle controls. |
| Startup | FastAPI lifespan initializes chat service; no learning worker startup/shutdown hooks. |

Implementation direction: persistent bounded worker, default OFF, evidence-based
goal selection, approved-source research, capability-restricted arithmetic Python
practice interpreter (no host execution), fail-closed container execution for
general code, verified lesson storage and future context retrieval, real event
stream, and human-only production upgrades. Unsupported services, unavailable
models/containers and failed benchmarks must be visible as limitations.
