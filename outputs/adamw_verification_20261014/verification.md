# AEXN AI verification — 2026-10-14

The existing application now has one AEXN AI chat, one composer and automatic internal routing. Implementation and focused regressions passed. Full live-server verification remains incomplete; do not interpret these results as production readiness.

## Repairs and behavior

- Removed visible model/reasoning selectors and the duplicate coding composer. Workspace moves the same chat/composer into its project layout.
- Connected chat to the shared backend model manager and provider adapter. Internal routing chooses conversation, reasoning, coding or the existing literary checkpoint; unavailable local models fail explicitly. Cloud keys alone never authorize cloud requests.
- Reused Think Engine, coding context/search/patch/test tools, image pipeline, uploads and history. Public progress messages contain no private reasoning.
- Coding requests prepare bounded multi-file plans and diffs. Approval is required to apply edits, run tests or restore files; stale previews are rejected. Original content is retained for Undo.
- Added approved Git checkpoint creation and previewed source rollback without resetting the user's Git index. Source snapshots are application-session scoped; newly created files survive restore.
- Untrusted tests require an existing local Docker sandbox image. They receive a disposable source copy, no original workspace mount and no network. Execution is blocked when the sandbox is unavailable.
- Corrected image Markdown rendering, escaped untrusted HTML, added code highlighting and retained copy controls.
- Repaired internal-directory exposure and workspace containment checks. Files and directories such as internal agent metadata and checkpoints are excluded from generated-edit targets.
- Refined light/dark styling and responsive workspace layout. Browser testing found mobile navigation covering chat; mobile navigation now starts closed.
- Existing reverse-engineering routes are proxied to their existing Node implementation, rather than duplicated.
- `npm start` now launches the Python application and existing Node companion through the project virtual environment; `npm run start:node` preserves standalone Node startup.

## Actual checks

| Check | Result | Scope |
| --- | --- | --- |
| JavaScript syntax | Passed | app.js, markdown.js, server.js |
| Python syntax | 15 files passed | Changed application/training modules |
| Focused Python regression suite | **115 passed in 81.35 seconds** | Unified ASGI routes, routing, Think Engine, coding, AdamW, training and BPE |
| Markdown/rendering Node tests | **6 passed** | Image cards, escaping, links, code and diff rendering |
| Brave desktop frontend fixture | **14 checks passed** | Actual frontend DOM with mocked API responses |
| Brave 390px mobile frontend fixture | **15 checks passed** | Same events plus closed navigation; no overflow or JS errors |
| Approved project test execution | Passed | Actual pytest subprocess on an isolated temporary project, through the ASGI route |
| Local image generation | Passed | Cached SD-Turbo pipeline, RTX 3050, four inference steps |
| Git diff check | Passed | No whitespace errors |

Python command: `.venv/Scripts/python.exe -m pytest tests/test_unified_aexn.py tests/test_ui_model_wiring.py tests/test_routing.py tests/test_think_engine.py tests/test_coding_agent.py tests/test_aexn_adamw.py tests/test_aexn_gpt_training.py tests/test_bpe_tokenizer.py -q --tb=short` with writable TEMP/TMP and a unique basetemp. Machine-readable results are in `regression_results.xml` beside this report.

ASGI tests run the real application handlers without TCP sockets; their coding model responses are controlled fixtures. Native browser fixtures use mocked fetch responses, not a running backend. Tests isolate new conversation/memory stores. Earlier broad checks touched existing conversation/memory/event stores; these were not reset because they already contained user modifications.

Initial retry without explicit writable TEMP failed on Windows sandbox temp-folder permissions. Setting TEMP/TMP to the permitted local temporary directory resolved those failures. Headless Brave failed during Crashpad startup; an already running native Brave browser was used for the frontend checks instead.

## AdamW experiment completed earlier in this task

The training system continues to use official PyTorch AdamW. Parameters remain configurable: lr 0.0003, betas 0.9/0.999, epsilon 1e-8, decay 0.01, clip 1.0, warmup 50, optional AMSGrad (default false). Biases and one-dimensional normalization parameters are excluded from decay. AMP unscales before clipping; skipped overflow updates do not advance the scheduler. Checkpoints restore model, optimizer, scheduler, scaler, step, hyperparameters and RNG state.

Short Shakespeare BPE experiment: 60 steps followed by a checkpoint resume for 10 more, 1,327,872 parameters, RTX 3050 CUDA AMP. Existing active checkpoints were not overwritten.

| Metric | Actual value |
| --- | --- |
| Initial evaluated training loss | 7.7811 |
| First / final training batch loss | 7.8136 / 6.6917 |
| Final validation loss | 6.6043 |
| Final validation perplexity | 738.26 |
| Learning rate at steps 1 / 50 / 60 / 70 | 0.000006 / 0.000300 / 0.00018612 / 0.00003166 |
| Pre-clipping gradient norm range | 0.902853–1.540227 |
| Post-clipping limit | 1.0 |
| Nonfinite/skipped updates | 0 |
| Peak allocated GPU memory | 65.36 MB |

Seven short tuning trials of 16 steps compared learning rates, beta values, weight decay, architecture and batch size. The 0.0008 learning-rate trial had the lowest short-run validation loss (6.9598). This is preliminary evidence, not a recommendation to replace the requested default. Full records: `initial_training_summary.json`, `training_summary.json`, `tuning_results.json`, `tuning_results_best.json`. Generated text remains poor; reduced Shakespeare loss does not establish conversational or coding quality.

## Files changed or added by this work

Application: `app/ui/app.py`, `app/ui/backend_service.py`, `app/ui/templates/index.html`, `app/ui/static/js/app.js`, `app/ui/static/js/markdown.js`, `app/ui/static/css/styles.css`, `app/ui/static/css/coding-workspace.css`.

Routing: `app/ai/model_manager.py`, `app/ai/router.py`, `app/ai/local_model.py`, `app/ai/aexn_gpt_provider.py`, `app/ai/chat_adapter.py`, `axen/agent/classifier.py`, `axen/agent/loop.py`, `axen/model/hf_model.py`.

Coding: `axen/coding/coding_service.py`, `axen/coding/coding_agent.py`, `axen/coding/workspace.py`, `axen/coding/command_runner.py`, `axen/coding/test_runner.py`.

Training: `axen/model/foundation.py`, `configs/aexn_gpt.yaml`, `scripts/train_aexn_gpt.py`, `scripts/tune_aexn_gpt.py`.

Launch/documentation: `run_ui.py`, `main.py`, `server.js`, `package.json`, `README.md`.

New tests: `tests/test_aexn_adamw.py`, `tests/test_ui_model_wiring.py`, `tests/test_unified_aexn.py`, `tests/test_ui_browser.py`, `tests/ui-markdown.test.js`. Some project files were already modified or untracked before work began; their pre-existing changes were preserved.

## Still unverified or limited

- Starting the real server was attempted. This managed environment blocks loopback socket operations; the native browser reported connection refused at port 8000. Live Send-to-model-to-browser behavior and live reverse-engineering proxy traffic therefore remain unverified.
- Real coding generation through a running configured Ollama provider was not verified here. Proposal/edit/test integration was verified with controlled model output.
- SSE progress is live, but response text is emitted after model inference completes. Native model-token streaming remains to be implemented.
- Docker is absent here. Refusal without a sandbox is tested; successful container execution is not.
- Existing document upload tools are reused; full PDF attachment-to-answer browser testing was not completed.
- The complete repository test suite was not certified. The reported 115 checks are the named focused suite.

No existing model, dataset, tokenizer, training checkpoint or feature was deleted. Experiment checkpoints and the generated image are separate output artifacts.
