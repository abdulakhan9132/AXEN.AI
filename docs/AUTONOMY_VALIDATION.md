# Autonomous learning validation — 2026-10-14

Validated on the existing Windows environment: Python 3.13.15,
PyTorch 2.6.0+cu124 and psutil 7.2.2. No production model, code upgrade or account
connection was activated.

## Test results

The focused evolution/learning/governance/chat/UI regression run passed **118
tests**, with no failures or warnings in the final run (84.76 seconds).
Machine-readable results: `outputs/autonomy-tests.xml`.

Coverage includes:

- Authorized, measured, prioritized and deduplicated goal generation.
- Worker lifecycle, resource pauses, bounded sessions, checkpoint recovery,
  failure handling, explicit retry and persistent emergency stop.
- Mock public research integration, HTTPS/domain/DNS validation, robots refusal,
  redirects, byte/request/source limits and source provenance.
- Authorized transcript intake, missing-caption limitations and connector
  consent, read-only permissions, revocation and secret-free audit records.
- Practice containment: import, attribute, file, arbitrary-call, numeric and
  execution-budget escape attempts rejected.
- Fixed practice/held-out evaluation, failed learning rejection, verified memory,
  future chat retrieval, example provenance and disjoint datasets.
- Actual persisted SSE events and event cursors.
- Code evolution baseline comparison and approval queue linkage, source-staleness
  checks, repository/policy boundaries, checkpoint-tampering rejection and rollback.
- Real small CPU neural training: changed candidate parameters, preserved base
  weights, saved checkpoint, `weights_only=True` reload with matching tensors,
  separate measured validation/held-out losses; no simulated result or activation.
- Existing controller, observations, store, models, meta/task agent, memory,
  conversation routing, model wiring and FastAPI UI behavior.
- Actual Chrome DOM tests of desktop and narrow dashboard controls, source links,
  escaped untrusted source text, pause/resume/emergency and layout; transport is
  explicitly mocked. Actual HTML/JS/CSS also passed the existing whole-app browser
  smoke. This is not a live authenticated-account browser test.

The initial browser harness had Brave startup issues and used Windows cp1252 to
read Chromium's UTF-8 DOM. Selecting installed Chrome, decoding UTF-8 explicitly
and controlling display scale corrected those test-environment failures.
Historical page-copy/branding assertions were updated to the already-present
AEXN branding and structural DOM checks.

An additional final boundary/cancellation/browser rerun follows the last
target-path restriction and is recorded in
`outputs/autonomy-final-boundary-tests.xml`: **36 passed** (43.87 seconds).
JavaScript syntax checks and `git diff --check` also passed.
The entire unrelated repository test suite was not run; the 118-test scope is
explicit in the XML file.

## Complete demonstrations

Public source:
`outputs/learning-demo-20261014-final-public/report.json`.

- Autonomous goal selected from actual failures in the fixed inclusive-sum
  exercise fixture.
- Retrieved approved official Python tutorial range material after robots checks.
- Recorded URL, UTC retrieval timestamp, content hash and untrusted status.
- Identified the exclusive stop concept and independently checked range semantics.
- Generated a bounded AST repair without access to held-out test cases.
- Executed the repair in the arithmetic capability interpreter.
- Practice score: **1/4 -> 4/4**; unseen score: **0/3 -> 3/3**.
- Persisted **one verified lesson**, demonstrated retrieval, and collected verified
  train/validation/held-out examples with provenance.
- Recorded all steps as actual worker events and ended the bounded session OFF.
- No production patch was justified by this exercise alone; no production code
  or weights were changed.

Explicit offline demonstration:
`outputs/learning-demo-20261014-bounded/report.json`.
It completed the same measured practice/verification/storage/retrieval cycle
using the clearly labeled operator-authored local range lesson.

The first default-limit attempt at
`outputs/learning-demo-20261014/report.json` correctly paused at high system RAM
usage and recorded **zero** completions/lessons. Successful demo commands used
explicit per-demo `--memory-busy-percent 99 --minimum-available-mb 64` bounds with
real psutil measurements. Default application limits remain 85% and 128 MB.

The CLI reports are deliberately in separate SQLite databases. They are evidence
artifacts; they do not silently seed the live user's memory. Enabling the actual
app worker uses its normal persistent SQLite memory and event stream.

## Limits of the evidence

The repaired fixture is not a baseline of a real LLM's general debugging ability.
This proves a bounded deterministic research-to-practice-to-memory pipeline and
future evidence retrieval, not general autonomous intelligence growth or a
production assistant accuracy gain. General curricula and semantic embeddings
remain future work.

Docker was absent. Missing-container refusal was tested; actual container
security isolation was not tested on this machine. Real connected accounts,
Windows vault secret provisioning, OAuth/caption downloads, media/speech
recognition and live-model checkpoint deployment were not exercised.

Additional other-agent modules and unrelated workspace modifications were
preserved; they are outside these validation claims.

## Root entry-point follow-up

After updating `main.py`, `run_ui.py`, `server.js` and `package.json`, **14 tests
passed** covering learning CLI dispatch, launcher working directory, explicit
offline/network demo selection, coding CLI/controller compatibility and import
safety (120.15 seconds). Node syntax, CLI help and diff whitespace checks passed.
The coding CLI was also updated to pass the current ModelRequest contract to
real AIModel implementations while preserving legacy injected models.
