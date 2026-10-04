# SafeAnes Local WebUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local React monitor that loads real VitalDB cases, shows all replay waveforms in a scrollable view, and displays real UC04/UC05 model outputs through FastAPI.

**Architecture:** A Vite-served React app proxies `/api` to a loopback-only FastAPI service. The service keeps each loaded `PatientTimeline` in Python memory, exposes bounded waveform windows and replay controls, and delegates inference to `SafeAnesInference`; existing signal preparation remains unchanged.

**Tech Stack:** React/React DOM 19.2.8, TypeScript, Vite 8.3.2, `@vitejs/plugin-react` 6.1.1, FastAPI 0.142.2, existing NumPy/VitalDB/PyTorch dependencies, pytest, and Playwright.

**Spec:** `docs/superpowers/specs/2026-10-03-safeanes-webui-design.md`

## Global Constraints

- The API binds only to `127.0.0.1`.
- Reuse `VitalDBSource.available_cases()` and `load_timeline()`, `PatientTimeline` window/numeric methods, `SafeAnesInference.predict_all()`, and `sanitize_recorded_vitals()`.
- Keep the loaded `PatientTimeline` in Python; only send a bounded time window to the browser.
- Render every signal in `PatientTimeline.waveforms` in an independently scrollable stack.
- UC04 needs at least 300 seconds of numeric history; UC05 needs at least 600 seconds of numeric history.
- Model errors and missing inputs remain explicit states; never fabricate a score or silently switch to synthetic data.
- Display model outputs as research-only, never as clinical alarms, diagnoses, or treatment advice.
- Keep Figma tokens and components editable; do not use a flattened full-screen screenshot as the design deliverable.

---

## File Structure

Create a single `webui/` source folder:

- `webui/package.json`, `webui/package-lock.json`: Vite scripts and locked React/tooling dependencies.
- `webui/index.html`, `webui/vite.config.ts`, `webui/tsconfig.json`: Vite entry and strict TypeScript settings; proxy `/api` to `127.0.0.1:8000`.
- `webui/src/main.tsx`, `webui/src/App.tsx`: React entry and responsive monitor shell.
- `webui/src/types.ts`, `webui/src/api/client.ts`: typed HTTP contracts and API functions.
- `webui/src/features/replay/useReplay.ts`: load/control/poll lifecycle and prediction cadence.
- `webui/src/components/ReplayControls.tsx`: case lookup/load, play/pause, seek, and speed.
- `webui/src/components/WaveformStack.tsx`, `WaveformTrace.tsx`: every signal row, bounded canvas drawing, status labels, and independent vertical scrolling.
- `webui/src/components/VitalsPanel.tsx`, `PredictionPanel.tsx`: recorded metrics and UC04/UC05 states.
- `webui/src/styles.css`: visual tokens, desktop/tablet layout, focus states, and reduced-motion rules.
- `webui/backend/__init__.py`, `app.py`, `schemas.py`, `service.py`: API factory/routes, Pydantic contracts, and per-process replay/prediction orchestration.
- `webui/requirements.txt`: FastAPI standard extra for the backend; existing replay requirements remain in the root dependency file.
- `webui/README.md`: Windows setup, start commands, URL, case-load flow, and limitations.
- `tests/webui/test_api.py`, `tests/webui/test_service.py`: deterministic API/service tests with fake VitalDB data and fake inference.

Create the Figma design file `SafeAnes Web Monitor` in the authenticated user's plan, then save its URL and node IDs in the implementation handoff/README. The user already supplied the vault path; the connected Figma MCP is authenticated, so no vault value is copied to app files or printed.

## Task 1: Build the Editable Figma Design

**Files:** Figma design file `SafeAnes Web Monitor`.

**Interfaces:** Create a desktop monitor frame (1440 × 900), a waiting/missing-input state, a ready UC04/UC05 state, and a tablet frame (1024 × 768). Components cover the application header, replay controls, signal row/status, vital tile, and model output card.

- [x] Create a Figma design file with the authenticated plan key `team::1686432664486521275`.
- [ ] Read file metadata and available libraries; search existing variables, text styles, and components before creating equivalents.
- [ ] If no suitable library exists, add reusable local components and named color/type/spacing/radius variables.
- [ ] Compose all four frames from editable layers and component instances; label example patient measurements and model outputs as illustrative.
- [ ] Inspect one full-frame render and returned node IDs; record the file URL and frame IDs for the code implementation.

**Verification:** The frames show the complete independently scrollable signal stack, model waiting and ready states, tablet reflow, and research-only copy. Text and controls are editable Figma layers.

## Task 2: Write API and Service Tests First

**Files:** Create `tests/webui/test_api.py` and `tests/webui/test_service.py`.

**Interfaces:** `create_app(source=None, inference=None, clock=None)` creates an injectable FastAPI app. Test doubles provide `available_cases(limit)`, `load_timeline(case_id)`, and `predict_all(timeline, end_sec)`.

- [x] Add fixtures with one `VitalDBCase`, a `PatientTimeline` containing multiple `TimelineSignal` statuses and a NaN sample, a fake inference runner, and a monotonic fake clock.
- [x] Add API tests for health, case listing, valid/unknown case load, replay frame content and payload bounds, invalid/expired replay IDs, play/pause/seek/speed, and deletion.
- [x] Add prediction tests for waiting-history statuses and ready scores/notes/thresholds.
- [ ] Add prediction tests for model missing-input and inference-error paths.
- [x] Add a serialization regression asserting every NumPy NaN becomes JSON `null` while signal status, rate, units, and reason are retained.
- [x] Run `python -m pytest tests/webui -q` and confirm the new tests fail because the WebUI API does not yet exist.

Representative contract assertion:

```python
def test_frame_contains_all_waveforms_and_json_null_for_nan(client, loaded_replay):
    response = client.get(f"/api/replays/{loaded_replay}/frame?window_seconds=12")
    assert response.status_code == 200
    body = response.json()
    assert {row["name"] for row in body["waveforms"]} == {"ecg", "pleth", "awp"}
    assert body["waveforms"][0]["samples"][-1] is None
```

## Task 3: Implement the FastAPI Bridge

**Files:** Create `webui/backend/__init__.py`, `app.py`, `schemas.py`, `service.py`, and `webui/requirements.txt`.

**Interfaces:** `ReplayService` exposes `available_cases(limit)`, `load_replay(case_id)`, `frame(replay_id, window_seconds)`, `control(replay_id, action, position_sec=None, speed=None)`, `predict(replay_id, end_sec)`, and `release(replay_id)`. `create_app()` injects a service and registers versioned `/api` endpoints.

- [x] Define response models for case summary, signal metadata/window, replay state/frame, and each `ModelPrediction` (`model_id`, status, timestamp, scores, thresholds, score kind, reason, notes).
- [x] Implement an in-memory replay registry keyed by an opaque ID; use `VitalDBSource` to find/load cases and retain timeline references in Python.
- [x] Implement a monotonic replay clock with bounded seek/speed validation and deterministic clock injection; frames call `TimelineSignal.window()`/`PatientTimeline.waveform_window()` for every mapped waveform and `numeric_history()` for recorded measurements.
- [x] Sanitize recorded arterial pressure with `sanitize_recorded_vitals()`; normalize non-finite samples/measurements to JSON null.
- [x] Implement prediction dispatch to `SafeAnesInference.predict_all()` without blocking the browser frame poll; deduplicate requests by the existing 30-second case-time stride.
- [x] Map invalid request, missing case/session, VitalDB access, and model failures to structured HTTP errors/statuses; do not switch to synthetic data.
- [x] Bind the app to `127.0.0.1` by default. Run the failing API/service tests, implement until they pass, then rerun `python -m pytest tests/webui -q`.

Backend run command:

```powershell
python -m pip install -r requirements-replay.txt -r webui/requirements.txt
python -m uvicorn webui.backend.app:app --host 127.0.0.1 --port 8000
```

## Task 4: Implement the React Monitor

**Files:** Create the files under `webui/` listed in File Structure.

**Interfaces:** `webui/src/api/client.ts` exports typed `health`, `findCases`, `loadReplay`, `getFrame`, `controlReplay`, `requestPredictions`, and `deleteReplay` functions. `useReplay()` returns current case, controls, frame, predictions, loading/error flags, and actions to the view components.

- [x] Add `package.json` scripts (`dev`, `build`, `preview`) and exact compatible dependency ranges; generate the lockfile with the installed Node 24.15.0.
- [x] Implement typed request/response schemas matching `webui/backend/schemas.py`; route `/api` through the Vite proxy.
- [x] Implement replay lifecycle: load the case, poll bounded frames while active, update displayed time locally, call inference once per 30-second stride, cancel stale requests on case change, and stop polling when paused/unmounted.
- [x] Implement canvas traces that decimate only for display, preserve NaN gaps, resize with the panel, and render every waveform row inside the dedicated vertical scroll container.
- [x] Implement replay controls, all recorded vital tiles, model result cards, and explicit waiting/missing/error states. Scores are rendered only when `status === READY`.
- [x] Add keyboard focus, text equivalents for color-coded signal states, responsive tablet reflow, and reduced-motion support.
- [x] Run `npm run build`; TypeScript and Vite production build pass. `npm ci` was not run because the active Vite server is using this install.

Frontend run command:

```powershell
Set-Location webui
npm ci
npm run dev -- --host 127.0.0.1
```

## Task 5: Add Local Run Guide and Browser Verification

**Files:** Create `webui/README.md`; adjust `tests/webui/test_api.py` or `test_service.py` only for observed defects.

- [x] Document the two local terminals, setup commands, browser URL `http://127.0.0.1:5173`, FastAPI URL `http://127.0.0.1:8000/docs`, VitalDB network/cache behavior, and the research-only limitation.
- [x] Start FastAPI and Vite and open the monitor in the browser.
- [x] Use Playwright to exercise case load, scroll to the last waveform, play/pause, seek, 2× speed changes, restart, and API error display. The case-results overlay interception found during this check is fixed and retested.
- [x] Load actual VitalDB cases and seek through 300/600 seconds and 18:30; confirm real UC04/UC05 waiting, missing-input, and ready outputs in the UI.
- [ ] If data cache, model packages, or network access prevents real replay, report that scope as unverified and preserve the exact error; do not replace it with a mock and call it an integrated pass.
- [ ] Run `git status --short` and `mcp__gitnexus__detect_changes({repo: "ecg-simulator", scope: "all"})`; review scope before reporting. Existing user-owned desktop changes are present, and GitNexus did not include the new untracked `webui/` tree in its changed-symbol analysis.

## Execution Notes

- Execute tasks inline because the user authorized this implementation and current instructions prohibit unrequested subagent delegation.
- Preserve pre-existing changes to the desktop simulator; do not stage or commit them as part of this WebUI task.
- After implementing the API, run focused API/service tests. After implementing the React view, run the production build and the browser flow; keep the final verification report scoped to the cases and environment actually exercised.
- Figma file creation succeeded, but editable frames/components and node IDs are still outstanding: Figma MCP `search_design_system`, `use_figma`, and the later metadata read all returned the Starter-plan tool-call limit. Do not describe the blank file as a completed design.
- Playwright reproduced the case-results panel intercepting Play/Restart. Results now render in normal layout flow and close after a case is selected; browser verification confirmed selection, 2× playback, pause freeze, seek, restart, and invalid-case error display.
- `npm ci` remains unrun because the active Vite process uses the installed `node_modules`; `npm run build` passed. The final workspace audit is also incomplete because unrelated user changes are already present and the GitNexus index omitted the untracked WebUI tree.
