# SafeAnes Local WebUI Design

**Status:** Architecture approved by the user; written-spec review pending.

## Goal

Add a polished local web monitor to the existing ECG simulator project. It must load real VitalDB replay cases, display every recorded waveform in a vertically scrollable view, and call the existing UC04/UC05 model runner through a local Python API.

## Approaches considered

1. **React + Vite + FastAPI (selected).** Gives the monitor a component-based, responsive front end while keeping VitalDB loading and model inference in the existing Python runtime. Vite proxies API requests to a loopback-only FastAPI process.
2. **FastAPI plus server-rendered HTML/vanilla JavaScript.** Fewer front-end dependencies, but the live waveform, replay controls, and multiple model states would become hand-built UI code.
3. **Next.js plus FastAPI.** Offers a full-stack React framework, but server rendering and a second application server do not help this local, data-heavy monitor.

The current machine has Node 24.15.0 and Python 3.12.4. Versions checked for this design are React/React DOM 19.2.8, Vite 8.3.2, `@vitejs/plugin-react` 6.1.1, and FastAPI 0.142.2. Vite 8 requires Node 20.19+ or 22.12+; the current Node runtime meets that requirement. Use a lockfile for the installed JavaScript dependency graph.

## Product experience

The primary screen is a desktop-first dark bedside monitor designed around a 1440 × 900 frame. A restrained navy shell surrounds a high-contrast waveform canvas; green, cyan, red, and amber accent colors identify signals, with text labels and status badges so meaning never depends on color alone.

- A compact header shows SafeAnes branding, local API/data status, and the research-only notice.
- A replay control rail lets the user find available cases, load a case by ID, play/pause, seek, and choose playback speed. Loading and errors are visible in the rail.
- The central waveform area renders every channel in `PatientTimeline.waveforms`. Each row has its signal name, units, sample rate, source/status, and its own readable scale. The stack owns a vertical scroll area so lower channels remain reachable without moving the global controls.
- A right-side measurement panel shows recorded HR, SpO₂, arterial pressure, respiratory rate, ETCO₂, and airway pressure when present. Missing values show an unavailable state; inconsistent arterial pressure continues through the existing sanitizer.
- A prediction panel shows separate UC04 and UC05 cards. It displays model scores only for ready predictions; waiting-for-history, missing-input, and inference-error states keep their reasons visible. A score is labeled as a model output with its horizon and source notes, never as a clinical alarm or recommendation.
- At tablet widths the control rail becomes a compact drawer and the measurement/prediction panel moves below the waveform stack. Waveforms retain an independent vertical scroll region.

Figma deliverables are editable design layers and reusable local components, not a flattened screenshot: (1) desktop replay with all signal rows and populated measurements, (2) waiting-for-history/missing-channel states, (3) ready UC04/UC05 results, and (4) tablet layout. Inspect available Figma libraries and existing design system before creating local tokens/components. If none are available, create a small token set for color, spacing, radius, and typography in the file.

## Runtime architecture

Create a self-contained `webui/` source folder:

- React/TypeScript/Vite owns navigation, replay controls, canvas waveform drawing, vitals, and prediction-state rendering.
- FastAPI owns the local API, replay clock/session, bounded waveform-window serialization, and calls into model inference.
- The API reuses `VitalDBSource.available_cases()` and `load_timeline()`, `PatientTimeline` window/numeric methods, `SafeAnesInference.predict_all()`, and `sanitize_recorded_vitals()`. It does not copy model feature preparation or infer from browser-generated data.
- The loaded `PatientTimeline` stays in the Python process. The API sends only signal metadata, the current bounded time window, numerics, status, and requested prediction results; it never serializes a full case timeline.
- Development requests go through Vite's `/api` proxy to FastAPI. FastAPI binds to `127.0.0.1`; no remote deployment or internet-facing API is in scope.

### API contract

- `GET /api/health` reports whether the local API is ready.
- `GET /api/cases?limit=...` returns matching de-identified case IDs and duration using the existing case discovery code.
- `POST /api/replays` accepts a case ID, loads its cached or public VitalDB timeline, and returns a replay ID, duration, and signal metadata/status.
- `GET /api/replays/{id}/frame?window_seconds=12` returns playback position/state, bounded samples for all mapped waveform channels, and the latest recorded numeric measurements. The query window and display sample rate are capped by the backend.
- `PATCH /api/replays/{id}` accepts play/pause, seek position, or playback speed changes.
- `POST /api/replays/{id}/predictions` evaluates the current case-time position and returns the existing UC04/UC05 prediction structures.
- `DELETE /api/replays/{id}` releases that replay session.

The browser polls frames at a modest interval and animates the displayed window locally. It asks for a new model prediction on the existing 30-second case-time stride. Backend request/response models validate payloads; NumPy non-finite values become JSON `null`. Unknown case/session, unavailable VitalDB, missing model packages, and malformed controls return explicit errors that the UI can show without substituting synthetic data.

## Prediction and signal states

Use the existing model input builders and inference results as the source of truth:

- UC04 remains `WAITING_FOR_HISTORY` until 300 seconds of numeric history are available; its required waveform validity gates can still produce a missing-input state.
- UC05 remains `WAITING_FOR_HISTORY` until 600 seconds of numeric history are available; required waveform validity gates can still produce a missing-input state. Unsupported optional tracks remain represented by their model-provided mask/notes.
- Model load/runtime failures are `ERROR`, never a fabricated zero score.
- Each signal separately reports available, missing, or not-supported status with units, sample rate, and reason where provided by the timeline.

All screens state: **Research only. Model scores are not clinical alarms, diagnoses, or treatment advice.**

## Failure handling and local safety

- The app starts without loading a case. The user explicitly searches/loads a case.
- VitalDB/network failures remain visible as retryable load errors. A cached timeline may load when already present, according to the existing cache behavior.
- API frame payloads are bounded and do not include patient names or the full de-identified metadata row.
- The API listens on loopback only and has no credential field. Figma credentials stay in the authenticated Figma connector; the vault file is not copied into the WebUI or returned by the API.
- Model scores carry the original score kind, thresholds, horizon, and notes from `ModelPrediction` so the interface does not relabel raw values as calibrated risk.

## Verification

Verification will include backend unit/API tests for case load, frame windows, playback controls, non-finite serialization, model waiting/error/ready states, and unknown cases; a front-end build/type check; and a Playwright browser flow through case load, scrolling the signal stack, play/pause/seek/speed controls, and prediction status updates.

The integrated local flow will then exercise an actual VitalDB cached/downloaded case and the local UC04/UC05 packages by seeking to at least 300 seconds and 600 seconds respectively and confirming the API/UI show the returned statuses/scores. This verifies the local environment only; it does not make the model clinically validated or verify any production deployment.

## Out of scope

- Changing the existing PyQt simulator or its replay/model feature preparation.
- Cloud hosting, remote multi-user access, authentication, database storage, and user accounts.
- Synthetic fallback data presented as a real VitalDB case.
- Clinical alarms, diagnoses, or treatment recommendations.
