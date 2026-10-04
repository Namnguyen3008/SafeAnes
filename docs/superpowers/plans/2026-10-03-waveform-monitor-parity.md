# Waveform Monitor Parity Implementation Plan

> **For agentic workers:** Execute inline in the current task; do not delegate overlapping edits.

**Goal:** Make the WebUI waveform monitor follow the ECG Simulator's bedside layout and 10-second trace presentation while preserving VitalDB availability and case-time alignment.

**Architecture:** Keep replay and inference as the existing shared frame contract. Render each waveform beside its matched numeric tile, keep model inference in its own research panel, expose a grid toggle, and request a 10-second frame with a higher bounded display sample rate.

**Tech Stack:** React, TypeScript, Canvas, CSS, FastAPI, pytest.

**Spec:** `C:/Users/Namdr/.codex/attachments/5008fd3b-01bd-4006-9ced-99332746f89b/goal-objective.md`, sections 1, 7, 8, 10, and 15.

## Global Constraints

- Waveforms, numerics, and predictions remain tied to the same replay case-time cursor.
- Missing or unsupported channels remain explicitly identified; never fabricate waveform or numeric values.
- UC04/UC05 inference behavior and its timestamps remain unchanged.
- Synthetic mode continues using the same monitor components.
- Keep existing uncommitted repository work intact.

---

### Task 1: Match Replay Trace Window and Sampling

**Files:**
- Modify: `webui/backend/service.py`
- Modify: `webui/backend/app.py`
- Modify: `webui/src/api/client.ts`
- Test: `tests/webui/test_monitor_presentation.py`

**Interface:** Frame defaults to a 10-second window. Waveform display sampling is capped at 250 Hz, retaining each signal's source rate when it is lower. Numerics and predictions keep the existing single-position sampling.

- [x] Add a service test that seeks to 30 seconds and expects a 300 Hz ECG source to return `display_rate_hz == 250` and exactly 2500 samples by default.
- [x] Add an API test that omits `window_seconds` and expects a 4 Hz ECG source to return 40 samples.
- [x] Run those focused tests and confirm they fail on the old 12-second/50 Hz defaults.
- [x] Update the service, FastAPI route, and TypeScript client defaults together.
- [x] Run the focused API/service tests.

### Task 2: Pair Monitor Numerics with Waveform Rows

**Files:**
- Modify: `webui/src/App.tsx`
- Modify: `webui/src/components/WaveformStack.tsx`
- Modify: `webui/src/components/WaveformTrace.tsx`
- Modify: `webui/src/components/VitalsPanel.tsx`
- Modify: `webui/src/styles.css`

**Interface:** ECG/HR, PLETH/SpO₂, ART/BP, RESP/RR, CO₂/EtCO₂, and airway pressure use matched row tiles. The grid defaults off and is controlled by a real Grid toggle. Unmapped or unavailable readings remain visibly blank with status text. UC04/UC05 remain in their own panel.

- [x] Reuse the existing recorded-vitals formatting and status rules to render the matching metric inside each waveform row.
- [x] Keep supplemental recorded values available in a compact details panel without duplicating the primary row tiles.
- [x] Move channel labels into the plot area and retain availability/sample-rate details in a low-profile row header.
- [x] Pass one grid-enabled state from App through WaveformStack to each canvas; turn grid lines off by default.
- [x] Match the 10-second sweep labels and use row heights that adapt to viewport height; keep each metric attached while scrolling.
- [x] Build the WebUI and inspect the live replay/synthetic views at desktop and narrow widths.

### Task 3: Review Cross-Component Impact

**Files:**
- Review: frame contract consumers and replay control/inference views.

- [x] Confirm the WebUI sends a 10-second window on initial load and subsequent frame refreshes.
- [x] Confirm pause still freezes the shared frame position and all channel rows remain scrollable with their tiles.
- [x] Confirm inference panels retain existing timestamps and statuses.
- [x] Review `git diff` and keep changes scoped to monitor rendering and frame display defaults.
