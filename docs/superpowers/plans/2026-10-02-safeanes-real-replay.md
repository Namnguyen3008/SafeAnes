# SafeAnes Real Replay and UC04/UC05 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add source-selectable synthetic/VitalDB monitoring with synchronized UC04/UC05 inference on an actual replay timeline.

**Architecture:** Keep the existing `RingBuffer` and monitor as the display path. Add a cached VitalDB case timeline and a replay worker that writes its real signals into the same monitor interface. A separate inference worker prepares the exact packaged model tensors at 30-second timestamps and reports per-model status/results to the on-monitor AI panel.

**Tech Stack:** Python 3.10+, PyQt6, pyqtgraph, NumPy, SciPy, optional `vitaldb` and PyTorch CPU runtime, pytest.

**Spec:** `docs/superpowers/specs/2026-10-02-safeanes-real-replay-design.md`

## Global Constraints

- At 1×, one case second advances per wall-clock second; pause holds case time and stops replay writes and inference scheduling.
- Monitor traces, recorded numerics, and predictions come from the same case and use the same case-time timestamp.
- Preserve the source model's exact tensor order, sampling rates, window lengths, masks, preprocessing, thresholds, and output labels.
- Missing waveforms are NaN and visibly marked `MISSING`, `INVALID`, or `NOT SUPPORTED`; never synthesize a VitalDB value.
- UC04 needs 300 seconds of numeric history and 30 seconds of waveform history. UC05 needs 600 seconds of numeric history and 60 seconds of waveform history. Both predict every 30 seconds.
- UC04/UC05 output is research-only and is not a clinical alarm or diagnosis.
- Model/runtime failures are isolated per model and never stop monitor replay.

---

### Task 1: Add case-timeline types and VitalDB case loading/cache

**Files:**
- Create: `core/data_sources.py`
- Create: `core/vitaldb.py`
- Test: `tests/test_vitaldb.py`
- Modify: `requirements-replay.txt`

**Interfaces:**
- `SignalStatus`: enum values `AVAILABLE`, `MISSING`, `INVALID`, `NOT_SUPPORTED`.
- `SignalInfo`: immutable track name, status, sample rate, and reason.
- `PatientTimeline`: immutable source/case metadata, wave arrays and rates, 1 Hz numeric arrays, static features, and a `window(channel, end_sec, seconds, target_hz)` operation that pads unavailable history with NaN.
- `VitalDBRepository.list_cases()` returns sorted metadata-only cases with the training-required tracks and duration eligibility; `load_case(case_id)` resolves training track candidates, fetches only that case's tracks, and uses a versioned compressed cache under application data.
- Optional dependencies are isolated in `requirements-replay.txt`; importing the simulator without them must remain valid.

- [ ] Write tests for case metadata filtering, exact UC04/UC05 track candidate resolution, no-waveform cache metadata, missing-track statuses, and NaN window padding; run `pytest tests/test_vitaldb.py -q` and confirm the expected import/attribute failures.
- [ ] Implement immutable timeline and signal status types in `core/data_sources.py`.
- [ ] Implement UC04 and UC05 track maps from the training notebooks, metadata catalogue filtering, selective `VitalFile` loading at training rates, cache read/write/version checks, and robust load errors in `core/vitaldb.py`.
- [ ] Add `vitaldb` and CPU-capable `torch` to the optional replay dependency manifest without adding either as an import-time requirement for Synthetic mode.
- [ ] Run `pytest tests/test_vitaldb.py -q` and confirm the new tests pass.

### Task 2: Add replay clock and ring-buffer producer

**Files:**
- Create: `core/replay.py`
- Test: `tests/test_replay.py`
- Reuse: `core/buffer.py`

**Interfaces:**
- `ReplayClock(duration_sec)` exposes `position_sec`, `playing`, `speed`, `play()`, `pause()`, `restart()`, `seek(position_sec)`, `set_speed(speed)`, and `advance(wall_seconds)`.
- `VitalDBReplayWorker(timeline, buffer)` owns a `ReplayClock`, fills six monitor channels at the monitor sample rate, emits position/end signals, and seeds the buffer after seek. `seek` preserves whether playback was running.
- Worker changes use a generation counter so data queued before a seek cannot advance the new position.

- [ ] Write deterministic tests proving 1×/2× clock advancement, pause stability, speed change, seek bounds, restart, end-of-case pause, and NaN for unavailable waveform rows; run `pytest tests/test_replay.py -q` and confirm these fail on the absent module.
- [ ] Implement the pure clock and waveform window interpolation from the existing case arrays.
- [ ] Implement a PyQt worker that writes 20–50 ms wall-time chunks into a six-channel `RingBuffer`, respecting replay speed and case end.
- [ ] Implement seek buffer reset and recent-history seeding without retaining samples from the previous seek generation.
- [ ] Run `pytest tests/test_replay.py -q` and confirm the new tests pass.

### Task 3: Reuse the existing bedside monitor for six signals and recorded numerics

**Files:**
- Modify: `ui/monitor.py` (`MonitorView.__init__`, `attach`, `refresh`, `update_traces`, and a new status setter)
- Create: `ui/numerics.py`
- Test: `tests/test_monitor_ui.py`

**Interfaces:**
- `MonitorView.set_channel_status(statuses)` updates the inline status text for ECG, Pleth, ART, RESP, CO₂, and AWP without creating a waveform for unavailable channels.
- `MonitorView.attach(buffer)` accepts either the existing four-channel Synthetic buffer or the six-channel replay buffer.
- `RecordedNumerics.set_values(values, case_time_sec, statuses)` displays HR, SpO₂, SBP/DBP/MAP, RR, and EtCO₂ from the case's recorded numeric tracks and shows `---` for missing or invalid values.

- [ ] Add monitor tests for four-channel Synthetic attachment, six-channel replay attachment, blank/status behavior for absent channels, and recorded numeric updates; run `pytest tests/test_monitor_ui.py -q` and confirm failures against current four-row monitor.
- [ ] Extend the existing monitor plot layout with CO₂ and AWP rows; preserve the existing HR/SpO₂/ART/RR tiles and Synthetic behavior.
- [ ] Add visible availability states beside waveform labels; absent channels remain NaN and do not render flat zero traces.
- [ ] Add the compact recorded numeric strip and timestamp, using the passed case-time values verbatim.
- [ ] Run focused monitor/UI tests and then existing `tests/test_sweep.py` to confirm producer/consumer compatibility.

### Task 4: Reproduce model contracts and run real per-model inference

**Files:**
- Create: `core/models/__init__.py`
- Create: `core/models/uc04_model.py` (matching architecture/loader source)
- Create: `core/models/uc05_model.py` (matching architecture/loader source)
- Create: `core/model_inference.py`
- Test: `tests/test_model_inference.py`
- Reuse artifacts: `../UC04/package_extracted/`, `../UC05/package_extracted/`

**Interfaces:**
- `model_readiness(model_key, timeline, case_time_sec)` returns a per-model state/reason and exact input window information.
- `prepare_uc04_input(timeline, case_time_sec)` produces the package's 8 tensors only when the 300-second numeric and 30-second waveform windows satisfy the notebook validity rules and ART/MAP requirements.
- `prepare_uc05_input(timeline, case_time_sec)` produces the package's 9 tensors only when the 600-second numeric window, 60-second waveform window, required modalities, SpO₂ history, and training inference eligibility rules are satisfied.
- `InferenceResult` records model key/version, case-time timestamp, raw output, processed score, defined threshold, input summary, status, and measured latency.
- `ModelRuntime.load()` loads each packaged directory independently and catches one model's failure without disabling the other.

- [ ] Add tests for exact array shapes/order, robust waveform normalization, missing masks, UC04's 300-second gate, UC05's 600-second gate, required modality checks, calibration, output horizon naming, and timestamp-preserving stale-generation rejection; run `pytest tests/test_model_inference.py -q` and confirm expected failures.
- [ ] Vendor the matching architecture/loader modules from the existing model source without changing model layer definitions or state-dict key names.
- [ ] Implement training-notebook preprocessing and model readiness checks using artifact schema/config/threshold/calibration metadata; never fill missing patient features with guessed values.
- [ ] Load `model.pt` and `respformer.pt` through strict state-dict loading, prepare tensors in a worker thread, and capture outputs/errors independently.
- [ ] Run focused tests; add a model-artifact smoke run that loads the two local packaged weights and checks a forward pass using training-contract tensor shapes.

### Task 5: Add source controls and SafeAnes AI display, then connect shared time

**Files:**
- Create: `ui/source_panel.py`
- Create: `ui/ai_panel.py`
- Modify: `ui/main_window.py` (`ControlPanel.__init__`, `MainWindow.__init__`, `bind`, `elapsed`, `signal_time`, `refresh_vitals`, `closeEvent`)
- Test: `tests/test_source_ui.py`

**Interfaces:**
- Source controls expose Synthetic/Real Replay, case-list refresh, case selection/load, play/pause/restart/seek, and speed selection.
- `SafeAnesAiPanel.update_state(model_key, state, reason, timestamp)` and `.update_result(result)` keep UC04 and UC05 states independent and show timestamped history.
- New `MainWindow` source handlers switch producers without changing the `MonitorView` instance. Synthetic-only patient edits, therapy, and scenarios are disabled during Real Replay; alarm settings remain enabled.
- Source header and clock display the case ID and replay position whenever Real Replay is selected.

- [ ] Write Qt tests for source selection, Synthetic-only control enablement, playback controls, per-model failure isolation, timestamp visibility, and result-history reset on seek; run `pytest tests/test_source_ui.py -q` and confirm expected missing-widget/signal failures.
- [ ] Add source and AI widgets, including the research-only note and optional developer detail view.
- [ ] Connect case loading/replay/inference signals to the existing MainWindow, route recorded numerics from the same `PatientTimeline`, and route both producers to the same monitor interface.
- [ ] On pause, stop replay and inference scheduling. On seek or source change, clear current predictions/history and reject stale worker results by generation id.
- [ ] Run the focused Qt tests and the existing simulator test suites with `QT_QPA_PLATFORM=offscreen`.

### Task 6: Prove the VitalDB end-to-end path and document setup

**Files:**
- Create: `tests/test_vitaldb_e2e.py` (opt-in live integration)
- Modify: `README.md`
- Verify: `docs/superpowers/specs/2026-10-02-safeanes-real-replay-design.md`

**Interfaces:**
- Live integration is opt-in through `SAFEANES_ENABLE_LIVE_INTEGRATION=1`; default unit tests never download VitalDB data.
- The live run loads VitalDB Case 1 (or a selected equivalent), confirms wave sample content and missing optional-channel status, advances to a model-eligible time, executes actual UC04/UC05 forward calls, verifies both results use the exact replay timestamp, then exercises pause/seek/resume/restart/speed on the replay clock.

- [ ] Write the opt-in live integration test and run it first to confirm its expected missing end-to-end behavior.
- [ ] Add source setup instructions for Synthetic and Real Replay, optional dependency installation, local case cache, playback controls, required input windows, and research-only model status.
- [ ] Run the existing full test suite and focused new tests in the project virtual environment with offscreen Qt.
- [ ] Run the opt-in live integration on VitalDB Case 1 with the local packaged model artifacts and record exactly which waveform and inference checks pass.
- [ ] Re-index the changed checkout with GitNexus, run `detect_changes`, inspect the affected flows, and report any unverified UI/production scope.
