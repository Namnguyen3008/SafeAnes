# SafeAnes Real Replay and UC04/UC05 Design

## Goal

Extend the existing ECG Simulator into a research bedside monitor that can replay one de-identified VitalDB case at real time and feed the same case timeline into the available UC04 and UC05 packages. Synthetic monitoring stays available through the existing monitor.

## Architecture

Keep the existing `RingBuffer` and `MonitorView` as the display path. The synthetic `SignalGenerator` remains the default producer. A VitalDB `ReplayWorker` writes resampled samples from one loaded case into a six-channel monitor buffer (`ecg`, `pleth`, `abp`, `resp`, `co2`, `awp`). Missing channels contain NaNs and carry an explicit status; they are never filled with a synthetic or default waveform.

Represent the loaded patient data as one immutable case timeline. It owns the case identifier, case duration, selected track names, waveforms at the rates used by the training pipelines, 1 Hz numeric histories, static case features, signal availability, and a generation identifier. The monitor, numeric display, and inference worker read from that same record and receive the same case-time timestamp. The synthetic source has an adapter to the same frame/status representation but does not produce AI output from synthetic data.

The source controls use a clock based on `time.monotonic()`. At 1× one wall-clock second advances one case second. Pause holds the case time and stops buffer writes and inference scheduling. Seek resets the buffer, vitals/alarm history, prediction history, and inference generation before replay continues from the requested case time. Speeds are 1×, 2×, 5×, and 10×.

VitalDB case metadata and one selected case are cached under the user's application-data directory. Only the requested case tracks are loaded. The cache is keyed by case id and a schema version so incompatible cached layouts are rejected and refetched.

## UC04 contract

Use the packaged `model.pt`, `config.json`, and `thresholds.json` with the matching `HypoFormer` loader/architecture from the model source. Reproduce the training notebook contract:

- Wave history: 30 seconds; ART 100 Hz (3,000 values), ECG 250 Hz (7,500), PPG 100 Hz (3,000), capnogram 62.5 Hz (1,875).
- Numeric history: five minutes at 1 Hz with ordered fields `MAP, SBP, DBP, HR, SPO2, RR, ETCO2` and a finite-value mask.
- Static fields: `age, bmi, asa, emop, sex_male`; unknown fields stay NaN and use the model's built-in missing mask.
- Training preprocessing: short in-range interpolation for numeric histories; each waveform rejects out-of-range samples, requires at least 90% valid ART or 80% valid optional wave data, interpolates remaining invalid values, and robust-scales by median and `max(IQR/1.349, std, 1e-3)`, clipped to [-8, 8].
- Outputs: raw logits for 5, 10, and 15 minute IOH horizons; display sigmoid scores. The saved 5 minute threshold is 0.5. The model target is MAP ≤65 mmHg sustained for at least 60 seconds.
- Inference cadence is 30 seconds. A result is timestamped at the end of its input window.

The package config marks this model as a smoke test with streaming validation disabled. The UI will identify it as research output and will not present its score as a clinical alarm.

## UC05 contract

Use packaged `respformer.pt`, `config.json`, `signal_schema.json`, `calibration_thresholds_metrics.json`, and `MODEL_CARD.md` with the matching `RespFormer` loader/architecture from the model source. Reproduce the training notebook contract:

- Wave history: 60 seconds; capnogram and airway pressure 62.5 Hz (3,750 values), PPG 125 Hz (7,500); Flow and Resp are optional 62.5 Hz channels.
- Numeric history: ten minutes at 1 Hz with the exact 18-field order in `signal_schema.json`, plus the finite-value mask.
- Static fields: `age, bmi, asa, sex_male, weight, height, emop`; unknown fields stay NaN and use the model's built-in missing mask.
- Training preprocessing: polyphase waveform resampling to the package rates, plausible-range filtering for capnogram/airway pressure, 80% minimum valid waveform fraction, linear interpolation of remaining invalid values, robust median/IQR scaling with the standard-deviation floor, and clipping to [-8, 8].
- Wait for the full ten-minute numeric history. Require capnogram, airway pressure, and PPG per the package schema; optional Flow and Resp remain masked when absent. Require at least 80% valid historical SpO2 and do not emit an eligible hypoxemia prediction when the current SpO2 is already below 90% or the training recovery-exclusion interval applies.
- Outputs: calibrated logits for 1, 3, and 5 minute hypoxemia heads using the package temperatures; only the 5 minute output has the packaged threshold. The separate ventilation-pattern output is labeled weak and signal-derived, per the model card.
- Inference cadence is 30 seconds. A result is timestamped at the end of its input window.

The displayed model state is per model. Missing artifacts, missing PyTorch, incomplete input windows, missing required signals, and inference exceptions remain distinct states and never create placeholder scores. UC05's saved validation/test event counts are small and its test record detected 0 of 1 test events; all outputs remain research-only.

## User experience

Add a Data Source tab for Synthetic and Real Replay, case selection/loading, play/pause/restart/seek/speed controls, case duration, current case time, and cache-aware status. Header text always identifies `SOURCE: Synthetic` or `SOURCE: VitalDB Real Replay · CASE <id>`.

The monitor has ECG, Pleth, ART, RESP, CO₂, and airway-pressure rows. Every channel shows `AVAILABLE`, `MISSING`, `INVALID`, or `NOT SUPPORTED`. Real monitor numerics come from the selected case's recorded numeric tracks at the current case time; the existing waveform-derived measurements remain for Synthetic mode. A new SafeAnes AI panel shows the actual available model output, threshold when defined, exact update time, state/reason, and a history trace. Research and developer detail labels do not imply a diagnosis or validated clinical risk.

Patient-editing, therapy, and scenario controls remain available in Synthetic mode and are disabled in Real Replay so replayed patient data cannot be altered by simulator controls. Alarm configuration remains usable in both modes.

## Demo case evidence

The VitalDB public metadata API currently lists Case 1 with a duration of 11,542 seconds and all mapped required tracks for both model packages: ECG II, ART, PLETH, Primus CO₂, Primus AWP, the UC04 MAP/pressure/HR/SpO₂/RR/EtCO₂ tracks, and all UC05 numeric feature track mappings. UC05 Flow and Resp tracks are absent and are optional. This is metadata-only confirmation; the waveform values and model forward passes still require runtime verification before describing Case 1 as end-to-end verified.

## Verification target

Automated tests cover the canonical timeline, status behavior, exact input-window shapes and preprocessing, real-time clock/pause/seek/speed behavior, stale-result rejection, and UI source state. A separate opt-in live integration run loads the selected VitalDB case, checks its waveform arrays, advances replay, prepares both model windows, executes real inference, and confirms both prediction timestamps equal the monitor case time. The full existing simulator test suite must still pass.
