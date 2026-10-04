# SafeAnes WebUI

A local research monitor for replaying one de-identified VitalDB case through its original case-time axis. The browser receives bounded waveform windows and recorded values from FastAPI; UC04 and UC05 use the existing Python input preparation and inference code.

## Run locally on Windows

Use two PowerShell terminals from the `ecg-simulator` directory.

**Terminal 1 — API and model runtime**

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-replay.txt -r webui\requirements.txt
.\.venv\Scripts\python.exe -m uvicorn webui.backend.app:app --host 127.0.0.1 --port 8000
```

**Terminal 2 — WebUI**

```powershell
Set-Location webui
npm ci
npm run dev
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173). FastAPI's local API explorer is at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

Choose **VitalDB replay**, enter a public case ID (the workspace has cached case 1), and select **Load case**. **Find cases** lists and filters up to 500 qualifying adult VitalDB records; any public case ID can also be entered directly. Playback uses 1× by default; 2×, 5×, and 10× are available for research review. Seeking updates the waveform window and requests a prediction at the selected case-time stride. Pausing stops background frame and inference polling.

The waveform monitor spans the workspace, with a shared time ruler and each recorded value beside its matching trace. Supplemental vitals and model output follow beneath the waveform stack. The signal list scrolls independently; each waveform row keeps its trace and value together while scrolling. The monitor includes every waveform returned by the loaded timeline, with the original track, rendered sample rate, units, and `AVAILABLE`, `MISSING`, `INVALID`, or `NOT SUPPORTED` state. Replay traces show a 10-second window at up to 250 Hz, bounded by each source channel's actual rate. The monitor grid is enabled on load and can be toggled from the header. The monitor never fills missing signals with synthetic traces. Non-finite samples are sent as JSON `null`; inconsistent recorded systolic/diastolic/MAP values are suppressed by the existing sanitizer.

The WebUI loads patient case data only from VitalDB. It has no synthetic patient cases or fallback signals; missing recordings and unavailable inputs are shown explicitly. The separate desktop simulator retains its own synthetic scenarios and alarm workflows.

## Local data and model requirements

- The API binds only to `127.0.0.1`; Vite proxies `/api` to that loopback service.
- VitalDB case metadata and tracks may require internet access on first use. Replayed case timelines use the existing local cache under `%LOCALAPPDATA%\SafeAnes\cache\vitaldb`.
- UC04 and UC05 weights and wrappers are discovered from the workspace paths used by `core/model_inference.py`. `SAFEANES_UC04_ARTIFACT_DIR`, `SAFEANES_UC05_ARTIFACT_DIR`, and `SAFEANES_MODEL_SOURCE_DIR` can override them.
- The WebUI uses `SafeAnesInference.predict_all()` and the existing contract. UC04 waits for 300 seconds of numeric history; UC05 waits for 600 seconds and checks its required waveform/numeric inputs. Missing input and inference errors remain visible and do not fall back to a mock score.
- The browser is a research interface. Model outputs are not clinical alarms, diagnoses, or treatment advice.

## API overview

- `GET /api/health`
- `GET /api/cases?limit=500`
- `POST /api/replays`
- `GET /api/replays/{replay_id}/frame?window_seconds=10`
- `PATCH /api/replays/{replay_id}` with `play`, `pause`, `seek`, `speed`, or `restart`
- `POST /api/replays/{replay_id}/predictions`
- `DELETE /api/replays/{replay_id}`

The Python contract and service tests run with:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\webui -q
```

The frontend production build runs with:

```powershell
Set-Location webui
npm ci
npm run build
```

## Figma source

[SafeAnes Web Monitor](https://www.figma.com/design/gkAFjfHaDfvj7qcPMCDoPo) is the created design file. Its canvas is still blank: the Figma MCP rejected the design write because this account has reached its Starter-plan tool-call limit. No frame or node IDs were generated. The editable implementation is in `src/` while MCP canvas access is unavailable.
