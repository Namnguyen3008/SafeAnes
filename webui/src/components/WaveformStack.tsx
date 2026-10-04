import type { ReplayFrame, WaveformWindow } from "../types";
import { WaveformVitalTile } from "./VitalsPanel";
import WaveformTrace from "./WaveformTrace";

const COLORS: Record<string, string> = {
  ecg: "#39ff14",
  ppg: "#22d3ee",
  pleth: "#22d3ee",
  art: "#ff4d5e",
  capno: "#22d3ee",
  awp: "#facc15",
  flow: "#ef9a4d",
  resp: "#facc15",
};

const LABELS: Record<string, string> = {
  ecg: "II",
  ppg: "PLETH",
  pleth: "PLETH",
  art: "ART",
  capno: "CO₂",
  awp: "AWP",
  flow: "Flow",
  resp: "RESP",
};

const SIGNAL_ORDER: Record<string, number> = {
  ecg: 0,
  pleth: 1,
  ppg: 1,
  art: 2,
  abp: 2,
  resp: 3,
  capno: 4,
  co2: 4,
  awp: 5,
  flow: 6,
};

function SignalRow({ signal, frame, gridEnabled, cursorSec }: {
  signal: WaveformWindow;
  frame: ReplayFrame | null;
  gridEnabled: boolean;
  cursorSec: number;
}) {
  const color = COLORS[signal.name.toLowerCase()] ?? "#a99cff";
  const label = LABELS[signal.name.toLowerCase()] ?? signal.name.toUpperCase();
  const statusClass = signal.status.toLowerCase().replaceAll(" ", "-");
  const isAvailable = signal.status === "AVAILABLE";
  return (
    <article className={`signal-row signal-row-${signal.name.toLowerCase()} ${isAvailable ? "" : "signal-row-unavailable"}`}>
      <div className="signal-plot-column">
        <div className="signal-row-meta" title={signal.track_name ?? (signal.units || undefined)}>
          <div className="signal-name-wrap">
            <span className="signal-name" style={{ color }}>{label}</span>
            {signal.units && <span className="signal-unit">{signal.units}</span>}
          </div>
          <span className={`state-tag ${statusClass}`}>
            <span className="state-dot" />
            {signal.status}
          </span>
          <span className="signal-rate" title="Rendered sample rate">
            {signal.display_rate_hz.toLocaleString()} Hz
          </span>
        </div>
        <WaveformTrace
          signal={signal}
          color={color}
          gridEnabled={gridEnabled}
          cursorSec={cursorSec}
          frameTimeSec={frame?.case_time_sec ?? cursorSec}
          emptyMessage={isAvailable ? undefined : signal.reason || `${signal.status} — no waveform samples in the source case`}
        />
      </div>
      <WaveformVitalTile signal={signal} frame={frame} color={color} />
    </article>
  );
}

function WaveformTimeRuler({ cursorSec }: { cursorSec: number }) {
  const sweepPhaseSec = ((cursorSec % 10) + 10) % 10;
  const sweepPhasePercent = (sweepPhaseSec / 10) * 100;
  return (
    <div className="waveform-time-ruler" aria-hidden="true">
      <div className="waveform-ruler-main">
        <span className="waveform-ruler-axis" />
        <div className="waveform-ruler-plot">
          <span className="ruler-start">0 s</span>
          <span className="ruler-middle">5 s</span>
          <span className="ruler-end">10 s</span>
          <span
            className="ruler-head"
            style={{
              left: `${sweepPhasePercent}%`,
              top: 0,
              bottom: 0,
              width: 1,
              backgroundColor: "#79d68d",
              boxShadow: "0 0 5px #79d68d",
            }}
          />
        </div>
      </div>
      <span className="waveform-ruler-value">READING</span>
    </div>
  );
}

export default function WaveformStack({ frame, gridEnabled, cursorSec }: {
  frame: ReplayFrame | null;
  gridEnabled: boolean;
  cursorSec: number;
}) {
  const signals = frame?.waveforms ?? [];
  const orderedSignals = [...signals].sort((left, right) => {
    const leftName = left.name.toLowerCase();
    const rightName = right.name.toLowerCase();
    return (SIGNAL_ORDER[leftName] ?? Number.MAX_SAFE_INTEGER)
      - (SIGNAL_ORDER[rightName] ?? Number.MAX_SAFE_INTEGER)
      || leftName.localeCompare(rightName);
  });
  const available = signals.filter((signal) => signal.status === "AVAILABLE").length;
  return (
    <section className="panel waveform-panel" id="signals" aria-labelledby="wave-heading">
      <div className="panel-heading waveform-heading">
        <div>
          <div className="section-kicker">PATIENT SIGNALS</div>
          <h2 id="wave-heading">Waveform timeline</h2>
          <p>Every recorded channel · synchronized to the case-time cursor</p>
        </div>
        <span className="channel-count">
          <span className="count-light" />
          {available} / {signals.length} available
        </span>
      </div>
      {!frame ? (
        <div className="wave-empty-state">
          <div className="empty-icon">⌁</div>
          <strong>Load a case to view patient signals</strong>
          <span>Every mapped VitalDB waveform will appear in this scrollable stack.</span>
        </div>
      ) : signals.length === 0 ? (
        <div className="wave-empty-state">
          <strong>No waveform channels were returned</strong>
          <span>The source response did not contain a waveform mapping.</span>
        </div>
      ) : (
        <>
          <WaveformTimeRuler cursorSec={cursorSec} />
          <div className="signal-scroll" tabIndex={0} aria-label="Scrollable waveform channels">
            {orderedSignals.map((signal) => (
              <SignalRow key={signal.name} signal={signal} frame={frame} gridEnabled={gridEnabled} cursorSec={cursorSec} />
            ))}
          </div>
        </>
      )}
      <footer className="waveform-footer">
        <span><kbd>Scroll</kbd> to inspect every channel</span>
        <span>{frame ? `${frame.waveforms.length} mapped channels` : "No case loaded"}</span>
      </footer>
    </section>
  );
}
