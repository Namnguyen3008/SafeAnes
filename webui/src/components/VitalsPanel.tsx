import { useLayoutEffect, useRef, useState } from "react";
import type { CSSProperties } from "react";

import type { NumericValue, ReplayFrame, WaveformWindow } from "../types";
import { formatValue } from "../types";

interface MetricDefinition {
  name: string;
  shortName: string;
  channels: string[];
  keys: string[];
  color: string;
  units: string;
}

const METRICS: MetricDefinition[] = [
  { name: "Heart rate", shortName: "HR", channels: ["ecg"], keys: ["HR"], color: "#47e36e", units: "bpm" },
  { name: "Oxygen saturation", shortName: "SpO₂", channels: ["pleth", "ppg"], keys: ["SPO2"], color: "#31d7ec", units: "%" },
  { name: "Arterial pressure", shortName: "ART", channels: ["art", "abp"], keys: ["SBP", "DBP", "MAP"], color: "#ff5c72", units: "mmHg" },
  { name: "Respiratory rate", shortName: "RR", channels: ["resp"], keys: ["RR", "RR_CO2"], color: "#f5c84b", units: "/min" },
  { name: "End-tidal CO₂", shortName: "EtCO₂", channels: ["capno", "co2"], keys: ["ETCO2", "ETCO2_UC05"], color: "#57c7f3", units: "mmHg" },
  { name: "Airway pressure", shortName: "AWP", channels: ["awp"], keys: ["AWP"], color: "#d4a94b", units: "cmH₂O" },
];

// VitalDB numerics can be sparse on the 1 Hz replay grid. Hold the last
// measured value through its usual cadence, then let longer gaps show as missing.
interface RecentReading {
  value: number;
  sampleTimeSec: number;
  intervalsSec: number[];
}

interface ReadingHistory {
  replayId: string | null;
  frameTimeSec: number;
  values: Record<string, RecentReading>;
}

const DEFAULT_NUMERIC_HOLD_SECONDS = 2;
const MAX_NUMERIC_HOLD_SECONDS = 8;

function useDisplayNumerics(frame: ReplayFrame | null): Record<string, NumericValue> {
  const history = useRef<ReadingHistory>({ replayId: null, frameTimeSec: 0, values: {} });
  const [displayNumerics, setDisplayNumerics] = useState<Record<string, NumericValue>>({});

  useLayoutEffect(() => {
    if (!frame) {
      history.current = { replayId: null, frameTimeSec: 0, values: {} };
      setDisplayNumerics((current) => Object.keys(current).length > 0 ? {} : current);
      return;
    }

    const previous = history.current;
    const reset = previous.replayId !== frame.replay_id || frame.case_time_sec + 0.25 < previous.frameTimeSec;
    const recent: Record<string, RecentReading> = reset ? {} : { ...previous.values };
    const next = { ...frame.numerics };

    for (const [name, reading] of Object.entries(frame.numerics)) {
      const rate = Number.isFinite(reading.sample_rate_hz) && reading.sample_rate_hz > 0
        ? reading.sample_rate_hz
        : 1;
      const sampleTimeSec = Math.floor(frame.case_time_sec * rate + 1e-8) / rate;
      const last = recent[name];

      if (reading.status !== "AVAILABLE") {
        delete recent[name];
        continue;
      }
      if (reading.value !== null && Number.isFinite(reading.value)) {
        const intervalsSec = last && sampleTimeSec > last.sampleTimeSec + 1e-6
          ? [...last.intervalsSec, sampleTimeSec - last.sampleTimeSec].slice(-7)
          : last?.intervalsSec ?? [];
        recent[name] = { value: reading.value, sampleTimeSec, intervalsSec };
        continue;
      }
      if (!last) continue;

      const isPrimusTrack = reading.track_name?.toLowerCase().startsWith("primus/") ?? false;
      let holdLimitSec = isPrimusTrack ? 7 : DEFAULT_NUMERIC_HOLD_SECONDS / rate;
      if (last.intervalsSec.length > 0) {
        const sorted = [...last.intervalsSec].sort((left, right) => left - right);
        const middle = Math.floor(sorted.length / 2);
        const median = sorted.length % 2 === 0
          ? (sorted[middle - 1] + sorted[middle]) / 2
          : sorted[middle];
        holdLimitSec = Math.ceil(median);
      }
      holdLimitSec = Math.min(MAX_NUMERIC_HOLD_SECONDS, Math.max(1 / rate, holdLimitSec));

      if (sampleTimeSec - last.sampleTimeSec < holdLimitSec - 1e-6) {
        next[name] = { ...reading, value: last.value };
      } else {
        delete recent[name];
      }
    }

    history.current = {
      replayId: frame.replay_id,
      frameTimeSec: frame.case_time_sec,
      values: recent,
    };
    setDisplayNumerics((current) => {
      const unchanged = Object.keys(current).length === Object.keys(next).length
        && Object.entries(next).every(([name, value]) => {
          const before = current[name];
          return before?.value === value.value
            && before?.status === value.status
            && before?.units === value.units
            && before?.track_name === value.track_name;
        });
      return unchanged ? current : next;
    });
  }, [frame]);

  return displayNumerics;
}

function statusFor(values: Array<NumericValue | undefined>): string {
  const available = values.find((value) => value?.status === "AVAILABLE");
  return available?.status ?? values.find(Boolean)?.status ?? "NOT SUPPORTED";
}

function MetricValue({ metric, numerics, monitorTile = false }: {
  metric: MetricDefinition;
  numerics: Record<string, NumericValue>;
  monitorTile?: boolean;
}) {
  const values = metric.keys.map((key) => numerics[key]);
  const availableValues = values.filter((value): value is NumericValue => value?.status === "AVAILABLE");
  const firstValue = availableValues.find((value) => value.value !== null)?.value ?? null;
  const displayStatus = statusFor(values);
  let main = formatValue(firstValue);
  let secondary = metric.units;

  if (metric.name === "Arterial pressure") {
    const sbp = numerics.SBP?.value ?? null;
    const dbp = numerics.DBP?.value ?? null;
    const map = numerics.MAP?.value ?? null;
    main = sbp !== null && dbp !== null ? `${formatValue(sbp)} / ${formatValue(dbp)}` : "— / —";
    secondary = map !== null ? `MAP ${formatValue(map)} ${metric.units}` : "MAP unavailable";
    if (displayStatus === "AVAILABLE" && (sbp === null || dbp === null)) {
      secondary = "Check recorded pressure values";
    }
  } else if (metric.name === "Respiratory rate" && values[0]?.value === null) {
    main = formatValue(values[1]?.value ?? null);
  }

  return (
    <article className={`metric-card ${monitorTile ? "waveform-vital-tile" : ""}`} style={{ "--metric-color": metric.color } as CSSProperties}>
      <div className="metric-label"><span className="metric-symbol" />{monitorTile ? metric.shortName : metric.name}</div>
      <div className={`metric-number ${metric.name === "Arterial pressure" ? "metric-number-small" : ""}`}>{main}</div>
      <div className="metric-footer">
        <span>{secondary}</span>
        <span className={`mini-state ${displayStatus.toLowerCase().replaceAll(" ", "-")}`}>{displayStatus}</span>
      </div>
    </article>
  );
}

export function WaveformVitalTile({ signal, frame, color }: {
  signal: WaveformWindow;
  frame: ReplayFrame | null;
  color: string;
}) {
  const channel = signal.name.toLowerCase();
  const metric = METRICS.find((candidate) => candidate.channels.includes(channel));
  const numerics = useDisplayNumerics(frame);
  if (metric) return <MetricValue metric={metric} numerics={numerics} monitorTile />;

  const numeric = Object.values(numerics).find((value) => value.name.toLowerCase() === channel);
  const status = numeric?.status ?? (signal.status === "AVAILABLE" ? "NOT RECORDED" : signal.status);
  return (
    <article className="metric-card waveform-vital-tile" style={{ "--metric-color": color } as CSSProperties}>
      <div className="metric-label"><span className="metric-symbol" />{signal.name.toUpperCase()}</div>
      <div className="metric-number">{numeric?.status === "AVAILABLE" ? formatValue(numeric.value) : "—"}</div>
      <div className="metric-footer">
        <span>{numeric?.units || signal.units || "recorded"}</span>
        <span className={`mini-state ${status.toLowerCase().replaceAll(" ", "-")}`}>{status}</span>
      </div>
    </article>
  );
}

export default function VitalsPanel({ frame }: { frame: ReplayFrame | null }) {
  const numerics = frame?.numerics ?? {};
  const primaryKeys = new Set(METRICS.flatMap((metric) => metric.keys));
  const additional = Object.values(numerics).filter((item) => !primaryKeys.has(item.name));
  return (
    <section className="panel vitals-panel" aria-labelledby="vitals-heading">
      <div className="panel-heading compact-heading">
        <div>
          <div className="section-kicker">SUPPLEMENTAL VALUES</div>
          <h2 id="vitals-heading">Additional readings</h2>
        </div>
        <span className="time-chip">{frame ? "CASE TIME" : "NO CASE"}</span>
      </div>
      {additional.length > 0 && (
        <details className="additional-values">
          <summary>Additional recorded values <span>{additional.length}</span></summary>
          <div className="additional-grid">
            {additional.map((item) => (
              <div className="additional-value" key={item.name}>
                <span>{item.name.replaceAll("_UC05", " · UC05")}</span>
                <strong>{item.status === "AVAILABLE" ? formatValue(item.value) : "—"}</strong>
                <small>{item.value === null ? item.status : item.units || "recorded"}</small>
              </div>
            ))}
          </div>
        </details>
      )}
      {frame && additional.length === 0 && <p className="panel-empty-note">Primary values are shown beside their waveform.</p>}
      {!frame && <p className="panel-empty-note">Recorded values appear after a case is loaded.</p>}
    </section>
  );
}
