import { useEffect, useMemo, useRef } from "react";

import type { WaveformWindow } from "../types";

const FALLBACK_RANGES: Record<string, [number, number]> = {
  ecg: [-1.2, 2.2],
  ppg: [-0.3, 1.6],
  pleth: [-0.3, 1.6],
  art: [0, 160],
  abp: [0, 160],
  capno: [0, 100],
  co2: [0, 100],
  awp: [-20, 120],
  flow: [-1, 1],
  resp: [-1.5, 1.5],
};

const TICK_STEPS: Record<string, number> = {
  ecg: 1,
  ppg: 0.5,
  pleth: 0.5,
  art: 50,
  abp: 50,
  capno: 20,
  co2: 20,
  awp: 20,
  flow: 1,
  resp: 1,
};

interface WaveformTraceProps {
  signal: WaveformWindow;
  color: string;
  gridEnabled: boolean;
  cursorSec: number;
  frameTimeSec: number;
  emptyMessage?: string;
}

function getPercentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  if (sorted.length === 1) return sorted[0];
  const idx = (sorted.length - 1) * p;
  const lower = Math.floor(idx);
  const upper = Math.ceil(idx);
  const weight = idx - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

function getRange(
  name: string,
  range: [number, number] | null | undefined,
  samples?: Array<number | null>,
): [number, number] {
  if (range && Number.isFinite(range[0]) && Number.isFinite(range[1]) && range[1] > range[0]) {
    return range;
  }
  const lname = name.toLowerCase();
  // Robust 1% - 99% percentile scaling matching desktop monitor (with 5% padding & non-negative filtering)
  if (samples && samples.length > 0) {
    let finite = samples.filter((s): s is number => s !== null && Number.isFinite(s));
    if (lname === "art" || lname === "abp" || lname === "co2" || lname === "capno") {
      finite = finite.filter((s) => s >= 0);
    }
    if (finite.length >= 4) {
      const sorted = [...finite].sort((a, b) => a - b);
      let low = getPercentile(sorted, 0.01);
      let high = getPercentile(sorted, 0.99);
      const span = high - low;
      if (span <= 1e-6) {
        const padding = Math.max(Math.abs((high + low) * 0.5) * 0.1, 1.0);
        low -= padding;
        high += padding;
      } else {
        const padding = Math.max(span * 0.05, 0.05);
        low -= padding;
        high += padding;
      }
      if (lname === "art" || lname === "abp" || lname === "co2" || lname === "capno") {
        low = Math.max(0, low);
      }
      if (high > low) {
        return [Number(low.toFixed(2)), Number(high.toFixed(2))];
      }
    }
  }
  return FALLBACK_RANGES[lname] ?? [-1, 1];
}

function niceStep(value: number): number {
  const exponent = Math.floor(Math.log10(value));
  const magnitude = 10 ** exponent;
  const normalized = value / magnitude;
  const step = normalized < 1.5 ? 1 : normalized < 3.5 ? 2 : normalized < 7.5 ? 5 : 10;
  return step * magnitude;
}

function getTicks(range: [number, number], name: string): number[] {
  const [low, high] = range;
  const span = high - low;
  const margin = span * 0.04;
  const preferred = TICK_STEPS[name.toLowerCase()] ?? 1;
  const step = span / preferred < 2 ? niceStep(span / 4) : preferred;
  const first = Math.ceil((low + margin) / step) * step;
  const ticks: number[] = [];
  for (let value = first; value < high - margin + 1e-9 && ticks.length < 8; value += step) {
    ticks.push(Number(value.toFixed(6)));
  }
  return ticks;
}

function formatTick(value: number): string {
  const rounded = Number(value.toFixed(2));
  return Object.is(rounded, -0) ? "0" : String(rounded);
}

export default function WaveformTrace({ signal, color, gridEnabled, cursorSec, frameTimeSec, emptyMessage }: WaveformTraceProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const isEcg = signal.name.toLowerCase().startsWith("ecg");
  const range = useMemo(
    () => getRange(signal.name, signal.display_range, signal.samples),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [signal.name, signal.display_range?.[0], signal.display_range?.[1], signal.samples],
  );
  const ticks = useMemo(() => getTicks(range, signal.name), [range, signal.name]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) return;
    const draw = () => {
      const bounds = canvas.getBoundingClientRect();
      if (bounds.width <= 0 || bounds.height <= 0) return;
      const ratio = Math.max(1, window.devicePixelRatio || 1);
      canvas.width = Math.round(bounds.width * ratio);
      canvas.height = Math.round(bounds.height * ratio);
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.clearRect(0, 0, bounds.width, bounds.height);

      if (gridEnabled && isEcg) {
        const minorGrid = isEcg ? "rgba(28, 69, 37, 0.34)" : "rgba(28, 42, 56, 0.36)";
        const majorGrid = isEcg ? "rgba(42, 96, 53, 0.58)" : "rgba(45, 65, 82, 0.62)";
        const xDivisions = Math.max(20, Math.floor(bounds.width / 7));
        const yDivisions = Math.max(8, Math.floor(bounds.height / 10));
        context.lineWidth = 1;
        for (let index = 0; index <= xDivisions; index += 1) {
          const x = (bounds.width * index) / xDivisions;
          context.strokeStyle = index % 5 === 0 ? majorGrid : minorGrid;
          context.beginPath();
          context.moveTo(x, 0);
          context.lineTo(x, bounds.height);
          context.stroke();
        }
        for (let index = 0; index <= yDivisions; index += 1) {
          const y = (bounds.height * index) / yDivisions;
          context.strokeStyle = index % 5 === 0 ? majorGrid : minorGrid;
          context.beginPath();
          context.moveTo(0, y);
          context.lineTo(bounds.width, y);
          context.stroke();
        }
      }

      const finite = signal.status === "AVAILABLE"
        ? signal.samples.filter((sample): sample is number => sample !== null && Number.isFinite(sample))
        : [];
      if (finite.length < 2) return;
      context.strokeStyle = color;
      context.lineWidth = isEcg ? 1.7 : 1.5;
      context.lineJoin = "miter";
      context.lineCap = "butt";
      context.shadowBlur = isEcg ? 2 : 0;
      context.shadowColor = color;
      context.beginPath();
      let penDown = false;
      const rate = signal.display_rate_hz;
      const windowSeconds = rate > 0 ? signal.samples.length / rate : 10;
      const wrapSeconds = (seconds: number) => ((seconds % windowSeconds) + windowSeconds) % windowSeconds;
      const sweepHeadSec = wrapSeconds(frameTimeSec);
      // Match the desktop's 80 ms erase bar and cover samples not yet returned for the interpolated cursor.
      const eraseGapSec = Math.min(windowSeconds, 0.08 + Math.max(0, cursorSec - frameTimeSec));
      let previousPhaseSec: number | null = null;
      signal.samples.forEach((sample, index) => {
        const sampleTimeSec = frameTimeSec - windowSeconds + index / rate;
        const phaseSec = wrapSeconds(sampleTimeSec);
        const timeAheadOfHeadSec = wrapSeconds(phaseSec - sweepHeadSec);
        const inEraseGap = timeAheadOfHeadSec < eraseGapSec;
        if (inEraseGap || sample === null || !Number.isFinite(sample)) {
          penDown = false;
          previousPhaseSec = phaseSec;
          return;
        }
        const x = (phaseSec / windowSeconds) * bounds.width;
        const y = bounds.height - ((sample - range[0]) / (range[1] - range[0])) * bounds.height;
        const crossedWrapPoint = previousPhaseSec !== null && phaseSec < previousPhaseSec;
        if (!penDown || crossedWrapPoint) {
          context.moveTo(x, y);
          penDown = true;
        } else {
          context.lineTo(x, y);
        }
        previousPhaseSec = phaseSec;
      });
      context.stroke();
      context.shadowBlur = 0;
    };
    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [signal.samples, signal.status, signal.display_rate_hz, color, range, isEcg, gridEnabled, cursorSec, frameTimeSec]);

  const hasSamples = signal.status === "AVAILABLE" && signal.samples.some((sample) => sample !== null && Number.isFinite(sample));
  return (
    <div className="trace-canvas-wrap">
      <div className="trace-y-axis" aria-hidden="true">
        {ticks.map((tick) => {
          const position = ((range[1] - tick) / (range[1] - range[0])) * 100;
          const edge = position < 5 ? "tick-top" : position > 95 ? "tick-bottom" : "";
          return <span className={`trace-y-tick ${edge}`} key={tick} style={{ top: `${position}%` }}>{formatTick(tick)}</span>;
        })}
      </div>
      <div className="trace-plot">
        <canvas
          ref={canvasRef}
          className="trace-canvas"
          role="img"
          aria-label={`${signal.name} waveform, ${signal.status.toLowerCase()}, ${signal.units || "no units"}`}
        />
        {!hasSamples && <div className="trace-empty">{emptyMessage ?? "No finite samples in this time window"}</div>}
      </div>
    </div>
  );
}
