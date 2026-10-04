import { useEffect, useMemo, useState } from "react";

import type { NumericValue, ReplayFrame, WaveformWindow } from "../../types";

const SYNTHETIC_DURATION = 3600;
const SAMPLE_RATE = 50;
const SAMPLE_COUNT = 600;
const SYNTHETIC_RANGES: Record<string, [number, number]> = {
  ecg: [-0.25, 1.05],
  pleth: [0.25, 0.9],
  ppg: [0.25, 0.9],
  art: [0.15, 0.7],
  capno: [0, 0.85],
  awp: [0, 0.55],
  flow: [-0.5, 0.5],
  resp: [0.15, 0.7],
};

function createWaveform(name: string, timeSec: number): WaveformWindow {
  const ecgPeriod = 60 / 72;
  const values = Array.from({ length: SAMPLE_COUNT }, (_, index) => {
    const time = timeSec - 12 + index / SAMPLE_RATE;
    const phase = ((time % ecgPeriod) + ecgPeriod) % ecgPeriod / ecgPeriod;
    let value = 0;
    if (name === "ecg") {
      value = 0.025 * Math.sin(time * 31);
      if (phase > 0.16 && phase < 0.22) value += 0.1 * Math.sin(((phase - 0.16) / 0.06) * Math.PI);
      if (phase > 0.34 && phase < 0.37) value -= 0.16 * Math.sin(((phase - 0.34) / 0.03) * Math.PI);
      if (phase > 0.37 && phase < 0.41) value += 0.9 * Math.sin(((phase - 0.37) / 0.04) * Math.PI);
      if (phase > 0.54 && phase < 0.77) value += 0.17 * Math.sin(((phase - 0.54) / 0.23) * Math.PI);
    } else if (name === "pleth" || name === "ppg") {
      value = phase < 0.28 ? phase / 0.28 : 1 - ((phase - 0.28) / 0.72);
      value = 0.3 + Math.max(0, value) * 0.52 + 0.018 * Math.sin(time * 5);
    } else if (name === "art") {
      value = phase < 0.08 ? phase / 0.08 : Math.exp(-(phase - 0.08) * 3.3);
      value = 0.22 + value * 0.42;
    } else if (name === "capno") {
      const breathPhase = ((time % 4.7) + 4.7) % 4.7 / 4.7;
      value = breathPhase < 0.08 ? 0 : breathPhase < 0.18 ? (breathPhase - 0.08) * 8 : breathPhase < 0.78 ? 0.76 : Math.max(0, 1 - (breathPhase - 0.78) * 4.5);
    } else if (name === "awp") {
      value = Math.max(0, Math.sin(time * (2 * Math.PI / 4.7))) * 0.5;
    } else if (name === "flow") {
      value = Math.sin(time * (2 * Math.PI / 4.7)) * 0.44;
    } else {
      value = 0.42 + 0.2 * Math.sin(time * (2 * Math.PI / 4.7));
    }
    return value;
  });
  const unitByName: Record<string, string> = {
    ecg: "mV", ppg: "a.u.", pleth: "a.u.", art: "mmHg", capno: "mmHg",
    awp: "cmH₂O", flow: "L/min", resp: "a.u.",
  };
  return {
    name,
    sample_rate_hz: SAMPLE_RATE,
    display_rate_hz: SAMPLE_RATE,
    status: "AVAILABLE",
    units: unitByName[name] ?? "",
    track_name: "SYNTHETIC TEST SOURCE",
    reason: null,
    display_range: SYNTHETIC_RANGES[name] ?? [-1, 1],
    samples: values,
  } as WaveformWindow;
}

function numeric(name: string, value: number, units: string): NumericValue {
  return {
    name,
    sample_rate_hz: 1,
    status: "AVAILABLE",
    units,
    track_name: "SYNTHETIC TEST SOURCE",
    reason: null,
    value,
  };
}

function createFrame(timeSec: number, playing: boolean, speed: number): ReplayFrame {
  const drift = Math.sin(timeSec / 20);
  return {
    replay_id: "synthetic-preview",
    case_id: 0,
    source: "Synthetic test source · Baseline",
    case_time_sec: timeSec,
    duration_sec: SYNTHETIC_DURATION,
    playing,
    speed,
    waveforms: ["ecg", "pleth", "art", "capno", "awp", "flow", "resp"].map((name) => createWaveform(name, timeSec)),
    numerics: {
      HR: numeric("HR", Math.round(72 + drift * 2), "bpm"),
      SPO2: numeric("SPO2", 98, "%"),
      SBP: numeric("SBP", 120, "mmHg"),
      DBP: numeric("DBP", 72, "mmHg"),
      MAP: numeric("MAP", 88, "mmHg"),
      RR: numeric("RR", 13, "/min"),
      ETCO2: numeric("ETCO2", 35, "mmHg"),
      ETCO2_UC05: numeric("ETCO2_UC05", 35, "mmHg"),
    },
    prediction_history: [],
  };
}

export function useSynthetic() {
  const [cursor, setCursor] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);

  useEffect(() => {
    if (!playing) return;
    let last = performance.now();
    const timer = window.setInterval(() => {
      const now = performance.now();
      const elapsed = (now - last) / 1000;
      last = now;
      setCursor((current) => {
        return Math.min(SYNTHETIC_DURATION, current + elapsed * speed);
      });
    }, 100);
    return () => window.clearInterval(timer);
  }, [playing, speed]);

  useEffect(() => {
    if (cursor >= SYNTHETIC_DURATION) setPlaying(false);
  }, [cursor]);

  const frame = useMemo(() => createFrame(cursor, playing, speed), [cursor, playing, speed]);
  return {
    frame,
    cursorSec: cursor,
    play: () => setPlaying(true),
    pause: () => setPlaying(false),
    seek: (value: number) => {
      setCursor(Math.max(0, Math.min(SYNTHETIC_DURATION, value)));
      setPlaying(false);
    },
    setSpeed: (value: number) => setSpeed(value),
    restart: () => {
      setCursor(0);
      setPlaying(false);
    },
  };
}
