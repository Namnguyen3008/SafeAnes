export type SignalStatus = "AVAILABLE" | "MISSING" | "INVALID" | "NOT SUPPORTED";
export type ModelStatus =
  | "READY"
  | "WAITING FOR HISTORY"
  | "MISSING REQUIRED INPUT"
  | "NOT ELIGIBLE"
  | "ERROR";

export interface SignalInfo {
  name: string;
  sample_rate_hz: number;
  status: SignalStatus;
  units: string;
  track_name: string | null;
  reason: string | null;
}

export interface NumericValue extends SignalInfo {
  value: number | null;
}

export interface WaveformWindow extends SignalInfo {
  display_rate_hz: number;
  display_range?: [number, number] | null;
  samples: Array<number | null>;
}

export interface ModelPrediction {
  model_id: string;
  status: ModelStatus;
  timestamp_sec: number;
  scores: Record<string, number>;
  thresholds: Record<string, number>;
  score_kind: string;
  reason: string;
  notes: string[];
}

export interface PredictionPoint {
  timestamp_sec: number;
  models: ModelPrediction[];
}

export interface ReplaySession {
  replay_id: string;
  case_id: number;
  source: string;
  duration_sec: number;
  case_time_sec: number;
  playing: boolean;
  speed: number;
  waveforms: SignalInfo[];
  numerics: SignalInfo[];
}

export interface ReplayFrame {
  replay_id: string;
  case_id: number;
  source: string;
  case_time_sec: number;
  duration_sec: number;
  playing: boolean;
  speed: number;
  waveforms: WaveformWindow[];
  numerics: Record<string, NumericValue>;
  prediction_history: PredictionPoint[];
}

export interface CaseSummary {
  case_id: number;
  duration_sec: number;
  source: string;
}

export interface PredictionResult {
  replay_id: string;
  case_id: number;
  timestamp_sec: number;
  models: ModelPrediction[];
  history: PredictionPoint[];
}

export type ReplayAction = "play" | "pause" | "seek" | "speed" | "restart";
export function formatTime(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const remainder = total % 60;
  return [hours, minutes, remainder].map((part) => String(part).padStart(2, "0")).join(":");
}

export function formatValue(value: number | null, digits = 0): string {
  if (value === null || !Number.isFinite(value)) return "—";
  return value.toFixed(digits);
}
