import type {
  CaseSummary,
  PredictionResult,
  ReplayAction,
  ReplayFrame,
  ReplaySession,
} from "../types";

interface ApiErrorBody {
  detail?: { message?: string; code?: string } | string;
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`/api${path}`, {
    ...init,
    headers: {
      Accept: "application/json",
      ...(init.body ? { "Content-Type": "application/json" } : {}),
      ...init.headers,
    },
  });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const body = (await response.json()) as ApiErrorBody;
      if (typeof body.detail === "string") message = body.detail;
      else if (body.detail?.message) message = body.detail.message;
      else if (body.detail?.code) message = body.detail.code;
    } catch {
      // The status line is enough when the server did not return JSON.
    }
    throw new Error(message);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export const checkHealth = () => request<{ status: "ok"; service: string }>("/health");

export async function findCases(limit = 500, signal?: AbortSignal): Promise<CaseSummary[]> {
  const result = await request<{ cases: CaseSummary[] }>(`/cases?limit=${limit}`, { signal });
  return result.cases;
}

export function loadReplay(caseId: number, signal?: AbortSignal): Promise<ReplaySession> {
  return request<ReplaySession>("/replays", {
    method: "POST",
    body: JSON.stringify({ case_id: caseId }),
    signal,
  });
}

export function getFrame(replayId: string, windowSeconds = 10): Promise<ReplayFrame> {
  return request<ReplayFrame>(
    `/replays/${encodeURIComponent(replayId)}/frame?window_seconds=${windowSeconds}`,
  );
}

export function controlReplay(
  replayId: string,
  action: ReplayAction,
  extra: { position_sec?: number; speed?: number } = {},
): Promise<{
  replay_id: string;
  case_id: number;
  case_time_sec: number;
  duration_sec: number;
  playing: boolean;
  speed: number;
}> {
  return request(`/replays/${encodeURIComponent(replayId)}`, {
    method: "PATCH",
    body: JSON.stringify({ action, ...extra }),
  });
}

export function requestPredictions(
  replayId: string,
  caseTimeSec: number,
): Promise<PredictionResult> {
  return request<PredictionResult>(`/replays/${encodeURIComponent(replayId)}/predictions`, {
    method: "POST",
    body: JSON.stringify({ case_time_sec: caseTimeSec }),
  });
}

export function deleteReplay(replayId: string): Promise<void> {
  return request<void>(`/replays/${encodeURIComponent(replayId)}`, { method: "DELETE" });
}
