import { useCallback, useEffect, useRef, useState } from "react";

import {
  controlReplay,
  deleteReplay,
  findCases as lookupCases,
  getFrame,
  loadReplay,
  requestPredictions,
} from "../../api/client";
import type { CaseSummary, ReplayAction, ReplayFrame, ReplaySession } from "../../types";

export function useReplay() {
  const [session, setSession] = useState<ReplaySession | null>(null);
  const [frame, setFrame] = useState<ReplayFrame | null>(null);
  const [cursorSec, setCursorSec] = useState(0);
  const [cases, setCases] = useState<CaseSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const sessionRef = useRef<ReplaySession | null>(null);
  const activeControllerRef = useRef<AbortController | null>(null);
  const generationRef = useRef(0);
  const lastStrideRef = useRef(-30);

  useEffect(() => {
    sessionRef.current = session;
  }, [session]);
  useEffect(() => {
    if (!frame) {
      setCursorSec(0);
      return;
    }
    if (!frame.playing) {
      setCursorSec(frame.case_time_sec);
      return;
    }
    const base = frame.case_time_sec;
    const startedAt = performance.now();
    const timer = window.setInterval(() => {
      const elapsed = (performance.now() - startedAt) / 1000;
      setCursorSec(Math.min(frame.duration_sec, base + elapsed * frame.speed));
    }, 1000 / 30);
    return () => window.clearInterval(timer);
  }, [frame?.case_time_sec, frame?.duration_sec, frame?.playing, frame?.speed]);

  const findCases = useCallback(async () => {
    setSearching(true);
    setError(null);
    try {
      const result = await lookupCases(500);
      setCases(result);
      if (result.length === 0) setError("VitalDB returned no cases matching the replay requirements.");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "VitalDB case search failed.");
    } finally {
      setSearching(false);
    }
  }, []);

  const loadCase = useCallback(async (rawCaseId: string) => {
    const caseId = Number(rawCaseId);
    if (!Number.isInteger(caseId) || caseId <= 0) {
      setError("Enter a positive VitalDB case ID.");
      return;
    }
    const generation = ++generationRef.current;
    activeControllerRef.current?.abort();
    const controller = new AbortController();
    activeControllerRef.current = controller;
    const previousId = sessionRef.current?.replay_id;
    setLoading(true);
    setError(null);
    try {
      const nextSession = await loadReplay(caseId, controller.signal);
      if (generation !== generationRef.current) {
        void deleteReplay(nextSession.replay_id).catch(() => undefined);
        return;
      }
      sessionRef.current = nextSession;
      setSession(nextSession);
      setFrame(null);
      setCursorSec(0);
      lastStrideRef.current = 0;
      const [initialFrame, initialPrediction] = await Promise.all([
        getFrame(nextSession.replay_id),
        requestPredictions(nextSession.replay_id, 0),
      ]);
      if (generation !== generationRef.current) return;
      setFrame({ ...initialFrame, prediction_history: initialPrediction.history });
      if (previousId && previousId !== nextSession.replay_id) {
        void deleteReplay(previousId).catch(() => undefined);
      }
    } catch (cause) {
      if (generation === generationRef.current) {
        setError(cause instanceof Error ? cause.message : "VitalDB case could not be loaded.");
        setSession(null);
        setFrame(null);
        sessionRef.current = null;
      }
    } finally {
      if (generation === generationRef.current) setLoading(false);
    }
  }, []);

  const control = useCallback(
    async (action: ReplayAction, extra: { position_sec?: number; speed?: number } = {}) => {
      const currentSession = sessionRef.current;
      if (!currentSession) return;
      setError(null);
      try {
        const nextState = await controlReplay(currentSession.replay_id, action, extra);
        let nextFrame = await getFrame(currentSession.replay_id);
        if (action === "seek" || action === "restart") {
          const target = action === "restart" ? 0 : (extra.position_sec ?? nextState.case_time_sec);
          const result = await requestPredictions(currentSession.replay_id, target);
          nextFrame = { ...nextFrame, prediction_history: result.history };
          lastStrideRef.current = Math.floor(target / 30) * 30;
        }
        setFrame(nextFrame);
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : "Replay control failed.");
      }
    },
    [],
  );

  useEffect(() => {
    if (!session || !frame?.playing) return;
    let cancelled = false;
    let busy = false;
    const tick = async () => {
      if (busy || cancelled) return;
      busy = true;
      try {
        const latest = await getFrame(session.replay_id);
        if (cancelled || sessionRef.current?.replay_id !== session.replay_id) return;
        setFrame(latest);
        if (latest.playing) {
          const stride = Math.floor(latest.case_time_sec / 30) * 30;
          if (stride >= 30 && stride > lastStrideRef.current) {
            lastStrideRef.current = stride;
            void requestPredictions(session.replay_id, stride)
              .then((result) => {
                if (cancelled || sessionRef.current?.replay_id !== session.replay_id) return;
                setFrame((current) => current
                  ? { ...current, prediction_history: result.history }
                  : current
                );
              })
              .catch((cause) => {
                if (!cancelled) setError(cause instanceof Error ? cause.message : "Prediction request failed.");
              });
          }
        }
      } catch (cause) {
        if (!cancelled) setError(cause instanceof Error ? cause.message : "Replay polling failed.");
      } finally {
        busy = false;
      }
    };
    const timer = window.setInterval(() => void tick(), 100);
    void tick();
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [session?.replay_id, frame?.playing]);

  useEffect(() => {
    return () => {
      activeControllerRef.current?.abort();
      const replayId = sessionRef.current?.replay_id;
      if (replayId) void deleteReplay(replayId).catch(() => undefined);
    };
  }, []);

  const pause = useCallback(() => control("pause"), [control]);
  const play = useCallback(() => control("play"), [control]);
  const seek = useCallback((position_sec: number) => control("seek", { position_sec }), [control]);
  const setSpeed = useCallback((speed: number) => control("speed", { speed }), [control]);
  const restart = useCallback(() => control("restart"), [control]);

  return {
    session,
    frame,
    cursorSec,
    cases,
    loading,
    searching,
    error,
    findCases,
    loadCase,
    play,
    pause,
    seek,
    setSpeed,
    restart,
    clearError: () => setError(null),
  };
}
