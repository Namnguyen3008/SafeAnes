import { useEffect, useRef, useState } from "react";

import type { CaseSummary, ReplayFrame, ReplaySession } from "../types";
import { formatTime } from "../types";

interface PlaybackActions {
  play: () => void;
  pause: () => void;
  seek: (position: number) => void;
  setSpeed: (speed: number) => void;
  restart: () => void;
}

interface ReplayControlsProps {
  session: ReplaySession | null;
  frame: ReplayFrame | null;
  cursorSec: number;
  cases: CaseSummary[];
  searching: boolean;
  loading: boolean;
  error: string | null;
  onFindCases: () => void;
  onLoadCase: (caseId: string) => void;
  actions: PlaybackActions;
  apiHealthy: boolean;
}

function SourceIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 12h4l2-5 4 11 2-6h6" /></svg>
  );
}

export default function ReplayControls(props: ReplayControlsProps) {
  const [caseInput, setCaseInput] = useState("1");
  const [caseFilter, setCaseFilter] = useState("");
  const [draftSeek, setDraftSeek] = useState(0);
  const [caseResultsOpen, setCaseResultsOpen] = useState(false);
  const seekRef = useRef<number | null>(null);
  const draggingSeekRef = useRef(false);
  const frame = props.frame;
  const hasFrame = Boolean(frame);
  const duration = frame?.duration_sec ?? props.session?.duration_sec ?? 0;
  const cursor = props.cursorSec;
  const playing = frame?.playing ?? false;
  const speed = frame?.speed ?? 1;
  const realLoaded = Boolean(props.session);
  const filteredCases = props.cases.filter((item) => String(item.case_id).includes(caseFilter.trim()));

  const commitSeek = () => {
    const position = seekRef.current;
    seekRef.current = null;
    draggingSeekRef.current = false;
    if (position !== null) props.actions.seek(position);
  };

  useEffect(() => {
    if (!draggingSeekRef.current) setDraftSeek(cursor);
  }, [cursor]);

  return (
    <section className="panel control-panel" aria-label="Replay controls">
      <div className="source-mode-row">
        <div>
          <div className="section-kicker">DATA SOURCE</div>
          <div className="source-fixed"><SourceIcon /><strong>VitalDB replay</strong></div>
        </div>
        <div className={"api-state " + (props.apiHealthy ? "connected" : "disconnected")}>
          <span className="api-state-dot" />
          {props.apiHealthy ? "Local API connected" : "Local API offline"}
        </div>
      </div>

      <div className="control-divider" />

      <div className="case-load-row">
        <div className="case-input-wrap">
          <label htmlFor="case-id">VitalDB case ID</label>
          <div className="case-input-line">
            <input
              id="case-id"
              inputMode="numeric"
              value={caseInput}
              onChange={(event) => setCaseInput(event.target.value.replace(/\D/g, ""))}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  setCaseResultsOpen(false);
                  props.onLoadCase(caseInput);
                }
              }}
              placeholder="Enter a public case ID"
              aria-describedby="case-source-hint"
            />
            <span className="input-suffix">ID</span>
          </div>
          <small id="case-source-hint">De-identified public VitalDB data</small>
        </div>
        <button
          className="button button-secondary"
          onClick={() => {
            const open = !caseResultsOpen;
            setCaseResultsOpen(open);
            if (open) {
              setCaseFilter("");
              props.onFindCases();
            }
          }}
          disabled={props.searching || !props.apiHealthy}
        >
          <span className="button-search-icon">⌕</span>{props.searching ? "Searching…" : "Find cases"}
        </button>
        <button
          className="button button-primary"
          onClick={() => {
            setCaseResultsOpen(false);
            props.onLoadCase(caseInput);
          }}
          disabled={props.loading || !props.apiHealthy}
        >
          {props.loading ? <span className="button-spinner" /> : <span>↧</span>}
          {props.loading ? "Loading case…" : "Load case"}
        </button>
      </div>

      {caseResultsOpen && props.cases.length > 0 && (
        <div className="case-results" aria-label="VitalDB public index candidates">
          <div className="case-results-label">Showing {props.cases.length} VitalDB cases</div>
          <label className="case-filter">
            <span>Filter by case ID</span>
            <input
              type="search"
              inputMode="numeric"
              aria-label="Filter VitalDB cases by ID"
              placeholder="Enter an ID to filter"
              value={caseFilter}
              onChange={(event) => setCaseFilter(event.target.value.replace(/\D/g, ""))}
            />
          </label>
          <div className="case-results-hint" aria-live="polite">
            {filteredCases.length} match{filteredCases.length === 1 ? "" : "es"} · signal quality is checked after load.
          </div>
          <div className="case-results-scroll">
            {filteredCases.length > 0 ? filteredCases.map((item) => (
              <button
                key={item.case_id}
                className={"case-result " + (caseInput === String(item.case_id) ? "selected" : "")}
                onClick={() => {
                  setCaseInput(String(item.case_id));
                  setCaseResultsOpen(false);
                }}
              >
                <span>Case {item.case_id}</span>
                <small>{formatTime(item.duration_sec)}</small>
              </button>
            )) : (
              <div className="case-results-empty">No listed case IDs match this filter.</div>
            )}
          </div>
          <div className="case-results-hint">Enter any public VitalDB case ID above to load it directly.</div>
        </div>
      )}

      {props.error && (
        <div className="inline-error" role="alert">
          <span className="error-symbol">!</span>
          <div><strong>Replay request failed</strong><p>{props.error}</p></div>
        </div>
      )}

      <div className="transport-row">
        <div className="transport-buttons">
          <button className="icon-button" onClick={props.actions.restart} disabled={!hasFrame} aria-label="Restart replay" title="Restart">
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 12a8 8 0 1 0 2.35-5.65L4 8.7M4 4v4.7h4.7" /></svg>
          </button>
          <button
            className={"play-button " + (playing ? "is-playing" : "")}
            onClick={playing ? props.actions.pause : props.actions.play}
            disabled={!hasFrame}
            aria-label={playing ? "Pause replay" : "Play replay"}
          >
            {playing ? <span className="pause-glyph" /> : <span className="play-glyph" />}
          </button>
          <div className="transport-time">
            <strong>{formatTime(cursor)}</strong>
            <small>CASE TIME / {formatTime(duration)}</small>
          </div>
        </div>

        <div className="seek-control">
          <div className="seek-labels"><span>00:00:00</span><span>{formatTime(duration)}</span></div>
          <input
            type="range"
            min={0}
            max={Math.max(1, duration)}
            step={1}
            value={draggingSeekRef.current ? draftSeek : cursor}
            disabled={!hasFrame}
            aria-label="Seek in case timeline"
            onPointerDown={() => { draggingSeekRef.current = true; }}
            onKeyDown={(event) => {
              if (event.key.startsWith("Arrow") || event.key === "Home" || event.key === "End") {
                draggingSeekRef.current = true;
              }
            }}
            onChange={(event) => {
              const value = Number(event.target.value);
              draggingSeekRef.current = true;
              seekRef.current = value;
              setDraftSeek(value);
            }}
            onPointerUp={commitSeek}
            onKeyUp={commitSeek}
            onBlur={commitSeek}
          />
        </div>

        <div className="speed-control">
          <label htmlFor="speed-select">SPEED</label>
          <select
            id="speed-select"
            value={speed}
            disabled={!hasFrame}
            onChange={(event) => props.actions.setSpeed(Number(event.target.value))}
          >
            {[0.5, 1, 2, 5, 10].map((value) => <option key={value} value={value}>{value}×</option>)}
          </select>
        </div>
      </div>
      {!realLoaded && !props.loading && (
        <div className="control-hint"><span>i</span> Search up to 500 cases or enter any public VitalDB case ID, then load it.</div>
      )}
    </section>
  );
}
