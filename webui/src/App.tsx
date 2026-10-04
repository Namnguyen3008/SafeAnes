import { useEffect, useState } from "react";

import { checkHealth } from "./api/client";
import PredictionPanel from "./components/PredictionPanel";
import ReplayControls from "./components/ReplayControls";
import VitalsPanel from "./components/VitalsPanel";
import WaveformStack from "./components/WaveformStack";
import { useReplay } from "./features/replay/useReplay";

import type { ReplayFrame } from "./types";
import { formatTime } from "./types";

function BrandMark() {
  return (
    <div className="brand-mark" aria-hidden="true">
      <svg viewBox="0 0 28 28"><path d="M4 15h5l2.4-5.7 4.5 10.1 2.8-6.1H24" /></svg>
    </div>
  );
}

function Header({ frame, apiHealthy, gridEnabled, onToggleGrid }: {
  frame: ReplayFrame | null;
  apiHealthy: boolean;
  gridEnabled: boolean;
  onToggleGrid: () => void;
}) {
  return (
    <header className="topbar">
      <div className="topbar-title">
        <span className="breadcrumb">SAFEANES <i>/</i> MONITOR</span>
        <span className="topbar-case">{frame ? "CASE " + frame.case_id : "NO CASE LOADED"}</span>
      </div>
      <div className="topbar-right">
        <div className={"top-status " + (apiHealthy ? "status-live" : "status-offline")}>
          <span />{apiHealthy ? "LOCAL API" : "API OFFLINE"}
        </div>
        <div className="source-badge source-real">
          <span className="source-badge-icon">V</span>
          <span>VitalDB</span>
        </div>
        <div className="time-live"><span className="time-live-dot" />{frame ? formatTime(frame.case_time_sec) : "--:--:--"}</div>
        <button
          className={"icon-button grid-toggle " + (gridEnabled ? "grid-enabled" : "")}
          aria-label="Toggle ECG paper grid"
          aria-pressed={gridEnabled}
          title={gridEnabled ? "Hide ECG paper grid" : "Show ECG paper grid"}
          onClick={onToggleGrid}
        >
          <svg viewBox="0 0 24 24" aria-hidden="true"><rect x="4" y="4" width="16" height="16" rx="1" /><path d="M4 10.7h16M4 17.3h16M10.7 4v16M17.3 4v16" /></svg>
          <span>Grid</span>
        </button>
      </div>
    </header>
  );
}

function Sidebar() {
  return (
    <aside className="sidebar">
      <a className="brand" href="#monitor" aria-label="SafeAnes monitor home">
        <BrandMark />
        <span className="brand-copy"><strong>SafeAnes</strong><small>RESEARCH MONITOR</small></span>
      </a>
      <div className="sidebar-section-label">WORKSPACE</div>
      <nav className="side-nav" aria-label="Monitor sections">
        <a className="side-nav-item active" href="#monitor"><span className="nav-glyph">⌂</span><span>Monitor</span><i>01</i></a>
        <a className="side-nav-item" href="#signals"><span className="nav-glyph">⌁</span><span>Signals</span></a>
        <a className="side-nav-item" href="#vitals"><span className="nav-glyph">◌</span><span>Vitals</span></a>
        <a className="side-nav-item" href="#inference"><span className="nav-glyph">◎</span><span>Model output</span></a>
      </nav>
      <div className="sidebar-bottom">
        <div className="connection-card">
          <span className="connection-icon">⌁</span>
          <div><strong>Local research</strong><small>Loopback only · de-identified</small></div>
          <span className="connection-dot" />
        </div>
        <div className="sidebar-disclaimer">Prototype only. Not for clinical decisions.</div>
        <div className="sidebar-version">SAFENES / WEBUI <span>0.1</span></div>
      </div>
    </aside>
  );
}

export default function App() {
  const replay = useReplay();
  const [apiHealthy, setApiHealthy] = useState(false);
  const [gridEnabled, setGridEnabled] = useState(true);
  const frame = replay.frame;
  const cursorSec = replay.cursorSec;
  const controls = {
    play: replay.play,
    pause: replay.pause,
    seek: replay.seek,
    setSpeed: replay.setSpeed,
    restart: replay.restart,
  };

  useEffect(() => {
    let active = true;
    void checkHealth().then(() => {
      if (active) setApiHealthy(true);
    }).catch(() => {
      if (active) setApiHealthy(false);
    });
    return () => { active = false; };
  }, []);

  return (
    <div className="app-shell" id="monitor">
      <Sidebar />
      <div className="app-main">
        <Header
          frame={frame}
          apiHealthy={apiHealthy}
          gridEnabled={gridEnabled}
          onToggleGrid={() => setGridEnabled((enabled) => !enabled)}
        />
        <main className="page-scroll">
          <div className="page-intro">
            <div>
              <div className="page-eyebrow"><span className="eyebrow-line" />OPERATING ROOM · RESEARCH</div>
              <h1>Perioperative monitor</h1>
              <p>Patient signals and model outputs on one synchronized case-time axis.</p>
            </div>
            <div className="session-meta">
              <span className="session-source-dot" />
              <div><small>SOURCE</small><strong>{frame ? "VitalDB real replay" : "Select a case"}</strong></div>
              <span className="meta-divider" />
              <div><small>CASE TIME</small><strong>{frame ? formatTime(cursorSec) : "—"}</strong></div>
              <span className="meta-divider" />
              <div><small>SESSION</small><strong>{frame ? "#" + frame.case_id : "IDLE"}</strong></div>
            </div>
          </div>

          <div className="monitor-workspace">
            <aside className="monitor-control-rail" aria-label="Replay controls">
              <ReplayControls
                session={replay.session}
                frame={frame}
                cursorSec={cursorSec}
                cases={replay.cases}
                searching={replay.searching}
                loading={replay.loading}
                error={replay.error}
                onFindCases={replay.findCases}
                onLoadCase={replay.loadCase}
                actions={controls}
                apiHealthy={apiHealthy}
              />
            </aside>

            <div className="monitor-primary-column">
              <WaveformStack frame={frame} gridEnabled={gridEnabled} cursorSec={cursorSec} />
              <div className="insight-column">
                <PredictionPanel frame={frame} />
                <div id="vitals"><VitalsPanel frame={frame} /></div>
                <div className="timeline-note">
                  <div className="timeline-note-icon">↔</div>
                  <p><strong>One case-time axis.</strong> Waveform windows, recorded values, and prediction timestamps follow the same replay cursor.</p>
                </div>
              </div>
            </div>
          </div>

          <footer className="page-footer">
            <span>SAFEANES RESEARCH PROTOTYPE</span>
            <span>De-identified public data · Model package validation is limited</span>
          </footer>
        </main>
      </div>
    </div>
  );
}
