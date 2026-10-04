import type { ModelPrediction, ReplayFrame } from "../types";
import { formatTime } from "../types";

function ModelCard({ model }: { model: ModelPrediction }) {
  const modelClass = model.model_id.toLowerCase();
  const statusClass = model.status.toLowerCase().replaceAll(" ", "-");
  const entries = Object.entries(model.scores);
  return (
    <article className={`model-card ${modelClass}`}>
      <div className="model-card-top">
        <div className="model-id">{model.model_id}</div>
        <span className={`model-status ${statusClass}`}>{model.status}</span>
      </div>
      {model.status === "READY" && entries.length > 0 ? (
        <>
          <div className="score-list">
            {entries.map(([name, score]) => (
              <div className="score-row" key={name}>
                <span>{name}</span>
                <strong>{Number.isFinite(score) ? score.toFixed(4) : "—"}</strong>
                {model.thresholds[name] !== undefined && (
                  <small>threshold {model.thresholds[name].toFixed(4)}</small>
                )}
              </div>
            ))}
          </div>
          <div className="model-caption">{model.score_kind || "Model output"}</div>
        </>
      ) : (
        <div className="model-reason">
          {model.reason || (model.status === "READY" ? "No score returned by model." : "No prediction is available at this case time.")}
        </div>
      )}
      <div className="model-time">Updated at {formatTime(model.timestamp_sec)}</div>
    </article>
  );
}

function HistoryChart({ frame }: { frame: ReplayFrame | null }) {
  const history = frame?.prediction_history ?? [];
  const points = history.flatMap((point) =>
    point.models.flatMap((model) => Object.values(model.scores).slice(0, 1).map((score) => ({
      model: model.model_id,
      time: point.timestamp_sec,
      score,
    }))),
  );
  const maxTime = Math.max(30, frame?.case_time_sec ?? 0);
  const x = (time: number) => 32 + (time / maxTime) * 320;
  const y = (score: number) => 82 - Math.max(0, Math.min(1, score)) * 58;
  const modelIds = ["UC04", "UC05"];
  const colors: Record<string, string> = { UC04: "#aa9bff", UC05: "#4bd0be" };
  return (
    <div className="history-plot" aria-label="Timestamped model output history">
      {points.length === 0 ? (
        <div className="history-empty">Prediction history appears as the replay advances.</div>
      ) : (
        <svg viewBox="0 0 372 104" role="img" aria-label="Raw model score history by case time">
          <line x1="32" y1="24" x2="352" y2="24" />
          <line x1="32" y1="52" x2="352" y2="52" />
          <line x1="32" y1="82" x2="352" y2="82" />
          {modelIds.map((id) => {
            const row = points.filter((point) => point.model === id);
            if (row.length === 0) return null;
            const path = row.map((point, index) => `${index === 0 ? "M" : "L"} ${x(point.time)} ${y(point.score)}`).join(" ");
            return <path key={id} d={path} stroke={colors[id]} fill="none" strokeWidth="2" />;
          })}
          {points.map((point, index) => (
            <circle key={`${point.model}-${point.time}-${index}`} cx={x(point.time)} cy={y(point.score)} r="3" fill={colors[point.model]} />
          ))}
        </svg>
      )}
      <div className="history-legend">
        {modelIds.map((id) => (
          <span key={id}><i style={{ background: colors[id] }} />{id}</span>
        ))}
        <small>Raw outputs · timestamped</small>
      </div>
    </div>
  );
}

export default function PredictionPanel({ frame }: { frame: ReplayFrame | null }) {
  const latest = frame?.prediction_history.at(-1);
  const models = latest?.models ?? [];
  return (
    <section className="panel ai-panel" id="inference" aria-labelledby="ai-heading">
      <div className="panel-heading ai-heading">
        <div>
          <div className="section-kicker">SAFEANES · RESEARCH OUTPUT</div>
          <h2 id="ai-heading">Model inference</h2>
        </div>
        <span className="sync-tag"><span /> SAME TIMELINE</span>
      </div>
      {!frame ? (
        <div className="model-mode-note">
          <span className="note-lock">⌁</span>
          <div>
            <strong>Waiting for a VitalDB replay</strong>
            <p>UC04 needs at least 5 minutes of history; UC05 needs at least 10 minutes. Signal quality and anesthesia timing can add further gates.</p>
          </div>
        </div>
      ) : models.length > 0 ? (
        <div className="model-cards">
          {models.map((model) => <ModelCard key={model.model_id} model={model} />)}
        </div>
      ) : (
        <div className="model-mode-note">
          <span className="note-lock">◷</span>
          <div>
            <strong>No inference at this timestamp</strong>
            <p>Predictions are requested every 30 case-seconds while replay is playing and once after seek.</p>
          </div>
        </div>
      )}
      <div className="history-heading">
        <span>Prediction history</span>
        <small>{latest ? "Latest · " + formatTime(latest.timestamp_sec) : "No results yet"}</small>
      </div>
      <HistoryChart frame={frame} />
      <p className="research-disclaimer">
        Research only. Scores are model outputs, not clinical alarms, diagnoses, or treatment advice.
      </p>
    </section>
  );
}
