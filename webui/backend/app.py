from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query, Response

from core.model_inference import SafeAnesInference
from core.vitaldb_source import VitalDBSource
from webui.backend.schemas import (
    CasesResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    ReplayControlRequest,
    ReplayControlResponse,
    ReplayCreatedResponse,
    ReplayFrameResponse,
)
from webui.backend.service import (
    CaseNotFound,
    CaseSourceError,
    ReplayNotFound,
    ReplayService,
    ReplayServiceError,
)


def _http_error(exc: ReplayServiceError) -> HTTPException:
    if isinstance(exc, CaseNotFound):
        code, status = "CASE_NOT_FOUND", 404
    elif isinstance(exc, ReplayNotFound):
        code, status = "REPLAY_NOT_FOUND", 404
    elif isinstance(exc, CaseSourceError):
        code, status = "VITALDB_UNAVAILABLE", 503
    else:
        code, status = "INVALID_REQUEST", 422
    return HTTPException(
        status_code=status,
        detail={"code": code, "message": str(exc)},
    )


def create_app(source=None, inference=None, clock=None) -> FastAPI:
    service = ReplayService(
        source=source if source is not None else VitalDBSource(),
        inference=inference if inference is not None else SafeAnesInference(),
        clock=clock,
    )
    app = FastAPI(
        title="SafeAnes Local Replay API",
        version="0.1.0",
        description="Loopback-only bridge for de-identified VitalDB research replay.",
    )
    app.state.replay_service = service

    @app.get("/api/health", response_model=HealthResponse)
    def health() -> dict:
        return {"status": "ok", "service": "SafeAnes Local Replay API"}

    @app.get("/api/cases", response_model=CasesResponse)
    def cases(limit: int = Query(default=50, ge=1, le=500)) -> dict:
        try:
            return service.available_cases(limit)
        except ReplayServiceError as exc:
            raise _http_error(exc) from exc

    @app.post("/api/replays", response_model=ReplayCreatedResponse, status_code=201)
    def load_replay(body: dict) -> dict:
        try:
            case_id = body.get("case_id")
            if isinstance(case_id, bool):
                raise ValueError("boolean case IDs are invalid")
            return service.load_replay(case_id)
        except ReplayServiceError as exc:
            raise _http_error(exc) from exc
        except (AttributeError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail={"code": "INVALID_CASE_ID", "message": str(exc)},
            ) from exc

    @app.get("/api/replays/{replay_id}/frame", response_model=ReplayFrameResponse)
    def frame(
        replay_id: str,
        window_seconds: float = Query(default=10.0, ge=1.0, le=30.0),
    ) -> dict:
        try:
            return service.frame(replay_id, window_seconds)
        except ReplayServiceError as exc:
            raise _http_error(exc) from exc

    @app.patch("/api/replays/{replay_id}", response_model=ReplayControlResponse)
    def control(replay_id: str, body: ReplayControlRequest) -> dict:
        try:
            return service.control(
                replay_id,
                action=body.action,
                position_sec=body.position_sec,
                speed=body.speed,
            )
        except ReplayServiceError as exc:
            raise _http_error(exc) from exc

    @app.post("/api/replays/{replay_id}/predictions", response_model=PredictionResponse)
    def predict(replay_id: str, body: PredictionRequest) -> dict:
        try:
            return service.predict(replay_id, body.case_time_sec)
        except ReplayServiceError as exc:
            raise _http_error(exc) from exc

    @app.delete("/api/replays/{replay_id}", status_code=204)
    def delete_replay(replay_id: str) -> Response:
        try:
            service.release(replay_id)
        except ReplayServiceError as exc:
            raise _http_error(exc) from exc
        return Response(status_code=204)

    return app


app = create_app()
