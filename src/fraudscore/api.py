"""FastAPI service: POST /score (single or batch), GET /health, GET /model."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, create_model

from fraudscore.data import FEATURES
from fraudscore.explain import reason_codes
from fraudscore.models import ModelBundle, load_bundle
from fraudscore.ratelimit import TokenBucketLimiter
from fraudscore.settings import Settings, get_settings

_feature_fields: dict[str, Any] = {
    name: (float, Field(allow_inf_nan=False, description=f"PCA component {name}"))
    for name in FEATURES
    if name != "Amount"
}
_feature_fields["Amount"] = (float, Field(ge=0, allow_inf_nan=False, description="Amount"))

Transaction: Any = create_model(
    "Transaction",
    __config__=ConfigDict(extra="forbid"),
    id=(str | None, Field(default=None, max_length=64, description="Caller reference")),
    **_feature_fields,
)


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transactions: list[Transaction] = Field(min_length=1)


class ReasonCode(BaseModel):
    feature: str
    value: float
    contribution: float = Field(description="SHAP contribution in log-odds")
    direction: Literal["increases_risk", "decreases_risk"]


class ScoreResult(BaseModel):
    id: str | None = None
    score: float = Field(ge=0, le=1, description="Calibrated fraud probability")
    decision: Literal["alert", "allow"]
    threshold: float
    reasons: list[ReasonCode] = Field(default_factory=list)
    model_version: str


class BatchResponse(BaseModel):
    results: list[ScoreResult]
    model_version: str


def score_rows(bundle: ModelBundle, rows: list[dict], *, explain: bool) -> list[ScoreResult]:
    frame = pd.DataFrame(rows)
    proba = bundle.proba(frame)
    codes = reason_codes(bundle, frame) if explain else [[] for _ in rows]
    return [
        ScoreResult(
            id=row.get("id"),
            score=float(p),
            decision="alert" if p >= bundle.threshold else "allow",
            threshold=bundle.threshold,
            reasons=[ReasonCode(**c) for c in cs],
            model_version=bundle.version,
        )
        for row, p, cs in zip(rows, proba, codes, strict=True)
    ]


def create_app(bundle: ModelBundle | None = None, settings: Settings | None = None) -> FastAPI:
    cfg = settings or get_settings()
    limiter = TokenBucketLimiter(cfg.rate_limit_per_minute, cfg.rate_limit_burst)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.bundle = bundle or load_bundle(cfg.model_path)
        yield

    app = FastAPI(title="fraud-scoring-service", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def rate_limit(request: Request, call_next):  # type: ignore[no-untyped-def]
        if request.url.path == "/score":
            client = request.client.host if request.client else "unknown"
            ok, retry = limiter.allow(client)
            if not ok:
                return JSONResponse(
                    {"detail": "rate limit exceeded"},
                    status_code=429,
                    headers={"Retry-After": str(max(1, round(retry)))},
                )
        return await call_next(request)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model_version": app.state.bundle.version}

    @app.get("/model")
    def model_info() -> dict[str, object]:
        b: ModelBundle = app.state.bundle
        return {
            "model_version": b.version,
            "kind": b.kind,
            "threshold": b.threshold,
            "cost_fp": b.cost_fp,
            "cost_fn": b.cost_fn,
            "features": b.features,
            "trained_on": b.trained_on,
        }

    @app.post("/score", response_model=ScoreResult | BatchResponse)
    def score(
        body: BatchRequest | Transaction,
        explain: Annotated[bool, Query(description="Include top-5 SHAP reason codes")] = True,
    ) -> ScoreResult | BatchResponse:
        b: ModelBundle = app.state.bundle
        if isinstance(body, BatchRequest):
            if len(body.transactions) > cfg.max_batch:
                raise HTTPException(413, f"batch larger than {cfg.max_batch}")
            results = score_rows(b, [t.model_dump() for t in body.transactions], explain=explain)
            return BatchResponse(results=results, model_version=b.version)
        return score_rows(b, [body.model_dump()], explain=explain)[0]

    return app


def __getattr__(name: str) -> FastAPI:
    """`uvicorn fraudscore.api:app` builds the default app lazily, so importing this module in
    tests does not require a trained model on disk."""
    if name == "app":
        return create_app()
    raise AttributeError(name)
