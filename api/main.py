"""FastAPI service exposing the fine-tuned text-to-SQL model.

Run locally:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health              liveness check
    POST /generate            {schema, question} -> {sql}
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import ProjectConfig
from src.inference import SQLPredictor

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_predictor: SQLPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor
    cfg = ProjectConfig.load("configs/default.yaml")
    logger.info("Loading model for serving...")
    _predictor = SQLPredictor(cfg)
    logger.info("Model loaded, ready to serve.")
    yield
    _predictor = None


app = FastAPI(
    title="Text-to-SQL LoRA API",
    description="Fine-tuned TinyLlama-1.1B (LoRA) that converts a table schema + natural-language question into SQL.",
    version="1.0.0",
    lifespan=lifespan,
)


class GenerateRequest(BaseModel):
    schema: str = Field(..., description="CREATE TABLE statement(s) describing the schema")
    question: str = Field(..., description="Natural-language question to convert to SQL")
    max_new_tokens: int | None = Field(None, ge=1, le=512)


class GenerateResponse(BaseModel):
    sql: str
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _predictor is not None}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")
    if not req.schema.strip() or not req.question.strip():
        raise HTTPException(status_code=422, detail="schema and question must be non-empty")

    start = time.perf_counter()
    try:
        sql = _predictor.predict(req.schema, req.question, req.max_new_tokens)
    except Exception as e:  # noqa: BLE001 - surface as 500 with message
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}") from e
    latency_ms = (time.perf_counter() - start) * 1000

    return GenerateResponse(sql=sql, latency_ms=round(latency_ms, 1))


# Serve the frontend (index.html, style.css, script.js) from the same
# FastAPI instance. Mounted last so it never shadows the /health or
# /generate routes above — Starlette matches routes in registration order.
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
