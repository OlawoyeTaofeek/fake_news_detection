"""
Phase 3: The backend API.

Run locally (from the project root):
    uvicorn api.main:app --reload
Then open http://127.0.0.1:8000/docs for interactive API documentation.
"""

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api import database
from api.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    StatsResponse,
)
from src.preprocess import clean_text
from src.text_stats import oov_ratio, word_count

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fake-news-api")

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.joblib"))

# Holds the loaded model so it is read from disk only once, at startup
state = {"model": None, "version": None}


def load_model() -> None:
    if not MODEL_PATH.exists():
        logger.warning("No model found at %s. Run `python -m src.train` first.", MODEL_PATH)
        return
    state["model"] = joblib.load(MODEL_PATH)
    meta_path = MODEL_PATH.parent / "metadata.json"
    state["version"] = (
        json.loads(meta_path.read_text()).get("model_version", "unknown")
        if meta_path.exists()
        else "unknown"
    )
    logger.info("Loaded model version %s", state["version"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    database.init_db()
    load_model()
    yield


app = FastAPI(
    title="Fake News Detector API",
    description="Flags misinformation-style writing. Built as an end-to-end MLOps teaching project.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allows a browser frontend (e.g. React on another port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root():
    return {"message": "Fake News Detector API. See /docs for usage."}


@app.get("/health", response_model=HealthResponse)
def health():
    loaded = state["model"] is not None
    return HealthResponse(
        status="ok" if loaded else "model_not_loaded",
        model_loaded=loaded,
        model_version=state["version"],
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model = state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train a model first.")

    start = time.perf_counter()
    cleaned = clean_text(req.text)  # same cleaning as training!
    prob_fake = float(model.predict_proba([cleaned])[0][1])
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    label = "fake" if prob_fake >= 0.5 else "real"
    confidence = prob_fake if label == "fake" else 1 - prob_fake
    request_id = str(uuid.uuid4())

    try:
        database.log_prediction(
            request_id=request_id,
            text=req.text,
            word_count=word_count(cleaned),
            oov_ratio=oov_ratio(model, cleaned),
            label=label,
            prob_fake=round(prob_fake, 4),
            confidence=round(confidence, 4),
            latency_ms=latency_ms,
            model_version=state["version"],
        )
    except Exception:  # logging must never break predictions for users
        logger.exception("Failed to log prediction %s", request_id)

    logger.info("prediction id=%s label=%s conf=%.3f latency=%sms", request_id, label, confidence, latency_ms)

    return PredictResponse(
        request_id=request_id,
        label=label,
        confidence=round(confidence, 4),
        prob_fake=round(prob_fake, 4),
        model_version=state["version"],
        latency_ms=latency_ms,
    )


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest):
    if not database.save_feedback(req.request_id, req.correct):
        raise HTTPException(status_code=404, detail="Unknown request_id")
    return FeedbackResponse(status="saved", request_id=req.request_id)


@app.get("/stats", response_model=StatsResponse)
def stats():
    """Live monitoring numbers, used by the frontend sidebar."""
    return StatsResponse(**database.get_stats())
