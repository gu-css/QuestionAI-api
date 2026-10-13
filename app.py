import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure root dir is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import threading
import time
import uvicorn
from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.pipeline import ClickbaitDetectionSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Base directories
MODELS_DIR = BASE_DIR / "models"

# Eagerly load the system (both K-Means and, if available, HDBSCAN categorizers)
logger.info("Initializing Clickbait Detection System...")
system = ClickbaitDetectionSystem.load(
    models_dir=str(MODELS_DIR),
    unsup_artifacts_dir=str(MODELS_DIR),
)

app = FastAPI(title="Clickbait Shield API")

# Allow CORS for the extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictBatchRequest(BaseModel):
    headlines: List[str]
    model: str = "kmeans"


class PredictRequest(BaseModel):
    headline: Optional[str] = None
    headlines: Optional[List[str]] = None
    data: Optional[List[str]] = None  # legacy alias for headlines
    model: str = "kmeans"


MAX_BATCH_SIZE = 100
VALID_MODELS = {"kmeans", "hdbscan"}

# Tiny per-IP rate limit: the endpoint is public with no auth, so one script
# could burn the free hosting quota. Local dev and tests bypass it below.
_RATE_LIMIT_MAX_REQUESTS = 60
_RATE_LIMIT_WINDOW_SECONDS = 60.0
_RATE_LIMIT_BYPASS_HOSTS = {"testclient", "127.0.0.1", "::1", "localhost"}
_rate_buckets: Dict[str, List[float]] = {}
_rate_lock = threading.Lock()


@app.middleware("http")
async def _rate_limit_middleware(request: Request, call_next):
    if request.url.path not in ("/health", "/api/health"):
        host = request.client.host if request.client else "unknown"
        if host not in _RATE_LIMIT_BYPASS_HOSTS:
            now = time.monotonic()
            with _rate_lock:
                if len(_rate_buckets) > 5000:  # drop stale entries so memory does not grow
                    _rate_buckets = {h: ts for h, ts in _rate_buckets.items() if any(now - t < _RATE_LIMIT_WINDOW_SECONDS for t in ts)}
                hits = [t for t in _rate_buckets.get(host, []) if now - t < _RATE_LIMIT_WINDOW_SECONDS]
                if len(hits) >= _RATE_LIMIT_MAX_REQUESTS:
                    return JSONResponse({"detail": "Rate limit exceeded. Try again later."}, status_code=429)
                hits.append(now)
                _rate_buckets[host] = hits
    return await call_next(request)


def _require_model(model: str) -> str:
    if model not in VALID_MODELS:
        raise HTTPException(status_code=422, detail=f"model must be one of {sorted(VALID_MODELS)}")
    return model


def _check_batch_size(headlines: List[str]) -> None:
    if len(headlines) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=413, detail=f"Batch too large: max {MAX_BATCH_SIZE} headlines")


def _health_payload() -> Dict[str, Any]:
    return {
        "status": "healthy" if getattr(system, "supervised", None) is not None else "degraded",
        "supervised_loaded": getattr(system, "supervised", None) is not None,
        "unsupervised_loaded": getattr(system, "unsupervised", None) is not None,
        "hdbscan_loaded": getattr(system, "hdbscan", None) is not None,
    }


def format_result(res: Dict[str, Any]) -> Dict[str, Any]:
    """Formats an internal prediction dict into the public API response schema."""
    is_clickbait = bool(res["is_clickbait"])
    return {
        "headline": res["headline"],
        "is_clickbait": is_clickbait,
        "clickbait_status": "clickbait" if is_clickbait else "not_clickbait",
        "probability": round(float(res["probability"]), 4),
        "confidence": round(float(res["confidence"]), 4),
        "bait_type": res["bait_type"] if is_clickbait else None,
    }

@app.get("/health")
@app.get("/api/health")
def health_check():
    payload = _health_payload()
    if not payload["supervised_loaded"]:
        raise HTTPException(status_code=503, detail="models not loaded")
    return payload

@app.post("/predict")
def predict_single(
    headline: Optional[str] = Query(None),
    model: str = Query("kmeans"),
    body: Optional[PredictRequest] = Body(None),
):
    # Supports both query params (extension) and a JSON body (tests/contract).
    if headline is None and body is not None:
        if body.headline is None and body.headlines is not None:
            _require_model(body.model)
            _check_batch_size(body.headlines)
            results = [format_result(r) for r in system.predict_many(body.headlines, categorizer=body.model)]
            return {"results": results, "total": len(results)}
        headline = body.headline
        model = body.model
    if headline is None:
        raise HTTPException(status_code=422, detail="headline is required")
    return format_result(system.predict(headline, categorizer=_require_model(model)))

@app.post("/api/predict")
def api_predict(req: PredictRequest):
    _require_model(req.model)
    headlines = req.headlines if req.headlines is not None else req.data
    if headlines is not None:
        _check_batch_size(headlines)
        results = [format_result(r) for r in system.predict_many(headlines, categorizer=req.model)]
        return {"results": results, "total": len(results)}
    if req.headline is not None:
        return format_result(system.predict(req.headline, categorizer=req.model))
    raise HTTPException(status_code=422, detail="headline or headlines is required")

@app.post("/predict_batch")
def predict_batch(req: PredictBatchRequest):
    _require_model(req.model)
    _check_batch_size(req.headlines)
    results = [format_result(r) for r in system.predict_many(req.headlines, categorizer=req.model)]
    return {"results": results, "total": len(results)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    workers = int(os.environ.get("WORKERS", "1"))
    logger.info(f"Starting FastAPI on port {port} with {workers} worker(s)")
    if workers > 1:
        # Import-string form so each worker process loads its own copy of the model.
        uvicorn.run("app:app", host="0.0.0.0", port=port, workers=workers)
    else:
        uvicorn.run(app, host="0.0.0.0", port=port)
