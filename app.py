import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure root dir is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
def predict_single(headline: str, model: str = "kmeans"):
    return format_result(system.predict(headline, categorizer=model))

@app.post("/predict_batch")
def predict_batch(req: PredictBatchRequest):
    results = [format_result(r) for r in system.predict_many(req.headlines, categorizer=req.model)]
    return {"results": results}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    workers = int(os.environ.get("WORKERS", "1"))
    logger.info(f"Starting FastAPI on port {port} with {workers} worker(s)")
    if workers > 1:
        # Import-string form so each worker process loads its own copy of the model.
        uvicorn.run("app:app", host="0.0.0.0", port=port, workers=workers)
    else:
        uvicorn.run(app, host="0.0.0.0", port=port)
