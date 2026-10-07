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

# Eagerly load the system
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

def process_headline(headline: str) -> Dict[str, Any]:
    """Processes a single headline through the decoupled prediction system."""
    if not isinstance(headline, str) or not headline.strip():
        return {
            "headline": headline if isinstance(headline, str) else str(headline or ""),
            "is_clickbait": False,
            "clickbait_status": "not_clickbait",
            "probability": 0.0,
            "confidence": 1.0,
            "bait_type": None,
        }

    res = system.predict(headline)
    is_clickbait = bool(res["is_clickbait"])
    prob = float(res["probability"])
    conf = float(res["confidence"])

    bait_type = res["bait_type"] if is_clickbait else None

    return {
        "headline": res["headline"],
        "is_clickbait": is_clickbait,
        "clickbait_status": "clickbait" if is_clickbait else "not_clickbait",
        "probability": round(prob, 4),
        "confidence": round(conf, 4),
        "bait_type": bait_type,
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
def predict_single(headline: str):
    return process_headline(headline)

@app.post("/predict_batch")
def predict_batch(req: PredictBatchRequest):
    results = [process_headline(h) for h in req.headlines]
    return {"results": results}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting FastAPI on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
