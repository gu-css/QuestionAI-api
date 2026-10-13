# QuestionAI-api

Clickbait detection API (FastAPI). Production runs K-Means only.

## Run

```sh
pip install -r requirements.txt
python app.py  # PORT=8000 by default
```

## Endpoints

- `GET /health` → `{"status": "healthy"}`
- `POST /predict?headline=...&model=kmeans`
- `POST /predict_batch` → `{"headlines": [...], "model": "kmeans"}`
- `POST /api/predict` → `{"headline": "..."}` or `{"headlines": [...]}`

Response: `headline, is_clickbait, clickbait_status, probability, confidence, bait_type`.

## HDBSCAN (local only)

The HDBSCAN model (~441 MB) is gitignored. Generate it with
`scripts/train_hdbscan.py` (deps in `requirements-hdbscan.txt`),
or download it from the
[modelos-hdbscan-v1 release](https://github.com/gu-css/QuestionAI-api/releases/tag/modelos-hdbscan-v1)
into `models/`.

## Deploy (Render)

Blueprint in `render.yaml` deploys the repo root.
