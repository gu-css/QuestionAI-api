#!/usr/bin/env python3
"""
train_hdbscan.py - Train & serialize the HDBSCAN unsupervised categorizer (Option A).

Replicates, standalone (without touching the research notebooks), the HDBSCAN
pipeline from the consolidated notebook:
    sentence-transformer embeddings (paraphrase-multilingual-MiniLM-L12-v2)
    -> UMAP (10d, cosine) -> HDBSCAN.

Serializes to:
    models/unsupervised_hdbscan.joblib
    models/cluster_topics_hdbscan.json

Requires the optional dependencies (install once):
    pip install -r requirements-hdbscan.txt

Usage:
    python scripts/train_hdbscan.py [--data PATH] [--min-cluster-size N]

Data: a CSV with a column of clickbait headlines. By default it reuses the
production dataset (data/detector-clickbait-br-processed.csv, filtering label==1).
To reproduce the exact research corpus, point --data at the `base.csv` used by
the consolidated notebook. The notebook auto-searched min_cluster_size over
[40, 50, 60, 70, 80, 100, 120] targeting 5-20 clusters with minimum noise.
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import pandas as pd

from src.pipeline import HDBSCANCategorizer

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "detector-clickbait-br-processed.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def load_clickbaits(path: str) -> list[str]:
    df = pd.read_csv(path)
    if "label" in df.columns:
        df = df[df["label"] == 1]
    col = "text" if "text" in df.columns else df.columns[0]
    return df[col].astype(str).tolist()


def main() -> int:
    parser = argparse.ArgumentParser(description="Train & serialize the HDBSCAN categorizer.")
    parser.add_argument("--data", default=DATA_PATH, help="CSV with clickbait headlines.")
    parser.add_argument("--min-cluster-size", type=int, default=None,
                        help="HDBSCAN min_cluster_size (default: 60).")
    args = parser.parse_args()

    texts = load_clickbaits(args.data)
    print(f"Loaded {len(texts)} clickbait headlines from {args.data}")

    categorizer = HDBSCANCategorizer()
    if args.min_cluster_size is not None:
        categorizer.min_cluster_size = args.min_cluster_size
    categorizer.fit(texts)

    n_noise = int((categorizer.labels_ == -1).sum())
    n_clusters = len(set(int(c) for c in categorizer.labels_) - {-1})
    print(f"Clusters: {n_clusters}, noise: {n_noise} ({100 * n_noise / len(texts):.1f}%)")

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "unsupervised_hdbscan.joblib")
    joblib.dump(categorizer, model_path, compress=3)
    print(f"Saved {model_path}")

    topics_path = os.path.join(MODELS_DIR, "cluster_topics_hdbscan.json")
    with open(topics_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in categorizer.cluster_topics.items()},
                  f, ensure_ascii=False, indent=2)
    print(f"Saved {topics_path}")
    print("\nDone. Set UNSUPERVISED_MODEL=hdbscan to use this backend.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
