#!/usr/bin/env python3
"""
scripts/train.py - Clickbait Detection & Categorization Training Pipeline.

Fulfills Milestone M1 (Model Training & Serialization for R1):
1. Downloads authoritative dataset from Hugging Face if not cached.
2. Performs data normalization, deduplication, and stratified 80/10/10 split.
3. Trains Supervised Champion Model (Approach C: Calibrated Hybrid SVM, LinearSVC + CalibratedClassifierCV sigmoid).
4. Evaluates on the held-out test partition (F1, Accuracy, Precision, Recall, ROC-AUC).
5. Retrains on Train + Validation partitions for production deployment.
6. Trains Unsupervised Champion Model (Spherical K-Means: TF-IDF + SVD 40d + L2 + KMeans, 6 interpretable archetypes).
7. Serializes independent model artifacts to:
   - models/supervised_model.joblib
   - models/unsupervised_model.joblib
   - models/metadata.json
   - models/cluster_topics.json
"""

import os
import sys
import json
import time
import urllib.request
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features import (
    normalize_text,
    clean_text,
    FINAL_STOPWORDS_SET,
    KEEP_STOPWORDS,
    STYLO_FEATURES,
    SENSATIONALIST_TRIGGERS,
)
from src.pipeline import (
    build_supervised_pipeline,
    UnsupervisedCategorizer,
    DEFAULT_CLUSTER_TOPICS,
)

DATA_URL = (
    "https://huggingface.co/datasets/rodrigoaraujorosa/detector-clickbait-br-datasets"
    "/resolve/main/detector-clickbait-br-processed.csv"
)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "detector-clickbait-br-processed.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RANDOM_STATE = 42


def ensure_dataset() -> str:
    """Download dataset from Hugging Face if not already present."""
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    if not os.path.exists(DATA_PATH):
        print(f"📥 Downloading dataset from {DATA_URL}...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
        print(f"✅ Downloaded to {DATA_PATH} ({os.path.getsize(DATA_PATH) / 1024:.1f} KB)")
    else:
        print(f"✅ Using cached dataset at {DATA_PATH} ({os.path.getsize(DATA_PATH) / 1024:.1f} KB)")
    return DATA_PATH


def prepare_data(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Applies the exact data lineage protocol from research notebooks:
    1. Normalization and removal of 23 superficial duplicates on text_norm
    2. Stratified 80/10/10 split (random_state=42)
    3. Final cleaning with H7 discriminative stopwords and removal of 110 clean duplicates
    """
    print("🧹 Loading and preprocessing dataset...")
    df_raw = pd.read_csv(data_path)
    print(f"   Raw samples: {len(df_raw):,}")

    df = df_raw.copy()
    df["text_norm"] = df["text"].apply(normalize_text)

    # Deduplication stage 1: text_norm duplicates
    dup_norm = df.duplicated("text_norm", keep="first")
    print(f"   Deduplication stage 1: {int(dup_norm.sum())} duplicates removed.")
    df = df[~dup_norm].reset_index(drop=True)

    # Stratified split: 80% train, 10% val, 10% test
    target = "label"
    idx = np.arange(len(df))
    idx_tmp, idx_test = train_test_split(
        idx, test_size=0.10, stratify=df[target], random_state=RANDOM_STATE
    )
    val_share = 0.10 / 0.90
    idx_train, idx_val = train_test_split(
        idx_tmp, test_size=val_share, stratify=df.loc[idx_tmp, target], random_state=RANDOM_STATE
    )

    df["particao"] = "treino"
    df.loc[idx_val, "particao"] = "validacao"
    df.loc[idx_test, "particao"] = "teste"

    # Deduplication stage 2: text_clean with priority train > val > test
    df["text_clean"] = df["text"].apply(lambda t: clean_text(t, FINAL_STOPWORDS_SET))
    ordem = df["particao"].map({"treino": 0, "validacao": 1, "teste": 2})
    dup_clean = df.assign(_o=ordem).sort_values("_o").duplicated("text_clean", keep="first").reindex(df.index)
    print(f"   Deduplication stage 2: {int(dup_clean.sum())} duplicates removed across partitions.")
    df = df[~dup_clean].reset_index(drop=True)

    df_train = df[df["particao"] == "treino"].reset_index(drop=True)
    df_val = df[df["particao"] == "validacao"].reset_index(drop=True)
    df_test = df[df["particao"] == "teste"].reset_index(drop=True)

    print(f"   Final dataset: {len(df):,} headlines")
    print(f"   - Treino: {len(df_train):,} ({100 * len(df_train) / len(df):.1f}%)")
    print(f"   - Validação: {len(df_val):,} ({100 * len(df_val) / len(df):.1f}%)")
    print(f"   - Teste: {len(df_test):,} ({100 * len(df_test) / len(df):.1f}%)")

    return df_train, df_val, df_test


def train_supervised_model(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Trains Approach C: Calibrated Hybrid SVM.
    Evaluates on test partition, then fits on train+val for production artifact.
    """
    print("\n🎯 Training Supervised Model: Calibrated Hybrid SVM (Approach C)...")
    t0 = time.time()

    # 1. Train on training set to evaluate generalization on test set
    eval_pipeline = build_supervised_pipeline(
        ngram_range=(1, 2),
        max_features=5000,
        min_df=2,
        sublinear_tf=True,
        style_weight=0.1,
        c_param=1.0,
        cv_folds=5,
        random_state=RANDOM_STATE,
    )

    eval_pipeline.fit(df_train["text"], df_train["label"])
    train_time = time.time() - t0
    print(f"   Fitted on {len(df_train):,} training samples in {train_time:.2f}s.")

    # Evaluate on held-out test partition
    y_test = df_test["label"].to_numpy()
    y_pred = eval_pipeline.predict(df_test["text"])
    y_proba = eval_pipeline.predict_proba(df_test["text"])[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred))
    rec = float(recall_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    roc_auc = float(roc_auc_score(y_test, y_proba))
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"   📊 Test Evaluation Metrics (Held-out Test set, n={len(df_test):,}):")
    print(f"      Accuracy:  {acc:.4f}")
    print(f"      Precision: {prec:.4f}")
    print(f"      Recall:    {rec:.4f}")
    print(f"      F1 Score:  {f1:.4f}")
    print(f"      ROC-AUC:   {roc_auc:.4f}")
    print(f"      Confusion Matrix: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")

    assert f1 >= 0.95, f"Supervised model F1 {f1:.4f} is below acceptance threshold 0.95"

    # 2. Retrain champion on train + val (8,463 samples) for production deployment
    print("   Retraining champion pipeline on Train + Validation partitions for production...")
    df_train_val = pd.concat([df_train, df_val], ignore_index=True)

    prod_pipeline = build_supervised_pipeline(
        ngram_range=(1, 2),
        max_features=5000,
        min_df=2,
        sublinear_tf=True,
        style_weight=0.1,
        c_param=1.0,
        cv_folds=5,
        random_state=RANDOM_STATE,
    )
    prod_pipeline.fit(df_train_val["text"], df_train_val["label"])
    print(f"   ✅ Production supervised model fitted on {len(df_train_val):,} samples.")

    metrics = {
        "test_accuracy": round(acc, 4),
        "test_precision": round(prec, 4),
        "test_recall": round(rec, 4),
        "test_f1": round(f1, 4),
        "test_roc_auc": round(roc_auc, 4),
        "confusion_matrix": cm,
        "train_samples": len(df_train_val),
        "test_samples": len(df_test),
    }

    return prod_pipeline, metrics


def train_unsupervised_model(df_train: pd.DataFrame) -> Tuple[UnsupervisedCategorizer, Dict[str, Any]]:
    """
    Trains Unsupervised Champion Model: Spherical K-Means.
    Extracts clickbait headlines (label == 1), maps into 6 interpretable archetypes.
    """
    print("\n🔍 Training Unsupervised Model: Spherical K-Means (Bait Categorizer)...")
    t0 = time.time()

    clickbaits = df_train[df_train["label"] == 1]["text"].tolist()
    print(f"   Corpus size: {len(clickbaits):,} clickbait headlines.")

    categorizer = UnsupervisedCategorizer(
        n_clusters=6,
        n_components=40,
        max_features=4000,
        random_state=RANDOM_STATE,
    )
    categorizer.fit(clickbaits)
    elapsed = time.time() - t0
    print(f"   Fitted 6 clusters in {elapsed:.2f}s.")

    topics_summary = {}
    for c in range(6):
        topic_name = categorizer.cluster_topics.get(c, "Sensacionalismo Geral")
        keywords = categorizer.cluster_keywords.get(c, [])
        topics_summary[str(c)] = {
            "topic": topic_name,
            "keywords": keywords,
        }
        print(f"   Cluster {c}: '{topic_name}' -> Top terms: {keywords[:5]}")

    return categorizer, topics_summary


def save_artifacts(
    supervised_model: Any,
    unsupervised_model: UnsupervisedCategorizer,
    sup_metrics: Dict[str, Any],
    topics_summary: Dict[str, Any],
) -> None:
    """Serializes all model artifacts independently to models/."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    sup_path = os.path.join(MODELS_DIR, "supervised_model.joblib")
    unsup_path = os.path.join(MODELS_DIR, "unsupervised_model.joblib")
    meta_path = os.path.join(MODELS_DIR, "metadata.json")
    topics_path = os.path.join(MODELS_DIR, "cluster_topics.json")

    print("\n💾 Serializing artifacts to models/ directory...")

    # 1. Independent supervised model serialization
    joblib.dump(supervised_model, sup_path, compress=3)
    print(f"   ✅ Saved {sup_path} ({os.path.getsize(sup_path) / 1024:.1f} KB)")

    # 2. Independent unsupervised model serialization
    joblib.dump(unsupervised_model, unsup_path, compress=3)
    print(f"   ✅ Saved {unsup_path} ({os.path.getsize(unsup_path) / 1024:.1f} KB)")

    # 3. Cluster topics mapping
    cluster_topics_map = {str(k): v for k, v in unsupervised_model.cluster_topics.items()}
    with open(topics_path, "w", encoding="utf-8") as f:
        json.dump(cluster_topics_map, f, ensure_ascii=False, indent=2)
    print(f"   ✅ Saved {topics_path}")

    # 4. Comprehensive metadata
    metadata = {
        "model_type": "Calibrated Hybrid SVM (Approach C)",
        "unsupervised_model_type": "Spherical K-Means (TF-IDF + SVD 40d + L2)",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "supervised_metrics": sup_metrics,
        "decision_threshold": 0.50,
        "uncertainty_zone": [0.10, 0.85],
        "stylometric_features": STYLO_FEATURES,
        "discriminative_stopwords_count": len(KEEP_STOPWORDS),
        "cluster_topics": cluster_topics_map,
        "cluster_details": topics_summary,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"   ✅ Saved {meta_path}")


def main() -> int:
    start_total = time.time()
    print("=" * 70)
    print("🚀 CLICKBAIT MODEL TRAINING PIPELINE (Milestone M1)")
    print("=" * 70)

    data_file = ensure_dataset()
    df_train, df_val, df_test = prepare_data(data_file)

    supervised_model, sup_metrics = train_supervised_model(df_train, df_val, df_test)
    unsupervised_model, topics_summary = train_unsupervised_model(df_train)

    save_artifacts(supervised_model, unsupervised_model, sup_metrics, topics_summary)

    total_time = time.time() - start_total
    print("\n" + "=" * 70)
    print(f"🎉 Pipeline completed successfully in {total_time:.2f}s!")
    print(f"   Supervised F1: {sup_metrics['test_f1']:.4f} | Accuracy: {sup_metrics['test_accuracy']:.4f}")
    print(f"   Ready for verification via scripts/verify_models.py")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
