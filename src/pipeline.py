"""
Clickbait Detection System - Independent Supervised & Unsupervised Pipelines.
Enforces strict architectural decoupling between:
1. Supervised Clickbait Classifier (Calibrated Hybrid SVM: LinearSVC + CalibratedClassifierCV)
2. Unsupervised Bait Type Categorizer (Spherical K-Means: TF-IDF + SVD 40d + L2 + KMeans)
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans

from src.features import (
    FeatureExtractor,
    STYLO_FEATURES,
    PT_STOPWORDS,
    normalize_text,
)

# Human-readable labels for the 6 clickbait archetypes discovered in the corpus
DEFAULT_CLUSTER_TOPICS: dict[int, str] = {
    0: "Polícia & Casos Chocantes",
    1: "Segredos & Estilo de Vida",
    2: "Ciência, Saúde & Mistérios",
    3: "Esportes & Competições",
    4: "Celebridades & Entretenimento",
    5: "Urgência & Curiosidade Sensacionalista",
}


def build_supervised_pipeline(
    ngram_range: tuple[int, int] = (1, 2),
    max_features: int = 5000,
    min_df: int = 2,
    sublinear_tf: bool = True,
    style_weight: float = 0.1,
    c_param: float = 1.0,
    cv_folds: int = 5,
    random_state: int = 42,
) -> Pipeline:
    """
    Builds the Champion Approach C Supervised Pipeline:
    1. FeatureExtractor: transforms raw text/DataFrame into text_clean and 6 stylometric features
    2. ColumnTransformer: TF-IDF (weight 1.0) on text_clean + StandardScaler (weight 0.1) on stylometric features
    3. LinearSVC wrapped in CalibratedClassifierCV(method='sigmoid', cv=cv_folds)
    """
    hybrid_prep = ColumnTransformer(
        transformers=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=ngram_range,
                    max_features=max_features,
                    min_df=min_df,
                    sublinear_tf=sublinear_tf,
                ),
                "text_clean",
            ),
            (
                "estilo",
                StandardScaler(),
                STYLO_FEATURES,
            ),
        ],
        transformer_weights={"tfidf": 1.0, "estilo": style_weight},
    )

    base_hybrid_estimator = Pipeline([
        ("prep", hybrid_prep),
        (
            "clf",
            LinearSVC(
                C=c_param,
                random_state=random_state,
                max_iter=10000,
            ),
        ),
    ])

    calibrated = CalibratedClassifierCV(
        estimator=base_hybrid_estimator,
        method="sigmoid",
        cv=cv_folds,
    )

    return Pipeline([
        ("extract", FeatureExtractor()),
        ("model", calibrated),
    ])


class SupervisedPredictor:
    """
    Self-contained inference engine for supervised clickbait detection.
    Completely decoupled from any clustering or topic models.
    """

    def __init__(self, model_path: Optional[str] = None, model: Optional[Any] = None) -> None:
        self.model = model
        self._init_fast_path()
        if model_path:
            self.load(model_path)

    def load(self, model_path: str) -> "SupervisedPredictor":
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Supervised model not found at: {model_path}")
        self.model = joblib.load(model_path)
        self._init_fast_path()
        return self

    def _init_fast_path(self) -> None:
        self._fast_ready = False
        try:
            if self.model is not None and hasattr(self.model, "named_steps"):
                extractor = self.model.named_steps.get("extract")
                cal_model = self.model.named_steps.get("model")
                if extractor and cal_model and hasattr(cal_model, "calibrated_classifiers_"):
                    ccs = cal_model.calibrated_classifiers_
                    self._fast_extractor = extractor
                    self._fast_tfidfs = []
                    self._fast_means = []
                    self._fast_scales = []
                    self._fast_coef_tfidfs = []
                    self._fast_coef_stylos = []
                    self._fast_intercepts = []
                    self._fast_calibrators = []

                    for cc in ccs:
                        prep = cc.estimator.named_steps["prep"]
                        clf = cc.estimator.named_steps["clf"]
                        tfidf = prep.named_transformers_["tfidf"]
                        estilo = prep.named_transformers_["estilo"]
                        coef = clf.coef_[0]

                        self._fast_tfidfs.append(tfidf)
                        self._fast_means.append(estilo.mean_)
                        self._fast_scales.append(estilo.scale_)
                        self._fast_coef_tfidfs.append(coef[:5000])
                        self._fast_coef_stylos.append(coef[5000:])
                        self._fast_intercepts.append(clf.intercept_[0])
                        self._fast_calibrators.append(cc.calibrators[0])

                    self._fast_n_ccs = len(ccs)
                    self._fast_ready = True
        except Exception:
            self._fast_ready = False

    def predict(self, texts: Union[str, Sequence[str], pd.DataFrame]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Supervised model is not loaded.")
        if isinstance(texts, str):
            texts = [texts]
        if self._fast_ready and isinstance(texts, (list, tuple)) and len(texts) == 1 and isinstance(texts[0], str):
            probs = self.predict_proba(texts)
            return np.array([int(probs[0, 1] >= 0.50)])
        return self.model.predict(texts)

    def predict_proba(self, texts: Union[str, Sequence[str], pd.DataFrame]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Supervised model is not loaded.")
        if isinstance(texts, str):
            texts = [texts]

        if self._fast_ready and isinstance(texts, (list, tuple)) and len(texts) == 1 and isinstance(texts[0], str):
            text = texts[0]
            df = self._fast_extractor.transform([text])
            clean = df["text_clean"].iloc[0]
            raw_stylo = df[["excl_count", "quest_presence", "char_count", "trigger_count", "upper_ratio", "has_num"]].to_numpy()[0]

            probs = 0.0
            for i in range(self._fast_n_ccs):
                xt = self._fast_tfidfs[i].transform([clean])
                scaled_stylo = (raw_stylo - self._fast_means[i]) / self._fast_scales[i] * 0.1
                df_val = xt.dot(self._fast_coef_tfidfs[i]) + np.dot(scaled_stylo, self._fast_coef_stylos[i]) + self._fast_intercepts[i]
                prob = self._fast_calibrators[i].predict(np.array([df_val]))[0]
                probs += prob

            prob_clickbait = probs / self._fast_n_ccs
            return np.array([[1.0 - prob_clickbait, prob_clickbait]])

        return self.model.predict_proba(texts)

    def predict_single(self, text: str, threshold: float = 0.50) -> dict[str, Any]:
        """
        Runs inference on a single headline and returns structured metrics.
        """
        probs = self.predict_proba([text])[0]
        prob_clickbait = float(probs[1])
        is_clickbait = bool(prob_clickbait >= threshold)

        # Distance from decision boundary normalized to [0, 1]
        confidence = round(float(abs(prob_clickbait - 0.5) * 2), 4)
        # Research notebook uncertainty zone: [0.10 <= prob < 0.85]
        uncertain = bool(0.10 <= prob_clickbait < 0.85)

        return {
            "headline": text,
            "is_clickbait": is_clickbait,
            "clickbait_status": "clickbait" if is_clickbait else "not_clickbait",
            "probability": round(prob_clickbait, 4),
            "confidence": confidence,
            "uncertain": uncertain,
        }


# Archetype definitions for mapping cluster keywords to human-readable categories
ARCHETYPE_DEFINITIONS: list[tuple[str, list[str]]] = [
    ("Polícia & Casos Chocantes", ["polícia", "crime", "cidade", "detalhes", "operação", "policial", "preso", "prisão", "tragédia", "investigação"]),
    ("Esportes & Competições", ["time", "futebol", "jogador", "campeonato", "estádio", "estrela", "técnico", "equipe", "clube", "copa"]),
    ("Ciência, Saúde & Mistérios", ["cientistas", "descoberta", "saúde", "cura", "estudo", "médicos", "remédio", "nasa", "pesquisadores"]),
    ("Segredos & Estilo de Vida", ["segredo", "descubra segredo", "dicas", "vida", "economizar", "hábito", "milionários", "dinheiro"]),
    ("Urgência & Curiosidade Sensacionalista", ["vai acreditar", "acreditar", "vai mudar", "mudar", "chocante", "ninguém esperava", "revelado"]),
    ("Política & Escândalos", ["escândalo", "governo", "promete", "política", "presidente", "lei", "decisão", "revelação"]),
]


def _match_topic_name(keywords: list[str], used_names: set[str]) -> str:
    """Matches cluster keywords to the most descriptive unused archetype."""
    best_name = None
    best_score = -1

    for name, terms in ARCHETYPE_DEFINITIONS:
        if name in used_names:
            continue
        score = sum(1 for kw in keywords for t in terms if t in kw or kw in t)
        if score > best_score:
            best_score = score
            best_name = name

    if best_name and best_score > 0:
        return best_name

    # Fallback to any unused name
    for name, _ in ARCHETYPE_DEFINITIONS:
        if name not in used_names:
            return name

    # Default formatted from top keywords
    if keywords:
        return f"Tendências: {keywords[0].capitalize()}"
    return "Sensacionalismo Geral"


def load_topics_from_csv(
    csv_path: str,
    model_name: str = "K-Means esférico",
) -> dict[int, str]:
    """
    Parses temas_por_cluster.csv from entrega artifacts and constructs
    a dictionary of cluster_id -> human-readable typology title.
    Maps the 6 descriptive Portuguese typologies for K-Means esférico:
      0: "Celebridades & Entretenimento"
      1: "Urgência & Curiosidade Sensacionalista"
      2: "Polícia & Casos Chocantes"
      3: "Esportes & Competições"
      4: "Ciência, Saúde & Mistérios"
      5: "Política, Economia & Sociedade"
    """
    if not os.path.exists(csv_path):
        return {}

    PREDEFINED_TITLES: dict[str, dict[int, str]] = {
        "K-Means esférico": {
            0: "Segredos & Estilo de Vida",
            1: "Ciência, Saúde & Mistérios",
            2: "Esportes & Competições",
            3: "Polícia & Casos Chocantes",
            4: "Urgência & Curiosidade Sensacionalista",
            5: "Política & Escândalos",
        },
        "HDBSCAN": {
            0: "Esportes & Competições",
            1: "Celebridades & Entretenimento",
            2: "Viagens & Turismo",
            3: "Polícia & Casos Chocantes",
            4: "Turismo & Destinos Internacionais",
            5: "Economia Nacional & Sociedade",
            6: "Medicina & Vacinas",
            7: "Saúde & Alimentação",
            8: "Curiosidades Científicas",
            9: "Tecnologia & Gadgets",
            10: "Política & Escândalos",
            11: "Economia & Investimentos",
            12: "Ciência & Descobertas Naturais",
            13: "Espaço, Clima & Planeta",
            -1: "Sensacionalismo Geral",
        },
    }

    try:
        df = pd.read_csv(csv_path)
        if "modelo" in df.columns and "cluster" in df.columns:
            df_model = df[df["modelo"] == model_name]
            if not df_model.empty:
                mapping = PREDEFINED_TITLES.get(model_name, {})
                has_keywords = "tema (top-palavras)" in df_model.columns
                topics: dict[int, str] = {}
                used_names: set[str] = set()
                for _, row in df_model.iterrows():
                    c = int(row["cluster"])
                    if has_keywords and pd.notna(row["tema (top-palavras)"]):
                        kws = [k.strip() for k in str(row["tema (top-palavras)"]).split(",") if k.strip()]
                        name = _match_topic_name(kws, used_names)
                        topics[c] = name
                        used_names.add(name)
                    else:
                        topics[c] = mapping.get(c, f"Cluster {c}")
                if topics:
                    return topics
    except Exception:
        pass

    return PREDEFINED_TITLES.get(model_name, {})


class UnsupervisedCategorizer(BaseEstimator):
    """
    Independent Bait Categorization Model using Spherical K-Means:
    TF-IDF (1,2) -> TruncatedSVD (40d) -> L2 Normalizer -> KMeans (k=6).
    Exposes cluster topic labels and keywords per cluster.
    """

    def __init__(
        self,
        n_clusters: int = 6,
        n_components: int = 40,
        max_features: int = 4000,
        random_state: int = 42,
    ) -> None:
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.max_features = max_features
        self.random_state = random_state

        self.vectorizer_ = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=3,
            max_features=self.max_features,
            sublinear_tf=True,
            stop_words=PT_STOPWORDS,
        )
        self.svd_ = TruncatedSVD(n_components=self.n_components, random_state=self.random_state)
        self.normalizer_ = Normalizer(copy=False)
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=15)

        self.cluster_topics: dict[int, str] = {}
        self.cluster_keywords: dict[int, list[str]] = {}

    def fit(self, texts: Sequence[str], y: Any = None) -> "UnsupervisedCategorizer":
        clean_texts = [normalize_text(t) for t in texts]
        Xs = self.vectorizer_.fit_transform(clean_texts)
        Xr = self.svd_.fit_transform(Xs)
        X = self.normalizer_.fit_transform(Xr)
        labels = self.kmeans_.fit_predict(X)

        # Extract top keywords per cluster
        feature_names = np.array(self.vectorizer_.get_feature_names_out())
        self.cluster_keywords = {}
        for c in range(self.n_clusters):
            mask = labels == c
            if np.any(mask):
                cluster_mean = Xs[mask].mean(axis=0)
                top_idx = np.asarray(cluster_mean).ravel().argsort()[::-1][:8]
                self.cluster_keywords[c] = list(feature_names[top_idx])
            else:
                self.cluster_keywords[c] = []

        # Map cluster IDs to human-readable topics based on keywords
        self.cluster_topics = {}
        used_topics: set[str] = set()
        for c in range(self.n_clusters):
            kws = self.cluster_keywords.get(c, [])
            topic_name = _match_topic_name(kws, used_topics)
            self.cluster_topics[c] = topic_name
            used_topics.add(topic_name)

        return self

    def predict(self, texts: Union[str, Sequence[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        clean_texts = [normalize_text(t) for t in texts]
        Xs = self.vectorizer_.transform(clean_texts)
        Xr = self.svd_.transform(Xs)
        X = self.normalizer_.transform(Xr)
        return self.kmeans_.predict(X)

    def categorize(self, text: str) -> dict[str, Any]:
        """
        Categorizes a clickbait headline into its archetype and top keywords.
        """
        cluster_id = int(self.predict([text])[0])
        topic_name = self.cluster_topics.get(cluster_id, "Sensacionalismo Geral")
        keywords = self.cluster_keywords.get(cluster_id, [])
        return {
            "cluster_id": cluster_id,
            "bait_type": topic_name,
            "keywords": keywords,
        }


class ClickbaitDetectionSystem:
    """
    Decoupled Orchestrator integrating Supervised Classifier and Unsupervised Categorizer.
    If the unsupervised model is missing or fails, the supervised classification continues
    normally and sets bait_type to null.
    """

    def __init__(
        self,
        supervised: Optional[SupervisedPredictor] = None,
        unsupervised: Optional[UnsupervisedCategorizer] = None,
    ) -> None:
        self.supervised = supervised
        self.unsupervised = unsupervised

    @classmethod
    def load(
        cls,
        models_dir: str = "models",
        unsup_artifacts_dir: Optional[str] = None,
        load_unsupervised: bool = True,
    ) -> "ClickbaitDetectionSystem":
        sup_path = os.path.join(models_dir, "supervised_model.joblib")
        supervised_predictor = None
        if os.path.exists(sup_path):
            supervised_predictor = SupervisedPredictor(model_path=sup_path)

        unsupervised_categorizer = None
        if load_unsupervised:
            # Candidate paths for the unsupervised model artifact
            candidate_paths: list[str] = []

            # 1. Explicit unsup_artifacts_dir if specified
            if unsup_artifacts_dir:
                candidate_paths.append(os.path.join(unsup_artifacts_dir, "unsupervised_model.joblib"))

            # 2. Prioritize entrega artifacts if models_dir is the default project models directory
            proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            project_models = os.path.abspath(os.path.join(proj_root, "models"))
            is_default_models = (
                models_dir == "models"
                or os.path.abspath(models_dir) == os.path.abspath("models")
                or os.path.abspath(models_dir) == project_models
            )
            if is_default_models:
                candidate_paths.append(
                    os.path.join(proj_root, "modelos_nao_supervisionado_entrega", "artefatos", "unsupervised_model.joblib")
                )
                candidate_paths.append("modelos_nao_supervisionado_entrega/artefatos/unsupervised_model.joblib")

            # 3. Fallback to models_dir
            candidate_paths.append(os.path.join(models_dir, "unsupervised_model.joblib"))

            for path in candidate_paths:
                if os.path.exists(path):
                    try:
                        loaded = joblib.load(path)
                        unsupervised_categorizer = loaded
                        break
                    except Exception:
                        continue

            # Ensure cluster topics are properly resolved:
            # 1. Preserve native cluster_topics from loaded model artifact if present and non-empty.
            # 2. Otherwise load authoritative mapping from cluster_topics.json.
            # 3. Fallback to temas_por_cluster.csv only if topics are still unpopulated.
            if unsupervised_categorizer is not None:
                has_native_topics = (
                    hasattr(unsupervised_categorizer, "cluster_topics")
                    and bool(unsupervised_categorizer.cluster_topics)
                )

                if not has_native_topics:
                    json_candidates: list[Optional[str]] = []
                    if unsup_artifacts_dir:
                        json_candidates.append(os.path.join(unsup_artifacts_dir, "cluster_topics.json"))
                    if is_default_models:
                        json_candidates.append(
                            os.path.join(proj_root, "modelos_nao_supervisionado_entrega", "artefatos", "cluster_topics.json")
                        )
                        json_candidates.append("modelos_nao_supervisionado_entrega/artefatos/cluster_topics.json")
                    json_candidates.append(os.path.join(models_dir, "cluster_topics.json"))

                    for json_path in filter(None, json_candidates):
                        if os.path.exists(json_path):
                            try:
                                with open(json_path, "r", encoding="utf-8") as f:
                                    raw_topics = json.load(f)
                                    unsupervised_categorizer.cluster_topics = {
                                        int(k): str(v) for k, v in raw_topics.items()
                                    }
                                break
                            except Exception:
                                pass

                # Fallback to temas_por_cluster.csv ONLY if cluster_topics is still missing/empty
                if not getattr(unsupervised_categorizer, "cluster_topics", None):
                    csv_candidates: list[Optional[str]] = []
                    if unsup_artifacts_dir:
                        csv_candidates.append(os.path.join(unsup_artifacts_dir, "temas_por_cluster.csv"))
                    if is_default_models:
                        csv_candidates.append(
                            os.path.join(proj_root, "modelos_nao_supervisionado_entrega", "artefatos", "temas_por_cluster.csv")
                        )
                        csv_candidates.append("modelos_nao_supervisionado_entrega/artefatos/temas_por_cluster.csv")
                    csv_candidates.append(os.path.join(models_dir, "temas_por_cluster.csv"))

                    for csv_path in filter(None, csv_candidates):
                        if os.path.exists(csv_path):
                            try:
                                topics = load_topics_from_csv(csv_path)
                                if topics and hasattr(unsupervised_categorizer, "cluster_topics"):
                                    unsupervised_categorizer.cluster_topics = {
                                        int(k): str(v) for k, v in topics.items()
                                    }
                            except Exception:
                                pass
                            break

                # Ensure all cluster_topics keys are integer typed
                if hasattr(unsupervised_categorizer, "cluster_topics") and unsupervised_categorizer.cluster_topics:
                    unsupervised_categorizer.cluster_topics = {
                        int(k): str(v) for k, v in unsupervised_categorizer.cluster_topics.items()
                    }

        return cls(supervised=supervised_predictor, unsupervised=unsupervised_categorizer)

    def predict(self, headline: str) -> dict[str, Any]:
        """
        Predicts whether a headline is clickbait and determines its bait category.
        Gracefully handles absence or failure of the unsupervised model.
        """
        if self.supervised is None:
            raise RuntimeError("Supervised model is not loaded in ClickbaitDetectionSystem.")

        sup_res = self.supervised.predict_single(headline)
        bait_type = None

        if sup_res["is_clickbait"]:
            if self.unsupervised is not None:
                try:
                    cat_res = self.unsupervised.categorize(headline)
                    bait_type = cat_res.get("bait_type")
                except Exception:
                    bait_type = "Sensacionalismo Geral"
            else:
                bait_type = None

        return {
            "headline": headline,
            "is_clickbait": sup_res["is_clickbait"],
            "clickbait_status": sup_res["clickbait_status"],
            "probability": sup_res["probability"],
            "confidence": sup_res["confidence"],
            "uncertain": sup_res["uncertain"],
            "bait_type": bait_type,
        }
