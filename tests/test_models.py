"""
tests/test_models.py - Comprehensive Unit & Integration Test Suite for ML Models.
Tests feature extraction, supervised classifier, unsupervised categorizer,
strict model decoupling, and adversarial edge cases.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import joblib

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features import (
    normalize_text,
    clean_text,
    prepare_features,
    STYLO_FEATURES,
    SENSATIONALIST_TRIGGERS,
)
from src.pipeline import (
    SupervisedPredictor,
    UnsupervisedCategorizer,
    ClickbaitDetectionSystem,
)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
SUP_PATH = os.path.join(MODELS_DIR, "supervised_model.joblib")
UNSUP_PATH = os.path.join(MODELS_DIR, "unsupervised_model.joblib")
META_PATH = os.path.join(MODELS_DIR, "metadata.json")


# ============================================================================
# Feature Engineering Tests
# ============================================================================

def test_normalize_text_removes_html_urls_and_punctuation():
    raw = "<b>Atenção:</b> Acesse https://site.com @usuario e confira o _segredo_!"
    norm = normalize_text(raw)
    assert "<" not in norm and ">" not in norm
    assert "https" not in norm
    assert "@" not in norm
    assert "!" not in norm
    assert "_" not in norm
    assert "atenção" in norm
    assert "segredo" in norm


def test_clean_text_preserves_discriminative_stopwords():
    # 'você', 'como', 'isso' are in KEEP_STOPWORDS and must be retained
    raw = "Descubra como você pode mudar isso agora"
    cleaned = clean_text(raw)
    assert "você" in cleaned
    assert "como" in cleaned
    assert "isso" in cleaned


def test_prepare_features_schema_and_types():
    texts = [
        "Você não vai acreditar!",
        "Governo anuncia corte de gastos nesta terça-feira com meta de 20 bilhões",
    ]
    df = prepare_features(texts)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "text_clean" in df.columns
    for feat in STYLO_FEATURES:
        assert feat in df.columns

    # First headline has exclamation mark
    assert df.loc[0, "excl_count"] == 1
    # Second headline has number
    assert df.loc[1, "has_num"] == 1


def test_prepare_features_handles_empty_and_special_strings():
    df = prepare_features("")
    assert len(df) == 1
    assert df.loc[0, "char_count"] == 0
    assert df.loc[0, "excl_count"] == 0

    df_none = prepare_features(None)
    assert len(df_none) == 1


# ============================================================================
# Model Loading & Inference Tests
# ============================================================================

@pytest.fixture(scope="module")
def supervised_predictor():
    assert os.path.exists(SUP_PATH), "supervised_model.joblib missing"
    return SupervisedPredictor(model_path=SUP_PATH)


@pytest.fixture(scope="module")
def unsupervised_categorizer():
    assert os.path.exists(UNSUP_PATH), "unsupervised_model.joblib missing"
    return joblib.load(UNSUP_PATH)


@pytest.fixture(scope="module")
def detection_system():
    return ClickbaitDetectionSystem.load(models_dir=MODELS_DIR)


def test_supervised_predictor_predict_and_proba(supervised_predictor):
    headline = "Descubra o segredo chocante dos famosos!"
    pred = supervised_predictor.predict(headline)
    proba = supervised_predictor.predict_proba(headline)

    assert isinstance(pred, np.ndarray)
    assert pred[0] in (0, 1)
    assert isinstance(proba, np.ndarray)
    assert proba.shape == (1, 2)
    assert np.isclose(proba.sum(), 1.0)
    assert proba[0, 1] > 0.80  # Clickbait probability should be high


def test_supervised_predict_single_contract(supervised_predictor):
    factual = "Banco Central eleva taxa de juros para conter inflação em 2026"
    res = supervised_predictor.predict_single(factual)

    assert res["headline"] == factual
    assert res["is_clickbait"] is False
    assert res["clickbait_status"] == "not_clickbait"
    assert 0.0 <= res["probability"] <= 0.20
    assert isinstance(res["confidence"], float)
    assert isinstance(res["uncertain"], bool)


def test_unsupervised_categorizer_outputs(unsupervised_categorizer):
    cb = "Jogador do time é flagrado em escândalo antes do clássico"
    cat = unsupervised_categorizer.categorize(cb)

    assert "cluster_id" in cat
    assert 0 <= cat["cluster_id"] < 6
    assert "bait_type" in cat
    assert isinstance(cat["bait_type"], str)
    assert len(cat["bait_type"]) > 3
    assert "keywords" in cat
    assert isinstance(cat["keywords"], list)
    assert cat["bait_type"] == "Esportes & Competições"


def test_detection_system_end_to_end(detection_system):
    cb_text = "Você não vai acreditar no que aconteceu com esse famoso!"
    res_cb = detection_system.predict(cb_text)

    assert res_cb["is_clickbait"] is True
    assert res_cb["clickbait_status"] == "clickbait"
    assert res_cb["bait_type"] is not None
    assert res_cb["probability"] > 0.90

    fact_text = "Ministério da Fazenda divulga portaria sobre novas regras tributárias"
    res_fact = detection_system.predict(fact_text)

    assert res_fact["is_clickbait"] is False
    assert res_fact["clickbait_status"] == "not_clickbait"
    assert res_fact["bait_type"] is None
    assert res_fact["probability"] < 0.10


def test_semantic_topic_predictions_sports_and_police(detection_system):
    """Verifies that sports and police categories are not inverted in ClickbaitDetectionSystem."""
    sports_headline = "Chocante: Jogador do time de futebol foi suspenso do campeonato!"
    res_sports = detection_system.predict(sports_headline)
    assert res_sports["is_clickbait"] is True
    assert res_sports["bait_type"] == "Esportes & Competições"

    police_headline = "Urgente: Operação da polícia prende quadrilha após investigação brutal na cidade!"
    res_police = detection_system.predict(police_headline)
    assert res_police["is_clickbait"] is True
    assert res_police["bait_type"] == "Polícia & Casos Chocantes"


# ============================================================================
# Strict Decoupling Tests
# ============================================================================

def test_strict_decoupling_supervised_isolated_from_unsupervised(supervised_predictor):
    # System with unsupervised=None
    system_isolated = ClickbaitDetectionSystem(supervised=supervised_predictor, unsupervised=None)

    # Clickbait headline
    headline = "Descubra o segredo inacreditável que vai mudar sua vida!"
    res = system_isolated.predict(headline)

    # Supervised classification must still work flawlessly
    assert res["is_clickbait"] is True
    assert res["clickbait_status"] == "clickbait"
    assert res["probability"] > 0.90
    # bait_type must be gracefully None
    assert res["bait_type"] is None


def test_strict_decoupling_faulty_unsupervised_handler(supervised_predictor):
    class FaultyUnsupervised:
        def categorize(self, text):
            raise RuntimeError("Simulated failure in unsupervised module")

    system_resilient = ClickbaitDetectionSystem(
        supervised=supervised_predictor, unsupervised=FaultyUnsupervised()
    )

    headline = "Veja o que aconteceu com esse cantor famoso!"
    res = system_resilient.predict(headline)

    # Supervised must not crash
    assert res["is_clickbait"] is True
    assert res["probability"] > 0.80
    assert res["bait_type"] == "Sensacionalismo Geral"


# ============================================================================
# Latency & Performance Test
# ============================================================================

def test_inference_latency_under_50ms(detection_system):
    import time
    headline = "Novo reforço de peso surpreende torcida do time!"

    # Warmup
    _ = detection_system.predict(headline)

    # Measure 50 iterations
    t0 = time.perf_counter()
    n_iters = 50
    for _ in range(n_iters):
        _ = detection_system.predict(headline)
    elapsed = (time.perf_counter() - t0) / n_iters

    # Per-headline latency must be well under 50ms
    assert elapsed < 0.050, f"Average latency too high: {elapsed * 1000:.2f} ms"
