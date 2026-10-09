#!/usr/bin/env python3
"""
scripts/verify_models.py - Model Verification and Acceptance Test Script.

Validates Acceptance Criterion R1:
- Loads exported artifacts from models/
- Runs inference on diverse test strings (clickbait and legitimate news)
- Asserts schema conformance and prediction validity
- Verifies strict architectural decoupling (supervised functions when unsupervised is absent)
- Exits 0 on success.
"""

import os
import sys
import json
from typing import List, Dict, Any

import joblib
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline import (
    SupervisedPredictor,
    UnsupervisedCategorizer,
    ClickbaitDetectionSystem,
)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
SUP_MODEL_PATH = os.path.join(MODELS_DIR, "supervised_model.joblib")
UNSUP_MODEL_PATH = os.path.join(MODELS_DIR, "unsupervised_model.joblib")
METADATA_PATH = os.path.join(MODELS_DIR, "metadata.json")
TOPICS_PATH = os.path.join(MODELS_DIR, "cluster_topics.json")

TEST_CASES = [
    {
        "headline": "Você não vai acreditar no que este cientista descobriu sobre o café!",
        "expected_clickbait": True,
        "description": "Clickbait clássico com gatilho de curiosidade e exclamação",
    },
    {
        "headline": "O pesadelo dos investidores: a decisão do Banco Central que vai destruir seus lucros!",
        "expected_clickbait": True,
        "description": "Clickbait sensacionalista financeiro com gatilho 'pesadelo' e exclamação",
    },
    {
        "headline": "Descubra o segredo dos milionários para economizar todo mês",
        "expected_clickbait": True,
        "description": "Clickbait de estilo de vida com gatilho 'descubra' e 'segredo'",
    },
    {
        "headline": "Banco Central mantém a taxa Selic em 10,5% ao ano",
        "expected_clickbait": False,
        "description": "Notícia factual factual com números e tom neutro",
    },
    {
        "headline": "Ministério da Fazenda divulga relatório bimestral de receitas e despesas",
        "expected_clickbait": False,
        "description": "Notícia governamental factual sem adjetivos",
    },
    {
        "headline": "Governo anuncia novas medidas econômicas nesta terça-feira",
        "expected_clickbait": False,
        "description": "Notícia padrão de política econômica",
    },
]


def test_artifact_existence() -> None:
    print("🧪 [Test 1/6] Verifying presence of required model artifacts...")
    for path, name in [
        (SUP_MODEL_PATH, "supervised_model.joblib"),
        (UNSUP_MODEL_PATH, "unsupervised_model.joblib"),
        (METADATA_PATH, "metadata.json"),
        (TOPICS_PATH, "cluster_topics.json"),
    ]:
        assert os.path.exists(path), f"Required artifact missing: {path}"
        size_kb = os.path.getsize(path) / 1024
        print(f"   ✔ Found {name} ({size_kb:.1f} KB)")
    print("   ✅ Artifact existence verified.\n")


def test_metadata_integrity() -> None:
    print("🧪 [Test 2/6] Verifying metadata.json integrity and metrics...")
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert "supervised_metrics" in meta, "Missing 'supervised_metrics' in metadata"
    metrics = meta["supervised_metrics"]
    print(f"   Model Type: {meta.get('model_type')}")
    print(f"   Test F1:        {metrics.get('test_f1'):.4f}")
    print(f"   Test Accuracy:  {metrics.get('test_accuracy'):.4f}")
    print(f"   Test Precision: {metrics.get('test_precision'):.4f}")
    print(f"   Test Recall:    {metrics.get('test_recall'):.4f}")
    print(f"   Test ROC-AUC:   {metrics.get('test_roc_auc'):.4f}")

    assert metrics.get("test_f1", 0) >= 0.95, f"F1 score {metrics.get('test_f1')} is below 0.95"
    print("   ✅ Metadata integrity and performance threshold verified.\n")


def test_direct_joblib_loading() -> None:
    print("🧪 [Test 3/6] Testing direct joblib.load on exported artifacts...")
    sup_model = joblib.load(SUP_MODEL_PATH)
    assert hasattr(sup_model, "predict"), "Supervised model missing predict method"
    assert hasattr(sup_model, "predict_proba"), "Supervised model missing predict_proba method"

    # Test direct raw string inference
    preds = sup_model.predict(["Teste de notícia informativa"])
    probs = sup_model.predict_proba(["Teste de notícia informativa"])
    assert isinstance(preds, np.ndarray), "predict did not return np.ndarray"
    assert isinstance(probs, np.ndarray), "predict_proba did not return np.ndarray"
    assert probs.shape[1] == 2, f"predict_proba shape expected (N, 2), got {probs.shape}"

    unsup_model = joblib.load(UNSUP_MODEL_PATH)
    assert hasattr(unsup_model, "predict"), "Unsupervised model missing predict method"
    clusters = unsup_model.predict(["Descubra o segredo surpreendente!"])
    assert isinstance(clusters, np.ndarray), "unsupervised predict did not return np.ndarray"
    print("   ✅ Direct joblib loading and base inference verified.\n")


def test_independent_predictors() -> None:
    print("🧪 [Test 4/6] Testing independent SupervisedPredictor and UnsupervisedCategorizer classes...")
    sup = SupervisedPredictor(model_path=SUP_MODEL_PATH)
    unsup = UnsupervisedCategorizer()
    unsup_loaded = joblib.load(UNSUP_MODEL_PATH)

    # Supervised independent test
    res = sup.predict_single("Você não vai acreditar nisso!")
    assert res["is_clickbait"] is True
    assert 0.0 <= res["probability"] <= 1.0

    # Unsupervised independent test
    cat = unsup_loaded.categorize("Jogador do time é flagrado em escândalo!")
    assert "cluster_id" in cat
    assert "bait_type" in cat
    assert cat["bait_type"] == "Rivalidade Esportiva", (
        f"Expected 'Rivalidade Esportiva', got '{cat['bait_type']}'"
    )
    print(f"   Categorization sample: '{cat['bait_type']}' (Cluster {cat['cluster_id']})")
    print("   ✅ Independent predictor interfaces verified.\n")


def test_system_orchestration() -> None:
    print("🧪 [Test 5/6] Testing ClickbaitDetectionSystem on full test benchmark...")
    system = ClickbaitDetectionSystem.load(models_dir=MODELS_DIR)

    all_passed = True
    for idx, case in enumerate(TEST_CASES, 1):
        headline = case["headline"]
        expected = case["expected_clickbait"]
        res = system.predict(headline)

        is_cb = res["is_clickbait"]
        status = res["clickbait_status"]
        prob = res["probability"]
        bait_type = res["bait_type"]

        # Validate schema
        assert isinstance(is_cb, bool), f"is_clickbait must be bool, got {type(is_cb)}"
        assert status in ("clickbait", "not_clickbait"), f"Invalid status: {status}"
        assert 0.0 <= prob <= 1.0, f"Probability out of range: {prob}"

        if is_cb:
            assert bait_type is not None, "Clickbait headline must have bait_type"
        else:
            assert bait_type is None, "Non-clickbait headline must have bait_type=None"

        match = (is_cb == expected)
        icon = "✔" if match else "✖"
        print(f"   [{idx}/{len(TEST_CASES)}] {icon} {case['description']}")
        print(f"       Headline:    '{headline}'")
        print(f"       Clickbait:   {is_cb} (prob: {prob:.1%}, conf: {res['confidence']:.2f})")
        print(f"       Bait Type:   {bait_type}")

        if not match:
            all_passed = False

    # Explicit semantic topic assertions (preventing topic inversion)
    print("   🔬 Testing semantic category alignment...")
    res_sports = system.predict("Chocante: Jogador do time de futebol foi suspenso do campeonato!")
    assert res_sports["is_clickbait"] is True
    assert res_sports["bait_type"] == "Rivalidade Esportiva", (
        f"Semantic inversion detected! Expected 'Rivalidade Esportiva', got '{res_sports['bait_type']}'"
    )
    print(f"       ✔ Sports headline -> '{res_sports['bait_type']}'")

    res_police = system.predict("Urgente: Operação da polícia prende quadrilha após investigação brutal na cidade!")
    assert res_police["is_clickbait"] is True
    assert res_police["bait_type"] == "Choque e Morbidez", (
        f"Semantic inversion detected! Expected 'Choque e Morbidez', got '{res_police['bait_type']}'"
    )
    print(f"       ✔ Police headline -> '{res_police['bait_type']}'")

    assert all_passed, "One or more benchmark test cases failed expectation"
    print("   ✅ System orchestration and benchmark passed.\n")


def test_strict_decoupling_fault_tolerance() -> None:
    print("🧪 [Test 6/6] Testing strict decoupling: Supervised works when Unsupervised is absent...")
    # Load system with unsupervised explicitly disabled/None
    sup_only_system = ClickbaitDetectionSystem.load(models_dir=MODELS_DIR, load_unsupervised=False)
    assert sup_only_system.unsupervised is None, "Unsupervised model should be None"

    headline = "Você não vai acreditar no que este cientista descobriu sobre o café!"
    res = sup_only_system.predict(headline)

    assert res["is_clickbait"] is True, "Supervised detection must work normally"
    assert res["bait_type"] is None, "bait_type must gracefully fall back to None when unsupervised is absent"
    assert 0.0 <= res["probability"] <= 1.0, "Probability must remain valid"

    print("   ✔ Successfully ran supervised prediction with missing unsupervised model")
    print(f"     Result: is_clickbait={res['is_clickbait']}, probability={res['probability']:.1%}, bait_type={res['bait_type']}")
    print("   ✅ Strict decoupling fault tolerance verified.\n")


def main() -> int:
    print("=" * 70)
    print("🔍 VERIFYING CLICKBAIT DETECTION & CATEGORIZATION MODELS")
    print("=" * 70 + "\n")

    try:
        test_artifact_existence()
        test_metadata_integrity()
        test_direct_joblib_loading()
        test_independent_predictors()
        test_system_orchestration()
        test_strict_decoupling_fault_tolerance()

        print("=" * 70)
        print("🎉 ALL ACCEPTANCE TESTS PASSED SUCCESSFULLY! (Acceptance Criterion R1: SATISFIED)")
        print("=" * 70)
        return 0
    except AssertionError as err:
        print(f"\n❌ VERIFICATION FAILED: {err}", file=sys.stderr)
        return 1
    except Exception as err:
        print(f"\n💥 UNEXPECTED ERROR: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
