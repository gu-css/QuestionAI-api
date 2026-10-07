"""
tests/test_challenger_m1.py - Empirical Adversarial Challenge Suite for Milestone M1.

Evaluates:
1. Probability Calibration Properties:
   - Boundaries [0.0, 1.0] across degenerate, extreme, and diverse inputs.
   - Monotonicity between raw SVM decision scores and Platt calibrated probabilities.
   - Perturbation monotonicity on clickbait feature intensifiers.
2. Fault Injection & Strict Decoupling:
   - Missing unsupervised model artifact.
   - Zero-byte corrupted artifact.
   - Random byte corrupted artifact.
   - Invalid object types and failing categorization methods.
   - Isolation of SupervisedPredictor from unsupervised clustering.
3. Brazilian Portuguese Linguistic Nuances & Edge Cases:
   - Diacritic variations (accented vs unaccented).
   - Slang and abbreviations.
   - Hard negatives (journalistic drama, disaster alerts, investigative reports).
   - Journalistic questions and question mark punctuation bias.
   - Degenerate inputs (empty, whitespace, emojis, extreme length, injection payloads).
"""

import os
import sys
import shutil
import pytest
import numpy as np
import pandas as pd
import joblib
from scipy.stats import spearmanr

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features import (
    prepare_features,
    normalize_text,
    clean_text,
    SENSATIONALIST_TRIGGERS,
    STYLO_FEATURES,
)
from src.pipeline import (
    SupervisedPredictor,
    UnsupervisedCategorizer,
    ClickbaitDetectionSystem,
)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
SUP_PATH = os.path.join(MODELS_DIR, "supervised_model.joblib")
UNSUP_PATH = os.path.join(MODELS_DIR, "unsupervised_model.joblib")


@pytest.fixture(scope="module")
def supervised_predictor():
    assert os.path.exists(SUP_PATH), "supervised_model.joblib must exist"
    return SupervisedPredictor(model_path=SUP_PATH)


@pytest.fixture(scope="module")
def detection_system():
    assert os.path.exists(SUP_PATH), "supervised_model.joblib must exist"
    return ClickbaitDetectionSystem.load(models_dir=MODELS_DIR)


# ============================================================================
# CHALLENGE SUITE 1: PROBABILITY CALIBRATION PROPERTIES
# ============================================================================

class TestProbabilityCalibration:
    """Empirical verification of probability calibration ranges, consistency and monotonicity."""

    def test_strict_probability_boundaries(self, supervised_predictor):
        """Test that for diverse inputs, probability and confidence stay strictly in [0.0, 1.0]."""
        test_inputs = [
            "",
            " ",
            "    \t\n   ",
            "!",
            "?",
            "...",
            "?!?!?!",
            "1234567890",
            "R$ 500,00",
            "https://globo.com/noticias",
            "<div><b>Título em HTML</b></div>",
            "🔥🚨😱💣",
            "Notícia neutra sobre inflação e economia brasileira",
            "VOCÊ NUNCA VAI ACREDITAR NO QUE ESSE FAMOSO FEZ COM A ESPOSA!!!",
            "A" * 5000,  # Extremely long headline
            "Governo anuncia plano de investimentos em infraestrutura",
            "O escândalo secreto que destruiu a carreira do cantor revelado!",
            "Cientistas desenvolvem vacina contra nova variante da gripe",
            "Veja o antes e depois surpreendente dessa atriz mirim",
            "Banco Central eleva taxa Selic em 0,5 ponto percentual",
        ]

        for text in test_inputs:
            res = supervised_predictor.predict_single(text)
            prob = res["probability"]
            conf = res["confidence"]

            assert not np.isnan(prob), f"Probability is NaN for: {text!r}"
            assert not np.isinf(prob), f"Probability is Inf for: {text!r}"
            assert 0.0 <= prob <= 1.0, f"Probability {prob} out of [0, 1] for: {text!r}"

            assert not np.isnan(conf), f"Confidence is NaN for: {text!r}"
            assert not np.isinf(conf), f"Confidence is Inf for: {text!r}"
            assert 0.0 <= conf <= 1.0, f"Confidence {conf} out of [0, 1] for: {text!r}"

            # Verify underlying predict_proba row sums to 1.0
            probs = supervised_predictor.predict_proba(text)[0]
            assert np.isclose(probs.sum(), 1.0, atol=1e-5), f"Probabilities do not sum to 1.0 for: {text!r}"

    def test_platt_calibration_mathematical_monotonicity_per_fold(self, supervised_predictor):
        """
        Verify that for each individual calibrated classifier fold in CalibratedClassifierCV,
        the calibrated probability is strictly 100% monotonic (Spearman correlation = 1.0)
        with respect to the raw LinearSVC decision function scores.
        Also verify that ensemble correlation exceeds 0.98.
        """
        headlines = [
            "Governo do Estado inaugura nova ponte sobre o rio Tietê",
            "Petrobras registra lucro recorde no terceiro trimestre",
            "Tribunal Superior Eleitoral homologa resultado de eleições municipais",
            "Previsão do tempo indica chuvas fortes no Sul durante o fim de semana",
            "Ministério da Saúde atualiza calendário nacional de vacinação",
            "Empresas de tecnologia anunciam investimentos em inteligência artificial",
            "Novo remédio promete curar insônia em apenas 5 minutos de uso",
            "O que ninguém te contou sobre a dieta milagrosa que secou a barriga das celebridades",
            "Você não vai acreditar no que este político disse em gravação secreta!",
            "CHOCANTE: O segredo bizarro da família real finalmente foi revelado na internet!!!",
            "O pesadelo começou: a decisão inacreditável que vai falir milhares de brasileiros!",
            "URGENTE: Veja agora o escândalo estarrecedor que parou o país inteiro hoje!",
        ]

        calibrated_cv = supervised_predictor.model.named_steps["model"]
        feature_extractor = supervised_predictor.model.named_steps["extract"]
        df_features = feature_extractor.transform(headlines)

        raw_scores_all_folds = []
        for i, calibrated_clf in enumerate(calibrated_cv.calibrated_classifiers_):
            raw_s = calibrated_clf.estimator.decision_function(df_features)
            raw_scores_all_folds.append(raw_s)
            cal_p = calibrated_clf.predict_proba(df_features)[:, 1]
            fold_corr, _ = spearmanr(raw_s, cal_p)
            assert np.isclose(fold_corr, 1.0, atol=1e-4), (
                f"Fold {i} violated strict Platt monotonicity: correlation={fold_corr:.6f}"
            )

        # Ensemble average correlation
        mean_raw_scores = np.mean(raw_scores_all_folds, axis=0)
        calibrated_probs = supervised_predictor.predict_proba(headlines)[:, 1]
        ensemble_corr, _ = spearmanr(mean_raw_scores, calibrated_probs)
        assert ensemble_corr > 0.98, (
            f"Ensemble Platt calibration correlation unexpectedly low: {ensemble_corr:.4f}"
        )

    def test_feature_intensification_monotonicity(self, supervised_predictor):
        """
        Test that intensifying clickbait stylometric features monotonically increases or preserves probability.
        """
        base = "Governo divulga novas regras para declaração do imposto de renda"
        prob_base = supervised_predictor.predict_single(base)["probability"]

        # Step 1: Add sensational trigger word
        step1 = f"{base}: descubra o segredo"
        prob_step1 = supervised_predictor.predict_single(step1)["probability"]

        # Step 2: Add curiosity gap and trigger phrase
        step2 = "Você não vai acreditar: descubra o segredo do imposto de renda!"
        prob_step2 = supervised_predictor.predict_single(step2)["probability"]

        # Step 3: Add strong sensationalism and uppercase
        step3 = "VOCÊ NÃO VAI ACREDITAR: DESCUBRA O SEGREDO CHOCANTE DO IMPOSTO DE RENDA!!!"
        prob_step3 = supervised_predictor.predict_single(step3)["probability"]

        assert prob_base <= prob_step1, f"Step 1 ({prob_step1}) should be >= base ({prob_base})"
        assert prob_step1 < prob_step2, f"Step 2 ({prob_step2}) should exceed Step 1 ({prob_step1})"
        assert prob_step2 <= prob_step3, f"Step 3 ({prob_step3}) should be >= Step 2 ({prob_step2})"

    def test_exclamation_scaling_monotonicity(self, supervised_predictor):
        """
        Test that increasing exclamation marks alone does not decrease clickbait probability.
        """
        sentence = "Veja o que aconteceu com o jogador após a partida"
        probs = [
            supervised_predictor.predict_single(f"{sentence}{'!' * i}")["probability"]
            for i in range(11)
        ]
        # P(0) should be <= P(10)
        assert probs[0] <= probs[-1], f"Adding 10 exclamations decreased prob: {probs[0]} -> {probs[-1]}"
        # Ensure non-negative trend overall
        for i in range(len(probs) - 1):
            assert probs[i] <= probs[i+1] + 1e-4, f"Decreased from {i} to {i+1}: {probs[i]} -> {probs[i+1]}"


# ============================================================================
# CHALLENGE SUITE 2: FAULT INJECTION & STRICT DECOUPLING
# ============================================================================

class TestFaultInjectionDecoupling:
    """Stress tests simulating missing, corrupted, or faulty unsupervised model artifacts."""

    def test_missing_unsupervised_model_file(self, tmp_path):
        """Verify ClickbaitDetectionSystem works when unsupervised_model.joblib does not exist."""
        shutil.copy(SUP_PATH, tmp_path / "supervised_model.joblib")
        shutil.copy(os.path.join(MODELS_DIR, "metadata.json"), tmp_path / "metadata.json")

        assert not (tmp_path / "unsupervised_model.joblib").exists()

        system = ClickbaitDetectionSystem.load(models_dir=str(tmp_path))

        assert system.supervised is not None
        assert system.unsupervised is None

        res_cb = system.predict("Você não vai acreditar no segredo chocante!")
        assert res_cb["is_clickbait"] is True
        assert res_cb["probability"] > 0.90
        assert res_cb["bait_type"] is None, "Missing unsupervised model must set bait_type to None"

        res_fact = system.predict("Governo aprova orçamento anual para a educação")
        assert res_fact["is_clickbait"] is False
        assert res_fact["bait_type"] is None

    def test_corrupted_empty_unsupervised_model_file(self, tmp_path):
        """Verify ClickbaitDetectionSystem gracefully survives a 0-byte corrupted unsupervised model."""
        shutil.copy(SUP_PATH, tmp_path / "supervised_model.joblib")
        (tmp_path / "unsupervised_model.joblib").touch()

        system = ClickbaitDetectionSystem.load(models_dir=str(tmp_path))
        assert system.supervised is not None
        assert system.unsupervised is None

        res = system.predict("Descubra o segredo que os médicos escondem!")
        assert res["is_clickbait"] is True
        assert res["bait_type"] is None

    def test_corrupted_garbage_bytes_unsupervised_model_file(self, tmp_path):
        """Verify ClickbaitDetectionSystem gracefully survives random corrupted binary bytes."""
        shutil.copy(SUP_PATH, tmp_path / "supervised_model.joblib")
        with open(tmp_path / "unsupervised_model.joblib", "wb") as f:
            f.write(b"\x00\xFF\xAA\x55GARBAGE_BINARY_CORRUPTED_DATA_HEADER\x00\x00\x00")

        system = ClickbaitDetectionSystem.load(models_dir=str(tmp_path))
        assert system.supervised is not None
        assert system.unsupervised is None

        res = system.predict("Veja o que aconteceu nessa praia paradisíaca!")
        assert res["is_clickbait"] is True
        assert res["bait_type"] is None

    def test_unsupervised_wrong_object_type(self, tmp_path):
        """Verify ClickbaitDetectionSystem survives when joblib contains an unexpected object."""
        shutil.copy(SUP_PATH, tmp_path / "supervised_model.joblib")
        joblib.dump({"invalid": "model_data"}, tmp_path / "unsupervised_model.joblib")

        system = ClickbaitDetectionSystem.load(models_dir=str(tmp_path))
        assert system.supervised is not None

        res = system.predict("Você não vai acreditar no que este cientista revelou!")
        assert res["is_clickbait"] is True
        assert res["bait_type"] in (None, "Sensacionalismo Geral")

    def test_unsupervised_runtime_exception_resilience(self, supervised_predictor):
        """Verify that an active runtime exception in categorize() never crashes the supervised prediction."""
        class ExplodingCategorizer:
            def categorize(self, text):
                raise ZeroDivisionError("Catastrophic internal failure in cluster distance math")

        system = ClickbaitDetectionSystem(supervised=supervised_predictor, unsupervised=ExplodingCategorizer())

        res = system.predict("Inacreditável: homem encontra tesouro no próprio quintal!")
        assert res["is_clickbait"] is True
        assert res["probability"] > 0.90
        assert res["bait_type"] == "Sensacionalismo Geral"

    def test_supervised_predictor_zero_unsupervised_coupling(self):
        """Assert SupervisedPredictor has zero dependency on unsupervised modules or clusterers."""
        pred = SupervisedPredictor(model_path=SUP_PATH)
        assert not hasattr(pred, "unsupervised")
        assert not hasattr(pred, "cluster_topics")
        assert not hasattr(pred, "categorize")
        assert not isinstance(pred.model, UnsupervisedCategorizer)


# ============================================================================
# CHALLENGE SUITE 3: BRAZILIAN PORTUGUESE LINGUISTIC NUANCES & EDGE CASES
# ============================================================================

class TestBrazilianPortugueseEdgeCases:
    """Stress tests on Brazilian Portuguese grammar, diacritics, slang, and journalistic nuances."""

    @pytest.mark.parametrize("accented,unaccented", [
        ("Você não vai acreditar no que aconteceu!", "Voce nao vai acreditar no que aconteceu!"),
        ("Descubra o segredo chocante revelado hoje", "Descubra o segredo chocante revelado hoje"),
        ("Médicos alertam para sintoma silencioso", "Medicos alertam para sintoma silencioso"),
    ])
    def test_diacritic_robustness_on_standard_triggers(self, supervised_predictor, accented, unaccented):
        """Headlines with major Portuguese triggers retain clickbait status regardless of accents."""
        res_acc = supervised_predictor.predict_single(accented)
        res_unacc = supervised_predictor.predict_single(unaccented)

        assert res_acc["is_clickbait"] == res_unacc["is_clickbait"]
        diff = abs(res_acc["probability"] - res_unacc["probability"])
        assert diff <= 0.15, f"Discrepancy ({diff:.4f}) between '{accented}' and '{unaccented}'"

    def test_diacritic_vulnerability_observation(self, supervised_predictor):
        """
        Document empirical vulnerability: unaccented words like 'escandalo' miss trigger dictionary
        ('escândalo' has circumflex in SENSATIONALIST_TRIGGERS) and TF-IDF feature vocabulary.
        """
        accented = "Veja o escândalo que abalou a política"
        unaccented = "Veja o escandalo que abalou a politica"

        res_acc = supervised_predictor.predict_single(accented)
        res_unacc = supervised_predictor.predict_single(unaccented)

        assert res_acc["is_clickbait"] is True
        assert res_acc["probability"] > 0.70
        # The unaccented variation drops significantly below threshold due to exact accent matching
        assert res_unacc["probability"] < res_acc["probability"]

    def test_hard_negatives_factual_news_with_intense_words(self, supervised_predictor):
        """
        Legitimate news reports containing dramatic words (Polícia, alerta, morte, operação, crise)
        must NOT be falsely flagged as clickbait when structured factually.
        """
        hard_negatives = [
            "Polícia Federal deflagra operação contra desvio de recursos públicos em Rondônia",
            "Ministério da Saúde emite alerta preventivo sobre aumento de casos de dengue no Sudeste",
            "Terremoto de magnitude 7.2 atinge costa do Japão e gera alerta temporário de tsunami",
            "Acidente na Rodovia Presidente Dutra interdita duas faixas no sentido São Paulo",
            "Tribunal Regional Federal condena ex-prefeito a cinco anos de prisão por improbidade",
            "Inflação oficial fecha o ano em 3,92%, dentro da meta estabelecida pelo Banco Central",
            "Cientistas da USP identificam nova proteína ligada à regeneração celular em mamíferos",
            "Ministro da Fazenda participa de reunião com líderes do Congresso para debater reforma",
        ]

        for headline in hard_negatives:
            res = supervised_predictor.predict_single(headline)
            assert not res["is_clickbait"], (
                f"False positive on legitimate factual news: '{headline}' "
                f"(probability: {res['probability']}, status: {res['clickbait_status']})"
            )

    def test_journalistic_quotes_and_factual_declarations(self, supervised_predictor):
        """
        Legitimate journalism frequently uses direct quotes and factual statements.
        Verify that sober statements are reliably classified as not_clickbait.
        """
        factual_quotes = [
            "'Não haverá aumento de impostos neste trimestre', afirma secretário da Fazenda",
            "Diretor do Banco Central declara que corte de juros dependerá do cenário externo",
            "Presidente da República assina decreto que amplia verba da segurança pública",
        ]

        for q in factual_quotes:
            res = supervised_predictor.predict_single(q)
            assert not res["is_clickbait"], (
                f"Legitimate statement falsely flagged as clickbait: '{q}' "
                f"(prob: {res['probability']})"
            )
            assert res["probability"] < 0.35

    def test_question_mark_punctuation_bias_observation(self, supervised_predictor):
        """
        Document empirical finding: dataset contains 99.78% clickbait for headlines with '?'.
        This causes questions to have strong positive shift, which can flag journalistic questions.
        Removing the '?' restores correct non-clickbait classification.
        """
        q_headline = "Por que os juros futuros caíram após a divulgação da ata do Copom?"
        statement_headline = "Por que os juros futuros caíram após a divulgação da ata do Copom"

        res_q = supervised_predictor.predict_single(q_headline)
        res_stmt = supervised_predictor.predict_single(statement_headline)

        # Document the question mark bias impact:
        assert res_q["probability"] > 0.80  # Shifted high due to dataset artifact
        assert res_stmt["probability"] < 0.05  # Without ?, recognizes the sober journalistic content
        assert res_stmt["is_clickbait"] is False

    def test_overt_clickbait_curiosity_triggers(self, supervised_predictor):
        """
        Test that overt curiosity-gap clickbait headlines are decisively detected with high confidence.
        """
        clickbaits = [
            "Você não vai acreditar no que aconteceu com esse famoso!",
            "O pesadelo começou: a decisão inacreditável que vai destruir seus lucros!",
            "Descubra o segredo chocante dos milionários para economizar todo mês",
            "Veja o que aconteceu quando este homem tentou enganar a polícia!",
            "Médicos revelam verdade oculta sobre alimento popular que você consome todo dia!",
        ]

        for cb in clickbaits:
            res = supervised_predictor.predict_single(cb)
            assert res["is_clickbait"] is True, f"Failed on: {cb}"
            assert res["probability"] > 0.90
            assert res["clickbait_status"] == "clickbait"

    def test_extreme_degenerate_inputs_no_crash(self, supervised_predictor, detection_system):
        """
        Adversarial degenerate inputs: empty strings, whitespace, unicode spam, code injection.
        System must handle gracefully without throwing uncaught exceptions.
        """
        degenerate_samples = [
            "",
            "   ",
            "\n\t\r",
            "???",
            "!!!",
            "12345 67890",
            "<script>alert(1)</script>",
            "SELECT * FROM users WHERE '1'='1';",
            "🚨🔥😱👀🎉",
            "a" * 10000,
            "Não " * 200,
            "Você " * 200,
            "BREAKING NEWS IN ENGLISH ONLY WITHOUT PORTUGUESE WORDS",
        ]

        for sample in degenerate_samples:
            res_sup = supervised_predictor.predict_single(sample)
            assert "is_clickbait" in res_sup
            assert "probability" in res_sup
            assert 0.0 <= res_sup["probability"] <= 1.0

            res_sys = detection_system.predict(sample)
            assert "is_clickbait" in res_sys
            assert "clickbait_status" in res_sys
            assert "probability" in res_sys
            assert 0.0 <= res_sys["probability"] <= 1.0
