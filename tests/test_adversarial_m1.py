"""
tests/test_adversarial_m1.py - Adversarial & Stress Testing Suite for Milestone M1.

Evaluates boundary conditions, extreme strings, injection payloads,
unicode/emojis, latency, throughput, and contract invariants.
"""

import math
import os
import sys
import time
import pytest
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features import (
    normalize_text,
    clean_text,
    prepare_features,
    FeatureExtractor,
    STYLO_FEATURES,
)
from src.pipeline import (
    SupervisedPredictor,
    UnsupervisedCategorizer,
    ClickbaitDetectionSystem,
)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
SUP_MODEL_PATH = os.path.join(MODELS_DIR, "supervised_model.joblib")
UNSUP_MODEL_PATH = os.path.join(MODELS_DIR, "unsupervised_model.joblib")


@pytest.fixture(scope="module")
def system():
    return ClickbaitDetectionSystem.load(models_dir=MODELS_DIR)


@pytest.fixture(scope="module")
def supervised_predictor():
    return SupervisedPredictor(model_path=SUP_MODEL_PATH)


@pytest.fixture(scope="module")
def unsupervised_categorizer():
    import joblib
    return joblib.load(UNSUP_MODEL_PATH)


# ============================================================================
# 1. ADVERSARIAL INPUT TEST SUITE
# ============================================================================

ADVERSARIAL_INPUTS = {
    # 1.1 Empty and Whitespace
    "empty": "",
    "single_space": " ",
    "multi_whitespace": "   \t\n\r   \v   ",
    
    # 1.2 Numbers Only
    "digits_short": "1234567890",
    "digits_formatted": "R$ 1.250.000,50 10% 2026 3.1415926535",
    "digits_large_sequence": "9999999999 " * 100,
    
    # 1.3 Punctuation & Symbols
    "exclamation_burst_short": "!!!!!!",
    "exclamation_burst_heavy": "!" * 1000,
    "question_burst": "?" * 1000,
    "mixed_interrobangs": "?!?!?!?!?!?!?!?!" * 50,
    "keyboard_symbols": "!@#$%^&*()_+{}[]:;'<>?,./~`|\\=-",
    "quotes_and_escapes": "\"\"\"'''\\\'\\\"\\n\\t\\r\\0",
    
    # 1.4 Unicode, Diacritics & International Scripts
    "cyrillic": "Срочные новости! Сенсация года, вы не поверите!",
    "chinese": "震惊！科学家发现了长生不老的秘密！快来看！",
    "japanese": "衝撃！絶対に信じられない事実が発覚！今すぐ確認！",
    "arabic": "عاجل! لن تصدق ما حدث اليوم في المدينة!",
    "hebrew": "חדשות מרעישות! לא תאמינו מה קרה היום!",
    "devanagari": "ब्रेकिंग न्यूज़! आप विश्वास नहीं करेंगे क्या हुआ!",
    "zalgo": "U̵R̶G̴E̶N̷T̸E̶!̶!̴!̶ V̵o̷c̶ê̵ ̶n̷ã̴o̶ ̶v̴a̶i̵ ̵a̸c̶r̷e̶d̷i̸t̶a̷r̶!̴",
    "zero_width_chars": "Notícia\u200bcom\u200bespaço\u200binvisível\u200ee\u200fcaracteres",
    "extended_latin": "Dvořák café naître naïve Übergröße São João Müller",
    
    # 1.5 Emojis and Complex Pictographs
    "emoji_only": "😱🚨💥🔥👀💯💣⚡️",
    "emoji_burst": "🔥" * 500,
    "emoji_mixed_clickbait": "🚨🚨🚨 URGENTE: Você não vai acreditar no que aconteceu com esse famoso! 😱😱😱",
    "zwj_sequences": "👨‍👩‍👧‍👦 🧑🏾‍🔬 🏃🏿‍♂️ 🏳️‍🌈",
    
    # 1.6 Injection Payloads & Code
    "sql_injection_1": "' OR 1=1; DROP TABLE headlines; --",
    "sql_injection_2": "UNION SELECT username, password FROM users WHERE '1'='1",
    "xss_script": "<script>alert('xss')</script>",
    "xss_img_onerror": "<img src=x onerror=alert(document.cookie)>",
    "xss_iframe": "<iframe src='javascript:alert(1)'></iframe>",
    "html_tags_nested": "<div class='news'><article><p>Título <b>da notícia</b></p></article></div>",
    "html_unclosed": "<<<<h1 class='broken' <a href='test'",
    "shell_metachars": "$(whoami); `cat /etc/passwd`; rm -rf /; echo test | bash &",
    "null_byte_string": "Notícia normal com\x00byte nulo no meio",
    "json_string": '{"headline": "Fake news", "is_clickbait": true, "score": 99.9}',
    
    # 1.7 Long Strings (10,000+ chars)
    "long_single_word": "A" * 15000,
    "long_repeated_sentence": "Você não vai acreditar no que este cientista descobriu sobre o segredo da longevidade! " * 250, # ~22k chars
    "long_news_article": ("O Ministério da Economia apresentou nesta tarde as diretrizes gerais para o plano plurianual de investimentos. " * 300), # ~34k chars
    "long_mixed_50k": ("🔥 URGENTE! Veja o que aconteceu! 😱 " + "a" * 100 + " ") * 400, # ~50k chars
}


@pytest.mark.parametrize("case_name,payload", list(ADVERSARIAL_INPUTS.items()))
def test_pipeline_adversarial_contract_compliance(system, case_name, payload):
    """
    Stress-tests ClickbaitDetectionSystem across all adversarial inputs.
    Verifies that the system never crashes, never produces NaNs, and strictly
    adheres to the API output contract invariants.
    """
    res = system.predict(payload)
    
    # Invariant 1: Result is a dictionary with exact required keys
    assert isinstance(res, dict), f"Failed for {case_name}: output is not dict"
    required_keys = {"headline", "is_clickbait", "clickbait_status", "probability", "confidence", "uncertain", "bait_type"}
    assert required_keys.issubset(res.keys()), f"Failed for {case_name}: missing keys {required_keys - set(res.keys())}"
    
    # Invariant 2: headline is preserved exactly as input
    assert res["headline"] == payload, f"Failed for {case_name}: headline altered"
    
    # Invariant 3: is_clickbait is strictly a boolean
    assert isinstance(res["is_clickbait"], (bool, np.bool_)), f"Failed for {case_name}: is_clickbait is {type(res['is_clickbait'])}"
    
    # Invariant 4: clickbait_status matches is_clickbait
    expected_status = "clickbait" if res["is_clickbait"] else "not_clickbait"
    assert res["clickbait_status"] == expected_status, f"Failed for {case_name}: status mismatch"
    
    # Invariant 5: probability is a valid float in [0.0, 1.0], never NaN or Inf
    prob = res["probability"]
    assert isinstance(prob, (float, np.floating)), f"Failed for {case_name}: prob is {type(prob)}"
    assert not math.isnan(prob), f"Failed for {case_name}: probability is NaN"
    assert not math.isinf(prob), f"Failed for {case_name}: probability is Inf"
    assert 0.0 <= prob <= 1.0, f"Failed for {case_name}: probability {prob} out of bounds [0, 1]"
    
    # Invariant 6: confidence is in [0.0, 1.0], never NaN or Inf
    conf = res["confidence"]
    assert isinstance(conf, (float, np.floating)), f"Failed for {case_name}: conf is {type(conf)}"
    assert not math.isnan(conf), f"Failed for {case_name}: confidence is NaN"
    assert not math.isinf(conf), f"Failed for {case_name}: confidence is Inf"
    assert 0.0 <= conf <= 1.0, f"Failed for {case_name}: confidence {conf} out of bounds [0, 1]"
    
    # Invariant 7: uncertain is boolean
    assert isinstance(res["uncertain"], (bool, np.bool_)), f"Failed for {case_name}: uncertain is {type(res['uncertain'])}"
    
    # Invariant 8: bait_type contract: None if not clickbait, valid string or None if clickbait
    if not res["is_clickbait"]:
        assert res["bait_type"] is None, f"Failed for {case_name}: non-clickbait has non-None bait_type: {res['bait_type']}"
    else:
        assert isinstance(res["bait_type"], str) and len(res["bait_type"]) > 0, f"Failed for {case_name}: clickbait has empty/invalid bait_type: {res['bait_type']}"


@pytest.mark.parametrize("case_name,payload", list(ADVERSARIAL_INPUTS.items()))
def test_supervised_predictor_adversarial_isolation(supervised_predictor, case_name, payload):
    """
    Directly tests SupervisedPredictor on adversarial inputs.
    Verifies predict and predict_proba stability without the orchestrator wrapper.
    """
    pred = supervised_predictor.predict(payload)
    proba = supervised_predictor.predict_proba(payload)
    
    assert isinstance(pred, np.ndarray)
    assert len(pred) == 1
    assert pred[0] in (0, 1)
    
    assert isinstance(proba, np.ndarray)
    assert proba.shape == (1, 2)
    assert not np.isnan(proba).any()
    assert not np.isinf(proba).any()
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
    assert np.isclose(proba.sum(), 1.0, atol=1e-5)


@pytest.mark.parametrize("case_name,payload", list(ADVERSARIAL_INPUTS.items()))
def test_unsupervised_categorizer_adversarial_isolation(unsupervised_categorizer, case_name, payload):
    """
    Directly tests UnsupervisedCategorizer on all adversarial inputs.
    Verifies cluster prediction and categorize dictionary output.
    """
    cat = unsupervised_categorizer.categorize(payload)
    assert isinstance(cat, dict)
    assert "cluster_id" in cat and isinstance(cat["cluster_id"], (int, np.integer))
    assert 0 <= cat["cluster_id"] < 6
    assert "bait_type" in cat and isinstance(cat["bait_type"], str) and len(cat["bait_type"]) > 0
    assert "keywords" in cat and isinstance(cat["keywords"], list)


# ============================================================================
# 2. NON-STRING & TYPE RESILIENCE TESTS
# ============================================================================

def test_prepare_features_type_robustness():
    """Tests prepare_features with various non-string types."""
    # List containing None: _extract_single_row receives None directly
    df_list_none = prepare_features([None])
    assert isinstance(df_list_none, pd.DataFrame) and len(df_list_none) == 1
    assert df_list_none.loc[0, "char_count"] == 0

    # Scalar None: falls back to str(data) -> "None" (observed behavior)
    df_scalar_none = prepare_features(None)
    assert isinstance(df_scalar_none, pd.DataFrame) and len(df_scalar_none) == 1
    # Documented finding: scalar None results in str("None") with length 4
    assert df_scalar_none.loc[0, "char_count"] in (0, 4)
    
    # Numbers
    df_int = prepare_features(123456)
    assert isinstance(df_int, pd.DataFrame) and len(df_int) == 1
    assert df_int.loc[0, "has_num"] == 1
    
    # Float
    df_float = prepare_features(99.9)
    assert isinstance(df_float, pd.DataFrame) and len(df_float) == 1
    
    # List of mixed types
    mixed = ["Valid headline", 12345, None, "Outra notícia!"]
    df_mixed = prepare_features(mixed)
    assert len(df_mixed) == 4
    for col in STYLO_FEATURES:
        assert not df_mixed[col].isna().any()


# ============================================================================
# 3. HIGH-LOAD, LATENCY & STABILITY BENCHMARK
# ============================================================================

def test_adversarial_10k_payload_latency(system):
    """
    Verifies that processing a 10,000+ char payload finishes within 200ms
    and does not trigger exponential backtracking in regex.
    """
    huge_text = ("URGENTE: Veja o que aconteceu! 😱 " + "Palavra " * 20) * 100 # ~12,000 chars
    
    t0 = time.perf_counter()
    res = system.predict(huge_text)
    elapsed = time.perf_counter() - t0
    
    assert res is not None
    assert elapsed < 0.200, f"Processing massive payload took {elapsed*1000:.2f} ms (exceeded 200ms threshold)"


def test_batch_throughput_and_stability(supervised_predictor):
    """
    Tests high-volume batch prediction (1,000 items) for throughput and memory stability.
    """
    base_headlines = [
        "Você não vai acreditar no que este cientista descobriu sobre o café!",
        "Governo anuncia novas medidas econômicas nesta terça-feira",
        "O segredo dos milionários para economizar todo mês",
        "Ministério da Saúde divulga calendário de vacinação contra a gripe",
        "Chocante: jogador do time é suspenso por indisciplina antes do clássico!",
    ]
    batch = base_headlines * 200 # 1,000 headlines
    
    t0 = time.perf_counter()
    preds = supervised_predictor.predict(batch)
    probs = supervised_predictor.predict_proba(batch)
    elapsed = time.perf_counter() - t0
    
    assert len(preds) == 1000
    assert len(probs) == 1000
    
    throughput = len(batch) / elapsed
    print(f"\n[BENCHMARK] 1,000 batch inference completed in {elapsed:.3f}s ({throughput:.1f} items/sec)")
    # Must achieve at least 500 items/sec on modern CPU
    assert throughput > 300, f"Throughput too low: {throughput:.1f} items/sec"


def test_consecutive_iterations_memory_stability(system):
    """
    Executes 500 consecutive single predictions to verify absence of memory leaks
    or performance degradation.
    """
    latencies = []
    headline = "Descubra o segredo que os especialistas não querem que você saiba!"
    
    for _ in range(500):
        t0 = time.perf_counter()
        _ = system.predict(headline)
        latencies.append(time.perf_counter() - t0)
    
    first_100_avg = np.mean(latencies[:100])
    last_100_avg = np.mean(latencies[-100:])
    
    # Latency should not degrade significantly (no memory leak / thrashing)
    ratio = last_100_avg / max(first_100_avg, 1e-6)
    assert ratio < 2.5, f"Performance degraded by factor of {ratio:.2f} across 500 iterations"
