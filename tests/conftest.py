"""Shared test configuration, fixtures, sample datasets, and contract validators.
Authoritative source: ORIGINAL_REQUEST.md & PROJECT.md
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Project directories. models/scripts/app live under api/ (PROJECT_ROOT),
# but extension/, docs/ and hf_space/ live at the repo root (one level up).
REPO_ROOT = PROJECT_ROOT.parent
MODELS_DIR = PROJECT_ROOT / "models"
EXTENSION_DIR = REPO_ROOT / "extension"
DOCS_DIR = REPO_ROOT / "docs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
HF_SPACE_DIR = REPO_ROOT / "hf_space"

# -----------------------------------------------------------------------------
# Curated Test Datasets (Derived from Brazilian news & clickbait datasets)
# -----------------------------------------------------------------------------

SAMPLE_CLICKBAIT_HEADLINES: List[str] = [
    "Você não vai acreditar no que aconteceu com esse famoso!",
    "O segredo chocante que os médicos não querem que você descubra!",
    "Veja 10 fotos inacreditáveis que vão mudar a sua vida para sempre!",
    "URGENTE: Saiba agora toda a verdade sobre a polêmica do momento!",
    "Isso é o que acontece com seu corpo se você comer isso todos os dias!",
    "Ela pensou que ninguém estava olhando e fez isso na frente de todos!",
    "O que essa celebridade fez nos bastidores chocou até a produção!",
    "Descubra o truque simples para dobrar sua renda em menos de uma semana!",
    "Ninguém esperava por isso: revelação bombástica agita as redes sociais!",
    "Você comete este erro gravíssimo todos os dias sem perceber!",
]

SAMPLE_LEGITIMATE_HEADLINES: List[str] = [
    "Ministério da Fazenda divulga balanço fiscal do terceiro trimestre nesta terça-feira",
    "Banco Central eleva taxa Selic para 11,25% ao ano em reunião do Copom",
    "Prefeitura de São Paulo inicia campanha de vacinação contra a gripe nas UBSs",
    "Cientistas descobrem fóssil inédito de réptil marinho no interior do Paraná",
    "Ibovespa fecha em alta de 0,85% impulsionado por ações dos setores elétrico e bancário",
    "Frente fria traz chuva moderada e derruba temperaturas no Sul e Sudeste",
    "Petrobras assina contrato de fornecimento de gás natural com distribuidora local",
    "Unicamp abre inscrições para o vestibular do próximo ano letivo",
    "Seleção Brasileira treina com elenco completo antes de partida pelas Eliminatórias",
    "Agência Nacional de Energia Elétrica mantém bandeira tarifária verde para o mês",
]

ADVERSARIAL_HEADLINES: List[Dict[str, Any]] = [
    {"name": "empty_string", "text": "", "expected_valid": False},
    {"name": "whitespace_only", "text": "    ", "expected_valid": False},
    {"name": "newlines_tabs", "text": "\t\n\r  \n", "expected_valid": False},
    {"name": "single_char", "text": "A", "expected_valid": False},
    {"name": "short_text_below_threshold", "text": "Notícia curta!", "expected_valid": False},
    {"name": "punctuation_flood", "text": "O QUE HOUVE???!!!!! NINGUÉM ESPERAVA POR ISSO!!!!!", "expected_valid": True},
    {"name": "all_uppercase", "text": "GOVERNO ANUNCIA NOVAS MEDIDAS ECONÔMICAS NESTA TERÇA-FEIRA", "expected_valid": True},
    {"name": "all_lowercase", "text": "governo anuncia novas medidas econômicas nesta terça-feira", "expected_valid": True},
    {"name": "emojis_and_symbols", "text": "😱🔥 URGENTE: Você não vai acreditar no que aconteceu com esse famoso! 👀🚨", "expected_valid": True},
    {"name": "html_xss_attempt", "text": "<script>alert('xss')</script> Veja o que aconteceu no evento oficial!", "expected_valid": True},
    {"name": "sql_injection_attempt", "text": "Headline' OR '1'='1' -- Segredo revelado pelo tribunal", "expected_valid": True},
    {"name": "control_characters", "text": "Notícia com caracteres \x00 nulos e \x1b escapes de terminal", "expected_valid": True},
    {"name": "extreme_length_5000", "text": "Manchete longa: " + ("notícia relevante " * 300), "expected_valid": True},
]

REAL_WORLD_E2E_DATASET: List[Dict[str, Any]] = [
    {"text": "Você não vai acreditar no que aconteceu com esse famoso!", "is_clickbait": True, "category": "gossip"},
    {"text": "Veja as fotos chocantes do antes e depois desta celebridade!", "is_clickbait": True, "category": "sensationalism"},
    {"text": "O método secreto que os ricos usam para pagar menos impostos!", "is_clickbait": True, "category": "finance_bait"},
    {"text": "Isso vai fazer você nunca mais querer comer pão na vida!", "is_clickbait": True, "category": "health_bait"},
    {"text": "URGENTE: O que aconteceu hoje deixou a internet em choque absoluto!", "is_clickbait": True, "category": "urgency"},
    {"text": "Ministério da Fazenda divulga balanço fiscal do terceiro trimestre nesta terça-feira", "is_clickbait": False, "category": "economy"},
    {"text": "Banco Central eleva taxa Selic para 11,25% ao ano em reunião do Copom", "is_clickbait": False, "category": "economy"},
    {"text": "Prefeitura de São Paulo inicia campanha de vacinação contra a gripe nas UBSs", "is_clickbait": False, "category": "health"},
    {"text": "Cientistas descobrem fóssil inédito de réptil marinho no interior do Paraná", "is_clickbait": False, "category": "science"},
    {"text": "Frente fria traz chuva moderada e derruba temperaturas no Sul e Sudeste", "is_clickbait": False, "category": "weather"},
]

# -----------------------------------------------------------------------------
# Component Availability Probes (Supports Progressive Testability)
# -----------------------------------------------------------------------------

def has_m1_model_artifacts() -> bool:
    """Checks if Milestone 1 model artifacts have been serialized to disk."""
    supervised_path = MODELS_DIR / "supervised_model.joblib"
    unsupervised_path = MODELS_DIR / "unsupervised_model.joblib"
    return supervised_path.exists() and unsupervised_path.stat().st_size > 0

def has_m1_verify_script() -> bool:
    """Checks if Milestone 1 verification script exists."""
    verify_script = SCRIPTS_DIR / "verify_models.py"
    return verify_script.exists()

def has_m2_app() -> bool:
    """Checks if Milestone 2 app.py exists."""
    app_path = PROJECT_ROOT / "app.py"
    return app_path.exists()

def is_m2_live_api_online(url: str = "http://127.0.0.1:8000") -> bool:
    """Probes whether the local Gradio/FastAPI server is actively listening."""
    import urllib.request
    try:
        req = urllib.request.Request(f"{url}/docs", headers={"User-Agent": "E2ETestProbe/1.0"})
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            return resp.status in (200, 307, 308)
    except Exception:
        # Check root or api/predict
        try:
            req = urllib.request.Request(f"{url}/", headers={"User-Agent": "E2ETestProbe/1.0"})
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                return resp.status in (200, 307, 308)
        except Exception:
            return False

def has_m3_extension_files() -> bool:
    """Checks if Milestone 3 Chrome Extension files exist."""
    manifest = EXTENSION_DIR / "manifest.json"
    content_js = EXTENSION_DIR / "content" / "content.js"
    return manifest.exists() and content_js.exists()

def has_m4_docs_files() -> bool:
    """Checks if Milestone 4 markdown report and diagrams exist."""
    report = DOCS_DIR / "relatorio_desenvolvimento.md"
    return report.exists()

def has_m4_pdf_file() -> bool:
    """Checks if Milestone 4 PDF file exists."""
    pdf = DOCS_DIR / "relatorio_desenvolvimento.pdf"
    return pdf.exists() and pdf.stat().st_size > 10000

# -----------------------------------------------------------------------------
# Interface Contract Validators
# -----------------------------------------------------------------------------

def validate_prediction_schema(data: Any) -> None:
    """Strictly asserts that an API response adheres to the PROJECT.md schema."""
    assert isinstance(data, dict), f"Prediction response must be a dict, got {type(data).__name__}"
    
    # Required keys per PROJECT.md § M2 ↔ M3 Interface Contract
    required_keys = ["headline", "is_clickbait", "clickbait_status", "probability", "confidence", "bait_type"]
    for key in required_keys:
        assert key in data, f"Missing required response key: '{key}' in {data}"
    
    # Field types and bounds
    assert isinstance(data["headline"], str), f"'headline' must be str, got {type(data['headline'])}"
    assert isinstance(data["is_clickbait"], bool), f"'is_clickbait' must be bool, got {type(data['is_clickbait'])}"
    assert data["clickbait_status"] in ("clickbait", "not_clickbait"), (
        f"'clickbait_status' must be 'clickbait' or 'not_clickbait', got '{data['clickbait_status']}'"
    )
    
    assert isinstance(data["probability"], (float, int)), f"'probability' must be float, got {type(data['probability'])}"
    assert 0.0 <= float(data["probability"]) <= 1.0, f"'probability' out of range [0, 1]: {data['probability']}"
    
    assert isinstance(data["confidence"], (float, int)), f"'confidence' must be float, got {type(data['confidence'])}"
    assert 0.0 <= float(data["confidence"]) <= 1.0, f"'confidence' out of range [0, 1]: {data['confidence']}"
    
    if data["is_clickbait"]:
        assert data["clickbait_status"] == "clickbait"
        assert data["bait_type"] is not None and isinstance(data["bait_type"], str) and len(data["bait_type"].strip()) > 0, (
            f"When is_clickbait is True, 'bait_type' must be a non-empty string, got {data['bait_type']}"
        )
    else:
        assert data["clickbait_status"] == "not_clickbait"
        assert data["bait_type"] is None, (
            f"When is_clickbait is False, 'bait_type' must be None, got {data['bait_type']}"
        )

# -----------------------------------------------------------------------------
# Pytest Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def clickbait_headlines() -> List[str]:
    return SAMPLE_CLICKBAIT_HEADLINES

@pytest.fixture
def legitimate_headlines() -> List[str]:
    return SAMPLE_LEGITIMATE_HEADLINES

@pytest.fixture
def adversarial_headlines() -> List[Dict[str, Any]]:
    return ADVERSARIAL_HEADLINES

@pytest.fixture
def real_world_dataset() -> List[Dict[str, Any]]:
    return REAL_WORLD_E2E_DATASET

@pytest.fixture
def contract_validator():
    return validate_prediction_schema
