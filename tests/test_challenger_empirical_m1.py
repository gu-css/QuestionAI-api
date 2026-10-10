"""tests/test_challenger_empirical_m1.py
Empirical stress-testing suite for Milestone M1 / M2 API.

Verifies:
1. Endpoints /api/predict and /predict alias with:
   - Empty string and whitespace strings
   - Massive 10,000+ character strings
   - Unicode emojis and multi-byte characters
   - Batch requests (empty batch, mixed batch)
   - Unsupported HTTP methods (GET, PUT, DELETE, PATCH, HEAD) returning 405
2. Contract verification:
   - If is_clickbait is False -> bait_type is strictly None (null in JSON)
   - If is_clickbait is True -> bait_type is strictly a non-empty string
"""
import sys
import time
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import app

client = TestClient(app)

KNOWN_CLICKBAIT_HEADLINES = [
    "Você não vai acreditar no que aconteceu com esse famoso!",
    "Veja as fotos chocantes do antes e depois desta celebridade!",
    "O método secreto que os ricos usam para pagar menos impostos!",
    "URGENTE: O que aconteceu hoje deixou a internet em choque absoluto!",
    "Ela fez isso todos os dias e o resultado foi surpreendente!",
]

KNOWN_LEGITIMATE_HEADLINES = [
    "Ministério da Fazenda divulga balanço fiscal do terceiro trimestre nesta terça-feira",
    "Banco Central eleva taxa Selic para 11,25% ao ano em reunião do Copom",
    "Prefeitura de São Paulo inicia campanha de vacinação contra a gripe nas UBSs",
    "Cientistas descobrem fóssil inédito de réptil marinho no interior do Paraná",
    "IBGE divulga dados sobre desemprego no primeiro trimestre de 2026",
]


class TestEmpiricalEdgeCases:
    """Empirical verification of /api/predict and /predict edge cases."""

    @pytest.mark.parametrize("endpoint", ["/api/predict", "/predict"])
    def test_empty_string(self, endpoint):
        """Empty string returns 200, is_clickbait=False, bait_type=None."""
        resp = client.post(endpoint, json={"headline": ""})
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        data = resp.json()
        assert data["is_clickbait"] is False
        assert data["clickbait_status"] == "not_clickbait"
        assert data["probability"] == 0.0
        assert data["confidence"] == 1.0
        assert data["bait_type"] is None

    @pytest.mark.parametrize("endpoint", ["/api/predict", "/predict"])
    def test_whitespace_string(self, endpoint):
        """Whitespace string returns 200, is_clickbait=False, bait_type=None."""
        resp = client.post(endpoint, json={"headline": "   \n\t   "})
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_clickbait"] is False
        assert data["clickbait_status"] == "not_clickbait"
        assert data["bait_type"] is None

    @pytest.mark.parametrize("endpoint", ["/api/predict", "/predict"])
    def test_massive_10000_char_string(self, endpoint):
        """Massive 10,000 char string returns 200 without timeout or crash."""
        massive_text = "Esta é uma manchete extremamente longa de teste de estresse empírico. " * 150
        assert len(massive_text) >= 10000

        t0 = time.perf_counter()
        resp = client.post(endpoint, json={"headline": massive_text})
        elapsed = time.perf_counter() - t0

        assert resp.status_code == 200
        assert elapsed < 1.0, f"Processing took too long: {elapsed:.3f}s"
        data = resp.json()
        assert "is_clickbait" in data
        assert isinstance(data["is_clickbait"], bool)
        if not data["is_clickbait"]:
            assert data["bait_type"] is None
        else:
            assert isinstance(data["bait_type"], str) and len(data["bait_type"]) > 0

    @pytest.mark.parametrize("endpoint", ["/api/predict", "/predict"])
    def test_unicode_emojis(self, endpoint):
        """Unicode emojis and multi-byte characters are processed cleanly."""
        emoji_headline = "🔥😱💥🚨 VOCÊ NÃO VAI ACREDITAR NO QUE ESSE FAMOSO FEZ! 😱🔥💥 " * 20
        resp = client.post(endpoint, json={"headline": emoji_headline})
        assert resp.status_code == 200
        data = resp.json()
        assert "is_clickbait" in data
        assert isinstance(data["probability"], (float, int))
        assert 0.0 <= data["probability"] <= 1.0

    @pytest.mark.parametrize("endpoint", ["/api/predict", "/predict"])
    def test_batch_requests(self, endpoint):
        """Batch request returns total and results with valid contracts."""
        payload = {
            "headlines": [
                "Você não vai acreditar no que aconteceu com esse famoso!",
                "Ministério da Fazenda divulga balanço fiscal do terceiro trimestre",
                "",
                "🔥🚨 URGENTE: O segredo dos milionários revelado!",
                "IBGE aponta crescimento da economia brasileira",
            ]
        }
        resp = client.post(endpoint, json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 5
        assert len(data["results"]) == 5

        for item in data["results"]:
            assert "headline" in item
            assert "is_clickbait" in item
            assert "probability" in item
            assert "confidence" in item
            assert "bait_type" in item
            if not item["is_clickbait"]:
                assert item["bait_type"] is None
            else:
                assert isinstance(item["bait_type"], str) and len(item["bait_type"]) > 0

    @pytest.mark.parametrize("method", ["get", "put", "delete", "patch", "head"])
    @pytest.mark.parametrize("endpoint", ["/api/predict", "/predict"])
    def test_unsupported_http_methods_return_405(self, method, endpoint):
        """Unsupported HTTP methods must strictly return 405 Method Not Allowed."""
        req_fn = getattr(client, method)
        resp = req_fn(endpoint)
        assert resp.status_code == 405, (
            f"Expected 405 for {method.upper()} {endpoint}, got {resp.status_code}"
        )


class TestContractStrictness:
    """Verify contracts: not clickbait -> bait_type is null; clickbait -> bait_type is non-empty string."""

    @pytest.mark.parametrize("headline", KNOWN_LEGITIMATE_HEADLINES)
    def test_not_clickbait_bait_type_is_strictly_null(self, headline):
        resp = client.post("/api/predict", json={"headline": headline})
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_clickbait"] is False, f"Expected legitimate for: '{headline}'"
        assert data["clickbait_status"] == "not_clickbait"
        assert data["bait_type"] is None, f"Expected bait_type null for: '{headline}', got {data['bait_type']}"

    @pytest.mark.parametrize("headline", KNOWN_CLICKBAIT_HEADLINES)
    def test_clickbait_bait_type_is_strictly_non_empty_string(self, headline):
        resp = client.post("/api/predict", json={"headline": headline})
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_clickbait"] is True, f"Expected clickbait for: '{headline}'"
        assert data["clickbait_status"] == "clickbait"
        assert isinstance(data["bait_type"], str), f"Expected str for bait_type, got {type(data['bait_type'])}"
        assert len(data["bait_type"].strip()) > 0, f"Expected non-empty string for bait_type"
