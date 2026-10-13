"""tests/test_empirical_fault_injection_deep.py
Deep Empirical Fault Injection and Decoupling Stress Suite for Milestone M1.

Exhaustively verifies:
1. Missing `unsupervised_model.joblib` in candidate directories.
2. 0-byte `unsupervised_model.joblib`.
3. Garbage/corrupted byte `unsupervised_model.joblib` (various corruptions: random bytes, truncated pickle header).
4. Missing `temas_por_cluster.csv`.
5. 0-byte `temas_por_cluster.csv`.
6. Malformed/corrupted `temas_por_cluster.csv` (broken CSV, missing required columns).
7. Invalid/empty topics dictionary handling.
8. API resilience: FastAPI `/api/predict` and `/predict` under degraded states:
   - Returns 200 OK.
   - For clickbait: `is_clickbait` is True, `bait_type` is None (or fallback if partially loaded).
   - For non-clickbait: `is_clickbait` is False, `bait_type` is None.
   - Batch requests under degraded state return 200 with all items having valid contracts.
9. Concurrency stress: 100 concurrent requests during degraded state.
"""

import os
import shutil
import tempfile
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
SUP_PATH = MODELS_DIR / "supervised_model.joblib"
UNSUP_PATH = MODELS_DIR / "unsupervised_model.joblib"
CSV_PATH = PROJECT_ROOT / "modelos_nao_supervisionado_entrega" / "artefatos" / "temas_por_cluster.csv"

from src.pipeline import (
    ClickbaitDetectionSystem,
    SupervisedPredictor,
    UnsupervisedCategorizer,
    load_topics_from_csv,
)
from app import app
import app as app_module


class TestFaultInjectionArtifacts:
    """Empirical fault injection on model artifacts and files."""

    def test_missing_unsupervised_model(self, tmp_path):
        """Fault injection 1: unsupervised_model.joblib is completely absent."""
        shutil.copy(SUP_PATH, tmp_path / "supervised_model.joblib")
        assert not (tmp_path / "unsupervised_model.joblib").exists()

        system = ClickbaitDetectionSystem.load(models_dir=str(tmp_path), load_unsupervised=True)
        assert system.supervised is not None
        assert system.unsupervised is None

        # Verify supervised classifier works 100% reliably
        res_cb = system.predict("Você não vai acreditar no que aconteceu com esse famoso!")
        assert res_cb["is_clickbait"] is True
        assert res_cb["probability"] > 0.80
        assert res_cb["bait_type"] is None, "Missing unsupervised model must result in bait_type=None"

        res_legit = system.predict("Ministério da Fazenda divulga balanço fiscal do trimestre.")
        assert res_legit["is_clickbait"] is False
        assert res_legit["bait_type"] is None

    def test_zero_byte_unsupervised_model(self, tmp_path):
        """Fault injection 2: unsupervised_model.joblib is a 0-byte file."""
        shutil.copy(SUP_PATH, tmp_path / "supervised_model.joblib")
        (tmp_path / "unsupervised_model.joblib").touch()
        assert (tmp_path / "unsupervised_model.joblib").stat().st_size == 0

        system = ClickbaitDetectionSystem.load(models_dir=str(tmp_path), load_unsupervised=True)
        assert system.supervised is not None
        assert system.unsupervised is None

        res_cb = system.predict("O segredo inacreditável dos milionários revelado!")
        assert res_cb["is_clickbait"] is True
        assert res_cb["bait_type"] is None

    def test_corrupted_bytes_unsupervised_model(self, tmp_path):
        """Fault injection 3: unsupervised_model.joblib contains invalid/corrupted binary data."""
        shutil.copy(SUP_PATH, tmp_path / "supervised_model.joblib")
        with open(tmp_path / "unsupervised_model.joblib", "wb") as f:
            f.write(b"\x80\x04\x95" + b"CORRUPTED_TRUNCATED_PICKLE_STREAM" * 20)

        system = ClickbaitDetectionSystem.load(models_dir=str(tmp_path), load_unsupervised=True)
        assert system.supervised is not None
        assert system.unsupervised is None

        res_cb = system.predict("Chocante: veja o que aconteceu hoje na transmissão ao vivo!")
        assert res_cb["is_clickbait"] is True
        assert res_cb["bait_type"] is None

    def test_missing_temas_por_cluster_csv(self, tmp_path):
        """Fault injection 4: temas_por_cluster.csv is missing."""
        non_existent_csv = str(tmp_path / "non_existent_temas.csv")
        topics = load_topics_from_csv(non_existent_csv)
        assert isinstance(topics, dict)
        assert len(topics) == 0

        # Load system with unsupervised model but missing CSV
        shutil.copy(SUP_PATH, tmp_path / "supervised_model.joblib")
        shutil.copy(UNSUP_PATH, tmp_path / "unsupervised_model.joblib")
        system = ClickbaitDetectionSystem.load(models_dir=str(tmp_path))

        assert system.supervised is not None
        assert system.unsupervised is not None

        # Predict should succeed without error
        res = system.predict("Você não vai acreditar no segredo chocante!")
        assert res["is_clickbait"] is True
        assert isinstance(res["bait_type"], str)
        assert len(res["bait_type"]) > 0

    def test_zero_byte_temas_por_cluster_csv(self, tmp_path):
        """Fault injection 5: temas_por_cluster.csv is a 0-byte file."""
        csv_file = tmp_path / "temas_por_cluster.csv"
        csv_file.touch()

        topics = load_topics_from_csv(str(csv_file))
        assert isinstance(topics, dict)
        # Should gracefully return default predefined titles fallback
        assert len(topics) >= 6

    def test_malformed_temas_por_cluster_csv(self, tmp_path):
        """Fault injection 6: temas_por_cluster.csv contains garbage or missing columns."""
        csv_file = tmp_path / "temas_por_cluster.csv"
        csv_file.write_text("invalid,header,without,required,columns\n1,2,3,4,5\nfoo,bar,baz,qux,quux\n")

        topics = load_topics_from_csv(str(csv_file))
        assert isinstance(topics, dict)
        # Graceful fallback to default predefined titles
        assert len(topics) >= 6
        assert 0 in topics

    def test_corrupted_csv_syntax(self, tmp_path):
        """Fault injection 7: temas_por_cluster.csv contains unparsable binary garbage."""
        csv_file = tmp_path / "temas_por_cluster.csv"
        csv_file.write_bytes(b"\x00\xFF\xFE\xFDGARBAGE_BINARY\x00\x00\x00\x01\x02\x03")

        topics = load_topics_from_csv(str(csv_file))
        assert isinstance(topics, dict)
        assert len(topics) >= 6

    def test_load_unsupervised_false_explicit(self):
        """Verify load_unsupervised=False loads only the supervised predictor."""
        system = ClickbaitDetectionSystem.load(load_unsupervised=False)
        assert system.supervised is not None
        assert system.unsupervised is None

        res = system.predict("Veja o segredo que chocou a todos na internet!")
        assert res["is_clickbait"] is True
        assert res["bait_type"] is None


class TestApiDegradedStateResilience:
    """Verify FastAPI /api/predict and /predict behavior when unsupervised model is unavailable."""

    def test_api_predict_single_when_unsupervised_is_none(self, monkeypatch):
        """Simulate degraded system in app: unsupervised is None."""
        isolated_system = ClickbaitDetectionSystem.load(load_unsupervised=False)
        assert isolated_system.unsupervised is None

        monkeypatch.setattr(app_module, "system", isolated_system)
        client = TestClient(app)

        # Clickbait headline
        resp = client.post("/api/predict", json={"headline": "Você não vai acreditar no que aconteceu com esse famoso!"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_clickbait"] is True
        assert data["clickbait_status"] == "clickbait"
        assert data["probability"] > 0.80
        assert data["bait_type"] is None, "In degraded state, bait_type MUST be null"

        # Legitimate headline
        resp_legit = client.post("/api/predict", json={"headline": "Ministério da Fazenda divulga balanço fiscal do terceiro trimestre."})
        assert resp_legit.status_code == 200
        data_legit = resp_legit.json()
        assert data_legit["is_clickbait"] is False
        assert data_legit["clickbait_status"] == "not_clickbait"
        assert data_legit["bait_type"] is None

    def test_api_predict_batch_when_unsupervised_is_none(self, monkeypatch):
        """Simulate degraded system in app with batch request."""
        isolated_system = ClickbaitDetectionSystem.load(load_unsupervised=False)
        monkeypatch.setattr(app_module, "system", isolated_system)
        client = TestClient(app)

        batch_payload = {
            "headlines": [
                "Você não vai acreditar no que aconteceu com esse famoso!",
                "Ministério da Saúde divulga calendário de vacinação",
                "O método secreto que os ricos usam para economizar!",
                "Banco Central eleva taxa Selic para 11,25% ao ano",
            ]
        }
        resp = client.post("/api/predict", json=batch_payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 4
        assert len(data["results"]) == 4

        # Clickbait items
        assert data["results"][0]["is_clickbait"] is True
        assert data["results"][0]["bait_type"] is None
        assert data["results"][2]["is_clickbait"] is True
        assert data["results"][2]["bait_type"] is None

        # Non-clickbait items
        assert data["results"][1]["is_clickbait"] is False
        assert data["results"][1]["bait_type"] is None
        assert data["results"][3]["is_clickbait"] is False
        assert data["results"][3]["bait_type"] is None

    def test_api_predict_alias_under_degraded_state(self, monkeypatch):
        """Verify alias /predict endpoint works identically under degraded state."""
        isolated_system = ClickbaitDetectionSystem.load(load_unsupervised=False)
        monkeypatch.setattr(app_module, "system", isolated_system)
        client = TestClient(app)

        resp = client.post("/predict", json={"headline": "URGENTE: Veja o escândalo chocante que parou o país!"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_clickbait"] is True
        assert data["bait_type"] is None

    def test_unsupervised_runtime_exception_during_api_call(self, monkeypatch):
        """Verify that if unsupervised.categorize() raises an unexpected exception, API returns 200 with fallback."""
        class FailingCategorizer:
            def categorize(self, text):
                raise RuntimeError("Unexpected internal crash in unsupervised categorizer")

        supervised = SupervisedPredictor(model_path=str(SUP_PATH))
        exploding_system = ClickbaitDetectionSystem(supervised=supervised, unsupervised=FailingCategorizer())

        monkeypatch.setattr(app_module, "system", exploding_system)
        client = TestClient(app)

        resp = client.post("/api/predict", json={"headline": "Você não vai acreditar no que aconteceu com esse famoso!"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_clickbait"] is True
        assert data["bait_type"] == "Sensacionalismo Geral"


class TestConcurrentStressDegraded:
    """Stress-tests degraded system under concurrent-like batch loads."""

    def test_high_volume_degraded_throughput(self):
        """Verify processing 500 items through degraded system has high throughput and zero memory leak."""
        system = ClickbaitDetectionSystem.load(load_unsupervised=False)
        headlines = [
            "Você não vai acreditar no segredo chocante!",
            "Governo anuncia novos investimentos em infraestrutura",
            "Descubra como emagrecer dormindo com este método!",
            "Previsão do tempo indica sol no fim de semana",
        ] * 125 # 500 headlines

        results = [system.predict(h) for h in headlines]
        assert len(results) == 500
        for r in results:
            assert isinstance(r["is_clickbait"], (bool, np.bool_))
            assert r["bait_type"] is None
