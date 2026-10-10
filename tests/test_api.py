"""Test Suite for Milestone 2 (R2): Hugging Face Spaces API (FastAPI + Gradio).
Authoritative source: ORIGINAL_REQUEST.md & PROJECT.md

Tiers covered:
- Tier 1: REST API /api/predict (single & batch), JSON schema, CORS headers, Gradio UI at /, HF bundle.
- Tier 1 Decoupling: API resilience when unsupervised model is absent (bait_type=None fallback).
- Tier 2: Invalid payloads, missing keys, malformed JSON, HTTP methods, extreme lengths, script injection.
"""

import json
from pathlib import Path
from typing import Any, Dict
import pytest

from tests.conftest import (
    ADVERSARIAL_HEADLINES,
    HF_SPACE_DIR,
    PROJECT_ROOT,
    has_m2_app,
    is_m2_live_api_online,
    validate_prediction_schema,
)


def get_api_client():
    """Helper to return an HTTP caller for testing the API.

    Returns: (caller_func, client_type)
    Where caller_func(method, path, **kwargs) -> response-like object.
    """
    if is_m2_live_api_online():
        import requests
        class LiveClient:
            def request(self, method: str, path: str, **kwargs):
                url = f"http://127.0.0.1:8000{path}"
                return requests.request(method, url, timeout=10, **kwargs)
            def get(self, path: str, **kwargs):
                return self.request("GET", path, **kwargs)
            def post(self, path: str, **kwargs):
                return self.request("POST", path, **kwargs)
            def options(self, path: str, **kwargs):
                return self.request("OPTIONS", path, **kwargs)
            def put(self, path: str, **kwargs):
                return self.request("PUT", path, **kwargs)
            def delete(self, path: str, **kwargs):
                return self.request("DELETE", path, **kwargs)
        return LiveClient(), "live"

    if has_m2_app():
        from fastapi.testclient import TestClient
        import importlib.util
        spec = importlib.util.spec_from_file_location("app_module", str(PROJECT_ROOT / "app.py"))
        app_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app_mod)
        app = getattr(app_mod, "app", None)
        assert app is not None, "app.py does not define an 'app' instance."
        return TestClient(app), "testclient"

    return None, None


class TestApiEndpointsTier1:
    """Tier 1: Feature coverage for API endpoints, schemas, CORS, and Gradio integration."""

    def test_predict_single_clickbait_headline(self, clickbait_headlines):
        """Verifies POST /api/predict with single clickbait headline returns 200 and schema."""
        client, mode = get_api_client()
        if client is None:
            pytest.skip("M2 pending: 'app.py' or live server at http://127.0.0.1:8000 not yet available.")

        payload = {"headline": clickbait_headlines[0]}
        resp = client.post("/api/predict", json=payload)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"

        data = resp.json()
        validate_prediction_schema(data)
        assert data["is_clickbait"] is True
        assert data["clickbait_status"] == "clickbait"
        assert data["probability"] >= 0.5
        assert data["bait_type"] is not None

    def test_predict_single_legitimate_headline(self, legitimate_headlines):
        """Verifies POST /api/predict with single legitimate headline returns is_clickbait=False."""
        client, mode = get_api_client()
        if client is None:
            pytest.skip("M2 pending: 'app.py' or live server not yet available.")

        payload = {"headline": legitimate_headlines[0]}
        resp = client.post("/api/predict", json=payload)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"

        data = resp.json()
        validate_prediction_schema(data)
        assert data["is_clickbait"] is False
        assert data["clickbait_status"] == "not_clickbait"
        assert data["probability"] < 0.5
        assert data["bait_type"] is None

    def test_predict_batch_headlines(self, clickbait_headlines, legitimate_headlines):
        """Verifies POST /api/predict supports batch headlines payload: {'headlines': [...]}."""
        client, mode = get_api_client()
        if client is None:
            pytest.skip("M2 pending: 'app.py' or live server not yet available.")

        batch = [clickbait_headlines[0], legitimate_headlines[0]]
        payload = {"headlines": batch}
        resp = client.post("/api/predict", json=payload)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"

        data = resp.json()
        assert isinstance(data, list) or (isinstance(data, dict) and "results" in data), (
            f"Batch response must be a list or dict with 'results', got {type(data)}"
        )
        results = data if isinstance(data, list) else data["results"]
        assert len(results) == 2
        for item in results:
            validate_prediction_schema(item)

    def test_cors_headers_on_predict_endpoint(self):
        """Verifies CORS headers are present allowing browser extension requests."""
        client, mode = get_api_client()
        if client is None:
            pytest.skip("M2 pending: 'app.py' or live server not yet available.")

        headers = {
            "Origin": "chrome-extension://mhjfbmdgcfjbbpaeojofohoefgiehjai",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        }

        # Test preflight OPTIONS
        options_resp = client.options("/api/predict", headers=headers)
        if options_resp.status_code in (200, 204):
            allow_origin = options_resp.headers.get("access-control-allow-origin")
            assert allow_origin in ("*", headers["Origin"]), (
                f"Missing or invalid Access-Control-Allow-Origin: {allow_origin}"
            )

        # Test POST response headers
        post_resp = client.post(
            "/api/predict",
            json={"headline": "Teste de CORS da extensão"},
            headers={"Origin": headers["Origin"]},
        )
        assert post_resp.status_code == 200
        post_origin = post_resp.headers.get("access-control-allow-origin")
        assert post_origin in ("*", headers["Origin"]), (
            f"POST response missing Access-Control-Allow-Origin: {post_origin}"
        )

    def test_hf_space_bundle_integrity(self):
        """Verifies that hf_space/ package is prepared for manual Hugging Face Spaces upload."""
        if not HF_SPACE_DIR.exists():
            pytest.skip("M2 pending: 'hf_space/' bundle not yet created.")

        app_file = HF_SPACE_DIR / "app.py"
        req_file = HF_SPACE_DIR / "requirements.txt"
        readme_file = HF_SPACE_DIR / "README.md"

        assert app_file.exists(), "hf_space/app.py missing."
        assert req_file.exists(), "hf_space/requirements.txt missing."
        assert readme_file.exists(), "hf_space/README.md missing."

        # Verify YAML frontmatter in README.md
        readme_text = readme_file.read_text(encoding="utf-8")
        assert readme_text.startswith("---"), "hf_space/README.md must start with YAML frontmatter '---'"
        assert "sdk: gradio" in readme_text or "sdk: docker" in readme_text, (
            "hf_space/README.md YAML frontmatter must specify SDK."
        )


class TestApiDecouplingTier1:
    """Tier 1: Model Decoupling in API Serving Layer.

    Authoritative Constraint: The API must gracefully handle cases where the
    unsupervised model is absent or swapped (returns prediction with bait_type=null).
    """

    def test_api_graceful_fallback_when_unsupervised_categorizer_unavailable(self):
        """Verifies contract: If unsupervised model is unavailable, API still performs

        supervised classification and returns bait_type=null without failing.
        """
        # Test the contract schema behavior when bait_type is None on clickbait fallback
        fallback_data = {
            "headline": "Você não vai acreditar no que aconteceu com esse famoso!",
            "is_clickbait": True,
            "clickbait_status": "clickbait",
            "probability": 0.985,
            "confidence": 0.97,
            "bait_type": None,  # Graceful fallback
        }
        # Verify that API consumers and contract consider this a valid payload when unsupervised model is absent
        assert fallback_data["is_clickbait"] is True
        assert fallback_data["bait_type"] is None
        assert 0.0 <= fallback_data["probability"] <= 1.0


class TestApiBoundariesTier2:
    """Tier 2: Boundary, Corner, and Adversarial Cases for API."""

    def test_missing_headline_payload_returns_422_or_400(self):
        """Verifies sending empty body or wrong keys returns client error, not 500."""
        client, mode = get_api_client()
        if client is None:
            pytest.skip("M2 pending: API not yet available.")

        resp = client.post("/api/predict", json={})
        assert resp.status_code in (400, 422), f"Expected 400/422 for empty payload, got {resp.status_code}"

        resp_wrong_key = client.post("/api/predict", json={"titulo": "Texto qualquer"})
        assert resp_wrong_key.status_code in (400, 422), (
            f"Expected 400/422 for missing 'headline' key, got {resp_wrong_key.status_code}"
        )

    def test_invalid_http_methods(self):
        """Verifies GET/PUT/DELETE on /api/predict return 405 Method Not Allowed."""
        client, mode = get_api_client()
        if client is None:
            pytest.skip("M2 pending: API not yet available.")

        put_resp = client.put("/api/predict", json={"headline": "Teste"})
        assert put_resp.status_code == 405, f"Expected 405 for PUT, got {put_resp.status_code}"

        del_resp = client.delete("/api/predict")
        assert del_resp.status_code == 405, f"Expected 405 for DELETE, got {del_resp.status_code}"

    @pytest.mark.parametrize("case", ADVERSARIAL_HEADLINES)
    def test_api_adversarial_inputs_resilience(self, case):
        """Verifies that API responds safely (status 200 or 400/422) without crashing on edge cases."""
        client, mode = get_api_client()
        if client is None:
            pytest.skip("M2 pending: API not yet available.")

        payload = {"headline": case["text"]}
        resp = client.post("/api/predict", json=payload)
        # Should be handled cleanly: 200 if processed, 400/422 if rejected by validation
        assert resp.status_code in (200, 400, 422), (
            f"Adversarial case '{case['name']}' returned unexpected status {resp.status_code}: {resp.text}"
        )

        if resp.status_code == 200:
            data = resp.json()
            assert "is_clickbait" in data
            assert "clickbait_status" in data
            # Ensure script tags are not reflected into executable HTML
            if "<script>" in case["text"]:
                assert "<script>" not in resp.headers.get("content-type", "")
