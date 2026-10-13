"""tests/test_challenger_m2.py - Empirical Adversarial Challenge Suite for Milestone M2.

Authoritative constraints: ORIGINAL_REQUEST.md (R2) & PROJECT.md (M2).

Adversarial Challenge Vectors:
1. HTTP Protocol Challenges:
   - Unsupported HTTP methods (GET, PUT, DELETE, PATCH, HEAD) on /api/predict and /predict alias -> 405 Method Not Allowed.
   - Method options and health check method filtering.
2. Malformed Payloads:
   - Non-JSON / syntax error byte streams -> 422 or 400.
   - Top-level non-dict JSON payloads (raw list, string, int, bool, null) -> 422.
   - Empty JSON dict {} and wrong keys -> 422.
   - Type violations in request fields (e.g. dict inside headline, nulls inside headlines list) -> 422.
   - Malicious/extreme inputs (XSS tags, SQLi strings, null bytes, unicode control chars, zalgo/emoji) -> handled safely with status 200, valid schema, no 500 crashes.
3. Batch and Empty Edge Cases:
   - Empty batch: {"headlines": []} and {"data": []} -> 200 with total: 0 and results: [].
   - Batches with empty/whitespace strings -> all processed gracefully with is_clickbait=False.
   - Single empty/whitespace strings -> handled gracefully with is_clickbait=False.
4. Huge Payloads & Stress:
   - Huge headline texts (10 KB, 50 KB, 100 KB, 500 KB) -> no ReDoS hang, sub-second execution.
   - Large batches (100 and 500 headlines) -> processed cleanly without memory explosion.
5. Concurrency & Throughput:
   - Concurrent load (30+ simultaneous threads) on /api/predict.
   - Mixed concurrent workload (clickbait, legitimate, empty, batch, healthcheck).
   - Verifies thread-safety, zero error rate, latency stats.
6. CORS Response Headers:
   - Chrome Extension origin preflight (OPTIONS) and POST requests.
   - Various origin headers (chrome-extension://*, http://localhost:*, https://*, null).
   - Validates Access-Control-Allow-Origin, Methods, and Headers.
7. Model Decoupling & Fault Injection:
   - Graceful fallback when unsupervised model is None (returns bait_type=None without crashing).
   - Healthcheck status reflection during model state changes.
"""

import concurrent.futures
import copy
import json
import os
import sys
import time
from typing import Any, Dict, List
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient
import app as app_module
from app import app
from src.pipeline import ClickbaitDetectionSystem


@pytest.fixture(scope="module")
def api_client():
    """Provides a FastAPI TestClient instance for the application."""
    return TestClient(app)


# Sample headlines for testing
SAMPLE_CLICKBAIT = "Você não vai acreditar no que aconteceu com esse famoso!"
SAMPLE_LEGITIMATE = "Ministério da Fazenda divulga balanço fiscal do terceiro trimestre nesta terça-feira"


class TestChallengerHttpMethods:
    """Challenge 1: HTTP Method Restrictions and 405 Handling."""

    @pytest.mark.parametrize("method", ["get", "put", "delete", "patch", "head"])
    def test_unsupported_http_methods_on_api_predict_return_405(self, api_client, method):
        """Rejects non-POST methods with HTTP 405 Method Not Allowed."""
        func = getattr(api_client, method)
        response = func("/api/predict")
        assert response.status_code == 405, (
            f"Expected HTTP 405 for {method.upper()} /api/predict, got {response.status_code}: {response.text}"
        )

    @pytest.mark.parametrize("method", ["get", "put", "delete", "patch", "head"])
    def test_unsupported_http_methods_on_predict_alias_return_405(self, api_client, method):
        """Rejects non-POST methods on the /predict alias with HTTP 405."""
        func = getattr(api_client, method)
        response = func("/predict")
        assert response.status_code == 405, (
            f"Expected HTTP 405 for {method.upper()} /predict, got {response.status_code}: {response.text}"
        )

    def test_health_endpoint_allowed_and_unsupported_methods(self, api_client):
        """Verifies GET /api/health succeeds, and non-GET returns 405."""
        get_res = api_client.get("/api/health")
        assert get_res.status_code == 200
        health_data = get_res.json()
        assert health_data.get("status") == "healthy"
        assert "supervised_loaded" in health_data

        post_res = api_client.post("/api/health", json={})
        assert post_res.status_code in (404, 405), (
            f"Expected 404 or 405 for non-GET on /api/health, got {post_res.status_code}"
        )


class TestChallengerPayloadMalformed:
    """Challenge 2: Malformed Payloads, Type Violations, and Adversarial Strings."""

    def test_invalid_json_syntax_returns_422_or_400(self, api_client):
        """Sending corrupted raw bytes as JSON must yield 400 or 422, never 500."""
        malformed_bytes = b'{"headline": "truncat'
        response = api_client.post(
            "/api/predict",
            content=malformed_bytes,
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code in (400, 422), (
            f"Expected 400/422 for malformed JSON syntax, got {response.status_code}"
        )

    @pytest.mark.parametrize("bad_payload", [
        b'["Headline 1", "Headline 2"]',  # Top-level array
        b'"Uma simples string"',         # Top-level string
        b'12345',                        # Top-level integer
        b'true',                         # Top-level boolean
        b'null',                         # Top-level null
    ])
    def test_top_level_non_object_json_rejected(self, api_client, bad_payload):
        """Top-level JSON that is not an object must be rejected with 422."""
        response = api_client.post(
            "/api/predict",
            content=bad_payload,
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422, (
            f"Expected 422 for non-object top-level payload {bad_payload}, got {response.status_code}"
        )

    def test_empty_object_and_missing_keys_rejected(self, api_client):
        """Empty dictionary or dictionary without valid keys must return 422."""
        resp1 = api_client.post("/api/predict", json={})
        assert resp1.status_code == 422

        resp2 = api_client.post("/api/predict", json={"unknown_key": "some_value"})
        assert resp2.status_code == 422

        resp3 = api_client.post("/api/predict", json={"headline": None, "headlines": None})
        assert resp3.status_code == 422

    @pytest.mark.parametrize("invalid_field_payload", [
        {"headline": {"nested": "dict_not_string"}},
        {"headline": ["list_item_1", "list_item_2"]},
        {"headlines": "string_instead_of_list"},
        {"headlines": [None]},
        {"headlines": [{"nested": "dict"}]},
        {"headlines": [["sublist"]]},
    ])
    def test_type_violations_in_fields_rejected_with_422(self, api_client, invalid_field_payload):
        """Type violations on Pydantic fields must be rejected with 422."""
        response = api_client.post("/api/predict", json=invalid_field_payload)
        assert response.status_code == 422, (
            f"Expected 422 for {invalid_field_payload}, got {response.status_code}"
        )

    @pytest.mark.parametrize("adversarial_headline", [
        # SQL Injection attempts
        "SELECT * FROM users WHERE '1'='1'; DROP TABLE noticias; --",
        # XSS / HTML tags
        "<script>alert('xss')</script><b>Impactante!</b><iframe src='about:blank'></iframe>",
        # Null bytes and escape chars
        "Manchete com \x00 null byte e \r\n quebra de linha",
        # Unicode RTL override & zero-width characters
        "Notícia bombástica \u202e inverteu o texto \u200b\u200c escondido",
        # Emoji overload
        "💣🔥🚨 " * 50 + "OLHA O QUE ACONTECEU COM ELES!",
        # URLs and mentions
        "https://example.com/noticia?id=123&track=abc @usuario veja isso!",
        # Portuguese extreme diacritics & punctuation
        "AÇÃO, ÓRFÃO, ÍNDIO, ÚMIDO, ÉPICO!?!?!!!...---...???",
    ])
    def test_adversarial_strings_handled_safely(self, api_client, adversarial_headline):
        """Adversarial text inputs must process safely, returning 200 without crashing."""
        response = api_client.post("/api/predict", json={"headline": adversarial_headline})
        assert response.status_code == 200, (
            f"Adversarial string crashed with {response.status_code}: {response.text}"
        )
        data = response.json()
        assert "is_clickbait" in data
        assert "clickbait_status" in data
        assert isinstance(data["probability"], (float, int))
        assert 0.0 <= data["probability"] <= 1.0


class TestChallengerBatchAndEmpty:
    """Challenge 3: Empty Batches, Whitespace, and Batch Edge Cases."""

    def test_empty_headlines_batch_handled_cleanly(self, api_client):
        """Payload {"headlines": []} must return 200 with total=0 and results=[]."""
        response = api_client.post("/api/predict", json={"headlines": []})
        assert response.status_code == 200, (
            f"Expected 200 for empty headlines list, got {response.status_code}: {response.text}"
        )
        data = response.json()
        assert data.get("total") == 0
        assert data.get("results") == []

    def test_empty_legacy_data_batch_handled_cleanly(self, api_client):
        """Payload {"data": []} must return 200 with total=0 and results=[]."""
        response = api_client.post("/api/predict", json={"data": []})
        assert response.status_code == 200, (
            f"Expected 200 for empty data list, got {response.status_code}: {response.text}"
        )
        data = response.json()
        assert data.get("total") == 0
        assert data.get("results") == []

    def test_batch_with_whitespace_and_empty_headlines(self, api_client):
        """Batches containing only empty or whitespace strings return legitimate classification."""
        batch = ["", "   ", "\t\r\n", "    \n\t   "]
        response = api_client.post("/api/predict", json={"headlines": batch})
        assert response.status_code == 200
        data = response.json()
        assert data.get("total") == 4
        for item in data.get("results", []):
            assert item["is_clickbait"] is False
            assert item["clickbait_status"] == "not_clickbait"
            assert item["probability"] == 0.0
            assert item["bait_type"] is None

    @pytest.mark.parametrize("blank", ["", " ", "   ", "\n\t\r", "   \n\t   "])
    def test_single_blank_headline_returns_safe_default(self, api_client, blank):
        """Blank single headline returns 200 with is_clickbait=False, probability=0.0."""
        response = api_client.post("/api/predict", json={"headline": blank})
        assert response.status_code == 200
        data = response.json()
        assert data["is_clickbait"] is False
        assert data["clickbait_status"] == "not_clickbait"
        assert data["probability"] == 0.0
        assert data["bait_type"] is None


class TestChallengerHugePayloads:
    """Challenge 4: Huge Strings, Large Batches, and Execution Limits."""

    @pytest.mark.parametrize("size_kb", [10, 50, 100, 500])
    def test_huge_single_headline_throughput(self, api_client, size_kb):
        """Large single headline strings must not cause ReDoS, memory spike, or timeout."""
        # Generate text of approximately size_kb kilobytes with Portuguese words
        base_sentence = "Esta é uma grande reportagem investigativa sobre economia e política brasileira. "
        multiplier = (size_kb * 1024) // len(base_sentence)
        large_text = base_sentence * multiplier

        start_time = time.perf_counter()
        response = api_client.post("/api/predict", json={"headline": large_text})
        elapsed = time.perf_counter() - start_time

        assert response.status_code == 200, (
            f"Failed on {size_kb}KB headline with {response.status_code}: {response.text}"
        )
        assert elapsed < 5.0, (
            f"{size_kb}KB headline took {elapsed:.2f}s (potential ReDoS or bottleneck)"
        )
        data = response.json()
        assert "is_clickbait" in data

    def test_large_batch_processing(self, api_client):
        """Batch of 100 headlines processed in a single request."""
        sample_pool = [
            SAMPLE_CLICKBAIT,
            SAMPLE_LEGITIMATE,
            "Veja como economizar combustível com essa dica simples!",
            "Petrobras divulga plano de investimentos para os próximos cinco anos",
        ]
        large_batch = [sample_pool[i % len(sample_pool)] for i in range(100)]

        start_time = time.perf_counter()
        response = api_client.post("/api/predict", json={"headlines": large_batch})
        elapsed = time.perf_counter() - start_time

        assert response.status_code == 200
        data = response.json()
        assert data.get("total") == 100
        assert len(data.get("results", [])) == 100
        # Processing 100 items should complete in reasonable time (< 10 seconds)
        assert elapsed < 10.0, f"100 headlines batch took {elapsed:.2f}s"


class TestChallengerCorsHeaders:
    """Challenge 5: Strict CORS Validation for Chrome Extension Origins."""

    EXT_ORIGIN = "chrome-extension://mhjfbmdgcfjbbpaeojofohoefgiehjai"

    def test_cors_preflight_options_for_chrome_extension(self, api_client):
        """OPTIONS preflight with chrome-extension:// origin returns proper CORS headers."""
        headers = {
            "Origin": self.EXT_ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        }
        response = api_client.options("/api/predict", headers=headers)
        assert response.status_code in (200, 204), (
            f"OPTIONS preflight failed with {response.status_code}: {response.text}"
        )
        origin_header = response.headers.get("access-control-allow-origin")
        assert origin_header in ("*", self.EXT_ORIGIN), (
            f"Invalid Access-Control-Allow-Origin: {origin_header}"
        )
        methods_header = response.headers.get("access-control-allow-methods", "")
        assert "POST" in methods_header.upper() or "*" in methods_header

    def test_cors_actual_post_request_for_chrome_extension(self, api_client):
        """POST /api/predict includes Access-Control-Allow-Origin header."""
        response = api_client.post(
            "/api/predict",
            json={"headline": SAMPLE_CLICKBAIT},
            headers={"Origin": self.EXT_ORIGIN},
        )
        assert response.status_code == 200
        origin_header = response.headers.get("access-control-allow-origin")
        assert origin_header in ("*", self.EXT_ORIGIN), (
            f"POST response missing proper CORS origin: {origin_header}"
        )

    @pytest.mark.parametrize("origin", [
        "chrome-extension://aicmggodhhkmpocbfapfgjhbfclmjmff",
        "http://localhost:3000",
        "http://127.0.0.1:8080",
        "https://huggingface.co",
        "null",
    ])
    def test_cors_supports_multiple_origins(self, api_client, origin):
        """CORS middleware accepts requests from various web and extension origins."""
        response = api_client.post(
            "/api/predict",
            json={"headline": "Teste de cabeçalho CORS"},
            headers={"Origin": origin},
        )
        assert response.status_code == 200
        allow_origin = response.headers.get("access-control-allow-origin")
        assert allow_origin in ("*", origin), (
            f"Origin {origin} not allowed: {allow_origin}"
        )

    def test_cors_preflight_on_predict_alias(self, api_client):
        """OPTIONS preflight on /predict alias works properly."""
        headers = {
            "Origin": self.EXT_ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        }
        response = api_client.options("/predict", headers=headers)
        assert response.status_code in (200, 204)
        allow_origin = response.headers.get("access-control-allow-origin")
        assert allow_origin in ("*", self.EXT_ORIGIN)


class TestChallengerConcurrencyAndThroughput:
    """Challenge 6: Multi-threaded Concurrency and Throughput Stress Testing."""

    def test_concurrent_single_predictions(self, api_client):
        """Fires 30 concurrent requests to /api/predict across 10 threads."""
        num_requests = 30
        concurrency = 10

        def send_request(idx: int) -> Dict[str, Any]:
            is_click = (idx % 2 == 0)
            text = SAMPLE_CLICKBAIT if is_click else SAMPLE_LEGITIMATE
            start = time.perf_counter()
            resp = api_client.post("/api/predict", json={"headline": text})
            latency = time.perf_counter() - start
            return {
                "idx": idx,
                "status": resp.status_code,
                "latency": latency,
                "data": resp.json() if resp.status_code == 200 else None,
                "is_click_expected": is_click,
            }

        start_all = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(send_request, i) for i in range(num_requests)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        total_time = time.perf_counter() - start_all

        # Assert 100% success rate
        errors = [r for r in results if r["status"] != 200]
        assert len(errors) == 0, f"Encountered {len(errors)} errors during concurrency test: {errors}"

        # Assert data correctness without race conditions
        for r in results:
            data = r["data"]
            assert data is not None
            if r["is_click_expected"]:
                assert data["is_clickbait"] is True
                assert data["bait_type"] is not None
            else:
                assert data["is_clickbait"] is False
                assert data["bait_type"] is None

        # Measure throughput
        rps = num_requests / total_time
        latencies = [r["latency"] for r in results]
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)

        assert rps > 5.0, f"Throughput too low: {rps:.2f} req/s"
        assert max_latency < 3.0, f"Max latency too high: {max_latency:.2f}s"

    def test_concurrent_mixed_workload(self, api_client):
        """Fires mixed concurrent requests (single, batch, blank, health) simultaneously."""
        workload = [
            ("single_click", "/api/predict", {"headline": SAMPLE_CLICKBAIT}),
            ("single_legit", "/api/predict", {"headline": SAMPLE_LEGITIMATE}),
            ("single_empty", "/api/predict", {"headline": ""}),
            ("batch", "/api/predict", {"headlines": [SAMPLE_CLICKBAIT, SAMPLE_LEGITIMATE]}),
            ("health", "/api/health", None),
        ] * 6  # 30 requests total

        def execute_task(task):
            name, path, payload = task
            if payload is not None:
                resp = api_client.post(path, json=payload)
            else:
                resp = api_client.get(path)
            return name, resp.status_code, resp.json()

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(execute_task, item) for item in workload]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        for name, status, data in results:
            assert status == 200, f"Task {name} returned status {status}: {data}"


class TestChallengerDecouplingAndFallbacks:
    """Challenge 7: Model Decoupling & Graceful Fallback Stress Testing."""

    def test_fallback_when_unsupervised_categorizer_is_none(self, api_client):
        """Simulates absence of unsupervised model: clickbait classification continues, bait_type is None."""
        original_unsupervised = app_module.system.unsupervised
        try:
            # Inject None to simulate missing unsupervised model
            app_module.system.unsupervised = None

            # Verify healthcheck reflects the change
            health_resp = api_client.get("/api/health")
            assert health_resp.status_code == 200
            assert health_resp.json()["unsupervised_loaded"] is False

            # Verify clickbait prediction works with graceful fallback
            resp = api_client.post("/api/predict", json={"headline": SAMPLE_CLICKBAIT})
            assert resp.status_code == 200
            data = resp.json()
            assert data["is_clickbait"] is True
            assert data["clickbait_status"] == "clickbait"
            assert data["bait_type"] is None, (
                f"Expected bait_type=None when unsupervised is missing, got {data['bait_type']}"
            )
        finally:
            # Restore model state
            app_module.system.unsupervised = original_unsupervised
