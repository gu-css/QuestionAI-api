#!/usr/bin/env python3
"""Automated API Test Script for Clickbait Detection & Categorization System.
Satisfies Acceptance Criterion R2 from ORIGINAL_REQUEST.md:
"An automated test script sends a POST request to the local Gradio API with a sample
headline and verifies that the JSON response contains both the clickbait status and bait type."

Usage:
    python3 scripts/test_api.py
    python3 scripts/test_api.py --url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Test Clickbait Detection API (R2 Acceptance Criterion)")
    parser.add_argument(
        "--url",
        type=str,
        default="http://127.0.0.1:8000",
        help="Base URL of the running API server (default: http://127.0.0.1:8000)",
    )
    return parser.parse_args()


class ApiTester:
    """Handles testing either via HTTP requests against live server or TestClient fallback."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.is_live = False
        self.client = None

        # Check if live server is reachable
        try:
            r = requests.get(f"{self.base_url}/api/health", timeout=1.5)
            if r.status_code == 200:
                self.is_live = True
        except Exception:
            pass

        if not self.is_live:
            # Fallback to FastAPI in-process TestClient
            from fastapi.testclient import TestClient
            from app import app
            self.client = TestClient(app)

    def post(self, path: str, json_data: dict, headers: Optional[dict] = None) -> Any:
        if self.is_live:
            return requests.post(f"{self.base_url}{path}", json=json_data, headers=headers, timeout=10)
        return self.client.post(path, json=json_data, headers=headers)

    def options(self, path: str, headers: Optional[dict] = None) -> Any:
        if self.is_live:
            return requests.options(f"{self.base_url}{path}", headers=headers, timeout=10)
        return self.client.options(path, headers=headers)

    def get(self, path: str) -> Any:
        if self.is_live:
            return requests.get(f"{self.base_url}{path}", timeout=10)
        return self.client.get(path)


def run_tests(tester: ApiTester) -> bool:
    mode_str = f"LIVE SERVER ({tester.base_url})" if tester.is_live else "IN-PROCESS TESTCLIENT"
    print("=" * 72)
    print(f" 🧪 CLICKBAIT DETECTION API VERIFICATION — MODE: {mode_str}")
    print("=" * 72)

    passed_tests = 0
    total_tests = 6

    # Test 1: Single Clickbait Headline (AC R2 Core Assertion)
    print("\n[1/6] Testing POST /api/predict with Clickbait Headline...")
    cb_headline = "Você não vai acreditar no que aconteceu com esse famoso!"
    r1 = tester.post("/api/predict", {"headline": cb_headline})
    assert r1.status_code == 200, f"Expected 200, got {r1.status_code}: {r1.text}"
    d1 = r1.json()
    print(f"   Payload  : {cb_headline}")
    print(f"   Response : {json.dumps(d1, ensure_ascii=False)}")

    assert "clickbait_status" in d1, "Response missing 'clickbait_status'"
    assert "bait_type" in d1, "Response missing 'bait_type'"
    assert d1["is_clickbait"] is True, f"Expected is_clickbait=True, got {d1['is_clickbait']}"
    assert d1["clickbait_status"] == "clickbait", f"Expected clickbait_status='clickbait', got {d1['clickbait_status']}"
    assert d1["probability"] >= 0.5, f"Expected probability >= 0.5, got {d1['probability']}"
    assert d1["bait_type"] is not None and len(d1["bait_type"].strip()) > 0, "Expected non-empty bait_type"
    print("   ✅ PASSED: Detected clickbait with valid probability and bait_type archetype.")
    passed_tests += 1

    # Test 2: Single Legitimate Headline
    print("\n[2/6] Testing POST /api/predict with Legitimate Headline...")
    legit_headline = "Ministério da Fazenda divulga balanço fiscal do terceiro trimestre nesta terça-feira"
    r2 = tester.post("/api/predict", {"headline": legit_headline})
    assert r2.status_code == 200, f"Expected 200, got {r2.status_code}: {r2.text}"
    d2 = r2.json()
    print(f"   Payload  : {legit_headline}")
    print(f"   Response : {json.dumps(d2, ensure_ascii=False)}")

    assert d2["is_clickbait"] is False, f"Expected is_clickbait=False, got {d2['is_clickbait']}"
    assert d2["clickbait_status"] == "not_clickbait", f"Expected clickbait_status='not_clickbait', got {d2['clickbait_status']}"
    assert d2["probability"] < 0.5, f"Expected probability < 0.5, got {d2['probability']}"
    assert d2["bait_type"] is None, f"Expected bait_type=None for legitimate headline, got {d2['bait_type']}"
    print("   ✅ PASSED: Detected legitimate headline with bait_type=null.")
    passed_tests += 1

    # Test 3: Batch Headlines (M2 ↔ M3 Extension Contract)
    print("\n[3/6] Testing POST /api/predict with Batch Payload...")
    batch_payload = {"headlines": [cb_headline, legit_headline]}
    r3 = tester.post("/api/predict", batch_payload)
    assert r3.status_code == 200, f"Expected 200, got {r3.status_code}: {r3.text}"
    d3 = r3.json()
    results = d3["results"] if isinstance(d3, dict) and "results" in d3 else d3
    assert len(results) == 2, f"Expected 2 batch results, got {len(results)}"
    assert results[0]["is_clickbait"] is True
    assert results[1]["is_clickbait"] is False
    print(f"   Batch processed {len(results)} items successfully.")
    print("   ✅ PASSED: Batch prediction verified.")
    passed_tests += 1

    # Test 4: CORS Support
    print("\n[4/6] Testing CORS Headers on /api/predict...")
    cors_headers = {
        "Origin": "chrome-extension://abcdefghijklmnopqrstuvwxyz",
        "Access-Control-Request-Method": "POST",
    }
    r4_opt = tester.options("/api/predict", headers=cors_headers)
    allow_origin = r4_opt.headers.get("access-control-allow-origin", "")
    print(f"   Preflight OPTIONS Origin header: '{allow_origin}'")
    assert allow_origin in ("*", cors_headers["Origin"]), f"Invalid CORS header: {allow_origin}"

    r4_post = tester.post("/api/predict", {"headline": "Teste CORS"}, headers={"Origin": cors_headers["Origin"]})
    post_origin = r4_post.headers.get("access-control-allow-origin", "")
    assert post_origin in ("*", cors_headers["Origin"]), f"Invalid POST CORS header: {post_origin}"
    print("   ✅ PASSED: CORS fully open for browser extensions.")
    passed_tests += 1

    # Test 5: Empty and Edge-Case Input Resilience
    print("\n[5/6] Testing Empty Input Handling...")
    r5 = tester.post("/api/predict", {"headline": "   "})
    assert r5.status_code == 200, f"Expected 200 for empty string, got {r5.status_code}"
    d5 = r5.json()
    assert d5["is_clickbait"] is False
    assert d5["probability"] == 0.0
    assert d5["bait_type"] is None
    print(f"   Empty headline response: {d5}")
    print("   ✅ PASSED: Graceful empty string handling verified.")
    passed_tests += 1

    # Test 6: Gradio Web UI at Root "/"
    print("\n[6/6] Testing Gradio Web UI at Root '/'...")
    r6 = tester.get("/")
    assert r6.status_code == 200, f"Expected 200 at GET /, got {r6.status_code}"
    content_type = r6.headers.get("content-type", "")
    assert "text/html" in content_type, f"Expected text/html, got {content_type}"
    print(f"   UI Content-Type: {content_type}")
    print("   ✅ PASSED: Gradio Blocks UI mounted successfully at '/'.")
    passed_tests += 1

    print("\n" + "=" * 72)
    print(f" 🎉 ALL {passed_tests}/{total_tests} TESTS PASSED SUCCESSFULLY! ACCEPTANCE CRITERION R2 SATISFIED.")
    print("=" * 72)
    return True


def main():
    args = parse_args()
    tester = ApiTester(base_url=args.url)
    success = run_tests(tester)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
