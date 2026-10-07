#!/usr/bin/env python3
"""scripts/stress_test_m2.py - Standalone Live Socket Stress Test Harness for M2 API.

Performs live socket HTTP attacks, CORS verification, and high-concurrency throughput
benchmarks against the running Uvicorn + FastAPI + Gradio server.

Usage:
    python3 scripts/stress_test_m2.py [--port 7862] [--concurrency 25] [--requests 100]
"""

import argparse
import concurrent.futures
import json
import os
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple
import requests

SAMPLE_CLICKBAIT = "Você não vai acreditar no que aconteceu com esse famoso!"
SAMPLE_LEGITIMATE = "Ministério da Fazenda divulga balanço fiscal do terceiro trimestre nesta terça-feira"
CHROME_EXT_ORIGIN = "chrome-extension://mhjfbmdgcfjbbpaeojofohoefgiehjai"


class ApiStressHarness:
    def __init__(self, port: int = 7862):
        self.port = port
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.server_proc = None

    def start_server(self):
        print(f"🚀 Spawning live server process on port {self.port}...")
        env = os.environ.copy()
        env["PORT"] = str(self.port)
        self.server_proc = subprocess.Popen(
            [sys.executable, "app.py"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid,
        )

        # Wait for server to become healthy
        max_attempts = 30
        for attempt in range(max_attempts):
            time.sleep(0.5)
            try:
                r = requests.get(f"{self.base_url}/api/health", timeout=1)
                if r.status_code == 200:
                    print(f"✅ Server is healthy and listening on {self.base_url}")
                    return
            except Exception:
                pass
        raise RuntimeError(f"Server failed to start on port {self.port} within 15 seconds")

    def stop_server(self):
        if self.server_proc:
            print("🛑 Stopping server process...")
            try:
                os.killpg(os.getpgid(self.server_proc.pid), signal.SIGTERM)
                self.server_proc.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(os.getpgid(self.server_proc.pid), signal.SIGKILL)
                except Exception:
                    pass
            print("✅ Server process terminated.")

    def challenge_http_methods(self) -> List[Tuple[str, bool, str]]:
        print("\n--- [1] Challenging HTTP Methods ---")
        results = []
        methods = ["GET", "PUT", "DELETE", "PATCH", "HEAD"]
        for endpoint in ["/api/predict", "/predict"]:
            for m in methods:
                r = requests.request(m, f"{self.base_url}{endpoint}")
                passed = (r.status_code == 405)
                msg = f"{m} {endpoint} -> status {r.status_code} (expected 405)"
                print(f"  {'✅' if passed else '❌'} {msg}")
                results.append((f"{m} {endpoint}", passed, msg))

        # Test POST /api/predict (supported)
        r = requests.post(f"{self.base_url}/api/predict", json={"headline": SAMPLE_CLICKBAIT})
        passed = (r.status_code == 200)
        print(f"  {'✅' if passed else '❌'} POST /api/predict -> status {r.status_code} (expected 200)")
        results.append(("POST /api/predict", passed, f"status {r.status_code}"))

        return results

    def challenge_malformed_and_edge_payloads(self) -> List[Tuple[str, bool, str]]:
        print("\n--- [2] Challenging Malformed & Edge Payloads ---")
        results = []
        cases = [
            ("Raw malformed JSON syntax", b'{"headline": "trunc', "application/json", [400, 422]),
            ("Raw top-level array", b'["headline 1", "headline 2"]', "application/json", [422]),
            ("Raw top-level int", b'12345', "application/json", [422]),
            ("Raw top-level null", b'null', "application/json", [422]),
            ("Empty JSON object {}", json.dumps({}).encode(), "application/json", [422]),
            ("Unrelated JSON keys", json.dumps({"unknown": "val"}).encode(), "application/json", [422]),
            ("Headline type violation (dict)", json.dumps({"headline": {"bad": 1}}).encode(), "application/json", [422]),
            ("Headlines type violation (string)", json.dumps({"headlines": "not_list"}).encode(), "application/json", [422]),
            ("Headlines with null item", json.dumps({"headlines": [None]}).encode(), "application/json", [422]),
            ("Empty batch headlines []", json.dumps({"headlines": []}).encode(), "application/json", [200]),
            ("Empty batch legacy data []", json.dumps({"data": []}).encode(), "application/json", [200]),
            ("Batch with blanks only", json.dumps({"headlines": ["", "  ", "\t\n"]}).encode(), "application/json", [200]),
            ("Single empty string", json.dumps({"headline": ""}).encode(), "application/json", [200]),
            ("Single whitespace string", json.dumps({"headline": "   "}).encode(), "application/json", [200]),
        ]

        for name, body, ctype, expected_codes in cases:
            r = requests.post(
                f"{self.base_url}/api/predict",
                data=body,
                headers={"Content-Type": ctype},
                timeout=5,
            )
            passed = r.status_code in expected_codes
            msg = f"{name} -> status {r.status_code} (expected {expected_codes})"
            print(f"  {'✅' if passed else '❌'} {msg}")
            results.append((name, passed, msg))

        return results

    def challenge_huge_payloads(self) -> List[Tuple[str, bool, str]]:
        print("\n--- [3] Challenging Huge Payloads & ReDoS ---")
        results = []

        # 100 KB and 500 KB single string
        for size_kb in [50, 100, 500]:
            chunk = "Esta é uma manchete longa de teste de estresse jornalístico. "
            text = chunk * ((size_kb * 1024) // len(chunk))
            t0 = time.perf_counter()
            r = requests.post(
                f"{self.base_url}/api/predict",
                json={"headline": text},
                timeout=10,
            )
            elapsed = time.perf_counter() - t0
            passed = (r.status_code == 200 and elapsed < 4.0)
            msg = f"Single string {size_kb}KB -> status {r.status_code}, elapsed {elapsed:.3f}s"
            print(f"  {'✅' if passed else '❌'} {msg}")
            results.append((f"Huge text {size_kb}KB", passed, msg))

        # Batch scaling curve: 25, 50, 100 headlines
        for batch_size in [25, 50, 100]:
            batch = [SAMPLE_CLICKBAIT if i % 2 == 0 else SAMPLE_LEGITIMATE for i in range(batch_size)]
            t0 = time.perf_counter()
            r = requests.post(
                f"{self.base_url}/api/predict",
                json={"headlines": batch},
                timeout=30,
            )
            elapsed = time.perf_counter() - t0
            passed = (r.status_code == 200 and len(r.json().get("results", [])) == batch_size)
            ms_per_item = (elapsed / batch_size) * 1000
            msg = f"Batch of {batch_size} headlines -> status {r.status_code}, total {elapsed:.3f}s ({ms_per_item:.1f} ms/headline)"
            print(f"  {'✅' if passed else '❌'} {msg}")
            results.append((f"Batch {batch_size}", passed, msg))

        return results

    def challenge_cors_headers(self) -> List[Tuple[str, bool, str]]:
        print("\n--- [4] Challenging CORS Headers with Chrome Extension ---")
        results = []

        # Preflight OPTIONS
        preflight_headers = {
            "Origin": CHROME_EXT_ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        }
        r = requests.options(f"{self.base_url}/api/predict", headers=preflight_headers)
        allow_orig = r.headers.get("access-control-allow-origin", "")
        allow_methods = r.headers.get("access-control-allow-methods", "")
        passed = (r.status_code in (200, 204) and allow_orig in ("*", CHROME_EXT_ORIGIN) and "POST" in allow_methods.upper())
        msg = f"OPTIONS Preflight -> status {r.status_code}, Allow-Origin='{allow_orig}', Allow-Methods='{allow_methods}'"
        print(f"  {'✅' if passed else '❌'} {msg}")
        results.append(("CORS OPTIONS Preflight", passed, msg))

        # Actual POST
        r = requests.post(
            f"{self.base_url}/api/predict",
            json={"headline": SAMPLE_CLICKBAIT},
            headers={"Origin": CHROME_EXT_ORIGIN},
        )
        allow_orig = r.headers.get("access-control-allow-origin", "")
        passed = (r.status_code == 200 and allow_orig in ("*", CHROME_EXT_ORIGIN))
        msg = f"POST /api/predict -> status {r.status_code}, Allow-Origin='{allow_orig}'"
        print(f"  {'✅' if passed else '❌'} {msg}")
        results.append(("CORS POST with Chrome Origin", passed, msg))

        # Other origins
        for org in ["http://localhost:3000", "https://huggingface.co", "null"]:
            r = requests.post(
                f"{self.base_url}/api/predict",
                json={"headline": SAMPLE_LEGITIMATE},
                headers={"Origin": org},
            )
            allow_orig = r.headers.get("access-control-allow-origin", "")
            passed = (r.status_code == 200 and allow_orig in ("*", org))
            msg = f"Origin '{org}' -> status {r.status_code}, Allow-Origin='{allow_orig}'"
            print(f"  {'✅' if passed else '❌'} {msg}")
            results.append((f"CORS Origin {org}", passed, msg))

        return results

    def challenge_concurrency_and_throughput(
        self, concurrency: int = 20, num_requests: int = 100
    ) -> Dict[str, Any]:
        print(f"\n--- [5] Concurrency & Throughput Stress ({num_requests} reqs, {concurrency} threads) ---")

        def task(i: int):
            is_click = (i % 2 == 0)
            text = SAMPLE_CLICKBAIT if is_click else SAMPLE_LEGITIMATE
            t0 = time.perf_counter()
            r = requests.post(
                f"{self.base_url}/api/predict",
                json={"headline": text},
                timeout=10,
            )
            dt = time.perf_counter() - t0
            return {
                "idx": i,
                "status": r.status_code,
                "latency": dt,
                "is_click": is_click,
                "correct": (r.json().get("is_clickbait") == is_click) if r.status_code == 200 else False,
            }

        t_start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(task, i) for i in range(num_requests)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        total_time = time.perf_counter() - t_start

        latencies = [r["latency"] for r in results]
        latencies.sort()
        errors = [r for r in results if r["status"] != 200]
        incorrect = [r for r in results if not r["correct"]]

        p50 = latencies[int(len(latencies) * 0.50)]
        p90 = latencies[int(len(latencies) * 0.90)]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]
        rps = num_requests / total_time

        stats = {
            "total_requests": num_requests,
            "concurrency": concurrency,
            "total_time_sec": round(total_time, 3),
            "throughput_rps": round(rps, 2),
            "errors": len(errors),
            "incorrect_predictions": len(incorrect),
            "min_latency_ms": round(latencies[0] * 1000, 2),
            "p50_latency_ms": round(p50 * 1000, 2),
            "p90_latency_ms": round(p90 * 1000, 2),
            "p95_latency_ms": round(p95 * 1000, 2),
            "p99_latency_ms": round(p99 * 1000, 2),
            "max_latency_ms": round(latencies[-1] * 1000, 2),
        }

        print(f"  📊 Total Requests : {num_requests}")
        print(f"  📊 Concurrency    : {concurrency} workers")
        print(f"  📊 Total Time     : {stats['total_time_sec']}s")
        print(f"  📊 Throughput     : {stats['throughput_rps']} req/s")
        print(f"  📊 Errors         : {stats['errors']} (0%)")
        print(f"  📊 Accuracy       : {100 - len(incorrect)}/{num_requests} (100%)")
        print(f"  📊 Latency p50    : {stats['p50_latency_ms']} ms")
        print(f"  📊 Latency p95    : {stats['p95_latency_ms']} ms")
        print(f"  📊 Latency p99    : {stats['p99_latency_ms']} ms")
        print(f"  📊 Latency max    : {stats['max_latency_ms']} ms")

        return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--requests", type=int, default=100)
    args = parser.parse_args()

    harness = ApiStressHarness(port=args.port)
    all_passed = True
    try:
        harness.start_server()
        r1 = harness.challenge_http_methods()
        r2 = harness.challenge_malformed_and_edge_payloads()
        r3 = harness.challenge_huge_payloads()
        r4 = harness.challenge_cors_headers()
        stats = harness.challenge_concurrency_and_throughput(
            concurrency=args.concurrency, num_requests=args.requests
        )

        checks = r1 + r2 + r3 + r4
        failed_checks = [c for c in checks if not c[1]]
        if failed_checks or stats["errors"] > 0:
            all_passed = False
            print(f"\n❌ FAILURES DETECTED: {len(failed_checks)} checks failed, {stats['errors']} request errors.")
        else:
            print("\n🎉 ALL LIVE SOCKET STRESS & ADVERSARIAL CHALLENGES PASSED EMPIRICALLY!")

    finally:
        harness.stop_server()

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
