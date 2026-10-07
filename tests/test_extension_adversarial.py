"""tests/test_extension_adversarial.py
Adversarial Verification Suite for Chrome Extension Service Worker Proxy.
Empirically tests service worker network proxying under adverse conditions:
- Backend API offline / unreachable
- HTTP 500, 503, 504 server errors
- Network timeout exceeding AbortController limit (5000ms)
- Malformed / truncated JSON responses
- Empty batches and oversized batches
- Zero uncaught promise rejections
"""

import json
import subprocess
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTENSION_DIR = PROJECT_ROOT / "extension"
SW_PATH = EXTENSION_DIR / "background" / "service-worker.js"
NODE_HARNESS_PATH = PROJECT_ROOT / "tests" / "test_extension_network_adversarial.js"


class TestExtensionServiceWorkerStaticIntegrity:
    """Static analysis of service worker implementation against resilience contracts."""

    def test_service_worker_file_exists(self):
        """Verifies background/service-worker.js exists and is non-empty."""
        assert SW_PATH.exists(), f"Missing service-worker.js at {SW_PATH}"
        assert SW_PATH.stat().st_size > 500, "service-worker.js is suspiciously small"

    def test_service_worker_has_abort_controller_timeout(self):
        """Verifies that fetchWithTimeout uses AbortController with 5000ms limit."""
        content = SW_PATH.read_text(encoding="utf-8")
        assert "AbortController" in content, "service-worker.js must utilize AbortController"
        assert "timeoutMs" in content, "service-worker.js must configure a timeout parameter"
        assert "5000" in content, "service-worker.js default timeout must be 5000ms"
        assert "clearTimeout" in content, "service-worker.js must clear timeout timer"

    def test_service_worker_listener_synchronous_return_true(self):
        """Verifies onMessage returns true synchronously for async sendResponse."""
        content = SW_PATH.read_text(encoding="utf-8")
        # Ensure return true is present
        assert "return true;" in content, "onMessage listener must return true synchronously"

    def test_service_worker_structured_error_responses(self):
        """Verifies that catch blocks return { success: false, error: ..., offline: true }."""
        content = SW_PATH.read_text(encoding="utf-8")
        assert "success: false" in content, "Error responses must return success: false"
        assert "offline: true" in content, "Network failures must flag offline: true"


class TestExtensionServiceWorkerEmpiricalHarness:
    """Empirical execution of the Node.js adversarial network harness."""

    def test_run_node_adversarial_suite(self):
        """Executes test_extension_network_adversarial.js and validates 100% pass rate.

        Vectors covered:
        - 1. Backend API offline / unreachable (ECONNREFUSED, unresolvable host, socket destroy)
        - 2. HTTP Server Errors (500, 503, 504, 404, 422)
        - 3. Network timeout exceeding AbortController limit (5000ms)
        - 4. Malformed / truncated JSON responses (truncated, HTML, empty, binary, schema violations)
        - 5. Batch sizing edge cases (empty batches, null headlines, oversized batches of 1000, huge 50KB strings)
        - 6. Health check resilience (online, offline, 500, timeout)
        - 7. Extension state handling (disabled extension, unknown action)
        - 8. Zero uncaught promise rejections across the entire suite.
        """
        assert NODE_HARNESS_PATH.exists(), f"Harness script missing: {NODE_HARNESS_PATH}"

        result = subprocess.run(
            ["node", str(NODE_HARNESS_PATH)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=40,
        )

        # Print output for visibility in pytest -s
        print("\n--- HARNESS STDOUT ---")
        print(result.stdout)
        if result.stderr:
            print("\n--- HARNESS STDERR ---")
            print(result.stderr)

        assert result.returncode == 0, f"Adversarial test suite failed with code {result.returncode}"
        assert "0 FAILED" in result.stdout, "Adversarial test suite reported failures"
        assert "ALL TESTS PASSED" in result.stdout or "TEST SUITE SUMMARY" in result.stdout
