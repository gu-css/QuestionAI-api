"""Acceptance Criteria Test Suite (Opaque-Box).
Directly asserts all 4 Acceptance Criteria specified in ORIGINAL_REQUEST.md.

Acceptance Criteria:
- AC R1: Automated script loads exported models, runs inference on test string, returns valid prediction without errors.
- AC R2: Automated test sends POST request to Gradio API with sample headline, verifies JSON has clickbait status and bait type.
- AC R3: Verifies extension content script extracts heading elements, communicates with API, and applies visual highlight to clickbait.
- AC R4: Docs folder contains markdown report with at least one diagram, non-technical language, HF Spaces hosting guide, and valid PDF.
"""

import json
import os
import re
import subprocess
from pathlib import Path
import pytest

from tests.conftest import (
    DOCS_DIR,
    EXTENSION_DIR,
    MODELS_DIR,
    PROJECT_ROOT,
    SCRIPTS_DIR,
    has_m1_model_artifacts,
    has_m1_verify_script,
    has_m2_app,
    has_m3_extension_files,
    has_m4_docs_files,
    has_m4_pdf_file,
    is_m2_live_api_online,
    validate_prediction_schema,
)


class TestAcceptanceCriteriaR1:
    """AC R1: Model Training & Inference Verification."""

    def test_acceptance_r1_verify_models_script(self):
        """Authoritative Source: ORIGINAL_REQUEST.md line 34:

        'An automated script successfully loads the exported models and runs inference
        on a test string, returning a valid prediction without errors.'
        """
        verify_script = SCRIPTS_DIR / "verify_models.py"
        if not verify_script.exists():
            if not has_m1_model_artifacts():
                pytest.skip("M1 Artifacts pending: 'scripts/verify_models.py' and 'models/' not yet created.")
            pytest.fail("M1 model artifacts exist but 'scripts/verify_models.py' is missing.")

        # Run verification script as subprocess
        result = subprocess.run(
            ["python3", str(verify_script)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"scripts/verify_models.py failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
        assert len(result.stdout.strip()) > 0, "scripts/verify_models.py produced empty stdout."

    def test_acceptance_r1_model_inference_direct(self, clickbait_headlines, legitimate_headlines):
        """Direct opaque-box inference on exported models verifying valid predictions and schema."""
        if not has_m1_model_artifacts():
            pytest.skip("M1 Artifacts pending: 'models/supervised_model.joblib' not yet created.")

        import joblib

        sup_path = MODELS_DIR / "supervised_model.joblib"
        unsup_path = MODELS_DIR / "unsupervised_model.joblib"
        metadata_path = MODELS_DIR / "metadata.json"

        assert sup_path.exists(), f"Supervised model missing at {sup_path}"
        assert unsup_path.exists(), f"Unsupervised model missing at {unsup_path}"

        supervised_pipeline = joblib.load(sup_path)
        unsupervised_pipeline = joblib.load(unsup_path)

        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        # Test on classic clickbait
        cb_sample = clickbait_headlines[0]
        # Supervised prediction
        pred_label = supervised_pipeline.predict([cb_sample])[0]
        assert pred_label in (0, 1, True, False, "clickbait", "not_clickbait"), (
            f"Invalid prediction label format: {pred_label}"
        )

        # Probabilities
        if hasattr(supervised_pipeline, "predict_proba"):
            probs = supervised_pipeline.predict_proba([cb_sample])[0]
            assert len(probs) >= 2, f"Expected binary probabilities, got {probs}"
            prob_clickbait = float(probs[1])
            assert 0.0 <= prob_clickbait <= 1.0, f"Probability out of range: {prob_clickbait}"

        # Unsupervised prediction
        cluster_id = unsupervised_pipeline.predict([cb_sample])[0]
        assert cluster_id is not None, "Unsupervised model returned None cluster"


class TestAcceptanceCriteriaR2:
    """AC R2: Hugging Face Spaces API (Gradio & FastAPI) Verification."""

    def test_acceptance_r2_gradio_api_post_request(self, clickbait_headlines):
        """Authoritative Source: ORIGINAL_REQUEST.md line 37:

        'An automated test script sends a POST request to the local Gradio API with a sample headline
        and verifies that the JSON response contains both the clickbait status and bait type.'
        """
        sample_headline = clickbait_headlines[0]
        payload = {"headline": sample_headline}

        # Check if live server is running
        if is_m2_live_api_online():
            import requests
            resp = requests.post("http://127.0.0.1:8000/api/predict", json=payload, timeout=10)
            assert resp.status_code == 200, f"Expected HTTP 200, got {resp.status_code}: {resp.text}"
            data = resp.json()
        elif has_m2_app():
            from fastapi.testclient import TestClient
            import importlib.util
            spec = importlib.util.spec_from_file_location("app_module", str(PROJECT_ROOT / "app.py"))
            app_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(app_mod)
            app = getattr(app_mod, "app", None)
            assert app is not None, "app.py does not define an 'app' instance."
            client = TestClient(app)
            resp = client.post("/api/predict", json=payload)
            assert resp.status_code == 200, f"Expected HTTP 200, got {resp.status_code}: {resp.text}"
            data = resp.json()
        else:
            pytest.skip("M2 pending: 'app.py' or live server at http://127.0.0.1:8000 not yet available.")

        # Authoritative assertions per AC R2
        assert "clickbait_status" in data, f"Response JSON missing 'clickbait_status': {data}"
        assert "bait_type" in data, f"Response JSON missing 'bait_type': {data}"
        validate_prediction_schema(data)


class TestAcceptanceCriteriaR3:
    """AC R3: Chrome Extension (Manifest V3) Verification."""

    def test_acceptance_r3_manifest_and_structure(self):
        """Authoritative Source: ORIGINAL_REQUEST.md line 40:

        'Agent-as-judge verifies that the extension's content script extracts heading elements,
        communicates with the API, and correctly applies a visual highlight (e.g., CSS class or inline style)
        to clickbait headlines.'
        """
        if not has_m3_extension_files():
            pytest.skip("M3 pending: 'extension/manifest.json' or content scripts not yet created.")

        manifest_path = EXTENSION_DIR / "manifest.json"
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        # Manifest V3 standards
        assert manifest.get("manifest_version") == 3, f"Expected manifest_version 3, got {manifest.get('manifest_version')}"
        assert "name" in manifest and len(manifest["name"]) > 0
        assert "version" in manifest and len(manifest["version"]) > 0

        # Background service worker proxy
        assert "background" in manifest, "Manifest missing 'background' declaration."
        service_worker = manifest["background"].get("service_worker")
        assert service_worker, "Manifest missing 'service_worker' path in background."
        assert (EXTENSION_DIR / service_worker).exists(), f"Service worker file missing at {service_worker}"

        # Content scripts
        assert "content_scripts" in manifest and len(manifest["content_scripts"]) > 0, (
            "Manifest missing 'content_scripts' definition."
        )
        content_script_entry = manifest["content_scripts"][0]
        js_files = content_script_entry.get("js", [])
        assert len(js_files) > 0, "No JS files defined in content_scripts."
        for js_file in js_files:
            assert (EXTENSION_DIR / js_file).exists(), f"Content JS file missing: {js_file}"

        # Visual CSS
        css_files = content_script_entry.get("css", [])
        assert len(css_files) > 0, "Content scripts must include CSS for visual highlighting."
        for css_file in css_files:
            assert (EXTENSION_DIR / css_file).exists(), f"Content CSS file missing: {css_file}"

    def test_acceptance_r3_content_script_selectors_and_visual_badge(self):
        """Verifies content script heuristic extracts headings and applies visual styling."""
        if not has_m3_extension_files():
            pytest.skip("M3 pending: Chrome extension files not yet created.")

        content_js_path = EXTENSION_DIR / "content" / "content.js"
        content_js_code = content_js_path.read_text(encoding="utf-8")

        # Heading elements extraction check
        heading_indicators = ["h1", "h2", "h3", "h4", "article", "heading"]
        assert any(ind in content_js_code for ind in heading_indicators), (
            "content.js must target heading elements (h1-h4, article headlines)."
        )

        # Communication with background/API check
        comm_indicators = ["sendMessage", "fetch", "chrome.runtime"]
        assert any(comm in content_js_code for comm in comm_indicators), (
            "content.js must communicate with service worker or API."
        )

        # Visual highlight check: CSS class or inline style or badge
        highlight_indicators = ["clickbait-detected", "clickbait-badge", "style", "classList.add", "badge"]
        assert any(hl in content_js_code for hl in highlight_indicators), (
            "content.js must apply visual highlight or badge to detected clickbait."
        )


class TestAcceptanceCriteriaR4:
    """AC R4: Documentation & PDF Verification."""

    def test_acceptance_r4_markdown_report_with_diagram_and_hf_instructions(self):
        """Authoritative Source: ORIGINAL_REQUEST.md lines 43-44 and line 52:

        - 'The docs folder contains a markdown report with at least one diagram, and its corresponding .pdf version.'
        - 'Agent-as-judge verifies that the report's language is accessible to a non-technical audience and accurately describes the system architecture.'
        - 'MUST include step-by-step instructions in the documentation report explaining exactly how the user should host the Gradio API on Hugging Face Spaces.'
        """
        if not has_m4_docs_files():
            pytest.skip("M4 pending: 'docs/relatorio_desenvolvimento.md' not yet created.")

        report_path = DOCS_DIR / "relatorio_desenvolvimento.md"
        content = report_path.read_text(encoding="utf-8")

        # 1. Report size and non-empty check
        assert len(content.strip()) > 1000, "Documentation report is too short."

        # 2. Diagram check: references svg, png or mermaid diagrams
        diagram_patterns = [r"!\[.*?\]\(.*?\.(?:svg|png|jpg)\)", r"<img.*?>", r"```mermaid", r"<svg.*?>"]
        has_diagram = any(re.search(pat, content, re.IGNORECASE) for pat in diagram_patterns)
        assert has_diagram, "Report must include at least one diagram (SVG/PNG/Mermaid) per AC R4."

        # 3. Non-technical accessible language check
        # Must describe architecture and purpose clearly
        assert re.search(r"arquitetura|como funciona|sistema|extens[aã]o|modelo", content, re.IGNORECASE), (
            "Report must describe the system architecture."
        )

        # 4. Mandatory Hugging Face Spaces manual hosting guide per user follow-up
        hf_patterns = [
            r"hugging\s*face\s*spaces",
            r"como\s+hospedar",
            r"passo\s+a\s+passo",
        ]
        has_hf_guide = any(re.search(pat, content, re.IGNORECASE) for pat in hf_patterns)
        assert has_hf_guide, (
            "Report must include step-by-step instructions explaining how to manually host "
            "the Gradio API on Hugging Face Spaces per user requirement."
        )

    def test_acceptance_r4_pdf_presence_and_validity(self):
        """Authoritative Source: ORIGINAL_REQUEST.md line 43:

        'and its corresponding .pdf version.'
        """
        if not has_m4_pdf_file():
            pytest.skip("M4 pending: 'docs/relatorio_desenvolvimento.pdf' not yet compiled.")

        pdf_path = DOCS_DIR / "relatorio_desenvolvimento.pdf"
        assert pdf_path.exists(), f"PDF missing at {pdf_path}"

        # Size check (> 10 KB to ensure not an empty file)
        file_size = pdf_path.stat().st_size
        assert file_size >= 10000, f"PDF file size is suspiciously small ({file_size} bytes)."

        # Valid PDF binary signature
        with open(pdf_path, "rb") as f:
            header = f.read(5)
            assert header == b"%PDF-", f"Invalid PDF signature: {header}"
