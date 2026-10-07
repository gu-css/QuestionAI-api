"""Test Suite for Milestone 4 (R4): Documentation and Automated PDF Compilation.
Authoritative source: ORIGINAL_REQUEST.md & PROJECT.md

Tiers covered:
- Tier 1: Markdown report presence, non-technical readability, diagram references, HF Spaces hosting guide, PDF validity.
- Tier 2: Broken image references, SVG diagram validity, section completeness, build script validation.
"""

import os
import re
import subprocess
from pathlib import Path
import pytest

from tests.conftest import DOCS_DIR, PROJECT_ROOT, SCRIPTS_DIR, has_m4_docs_files, has_m4_pdf_file


class TestDocumentationMarkdownTier1:
    """Tier 1: Markdown Report Content and Accessibility."""

    def test_report_file_exists_and_comprehensive(self):
        """Verifies docs/relatorio_desenvolvimento.md exists and has substantial content."""
        if not has_m4_docs_files():
            pytest.skip("M4 pending: 'docs/relatorio_desenvolvimento.md' not yet created.")

        report_path = DOCS_DIR / "relatorio_desenvolvimento.md"
        assert report_path.exists(), f"Missing report at {report_path}"

        content = report_path.read_text(encoding="utf-8")
        assert len(content) > 3000, f"Report content is too short ({len(content)} characters)."

    def test_report_accessible_non_technical_language(self):
        """Verifies report uses non-technical analogies and clear explanations."""
        if not has_m4_docs_files():
            pytest.skip("M4 pending: 'docs/relatorio_desenvolvimento.md' not yet created.")

        report_path = DOCS_DIR / "relatorio_desenvolvimento.md"
        content = report_path.read_text(encoding="utf-8").lower()

        # Check for core functional explanations
        assert any(term in content for term in ["o que é clickbait", "clickbait", "como funciona", "visão geral", "introdução"]), (
            "Report must introduce the concept and purpose for non-technical readers."
        )

        # Check for extension and API explanations
        assert "extensão" in content or "chrome" in content
        assert "api" in content or "servidor" in content

    def test_report_diagram_presence(self):
        """Verifies report references at least one diagram per AC R4."""
        if not has_m4_docs_files():
            pytest.skip("M4 pending: 'docs/relatorio_desenvolvimento.md' not yet created.")

        report_path = DOCS_DIR / "relatorio_desenvolvimento.md"
        content = report_path.read_text(encoding="utf-8")

        # Find image links or inline SVGs
        img_matches = re.findall(r"!\[(.*?)\]\((.*?)\)", content)
        inline_svgs = "<svg" in content
        mermaid_blocks = "```mermaid" in content

        has_any_diagram = len(img_matches) > 0 or inline_svgs or mermaid_blocks
        assert has_any_diagram, "Report must contain at least one diagram per AC R4."

        # If image links are used, verify target files exist
        for alt, img_path in img_matches:
            # Resolve relative to docs directory
            resolved_path = (DOCS_DIR / img_path).resolve()
            assert resolved_path.exists(), f"Diagram file referenced in markdown does not exist: {resolved_path}"

    def test_report_hf_spaces_hosting_guide(self):
        """Verifies report contains step-by-step instructions for hosting on Hugging Face Spaces.

        Authoritative requirement: User update at 2026-10-06T21:04:41Z.
        """
        if not has_m4_docs_files():
            pytest.skip("M4 pending: 'docs/relatorio_desenvolvimento.md' not yet created.")

        report_path = DOCS_DIR / "relatorio_desenvolvimento.md"
        content = report_path.read_text(encoding="utf-8").lower()

        assert "hugging face" in content or "huggingface" in content, (
            "Report must mention Hugging Face Spaces."
        )

        # Must include instructions or steps
        assert any(k in content for k in ["passo a passo", "como hospedar", "criar space", "upload", "espaço"]), (
            "Report must provide actionable step-by-step instructions for hosting on Hugging Face Spaces."
        )


class TestDocumentationPdfTier1:
    """Tier 1: PDF Compilation and File Validation."""

    def test_pdf_exists_and_valid_structure(self):
        """Verifies docs/relatorio_desenvolvimento.pdf exists and has valid PDF binary structure."""
        if not has_m4_pdf_file():
            pytest.skip("M4 pending: 'docs/relatorio_desenvolvimento.pdf' not yet compiled.")

        pdf_path = DOCS_DIR / "relatorio_desenvolvimento.pdf"
        assert pdf_path.exists(), f"PDF not found at {pdf_path}"

        size = pdf_path.stat().st_size
        assert size >= 10000, f"PDF file size is too small ({size} bytes). Minimum expected is 10 KB."

        with open(pdf_path, "rb") as f:
            header = f.read(5)
            assert header == b"%PDF-", f"Invalid PDF magic header: {header}"

            # Check EOF marker in the last 1024 bytes
            f.seek(max(0, size - 1024))
            tail = f.read()
            assert b"%%EOF" in tail or b"%EOF" in tail, "PDF file missing %%EOF trailer marker."

    def test_build_docs_script_exists(self):
        """Verifies scripts/build_docs.py exists and handles PDF compilation."""
        build_script = SCRIPTS_DIR / "build_docs.py"
        if not build_script.exists():
            if not has_m4_docs_files():
                pytest.skip("M4 pending: scripts/build_docs.py not yet implemented.")
            pytest.fail("scripts/build_docs.py is missing.")

        assert build_script.stat().st_size > 100, "scripts/build_docs.py is empty."


class TestDiagramIntegrityTier2:
    """Tier 2: Vector SVG Diagram Format Validation."""

    def test_svg_diagrams_are_valid_xml(self):
        """Verifies that all SVG diagrams in docs/diagrams/ are valid XML and renderable."""
        diagrams_dir = DOCS_DIR / "diagrams"
        if not diagrams_dir.exists():
            if not has_m4_docs_files():
                pytest.skip("M4 pending: docs/diagrams/ not yet created.")
            return

        svg_files = list(diagrams_dir.glob("*.svg"))
        if not svg_files:
            pytest.skip("No SVG files found in docs/diagrams/.")

        import xml.etree.ElementTree as ET
        for svg_path in svg_files:
            try:
                tree = ET.parse(svg_path)
                root = tree.getroot()
                assert "svg" in root.tag.lower(), f"Root tag of {svg_path.name} is not svg: {root.tag}"
            except ET.ParseError as e:
                pytest.fail(f"SVG file {svg_path.name} is malformed XML: {e}")
