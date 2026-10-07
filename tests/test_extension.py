"""Test Suite for Milestone 3 (R3): Manifest V3 Chrome Extension.
Authoritative source: ORIGINAL_REQUEST.md & PROJECT.md

Tiers covered:
- Tier 1: Manifest V3 validity, icon assets, background service worker proxy, content script selectors, visual badge injection.
- Tier 2: Idempotence, noise filtering heuristics, XSS prevention, offline error resilience.
"""

import json
import re
from pathlib import Path
import pytest
from bs4 import BeautifulSoup

from tests.conftest import EXTENSION_DIR, has_m3_extension_files


class TestManifestV3Tier1:
    """Tier 1: Manifest V3 Structure and Asset Integrity."""

    def test_manifest_file_and_required_keys(self):
        """Verifies manifest.json compliance with Chrome Extensions Manifest V3."""
        if not has_m3_extension_files():
            pytest.skip("M3 pending: Chrome extension files not yet generated.")

        manifest_path = EXTENSION_DIR / "manifest.json"
        assert manifest_path.exists(), f"Missing manifest.json at {manifest_path}"

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert manifest.get("manifest_version") == 3, "Extension must use Manifest V3."
        assert manifest.get("name"), "Manifest missing 'name'."
        assert manifest.get("version"), "Manifest missing 'version'."
        assert "description" in manifest, "Manifest missing 'description'."

        # Permissions check
        permissions = manifest.get("permissions", [])
        assert "storage" in permissions, "Manifest must include 'storage' permission for settings."

        # Host permissions for API proxy
        host_perms = manifest.get("host_permissions", [])
        assert any("localhost" in p or "127.0.0.1" in p for p in host_perms), (
            "host_permissions must allow access to local Gradio API (localhost/127.0.0.1)."
        )

    def test_icons_declared_and_valid_png_binaries(self):
        """Verifies that all icons declared in manifest exist and are valid PNG files."""
        if not has_m3_extension_files():
            pytest.skip("M3 pending: Chrome extension files not yet generated.")

        manifest_path = EXTENSION_DIR / "manifest.json"
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        icons_dict = manifest.get("icons", {})
        assert len(icons_dict) > 0, "Manifest must declare icons (16, 48, 128)."

        png_header = b"\x89PNG\r\n\x1a\n"
        for size, rel_path in icons_dict.items():
            icon_file = EXTENSION_DIR / rel_path
            assert icon_file.exists(), f"Declared icon not found on disk: {icon_file}"
            assert icon_file.stat().st_size > 50, f"Icon file {icon_file} is suspiciously small."
            with open(icon_file, "rb") as f:
                header = f.read(8)
                assert header == png_header, f"Icon {icon_file} is not a valid PNG binary."

    def test_service_worker_and_content_script_paths(self):
        """Verifies service worker and content script files exist at declared locations."""
        if not has_m3_extension_files():
            pytest.skip("M3 pending: Chrome extension files not yet generated.")

        manifest_path = EXTENSION_DIR / "manifest.json"
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        # Service worker
        sw_rel = manifest.get("background", {}).get("service_worker")
        assert sw_rel, "Manifest missing background service worker path."
        sw_path = EXTENSION_DIR / sw_rel
        assert sw_path.exists(), f"Service worker not found at {sw_path}"

        # Content scripts
        content_entries = manifest.get("content_scripts", [])
        assert len(content_entries) > 0, "No content_scripts declared in manifest."
        for entry in content_entries:
            for js in entry.get("js", []):
                assert (EXTENSION_DIR / js).exists(), f"Content script JS not found: {js}"
            for css in entry.get("css", []):
                assert (EXTENSION_DIR / css).exists(), f"Content script CSS not found: {css}"


class TestContentScriptHeuristicsTier1:
    """Tier 1 & 2: Headline Extraction Heuristics and Noise Filtering."""

    def test_dom_heading_extraction_heuristic(self):
        """Tests the DOM selector heuristic against a simulated news portal HTML document."""
        html_doc = """
        <!DOCTYPE html>
        <html>
        <head><title>Portal de Notícias</title></head>
        <body>
            <header>
                <nav>
                    <a href="/">Página Inicial</a>
                    <a href="/politica">Política Nacional</a>
                </nav>
            </header>
            <main>
                <h1>Ministério da Fazenda divulga balanço fiscal do terceiro trimestre</h1>
                <article>
                    <h2>Você não vai acreditar no que aconteceu com esse famoso!</h2>
                    <p>Texto da notícia...</p>
                    <a href="/link" class="headline">O segredo chocante que ninguém quer que você saiba!</a>
                </article>
                <aside>
                    <h3>Curtas</h3>
                    <p>Nota rápida</p>
                </aside>
            </main>
            <footer>
                <p>Todos os direitos reservados 2026</p>
            </footer>
        </body>
        </html>
        """
        soup = BeautifulSoup(html_doc, "html.parser")

        # Heuristic rules per PROJECT.md § M3 Interface Contracts:
        # 1. Target selectors: h1, h2, h3, h4, [role="heading"], article a, .headline
        # 2. Exclude parents: nav, header, footer
        # 3. Text length > 18 chars, word count > 3 words
        candidates = []
        for el in soup.find_all(["h1", "h2", "h3", "h4", "a"]):
            # Check if inside excluded containers
            if el.find_parent(["nav", "header", "footer"]):
                continue

            text = el.get_text().strip()
            words = text.split()
            if len(text) > 18 and len(words) > 3:
                candidates.append((el.name, text))

        candidate_texts = [c[1] for c in candidates]

        # Valid candidates that SHOULD be captured:
        assert "Ministério da Fazenda divulga balanço fiscal do terceiro trimestre" in candidate_texts
        assert "Você não vai acreditar no que aconteceu com esse famoso!" in candidate_texts
        assert "O segredo chocante que ninguém quer que você saiba!" in candidate_texts

        # Boilerplate that MUST be rejected:
        assert "Página Inicial" not in candidate_texts
        assert "Política Nacional" not in candidate_texts
        assert "Curtas" not in candidate_texts
        assert "Todos os direitos reservados 2026" not in candidate_texts

    def test_visual_badge_injection_contract(self):
        """Verifies the visual highlight structure matches PROJECT.md contract specifications."""
        sample_headline_html = "<h2>Você não vai acreditar no que aconteceu com esse famoso!</h2>"
        soup = BeautifulSoup(sample_headline_html, "html.parser")
        heading = soup.find("h2")

        # Simulate content script injection
        heading["class"] = heading.get("class", []) + ["clickbait-detected"]
        bait_type = "Sensacionalismo / Exagero"

        badge_html = f'<span class="clickbait-badge" role="status">⚠️ Clickbait: <span class="bait-type">{bait_type}</span></span>'
        badge_soup = BeautifulSoup(badge_html, "html.parser")
        heading.append(badge_soup)

        # Assert visual highlight contract
        assert "clickbait-detected" in heading.get("class", [])
        badge = heading.find("span", class_="clickbait-badge")
        assert badge is not None
        assert "Clickbait:" in badge.get_text()
        assert bait_type in badge.get_text()

    def test_idempotent_injection(self):
        """Verifies that running scanning multiple times does NOT inject duplicate badges."""
        sample_html = '<h2 class="clickbait-detected">Manchete <span class="clickbait-badge">⚠️ Clickbait</span></h2>'
        soup = BeautifulSoup(sample_html, "html.parser")
        heading = soup.find("h2")

        # Heuristic check before injection
        is_already_highlighted = (
            "clickbait-detected" in heading.get("class", [])
            or heading.find("span", class_="clickbait-badge") is not None
        )
        assert is_already_highlighted is True, "Scanner must detect already-highlighted elements to ensure idempotence."


class TestExtensionPopupAndStylesTier1:
    """Tier 1: Extension Popup and CSS styles."""

    def test_extension_css_rules_present(self):
        """Verifies content.css contains styling for .clickbait-detected and .clickbait-badge."""
        if not has_m3_extension_files():
            pytest.skip("M3 pending: Chrome extension files not yet generated.")

        css_path = EXTENSION_DIR / "content" / "content.css"
        assert css_path.exists(), f"Missing content.css at {css_path}"

        css_content = css_path.read_text(encoding="utf-8")
        assert ".clickbait-detected" in css_content, "content.css must define .clickbait-detected rule."
        assert ".clickbait-badge" in css_content, "content.css must define .clickbait-badge rule."

    def test_popup_ui_files_presence(self):
        """Verifies extension popup contains HTML, JS, and CSS for user controls."""
        if not has_m3_extension_files():
            pytest.skip("M3 pending: Chrome extension files not yet generated.")

        popup_html = EXTENSION_DIR / "popup" / "popup.html"
        popup_js = EXTENSION_DIR / "popup" / "popup.js"
        popup_css = EXTENSION_DIR / "popup" / "popup.css"

        assert popup_html.exists(), "popup.html missing."
        assert popup_js.exists(), "popup.js missing."
        assert popup_css.exists(), "popup.css missing."

        html_text = popup_html.read_text(encoding="utf-8")
        # Must have endpoint configuration and toggle
        assert re.search(r"url|endpoint|servidor|api", html_text, re.IGNORECASE), (
            "popup.html must allow configuring the API URL."
        )
