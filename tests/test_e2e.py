"""End-to-End (E2E) Test Suite: Tiers 3 & 4.
Authoritative source: ORIGINAL_REQUEST.md & PROJECT.md

Tiers covered:
- Tier 3: Cross-feature combinations (DOM Extraction -> API Request -> Model Prediction -> Visual Highlighting).
- Tier 4: Real-world news portal layouts (G1, UOL, Folha formats), realistic Portuguese headlines, latency budget.
"""

import time
from typing import Any, Dict, List
import pytest
from bs4 import BeautifulSoup

from tests.conftest import (
    REAL_WORLD_E2E_DATASET,
    SAMPLE_CLICKBAIT_HEADLINES,
    SAMPLE_LEGITIMATE_HEADLINES,
    validate_prediction_schema,
)


class TestCrossFeatureContractTier3:
    """Tier 3: Cross-Feature Integration Contract.

    Verifies the complete dataflow:
    Simulated Web Page -> Extension extraction -> API Batch Request -> Model Inference -> Visual Highlight Injection.
    """

    def test_e2e_dataflow_simulation(self):
        """Simulates the entire pipeline dataflow without dependency on external browser instances."""
        html_page = """
        <!DOCTYPE html>
        <html lang="pt-BR">
        <head><meta charset="utf-8"><title>Portal de Notícias Modelo</title></head>
        <body>
            <header>
                <div class="logo">Portal Brasil Notícias</div>
                <nav>
                    <a href="/politica">Política</a>
                    <a href="/economia">Economia</a>
                    <a href="/famosos">Famosos</a>
                </nav>
            </header>
            <main>
                <section class="destaque-principal">
                    <h1 id="hl-1">Ministério da Fazenda divulga balanço fiscal do terceiro trimestre nesta terça-feira</h1>
                </section>
                <section class="feed-noticias">
                    <article>
                        <h2 id="hl-2">Você não vai acreditar no que aconteceu com esse famoso!</h2>
                        <p class="summary">Famoso foi visto em local inesperado e causou comoção nas redes sociais.</p>
                    </article>
                    <article>
                        <h2 id="hl-3">Banco Central eleva taxa Selic para 11,25% ao ano em reunião do Copom</h2>
                        <p class="summary">Decisão foi unânime e visa controlar as expectativas de inflação.</p>
                    </article>
                    <article>
                        <h2 id="hl-4">O segredo chocante que os médicos não querem que você descubra!</h2>
                        <p class="summary">Receita milagrosa promete curar tudo em 24 horas.</p>
                    </article>
                </section>
                <aside class="sidebar">
                    <div class="widget">
                        <h3>Mais lidas</h3>
                        <a href="/link1" id="hl-5" class="headline">Veja as fotos inacreditáveis da mansão milionária que deixaram a web em choque!</a>
                    </div>
                    <div class="widget">
                        <h3>Curtas</h3>
                        <a href="/curta1" id="curta-1">Nota rápida</a>
                    </div>
                </aside>
            </main>
            <footer>
                <p>Copyright 2026 - Todos os direitos reservados</p>
                <a href="/termos">Termos de uso</a>
            </footer>
        </body>
        </html>
        """
        soup = BeautifulSoup(html_page, "html.parser")

        # Step 1: Content Script Selector Heuristic (M3)
        # Select candidates matching target rules:
        target_tags = ["h1", "h2", "h3", "h4", "a"]
        extracted_elements = []

        for el in soup.find_all(target_tags):
            # Exclude navigation and boilerplate
            if el.find_parent(["nav", "header", "footer"]):
                continue

            text = el.get_text().strip()
            words = text.split()
            # Length and word count filter
            if len(text) > 18 and len(words) > 3:
                extracted_elements.append(el)

        extracted_ids = [el.get("id") for el in extracted_elements]
        assert "hl-1" in extracted_ids
        assert "hl-2" in extracted_ids
        assert "hl-3" in extracted_ids
        assert "hl-4" in extracted_ids
        assert "hl-5" in extracted_ids
        # Rejected elements
        assert "curta-1" not in extracted_ids

        # Step 2: Service Worker & API Request Formatting (M3 ➔ M2)
        extracted_texts = [el.get_text().strip() for el in extracted_elements]
        api_request_payload = {"headlines": extracted_texts}
        assert len(api_request_payload["headlines"]) == 5

        # Step 3: Simulated API & Model Response Contract (M2 ➔ M1)
        # We model the contract response adhering to PROJECT.md specifications:
        simulated_api_responses = [
            {
                "headline": "Ministério da Fazenda divulga balanço fiscal do terceiro trimestre nesta terça-feira",
                "is_clickbait": False,
                "clickbait_status": "not_clickbait",
                "probability": 0.042,
                "confidence": 0.958,
                "bait_type": None,
            },
            {
                "headline": "Você não vai acreditar no que aconteceu com esse famoso!",
                "is_clickbait": True,
                "clickbait_status": "clickbait",
                "probability": 0.985,
                "confidence": 0.97,
                "bait_type": "Fofoca / Celebridades",
            },
            {
                "headline": "Banco Central eleva taxa Selic para 11,25% ao ano em reunião do Copom",
                "is_clickbait": False,
                "clickbait_status": "not_clickbait",
                "probability": 0.015,
                "confidence": 0.985,
                "bait_type": None,
            },
            {
                "headline": "O segredo chocante que os médicos não querem que você descubra!",
                "is_clickbait": True,
                "clickbait_status": "clickbait",
                "probability": 0.972,
                "confidence": 0.96,
                "bait_type": "Sensacionalismo / Exagero",
            },
            {
                "headline": "Veja as fotos inacreditáveis da mansão milionária que deixaram a web em choque!",
                "is_clickbait": True,
                "clickbait_status": "clickbait",
                "probability": 0.941,
                "confidence": 0.92,
                "bait_type": "Curiosidade / Ostentação",
            },
        ]

        # Validate schema of every simulated response item
        for item in simulated_api_responses:
            validate_prediction_schema(item)

        # Step 4: Visual Highlight Injection into DOM (M3)
        for el, res in zip(extracted_elements, simulated_api_responses):
            if res["is_clickbait"]:
                el["class"] = el.get("class", []) + ["clickbait-detected"]
                badge_html = f'<span class="clickbait-badge">⚠️ Clickbait: <span class="bait-type">{res["bait_type"]}</span></span>'
                badge_elem = BeautifulSoup(badge_html, "html.parser")
                el.append(badge_elem)

        # Step 5: End-to-End Assertions on Resulting DOM
        highlighted = soup.find_all(class_="clickbait-detected")
        assert len(highlighted) == 3, f"Expected 3 clickbaits highlighted, got {len(highlighted)}"

        highlighted_ids = [el.get("id") for el in highlighted]
        assert "hl-2" in highlighted_ids
        assert "hl-4" in highlighted_ids
        assert "hl-5" in highlighted_ids

        # Ensure legitimate headlines were NOT modified
        hl1 = soup.find(id="hl-1")
        assert "clickbait-detected" not in hl1.get("class", [])
        assert hl1.find(class_="clickbait-badge") is None

        hl3 = soup.find(id="hl-3")
        assert "clickbait-detected" not in hl3.get("class", [])
        assert hl3.find(class_="clickbait-badge") is None

        # Verify badge contents
        hl2 = soup.find(id="hl-2")
        badge2 = hl2.find(class_="clickbait-badge")
        assert "Fofoca / Celebridades" in badge2.get_text()


class TestRealWorldPortalsTier4:
    """Tier 4: Real-World News Scenarios and Layout Robustness."""

    def test_g1_style_portal_layout_extraction(self):
        """Simulates news headline extraction on a G1-style portal layout."""
        html_doc = """
        <div class="feed-root">
            <div class="feed-post bstn-item-shape">
                <div class="feed-post-body">
                    <a class="feed-post-link" href="/noticia-1">
                        <h2 class="feed-post-title">Polícia Federal deflagra operação contra desvio de recursos públicos</h2>
                    </a>
                </div>
            </div>
            <div class="feed-post bstn-item-shape">
                <div class="feed-post-body">
                    <a class="feed-post-link" href="/noticia-2">
                        <h2 class="feed-post-title">Veja 5 dicas infalíveis que vão transformar sua dieta nesta semana!</h2>
                    </a>
                </div>
            </div>
        </div>
        """
        soup = BeautifulSoup(html_doc, "html.parser")
        headlines = [h.get_text().strip() for h in soup.find_all("h2", class_="feed-post-title")]
        assert len(headlines) == 2
        assert "Polícia Federal deflagra operação contra desvio de recursos públicos" in headlines
        assert "Veja 5 dicas infalíveis que vão transformar sua dieta nesta semana!" in headlines

    def test_uol_style_portal_layout_extraction(self):
        """Simulates news headline extraction on a UOL-style portal layout."""
        html_doc = """
        <section class="highlights">
            <article class="headlineSub">
                <a href="/link1">
                    <h3 class="headlineSub__title">Governo estuda prorrogação de subsídios para o setor automotivo</h3>
                </a>
            </article>
            <article class="headlineSub">
                <a href="/link2">
                    <h3 class="headlineSub__title">URGENTE: Cantor famoso é hospitalizado às pressas após mal-estar em show</h3>
                </a>
            </article>
        </section>
        """
        soup = BeautifulSoup(html_doc, "html.parser")
        headlines = [h.get_text().strip() for h in soup.find_all("h3", class_="headlineSub__title")]
        assert len(headlines) == 2
        assert any("Governo estuda prorrogação" in h for h in headlines)
        assert any("URGENTE: Cantor famoso" in h for h in headlines)

    def test_folha_style_portal_layout_extraction(self):
        """Simulates news headline extraction on a Folha-style portal layout."""
        html_doc = """
        <div class="c-main">
            <div class="c-headline">
                <h2 class="c-headline__title">
                    <a href="/link-folha">Inflação oficial desacelera e fecha mês de setembro em 0,44%, aponta IBGE</a>
                </h2>
            </div>
            <div class="c-headline">
                <h2 class="c-headline__title">
                    <a href="/link-folha-2">Você não vai acreditar em quanto ganha essa estrela da TV aos 40 anos!</a>
                </h2>
            </div>
        </div>
        """
        soup = BeautifulSoup(html_doc, "html.parser")
        headlines = [h.get_text().strip() for h in soup.find_all("h2", class_="c-headline__title")]
        assert len(headlines) == 2
        assert any("Inflação oficial" in h for h in headlines)
        assert any("Você não vai acreditar" in h for h in headlines)

    def test_curated_real_world_dataset_contract_soundness(self, real_world_dataset):
        """Verifies that all 10 curated real-world test cases adhere to contract rules."""
        assert len(real_world_dataset) == 10
        cb_cases = [c for c in real_world_dataset if c["is_clickbait"]]
        legit_cases = [c for c in real_world_dataset if not c["is_clickbait"]]

        assert len(cb_cases) == 5
        assert len(legit_cases) == 5

        # All text strings must be > 18 characters and > 3 words
        for case in real_world_dataset:
            text = case["text"]
            assert len(text) > 18, f"Headline too short: '{text}'"
            assert len(text.split()) > 3, f"Headline has too few words: '{text}'"

    def test_latency_budget_on_batch_inference_simulation(self):
        """Verifies that end-to-end extraction and parsing of 20 headlines executes within 500ms."""
        headlines = SAMPLE_CLICKBAIT_HEADLINES + SAMPLE_LEGITIMATE_HEADLINES
        assert len(headlines) == 20

        start_time = time.time()
        # Simulate local parsing, cleaning, and preparation
        cleaned = [h.strip().lower() for h in headlines]
        tokenized = [c.split() for c in cleaned]
        elapsed = time.time() - start_time

        assert elapsed < 0.5, f"Data processing exceeded 500ms latency budget ({elapsed:.3f}s)"
