#!/usr/bin/env python3
"""Automated Documentation & PDF Compilation Script.
Converts Markdown report (docs/relatorio_desenvolvimento.md) with vector diagrams
and custom print CSS into docs/relatorio_desenvolvimento.pdf.

Workflow:
1. Compiles Markdown to standalone HTML with embedded resources (SVG diagrams, CSS) via Pandoc.
2. Renders standalone HTML to print-perfect PDF using headless Google Chrome.
3. Provides automated fallback chains to XeLaTeX / pdfLaTeX or ReportLab if headless Chrome is unavailable.
4. Validates output PDF structure (header signature, trailer EOF, and minimum byte size).

Authoritative source: ORIGINAL_REQUEST.md AC R4 & PROJECT.md
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
DEFAULT_INPUT_MD = DOCS_DIR / "relatorio_desenvolvimento.md"
DEFAULT_OUTPUT_PDF = DOCS_DIR / "relatorio_desenvolvimento.pdf"
DEFAULT_CSS = DOCS_DIR / "style.css"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile technical documentation Markdown report to PDF."
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=DEFAULT_INPUT_MD,
        help=f"Path to input Markdown file (default: {DEFAULT_INPUT_MD})"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_OUTPUT_PDF,
        help=f"Path to output PDF file (default: {DEFAULT_OUTPUT_PDF})"
    )
    parser.add_argument(
        "--css", "-c",
        type=Path,
        default=DEFAULT_CSS,
        help=f"Path to custom print CSS file (default: {DEFAULT_CSS})"
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary HTML file for debugging."
    )
    return parser.parse_args()


def find_chrome_binary() -> str | None:
    """Detects available Google Chrome or Chromium binaries on system PATH."""
    candidates = [
        "google-chrome",
        "google-chrome-stable",
        "chromium",
        "chromium-browser",
        "/usr/bin/google-chrome",
        "/usr/bin/chromium",
    ]
    for candidate in candidates:
        bin_path = shutil.which(candidate)
        if bin_path:
            return bin_path
    return None


def validate_pdf_binary(pdf_path: Path) -> bool:
    """Validates that generated PDF meets structural integrity standards."""
    if not pdf_path.exists():
        print(f"❌ Erro: Arquivo PDF não encontrado em {pdf_path}")
        return False

    size = pdf_path.stat().st_size
    if size < 10000:
        print(f"❌ Erro: Tamanho do PDF insuficiente ({size} bytes). Mínimo esperado: 10.000 bytes.")
        return False

    with open(pdf_path, "rb") as f:
        header = f.read(5)
        if header != b"%PDF-":
            print(f"❌ Erro: Assinatura mágica de cabeçalho PDF inválida: {header}")
            return False

        f.seek(max(0, size - 1024))
        tail = f.read()
        if b"%%EOF" not in tail and b"%EOF" not in tail:
            print("❌ Erro: Marcador %%EOF de final de arquivo ausente no PDF.")
            return False

    return True


def compile_via_chrome(
    input_md: Path,
    output_pdf: Path,
    css_path: Path,
    chrome_bin: str,
    keep_temp: bool = False
) -> bool:
    """Primary pipeline: Pandoc standalone HTML + headless Chrome print-to-pdf."""
    print("🔹 Iniciando compilação primária via Pandoc + Headless Chrome...")

    pandoc_bin = shutil.which("pandoc")
    if not pandoc_bin:
        print("⚠️ Pandoc não encontrado no PATH do sistema.")
        return False

    # Create temporary standalone HTML file
    temp_dir = Path(tempfile.mkdtemp(prefix="build_docs_"))
    temp_html = temp_dir / "relatorio_temp.html"

    try:
        # Step 1: Markdown to HTML with embedded resources (images, SVGs, CSS)
        pandoc_cmd = [
            pandoc_bin,
            str(input_md.resolve()),
            "-o", str(temp_html.resolve()),
            "--standalone",
            "--embed-resources",
            f"--resource-path={input_md.parent.resolve()}",
            "--metadata", "title=Relatório de Desenvolvimento Técnico & Guia Pedagógico",
        ]
        if css_path.exists():
            pandoc_cmd.append(f"--css={css_path.resolve()}")

        print(f"   Executando Pandoc: {' '.join(pandoc_cmd[:4])} ...")
        proc_pandoc = subprocess.run(
            pandoc_cmd,
            cwd=str(input_md.parent),
            capture_output=True,
            text=True,
            check=False
        )
        if proc_pandoc.returncode != 0:
            print(f"⚠️ Falha no Pandoc: {proc_pandoc.stderr}")
            return False

        if not temp_html.exists() or temp_html.stat().st_size == 0:
            print("⚠️ Arquivo HTML intermediário não foi gerado.")
            return False

        # Step 2: Headless Chrome to PDF
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        chrome_cmd = [
            chrome_bin,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            f"--print-to-pdf={output_pdf.resolve()}",
            str(temp_html.resolve())
        ]
        print(f"   Executando Headless Chrome para renderização A4...")
        proc_chrome = subprocess.run(
            chrome_cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False
        )

        if validate_pdf_binary(output_pdf):
            print(f"✅ Sucesso! PDF gerado via Chrome: {output_pdf} ({output_pdf.stat().st_size:,} bytes)")
            return True
        else:
            print(f"⚠️ PDF gerado pelo Chrome não passou na validação: {proc_chrome.stderr}")
            return False

    finally:
        if not keep_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            print(f"   Arquivo temporário mantido para inspeção em: {temp_html}")


def compile_via_latex(input_md: Path, output_pdf: Path) -> bool:
    """Fallback 1: Pandoc with XeLaTeX or pdfLaTeX engine."""
    pandoc_bin = shutil.which("pandoc")
    if not pandoc_bin:
        return False

    engine = shutil.which("xelatex") or shutil.which("pdflatex")
    if not engine:
        return False

    print(f"🔹 Tentando compilação alternativa via Pandoc + {Path(engine).name}...")
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        pandoc_bin,
        str(input_md.resolve()),
        "-o", str(output_pdf.resolve()),
        f"--pdf-engine={engine}",
        "-V", "geometry:margin=2cm",
    ]
    proc = subprocess.run(cmd, cwd=str(input_md.parent), capture_output=True, text=True, check=False)
    if proc.returncode == 0 and validate_pdf_binary(output_pdf):
        print(f"✅ Sucesso! PDF gerado via LaTeX: {output_pdf} ({output_pdf.stat().st_size:,} bytes)")
        return True
    return False


def compile_via_reportlab(input_md: Path, output_pdf: Path) -> bool:
    """Fallback 2: Native Python ReportLab PDF generation."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    except ImportError:
        return False

    print("🔹 Tentando compilação de emergência via ReportLab...")
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(str(output_pdf), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    text = input_md.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            story.append(Spacer(1, 6))
            continue
        if line.startswith("# "):
            story.append(Paragraph(f"<b><font size=18>{line[2:]}</font></b>", styles["Heading1"]))
        elif line.startswith("## "):
            story.append(Paragraph(f"<b><font size=14>{line[3:]}</font></b>", styles["Heading2"]))
        elif line.startswith("### "):
            story.append(Paragraph(f"<b><font size=12>{line[4:]}</font></b>", styles["Heading3"]))
        else:
            safe_line = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            story.append(Paragraph(safe_line, styles["Normal"]))

    doc.build(story)
    return validate_pdf_binary(output_pdf)


def main() -> int:
    args = parse_args()

    print("=" * 72)
    print(" 📄 COMPILADOR DE DOCUMENTAÇÃO TÉCNICA E PEDAGÓGICA (MARKDOWN -> PDF)")
    print("=" * 72)
    print(f" Arquivo de Entrada : {args.input}")
    print(f" Arquivo de Saída   : {args.output}")
    print(f" Folha de Estilos   : {args.css}")
    print("-" * 72)

    if not args.input.exists():
        print(f"❌ Erro fatal: Arquivo de entrada {args.input} não existe.")
        return 1

    chrome_bin = find_chrome_binary()
    success = False

    if chrome_bin:
        success = compile_via_chrome(
            args.input,
            args.output,
            args.css,
            chrome_bin,
            keep_temp=args.keep_temp
        )

    if not success:
        print("⚠️ Compilação com Chrome indisponível ou falhou. Testando mecanismos de fallback...")
        success = compile_via_latex(args.input, args.output)

    if not success:
        success = compile_via_reportlab(args.input, args.output)

    print("-" * 72)
    if success:
        print(f"🎉 Compilação concluída com sucesso!")
        print(f"📍 PDF Final: {args.output.resolve()}")
        print(f"📊 Tamanho: {args.output.stat().st_size:,} bytes")
        return 0
    else:
        print("❌ Falha crítica: Nenhum mecanismo de compilação PDF foi capaz de gerar o arquivo.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
