"""Konversi dokumen teknis Markdown → HTML → PDF.

Memerlukan: `markdown` (pip, di venv) + `weasyprint` (CLI di PATH).
Jalankan: python scripts/export_pdf.py
Output: docs/dokumen-teknis-ml-climate.pdf
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "docs" / "dokumen-teknis-ml-climate.md"
PDF = ROOT / "docs" / "dokumen-teknis-ml-climate.pdf"

CSS = """
@page { size: A4; margin: 2cm 1.8cm; @bottom-right { content: counter(page); color:#888; font-size:9px; } }
body { font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 10.5px; line-height: 1.55; color:#1e293b; }
h1 { font-size: 22px; color:#0c4a6e; border-bottom:3px solid #0ea5e9; padding-bottom:6px; }
h2 { font-size: 16px; color:#0369a1; margin-top:22px; border-bottom:1px solid #e2e8f0; padding-bottom:3px; }
h3 { font-size: 13px; color:#0284c7; margin-top:16px; }
table { border-collapse: collapse; width:100%; margin:10px 0; font-size:9.5px; }
th { background:#f0f9ff; color:#0c4a6e; text-align:left; }
th, td { border:1px solid #cbd5e1; padding:4px 7px; vertical-align:top; }
code { background:#f1f5f9; padding:1px 4px; border-radius:3px; font-family:'DejaVu Sans Mono',monospace; font-size:9px; color:#be123c; }
pre { background:#0f172a; color:#e2e8f0; padding:10px 12px; border-radius:6px; overflow-x:auto; font-size:8.5px; line-height:1.4; }
pre code { background:none; color:#e2e8f0; padding:0; }
blockquote { border-left:4px solid #0ea5e9; background:#f0f9ff; margin:10px 0; padding:6px 12px; color:#334155; }
a { color:#0284c7; text-decoration:none; }
hr { border:none; border-top:1px solid #e2e8f0; margin:18px 0; }
"""


def main() -> None:
    html_body = markdown.markdown(
        MD.read_text(),
        extensions=["tables", "fenced_code", "toc", "sane_lists"],
    )
    html = f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html_body}</body></html>"

    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as f:
        f.write(html)
        tmp_html = f.name

    try:
        subprocess.run(["weasyprint", tmp_html, str(PDF)], check=True)
    except FileNotFoundError:
        sys.exit("weasyprint tidak ditemukan di PATH. Install: pip install weasyprint")
    print(f"OK: {PDF} ({PDF.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
