"""Render Markdown documents to styled HTML and PDF (headless Chrome).
usage: python pipeline/render_pdf.py file1.md [file2.md ...]"""
import html
import re
import subprocess
import sys
from pathlib import Path

import markdown

sys.path.insert(0, str(Path(__file__).resolve().parent))
import college_config  # noqa: E402

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
CSS = """
@page { size: Letter; margin: 16mm 15mm 18mm 15mm; }
:root { --ink:#1d2433; --muted:#5b6475; --line:#e3e6ec; --accent:__ACCENT__; --gold:__GOLD__; }
* { box-sizing: border-box; }
body { font-family: "Charter", "Georgia", serif; color: var(--ink); font-size: 10.2pt; line-height: 1.45; margin: 0; }
h1 { font-family: "Avenir Next", "Helvetica Neue", Arial, sans-serif; font-size: 22pt; color: var(--accent);
     border-bottom: 3px solid var(--gold); padding-bottom: 6px; margin: 0 0 6px; letter-spacing: -0.2px; }
h1 + p em { color: var(--muted); font-size: 11pt; }
h2 { font-family: "Avenir Next", "Helvetica Neue", Arial, sans-serif; font-size: 14.5pt; color: var(--accent);
     margin: 22px 0 6px; padding-top: 6px; border-top: 1px solid var(--line); break-after: avoid; }
h3 { font-family: "Avenir Next", "Helvetica Neue", Arial, sans-serif; font-size: 11pt; margin: 14px 0 4px;
     color: #2b3445; break-after: avoid; }
p { margin: 4px 0 8px; }
ul { margin: 2px 0 8px; padding-left: 18px; }
li { margin: 2px 0; break-inside: avoid; }
code { font-size: 9pt; background: #f3f4f7; padding: 0 3px; border-radius: 3px; }
a { color: #1f4e9a; text-decoration: none; word-break: break-all; }
.cite { color: var(--muted); font-size: 8pt; font-family: "Avenir Next", Arial, sans-serif; white-space: nowrap; }
strong { color: #111827; }
table { border-collapse: collapse; width: 100%; font-size: 9pt; margin: 6px 0 12px; }
th, td { border: 1px solid var(--line); padding: 4px 6px; text-align: left; vertical-align: top; }
th { background: #faf6f0; }
blockquote { margin: 8px 0; padding: 6px 12px; border-left: 3px solid var(--gold); background: #fbf8f2; color: #3a3f4a; }
"""


def render(md_path):
    cfg = college_config.for_path(md_path)  # college colours from config/college.json
    css = CSS.replace("__ACCENT__", cfg["pdf_accent"]).replace("__GOLD__", cfg["pdf_gold"])
    md_path = Path(md_path)
    text = md_path.read_text()
    body = markdown.markdown(text, extensions=["extra", "sane_lists"])
    body = re.sub(r"\[(S\d+(?:, S\d+)*)\]", r'<span class="cite">[\1]</span>', body)
    title = html.escape(text.splitlines()[0].lstrip("# ").strip()) if text else md_path.stem
    doc = f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title><style>{css}</style></head><body>{body}</body></html>"
    html_path = md_path.with_suffix(".html")
    html_path.write_text(doc)
    pdf_path = md_path.with_suffix(".pdf")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf_path}",
                    "--virtual-time-budget=20000", html_path.resolve().as_uri()],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=900)
    return pdf_path


if __name__ == "__main__":
    for p in sys.argv[1:]:
        out = render(p)
        print(out, out.stat().st_size)
