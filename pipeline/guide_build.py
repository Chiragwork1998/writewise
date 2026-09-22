"""Assemble the student guide: checked chapters -> one branded HTML -> PDF with page numbers.

Citations [RES-12, DRES-3] become superscript note numbers; each chapter ends with its numbered source pages.
Refuses to build a chapter that does not pass guide_check.py.
usage: python pipeline/guide_build.py colleges/usc "University of Southern California" "WriteWise"
writes: colleges/<slug>/guide/<slug>_student_guide.html and .pdf
"""
import html
import json
import re
import subprocess
import sys
from pathlib import Path

import markdown

college, SCHOOL, BRAND = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
root = Path(__file__).resolve().parent.parent
G = college / "guide"
ORDER = [("GEN", "Orientation", "Size, admissions, cost and campus"),
         ("CUL", "Culture", "Ethos, mission statement and traditions"),
         ("EXT", "Extracurriculars", "Clubs and organizations"),
         ("QRK", "Quirks", "Fun, unusual, human clubs and traditions"),
         ("ACA", "Academics", "Courses and the professors who teach them"),
         ("RES", "Research", "Research opportunities with specific professors"),
         ("SOC", "Social Impact", "Nonprofit and community service alignment"),
         ("INN", "Innovative Programs", "Signature, distinctive or rare programs"),
         ("INT", "Intellectual Alignment", "Academic philosophy and way of thinking"),
         ("DIV", "Diversity of Community", "International support and cultural integration"),
         ("NEW", "In the News", "The school in the news")]
CITE = re.compile(r"\s?\[((?:D?[A-Z]{3}-\d+)(?:\s*,\s*D?[A-Z]{3}-\d+)*)\]")


def short_url(u, n=78):
    u = re.sub(r"^https?://(www\.)?", "", u).rstrip("/")
    return u if len(u) <= n else u[: n - 1] + "…"


chapters, toc = [], []
for idx, (code, label, sub) in enumerate(ORDER):
    md_path = G / "chapters" / f"{code}.md"
    if not md_path.exists():
        sys.exit(f"missing chapter {code}")
    chk = subprocess.run([sys.executable, str(root / "pipeline" / "guide_check.py"), str(college), code],
                         capture_output=True, text=True)
    if chk.returncode != 0:
        sys.exit(f"chapter {code} fails the checker:\n{chk.stdout[:2000]}")
    pack = json.loads((G / "packs" / f"{code}.json").read_text())
    text = md_path.read_text()
    m = re.match(r"#\s+(.+)\n", text)
    title = m.group(1).strip() if m else label
    text = text[m.end():] if m else text
    notes, note_no = [], {}

    def sub_cite(mt):
        nums = []
        for i in [x.strip() for x in mt.group(1).split(",")]:
            for u in pack[i]["sources"][:1]:
                if u not in note_no:
                    notes.append(u)
                    note_no[u] = len(notes)
                nums.append(note_no[u])
        nums = sorted(set(nums))
        links = ",".join(f'<a href="#{code}-s{k}">{k}</a>' for k in nums)
        return f'<sup class="n">{links}</sup>'

    text = CITE.sub(sub_cite, text)
    text = re.sub(r"^(>\s*\*\*At a glance\*\*)\s*\n(?=>\s*-)", r"\1\n>\n", text, flags=re.M | re.I)
    body = markdown.markdown(text, extensions=["tables", "sane_lists"])
    body = body.replace("<blockquote>", '<aside class="glance">').replace("</blockquote>", "</aside>")
    num = "" if code == "GEN" else f"{idx:02d}"
    src = "".join(f'<li id="{code}-s{k}"><a href="{html.escape(u, quote=True)}">{html.escape(short_url(u, 120))}</a></li>'
                  for k, u in enumerate(notes, 1))
    chapters.append(f"""
<section class="chapter" id="{code}">
  <header class="opener">
    <p class="cat">{'Before you start' if code == 'GEN' else f'Category {num} · {html.escape(label)}'}</p>
    <h1>{html.escape(title)}</h1>
    <p class="sub">{html.escape(sub)}</p>
  </header>
  <div class="body">{body}</div>
  <div class="sources"><h4>Sources for this chapter</h4><ol>{src}</ol></div>
</section>""")
    toc.append((num, title, sub, code))

toc_html = "".join(f'<li><span class="tn">{n or "—"}</span><a href="#{c}"><b>{html.escape(t)}</b><span>{html.escape(s)}</span></a></li>'
                   for n, t, s, c in toc)
DATE = "September 2026"
page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>{html.escape(SCHOOL)} — Student Guide</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Libre+Franklin:wght@500;700;800&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap">
<style>
:root{{--red:#990000;--red-dark:#6E0000;--ink:#1F1B1A;--muted:#6A625F;--rule:#E6DEDA;--tint:#FBF4F2;--paper:#FFFFFF}}
@page{{size:Letter;margin:15mm 16mm 16mm 16mm}}
*{{box-sizing:border-box}}
html,body{{margin:0;background:var(--paper);color:var(--ink)}}
body{{font:9.3pt/1.4 "Source Serif 4",Charter,Georgia,serif}}
h1,h2,h3,h4,.cat,.sub,.kicker,th,sup.n,.toc,.glance strong{{font-family:"Libre Franklin","Helvetica Neue",Arial,sans-serif}}
.cover{{height:279.3mm;width:215.9mm;background:var(--red);color:#fff;padding:26mm 22mm;display:flex;flex-direction:column;justify-content:space-between;break-after:page}}
.cover .kicker{{letter-spacing:.2em;text-transform:uppercase;font-size:10pt;font-weight:700;opacity:.85}}
.cover h1{{font-size:44pt;line-height:1.02;font-weight:800;margin:10mm 0 6mm;letter-spacing:-.01em}}
.cover .deck{{font-size:14pt;line-height:1.4;max-width:130mm;opacity:.95}}
.cover .bar{{border-top:2px solid rgba(255,255,255,.6);padding-top:5mm;display:flex;justify-content:space-between;font:600 11pt "Libre Franklin",Arial,sans-serif}}
.cover .brand{{font-size:20pt;font-weight:800;letter-spacing:.01em}}
.intro{{break-after:page}}
.intro h2{{font-size:20pt;color:var(--red);margin:0 0 4mm}}
.intro p{{margin:0 0 3mm}}
.toc{{list-style:none;padding:0;margin:6mm 0 0}}
.toc li{{display:grid;grid-template-columns:12mm 1fr;padding:1.6mm 0;border-bottom:1px solid var(--rule)}}
.toc .tn{{font-weight:800;color:var(--red);font-size:12pt}}
.toc a{{color:inherit;text-decoration:none}}
.toc b{{display:block;font-size:11pt}}
.toc span{{font-size:9pt;color:var(--muted)}}
.chapter{{break-before:page}}
.opener{{border-top:6px solid var(--red);padding-top:4mm;margin-bottom:4mm}}
.opener .cat{{margin:0;color:var(--red);font-weight:700;font-size:9pt;letter-spacing:.14em;text-transform:uppercase}}
.opener h1{{font-size:24pt;font-weight:800;margin:1.5mm 0 1mm;line-height:1.1}}
.opener .sub{{margin:0;color:var(--muted);font-size:10.5pt}}
.body h2{{font-size:13pt;font-weight:700;color:var(--red-dark);margin:5mm 0 1.5mm;break-after:avoid}}
.body h3{{font-size:11.5pt;font-weight:700;margin:5mm 0 1.5mm;break-after:avoid}}
.body p{{margin:0 0 2mm;orphans:3;widows:3}}
.body ul,.body ol{{margin:0 0 3mm;padding-left:5mm}}
.body li{{margin:0 0 .8mm}}
.body strong{{font-weight:600}}
.glance{{background:var(--tint);border-left:4px solid var(--red);padding:3mm 4.5mm;margin:3mm 0 4mm;break-inside:avoid}}
.glance p{{margin:0 0 1.5mm}}
.glance strong{{color:var(--red);font-size:9.5pt;letter-spacing:.1em;text-transform:uppercase;font-weight:700}}
.glance ul{{margin:0}}
table{{border-collapse:collapse;width:100%;margin:2.5mm 0 4mm;font-size:8.3pt;line-height:1.3;break-inside:auto}}
th{{background:var(--red);color:#fff;text-align:left;padding:1.5mm 2mm;font-weight:700;font-size:7.6pt;text-transform:uppercase;letter-spacing:.05em}}
td{{padding:1.2mm 2mm;border-bottom:1px solid var(--rule);vertical-align:top}}
tr{{break-inside:avoid}}
tr:nth-child(even) td{{background:#FDFAF9}}
sup.n{{font-size:6pt;color:var(--red);font-weight:700;margin-left:.4pt;line-height:0}}
.sources{{margin-top:6mm;border-top:1px solid var(--rule);padding-top:2.5mm;font-family:"Libre Franklin",Arial,sans-serif}}
.sources h4{{margin:0 0 1.5mm;font-size:8pt;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}}
.sources ol{{columns:3;column-gap:5mm;margin:0;padding-left:4.5mm;font-size:5.6pt;line-height:1.3;color:var(--muted)}}
.sources li{{break-inside:avoid;overflow-wrap:anywhere}}
.sources a{{color:var(--muted);text-decoration:none}}
sup.n a{{color:var(--red);text-decoration:none}}
</style></head><body>
<section class="cover">
  <div><p class="kicker">Student Guide · 10 categories</p>
  <h1>{html.escape(SCHOOL)}</h1>
  <p class="deck">What it is really like to study here: culture, clubs, quirks, academics, research, service, signature programs, ways of thinking, community and the news. Every fact traced to its source.</p></div>
  <div class="bar"><span class="brand">{html.escape(BRAND)}</span><span>{DATE}</span></div>
</section>
<section class="intro">
  <h2>How to use this guide</h2>
  <p>This guide looks at {html.escape(SCHOOL)} through ten lenses that matter when you are deciding where you fit. Each chapter explains what we found, what it means for a student, and what to ask next.</p>
  <p>Everything factual comes from USC's official websites, its student-organization directory and Fall 2026 Schedule of Classes, and independent news outlets, gathered in September 2026. Small red numbers point to the source pages listed, with their full web addresses, at the end of each chapter, so you can check any claim yourself. Details such as deadlines, course offerings and fees change, so confirm anything you plan around with USC directly.</p>
  <ul class="toc">{toc_html}</ul>
  <p style="margin-top:3mm;font-size:8pt;color:var(--muted)">Prepared by {html.escape(BRAND)}. Not an official publication of the University of Southern California.</p>
</section>
{''.join(chapters)}
</body></html>"""
slug = college.name
out_html = G / f"{slug}_student_guide.html"
out_html.write_text(page)
cover_start, cover_end = page.index('<section class="cover">'), page.index('<section class="intro">')
cover_html = G / "_cover.html"
cover_html.write_text(page[:cover_start] + page[cover_start:cover_end] + "</body></html>")
main_html = G / "_main.html"
main_html.write_text(page[:cover_start] + page[cover_end:])
out_pdf = G / f"{slug}_student_guide.pdf"
footer = (f'<div style="width:100%;font:7pt Arial,sans-serif;color:#8a817d;padding:0 16mm;display:flex;justify-content:space-between">'
          f'<span>{html.escape(SCHOOL)} · Student Guide · {html.escape(BRAND)}</span><span class="pageNumber"></span></div>')
script = f"""
import asyncio
from playwright.async_api import async_playwright
from pypdf import PdfReader, PdfWriter
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page()
        await pg.goto({json.dumps(cover_html.resolve().as_uri())}, wait_until="networkidle")
        await pg.pdf(path={json.dumps(str((G / "_cover.pdf").resolve()))}, format="Letter", print_background=True,
                     margin={{"top": "0", "bottom": "0", "left": "0", "right": "0"}}, page_ranges="1")
        await pg.goto({json.dumps(main_html.resolve().as_uri())}, wait_until="networkidle")
        await pg.pdf(path={json.dumps(str((G / "_main.pdf").resolve()))}, format="Letter", print_background=True,
                     display_header_footer=True, header_template="<div></div>", footer_template={json.dumps(footer)},
                     margin={{"top": "15mm", "bottom": "16mm", "left": "16mm", "right": "16mm"}})
        await b.close()
    w = PdfWriter()
    for f in ({json.dumps(str((G / "_cover.pdf").resolve()))}, {json.dumps(str((G / "_main.pdf").resolve()))}):
        for page in PdfReader(f).pages:
            w.add_page(page)
    with open({json.dumps(str(out_pdf.resolve()))}, "wb") as fh:
        w.write(fh)
asyncio.run(main())
"""
subprocess.run([str(root / ".venv-crawl4ai" / "bin" / "python"), "-c", script], check=True)
for f in ("_cover.html", "_main.html", "_cover.pdf", "_main.pdf"):
    (G / f).unlink(missing_ok=True)
d = out_pdf.read_bytes()
PAGE_RX = re.compile(rb"/Type\s*/Page[^s]")
print(f"wrote {out_pdf} | pages {len(PAGE_RX.findall(d))} | {len(d) / 1e6:.1f} MB")
