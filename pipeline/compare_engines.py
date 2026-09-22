"""Three-way fetching comparison on the same URLs: Firecrawl cloud, Firecrawl self-hosted, Crawl4AI (plus Crawl4AI
stealth on the pages self-hosted Firecrawl could not fetch).

Text-level measures need no AI: did the page come back, how close is its text to cloud's, and how many of the verbatim
quotes behind cloud's verified facts are still present. Optional AI-level measures read extract results produced by
claude_extract_report.py (blind extraction test, no DeepSeek) when present.

usage: python pipeline/compare_engines.py colleges/usc_control/sample.txt colleges/usc_crawl4ai/blocked_urls.txt
"""
import argparse
import collections
import difflib
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from firecrawl_client import url_id  # noqa: E402

src = (Path(__file__).parent / "build_docs.py").read_text()
ns = {"re": re, "unicodedata": __import__("unicodedata")}
exec(src[src.index("def ascii_fold"): src.index("def key(")], ns)
norm, verify = ns["norm"], ns["verify"]

ap = argparse.ArgumentParser()
ap.add_argument("sample")
ap.add_argument("blocked")
args = ap.parse_args()

ROOT = Path("colleges")
ENGINES = [("Firecrawl cloud", ROOT / "usc"), ("Firecrawl self-hosted", ROOT / "usc_selfhost"),
           ("Crawl4AI", ROOT / "usc_crawl4ai")]
STEALTH = ("Crawl4AI stealth", ROOT / "usc_crawl4ai_stealth")
sample_all = [l.strip() for l in open(args.sample) if l.strip()]


def attempted(d):
    p = d / "logs" / "fetch_log.jsonl"
    return {json.loads(l)["url"] for l in p.open()} if p.exists() else set()


# only pages every engine actually tried (runs were stopped before some slow crawl-delay pages)
tried = set.intersection(*(attempted(d) for _, d in [("", ROOT / "usc_selfhost"), ("", ROOT / "usc_crawl4ai")]))
sample = [u for u in sample_all if u in tried]
not_tried = len(sample_all) - len(sample)
blocked = [l.strip() for l in open(args.blocked) if l.strip()]
IMG = re.compile(r"!\[[^\]]*\]\([^)]*\)")
LINK = re.compile(r"\[([^\]]*)\]\((?:[^()]|\([^)]*\))*\)")
CHALLENGE = re.compile(r"just a moment|attention required|verify you are human|checking your browser|enable javascript "
                       r"and cookies|cf-ray|access denied|request unsuccessful|incapsula", re.I)


def page(d, u):
    p = d / "pages" / f"{url_id(u)}.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    return r if r.get("status") == "ok" else None


def text(r):
    return LINK.sub(r"\1", IMG.sub("", (r or {}).get("markdown") or ""))


def real(r):
    t = text(r)
    return len(t) > 200 and not (len(t) < 5000 and CHALLENGE.search(t))


cloud_ev = collections.defaultdict(list)
for l in (ROOT / "usc" / "extract" / "facts_raw.jsonl").open():
    d = json.loads(l)
    if d["verification"] in ("exact", "near"):
        cloud_ev[d["url"]].append(d["evidence"])


def measure(d, urls):
    got = sim_n = 0
    sims, secs = [], []
    kept = total = 0
    for u in urls:
        r = page(d, u)
        c = page(ROOT / "usc", u)
        if r and real(r):
            got += 1
            if r.get("elapsed"):
                secs.append(r["elapsed"])
        if c and d != ROOT / "usc":
            cw, sw = norm(text(c)).split()[:5000], norm(text(r)).split()[:5000] if r else []
            s = difflib.SequenceMatcher(None, cw, sw, autojunk=False).ratio() if cw and sw else 0.0
            sims.append(s)
            sim_n += s >= 0.8
        ev = cloud_ev.get(u, [])
        if ev:
            pn = norm(text(r)) if r else ""
            total += len(ev)
            kept += sum(1 for e in ev if pn and verify(e, pn) in ("exact", "exact_words", "near"))
    chars = [len(text(page(d, u))) for u in urls if page(d, u)]
    return {"urls": len(urls), "got": got, "sim80": sim_n, "sim_median": statistics.median(sims) if sims else None,
            "kept": kept, "total": total, "sec_median": statistics.median(secs) if secs else None,
            "chars_median": statistics.median(chars) if chars else None}


md = ["# Fetching engines compared — Firecrawl cloud vs Firecrawl self-hosted vs Crawl4AI", "",
      f"Same {len(sample)} randomly chosen USC pages for every engine (drawn from the cloud run; {not_tried} more were not "
      f"tried because the runs were stopped before slow crawl-delay pages, mostly Daily Trojan), plus the {len(blocked)} "
      "pages self-hosted Firecrawl could not fetch. Text is compared after removing link and image markup so each "
      "engine's markdown style does not count as a difference. Robots rules and crawl-delays were honoured in every run.", "",
      "## 1. Random sample", "",
      "| Measure | " + " | ".join(n for n, _ in ENGINES) + " |", "|---|" + "---:|" * len(ENGINES)]
res = {n: measure(d, sample) for n, d in ENGINES}
pct = lambda a, b: f"{a:,} ({a / max(b, 1) * 100:.1f}%)"
md += [
    "| Pages returned with real content | " + " | ".join(pct(res[n]["got"], res[n]["urls"]) for n, _ in ENGINES) + " |",
    "| Page text ≥80% same as cloud | — | " + " | ".join(pct(res[n]["sim80"], res[n]["urls"]) for n, _ in ENGINES[1:]) + " |",
    "| Median text similarity to cloud | — | " + " | ".join(f"{res[n]['sim_median']:.2f}" for n, _ in ENGINES[1:]) + " |",
    "| Cloud's fact quotes still present | " + " | ".join(pct(res[n]["kept"], res[n]["total"]) for n, _ in ENGINES) + " |",
    "| Median characters per page | " + " | ".join(f"{res[n]['chars_median']:,.0f}" for n, _ in ENGINES) + " |",
    "| Median seconds per page | " + " | ".join(f"{res[n]['sec_median']}" for n, _ in ENGINES) + " |", ""]

md += ["**By page type (quotes still present)**", "", "| Page type | Pages | " + " | ".join(n for n, _ in ENGINES[1:]) + " |",
       "|---|---:|" + "---:|" * (len(ENGINES) - 1)]
by = collections.defaultdict(list)
for u in sample:
    c = page(ROOT / "usc", u)
    by[((c or {}).get("meta") or {}).get("bucket") or "other"].append(u)
for b, us in sorted(by.items(), key=lambda kv: -len(kv[1])):
    cells = []
    for n, d in ENGINES[1:]:
        m = measure(d, us)
        cells.append(f"{m['kept'] / max(m['total'], 1) * 100:.0f}%" if m["total"] else "—")
    md.append(f"| {b} | {len(us)} | " + " | ".join(cells) + " |")
md.append("")

md += ["## 2. Pages blocked for self-hosted Firecrawl", "",
       "| Engine | Real content | Challenge / error pages |", "|---|---:|---:|"]
blk_rows = []
for n, d in ENGINES + [STEALTH]:
    ok = sum(1 for u in blocked if real(page(d, u)))
    md.append(f"| {n} | {ok} of {len(blocked)} | {len(blocked) - ok} |")
per_host = collections.defaultdict(lambda: collections.Counter())
for u in blocked:
    h = u.split("/")[2]
    for n, d in ENGINES + [STEALTH]:
        per_host[h][n] += 1 if real(page(d, u)) else 0
    per_host[h]["_total"] += 1
md += ["", "| Site | Pages | " + " | ".join(n for n, _ in ENGINES + [STEALTH]) + " |", "|---|---:|" + "---:|" * 4]
for h, c in sorted(per_host.items(), key=lambda kv: -kv[1]["_total"]):
    md.append(f"| {h} | {c['_total']} | " + " | ".join(str(c[n]) for n, _ in ENGINES + [STEALTH]) + " |")
md.append("")

claude = ROOT / "usc_engines" / "claude" / "section.md"  # blind AI extraction test (claude_extract_report.py)
if claude.exists():
    md += claude.read_text().splitlines()

out = ROOT / "usc_crawl4ai" / "engine_comparison.md"
out.write_text("\n".join(md) + "\n")
print("\n".join(md))
