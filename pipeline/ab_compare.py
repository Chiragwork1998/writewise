"""Compare self-hosted vs cloud Firecrawl output on the A/B sample: success, blocking, content kept, verified facts
still present, speed. usage: python pipeline/ab_compare.py colleges/usc colleges/usc_ab"""
import collections
import difflib
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from firecrawl_client import url_id  # noqa: E402

cloud_dir, ab_dir = Path(sys.argv[1]), Path(sys.argv[2])
src = (Path(__file__).parent / "build_docs.py").read_text()
ns = {"re": re, "unicodedata": __import__("unicodedata")}
exec(src[src.index("def ascii_fold"): src.index("def key(")], ns)
norm, verify = ns["norm"], ns["verify"]

facts = collections.defaultdict(list)
for l in (cloud_dir / "extract" / "facts_raw.jsonl").read_text().splitlines():
    d = json.loads(l)
    if d["verification"] == "exact":
        facts[d["url"]].append(d["evidence"])

BLOCK = re.compile(r"just a moment|attention required|access denied|verify you are (a )?human|captcha|cf-browser-verification|"
                   r"request blocked|forbidden|are you a robot|enable javascript and cookies", re.I)
rows = [json.loads(l) for l in (ab_dir / "ab_results.jsonl").read_text().splitlines()]
res = []
for r in rows:
    c = json.loads((cloud_dir / "pages" / f"{url_id(r['url'])}.json").read_text())
    cmd = c.get("markdown") or ""
    s = {}
    sp = ab_dir / "pages" / f"{url_id(r['url'])}.json"
    smd = json.loads(sp.read_text()).get("markdown", "") if sp.exists() else ""
    cw, sw = norm(cmd).split(), norm(smd).split()
    sim = difflib.SequenceMatcher(None, cw[:6000], sw[:6000], autojunk=False).ratio() if cw and sw else 0.0
    fl = facts.get(r["url"], [])
    pn = norm(smd)
    kept = sum(1 for e in fl if verify(e, pn) in ("exact", "near")) if fl else None
    blocked = bool(smd) and len(sw) < 250 and bool(BLOCK.search(smd[:3000]))
    res.append({**r, "cloud_chars": len(cmd), "cloud_elapsed": c.get("elapsed"), "similarity": round(sim, 3),
                "len_ratio": round(len(sw) / max(len(cw), 1), 2), "facts_total": len(fl), "facts_kept": kept,
                "blocked": blocked or r.get("http_status") in (401, 403, 429)})

(ab_dir / "ab_compare.jsonl").write_text("\n".join(json.dumps(x) for x in res) + "\n")


def summarize(group):
    n = len(group)
    ok = [x for x in group if x["status"] == "ok" and x["chars"] > 200]
    good = [x for x in ok if x["similarity"] >= 0.8]
    ft = sum(x["facts_total"] for x in group)
    fk = sum(x["facts_kept"] or 0 for x in group)
    el = [x["elapsed"] for x in group if x["status"] == "ok" and x["elapsed"]]
    cel = [x["cloud_elapsed"] for x in group if x.get("cloud_elapsed")]
    return {"pages": n, "usable": f"{len(ok)}/{n}", "same content (>=80% match)": f"{len(good)}/{n}",
            "blocked": sum(1 for x in group if x["blocked"]),
            "verified facts still present": f"{fk}/{ft} ({fk / ft * 100:.0f}%)" if ft else "n/a",
            "median sec self": round(statistics.median(el), 1) if el else None,
            "median sec cloud": round(statistics.median(cel), 1) if cel else None}


strata = collections.defaultdict(list)
for x in res:
    strata[x["stratum"]].append(x)
md = ["# Self-hosted vs Cloud Firecrawl — A/B test on USC pages", "",
      "Same URLs, same scrape options. Cloud results come from the production run; self-hosted Firecrawl "
      "v2.11.162 ran in Docker on a MacBook (Apple Silicon, Docker VM with 10 CPUs / 7.7 GB), from a home connection.", "",
      "| Group | Pages | Usable | Same content | Blocked | Verified facts still present | Median s (self) | Median s (cloud) |",
      "|---|---:|---:|---:|---:|---:|---:|---:|"]
allrow = summarize(res)
for name, g in sorted(strata.items(), key=lambda kv: -len(kv[1])):
    s = summarize(g)
    md.append(f"| {name} | {s['pages']} | {s['usable']} | {s['same content (>=80% match)']} | {s['blocked']} | "
              f"{s['verified facts still present']} | {s['median sec self']} | {s['median sec cloud']} |")
md.append(f"| **All** | {allrow['pages']} | {allrow['usable']} | {allrow['same content (>=80% match)']} | {allrow['blocked']} | "
          f"{allrow['verified facts still present']} | {allrow['median sec self']} | {allrow['median sec cloud']} |")
sched = [x for x in res if x["stratum"].startswith("schedule")]
if sched:
    errs = collections.Counter((x["actions_status"], (x["actions_error"] or "")[:120]) for x in sched)
    md += ["", "## Click-to-expand schedule pages", "", f"Actions requests: {dict(errs)}"]
worst = sorted([x for x in res if x["status"] != "ok" or x["similarity"] < 0.6], key=lambda x: x["similarity"])[:25]
md += ["", "## Pages that failed or differ most", "", "| Page | Group | Status | Similarity | Self chars / cloud chars | Note |", "|---|---|---|---:|---|---|"]
for x in worst:
    md.append(f"| {x['url'][:80]} | {x['stratum']} | {x['status']} {x.get('http_status') or ''} | {x['similarity']} | "
              f"{x['chars']:,} / {x['cloud_chars']:,} | {(x.get('error') or '')[:80].replace('|', '/')} |")
(ab_dir / "ab_report.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
