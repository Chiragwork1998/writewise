"""Score the blind Claude extraction test prepared by claude_extract_prep.py.

Every fact's quote is verified against its own input text with extract_facts.py's checker (exact, or >=85% of 5-word
runs = near). Then, per page, facts from each engine's text are matched to facts from the cloud text by overlapping
quotes. Cloud text was extracted twice (cloud_a, cloud_b): their agreement is the randomness floor.

usage: python pipeline/claude_extract_report.py
writes: colleges/usc_engines/claude/result.json and appends nothing else; prints a markdown section
"""
import collections
import json
import re
import statistics
from pathlib import Path
from urllib.parse import urlparse

OUT = Path("colleges/usc_engines/claude")
src = (Path(__file__).parent / "extract_facts.py").read_text()
ns = {"re": re, "collections": collections, "urlparse": urlparse, "unicodedata": __import__("unicodedata")}
exec(src[src.index("def norm(s):"): src.index("# ------------------------------------------------------------------ prompt")], ns)
norm, verify = ns["norm"], ns["verify"]
CONTACT = re.compile(r"\(\d{3}\)\s*\d{3}-\d{4}|\b\d{3}[-.]\d{3}[-.]\d{4}\b|[\w.+-]+@[\w-]+\.[\w.]+|mail code", re.I)
key = json.loads((OUT / "key.json").read_text())

facts = {}
missing, bad_lines = [], 0
for i, k in key.items():
    p = OUT / "out" / f"{i}.jsonl"
    if not p.exists():
        missing.append(i)
        continue
    page_norm = norm((OUT / "in" / f"{i}.txt").read_text().split("PAGE TEXT:\n", 1)[1])
    rows = []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            fa = json.loads(line)
        except json.JSONDecodeError:
            bad_lines += 1
            continue
        if not isinstance(fa, dict) or not fa.get("f") or not fa.get("e"):
            continue
        if CONTACT.search(str(fa["e"])) or CONTACT.search(str(fa["f"])):
            continue
        fa["v"] = verify(str(fa["e"]), page_norm)
        rows.append(fa)
    facts[i] = rows

by_page = collections.defaultdict(dict)
for i, k in key.items():
    if i in facts:
        by_page[k["url"]][k["version"]] = facts[i]
complete = {u: v for u, v in by_page.items() if len(v) == 4}
OK = ("exact", "near")


def grams(s, n=5):
    w = norm(s).split()
    return {" ".join(w[j:j + n]) for j in range(max(1, len(w) - n + 1))}


def match_share(xs, ys):
    yg = [grams(y["e"]) for y in ys]
    hit = sum(1 for x in xs if any(len(grams(x["e"]) & h) / max(1, min(len(grams(x["e"])), len(h))) >= 0.5 for h in yg))
    return hit, len(xs)


res = {"pages": len(complete), "missing_outputs": len(missing), "bad_json_lines": bad_lines, "versions": {}, "pairs": {},
       "by_set": {}}
for v in ("cloud_a", "cloud_b", "self", "c4a"):
    allf = [f for pv in complete.values() for f in pv[v]]
    ver = [f for f in allf if f["v"] in OK]
    res["versions"][v] = {"facts": len(allf), "verified": len(ver),
                          "verify_rate": len(ver) / max(len(allf), 1),
                          "zero_pages": sum(1 for pv in complete.values() if not [f for f in pv[v] if f["v"] in OK])}


def pair(x, y, pages):
    h1 = t1 = h2 = t2 = 0
    for pv in pages.values():
        xs = [f for f in pv[x] if f["v"] in OK]
        ys = [f for f in pv[y] if f["v"] in OK]
        a, b = match_share(xs, ys)
        c, d = match_share(ys, xs)
        h1 += a; t1 += b; h2 += c; t2 += d
    return {"x_in_y": h1 / max(t1, 1), "y_in_x": h2 / max(t2, 1), "mean": (h1 / max(t1, 1) + h2 / max(t2, 1)) / 2}


for x, y in (("cloud_a", "cloud_b"), ("cloud_a", "self"), ("cloud_b", "self"), ("cloud_a", "c4a"), ("cloud_b", "c4a"),
             ("self", "c4a")):
    res["pairs"][f"{x}|{y}"] = pair(x, y, complete)
for s in ("differs", "near-identical"):
    sub = {u: pv for u, pv in complete.items() if next(k["set"] for k in key.values() if k["url"] == u) == s}
    res["by_set"][s] = {"pages": len(sub),
                        "noise": pair("cloud_a", "cloud_b", sub)["mean"],
                        "self": (pair("cloud_a", "self", sub)["mean"] + pair("cloud_b", "self", sub)["mean"]) / 2,
                        "c4a": (pair("cloud_a", "c4a", sub)["mean"] + pair("cloud_b", "c4a", sub)["mean"]) / 2}
(OUT / "result.json").write_text(json.dumps(res, indent=1))

V = res["versions"]
P = res["pairs"]
noise = P["cloud_a|cloud_b"]["mean"] * 100
selfm = (P["cloud_a|self"]["mean"] + P["cloud_b|self"]["mean"]) / 2 * 100
c4am = (P["cloud_a|c4a"]["mean"] + P["cloud_b|c4a"]["mean"]) / 2 * 100
avg_cloud = (V["cloud_a"]["verified"] + V["cloud_b"]["verified"]) / 2
md = ["## 3. Facts the AI extracts from each engine's text (blind test, no DeepSeek)", "",
      f"{res['pages']} random sample pages. Each engine's cleaned text was given to Claude extraction agents under a random "
      "ID, so the extractor could not tell which engine produced it, with the same rules as the DeepSeek pipeline. Every "
      "version of a page went to a different agent. Cloud text was extracted twice to measure how much two runs on "
      "identical text differ. Quotes were verified with the pipeline's own checker.", "",
      "| Measure | Cloud text (run 1) | Cloud text (run 2) | Self-hosted Firecrawl | Crawl4AI |", "|---|---:|---:|---:|---:|",
      "| Verified facts | " + " | ".join(f"{V[v]['verified']:,}" for v in ("cloud_a", "cloud_b", "self", "c4a")) + " |",
      "| Quote verified | " + " | ".join(f"{V[v]['verify_rate'] * 100:.1f}%" for v in ("cloud_a", "cloud_b", "self", "c4a")) + " |",
      "| Pages with no facts | " + " | ".join(str(V[v]["zero_pages"]) for v in ("cloud_a", "cloud_b", "self", "c4a")) + " |",
      f"| Facts vs cloud average | 100% | 100% | {V['self']['verified'] / max(avg_cloud, 1) * 100:.0f}% | "
      f"{V['c4a']['verified'] / max(avg_cloud, 1) * 100:.0f}% |", "",
      "| Agreement with cloud-text facts (share of facts matched both ways) | Value |", "|---|---:|",
      f"| Two runs on identical cloud text (randomness floor) | {noise:.0f}% |",
      f"| Self-hosted Firecrawl text vs cloud text | {selfm:.0f}% |",
      f"| Crawl4AI text vs cloud text | {c4am:.0f}% |", "",
      "| Page group | Pages | Randomness floor | Self-hosted | Crawl4AI |", "|---|---:|---:|---:|---:|"]
for s, r in res["by_set"].items():
    md.append(f"| Text {'differs between engines' if s == 'differs' else 'near-identical'} | {r['pages']} | "
              f"{r['noise'] * 100:.0f}% | {r['self'] * 100:.0f}% | {r['c4a'] * 100:.0f}% |")
md += ["", f"Gap beyond randomness: self-hosted {max(0, noise - selfm):.0f} points, Crawl4AI {max(0, noise - c4am):.0f} points.", ""]
if missing or bad_lines:
    md.append(f"_Excluded: {len(missing)} missing outputs, {bad_lines} malformed lines._\n")
(OUT / "section.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
