"""Separate AI run-to-run randomness from the effect of self-hosted fetching.

On a random sample of pages, the CLOUD page text is re-extracted twice (A1, A2) with the current code and prompt, and
compared with the self-hosted extraction of the same URLs (S). A1 vs A2 = pure randomness; A vs S = randomness plus
fetching differences. The original cloud extraction (O) is included to show drift since the cloud run.

usage:
  python pipeline/control_randomness.py select colleges/usc colleges/usc_selfhost colleges/usc_control [N]
  python pipeline/extract_facts.py colleges/usc "..." --sample colleges/usc_control/sample.txt --out colleges/usc_control/cloud_a1
  python pipeline/extract_facts.py colleges/usc "..." --sample colleges/usc_control/sample.txt --out colleges/usc_control/cloud_a2
  python pipeline/control_randomness.py report colleges/usc colleges/usc_selfhost colleges/usc_control
"""
import collections
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
src = (Path(__file__).parent / "build_docs.py").read_text()
ns = {"re": re, "unicodedata": __import__("unicodedata")}
exec(src[src.index("def ascii_fold"): src.index("def key(")], ns)
norm = ns["norm"]

mode, cloud, selfd, ctl = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
SKIP_URL = re.compile(r"classes\.usc\.edu/term/|engage\.usc\.edu/club_signup")
OK = ("exact", "exact_words", "near")

if mode == "select":
    n = int(sys.argv[5]) if len(sys.argv) > 5 else 300
    extracted = {json.loads(l)["key"].rsplit("#", 1)[0] for l in (cloud / "extract" / "extracted_pages.jsonl").open()}
    attempted = {json.loads(l)["url"] for l in (selfd / "logs" / "fetch_log.jsonl").open()}
    urls = []
    for f in (cloud / "pages").glob("*.json"):
        r = json.loads(f.read_text())
        u = r.get("url")
        if r.get("status") == "ok" and u in extracted and u in attempted and not SKIP_URL.search(u):
            urls.append(u)
    random.Random(20260916).shuffle(urls)
    ctl.mkdir(parents=True, exist_ok=True)
    (ctl / "sample.txt").write_text("\n".join(sorted(urls[:n])) + "\n")
    print(f"eligible {len(urls)}; sample {min(n, len(urls))} -> {ctl / 'sample.txt'}")
    sys.exit()

sample = [l.strip() for l in (ctl / "sample.txt").open() if l.strip()]
sset = set(sample)


def load(path):
    by = collections.defaultdict(list)
    for l in Path(path).open():
        d = json.loads(l)
        if d["url"] in sset and d["verification"] in OK:
            by[d["url"]].append(d)
    return by


sets = {"O": load(cloud / "extract" / "facts_raw.jsonl"),
        "A1": load(ctl / "cloud_a1" / "facts_raw.jsonl"),
        "A2": load(ctl / "cloud_a2" / "facts_raw.jsonl"),
        "S": load(selfd / "extract" / "facts_raw.jsonl")}


def grams(s, k=5):
    w = norm(s).split()
    return {" ".join(w[i:i + k]) for i in range(max(1, len(w) - k + 1))}


def covered(xs, ys):
    """How many facts in xs have a matching fact (overlapping evidence) in ys, per page."""
    hit = tot = 0
    for u in sample:
        yg = [grams(y["evidence"]) for y in ys.get(u, [])]
        for x in xs.get(u, []):
            tot += 1
            g = grams(x["evidence"])
            if any(len(g & h) / max(1, min(len(g), len(h))) >= 0.5 for h in yg):
                hit += 1
    return hit, tot


def cats(by):
    return collections.Counter(d.get("category") for v in by.values() for d in v)


res = {"pages": len(sample), "counts": {k: sum(len(v) for v in by.values()) for k, by in sets.items()},
       "categories": {k: cats(by) for k, by in sets.items()}, "overlap": {}}
for x, y in (("A1", "A2"), ("A2", "A1"), ("A1", "S"), ("S", "A1"), ("A2", "S"), ("S", "A2"), ("O", "A1"), ("A1", "O")):
    h, t = covered(sets[x], sets[y])
    res["overlap"][f"{x}->{y}"] = [h, t]
# pages where one side has zero facts
res["zero_pages"] = {k: sum(1 for u in sample if not by.get(u)) for k, by in sets.items()}
(ctl / "control_result.json").write_text(json.dumps(res, indent=1))
print(json.dumps({k: res[k] for k in ("pages", "counts", "overlap", "zero_pages")}, indent=1))
