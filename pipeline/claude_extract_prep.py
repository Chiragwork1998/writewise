"""Prepare a blind, DeepSeek-free extraction test of the three fetching engines (Claude agents do the extraction).

For pages every engine returned, the page text is cleaned exactly like extract_facts.py (site boilerplate learned from
that engine's own sample pages), then written under random IDs so the extractor cannot tell which engine produced it.
Cloud text is included twice (two IDs, different batches) to measure extractor randomness.
Picks the pages whose text differs most between engines plus a random control set.

usage: python pipeline/claude_extract_prep.py [n_diff=30] [n_random=15] [batches=15]
writes: colleges/usc_engines/claude/{RULES.md, in/<id>.txt, key.json, batches.json}
"""
import collections
import difflib
import json
import random
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent))
from firecrawl_client import url_id  # noqa: E402

n_diff = int(sys.argv[1]) if len(sys.argv) > 1 else 30
n_rand = int(sys.argv[2]) if len(sys.argv) > 2 else 15
n_batches = int(sys.argv[3]) if len(sys.argv) > 3 else 15
ROOT = Path("colleges/usc_engines")
OUT = ROOT / "claude"
ENG = {"cloud": "usc", "self": "usc_selfhost", "c4a": "usc_crawl4ai"}
MAXC = 30000
src = (Path(__file__).parent / "extract_facts.py").read_text()
ns = {"re": re, "collections": collections, "urlparse": urlparse, "unicodedata": __import__("unicodedata")}
exec(src[src.index("LINK = re.compile"): src.index("host_lines = collections.defaultdict")], ns)
exec(src[src.index("def norm(s):"): src.index("# ------------------------------------------------------------------ prompt")], ns)
simplify, norm = ns["simplify"], ns["norm"]
sample = [l.strip() for l in open("colleges/usc_control/sample.txt") if l.strip()]
SKIP = re.compile(r"classes\.usc\.edu/(term|api)/|engage\.usc\.edu/club_signup")


def cleaner(pages):
    host_lines, host_pages = collections.defaultdict(collections.Counter), collections.Counter()
    for r in pages.values():
        h = urlparse(r["url"]).hostname
        host_pages[h] += 1
        for l in {simplify(x) for x in r["markdown"].splitlines()}:
            if l:
                host_lines[h][l] += 1

    def clean(r):
        h = urlparse(r["url"]).hostname
        n = host_pages[h]
        out, prev = [], None
        for raw in r["markdown"].splitlines():
            l = simplify(raw)
            if not l or l == prev:
                continue
            if n >= 5 and host_lines[h][l] / n >= 0.4 and len(l) < 300:
                continue
            if re.fullmatch(r"[-*•|#>\s\\]*", l) or re.search(r"cookie|consent preferences|privacy notice", l, re.I) and len(l) < 200:
                continue
            out.append(l)
            prev = l
        return "\n".join(out)
    return clean


texts = {}
for e, d in ENG.items():
    pages = {}
    for u in sample:
        p = Path("colleges") / d / "pages" / f"{url_id(u)}.json"
        if p.exists():
            r = json.loads(p.read_text())
            if r.get("status") == "ok" and r.get("markdown") and not SKIP.search(u):
                pages[u] = r
    clean = cleaner(pages)
    texts[e] = {u: clean(r)[:MAXC] for u, r in pages.items()}
common = [u for u in sample if all(len(texts[e].get(u, "")) >= 200 for e in ENG)]


def sim(a, b):
    return difflib.SequenceMatcher(None, norm(a).split()[:4000], norm(b).split()[:4000], autojunk=False).ratio()


scored = sorted(((min(sim(texts["cloud"][u], texts["self"][u]), sim(texts["cloud"][u], texts["c4a"][u])), u) for u in common))
# plain random sample (about 70% of pages differ somewhat between engines, so no stratification is needed)
rng = random.Random(20260916)
chosen = rng.sample(common, min(n_diff + n_rand, len(common)))
diff = [u for s, u in scored if s < 0.97 and u in chosen]
(OUT / "in").mkdir(parents=True, exist_ok=True)
key, versions = {}, []
used = set()
for u in chosen:
    for v in ("cloud_a", "cloud_b", "self", "c4a"):
        while True:
            i = "%06x" % rng.randrange(16 ** 6)
            if i not in used:
                used.add(i)
                break
        e = v.split("_")[0]
        key[i] = {"url": u, "version": v, "engine": e, "set": "differs" if u in diff else "near-identical",
                  "sim": next(s for s, x in scored if x == u)}
        (OUT / "in" / f"{i}.txt").write_text(f"URL: {u}\n\nPAGE TEXT:\n{texts[e][u]}\n")
        versions.append(i)
# every version of the same page goes to a different batch
batches = [[] for _ in range(n_batches)]
for u in chosen:
    ids = [i for i in versions if key[i]["url"] == u]
    order = sorted(range(n_batches), key=lambda b: (len(batches[b]), rng.random()))
    for b, i in zip(order, ids):  # four least-loaded, distinct batches
        batches[b].append(i)
for b in batches:
    rng.shuffle(b)
(OUT / "key.json").write_text(json.dumps(key, indent=1))
(OUT / "batches.json").write_text(json.dumps(batches, indent=1))
system = src[src.index('SYSTEM = f"""') + len('SYSTEM = f"""'): src.index('"""', src.index('SYSTEM = f"""') + 14)]
system = system.replace("{args.school}", "University of Southern California (USC)").replace("{{", "{").replace("}}", "}")
(OUT / "RULES.md").write_text(system + "\n\nRETURN AT MOST 40 FACTS per page: choose the most specific and important.\n")
sizes = [len(texts["cloud"][u]) for u in chosen]
print(f"common pages {len(common)}; differing (<0.97) {len([1 for s, _ in scored if s < 0.97])}; chosen {len(chosen)} "
      f"({len(diff)} with text differences, {len(chosen) - len(diff)} near-identical); texts {len(versions)}; batch sizes {[len(b) for b in batches]}; "
      f"cloud chars median {sorted(sizes)[len(sizes)//2]:,} max {max(sizes):,}")
