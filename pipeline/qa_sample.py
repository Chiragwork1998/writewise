"""Quality check before delivery: pick random verified facts for a person to check against the live page.

Writes output/qa_sample.md with a table (fact, quote, link, PASS/FAIL column to fill in). Aim for 0 wrong facts in 50;
anything wrong means stop and investigate before delivering.
usage: python pipeline/qa_sample.py colleges/<slug> [N=50] [seed]
"""
import json
import random
import sys
from pathlib import Path

college = Path(sys.argv[1])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
seed = int(sys.argv[3]) if len(sys.argv) > 3 else random.randrange(10**6)
rows = [json.loads(l) for l in (college / "extract" / "facts_raw.jsonl").open()]
ok = [r for r in rows if r.get("verification") in ("exact", "near")]
pick = random.Random(seed).sample(ok, min(n, len(ok)))
md = [f"# QA sample — {college.name}", "", f"{len(pick)} random verified facts out of {len(ok):,} (seed {seed}). "
      "Open each link and mark PASS if the page really says this and the fact sentence is a fair reading of the quote.", "",
      "| # | Category | Fact | Quote on page | Source | PASS/FAIL |", "|---|---|---|---|---|---|"]
esc = lambda s: str(s or "").replace("|", "/").replace("\n", " ")
for i, r in enumerate(pick, 1):
    md.append(f"| {i} | {esc(r.get('category'))} | {esc(r['fact'])} | {esc(r['evidence'])} | {r['url']} |  |")
out = college / "output" / "qa_sample.md"
out.write_text("\n".join(md) + "\n")
print(f"wrote {out} ({len(pick)} facts, seed {seed})")
