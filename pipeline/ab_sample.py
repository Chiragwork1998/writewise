"""Pick a stratified sample of pages already scraped by Firecrawl Cloud for a self-hosted A/B comparison.
Hard cases are over-represented: anti-bot (stealth) pages, click-to-expand schedules, PDFs, outside news.
usage: python pipeline/ab_sample.py colleges/usc colleges/usc_ab 300"""
import collections
import json
import random
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent))
from firecrawl_client import Robots  # noqa: E402

src, dst, N = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
random.seed(42)
robots = Robots(src / "discovery" / "sitemaps")
facts_by_url = collections.Counter()
for l in (src / "extract" / "facts_raw.jsonl").read_text().splitlines():
    d = json.loads(l)
    if d["verification"] == "exact":
        facts_by_url[d["url"]] += 1

pages = []
for f in (src / "pages").glob("*.json"):
    r = json.loads(f.read_text())
    if r.get("status") != "ok":
        continue
    h = urlparse(r["url"]).hostname or ""
    if robots.delay(h) > 10:
        continue  # keep the test quick; pacing for these hosts is identical either way
    pages.append({"url": r["url"], "host": h, "bucket": (r.get("meta") or {}).get("bucket") or "other",
                  "proxy": (r.get("metadata") or {}).get("proxyUsed"), "pdf": r["url"].lower().endswith(".pdf"),
                  "facts": facts_by_url[r["url"]]})

chosen, per_host = {}, collections.Counter()


def pick(label, pool, n, host_cap=6):
    random.shuffle(pool)
    k = 0
    for p in pool:
        if k >= n:
            break
        if p["url"] in chosen or per_host[p["host"]] >= host_cap:
            continue
        chosen[p["url"]] = dict(p, stratum=label)
        per_host[p["host"]] += 1
        k += 1


pick("anti-bot (needed stealth proxy)", [p for p in pages if p["proxy"] == "stealth"], 45, host_cap=10)
pick("schedule page (needs clicks)", [p for p in pages if p["bucket"].startswith("classes")], 20, host_cap=20)
pick("pdf", [p for p in pages if p["pdf"]], 8)
pick("outside news", [p for p in pages if p["bucket"] == "external_news" or not p["host"].endswith("usc.edu")], 25)
pick("faculty profile", [p for p in pages if p["bucket"] == "faculty_profile"], 40, host_cap=12)
rest = [p for p in pages if p["facts"] >= 3]
for b in ["seed_chatgpt", "research", "academics", "culture", "extracurriculars", "social_impact",
          "diversity_international", "innovative_programs", "intellectual_alignment", "quirks", "news"]:
    pick(f"content: {b}", [p for p in rest if p["bucket"] == b], 16)
pick("content: other", rest, N - len(chosen))

(dst / "pages").mkdir(parents=True, exist_ok=True)
(dst / "logs").mkdir(exist_ok=True)
(dst / "discovery").mkdir(exist_ok=True)
link = dst / "discovery" / "sitemaps"
if not link.exists():
    link.symlink_to((src / "discovery" / "sitemaps").resolve())
with (dst / "sample.jsonl").open("w") as f:
    for p in chosen.values():
        f.write(json.dumps(p) + "\n")
print(len(chosen), collections.Counter(p["stratum"] for p in chosen.values()).most_common())
