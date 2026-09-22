"""Turn DeepSeek ratings into a scrape queue: dedupe URL variants, drop anything already fetched or queued,
take every r=3 page, then the best r=2 pages under per-host caps and per-category quotas."""
import collections, json, sys
from pathlib import Path
from urllib.parse import urlparse

college, ratings, out, target = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
from college_config import load as load_config, root_or_default  # noqa: E402
CFG = load_config(college)
root_host = root_or_default(CFG, sys.argv[5] if len(sys.argv) > 5 else None)

def canon(u):
    p = urlparse(u)
    host = (p.hostname or "").lower()
    if host == root_host:
        host = "www." + root_host
    return f"https://{host}{p.path.rstrip('/') or '/'}" + (f"?{p.query}" if p.query else "")

taken = set()
for l in (college / "logs" / "fetch_log.jsonl").read_text().splitlines():
    d = json.loads(l)
    if d["status"] == "ok":
        taken.add(canon(d["url"]))
for q in (college / "queues").glob("*.jsonl"):
    if q.resolve() == out.resolve():
        continue
    for l in q.read_text().splitlines():
        if l.strip():
            taken.add(canon(json.loads(l)["url"]))

best = {}
for l in ratings.read_text().splitlines():
    d = json.loads(l)
    c = canon(d["url"])
    if c in taken:
        continue
    if c not in best or (d["r"], d["score"]) > (best[c]["r"], best[c]["score"]):
        best[c] = d

CAT = {"RES": "research", "ACA": "academics", "SOC": "social_impact", "DIV": "diversity_international",
       "INN": "innovative_programs", "CUL": "culture", "EXT": "extracurriculars", "NEW": "news",
       "INT": "intellectual_alignment", "QRK": "quirks"}
QUOTA = {"RES": 700, "ACA": 650, "SOC": 380, "DIV": 330, "INN": 320, "CUL": 230, "EXT": 220, "NEW": 140, "INT": 110, "QRK": 80}
HOST_CAP = collections.defaultdict(lambda: 60, CFG["queue_host_caps"])
rows = sorted(best.values(), key=lambda d: (-d["r"], -d["score"]))
host_n, cat_n, chosen = collections.Counter(), collections.Counter(), []
for d in rows:
    if d["r"] < 2 or len(chosen) >= target:
        continue
    c = d["c"] if d["c"] in QUOTA else "ACA"
    if d["r"] == 2 and (cat_n[c] >= QUOTA[c] or host_n[d["host"]] >= HOST_CAP[d["host"]]):
        continue
    host_n[d["host"]] += 1
    cat_n[c] += 1
    chosen.append({"url": d["url"], "bucket": CAT[c], "priority": d["r"] * 100 + int(d["score"]), "rating": d["r"],
                   "title": d["title"]})
with out.open("w") as f:
    for x in chosen:
        f.write(json.dumps(x) + "\n")
print("unique rated not yet taken:", len(best), "| chosen:", len(chosen))
print("by category:", {CAT[k]: v for k, v in cat_n.items()})
print("r=3:", sum(1 for x in chosen if x["rating"] == 3), "| top hosts:", host_n.most_common(15))
