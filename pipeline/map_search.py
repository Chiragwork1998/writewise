"""Targeted discovery for thin categories: Firecrawl /map with `search` on key sites (1 credit per call; results are
relevance-ordered with titles). New URLs are appended to discovery/candidates_thin.jsonl for DeepSeek rating.
usage: python pipeline/map_search.py colleges/usc"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent))
import college_config  # noqa: E402
from firecrawl_client import default_fetcher  # noqa: E402

college = Path(sys.argv[1])
cfg = college_config.load(college)
f = default_fetcher(college)
out_dir = college / "discovery" / "map_search"
out_dir.mkdir(exist_ok=True)

def sites_for(cfg):
    """The college's own undergraduate-facing hosts. search_sites overrides; otherwise undergrad_hosts, with
    bare subdomain labels ("admission") expanded against root_domain."""
    if cfg.get("search_sites"):
        return list(cfg["search_sites"])
    root = cfg.get("root_domain")
    hosts = []
    for h in cfg.get("undergrad_hosts") or []:
        host = h if "." in h else (f"{h}.{root}" if root else None)
        if host and host not in hosts:
            hosts.append(host)
    return hosts


SITES = sites_for(cfg)
if not SITES:
    raise SystemExit(f"no hosts to search: set search_sites or undergrad_hosts + root_domain "
                     f"in {college}/config/college.json")
TERMS = {
    "QRK": "traditions fun unusual quirky",
    "CUL": "mission values history heritage",
    "SOC": "community service volunteer service-learning nonprofit",
    "DIV": "international students cultural center identity belonging",
    "INN": "interdisciplinary signature program innovative new",
    "INT": "philosophy approach vision dean's message",
    "EXT": "student organizations clubs get involved",
}
jobs = [(s, c, t) for s in SITES for c, t in TERMS.items()]


def run(job):
    site, code, term = job
    p = out_dir / f"{site}__{code}.json"
    if p.exists():
        return 0
    for attempt in range(3):
        try:
            j = f.map(f"https://{site}/", limit=60, search=term)
            if j.get("success"):
                p.write_text(json.dumps(j))
                return len(j.get("links", []))
        except Exception:
            pass
        time.sleep(20)
    return 0


with ThreadPoolExecutor(2) as ex:
    n = sum(ex.map(run, jobs))
seen = set()
rows = []
for p in out_dir.glob("*.json"):
    site, code = p.stem.split("__")
    for rank, l in enumerate(json.load(open(p)).get("links", [])):
        u = l.get("url", "").split("#")[0]
        if not u or u in seen:
            continue
        seen.add(u)
        rows.append({"url": u, "host": urlparse(u).hostname, "title": l.get("title") or "", "desc": l.get("description") or "",
                     "score": 10 - rank * 0.1, "hint": code})
with (college / "discovery" / "candidates_thin.jsonl").open("w") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")
print(f"map-search calls={len(jobs)} urls={n} unique candidates={len(rows)}")
