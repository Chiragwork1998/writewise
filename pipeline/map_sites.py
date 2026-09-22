"""Firecrawl /map for a list of sites (1 credit each). Saves discovery/maps/<site>.json."""
import json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from firecrawl_client import default_fetcher

college = sys.argv[1]
sites = [s.strip() for s in Path(sys.argv[2]).read_text().split() if s.strip() and not s.startswith("#")]
f = default_fetcher(college)
out = Path(college) / "discovery" / "maps"

def run(site):
    p = out / f"{site}.json"
    if p.exists():
        return site, len(json.loads(p.read_text()).get("links", [])), "cached"
    try:
        j = f.map(f"https://{site}/")
    except Exception as e:
        return site, 0, f"error {e}"
    p.write_text(json.dumps(j))
    return site, len(j.get("links", [])), "ok" if j.get("success") else str(j)[:200]

with ThreadPoolExecutor(2) as ex:
    for site, n, st in ex.map(run, sites):
        print(f"{site:40} {n:6} {st}", flush=True)
