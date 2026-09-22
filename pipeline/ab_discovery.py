"""Compare self-hosted discovery with the cloud run: /map on the same sites and /search on the same queries."""
import json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse
sys.path.insert(0, str(Path(__file__).parent))
from firecrawl_client import default_fetcher

cloud, self_dir = Path(sys.argv[1]), Path(sys.argv[2])
f = default_fetcher(self_dir)
out = self_dir / "discovery_selfhost"; out.mkdir(exist_ok=True)

# --- maps
sites = sorted(p.stem for p in (cloud / "discovery" / "maps").glob("*.json"))
def do_map(site):
    p = out / f"{site}.json"
    if not p.exists():
        try:
            p.write_text(json.dumps(f.map(f"https://{site}/", limit=100000)))
        except Exception as e:
            p.write_text(json.dumps({"success": False, "error": str(e)[:200]}))
    cl = json.loads((cloud / "discovery" / "maps" / f"{site}.json").read_text()).get("links", []) or []
    sh = json.loads(p.read_text()).get("links", []) or []
    cu = {l["url"].split("#")[0].rstrip("/") for l in cl if isinstance(l, dict) and l.get("url")}
    su = {(l["url"] if isinstance(l, dict) else l).split("#")[0].rstrip("/") for l in sh}
    ct = sum(1 for l in cl if isinstance(l, dict) and l.get("title"))
    st = sum(1 for l in sh if isinstance(l, dict) and l.get("title"))
    return {"site": site, "cloud_urls": len(cu), "self_urls": len(su), "overlap": len(cu & su),
            "self_only": len(su - cu), "cloud_titles": ct, "self_titles": st}
with ThreadPoolExecutor(4) as ex:
    maps = list(ex.map(do_map, sites))
json.dump(maps, (out / "_map_comparison.json").open("w"), indent=1)
tot = lambda k: sum(m[k] for m in maps)
print(f"MAP: sites={len(maps)} cloud_urls={tot('cloud_urls')} self_urls={tot('self_urls')} "
      f"overlap={tot('overlap')} self_only={tot('self_only')} cloud_titles={tot('cloud_titles')} self_titles={tot('self_titles')}")

# --- search
queries = []
for qf in ("news_queries.txt", "news_queries_2.txt"):
    p = cloud / "discovery" / qf
    if p.exists():
        queries += [q.strip() for q in p.read_text().splitlines() if q.strip() and not q.startswith("#")]
cloud_res = {}
for l in (cloud / "discovery" / "search_results.jsonl").read_text().splitlines():
    d = json.loads(l); cloud_res.setdefault(d["query"], []).append(d["url"])
rows = []
for q in queries:
    try:
        j = f.search(q, limit=10)
    except Exception as e:
        j = {"success": False, "error": str(e)[:120]}
    data = j.get("data") or {}
    items = (data.get("web") if isinstance(data, dict) else data) or []
    su = {i.get("url") for i in items if i.get("url")}
    cu = set(cloud_res.get(q, []))
    ext = lambda s: {u for u in s if u and not (urlparse(u).hostname or "").endswith("usc.edu")}
    rows.append({"query": q, "cloud": len(cu), "self": len(su), "overlap": len(cu & su),
                 "cloud_external": len(ext(cu)), "self_external": len(ext(su)), "error": j.get("error")})
    time.sleep(1)
json.dump(rows, (out / "_search_comparison.json").open("w"), indent=1)
s = lambda k: sum(r[k] for r in rows)
print(f"SEARCH: queries={len(rows)} cloud_results={s('cloud')} self_results={s('self')} overlap={s('overlap')} "
      f"cloud_external={s('cloud_external')} self_external={s('self_external')} empty_self={sum(1 for r in rows if r['self'] == 0)}")
