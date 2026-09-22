"""Stage 1b — turn discover_hosts.py probe results into discovery/sites.json, the input inventory.py needs.

Groups hostnames that land on the same final site (aliases), keeps the one with the most content.
usage: python pipeline/build_sites.py colleges/<slug> <root_domain>
"""
import collections
import json
import sys
from pathlib import Path

college, root = Path(sys.argv[1]), sys.argv[2].lower()
rows = [json.loads(l) for l in (college / "discovery" / "hosts.jsonl").read_text().splitlines()]
ok = [r for r in rows if r.get("status") and r["status"] < 400 and r.get("final_host")]
by_final = collections.defaultdict(list)
for r in ok:
    by_final[r["final_host"]].append(r)
sites = []
for fh, rs in by_final.items():
    r = max(rs, key=lambda x: x.get("bytes", 0))
    sites.append({"site": fh, "aliases": sorted({x["host"] for x in rs}), "title": r.get("title"),
                  "desc": r.get("description"), "gen": r.get("generator"), "bytes": r.get("bytes"),
                  "final_url": r.get("final_url")})
sites.sort(key=lambda s: s["site"])
(college / "discovery" / "sites.json").write_text(json.dumps(sites, indent=1))
off = [s["site"] for s in sites if not s["site"].endswith(root)]
blocked = [r["host"] for r in rows if r.get("status") in (401, 403, 406, 429)]
print(f"sites: {len(sites):,} | redirect off {root}: {len(off)} | blocked probes: {len(blocked)}")
if off:
    print("off-domain (check: affiliated or unrelated?):", off[:40])
