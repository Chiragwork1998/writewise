"""Stage 0 — list every hostname under the college's domain from public certificate-transparency logs (crt.sh).

Free, no API key. crt.sh is slow and sometimes times out: re-run, or wait a few minutes.
usage: python pipeline/fetch_ct_hosts.py colleges/<slug> <root_domain>      e.g. colleges/usc usc.edu
writes: discovery/subdomains.txt (all names), discovery/subdomains_l1.txt (first-level names only)
"""
import json
import sys
import time
from pathlib import Path

import httpx

college, root = Path(sys.argv[1]), sys.argv[2].lower().strip(".")
disc = college / "discovery"
disc.mkdir(parents=True, exist_ok=True)
data = None
for attempt in range(4):
    try:
        r = httpx.get("https://crt.sh/", params={"q": f"%.{root}", "output": "json"}, timeout=180)
        if r.status_code == 200:
            data = r.json()
            break
        print(f"crt.sh answered HTTP {r.status_code}; retrying", flush=True)
    except Exception as e:
        print(f"crt.sh failed ({type(e).__name__}); retrying", flush=True)
    time.sleep(20 * (attempt + 1))
if data is None:
    sys.exit("crt.sh did not answer after 4 tries. Try again later.")
names = set()
for row in data:
    for n in row.get("name_value", "").split("\n"):
        n = n.strip().lower().lstrip("*.")
        if n == root or n.endswith("." + root):
            names.add(n)
first = sorted(n for n in names if n.count(".") == root.count(".") + 1)
(disc / "subdomains.txt").write_text("\n".join(sorted(names)) + "\n")
(disc / "subdomains_l1.txt").write_text("\n".join(first) + "\n")
print(f"hostnames: {len(names):,} (first-level: {len(first):,}) -> {disc / 'subdomains.txt'}")
