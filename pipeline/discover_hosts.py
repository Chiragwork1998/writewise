"""Step 1 — host discovery (no Firecrawl credits).

Probes every hostname from certificate-transparency logs plus seed hosts, follows
redirects, and records what each one actually serves so we can classify it.

usage: python pipeline/discover_hosts.py colleges/usc
"""
import json
import re
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import httpx

warnings.filterwarnings("ignore")

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")

college = Path(sys.argv[1])
disc = college / "discovery"
names = {l.strip().lower() for l in (disc / "subdomains.txt").read_text().split()}
seeds = (disc / "seed_hosts.txt")
if seeds.exists():
    names |= {l.strip().lower() for l in seeds.read_text().split() if l.strip() and not l.startswith("#")}
names = sorted(n for n in names if n and "*" not in n and " " not in n)


def probe(host):
    rec = {"host": host}
    for scheme in ("https", "http"):
        try:
            with httpx.Client(follow_redirects=True, timeout=12, verify=False,
                              headers={"User-Agent": UA}) as c:
                r = c.get(f"{scheme}://{host}/")
            html = r.text[:200000] if "html" in r.headers.get("content-type", "") else ""
            t = re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I)
            d = re.search(r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']*)', html, re.I)
            g = re.search(r'<meta[^>]+name=["\']generator["\'][^>]+content=["\']([^"\']*)', html, re.I)
            rec.update(
                status=r.status_code,
                final_url=str(r.url),
                final_host=urlparse(str(r.url)).hostname,
                title=re.sub(r"\s+", " ", t.group(1)).strip()[:200] if t else "",
                description=d.group(1)[:300] if d else "",
                generator=g.group(1)[:80] if g else ("wordpress" if "wp-content" in html else ""),
                bytes=len(r.content),
                server=r.headers.get("server", ""),
            )
            return rec
        except Exception as e:
            rec["error"] = type(e).__name__
    return rec


with ThreadPoolExecutor(48) as ex:
    results = list(ex.map(probe, names))

out = disc / "hosts.jsonl"
with out.open("w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

ok = [r for r in results if r.get("status") and r["status"] < 400]
blocked = [r for r in results if r.get("status") in (401, 403, 406, 429)]
finals = {r["final_host"] for r in ok}
print(f"probed={len(results)} live={len(ok)} blocked={len(blocked)} unique_final_sites={len(finals)}")
