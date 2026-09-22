"""Step 2 — site harvest + URL inventory (no Firecrawl credits).

1. Expands the site list by following links from every known homepage and hub page
   (certificate logs miss hosts covered by wildcard certificates).
2. For every site: robots.txt (crawl-delay, disallow rules, sitemaps) and a full
   sitemap expansion, so we know how many URLs exist before spending credits.

usage: python pipeline/inventory.py colleges/usc usc.edu
"""
import gzip
import json
import re
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx

warnings.filterwarnings("ignore")
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")

college = Path(sys.argv[1])
root_domain = sys.argv[2]
disc = college / "discovery"
(disc / "sitemaps").mkdir(exist_ok=True)

from college_config import alt, load as load_config  # noqa: E402
CFG = load_config(college)
AFFILIATED = set(CFG["affiliated_domains"])  # the college's own sites on other domains
SKIP_HOST = re.compile(r"(^|\.)(login|shibboleth|adfs|sslvpn|vpn|smail|mail|webmail|myphpadmin|phpmyadmin|"
                       r"git|wiki|packages|webreg|eform|survey|feedback|help|support|lab|grad|"
                       + alt(CFG["skip_host_patterns"], escape=False) + r")\.")
HUBS = [
    f"https://www.{root_domain}/", f"https://www.{root_domain}/academics/", f"https://www.{root_domain}/research/",
    f"https://www.{root_domain}/student-life/", f"https://www.{root_domain}/about/",
    f"https://departmentsdirectory.{root_domain}/", f"https://research.{root_domain}/",
    f"https://studentaffairs.{root_domain}/", f"https://undergrad.{root_domain}/",
]


def client():
    return httpx.Client(follow_redirects=True, timeout=20, verify=False, headers={"User-Agent": UA})


def in_scope(host):
    return bool(host) and (host == root_domain or host.endswith("." + root_domain) or
                           any(host == a or host.endswith("." + a) for a in AFFILIATED))


def links_on(url):
    try:
        with client() as c:
            r = c.get(url)
        if "html" not in r.headers.get("content-type", ""):
            return str(r.url), set()
        hrefs = re.findall(r'href=["\']([^"\'#]+)', r.text)
        hosts = {urlparse(urljoin(str(r.url), h)).hostname for h in hrefs}
        return str(r.url), {h.lower() for h in hosts if h and in_scope(h.lower())}
    except Exception:
        return None, set()


def final_host(host):
    for scheme in ("https", "http"):
        try:
            with client() as c:
                r = c.get(f"{scheme}://{host}/")
            return host, urlparse(str(r.url)).hostname, r.status_code
        except Exception:
            continue
    return host, None, None


# ---- 1. harvest sites --------------------------------------------------------------
sites = {s["site"] for s in json.loads((disc / "sites.json").read_text()) if in_scope(s["site"])}
known_hosts = {json.loads(l)["host"] for l in (disc / "hosts.jsonl").read_text().splitlines()}
status = {}
frontier = [f"https://{s}/" for s in sites] + HUBS
for round_no in range(3):
    with ThreadPoolExecutor(32) as ex:
        found = set().union(*[h for _, h in ex.map(links_on, frontier)])
    new = sorted(h for h in found - known_hosts if not SKIP_HOST.search(h))
    known_hosts |= set(new)
    with ThreadPoolExecutor(32) as ex:
        probed = list(ex.map(final_host, new))
    added = set()
    for h, fh, st in probed:
        status[h] = st
        if fh and st and st < 400 and in_scope(fh) and fh not in sites:
            sites.add(fh)
            added.add(fh)
    print(f"harvest round {round_no + 1}: {len(new)} new hostnames, {len(added)} new live sites, total {len(sites)}", flush=True)
    frontier = [f"https://{s}/" for s in added]
    if not added:
        break

sites = sorted(s for s in sites if not SKIP_HOST.search(s))
(disc / "sites_all.txt").write_text("\n".join(sites))


# ---- 2. robots + sitemaps per site ---------------------------------------------------
def fetch(c, url):
    try:
        r = c.get(url)
        if r.status_code >= 400:
            return ""
        b = r.content
        if b[:2] == b"\x1f\x8b":
            b = gzip.decompress(b)
        return b.decode("utf-8", "replace")
    except Exception:
        return ""


def parse_robots(txt):
    """Rules for the '*' group (what Firecrawl and we fall under)."""
    groups, cur, in_rules = [], None, False
    for raw in txt.splitlines():
        line = raw.split("#", 1)[0].strip()
        if ":" not in line:
            continue
        k, v = [x.strip() for x in line.split(":", 1)]
        k = k.lower()
        if k == "user-agent":
            if cur is None or in_rules:
                cur = {"agents": [], "disallow": [], "allow": [], "delay": None}
                groups.append(cur)
                in_rules = False
            cur["agents"].append(v.lower())
        elif cur is not None and k in ("disallow", "allow"):
            in_rules = True
            if v:
                cur[k].append(v)
        elif cur is not None and k == "crawl-delay":
            in_rules = True
            try:
                cur["delay"] = float(v)
            except ValueError:
                pass
    star = [g for g in groups if "*" in g["agents"]]
    rules = {"disallow": [], "allow": [], "delay": None,
             "blocks_ai_bots": any(a in ("claudebot", "gptbot", "*ai*") for g in groups for a in g["agents"] if g["disallow"] == ["/"])}
    for g in star:
        rules["disallow"] += g["disallow"]
        rules["allow"] += g["allow"]
        if g["delay"] is not None:
            rules["delay"] = max(rules["delay"] or 0, g["delay"])
    rules["sitemaps"] = re.findall(r"(?im)^\s*sitemap:\s*(\S+)", txt)
    return rules


def expand(c, url, seen, depth=0):
    if url in seen or depth > 4 or len(seen) > 400:
        return []
    seen.add(url)
    x = fetch(c, url)
    if "<sitemapindex" in x:
        out = []
        for loc in re.findall(r"<loc>\s*(.*?)\s*</loc>", x, re.S):
            out += expand(c, loc.replace("&amp;", "&"), seen, depth + 1)
        return out
    entries = re.findall(r"<url>(.*?)</url>", x, re.S)
    out = []
    for e in entries:
        loc = re.search(r"<loc>\s*(.*?)\s*</loc>", e, re.S)
        lm = re.search(r"<lastmod>\s*(.*?)\s*</lastmod>", e, re.S)
        if loc:
            out.append([loc.group(1).replace("&amp;", "&"), lm.group(1) if lm else None])
    return out


def inventory(site):
    with client() as c:
        robots_txt = fetch(c, f"https://{site}/robots.txt")
        rules = parse_robots(robots_txt)
        candidates = rules["sitemaps"] + [f"https://{site}/sitemap_index.xml", f"https://{site}/sitemap.xml",
                                          f"https://{site}/wp-sitemap.xml"]
        urls, seen = [], set()
        for cand in candidates:
            if urlparse(cand).hostname and not in_scope(urlparse(cand).hostname):
                continue
            got = expand(c, cand, seen)
            urls += got
            if got and cand not in rules["sitemaps"]:
                break  # first standard sitemap that works is enough
    dedup = {}
    for u, lm in urls:
        dedup.setdefault(u.strip(), lm)
    rec = {"site": site, "robots": rules, "robots_found": bool(robots_txt),
           "sitemap_urls": [[u, lm] for u, lm in dedup.items()]}
    (disc / "sitemaps" / f"{site}.json").write_text(json.dumps(rec))
    return site, len(dedup), rules["delay"], rules["blocks_ai_bots"]


with ThreadPoolExecutor(24) as ex:
    rows = list(ex.map(inventory, sites))

rows.sort(key=lambda r: -r[1])
total = sum(r[1] for r in rows)
with (disc / "inventory_summary.tsv").open("w") as f:
    f.write("site\tsitemap_urls\tcrawl_delay\tblocks_ai_bots\n")
    for r in rows:
        f.write("\t".join(map(str, r)) + "\n")
print(f"sites={len(rows)} sites_with_sitemaps={sum(1 for r in rows if r[1])} total_sitemap_urls={total}")
for r in rows[:40]:
    print(r)
