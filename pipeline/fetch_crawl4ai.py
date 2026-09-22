"""Fetch a URL list with Crawl4AI (open source, runs a local headless Chromium) and save pages in the same record format
as firecrawl_client.py, so the same extraction and comparison scripts can read them.

Same politeness as the Firecrawl runs: robots.txt disallow rules and crawl-delay per host (from the college's
discovery/sitemaps), at most 3 parallel pages per host (1 when the host asks for a delay of 2 s or more).
Full-page markdown (no main-content filter), matching the Firecrawl runs' onlyMainContent=false.

usage: .venv-crawl4ai/bin/python pipeline/fetch_crawl4ai.py colleges/<slug> queues/round2.jsonl [more queues or .txt ...]
         [--robots colleges/<slug>] [--stealth] [--concurrency 8] [--after-finished colleges/usc_selfhost/logs/scrape.log]
"""
import argparse
import asyncio
import collections
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent))
from firecrawl_client import Robots, normalize, url_id  # noqa: E402

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("college")
ap.add_argument("urls", nargs="+")
ap.add_argument("--robots", default="", help="college dir whose discovery/sitemaps hold robots rules (default: the college)")
ap.add_argument("--stealth", action="store_true")
ap.add_argument("--concurrency", type=int, default=8)
ap.add_argument("--after-finished", default="", help="wait for FINISHED in this log before touching hosts it is still "
                                                    "crawling with a crawl-delay (avoids doubling the load on them)")
args = ap.parse_args()

college = Path(args.college)
pages = college / "pages"
logs = college / "logs"
pages.mkdir(parents=True, exist_ok=True)
logs.mkdir(parents=True, exist_ok=True)
log_path = logs / "fetch_log.jsonl"
robots = Robots(Path(args.robots or args.college) / "discovery" / "sitemaps")

urls, seen, queue_meta = [], set(), {}
for f in args.urls:  # queue files (.jsonl with url/bucket/priority, highest priority first) or plain URL lists
    rows = []
    for line in Path(f).read_text().splitlines():
        if not line.strip():
            continue
        if f.endswith(".jsonl"):
            d = json.loads(line)
            rows.append((normalize(d["url"]), {k: v for k, v in d.items() if k in ("bucket", "priority")}))
        else:
            rows.append((normalize(line), {}))
    rows.sort(key=lambda x: -(x[1].get("priority") or 0))
    for u, m in rows:
        if u and u not in seen:
            seen.add(u)
            urls.append(u)
            queue_meta[u] = m

browser_cfg = BrowserConfig(headless=True, enable_stealth=args.stealth, verbose=False)
run_cfg = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, page_timeout=90000, verbose=False,
                           remove_overlay_elements=False, check_robots_txt=False)  # robots handled below, same as Firecrawl runs

host_next = collections.defaultdict(float)
host_sem = {}
global_sem = asyncio.Semaphore(args.concurrency)
stats = collections.Counter()


def log(rec, code):
    line = {"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "url": rec["url"], "status": rec["status"],
            "api_http": code, "credits": 0, "chars": len(rec.get("markdown", "") or ""),
            "bucket": (rec.get("meta") or {}).get("bucket"), "elapsed": rec.get("elapsed"),
            "error": rec.get("error")}
    with log_path.open("a") as fh:
        fh.write(json.dumps(line) + "\n")


async def gate(host):
    if not args.after_finished:
        return
    if robots.delay(host) < 2:
        return
    while "FINISHED" not in Path(args.after_finished).read_text():
        await asyncio.sleep(30)


async def one(crawler, url, meta):
    out = pages / f"{url_id(url)}.json"
    if out.exists():
        stats["cached"] += 1
        return
    if not robots.allowed(url):
        rec = {"url": url, "status": "robots_disallowed", "meta": meta}
        log(rec, None)
        stats["robots_disallowed"] += 1
        return
    host = urlparse(url).hostname
    delay = robots.delay(host)
    await gate(host)
    sem = host_sem.setdefault(host, asyncio.Semaphore(1 if delay >= 2 else 3))
    async with sem:
        wait = host_next[host] - time.time()
        if wait > 0:
            await asyncio.sleep(wait)
        host_next[host] = time.time() + max(delay, 0.5)
        async with global_sem:
            t0 = time.time()
            try:
                res = await crawler.arun(url, config=run_cfg)
                err = None if res.success else (res.error_message or "failed")
            except Exception as e:  # browser crash, timeout outside crawl4ai's own handling
                res, err = None, f"{type(e).__name__}: {e}"
            elapsed = round(time.time() - t0, 1)
    code = getattr(res, "status_code", None) if res else None
    md = ""
    if res is not None and res.success:
        m = res.markdown
        md = (getattr(m, "raw_markdown", None) or str(m or "")) if m is not None else ""
    if res is not None and res.success and md.strip() and (code is None or code < 400):
        meta_d = res.metadata or {}
        rec = {"url": url, "status": "ok", "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "http_status": code, "final_url": res.redirected_url or url,
               "title": meta_d.get("title") or meta_d.get("og:title"),
               "description": meta_d.get("description") or meta_d.get("og:description"),
               "published": meta_d.get("article:published_time"), "modified": meta_d.get("article:modified_time"),
               "language": None, "content_type": (res.response_headers or {}).get("content-type"), "credits": 0,
               "markdown": md, "raw_html": "", "links": [l.get("href") for l in (res.links or {}).get("internal", [])]
               + [l.get("href") for l in (res.links or {}).get("external", [])],
               "metadata": meta_d, "meta": meta, "engine": "crawl4ai" + ("+stealth" if args.stealth else ""),
               "elapsed": elapsed}
        out.write_text(json.dumps(rec))
        log(rec, code)
        stats["ok"] += 1
    else:
        rec = {"url": url, "status": "error", "http": code, "meta": meta, "elapsed": elapsed,
               "error": (err or f"http {code}, {len(md)} chars")[:500],
               "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        log(rec, code)
        stats["error"] += 1
    done = sum(stats.values())
    if done % 25 == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {done}/{len(urls)} | {dict(stats)}", flush=True)


async def main():
    buckets = dict(queue_meta)
    if args.robots:  # comparison runs: reuse bucket labels from the reference college's pages
        for u in urls:
            p = Path(args.robots) / "pages" / f"{url_id(u)}.json"
            if p.exists() and not buckets.get(u):
                buckets[u] = (json.loads(p.read_text()).get("meta") or {})
    start = time.time()
    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        await asyncio.gather(*(one(crawler, u, buckets.get(u, {})) for u in urls))
    print(f"FINISHED {dict(stats)} in {(time.time() - start) / 60:.1f} min", flush=True)


asyncio.run(main())
