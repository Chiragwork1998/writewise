"""Scrape a queue of URLs through Firecrawl.

queue file: JSONL of {"url": ..., "bucket": ..., "priority": int, "expand": bool}
A dispatcher hands a worker the highest-priority URL whose host is ready (crawl-delay respected), so slow hosts never
tie up worker threads.
usage: python pipeline/run_queue.py colleges/usc queues/x.jsonl [--max-credits N] [--threads N]
"""
import argparse
import collections
import json
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent))
from college_config import load as load_config  # noqa: E402
from firecrawl_client import default_fetcher, expand_actions, normalize  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("college")
ap.add_argument("queue", nargs="+")
ap.add_argument("--max-credits", type=int, default=10**9)
ap.add_argument("--threads", type=int, default=12)
args = ap.parse_args()

fetcher = default_fetcher(args.college)
fetcher.default_gap = 0  # pacing is handled here
items, seen = [], set()
for qf in args.queue:
    for line in Path(qf).read_text().splitlines():
        if not line.strip():
            continue
        it = json.loads(line)
        it["url"] = normalize(it["url"])
        if it["url"] in seen or fetcher.done(it["url"]):
            continue
        seen.add(it["url"])
        items.append(it)

pending = collections.defaultdict(list)
for it in sorted(items, key=lambda x: -x.get("priority", 0)):
    pending[urlparse(it["url"]).hostname].append(it)
total = len(items)
print(f"queued {total} new URLs across {len(pending)} hosts", flush=True)

CFG = load_config(args.college)
HOST_CAP = CFG["fetch_host_caps"]  # per-host parallel requests for hosts without crawl-delay
EXPAND_ACTIONS = expand_actions(CFG["expand_button_text"])
lock = threading.Lock()
next_ok = collections.defaultdict(float)
inflight = collections.Counter()
stats = collections.Counter()
spent = [0]
start = time.time()


def take():
    while True:
        with lock:
            if spent[0] >= args.max_credits:
                return None
            if not any(pending.values()):
                return None
            now = time.time()
            best = None
            for h, lst in pending.items():
                if not lst:
                    continue
                delay = fetcher.robots.delay(h)
                cap = 1 if delay >= 2 else HOST_CAP.get(h, 3)
                if next_ok[h] > now or inflight[h] >= cap:
                    continue
                if best is None or lst[0].get("priority", 0) > pending[best][0].get("priority", 0):
                    best = h
            if best is not None:
                it = pending[best].pop(0)
                delay = fetcher.robots.delay(best)
                next_ok[best] = now + max(delay, 0.5)
                inflight[best] += 1
                return it
        time.sleep(0.2)


def worker():
    while True:
        it = take()
        if it is None:
            return
        h = urlparse(it["url"]).hostname
        try:
            rec = fetcher.scrape(it["url"], meta={"bucket": it.get("bucket"), "priority": it.get("priority")},
                                 actions=EXPAND_ACTIONS if it.get("expand") else None,
                                 pdf_max_pages=it.get("pdf_max_pages", 12))
        except Exception as e:
            rec = {"status": f"exception {type(e).__name__}"}
        with lock:
            inflight[h] -= 1
            delay = fetcher.robots.delay(h)
            next_ok[h] = max(next_ok[h], time.time() + delay)
            stats[rec["status"]] += 1
            spent[0] += rec.get("credits") or (1 if rec["status"] == "ok" else 0)
            n = sum(stats.values())
            if n % 25 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] {n}/{total} done | credits≈{spent[0]} | {dict(stats)} | "
                      f"{(time.time() - start) / 60:.1f} min", flush=True)


threads = [threading.Thread(target=worker, daemon=True) for _ in range(args.threads)]
for t in threads:
    t.start()
for t in threads:
    t.join()
print(f"FINISHED {dict(stats)} credits≈{spent[0]} in {(time.time() - start) / 60:.1f} min", flush=True)
