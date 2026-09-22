"""Re-scrape the A/B sample through a self-hosted Firecrawl with the same options the cloud run used.
usage: FIRECRAWL_API_URL=http://127.0.0.1:3002/v2 python pipeline/ab_run.py colleges/usc_ab"""
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import firecrawl_client as fc  # noqa: E402

college = Path(sys.argv[1])
assert "api.firecrawl.dev" not in fc.API, "point FIRECRAWL_API_URL at the self-hosted instance"
fetcher = fc.Fetcher(college, [("self-hosted", 8, 10000)], default_host_gap=1.0)
sample = [json.loads(l) for l in (college / "sample.jsonl").read_text().splitlines()]
out = college / "ab_results.jsonl"
lock = threading.Lock()
start = time.time()


def run(p):
    t0 = time.time()
    rec_actions = None
    if p["stratum"].startswith("schedule"):
        rec_actions = fetcher.scrape(p["url"], meta={"bucket": p["bucket"]}, actions=fc.EXPAND_ALL_ACTIONS, retries=0)
        if rec_actions.get("status") == "ok":
            rec = rec_actions
        else:
            rec = fetcher.scrape(p["url"], meta={"bucket": p["bucket"]}, retries=1)
    else:
        rec = fetcher.scrape(p["url"], meta={"bucket": p["bucket"]}, retries=1)
    row = {"url": p["url"], "stratum": p["stratum"], "status": rec.get("status"), "error": rec.get("error"),
           "http_status": rec.get("http_status"), "chars": len(rec.get("markdown") or ""),
           "elapsed": rec.get("elapsed") or round(time.time() - t0, 1),
           "actions_status": None if rec_actions is None else rec_actions.get("status"),
           "actions_error": None if rec_actions is None else rec_actions.get("error")}
    with lock:
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")


with ThreadPoolExecutor(8) as ex:
    for i, _ in enumerate(ex.map(run, sample), 1):
        if i % 25 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] {i}/{len(sample)} in {(time.time() - start) / 60:.1f} min", flush=True)
print(f"DONE {len(sample)} pages in {(time.time() - start) / 60:.1f} min", flush=True)
