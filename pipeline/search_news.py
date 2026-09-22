"""External references via Firecrawl /search: independent coverage of the school (category 10).

Saves every result (title, URL, snippet, date) to discovery/search_results.jsonl and writes a scrape queue of the
most relevant non-university articles.
usage: python pipeline/search_news.py colleges/usc queries.txt queues/news.jsonl
"""
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent))
from firecrawl_client import default_fetcher  # noqa: E402

college, qfile, out_q = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
from college_config import alt, load as load_config, root_or_default  # noqa: E402
CFG = load_config(college)
ROOT = root_or_default(CFG)
f = default_fetcher(college)
res_path = college / "discovery" / "search_results.jsonl"
done_q = set()
if res_path.exists():
    done_q = {json.loads(l)["query"] for l in res_path.read_text().splitlines()}

queries = [q.strip() for q in qfile.read_text().splitlines() if q.strip() and not q.startswith("#")]
for q in queries:
    if q in done_q:
        continue
    for attempt in range(3):
        j = f.search(q, limit=10)
        if j.get("success"):
            break
        time.sleep(15)
    data = j.get("data") or {}
    items = data.get("web", []) if isinstance(data, dict) else data
    items += (data.get("news", []) if isinstance(data, dict) else [])
    with res_path.open("a") as fh:
        for it in items:
            fh.write(json.dumps({"query": q, "url": it.get("url"), "title": it.get("title"),
                                 "snippet": it.get("description") or it.get("snippet"), "date": it.get("date"),
                                 "position": it.get("position")}) + "\n")
    print(f"{len(items):3} results | {q}", flush=True)
    time.sleep(2)

TRUSTED = re.compile("(" + alt(CFG["trusted_news_domains"]) + ")")
rows = [json.loads(l) for l in res_path.read_text().splitlines()]
seen, queue = set(), []
for r in rows:
    u = (r.get("url") or "").split("#")[0]
    h = urlparse(u).hostname or ""
    if not u or u in seen or h.endswith(ROOT) or not TRUSTED.search(h):
        continue
    seen.add(u)
    queue.append({"url": u, "bucket": "external_news", "priority": 60 - (r.get("position") or 5), "title": r.get("title"),
                  "query": r["query"]})
with out_q.open("w") as fh:
    for x in queue:
        fh.write(json.dumps(x) + "\n")
print(f"results={len(rows)} unique trusted external articles queued={len(queue)}")
