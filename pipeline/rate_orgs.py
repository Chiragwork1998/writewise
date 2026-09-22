"""Tag every student organization with the research categories it evidences (from its own mission text).
Output: extract/organizations_tagged.jsonl (adds primary/secondary category and quirk/impact/culture scores)."""
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from firecrawl_client import load_env  # noqa: E402

college = Path(sys.argv[1])
load_env(Path(__file__).resolve().parent.parent / ".env")
KEY = os.environ["DEEPSEEK_API_KEY"]
orgs = [json.loads(l) for l in (college / "extract" / "organizations.jsonl").read_text().splitlines()]

SYSTEM = """You tag university student organizations using ONLY the given name, directory categories and mission text.
For each organization return:
- "q": quirkiness 0-3 (3 = genuinely fun, unusual, whimsical or distinctive: e.g. building escape rooms, a cappella
  for a niche genre, quidditch, cheese appreciation; 0 = ordinary professional/academic/cultural club)
- "s": social impact 0-3 (3 = the mission is direct community service, nonprofit work, advocacy or civic engagement)
- "d": diversity/cultural/international/identity/faith community 0-3
- "r": research/academic-enrichment 0-3
- "p": pre-professional/career 0-3
Judge only from the text; do not assume. Return json {"items":[{"i":0,"q":0,"s":0,"d":0,"r":0,"p":0}, ...]}"""

out = []
lock = threading.Lock()


def run(batch):
    lines = "\n".join(f"{i}. {o['name']} | {', '.join(o['categories'])} | {(o['mission'] or '')[:400]}"
                      for i, o in batch)
    for attempt in range(4):
        try:
            r = httpx.post("https://api.deepseek.com/chat/completions", timeout=180, headers={"Authorization": f"Bearer {KEY}"},
                           json={"model": "deepseek-flash", "thinking": {"type": "disabled"}, "temperature": 0,
                                 "response_format": {"type": "json_object"},
                                 "messages": [{"role": "system", "content": SYSTEM},
                                              {"role": "user", "content": lines}]})
            items = json.loads(r.json()["choices"][0]["message"]["content"])["items"]
            got = {int(x["i"]): x for x in items}
            with lock:
                for i, o in batch:
                    x = got.get(i, {})
                    o2 = dict(o)
                    o2["scores"] = {k: int(x.get(k, 0)) for k in "qsdrp"}
                    out.append(o2)
            return
        except Exception:
            time.sleep(3 * (attempt + 1))


idx = list(enumerate(orgs))
batches = [idx[i:i + 40] for i in range(0, len(idx), 40)]
with ThreadPoolExecutor(12) as ex:
    list(ex.map(run, batches))
out.sort(key=lambda o: o["name"].lower())
with (college / "extract" / "organizations_tagged.jsonl").open("w") as f:
    for o in out:
        f.write(json.dumps(o) + "\n")
print(f"tagged {len(out)}/{len(orgs)}; quirky(q>=2)={sum(1 for o in out if o['scores']['q'] >= 2)} "
      f"impact(s>=2)={sum(1 for o in out if o['scores']['s'] >= 2)} diversity(d>=2)={sum(1 for o in out if o['scores']['d'] >= 2)}")
print("quirky examples:", [o["name"] for o in out if o["scores"]["q"] == 3][:40])
