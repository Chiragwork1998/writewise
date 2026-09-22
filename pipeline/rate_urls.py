"""DeepSeek relevance rating of candidate URLs (title + description + URL only, no scraping).

usage: python pipeline/rate_urls.py colleges/usc candidates.jsonl ratings.jsonl "University of Southern California"
"""
import json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import httpx
sys.path.insert(0, str(Path(__file__).parent))
from firecrawl_client import load_env

college, cand_path, out_path, school = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
from college_config import load as load_config  # noqa: E402
Y = load_config(college)["current_year"]
load_env(Path(__file__).resolve().parent.parent / ".env")
KEY = os.environ["DEEPSEEK_API_KEY"]

SYSTEM = f"""You triage web pages for a college-fit research product about {school}. The product helps prospective
UNDERGRADUATE students understand a university in depth across 10 categories:
CUL culture: mission, values, ethos, history, traditions, school spirit, identity
EXT extracurriculars: specific clubs, student organizations, Greek life, student government, ensembles, club sports
QRK quirks: fun, unusual, human traditions/clubs/rituals/lore that make the school distinctive
ACA academics: specific majors, minors, courses, curricula, teaching, advising, honors, faculty who teach undergrads
RES research: specific research opportunities, labs, centers, undergraduate research programs, professors' research
SOC social impact: community service, service-learning, nonprofit partners, civic engagement, sustainability action
INN innovative programs: signature, rare, interdisciplinary, entrepreneurial or first-of-their-kind programs
INT intellectual alignment: academic philosophy, pedagogy, ways of thinking, deans' visions, open dialogue
DIV diversity & international: international student support, cultural centers, identity communities, religious life
NEW external/news: notable news about the school's students, academics, research, culture (prefer {Y - 2}-{Y})

Rate each page from its URL, title and description:
3 = core source: likely contains specific, substantive, citable facts in a category for undergraduates
2 = useful supporting detail
1 = marginal
0 = irrelevant: admin/HR/IT/finance/facilities, clinical patient care, graduate-only professional or executive
    programs with nothing distinctive, event listings, job posts, fundraising, login/forms, publication lists,
    staff pages, generic navigation, archived news older than {Y - 3} unless about traditions/history
Return json only: {{"items":[{{"i":<index>,"r":<0-3>,"c":"<one code>"}}, ...]}} with one entry per input line."""

cands = [json.loads(l) for l in cand_path.read_text().splitlines()]
done = {}
if out_path.exists():
    for l in out_path.read_text().splitlines():
        d = json.loads(l); done[d["url"]] = d
todo = [c for c in cands if c["url"] not in done]
print("candidates", len(cands), "to rate", len(todo), flush=True)
lock = threading.Lock()
usage = {"in": 0, "out": 0, "calls": 0}

def call(batch):
    lines = "\n".join(f"{i} {c['url']} | {c.get('title','')[:110]}" for i, c in enumerate(batch))
    for attempt in range(4):
        try:
            r = httpx.post("https://api.deepseek.com/chat/completions", timeout=180,
                           headers={"Authorization": f"Bearer {KEY}"},
                           json={"model": "deepseek-flash", "thinking": {"type": "disabled"}, "temperature": 0,
                                 "response_format": {"type": "json_object"},
                                 "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": lines}]})
            j = r.json()
            items = json.loads(j["choices"][0]["message"]["content"])["items"]
            got = {int(it["i"]): it for it in items if "i" in it}
            with lock:
                usage["in"] += j["usage"]["prompt_tokens"]; usage["out"] += j["usage"]["completion_tokens"]; usage["calls"] += 1
                with out_path.open("a") as f:
                    for i, c in enumerate(batch):
                        it = got.get(i)
                        if it is None:
                            continue
                        f.write(json.dumps({"url": c["url"], "r": int(it.get("r", 0)), "c": it.get("c", ""),
                                            "host": c["host"], "score": c["score"], "title": c.get("title", "")}) + "\n")
            return
        except Exception as e:
            time.sleep(3 * (attempt + 1))

batches = [todo[i:i + 50] for i in range(0, len(todo), 50)]
with ThreadPoolExecutor(16) as ex:
    for n, _ in enumerate(ex.map(call, batches)):
        if n % 20 == 0:
            print(f"{n}/{len(batches)} batches | tokens in {usage['in']} out {usage['out']}", flush=True)
print("DONE", usage, flush=True)
