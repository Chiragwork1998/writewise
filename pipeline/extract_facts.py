"""Fact extraction with verbatim-evidence verification.

For every scraped page (except ones parsed deterministically), DeepSeek extracts facts in the 10 categories.
Each fact must carry an exact quote from the page; a script then checks the quote really appears in the page
text. Facts whose quote cannot be found are kept only in the audit file and never reach the documents.

usage: python pipeline/extract_facts.py colleges/usc "University of Southern California (USC)" [--limit N] [--model deepseek-flash]
"""
import argparse
import collections
import glob
import json
import os
import re
import sys
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from firecrawl_client import load_env  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("college")
ap.add_argument("school")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--model", default="deepseek-flash")
ap.add_argument("--threads", type=int, default=24)
ap.add_argument("--only", default="", help="substring filter on URL")
ap.add_argument("--sample", default="", help="file of URLs (one per line) to process; site boilerplate is still "
                                          "learned from all pages so cleaning matches a full run")
ap.add_argument("--out", default="", help="write extract files to this directory instead of <college>/extract")
args = ap.parse_args()

college = Path(args.college)
load_env(Path(__file__).resolve().parent.parent / ".env")
KEY = os.environ["DEEPSEEK_API_KEY"]
out_dir = Path(args.out) if args.out else college / "extract"
out_dir.mkdir(parents=True, exist_ok=True)
facts_path = out_dir / "facts_raw.jsonl"
done_path = out_dir / "extracted_pages.jsonl"

from college_config import alt, load as load_config  # noqa: E402
CFG = load_config(college)
SKIP_URL = re.compile(alt(CFG["structured_url_patterns"], escape=False))  # parsed by code: never sent to the AI
CHUNK = 12000
FACT_CAP = {"faculty_profile": 14, "external_news": 15, "news": 15, "seed_chatgpt": 45}
FOCUS = {
    "faculty_profile": "This is a faculty profile: capture title and department(s), research areas, labs/centers they "
                       "lead or belong to, courses they teach, work with undergraduates, and the 3 most notable honors. "
                       "Skip degrees, publication lists and minor service roles.",
    "external_news": "This is an independent news article about the university: capture what happened, when, who was "
                     "involved, and any figures or outcomes. Keep critical as well as positive points.",
    "news": "This is a news story: capture what happened, when, who was involved, and figures or outcomes.",
}
CONTACT = re.compile(r"\(\d{3}\)\s*\d{3}-\d{4}|\b\d{3}[-.]\d{3}[-.]\d{4}\b|[\w.+-]+@[\w-]+\.[\w.]+|mail code", re.I)

# ------------------------------------------------------------------ cleaning
pages = []
for f in glob.glob(str(college / "pages" / "*.json")):
    r = json.load(open(f))
    if r.get("status") != "ok" or not r.get("markdown") or SKIP_URL.search(r["url"]):
        continue
    if args.only and args.only not in r["url"]:
        continue
    pages.append(r)

LINK = re.compile(r"!\[[^\]]*\]\([^)]*\)")          # images
MDLINK = re.compile(r"\[([^\]]*)\]\((?:[^()]|\([^)]*\))*\)")  # links -> text


def simplify(line):
    line = LINK.sub("", line)
    line = MDLINK.sub(r"\1", line)
    return line.strip()


host_lines = collections.defaultdict(collections.Counter)
host_pages = collections.Counter()
for r in pages:
    h = urlparse(r["url"]).hostname
    host_pages[h] += 1
    for l in {simplify(x) for x in r["markdown"].splitlines()}:
        if l:
            host_lines[h][l] += 1


def clean_page(r):
    h = urlparse(r["url"]).hostname
    n = host_pages[h]
    out, prev = [], None
    for raw in r["markdown"].splitlines():
        l = simplify(raw)
        if not l or l == prev:
            continue
        if n >= 5 and host_lines[h][l] / n >= 0.4 and len(l) < 300:
            continue  # site-wide navigation / footer
        if re.fullmatch(r"[-*•|#>\s\\]*", l) or re.search(r"cookie|consent preferences|privacy notice", l, re.I) and len(l) < 200:
            continue
        out.append(l)
        prev = l
    return "\n".join(out)


def norm(s):
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("–", "-").replace("—", "-")
    s = re.sub(r"[*_`#>|\\\[\]]", "", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def verify(evidence, page_norm):
    e = norm(evidence)
    if len(e) < 12:
        return "too_short"
    if e in page_norm:
        return "exact"
    words = e.split()
    if len(words) < 6:
        return "failed"
    grams = [" ".join(words[i:i + 5]) for i in range(len(words) - 4)]
    hit = sum(1 for g in grams if g in page_norm) / len(grams)
    return "near" if hit >= 0.85 else "failed"


# ------------------------------------------------------------------ prompt
SYSTEM = f"""You extract facts from one web page for a research database about {args.school}, used by prospective
undergraduate students to judge fit. Accuracy is paramount: a wrong fact is far worse than a missing one.

RULES
1. Extract ONLY what the page text explicitly states. No outside knowledge, no guessing, no interpretation.
2. Every fact needs "e" (evidence): an EXACT, contiguous, verbatim copy of the page text (15-250 characters) that by
   itself proves the fact. Copy characters exactly; do not paraphrase, merge sentences, or fix typos.
3. "f" (fact): one self-contained sentence of at most 35 words naming its subject explicitly (never "they"/"he"/"the
   program"). Keep numbers, names, dates, amounts, eligibility and requirements exactly as written.
4. "p" (period): the time the fact applies to if the page states it (e.g. "Fall 2025", "2024-25", "since 1961"), else null.
5. Be exhaustive about specifics: named clubs, programs, courses (codes), professors, labs, centers, research
   opportunities, partners, traditions, services, requirements, deadlines, funding amounts, statistics, dates.
   Skip navigation, generic marketing phrases with no specifics, cookie text, event logistics, and NEVER extract
   personal phone numbers, email addresses, office room numbers or mail codes.
6. "c" (category): one code. CUL culture (mission, values, history, traditions, spirit, identity); EXT extracurriculars
   (clubs, orgs, Greek life, student government, ensembles, teams); QRK quirks (fun, unusual, human traditions/clubs/
   lore); ACA academics (majors, minors, courses, curricula, teaching, advising, honors); RES research (opportunities,
   labs, centers, undergraduate research, professors' research); SOC social impact (service, service-learning,
   nonprofits, civic engagement, community partners, sustainability action); INN innovative programs (signature,
   rare, interdisciplinary, entrepreneurial, first-of-kind); INT intellectual alignment (academic philosophy,
   pedagogy, ways of thinking); DIV diversity & international (international support, cultural centers, identity
   communities, religious life); NEW news (dated news events about the school); GEN general facts (stats,
   admissions, cost, campus, leadership) that fit nowhere else.
7. "n" and "t": the main subject's exact name as on the page, and its type, one of university|school|department|program|
   course|professor|person|lab|center|organization|tradition|service|office|partner|facility|award|event|publication|other.
8. "r" (relations): links between named entities that the evidence states, as [subject, PREDICATE, object], using predicates
   such as PART_OF, OFFERED_BY, TEACHES, DIRECTS, MEMBER_OF, RESEARCHES, PARTNERS_WITH, FUNDS, AWARDED, LOCATED_IN,
   HOSTS, OPEN_TO, REQUIRES, FOUNDED_IN, AFFILIATED_WITH. Empty list if none.
9. If the page has nothing useful, return an empty facts list.

Return json only:
{{"page_type": "<short label>", "page_date": "<date shown on page or null>",
  "facts": [{{"c": "RES", "f": "...", "e": "...", "p": null, "n": "...", "t": "lab", "r": [["...", "DIRECTS", "..."]]}}]}}"""

lock = threading.Lock()
usage = collections.Counter()
done = set()
if done_path.exists():
    done = {json.loads(l)["key"] for l in done_path.read_text().splitlines()}


def call(user):
    for attempt in range(5):
        try:
            r = httpx.post("https://api.deepseek.com/chat/completions", timeout=300,
                           headers={"Authorization": f"Bearer {KEY}"},
                           json={"model": args.model, "thinking": {"type": "disabled"}, "temperature": 0,
                                 "max_tokens": 7000, "response_format": {"type": "json_object"},
                                 "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]})
            if r.status_code == 429:
                time.sleep(10 * (attempt + 1))
                continue
            j = r.json()
            content = j["choices"][0]["message"]["content"] or ""
            if j["choices"][0].get("finish_reason") == "length":
                return None, j.get("usage", {})
            return json.loads(content), j.get("usage", {})
        except json.JSONDecodeError:
            time.sleep(2)
        except Exception:
            time.sleep(5 * (attempt + 1))
    return None, {}


def work(r):
    text = clean_page(r)
    if len(text) < 200:
        return
    page_norm = norm(text)
    chunks = []
    buf = ""
    for block in re.split(r"\n(?=#)", text):
        if len(buf) + len(block) > CHUNK and buf:
            chunks.append(buf)
            buf = ""
        while len(block) > CHUNK:
            chunks.append(block[:CHUNK])
            block = block[CHUNK:]
        buf += block + "\n"
    if buf.strip():
        chunks.append(buf)
    expanded = []
    for chunk in chunks[:14]:
        expanded.append(chunk)
    chunks = expanded
    for ci, chunk in enumerate(chunks):
        key = f"{r['url']}#{ci}"
        if key in done:
            continue
        bucket = (r.get("meta") or {}).get("bucket") or ""
        cap = FACT_CAP.get(bucket, 40)
        host = urlparse(r["url"]).hostname or ""
        if bucket == "news" or "/news/" in r["url"] or host in CFG["news_hosts_capped"]:
            cap = min(cap, 15)
        guide = FOCUS.get(bucket, "")
        head = f"URL: {r['url']}\nTITLE: {r.get('title') or ''}\nPUBLISHED: {r.get('published') or 'unknown'}\n" \
               f"PART {ci + 1} of {len(chunks)}\nRETURN AT MOST {cap} FACTS: choose the most specific and important. {guide}\n\nPAGE TEXT:\n"
        res, u = call(head + chunk)
        facts_list = (res or {}).get("facts", []) if res else None
        if res is None and len(chunk) > 3000:
            half = len(chunk) // 2
            cut = chunk.rfind("\n", 0, half) if chunk.rfind("\n", 0, half) > 1000 else half
            facts_list, u = [], {}
            for part in (chunk[:cut], chunk[cut:]):
                sub, su = call(head + part)
                if sub:
                    facts_list += sub.get("facts", []) or []
                    for k2, v2 in su.items():
                        if isinstance(v2, int):
                            u[k2] = u.get(k2, 0) + v2
                    res = res or {"page_type": sub.get("page_type"), "page_date": sub.get("page_date")}
        if res is None:
            with lock:
                usage["failed_calls"] += 1
            continue
        rows = []
        for fa in facts_list or []:
            if not isinstance(fa, dict) or not fa.get("f") or not fa.get("e"):
                continue
            if CONTACT.search(fa["e"]) or CONTACT.search(fa["f"]):
                continue
            v = verify(fa["e"], page_norm)
            rows.append({"url": r["url"], "title": r.get("title"), "fetched_at": r.get("fetched_at"),
                         "page_published": r.get("published"), "page_type": res.get("page_type"),
                         "page_date": res.get("page_date"), "bucket": (r.get("meta") or {}).get("bucket"),
                         "category": fa.get("c"), "fact": fa["f"], "evidence": fa["e"], "period": fa.get("p"),
                         "entity": {"name": fa.get("n"), "type": fa.get("t")}, "relations": fa.get("r") or [],
                         "verification": v})
        with lock:
            usage["in"] += u.get("prompt_tokens", 0)
            usage["cached"] += u.get("prompt_cache_hit_tokens", 0)
            usage["out"] += u.get("completion_tokens", 0)
            usage["chunks"] += 1
            usage["facts"] += len(rows)
            usage["verified"] += sum(1 for x in rows if x["verification"] in ("exact", "near"))
            with facts_path.open("a") as f:
                for x in rows:
                    f.write(json.dumps(x) + "\n")
            with done_path.open("a") as f:
                f.write(json.dumps({"key": key, "facts": len(rows)}) + "\n")
            done.add(key)


# spend the remaining budget where it adds the most: profiles and outside coverage first, then thin categories
BUCKET_RANK = {"seed_chatgpt": 0, "faculty_profile": 1, "external_news": 1, "quirks": 2, "intellectual_alignment": 2,
               "innovative_programs": 2, "social_impact": 3, "diversity_international": 3, "culture": 3,
               "extracurriculars": 3, "news": 4, "academics": 5, "research": 5}
pages.sort(key=lambda r: (BUCKET_RANK.get((r.get("meta") or {}).get("bucket"), 2),
                          -((r.get("meta") or {}).get("priority") or 0)))
if args.sample:
    wanted = {l.strip() for l in open(args.sample) if l.strip()}
    pages = [r for r in pages if r["url"] in wanted]
todo = pages[: args.limit] if args.limit else pages
print(f"pages to process: {len(todo)}", flush=True)
start = time.time()
with ThreadPoolExecutor(args.threads) as ex:
    for i, _ in enumerate(ex.map(work, todo)):
        if i % 50 == 0:
            cost = (usage["in"] - usage["cached"]) * 0.30 / 1e6 + usage["cached"] * 0.006 / 1e6 + usage["out"] * 1.2 / 1e6
            print(f"[{time.strftime('%H:%M:%S')}] {i}/{len(todo)} pages | {dict(usage)} | max-cost≈${cost:.2f} | "
                  f"{(time.time() - start) / 60:.1f} min", flush=True)
cost = (usage["in"] - usage["cached"]) * 0.30 / 1e6 + usage["cached"] * 0.006 / 1e6 + usage["out"] * 1.2 / 1e6
print(f"DONE {dict(usage)} peak-price cost≈${cost:.2f}", flush=True)
