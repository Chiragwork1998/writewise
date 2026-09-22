"""Crawl + extraction statistics report, coverage metrics, recall test and per-college cost estimate.
usage: python pipeline/build_stats.py colleges/usc "University of Southern California" usc.edu"""
import collections
import glob
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from firecrawl_client import load_env  # noqa: E402

college, SCHOOL, ROOT = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
from college_config import load as load_config  # noqa: E402
CFG = load_config(college)
SHORT = CFG["short"] or SCHOOL
load_env(Path(__file__).resolve().parent.parent / ".env")
disc, logs, ext, out = college / "discovery", college / "logs", college / "extract", college / "output"
out.mkdir(exist_ok=True)

# ---------------- discovery
hosts = [json.loads(l) for l in (disc / "hosts.jsonl").read_text().splitlines()]
live = [h for h in hosts if h.get("status") and h["status"] < 400]
sites_all = (disc / "sites_all.txt").read_text().split()
sitemap_total = 0
for f in glob.glob(str(disc / "sitemaps" / "*.json")):
    sitemap_total += len(json.load(open(f))["sitemap_urls"])
maps = [json.load(open(f)) for f in glob.glob(str(disc / "maps" / "*.json"))]
map_urls = sum(len(m.get("links", [])) for m in maps)
ratings = [json.loads(l) for l in (disc / "ratings.jsonl").read_text().splitlines()] if (disc / "ratings.jsonl").exists() else []

# ---------------- fetching
log = [json.loads(l) for l in (logs / "fetch_log.jsonl").read_text().splitlines()]
scrapes = [r for r in log if r["status"] not in ("map", "search")]
ok = {}
for r in scrapes:
    if r["status"] == "ok":
        ok[r["url"]] = r
status = collections.Counter(r["status"] for r in scrapes)
bucket_pages = collections.Counter((r.get("bucket") or "other") for r in ok.values())
host_pages = collections.Counter(urlparse(u).hostname for u in ok)
credits = collections.Counter()
for r in log:
    kind = r["status"] if r["status"] in ("map", "search") else (r.get("bucket") or "other")
    credits[kind] += r.get("credits") or 0
fc_total = sum(credits.values())
SELF = "selfhost" in college.name  # self-hosted Firecrawl: "credits" are just request counts, no cloud billing
FC = "self-hosted Firecrawl" if SELF else "Firecrawl"
chars = sum(r.get("chars") or 0 for r in ok.values())

balances = {}
for name, label in (("FIRECRAWL_API_KEY", "main key (Hobby plan)"), ("FIRECRAWL_API_KEY_2", "second key (free plan)")):
    k = os.environ.get(name)
    if not k:
        continue
    try:
        j = httpx.get("https://api.firecrawl.dev/v2/team/credit-usage", headers={"Authorization": f"Bearer {k}"}, timeout=30).json()
        balances[label] = j.get("data", {}).get("remainingCredits")
    except Exception:
        pass
try:
    ds = httpx.get("https://api.deepseek.com/user/balance", headers={"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}"},
                   timeout=30).json()["balance_infos"][0]["total_balance"]
except Exception:
    ds = None

# ---------------- extraction + datasets
raw = [json.loads(l) for l in (ext / "facts_raw.jsonl").read_text().splitlines()]
bstats = json.loads((out / "build_stats.json").read_text()) if (out / "build_stats.json").exists() else {}
orgs = [json.loads(l) for l in (ext / "organizations_tagged.jsonl").read_text().splitlines()]
courses = []
for cf in glob.glob(str(ext / "courses_*.jsonl")):
    courses += [json.loads(l) for l in open(cf)]
ug = [c for c in courses if c["number"][:1] in CFG["undergrad_course_first_digits"]]
ug_instr = {n for c in ug for n in c["instructors"]}
entities = {(str((f.get("entity") or {}).get("type")), re.sub(r"\W+", " ", str((f.get("entity") or {}).get("name")).lower()).strip()) for f in raw}
relations = sum(len(f.get("relations") or []) for f in raw)
fac_pages = [r for r in ok.values() if r.get("bucket") == "faculty_profile"]
fac_queue = []
for q in glob.glob(str(college / "queues" / "faculty*.jsonl")):
    fac_queue += [json.loads(l) for l in open(q)]
fac_done = {x["instructor"] for x in fac_queue if x["url"] in ok}
cats = collections.Counter(f.get("category") for f in raw)

# ---------------- recall test (items named in the earlier ChatGPT research)
RECALL = CFG["recall_items"]  # config/college.json: things a good researcher expects to find
blob_facts = "\n".join(f"{f['fact']} || {f['evidence']} || {(f.get('entity') or {}).get('name')}" for f in raw)
blob_orgs = "\n".join(f"{o['name']} || {o['mission']}" for o in orgs)
blob_courses = "\n".join(f"{c['code']} {c['title']} || {', '.join(c['instructors'])}" for c in courses)
# ambiguous names get a pattern that requires the right context
CONTEXT = CFG["recall_context"]  # ambiguous names get a pattern that requires the right context
recall = []
for item in RECALL:
    if item in CONTEXT:
        rx = re.compile(CONTEXT[item])
    else:
        rx = re.compile(r"(?<![A-Za-z])" + re.escape(item) + r"(?![A-Za-z])", 0 if item in CFG["recall_case_sensitive"] else re.I)
    where, proof = [], ""
    for n, b in (("org directory", blob_orgs), ("class schedule", blob_courses), ("facts", blob_facts)):
        m = rx.search(b)
        if m:
            where.append(n)
            if not proof:
                ls = b.rfind("\n", 0, m.start()) + 1
                le = b.find("\n", m.end())
                proof = b[ls: le if le > 0 else None].split(" || ")[0][:160]
    recall.append((item, where, proof))
found = sum(1 for _, w, _ in recall if w)

# ---------------- write report
now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
verified = bstats.get("facts", {}).get("verified_unique")
vex = sum(v for k, v in bstats.get("facts", {}).items() if k in ("verify_exact", "verify_exact_words", "verify_near"))
headline = (f"{len(ok):,} pages scraped from {len(host_pages):,} sites via {FC} ({'0 cloud credits' if SELF else f'{fc_total:,} credits'}); "
            f"{verified or 0:,} verified facts, {len(orgs):,} student organizations, {len(courses):,} scheduled courses "
            f"with {len({n for c in courses for n in c['instructors']}):,} instructors")
(out / "run_stats.json").write_text(json.dumps({"headline": headline}))

md = [f"# {SCHOOL} — Crawl & Extraction Report", f"*Generated {now}*\n", f"**Headline:** {headline}.\n"]
md += ["## 1. Discovery (finding what exists)\n",
       *(["*Self-hosted run: the URL list and discovery below were reused from the cloud run so both fetch the same pages. See comparison_report for self-hosted map/search results.*\n"] if SELF else []),
       "| Step | Result |", "|---|---|",
       f"| Hostnames found in public certificate logs + seeds | {len(hosts):,} |",
       f"| Hostnames that answered as websites | {len(live):,} |",
       f"| Distinct sites after following links from every homepage | {len(sites_all):,} |",
       f"| URLs listed in those sites' sitemaps | {sitemap_total:,}" + (f" ({CFG['sitemap_note']})" if CFG["sitemap_note"] else "") + " |",
       f"| Firecrawl map calls / URLs returned (with titles) | {len(maps)} / {map_urls:,} |",
       f"| Candidate URLs rated for relevance by AI (title + URL only) | {len(ratings):,} (core: {sum(1 for r in ratings if r['r'] == 3):,}; useful: {sum(1 for r in ratings if r['r'] == 2):,}) |", ""]
md += ["## 2. Scraping (Firecrawl)\n", "| Metric | Value |", "|---|---|",
       f"| Pages scraped successfully | {len(ok):,} |",
       f"| Sites represented | {len(host_pages):,} |",
       f"| Text captured | {chars / 1e6:.1f} million characters |",
       f"| Skipped because robots.txt disallows | {status.get('robots_disallowed', 0):,} |",
       f"| Failed after retries | {status.get('error', 0):,} |",
       (f"| Firecrawl requests (self-hosted, not billed) | {fc_total:,} |" if SELF else f"| Firecrawl credits used (logged) | {fc_total:,} |"),
       f"| Credits remaining now | " + ", ".join(f"{k}: {v:,}" for k, v in balances.items() if v is not None) + " |", "",
       "**Credits by purpose**\n", "| Purpose | Credits |", "|---|---:|"]
md += [f"| {k} | {v:,} |" for k, v in credits.most_common()]
md += ["", "**Pages by purpose**\n", "| Purpose | Pages |", "|---|---:|"]
md += [f"| {k} | {v:,} |" for k, v in bucket_pages.most_common()]
md += ["", "**Top 30 sites by pages scraped**\n", "| Site | Pages |", "|---|---:|"]
md += [f"| {h} | {n:,} |" for h, n in host_pages.most_common(30)]
md += ["", "## 3. Extraction & verification\n", "| Metric | Value |", "|---|---|",
       f"| Facts extracted by AI (with verbatim quotes) | {len(raw):,} |",
       f"| Quotes confirmed present on the source page | {vex:,} ({(vex / max(len(raw), 1)) * 100:.1f}%) |",
       f"| Unique verified facts used in documents | {verified or 0:,} |",
       f"| Distinct entities named | {len(entities):,} |",
       f"| Relationships extracted (e.g. professor DIRECTS lab) | {relations:,} |",
       f"| Student organizations (official directory, parsed exactly) | {len(orgs):,} |",
       f"| Scheduled courses parsed exactly / undergraduate | {len(courses):,} / {len(ug):,} |",
       f"| Sections with instructor, schedule and enrollment | {sum(len(c['sections']) for c in courses):,} |",
       f"| Instructors of undergraduate courses | {len(ug_instr):,} |",
       f"| Of those, faculty profile pages scraped | {len(fac_done):,} |",
       *([] if SELF else [f"| DeepSeek balance now | ${ds} |"]), "",
       "**Facts by category code**\n", "| Code | Facts |", "|---|---:|"]
md += [f"| {k} | {v:,} |" for k, v in cats.most_common()]
md += ["", f"## 4. Recall test — {found}/{len(RECALL)} items from the earlier ChatGPT research found\n",
       "Items ChatGPT named, searched for in the verified dataset. \"Not found\" means the item does not appear in any "
       f"page we scraped or in {SHORT}'s official directories — it may be renamed, defunct, or on a page outside the budget.\n",
       "| Item | Found in | Example of what was captured |", "|---|---|---|"]
md += [f"| {i} | {', '.join(w) if w else '**not found**'} | {pr.replace('|', '/')} |" for i, w, pr in recall]
md += ["", "## 5. Cost per college (this run)\n",
       *([f"- Firecrawl: self-hosted in Docker, no credits ({fc_total:,} requests). Machine load is in comparison_report.",
          "- DeepSeek: fact extraction and digest only (URL triage and organization tagging were reused); spend is "
          "tracked in comparison_report."] if SELF else [
       f"- Firecrawl: {fc_total:,} credits. At list prices that is about ${fc_total * 19 / 5000:,.0f} on the Hobby plan "
       f"($19 per 5,000), ${fc_total * 83 / 100000:,.2f} on Standard billed yearly ($83 per 100,000), "
       f"${fc_total * 333 / 500000:,.2f} on Growth billed yearly.",
       "- DeepSeek (URL triage, fact extraction, organization tagging, digest): record the balance before and after the run; "
       "the account balance alone cannot show one college's spend."]),
       "- A fuller crawl (every rated-useful page plus all faculty profiles) would add roughly "
       f"{max(0, sum(1 for r in ratings if r['r'] >= 2) - bucket_pages.get(CFG['schedule_bucket'] or '', 0)):,} more pages of candidates "
       "already identified.", "",
       "## 6. Known gaps\n",
       f"- {status.get('robots_disallowed', 0)} URLs were skipped because the site's robots.txt disallows them; crawl-delay "
       "requests (e.g. 10–120 seconds per page) were honoured, which limited how many pages some sites could provide.",
       "- Budget: only the highest-rated pages were scraped; the candidate list above shows what remains.",
       *[f"- {g}" for g in CFG["known_gaps"]]]
(out / "crawl_report.md").write_text("\n".join(md) + "\n")
print(headline)
print(f"recall {found}/{len(RECALL)}; not found:", [i for i, w, _ in recall if not w])
