"""Simulate a lean crawl on already-scraped data: choose pages the way a budgeted run would (using only information
available BEFORE scraping — URL, title, AI relevance rating), then measure what survives per category.

usage: python pipeline/simulate_budget.py colleges/usc "University of Southern California" usc.edu 500
"""
import collections
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent))
from build_graph import build, load_context  # noqa: E402

college, school, root, BUDGET = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
ns = load_context(college, school, root)
facts, orgs, courses, page_meta = ns["facts"], ns["orgs"], ns["courses"], ns["page_meta"]

log = [json.loads(l) for l in (college / "logs" / "fetch_log.jsonl").read_text().splitlines()]
bucket = {r["url"]: r.get("bucket") for r in log if r["status"] == "ok"}
extracted = {json.loads(l)["key"].rsplit("#", 1)[0] for l in (college / "extract" / "extracted_pages.jsonl").read_text().splitlines()}
ratings = {}
for l in (college / "discovery" / "ratings.jsonl").read_text().splitlines():
    d = json.loads(l)
    if d["url"] not in ratings or d["r"] > ratings[d["url"]]["r"]:
        ratings[d["url"]] = d

# ------------------------------------------------------------------ the lean recipe
plan = collections.OrderedDict()
spent = 0


def take(label, urls, cap, cost_each=1):
    global spent
    chosen = []
    for u in urls:
        if len(chosen) >= cap or spent + cost_each > BUDGET:
            break
        if u in plan:
            continue
        plan[u] = label
        chosen.append(u)
        spent += cost_each
    return chosen


spent += 15      # Firecrawl map on ~15 key sites (discovery; titles for rating)
spent += 20      # 10 targeted site: news searches (2 credits per 10 results)
spent += 10      # retries
ORG_DIR = orgs[0]["source_url"] if orgs else None
take("org directory", [ORG_DIR] if ORG_DIR else [], 1)

UG_SCHOOLS = ["DRNS", "ENGV", "BUS", "ANSC", "CNMA", "FINE", "MUS", "DNCR", "THTR", "ARCH", "PPDP", "ACAD", "ACTN", "OCST", "UGST"]
dept_pages = sorted({c["source_url"] for c in courses if c["term"] == "Fall 2026" and c["school_code"] in UG_SCHOOLS},
                    key=lambda u: (UG_SCHOOLS.index(re.search(r"/school/([^/]+)/", u).group(1)), u))
take("schedule index", ["schedule-index"], 1)
take("course schedule (undergraduate schools)", dept_pages, 140)

candidates = [u for u in bucket if u in extracted and bucket[u] not in ("classes_fall2026", "classes_spring2026")]
LISTING = re.compile(r"directory|our-faculty|/faculty/?$|/people/?$|majors|minors|degree|/programs/?$|academic-programs|"
                     r"centers-and-institutes|/centers/?$|institutes|research-areas|/labs/?$|undergraduate-research|"
                     r"research-opportunit|faculty-mentors|student-organizations|/clubs|assembl|traditions|timeline|"
                     r"history|facts-and-stats|/ge/courses|cultural-cent|centers-offices|/programs$|service-learning", re.I)


def pre_score(u):
    r = ratings.get(u, {})
    title = (page_meta.get(u, {}).get("title") or "")
    return (r.get("r", 1), 1 if LISTING.search(u + " " + title) else 0)


listing = sorted([u for u in candidates if LISTING.search(u + " " + (page_meta.get(u, {}).get("title") or ""))
                  and ratings.get(u, {}).get("r", 1) >= 2], key=lambda u: pre_score(u), reverse=True)
take("directories & listing pages", listing, 60)

QUOTA = {"CUL": 25, "EXT": 20, "QRK": 15, "ACA": 35, "RES": 40, "SOC": 30, "INN": 30, "INT": 15, "DIV": 25, "NEW": 10}
by_cat = collections.defaultdict(list)
for u in candidates:
    r = ratings.get(u)
    if r and r["r"] >= 2 and not re.search(r"^https?://([^/]+\.)?(usc\.edu)", u) is None:
        by_cat[r["c"] if r["c"] in QUOTA else "ACA"].append(u)
for c in by_cat:
    by_cat[c].sort(key=lambda u: (-ratings[u]["r"], -ratings[u].get("score", 0)))
for c, q in QUOTA.items():
    take(f"core pages: {c}", by_cat.get(c, []), q)

external = [u for u in candidates if not (urlparse(u).hostname or "").endswith(root)]
TRUST = re.compile(r"latimes|nytimes|insidehighered|chronicle|apnews|laist|calmatters|edsource|dailytrojan|uscannenbergmedia|wikipedia|govtech|techstars")
external.sort(key=lambda u: (0 if TRUST.search(u) else 1, u))
take("independent news articles", external, 15)
remaining = [u for u in sorted(candidates, key=pre_score, reverse=True)]
take("fill to budget (best remaining)", remaining, 10**6)

lean_pages = {u for u in plan if u.startswith("http")}
lean_courses = {u for u in lean_pages if "/term/" in u}
full_pages = set(bucket)

# ------------------------------------------------------------------ measure
def cat_counts(page_filter):
    out = collections.Counter()
    ents = collections.defaultdict(set)
    for fa in facts:
        if page_filter is not None and not (set(fa["sources"]) & page_filter):
            continue
        out[fa["code"]] += 1
        ents[fa["code"]].add(ns["resolve"](((fa.get("entity") or {}).get("type") or "other").lower(),
                                           (fa.get("entity") or {}).get("name") or "")[0])
    return out, {k: len(v) for k, v in ents.items()}


full_f, full_e = cat_counts(None)
lean_f, lean_e = cat_counts(lean_pages)
full_nodes, full_edges = build(ns)
lean_nodes, lean_edges = build(ns, page_filter=lean_pages, course_filter=lean_courses)

ug_full = [c for c in courses if c["number"][:1] in "1234"]
ug_lean = [c for c in ug_full if c["source_url"] in lean_courses]
ins_full = {n for c in ug_full for n in c["instructors"]}
ins_lean = {n for c in ug_lean for n in c["instructors"]}

RECALL = ["Bhaskar Krishnamachari", "Autonomous Networks Research Group", "GEOL 150", "Lowell Stott", "ENST 492",
          "Security and Political Economy", "Escape SC", "advocaSC", "CLOVER", "Trojan Research Association",
          "180 Degrees Consulting", "SC Outfitters", "Joint Educational Project", "Peace Project", "WonderKids",
          "Iovine and Young", "Interdisciplinary Major", "Dornsife Toolkit", "Traveler", "Tommy Trojan", "Spirit of Troy",
          "Tirebiter", "Office of International Services", "International Student Assembly", "La CASA", "SURF", "CURVE",
          "Bridge Undergraduate Science", "Open Dialogue Project", "Thematic Option", "Room to Read"]


def recall(page_filter, course_filter):
    blob = "\n".join(fa["fact"] + " " + fa["evidence"] for fa in facts
                     if page_filter is None or set(fa["sources"]) & page_filter)
    blob += "\n".join(o["name"] + " " + (o["mission"] or "") for o in orgs)
    blob += "\n".join(c["code"] + " " + c["title"] + " " + " ".join(c["instructors"]) for c in courses
                      if course_filter is None or c["source_url"] in course_filter)
    return {i for i in RECALL if re.search(re.escape(i), blob, 0 if i in ("CLOVER", "SURF", "CURVE") else re.I)}


rec_full, rec_lean = recall(None, None), recall(lean_pages, lean_courses)
NAMES = {c: n for _, c, n, _ in ns["CATS"]}
NAMES["GEN"] = "General facts"
rows = []
for c in ["CUL", "EXT", "QRK", "ACA", "RES", "SOC", "INN", "INT", "DIV", "NEW", "GEN"]:
    rows.append((NAMES[c], full_f[c], lean_f[c], full_e.get(c, 0), lean_e.get(c, 0)))

md = [f"# What a {BUDGET}-credit crawl keeps — simulation on USC", "",
      f"Pages were chosen the way a budgeted run would choose them (URL, title and AI rating only — no hindsight), "
      f"from the {len(full_pages):,} pages this run scraped. Faculty profiles and news articles still being processed "
      f"are not yet included on either side.", "",
      "## Budget used", "", "| Part | Credits |", "|---|---:|", "| Discovery (Firecrawl map) | 15 |",
      "| News searches | 20 |", "| Retries | 10 |"]
parts = collections.Counter(plan.values())
md += [f"| {k} | {v} |" for k, v in parts.items()]
md += [f"| **Total** | **{spent}** |", "", f"**Pages read:** {len(lean_pages):,} (full run so far: {len(full_pages):,}).", "",
       "## Coverage by category", "", "| Category | Facts (full) | Facts (lean) | Kept | Entities (full) | Entities (lean) | Kept |",
       "|---|---:|---:|---:|---:|---:|---:|"]
for name, ff, lf, fe, le in rows:
    md.append(f"| {name} | {ff:,} | {lf:,} | {lf / max(ff, 1) * 100:.0f}% | {fe:,} | {le:,} | {le / max(fe, 1) * 100:.0f}% |")
md += ["", "## Structured datasets & graph", "", "| Item | Full | Lean | Kept |", "|---|---:|---:|---:|",
       f"| Student organizations with missions | {len(orgs):,} | {len(orgs):,} | 100% |",
       f"| Undergraduate courses (Fall 2026) | {len(ug_full):,} | {len(ug_lean):,} | {len(ug_lean) / max(len(ug_full), 1) * 100:.0f}% |",
       f"| Instructors of undergraduate courses | {len(ins_full):,} | {len(ins_lean):,} | {len(ins_lean) / max(len(ins_full), 1) * 100:.0f}% |",
       f"| Graph nodes | {len(full_nodes):,} | {len(lean_nodes):,} | {len(lean_nodes) / len(full_nodes) * 100:.0f}% |",
       f"| Graph edges (with provenance) | {len(full_edges):,} | {len(lean_edges):,} | {len(lean_edges) / len(full_edges) * 100:.0f}% |",
       f"| Recall checklist items found | {len(rec_full)}/{len(RECALL)} | {len(rec_lean)}/{len(RECALL)} | |", "",
       "Recall items lost in the lean run: " + (", ".join(sorted(rec_full - rec_lean)) or "none")]
(college / "output" / f"simulation_{BUDGET}.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
