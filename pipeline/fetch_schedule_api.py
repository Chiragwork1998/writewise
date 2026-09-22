"""Course schedule via the Schedule of Classes' own public read-only data endpoint, fetched through Firecrawl
(self-hosted works, since no browser clicks are needed). Produces extract/courses_<term>.jsonl in the same shape
as the click-based cloud parser, so the two runs can be compared field by field.

usage: FIRECRAWL_SELFHOST=1 FIRECRAWL_API_URL=http://127.0.0.1:3002/v2 \
       python pipeline/fetch_schedule_api.py colleges/usc_selfhost 20263
"""
import html
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from firecrawl_client import default_fetcher  # noqa: E402

college, TERM = Path(sys.argv[1]), sys.argv[2]
TERM_NAME = {"20263": "Fall 2026", "20261": "Spring 2026", "20262": "Summer 2026"}.get(TERM, TERM)
BASE = "https://classes.usc.edu"
f = default_fetcher(college)
raw_dir = college / "schedule_api"
raw_dir.mkdir(exist_ok=True)


def get_json(url):
    """Scrape a JSON endpoint through Firecrawl and parse the body out of the returned document."""
    rec = f.scrape(url, meta={"bucket": "schedule_api"}, formats=("rawHtml", "markdown"))
    if rec.get("status") != "ok":
        return None, rec.get("error")
    text = rec.get("raw_html") or rec.get("markdown") or ""
    m = re.search(r"\{.*\}", html.unescape(re.sub(r"<[^>]+>", "", text)), re.S)
    if not m:
        return None, "no JSON in response"
    try:
        return json.loads(m.group(0)), None
    except Exception as e:
        return None, f"JSON parse failed: {e}"


# 1. department list for the term (from the public catalogue page)
index = f.scrape(f"{BASE}/term/{TERM}/catalogue/school", meta={"bucket": "schedule_index"})
pairs = sorted({(m.group(1), m.group(2)) for m in
                (re.match(rf"{BASE}/term/{TERM}/school/([^/]+)/program/([^/?#]+)$", l.split("#")[0])
                 for l in index.get("links", []) if l) if m})
print(f"departments found: {len(pairs)}", flush=True)


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def fmt_units(units, max_units):
    vals = [v for v in (num(u) for u in (units or [])) if v is not None]
    if not vals:
        return None
    lo, hi = min(vals), max(vals)
    s = f"{lo:.1f}" if lo == hi else f"{lo:.1f} - {hi:.1f}"
    mx = num(max_units)
    return f"{s} Unit{'' if (lo == hi and lo == 1) else 's'}" + (f", Max {mx:.1f}" if mx else "")


def fmt_schedule(sched):
    out = []
    for s in sched or []:
        days = ", ".join(s.get("days") or [])
        t = f"{s.get('startTime')}-{s.get('endTime')}" if s.get("startTime") else "TBA"
        out.append(", ".join(x for x in (days, t) if x))
    return "; ".join(out) or "TBA"


def one(pair):
    school, program = pair
    api = f"{BASE}/api/Courses/CoursesByTermSchoolProgram?termCode={TERM}&school={school}&program={program}"
    page_url = f"{BASE}/term/{TERM}/school/{school}/program/{program}"
    data, err = get_json(api)
    if not data:
        return school, program, [], err
    (raw_dir / f"{school}__{program}.json").write_text(json.dumps(data))
    courses = []
    for c in data.get("courses", []):
        pc = c.get("publishedCourseCode") or c.get("scheduledCourseCode") or {}
        code = (pc.get("courseSpace") or c.get("fullCourseName") or "").strip()
        num = (pc.get("number") or c.get("classNumber") or "") + (pc.get("suffix") or "")
        secs = []
        for s in c.get("sections", []):
            names = [" ".join(x for x in (i.get("firstName"), i.get("lastName")) if x).strip()
                     for i in (s.get("instructors") or [])]
            secs.append({"id": s.get("sisSectionId"), "status": "CANCELLED" if s.get("isCancelled") else
                         ("FULL" if s.get("isFull") else "OPEN"), "units": str(s.get("units")) if s.get("units") else None,
                         "type": s.get("rnrMode"), "schedule": fmt_schedule(s.get("schedule")),
                         "instructors": [n for n in names if n],
                         "syllabus_url": s.get("syllabus"),
                         "registered": (f"{s.get('registeredSeats')} / {s.get('totalSeats')}"
                                        if s.get("totalSeats") is not None else None),
                         "waitlisted": s.get("waitlistedSeats"), "session": s.get("session"), "notes": s.get("notes")})
        courses.append({
            "code": code, "dept": pc.get("prefix") or program, "number": num,
            "cross_listed": "yes" if c.get("isCrossListed") else None,
            "title": (c.get("name") or "").strip(), "units": fmt_units(c.get("courseUnits"), c.get("maxUnits")),
            "available_seats": c.get("remainingSectionSeats"), "overall_status": None,
            "description": (c.get("description") or "").strip() or None,
            "general_education": f"GE code {c['geCode']}" if c.get("geCode") else None,
            "prerequisites": c.get("prerequisiteCourseCodes"), "corequisites": c.get("corequisiteCourseCodes"),
            "restrictions": {k: c.get(k) for k in ("majorRestrictions", "schoolRestrictions", "courseRestrictions") if c.get(k)},
            "duplicate_credit": c.get("duplicateCredit"), "recommended_prep": c.get("recommendedPrep"),
            "course_notes": c.get("courseNotes"), "sections": secs, "term": TERM_NAME,
            "school_code": school, "program_code": program, "source_url": page_url, "data_url": api,
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "instructors": sorted({n for s in secs for n in s["instructors"]}),
        })
    return school, program, courses, None


all_courses, errors = [], []
with ThreadPoolExecutor(6) as ex:
    for i, (school, program, courses, err) in enumerate(ex.map(one, pairs), 1):
        all_courses += courses
        if err:
            errors.append((school, program, err))
        if i % 50 == 0:
            print(f"{i}/{len(pairs)} departments, {len(all_courses)} courses", flush=True)
(college / "extract").mkdir(exist_ok=True)
with (college / "extract" / f"courses_{TERM}.jsonl").open("w") as fh:
    for c in all_courses:
        fh.write(json.dumps(c) + "\n")
ug = [c for c in all_courses if c["number"][:1] in "1234"]
print(f"departments={len(pairs)} courses={len(all_courses)} undergrad={len(ug)} "
      f"sections={sum(len(c['sections']) for c in all_courses)} "
      f"instructors={len({n for c in all_courses for n in c['instructors']})} errors={len(errors)}")
for e in errors[:10]:
    print("  error:", e)
