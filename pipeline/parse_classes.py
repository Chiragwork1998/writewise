"""Deterministic parser for Schedule-of-Classes department pages (expanded via Firecrawl actions).

Produces extract/courses.jsonl — one record per course with its sections and instructors, every field copied
verbatim from the page, plus the source URL and fetch time.
"""
import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import college_config  # noqa: E402

college = Path(sys.argv[1])
cfg = college_config.load(college)
# schedule_terms maps the college's own term codes to readable names, e.g. {"20263": "Fall 2026"}
TERMS = {str(k): v for k, v in (cfg.get("schedule_terms") or {}).items()}
TERM = sys.argv[2] if len(sys.argv) > 2 else (next(iter(TERMS), None) or "20263")
TERM_NAME = TERMS.get(TERM, TERM)

COURSE_RX = re.compile(r"^([A-Z]{2,5}) (\d{3}[A-Z]?)(?: \(([^)]*)\))? - (.+)$")
UNITS_RX = re.compile(r"^(\d+(?:\.\d+)?(?: - \d+(?:\.\d+)?)? Units?)(?:, Max [\d.]+)?(?:\s*\\?\|\s*Available Seats: (\d+))?")
SECTION_ID = re.compile(r"^\d{5}[A-Z]?$")
TYPES = {"Lecture", "Lab", "Discussion", "Quiz", "Seminar", "Lecture-Lab", "Lecture-Discussion", "Studio",
         "Clinical", "Practicum", "Performance", "Workshop", "Internship", "Independent Study", "Online"}


def clean(s):
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)  # drop markdown links, keep text
    return s.replace("\\", "").strip()


def parse_page(rec):
    school_prog = re.search(r"/school/([^/]+)/program/([^/?#]+)", rec["url"])
    lines = [l.rstrip() for l in rec["markdown"].splitlines()]
    lines = [l for l in lines if l.strip()]
    courses, cur, sec = [], None, None
    in_table, pending_status = False, None
    i = 0
    while i < len(lines):
        raw = lines[i].strip()
        line = clean(raw)
        m = COURSE_RX.match(line)
        if m:
            cur = {"code": f"{m.group(1)} {m.group(2)}", "dept": m.group(1), "number": m.group(2),
                   "cross_listed": m.group(3), "title": m.group(4).strip(), "units": None, "available_seats": None,
                   "overall_status": None, "description": [], "general_education": None, "sections": [],
                   "term": TERM_NAME, "school_code": school_prog.group(1) if school_prog else None,
                   "program_code": school_prog.group(2) if school_prog else None,
                   "source_url": rec["url"], "fetched_at": rec.get("fetched_at")}
            courses.append(cur)
            sec, in_table, pending_status = None, False, None
            um = UNITS_RX.match(clean(lines[i + 1])) if i + 1 < len(lines) else None
            if um:
                cur["units"] = um.group(1)
                cur["available_seats"] = um.group(2)
                i += 1
            i += 1
            continue
        if cur is None:
            i += 1
            continue
        if line.startswith("SECTIONUNITSTYPE"):
            in_table = True
            i += 1
            continue
        if line.startswith("fiber_manual_record"):
            status = line.replace("fiber_manual_record", "").strip() or None
            if in_table:
                pending_status = status
            else:
                cur["overall_status"] = status
            i += 1
            continue
        if not in_table:
            if line.startswith("General Education"):
                cur["general_education"] = line.split(":", 1)[1].strip() if ":" in line else line
            elif not line.startswith("#") and "Need help with Registration" not in line and len(line) > 25:
                cur["description"].append(line)
            i += 1
            continue
        if SECTION_ID.match(line):
            sec = {"id": line, "status": pending_status or "OPEN"}
            pending_status = None
            cur["sections"].append(sec)
        elif sec is None or line.startswith("## "):
            if line.startswith("## "):
                in_table = False
        elif re.match(r"^\d+(\.\d+)?(-\d+(\.\d+)?)?$", line) and "units" not in sec:
            sec["units"] = line
        elif "type" not in sec and (line in TYPES or (line.split()[0] in TYPES and len(line) < 30)):
            sec["type"] = line
        elif "Sign In to View" in raw:
            after = clean(raw.split(")", 1)[1]) if ")" in raw else ""
            sec["instructors"] = [n.strip() for n in re.split(r",\s*|\s+and\s+", after) if n.strip()]
        elif "View Syllabus" in raw:
            u = re.search(r"\((https?://[^)\s]+)", raw)
            sec["syllabus_url"] = u.group(1) if u else None
        elif re.match(r"^\d+ / \d+$", line):
            sec["registered"] = line
        elif line in ("View More", "Not Available"):
            pass
        elif "type" in sec and "schedule" not in sec and "instructors" not in sec:
            sec["schedule"] = line
        i += 1
    for c in courses:
        c["description"] = " ".join(c["description"]) or None
        c["instructors"] = sorted({n for s in c["sections"] for n in s.get("instructors", [])})
    return courses


pages_dir = college / "pages"
out = college / "extract"
out.mkdir(exist_ok=True)
all_courses = []
for f in glob.glob(str(pages_dir / "*.json")):
    rec = json.load(open(f))
    if rec.get("status") != "ok" or f"/term/{TERM}/school/" not in rec["url"]:
        continue
    if "SECTIONUNITSTYPE" not in rec["markdown"]:
        continue  # not expanded — needs refetch
    all_courses.extend(parse_page(rec))
with (out / f"courses_{TERM}.jsonl").open("w") as fh:
    for c in all_courses:
        fh.write(json.dumps(c) + "\n")
ug_digits = cfg.get("undergrad_course_first_digits") or "1234"
ug = [c for c in all_courses if c["number"][:1] in ug_digits]
instructors = {n for c in all_courses for n in c["instructors"]}
ug_instructors = {n for c in ug for n in c["instructors"]}
print(f"departments={len({c['source_url'] for c in all_courses})} courses={len(all_courses)} undergrad_courses={len(ug)} "
      f"sections={sum(len(c['sections']) for c in all_courses)} instructors={len(instructors)} undergrad_instructors={len(ug_instructors)}")
