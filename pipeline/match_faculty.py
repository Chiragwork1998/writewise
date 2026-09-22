"""Match Schedule-of-Classes instructors to faculty profile pages and queue the profiles to scrape.

Priority = people teaching undergraduate (100-499) lecture/seminar/studio sections, weighted by how many such
courses they teach. One primary profile per person, preferring the site of the school that offers the course.

usage: python pipeline/match_faculty.py colleges/usc 20263 queues/faculty.jsonl MAX
"""
import collections
import glob
import json
import re
import sys
import unicodedata
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

college, term, out_path, max_n = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), int(sys.argv[4])
from college_config import load as load_config, root_or_default  # noqa: E402
CFG = load_config(college)
ROOT = root_or_default(CFG)
NEWSROOMS = tuple(CFG["news_hosts_capped"])

courses = [json.loads(l) for l in (college / "extract" / f"courses_{term}.jsonl").read_text().splitlines()]
TEACHING_TYPES = re.compile(r"Lecture|Seminar|Studio|Performance|Workshop|Practicum|Clinical|Online", re.I)

people = collections.defaultdict(lambda: {"ug_courses": set(), "all_courses": set(), "schools": collections.Counter()})
for c in courses:
    for s in c["sections"]:
        for n in s.get("instructors", []):
            p = people[n]
            p["all_courses"].add(c["code"])
            p["schools"][c["school_code"]] += 1
            if c["number"][:1] in "1234" and TEACHING_TYPES.search(s.get("type") or "Lecture"):
                p["ug_courses"].add(c["code"])


def ascii_tokens(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return [t for t in re.split(r"[^a-z]+", s) if t]


def key_of(name):
    t = [x for x in ascii_tokens(name) if len(x) > 1 and x not in ("jr", "sr", "ii", "iii", "dr", "phd", "md")]
    return (t[0], t[-1]) if len(t) >= 2 else None


# ---- profile index: URL -> candidate name keys (from path pattern or map title)
titles = {}
for f in glob.glob(str(college / "discovery" / "maps" / "*.json")):
    for l in json.load(open(f)).get("links", []):
        if l.get("title"):
            titles[l["url"].split("#")[0]] = l["title"]
prof_urls = json.load(open(college / "discovery" / "profile_urls.json"))

index = collections.defaultdict(set)
for u in prof_urls:
    p = urlparse(u)
    path = unquote(p.path)
    keys = set()
    m = re.search(r"/directory/faculty/([^/]+)/([^/]+)", path)
    if m:
        keys.add(key_of(f"{m.group(2)} {m.group(1)}"))
    q = parse_qs(p.query)
    if "lname" in q and "fname" in q:
        keys.add(key_of(f"{q['fname'][0]} {q['lname'][0]}"))
    m = re.search(r"/(?:profile|personnel|people|faculty|bio|our-people)/([a-z0-9-]+)/?$", path, re.I)
    if m and not re.search(r"\d{3,}", m.group(1)):
        slug = re.sub(r"-\d+$", "", m.group(1))
        k = key_of(slug.replace("-", " "))
        if k:
            keys.add(k)
    t = titles.get(u) or titles.get(u.rstrip("/")) or titles.get(u + "/")
    if t:
        k = key_of(re.split(r"\s[|\-–—]\s", t)[0])
        if k:
            keys.add(k)
    for k in keys:
        if k:
            index[k].add(u)

# second index: any discovered page whose title is exactly a person's name (e.g. "William Kanengiser - USC Thornton")
title_index = collections.defaultdict(set)
for u, t in titles.items():
    h = urlparse(u).hostname or ""
    if not h.endswith(ROOT) or re.search(r"/(news|stories|story|blog|events?|spotlights?)/", u):
        continue
    first = re.split(r"\s[|\-–—:]\s|,\s", t)[0].strip()
    toks = ascii_tokens(first)
    if 2 <= len(toks) <= 4 and not re.search(r"\d", first):
        k = key_of(first)
        if k:
            title_index[k].add(u)

SCHOOL_HOST = CFG["school_hosts"]  # schedule school code -> school website host

fetched = {json.loads(l)["url"] for l in (college / "logs" / "fetch_log.jsonl").read_text().splitlines()
           if json.loads(l)["status"] == "ok"}

ranked = sorted(people.items(), key=lambda kv: (-len(kv[1]["ug_courses"]), -len(kv[1]["all_courses"])))
chosen, unmatched, seen_urls = [], [], set()
for name, p in ranked:
    if not p["ug_courses"]:
        continue
    k = key_of(name)
    cands = sorted(index.get(k, set()) or title_index.get(k, set())) if k else []
    if not cands:
        unmatched.append(name)
        continue
    home = SCHOOL_HOST.get(p["schools"].most_common(1)[0][0], "")

    def rank(u):
        h = urlparse(u).hostname or ""
        return (0 if home and h.endswith(home) else 2 if NEWSROOMS and h.endswith(NEWSROOMS) else 1, len(u))

    best = sorted(cands, key=rank)[0]
    if best in seen_urls:
        continue
    seen_urls.add(best)
    if best in fetched:
        continue
    chosen.append({"url": best, "bucket": "faculty_profile", "priority": 50 + 5 * len(p["ug_courses"]),
                   "instructor": name, "ug_courses": sorted(p["ug_courses"])[:12]})
    if len(chosen) >= max_n:
        break

with out_path.open("w") as f:
    for x in chosen:
        f.write(json.dumps(x) + "\n")
ug_people = sum(1 for p in people.values() if p["ug_courses"])
print(f"instructors={len(people)} teaching_undergrad={ug_people} matched_to_profile={len(chosen) + len(seen_urls & fetched)} "
      f"queued={len(chosen)} unmatched={len(unmatched)}")
print("hosts:", collections.Counter(urlparse(x['url']).hostname for x in chosen).most_common(15))
print("sample unmatched:", unmatched[:25])
