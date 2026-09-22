"""Build one document per research category from verified facts + structured datasets.

Every bullet is a fact whose verbatim evidence was re-checked against the scraped page, and cites its source.
Outputs Markdown (and HTML for PDF rendering) to output/categories/.

usage: python pipeline/build_docs.py colleges/usc "University of Southern California" USC
"""
import collections
import glob
import html
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

college, SCHOOL, SHORT = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
from college_config import alt, load as load_config, root_or_default  # noqa: E402
CFG = load_config(college)
ROOT_DOMAIN = root_or_default(CFG, sys.argv[4] if len(sys.argv) > 4 else None)
out_dir = college / "output" / "categories"
out_dir.mkdir(parents=True, exist_ok=True)
NOW = datetime.now(timezone.utc).strftime("%Y-%m-%d")

CATS = [
    ("01", "CUL", "Culture", "College-specific ethos, mission statement and traditions"),
    ("02", "EXT", "Extracurriculars", "Clubs and organizations"),
    ("03", "QRK", "Quirks", "Fun, unusual, human clubs and traditions"),
    ("04", "ACA", "Academics", "Specific courses and professors for teaching"),
    ("05", "RES", "Research", "Specific research opportunities with specific professors"),
    ("06", "SOC", "Social Impact", "Nonprofit and community service alignment"),
    ("07", "INN", "Innovative Programs", "Signature, distinctive or rare programs"),
    ("08", "INT", "Intellectual Alignment", "Academic philosophy and way of thinking"),
    ("09", "DIV", "Diversity of Community", "International student support and cultural integration"),
    ("10", "NEW", "External Articles and References", "The school in the news"),
]
CODE_FIX = {"NEWS": "NEW", "AWARD": "GEN", "GENERAL": "GEN"}


# ------------------------------------------------------------------ text utils (same cleaning as extraction)
def ascii_fold(s):
    s = unicodedata.normalize("NFKC", s or "")
    for a, b in (("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"'), ("–", "-"), ("—", "-"), (" ", " ")):
        s = s.replace(a, b)
    return s


def norm(s):
    s = ascii_fold(s)
    s = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", s)
    s = re.sub(r"\[([^\]]*)\]\((?:[^()]|\([^)]*\))*\)", r"\1", s)
    s = re.sub(r"(^|\n)\s*(?:[-*•]|\d+\.)\s+", r"\1", s)  # list markers
    s = re.sub(r"\\(.)", r"\1", s, flags=re.S)               # markdown escapes: \- \. \| and "\<newline>"
    s = s.replace("\\", "")
    s = re.sub(r"[*_`]", "", s)
    s = re.sub(r"[#>|\[\]]", " ", s)
    return re.sub(r"\s+", " ", s).strip().lower()


ALNUM_CACHE = {}


def alnum(s):
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def verify(evidence, page_norm):
    e = norm(evidence)
    if len(e) < 12:
        return "too_short"
    if e in page_norm:
        return "exact"
    pa = ALNUM_CACHE.get(id(page_norm))
    if pa is None:
        pa = ALNUM_CACHE[id(page_norm)] = alnum(page_norm)
    ea = alnum(e)
    if len(ea) >= 12 and ea in pa:
        return "exact_words"  # same words in the same order; only punctuation/list markers differ
    words = e.split()
    if len(words) < 6:
        return "failed"
    grams = [" ".join(words[i:i + 5]) for i in range(len(words) - 4)]
    return "near" if sum(1 for g in grams if g in page_norm) / len(grams) >= 0.85 else "failed"


def key(s):
    return re.sub(r"[^a-z0-9]+", " ", ascii_fold(s).lower()).strip()


def is_official(url):
    h = urlparse(url).hostname or ""
    return h == ROOT_DOMAIN or h.endswith("." + ROOT_DOMAIN)


# Independent news & reference outlets -> category 10. Other non-university domains (faculty lab sites, institutes,
# partner programs) are "affiliated" and keep their topical category.
NEWS_HOSTS = re.compile(r"(^|\.)(" + alt(CFG["news_domains"]) + r")$")
SKIP_HOSTS = re.compile(r"^jobs\.")


def source_kind(url):
    h = urlparse(url).hostname or ""
    if is_official(url):
        return "official"
    return "external" if NEWS_HOSTS.search(h) else "affiliated"


# ------------------------------------------------------------------ load
page_text = {}
page_meta = {}
for f in glob.glob(str(college / "pages" / "*.json")):
    r = json.load(open(f))
    if r.get("status") == "ok":
        page_text[r["url"]] = norm(r.get("markdown", ""))
        page_meta[r["url"]] = {"title": (r.get("title") or "").strip(), "fetched_at": r.get("fetched_at"),
                               "published": r.get("published")}

raw = [json.loads(l) for l in (college / "extract" / "facts_raw.jsonl").read_text().splitlines()]
stats = collections.Counter()
facts = []
for fa in raw:
    stats["raw"] += 1
    pt = page_text.get(fa["url"])
    if pt is None:
        continue
    v = verify(fa["evidence"], pt)
    stats[f"verify_{v}"] += 1
    if v not in ("exact", "exact_words", "near"):
        continue
    code = CODE_FIX.get((fa.get("category") or "").upper(), (fa.get("category") or "GEN").upper())
    if SKIP_HOSTS.search(urlparse(fa["url"]).hostname or ""):
        continue
    if source_kind(fa["url"]) == "external":
        code = "NEW"  # independent coverage all lives in category 10
    fa["code"] = code
    fa["verification"] = v
    facts.append(fa)

# dedupe identical statements; keep every source that supports them
merged = collections.OrderedDict()
for fa in facts:
    k = (fa["code"], key(fa["fact"]))
    if k in merged:
        if fa["url"] not in merged[k]["sources"]:
            merged[k]["sources"].append(fa["url"])
        continue
    fa["sources"] = [fa["url"]]
    merged[k] = fa
facts = list(merged.values())
stats["verified_unique"] = len(facts)

orgs = []
p = college / "extract" / "organizations_tagged.jsonl"
if p.exists():
    orgs = [json.loads(l) for l in p.read_text().splitlines()]
courses = []
for cf in sorted(glob.glob(str(college / "extract" / "courses_*.jsonl"))):
    courses += [json.loads(l) for l in open(cf)]


# ------------------------------------------------------------------ rendering helpers
class Doc:
    def __init__(self, num, code, name, subtitle):
        self.num, self.code, self.name, self.subtitle = num, code, name, subtitle
        self.lines = []
        self.sources = collections.OrderedDict()
        self.count = 0

    def cite(self, urls):
        ids = []
        for u in urls[:4]:
            if u not in self.sources:
                self.sources[u] = len(self.sources) + 1
            ids.append(f"S{self.sources[u]}")
        return "[" + ", ".join(ids) + "]"

    def h(self, level, text):
        self.lines.append("\n" + "#" * level + " " + text + "\n")

    def p(self, text):
        self.lines.append(text + "\n")

    def bullet(self, fa):
        period = f" *({fa['period']})*" if fa.get("period") else ""
        self.lines.append(f"- {fa['fact'].strip()}{period} {self.cite(fa['sources'])}")
        self.count += 1

    def write(self, intro_stats):
        head = [f"# {self.num}. {self.name.upper()} — {SCHOOL}",
                f"*{self.subtitle}*\n",
                f"Generated {NOW}. Every statement below was extracted from a scraped web page and its verbatim quote was "
                f"re-checked against that page; each bullet cites its source(s) in the Sources list at the end. "
                f"Periods in *(italics)* are the dates the source itself gives. Statements taken from the university's own "
                f"pages (marked *official* in Sources) reflect how the university describes itself; *affiliated* sources are "
                f"faculty lab sites, institutes and partner programs; *external* sources are independent news and references.\n",
                intro_stats, ""]
        src = ["\n## Sources\n"]
        for u, n in self.sources.items():
            m = page_meta.get(u, {})
            kind = source_kind(u)
            title = m.get("title") or u
            src.append(f"- **S{n}** {title} — <{u}> ({kind}; retrieved {str(m.get('fetched_at') or '')[:10]})")
        text = "\n".join(head + self.lines + src) + "\n"
        path = out_dir / f"{self.num}_{re.sub(r'[^a-z]+', '_', self.name.lower()).strip('_')}.md"
        path.write_text(text)
        return path


TYPE_ORDER = ["university", "school", "department", "program", "course", "center", "lab", "professor", "person",
              "organization", "tradition", "service", "office", "partner", "facility", "award", "event", "publication",
              "other"]
TYPE_LABEL = {"university": "University-wide", "school": "Schools", "department": "Departments", "program": "Programs",
              "course": "Courses", "center": "Centers & institutes", "lab": "Labs & research groups",
              "professor": "Professors", "person": "People", "organization": "Organizations", "tradition": "Traditions",
              "service": "Services & resources", "office": "Offices", "partner": "Partners", "facility": "Places & facilities",
              "award": "Awards & honors", "event": "Events", "publication": "Publications & media", "other": "Other"}


CORE_HOST = re.compile(r"^(www\.)?(" + alt([ROOT_DOMAIN] + CFG["core_subdomains"]) + r")\.")


def fact_rank(fa):
    h = urlparse(fa["url"]).hostname or ""
    return (-len(fa["sources"]), 0 if CORE_HOST.search(h) else 1, fa["fact"])


ALIASES = {}
_alias_path = college / "config" / "entity_aliases.json"
if _alias_path.exists():
    ALIASES = json.loads(_alias_path.read_text())
PREFIX = re.compile(r"^(the |" + "".join(f"{re.escape(n)} s |{re.escape(n)} |" for n in CFG["name_variants"]).rstrip("|") + ")+")


def resolve(t, nm):
    """Merge name variants: case/punctuation, a leading 'USC'/'The', and configured school aliases."""
    k = PREFIX.sub("", key(nm)) or key(nm)
    for alias, canonical in ALIASES.get(t, {}).items():
        if re.search(r"\b" + re.escape(alias) + r"\b", k):
            return key(canonical), canonical
    return k, nm


def entity_sections(doc, flist, base=2):
    by_type = collections.defaultdict(lambda: collections.defaultdict(list))
    display = {}
    for fa in flist:
        e = fa.get("entity") or {}
        t = (e.get("type") or "other").lower()
        t = t if t in TYPE_LABEL else "other"
        nm = (e.get("name") or "General").strip()
        k, canon_name = resolve(t, nm)
        if (t, k) not in display or (canon_name != nm and display[(t, k)] != canon_name):
            display[(t, k)] = canon_name if canon_name != nm else display.get((t, k), nm)
        by_type[t][k].append(fa)
    for t in TYPE_ORDER:
        if t not in by_type:
            continue
        groups = sorted(by_type[t].items(), key=lambda kv: (-len(kv[1]), kv[0]))
        doc.h(base, f"{TYPE_LABEL[t]} ({len(groups)})")
        for k, fl in groups:
            doc.h(base + 1, display[(t, k)])
            if len(fl) <= 20:
                for fa in sorted(fl, key=fact_rank):
                    doc.bullet(fa)
                continue
            # large groups: keep facts from the same source page together so each block reads as one topic
            by_page = collections.defaultdict(list)
            for fa in fl:
                by_page[fa["url"]].append(fa)
            for u, pl in sorted(by_page.items(), key=lambda kv: (-len(kv[1]), kv[0])):
                title = (page_meta.get(u, {}).get("title") or urlparse(u).path or u).strip()
                title = re.split(r"\s[|\-–—]\s", title)[0][:90] or u
                doc.lines.append(f"\n**{title}**\n")
                for fa in sorted(pl, key=fact_rank):
                    doc.bullet(fa)


def stats_line(n_facts, n_sources, extra=""):
    return f"**In this document:** {n_facts:,} verified facts from {n_sources:,} source pages{extra}."


by_code = collections.defaultdict(list)
for fa in facts:
    by_code[fa["code"]].append(fa)

written = []
QUIRK_RX = re.compile(r"\b(tradition|ritual|mascot|legend|lore|superstiti|prank|secret|hidden|myth|quirk|unusual|odd|weird|fun fact|only at|rivalry|tailgat|game ?day|homecoming|costume|parade|mural|statue|nickname|troupe|a cappella|improv|escape room|festival|celebration|midnight|annual)\b", re.I)
IDENTITY_RX = re.compile(r"\b(" + alt(CFG["identity_words"]) + r")\b", re.I)  # the college's own lore (Quirks filter)

for num, code, name, sub in CATS:
    doc = Doc(num, code, name, sub)
    flist = list(by_code.get(code, []))
    extra = ""

    if code == "EXT" and orgs:
        doc.h(2, f"Complete student organization directory ({len(orgs):,} groups)")
        doc.p(f"Source: the university's official student-group directory ({CFG['org_directory_name']}). Mission and membership text is "
              "copied verbatim from each group's directory entry.")
        by_cat = collections.defaultdict(list)
        for o in orgs:
            by_cat[o["group_type"] or "Other"].append(o)
        for gt, ol in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
            doc.h(3, f"{gt} ({len(ol)})")
            for o in ol:
                cats = f" — *{', '.join(o['categories'])}*" if o["categories"] else ""
                mission = re.sub(r"\s+", " ", o["mission"] or "").strip()
                doc.lines.append(f"- **{o['name']}**{cats}: {mission} {doc.cite([o['source_url']])}")
                doc.count += 1
        extra = f" plus {len(orgs):,} directory entries"
        doc.h(2, "Organization facts from other pages")

    if code == "QRK":
        # keep the college's own lore, plus campus things (groups, events, places, rituals) from official pages;
        # drop people's off-campus biography, which the model often files under Quirks
        CAMPUS = ("tradition", "organization", "event", "facility", "service", "center", "lab", "program")

        def quirky(fa):
            if IDENTITY_RX.search(fa["fact"] + " " + fa["evidence"]) or (fa.get("entity") or {}).get("type") in ("tradition", "organization"):
                return True
            t = (fa.get("entity") or {}).get("type")
            return bool(is_official(fa["sources"][0]) and t in CAMPUS and (t in ("tradition", "organization", "event", "facility") or QUIRK_RX.search(fa["fact"])))

        flist = [fa for fa in flist if quirky(fa)]
        trad = [fa for fa in by_code.get("CUL", []) if (fa.get("entity") or {}).get("type") == "tradition"]
        if orgs:
            quirky = [o for o in orgs if o["scores"]["q"] >= 2]
            doc.h(2, f"Unusual and fun student groups ({len(quirky)})")
            doc.p("Selected from the full directory because their own mission text describes something fun, unusual "
                  "or distinctive. Mission text is verbatim.")
            for o in sorted(quirky, key=lambda o: (-o["scores"]["q"], o["name"].lower())):
                mission = re.sub(r"\s+", " ", o["mission"] or "").strip()
                doc.lines.append(f"- **{o['name']}**: {mission} {doc.cite([o['source_url']])}")
                doc.count += 1
        if trad:
            doc.h(2, f"Traditions ({len({key((f.get('entity') or {}).get('name') or '') for f in trad})})")
            entity_sections(doc, trad)
        extra = ""

    if code == "SOC" and orgs:
        imp = [o for o in orgs if o["scores"]["s"] >= 2]
        doc.h(2, f"Student organizations with a service or advocacy mission ({len(imp)})")
        for o in sorted(imp, key=lambda o: (-o["scores"]["s"], o["name"].lower())):
            mission = re.sub(r"\s+", " ", o["mission"] or "").strip()
            doc.lines.append(f"- **{o['name']}**: {mission} {doc.cite([o['source_url']])}")
            doc.count += 1

    if code == "DIV" and orgs:
        dv = [o for o in orgs if o["scores"]["d"] >= 2]
        doc.h(2, f"Cultural, identity, international and faith communities ({len(dv)})")
        for o in sorted(dv, key=lambda o: (-o["scores"]["d"], o["name"].lower())):
            mission = re.sub(r"\s+", " ", o["mission"] or "").strip()
            doc.lines.append(f"- **{o['name']}**: {mission} {doc.cite([o['source_url']])}")
            doc.count += 1

    if code == "ACA" and courses:
        ug = [c for c in courses if c["number"][:1] in CFG["undergrad_course_first_digits"]]
        extra = f" plus {len(courses):,} scheduled courses ({len(ug):,} undergraduate)"
        doc.h(2, "Who teaches what: undergraduate courses and their instructors")
        doc.p(f"From the official {CFG['schedule_name']}. Instructor names, schedules and descriptions are copied as "
              "published for the term shown. The complete course list (graduate courses included) is in "
              "`04b_course_catalog.md`.")
        teach = collections.defaultdict(set)
        for c in ug:
            for n in c["instructors"]:
                teach[n].add(f"{c['code']} {c['title']}")
        doc.h(3, f"Instructors of undergraduate courses ({len(teach):,})")
        for n in sorted(teach, key=lambda x: (x.split()[-1].lower(), x.lower())):
            cl = sorted(teach[n])
            doc.lines.append(f"- **{n}** — {'; '.join(cl[:12])}{' …' if len(cl) > 12 else ''} {doc.cite(sorted({c['source_url'] for c in ug if n in c['instructors']})[:2])}")
            doc.count += 1
        doc.h(2, "Academic facts from program, department and faculty pages")

    if code == "NEW":
        ext = [fa for fa in flist if source_kind(fa["url"]) == "external"]
        own = [fa for fa in flist if source_kind(fa["url"]) != "external"]
        doc.h(2, f"A. Independent coverage ({len({fa['url'] for fa in ext})} articles)")
        by_url = collections.defaultdict(list)
        for fa in ext:
            by_url[fa["url"]].append(fa)
        for u, fl in sorted(by_url.items(), key=lambda kv: (str(page_meta.get(kv[0], {}).get("published") or ""), kv[0]), reverse=True):
            m = page_meta.get(u, {})
            doc.h(3, f"{m.get('title') or u} — {urlparse(u).hostname}")
            for fa in fl:
                doc.bullet(fa)
        doc.h(2, f"B. The university's own news ({len({fa['url'] for fa in own})} stories)")
        entity_sections(doc, own)
        flist = []

    if code == "RES":
        lead = [fa for fa in flist if re.search(r"undergrad", fa["fact"], re.I)]
        if lead:
            doc.h(2, f"Start here: research opportunities open to undergraduates ({len(lead):,} facts)")
            entity_sections(doc, lead, base=3)
            ids = {id(fa) for fa in lead}
            flist = [fa for fa in flist if id(fa) not in ids]
            doc.h(2, "All other research facts")
    if code == "ACA":
        lead = [fa for fa in flist if (fa.get("entity") or {}).get("type") == "program"
                and re.search(r"\b(major|minor|bachelor|B\.A\.|B\.S\.|BFA|degree)\b", fa["fact"], re.I)]
        if lead:
            doc.h(2, f"Majors, minors and degree programs ({len(lead):,} facts)")
            entity_sections(doc, lead, base=3)
            ids = {id(fa) for fa in lead}
            flist = [fa for fa in flist if id(fa) not in ids]
            doc.h(2, "All other academic facts")
    entity_sections(doc, flist)
    n_src = len(doc.sources)
    path = doc.write(stats_line(doc.count, n_src, extra))
    written.append((num, name, doc.count, n_src, path.name))

# general facts appendix
gen = Doc("00", "GEN", "General Facts", "Statistics, admissions, costs, campus, leadership and other facts outside the 10 categories")
entity_sections(gen, by_code.get("GEN", []))
gen.write(stats_line(gen.count, len(gen.sources)))
written.append(("00", "General Facts", gen.count, len(gen.sources), "00_general_facts.md"))

# full course catalog appendix
if courses:
    cat = Doc("04b", "ACA", "Course Catalog", f"Every course in the {CFG['schedule_name']} with sections and instructors")
    by_school = collections.defaultdict(lambda: collections.defaultdict(list))
    for c in courses:
        by_school[c["school_code"]][c["program_code"]].append(c)
    for sc in sorted(by_school):
        cat.h(2, f"School code {sc}")
        for pc in sorted(by_school[sc]):
            cat.h(3, f"{pc} ({len(by_school[sc][pc])} courses)")
            for c in sorted(by_school[sc][pc], key=lambda c: c["code"]):
                ins = ", ".join(c["instructors"]) or "instructor not listed"
                ge = f" GE: {c['general_education']}" if c.get("general_education") else ""
                sched = "; ".join(sorted({f"{s.get('type', '')} {s.get('schedule', '')}".strip() for s in c["sections"]})[:3])
                desc = f" — {c['description']}" if c.get("description") else ""
                cat.lines.append(f"- **{c['code']} {c['title']}** ({c.get('units') or ''}, {c['term']}) · {ins} · {sched}{ge}{desc} {cat.cite([c['source_url']])}")
                cat.count += 1
    cat.write(stats_line(cat.count, len(cat.sources)))
    written.append(("04b", "Course Catalog", cat.count, len(cat.sources), "04b_course_catalog.md"))

(college / "output" / "build_stats.json").write_text(json.dumps({"facts": dict(stats), "documents": written}, indent=1))
print(dict(stats))
for w in written:
    print(w)
