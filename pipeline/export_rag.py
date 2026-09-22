"""Export one college's verified data as a clean, stack-agnostic bundle for the RAG pipeline (no API calls).

usage: python pipeline/export_rag.py colleges/usc "University of Southern California" USC usc.edu [--courses-extra colleges/usc_selfhost]
writes: colleges/<slug>/export/<slug>_rag_v1/ and <slug>_rag_v1.zip
"""
import argparse
import csv
import datetime
import gzip
import hashlib
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent))
from firecrawl_client import url_id  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("college")
ap.add_argument("school")
ap.add_argument("short")
ap.add_argument("root")
ap.add_argument("--courses-extra", default="", help="college dir with a newer courses_<term>.jsonl to merge (preferred)")
args = ap.parse_args()

college = Path(args.college)
slug = college.name
SCHEMA = "2.0"
VERSION = "v2"
out = college / "export" / f"{slug}_rag_{VERSION}"
if out.exists():
    shutil.rmtree(out)
(out / "guide").mkdir(parents=True)

src = (Path(__file__).parent / "build_docs.py").read_text()
ns = {"__name__": "export_import"}
saved = sys.argv
sys.argv = [saved[0], str(college), args.school, args.short, args.root]
exec(src[: src.index("# ------------------------------------------------------------------ rendering helpers")], ns)
sys.argv = saved
facts, orgs, CATS, source_kind = ns["facts"], ns["orgs"], ns["CATS"], ns["source_kind"]
CAT_NAME = {code: name for _, code, name, _ in CATS}
CAT_NAME["GEN"] = "General Facts"


def is_undergraduate_number(number) -> bool:
    """Undergraduate by course number, across the numbering schemes colleges actually use.

    The old rule was `number[:1] in "1234"`, which is right for a three-digit scheme where 500
    and up is graduate, and wrong nearly everywhere else. Brown writes CSCI 0111 and Swarthmore
    CPSC 021, whose first digit is 0; Cornell writes CS 4780 and Purdue CS 18000, where the
    level lives in the FIRST digit of a four- or five-digit number, not the whole of it. Under
    the old rule a zero-padded college has every course flagged graduate and its Academics
    chapter empties out, with nothing in the logs to say why.

    The convention that actually generalises: normalise to the leading level digit, where a
    leading zero means the hundreds scheme with a pad. Anything at or above 5 is graduate.
    """
    raw = "".join(ch for ch in str(number or "") if ch.isdigit())
    if not raw:
        return False
    if len(raw) >= 4 and raw[0] == "0":     # 0111 -> 111, a padded three-digit number
        raw = raw.lstrip("0") or "0"
    level = raw[0]
    if len(raw) >= 5:                        # 18000 -> level 1
        level = raw[0]
    return level in "01234"

def sid(kind, *parts):
    return f"{slug}-{kind}-" + hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:12]


page_cache = {}


def page(u):
    if u not in page_cache:
        p = college / "pages" / f"{url_id(u)}.json"
        page_cache[u] = json.loads(p.read_text()) if p.exists() else {}
    return page_cache[u]


def year_of(*vals):
    ys = [int(y) for v in vals for y in re.findall(r"\b(19[5-9]\d|20[0-3]\d)\b", str(v or ""))]
    return max(ys) if ys else None


# ---------------- facts
counts = {}
with (out / "facts.jsonl").open("w") as fh:
    seen = set()
    for f in facts:
        code = f["code"] if f["code"] in CAT_NAME else "GEN"
        fid = sid("fact", code, f["fact"])
        if fid in seen:
            continue
        seen.add(fid)
        prim = f["sources"][0]
        pg = page(prim)
        ent = f.get("entity") or {}
        rec = {
            "id": fid, "college_id": slug, "category_code": code, "category": CAT_NAME[code],
            "category_raw": f["code"] if f["code"] != code else None,
            "fact": f["fact"], "evidence_quote": f["evidence"],
            "entity_name": ent.get("name"), "entity_type": ent.get("type"),
            "relations": [r for r in (f.get("relations") or []) if isinstance(r, list) and len(r) == 3],
            "period": f.get("period"), "year": year_of(f.get("period"), f["fact"]) or year_of(pg.get("published")),
            "source_url": prim, "source_urls": f["sources"], "source_count": len(f["sources"]),
            "source_kind": source_kind(prim), "source_host": (urlparse(prim).hostname or "").replace("www.", ""),
            "source_title": pg.get("title"), "page_published": pg.get("published"),
            "retrieved_at": f.get("fetched_at") or pg.get("fetched_at"), "verification": f.get("verification"),
        }
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
counts["facts"] = len(seen)

# ---------------- organizations
org_ids = set()
with (out / "organizations.jsonl").open("w") as fh:
    for o in orgs:
        oid = sid("org", o["name"], o.get("link") or o.get("mission") or "")
        if oid in org_ids:  # the directory lists a few groups twice, identically
            continue
        org_ids.add(oid)
        sc = o.get("scores") or {}
        rec = {"id": oid, "college_id": slug, "name": o["name"], "group_type": o.get("group_type"),
               "categories": o.get("categories") or [], "mission": o.get("mission"),
               "membership_benefits": o.get("membership_benefits"), "membership_duration": o.get("membership_duration"),
               "profile_url": o.get("link"), "source_url": o.get("source_url"), "retrieved_at": o.get("fetched_at"),
               "fit_scores": {"quirky": sc.get("q"), "social_impact": sc.get("s"), "diversity_community": sc.get("d"),
                              "research": sc.get("r"), "pre_professional": sc.get("p")}}
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
counts["organizations"] = len(org_ids)
counts["organizations_duplicates_removed"] = len(orgs) - len(org_ids)

# ---------------- courses (newer official data feed preferred, older schedule pages fill gaps)
UGD = ns["CFG"]["undergrad_course_first_digits"]


def prereq_text(p):
    """USC's feed gives nested requirement groups; render as 'CSCI 356 or EE 354; CSCI 201 or EE 250'."""
    if not p:
        return None
    if isinstance(p, str):
        return p
    groups = []
    for grp in p if isinstance(p, list) else [p]:
        opts = []
        for o in (grp.get("courseOptions") or []) if isinstance(grp, dict) else []:
            name = o.get("courseSpace") or " ".join(x for x in (o.get("prefix"), o.get("number")) if x)
            ands = [a.get("courseSpace") for a in (o.get("ands") or []) if isinstance(a, dict) and a.get("courseSpace")]
            opts.append(" and ".join([name] + ands))
        if opts:
            groups.append(" or ".join(opts))
    return "; ".join(groups) or None


def load_courses(d):
    rows = {}
    for p in sorted((Path(d) / "extract").glob("courses_*.jsonl")):
        for l in p.read_text().splitlines():
            c = json.loads(l)
            k = (c.get("term"), c["code"])
            if k not in rows or len(c.get("sections") or []) > len(rows[k].get("sections") or []):
                rows[k] = c
    return rows


base = load_courses(college)
extra = load_courses(args.courses_extra) if args.courses_extra else {}
merged = dict(base)
merged.update(extra)
with (out / "courses.jsonl").open("w") as fh:
    for (term, code), c in sorted(merged.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        rec = {"id": sid("course", term, code), "college_id": slug, "term": term, "code": code, "dept": c.get("dept"),
               "number": c.get("number"), "title": c.get("title"), "is_undergraduate": is_undergraduate_number(c.get("number")),
               "units": c.get("units"), "description": c.get("description"), "general_education": c.get("general_education"),
               "prerequisites_text": prereq_text(c.get("prerequisites")), "corequisites_text": prereq_text(c.get("corequisites")),
               "prerequisites": c.get("prerequisites"), "corequisites": c.get("corequisites"),
               "restrictions": c.get("restrictions"), "recommended_prep": c.get("recommended_prep"),
               "cross_listed": c.get("cross_listed"), "instructors": c.get("instructors") or [],
               "sections": [{k: s.get(k) for k in ("id", "type", "schedule", "instructors", "registered", "status", "units")}
                            for s in c.get("sections") or []],
               "school_code": c.get("school_code"), "source_url": c.get("source_url"), "retrieved_at": c.get("fetched_at"),
               "data_origin": "official_data_feed" if (term, code) in extra else "schedule_pages"}
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
counts["courses"] = len(merged)
counts["courses_from_feed"] = sum(1 for k in merged if k in extra)

# ---------------- graph
g = college / "output" / "graph"
for name in ("nodes.csv", "edges.csv"):
    shutil.copy(g / name, out / f"graph_{name}")
with (g / "nodes.csv").open() as fh:
    counts["graph_nodes"] = sum(1 for _ in fh) - 1
with (g / "edges.csv").open() as fh:
    counts["graph_edges"] = sum(1 for _ in fh) - 1

# ---------------- guide chapters (citations resolved to source URLs as footnotes)
CITE = re.compile(r"\s?\[((?:D?[A-Z]{3}-\d+)(?:\s*,\s*D?[A-Z]{3}-\d+)*)\]")
gdir = college / "guide"
chapters = 0
for md in sorted((gdir / "chapters").glob("*.md")) if (gdir / "chapters").exists() else []:
    code = md.stem
    pack = json.loads((gdir / "packs" / f"{code}.json").read_text())
    notes, num = [], {}

    def fn(m):
        refs = []
        for i in [x.strip() for x in m.group(1).split(",")]:
            u = pack[i]["sources"][0]
            if u not in num:
                notes.append(u)
                num[u] = len(notes)
            refs.append(f"[^{num[u]}]")
        return "".join(sorted(set(refs), key=refs.index))

    body = CITE.sub(fn, md.read_text())
    body += "\n\n" + "\n".join(f"[^{i}]: {u}" for i, u in enumerate(notes, 1)) + "\n"
    front = f"---\ncollege_id: {slug}\nchapter_code: {code}\ncategory: {CAT_NAME.get(code, 'Orientation')}\n---\n\n"
    (out / "guide" / f"{code}.md").write_text(front + body)
    chapters += 1
counts["guide_chapters"] = chapters

# ---------------- cleaned page passages, ready to embed (site menus and footers removed)
ex = (Path(__file__).parent / "extract_facts.py").read_text()
cl = {"re": re, "collections": __import__("collections"), "urlparse": urlparse, "unicodedata": __import__("unicodedata")}
exec(ex[ex.index("LINK = re.compile"): ex.index("host_lines = collections.defaultdict")], cl)
simplify = cl["simplify"]
SKIP_PAGE = re.compile(alt_pat) if (alt_pat := "|".join(ns["CFG"]["structured_url_patterns"]) or "(?!x)x") else None
raw_pages = []
for pth in sorted((college / "pages").glob("*.json")):
    r = json.loads(pth.read_text())
    if r.get("status") == "ok" and r.get("markdown") and not SKIP_PAGE.search(r["url"]):
        raw_pages.append(r)
host_lines, host_pages = __import__("collections").defaultdict(__import__("collections").Counter), __import__("collections").Counter()
for r in raw_pages:
    h = urlparse(r["url"]).hostname
    host_pages[h] += 1
    for l in {simplify(x) for x in r["markdown"].splitlines()}:
        if l:
            host_lines[h][l] += 1


def clean_lines(r):
    h = urlparse(r["url"]).hostname
    n = host_pages[h]
    out_l, prev = [], None
    for raw in r["markdown"].splitlines():
        l = simplify(raw)
        if not l or l == prev:
            continue
        if n >= 5 and host_lines[h][l] / n >= 0.4 and len(l) < 300:
            continue
        if re.fullmatch(r"[-*•|#>\s\\]*", l) or (re.search(r"cookie|consent preferences|privacy notice", l, re.I) and len(l) < 200):
            continue
        out_l.append(l)
        prev = l
    return out_l


MAXC, OVERLAP = 2400, 350  # ~600 tokens per passage, ~15% overlap


def tail_overlap(buf, want=OVERLAP):
    """The last `want` characters of buf, moved forward to a clean word start.

    A raw slice cut half of every passage mid-word -- "lifornia President Michael V. Drake
    said" for California -- because the carry-over is measured in characters. Half of the
    26,960 passages in the first index began that way. The reports survived it (the writer
    reads past the fragment and the verifier catches anything ungrounded) but the passage's
    first token is wasted and its embedding is computed on a non-word, so it is worth the
    handful of characters this gives up.
    """
    if len(buf) <= want:
        return buf
    tail = buf[-want:]
    line = tail.find("\n")
    if 0 <= line <= want // 3:      # a line start is the cleanest cut when one is near
        return tail[line + 1:]
    m = re.search(r"\s", tail)
    return tail[m.end():] if m else tail
nchunks = 0
with (out / "page_chunks.jsonl").open("w") as fh:
    for r in raw_pages:
        lines = clean_lines(r)
        if not lines:
            continue
        buf, chunks = "", []
        for l in lines:
            if len(buf) + len(l) + 1 > MAXC and buf:
                chunks.append(buf)
                buf = tail_overlap(buf)
            buf += l + "\n"
        if buf.strip():
            chunks.append(buf)
        for i, ch in enumerate(chunks):
            title = r.get("title") or ""
            fh.write(json.dumps({"id": sid("chunk", r["url"], i), "college_id": slug, "page_id": sid("page", r["url"]),
                                 "url": r["url"], "title": title, "chunk_index": i, "chunk_count": len(chunks),
                                 "source_kind": source_kind(r["url"]), "retrieved_at": r.get("fetched_at"),
                                 "text": (f"{title}\n\n" if title else "") + ch.strip()}, ensure_ascii=False) + "\n")
            nchunks += 1
counts["page_chunks"] = nchunks

# ---------------- full page texts
with gzip.open(out / "pages.jsonl.gz", "wt", encoding="utf-8") as fh:
    n = 0
    for p in sorted((college / "pages").glob("*.json")):
        r = json.loads(p.read_text())
        if r.get("status") != "ok" or not r.get("markdown"):
            continue
        fh.write(json.dumps({"id": sid("page", r["url"]), "college_id": slug, "url": r["url"], "title": r.get("title"),
                             "source_kind": source_kind(r["url"]), "published": r.get("published"),
                             "retrieved_at": r.get("fetched_at"), "content_type": r.get("content_type"),
                             "markdown": r["markdown"]}, ensure_ascii=False) + "\n")
        n += 1
counts["pages"] = n

# ---------------- manifest + README
created = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
files = {}
for p in sorted(out.rglob("*")):
    if p.is_file():
        files[str(p.relative_to(out))] = {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
(out / "manifest.json").write_text(json.dumps({"college_id": slug, "college": args.school, "schema_version": SCHEMA,
                                               "created_at": created, "snapshot": "2026-09-15/16", "counts": counts,
                                               "files": files}, indent=1))
readme = (Path(__file__).parent / "export_readme_template.md").read_text()
for k, v in {"SLUG": slug, "SCHOOL": args.school, "CREATED": created, **{f"N_{k.upper()}": f"{v:,}" for k, v in counts.items()}}.items():
    readme = readme.replace("{{" + k + "}}", str(v))
(out / "README.md").write_text(readme)

zpath = college / "export" / f"{slug}_rag_{VERSION}.zip"
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(out.rglob("*")):
        if p.is_file():
            z.write(p, f"{slug}_rag_{VERSION}/{p.relative_to(out)}")
print(json.dumps(counts), f"| zip {zpath} {zpath.stat().st_size / 1e6:.1f} MB")
