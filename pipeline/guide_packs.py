"""Build per-chapter source packs for the student guide (no API calls).

For each category: the strongest verified facts (digest ranking: multi-source, core sites, recent, undergraduate-relevant,
capped per entity and per page) plus structured data lines (club directory, course schedule), all with citable IDs.
usage: python pipeline/guide_packs.py colleges/usc "University of Southern California" USC usc.edu [QRK,INT,DIV]
writes: colleges/<slug>/guide/packs/<CODE>.txt (for writers) and <CODE>.json (id -> full fact, for the checker)
"""
import collections
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent))
from college_config import alt, load as load_config  # noqa: E402

college, SCHOOL, SHORT, ROOT = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
ONLY = [c.strip().upper() for c in sys.argv[5].split(",")] if len(sys.argv) > 5 else None  # rebuild only these chapter packs
CFG = load_config(college)
out = college / "guide" / "packs"
out.mkdir(parents=True, exist_ok=True)

src = (Path(__file__).parent / "build_docs.py").read_text()
ns = {"__name__": "guide_import"}
saved = sys.argv
sys.argv = [saved[0], str(college), SCHOOL, SHORT, ROOT]
exec(src[: src.index("# ------------------------------------------------------------------ rendering helpers")], ns)
sys.argv = saved
facts, orgs, courses, key = ns["facts"], ns["orgs"], ns["courses"], ns["key"]

dg = (Path(__file__).parent / "build_digest.py").read_text()
g = {"re": re, "collections": collections, "urlparse": urlparse, "ns": ns, "facts": facts, "orgs": orgs,
     "courses": courses, "key": key, "CFG": CFG, "alt": alt, "ROOT_DOMAIN": ROOT,
     "GOOD_HOST": re.compile(r"^(www\.)?(" + alt([ROOT] + CFG["digest_good_subdomains"]) + ")")}
exec(dg[dg.index("UG = re.compile"): dg.index("SYSTEM = f\"\"\"")], g)
candidates, extra_context = g["candidates"], g["extra_context"]

LIMIT = {"ACA": 900, "RES": 900, "NEW": 700, "INT": 700, "QRK": 450, "GEN": 500}
CHAPTERS = [("GEN", "USC at a glance", "Size, admissions, cost, campus and leadership")] + \
           [(c, n, s) for _, c, n, s in ns["CATS"]]
UGD = CFG["undergrad_course_first_digits"]


def org_line(o, n=220):
    mission = re.sub(r"\s+", " ", o["mission"] or "")[:n]
    cats = ", ".join(o["categories"][:3])
    return f"{o['name']} ({o['group_type']}; {cats}): {mission}"


def data_lines(code):
    lines = [l.strip() for l in extra_context(code).splitlines() if l.strip()] if code != "GEN" else []
    if code == "EXT" and orgs:
        by = collections.defaultdict(list)
        for o in orgs:
            by[o["group_type"]].append(o["name"])
        for t, names in sorted(by.items(), key=lambda kv: -len(kv[1])):
            lines.append(f"DIRECTORY GROUP TYPE {t} ({len(names)} groups): " + "; ".join(sorted(names)))
        tag = collections.defaultdict(list)
        for o in orgs:
            for c in o["categories"][:2]:
                tag[c].append(o)
        seen = set()
        for c, ol in sorted(tag.items(), key=lambda kv: -len(kv[1])):
            for o in sorted(ol, key=lambda o: -sum(o["scores"].values()))[:8]:
                if o["name"] not in seen:
                    seen.add(o["name"])
                    lines.append("ORG " + org_line(o))
    if code == "QRK":
        lines += ["ORG " + org_line(o) for o in sorted([o for o in orgs if o["scores"]["q"] >= 2], key=lambda o: -o["scores"]["q"])]
    if code == "SOC":
        lines += ["ORG " + org_line(o, 180) for o in orgs if o["scores"]["s"] >= 3]
        lines.append("OTHER SERVICE-MINDED GROUPS (score 2): " + "; ".join(o["name"] for o in orgs if o["scores"]["s"] == 2))
    if code == "DIV":
        lines += ["ORG " + org_line(o, 180) for o in orgs if o["scores"]["d"] >= 3]
        lines.append("OTHER CULTURAL/IDENTITY/FAITH GROUPS (score 2): " + "; ".join(o["name"] for o in orgs if o["scores"]["d"] == 2))
    if code == "RES":
        lines += ["ORG " + org_line(o, 160) for o in orgs if o["scores"].get("r", 0) >= 2]
    if code == "INN":
        lines += ["ORG " + org_line(o, 160) for o in orgs if o["scores"].get("p", 0) >= 3][:60]
    if code == "ACA" and courses:
        ug = [c for c in courses if c["number"][:1] in UGD]
        dept = collections.Counter(c["dept"] for c in ug)
        lines.append("UNDERGRADUATE COURSES BY DEPARTMENT CODE (Fall 2026): " + "; ".join(f"{d} {n}" for d, n in dept.most_common(60)))
        teach = collections.defaultdict(set)
        for c in ug:
            for s in c["sections"]:
                if re.search(r"Lecture|Seminar|Studio", s.get("type") or ""):
                    for n in s["instructors"]:
                        teach[n].add(c["code"])
        for n, cs in sorted(teach.items(), key=lambda kv: -len(kv[1]))[:40]:
            lines.append(f"INSTRUCTOR {n} teaches these undergraduate courses in Fall 2026: {', '.join(sorted(cs)[:10])}")
        ge = [c for c in ug if c.get("general_education")][:40]
        for c in ge:
            lines.append(f"GE COURSE {c['code']} {c['title']}: satisfies {c['general_education']}; {(c.get('description') or '')[:160]}")
        small = [c for c in ug if c["number"][:1] == "1" and re.search(r"seminar|freshman|first-year", (c["title"] + " " + (c.get("description") or "")), re.I)][:25]
        for c in small:
            lines.append(f"FIRST-YEAR COURSE {c['code']} {c['title']}: {(c.get('description') or '')[:180]}")
    return lines


index = []
for code, name, sub in CHAPTERS:
    if ONLY and code not in ONLY:
        continue
    fl = candidates(code, limit=LIMIT.get(code, 600))
    if code == "QRK":
        extra = [fa for fa in facts if fa["code"] == "CUL" and (fa.get("entity") or {}).get("type") == "tradition"]
        fl += extra[:150]
    idmap, txt = {}, []
    for i, fa in enumerate(fl, 1):
        fid = f"{code}-{i}"
        idmap[fid] = {"fact": fa["fact"], "evidence": fa["evidence"], "period": fa.get("period"),
                      "entity": (fa.get("entity") or {}).get("name"), "sources": fa["sources"]}
        host = (urlparse(fa["sources"][0]).hostname or "").replace("www.", "")
        date = f" | when: {fa['period']}" if fa.get("period") else ""
        more = f" +{len(fa['sources']) - 1}" if len(fa["sources"]) > 1 else ""
        txt.append(f"{fid} | {fa['fact']} | source: {host}{more}{date}")
    dl = data_lines(code)
    for j, l in enumerate(dl, 1):
        did = f"D{code}-{j}"
        idmap[did] = {"fact": l, "evidence": "", "period": None, "entity": None,
                      "sources": [orgs[0]["source_url"]] if l.startswith(("ORG", "DIRECTORY", "OTHER", "DATASET: official", "Directory")) and orgs
                      else ([courses[0]["source_url"]] if courses else [])}
        txt.append(f"{did} | {l}")
    (out / f"{code}.json").write_text(json.dumps(idmap))
    (out / f"{code}.txt").write_text(f"CHAPTER: {name} — {sub}\nSCHOOL: {SCHOOL} ({SHORT})\n"
                                     f"{len(fl)} verified facts and {len(dl)} data lines follow. Cite by ID.\n\n" + "\n".join(txt) + "\n")
    index.append((code, name, len(fl), len(dl), len("\n".join(txt))))
for r in index:
    print(f"{r[0]:4} {r[1]:36} facts {r[2]:4}  data {r[3]:4}  chars {r[4]:,}")
