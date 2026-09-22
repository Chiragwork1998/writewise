"""Digestible summary across the 10 categories, built ONLY from verified facts.

For each category: pick the strongest candidate facts, ask DeepSeek to write concise highlight bullets that cite
fact IDs, then machine-check every bullet — any name or number in a bullet must appear in the facts it cites,
otherwise the bullet is dropped. Output: output/digest.md (+ PDF via render_pdf.py).

usage: python pipeline/build_digest.py colleges/usc "University of Southern California" USC
"""
import collections
import glob
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from firecrawl_client import load_env  # noqa: E402

college, SCHOOL, SHORT = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
from college_config import alt, load as load_config, root_or_default  # noqa: E402
CFG = load_config(college)
ROOT_DOMAIN = root_or_default(CFG, sys.argv[4] if len(sys.argv) > 4 else None)
MODEL = os.environ.get("DIGEST_MODEL", "deepseek-v4-pro")
load_env(Path(__file__).resolve().parent.parent / ".env")
KEY = os.environ["DEEPSEEK_API_KEY"]

# reuse the verified, deduplicated facts exactly as the category documents use them
src = (Path(__file__).parent / "build_docs.py").read_text()
ns = {"__name__": "digest_import"}
sys_argv = sys.argv
sys.argv = [sys.argv[0], str(college), SCHOOL, SHORT, ROOT_DOMAIN]
exec(src[: src.index("# ------------------------------------------------------------------ rendering helpers")], ns)
sys.argv = sys_argv
facts, orgs, courses, page_meta, is_official, key = (ns["facts"], ns["orgs"], ns["courses"], ns["page_meta"],
                                                      ns["is_official"], ns["key"])
CATS = ns["CATS"]

UG = re.compile(r"undergrad|first-year|freshm|student|major|minor|club|open to", re.I)
GOOD_HOST = re.compile(r"^(www\.)?(" + alt([ROOT_DOMAIN] + CFG["digest_good_subdomains"]) + ")")


OLD_YEAR = re.compile(r"\b(19[5-9]\d|20[01]\d|202[01])\b")
NEW_YEAR = re.compile(r"\b(2024|2025|2026)\b")
LOW_VALUE = re.compile(r"suggested lectures|Sections? \d+\.\d|textbook|published a post|posted as a video|welcomed students to|"
                       r"will be held on|takes place from|room \d|office hours", re.I)
RES_GOOD = re.compile(r"undergrad|open to|welcome|apply|fellowship|stipend|funding|\$\d|mentor|lab|research assistant", re.I)
ACA_GOOD = re.compile(r"\b(major|minor|degree|B\.A\.|B\.S\.|curriculum|honors|general education|course)\b", re.I)


def fscore(fa, group_size, code=None):
    s = min(group_size, 25) * 0.4
    h = urlparse(fa["url"]).hostname or ""
    kind = ns["source_kind"](fa["url"])
    if code == "NEW":
        s += 5 if kind == "external" else -2
    elif GOOD_HOST.search(h):
        s += 3
    text = fa["fact"] + " " + str(fa.get("period") or "")
    if LOW_VALUE.search(text):
        s -= 6
    if code not in ("CUL", "QRK") and OLD_YEAR.search(text) and not NEW_YEAR.search(text):
        s -= 4
    if NEW_YEAR.search(text):
        s += 1.5
    if code == "RES" and RES_GOOD.search(fa["fact"]):
        s += 3
    if code == "ACA" and ACA_GOOD.search(fa["fact"]):
        s += 2
    if UG.search(fa["fact"]):
        s += 2
    if re.search(r"\d", fa["fact"]):
        s += 1
    if (fa.get("entity") or {}).get("type") in ("tradition", "program", "organization", "center", "lab", "course"):
        s += 1.5
    s += min(len(fa.get("sources", [])), 4) * 0.8
    if fa.get("period") and re.search(r"202[4-6]", str(fa["period"])):
        s += 1
    return s


def candidates(code, limit=420, per_entity=6):
    fl = [fa for fa in facts if fa["code"] == code]
    groups = collections.Counter(key((fa.get("entity") or {}).get("name") or "") for fa in fl)
    fl.sort(key=lambda fa: -fscore(fa, groups[key((fa.get("entity") or {}).get("name") or "")], code))
    used, out, per_url = collections.Counter(), [], collections.Counter()
    for fa in fl:
        k = key((fa.get("entity") or {}).get("name") or "")
        if used[k] >= per_entity or per_url[fa["url"]] >= 8:
            continue
        used[k] += 1
        per_url[fa["url"]] += 1
        out.append(fa)
        if len(out) >= limit:
            break
    return out


def extra_context(code):
    """Structured datasets that give a category real numbers."""
    lines = []
    if code == "EXT" and orgs:
        c = collections.Counter(o["group_type"] for o in orgs)
        lines.append(f"DATASET: official student-group directory lists {len(orgs)} groups; by type: " +
                     "; ".join(f"{k}: {v}" for k, v in c.most_common()))
        cc = collections.Counter(x for o in orgs for x in o["categories"])
        lines.append("Directory category tags (a group can have several): " + "; ".join(f"{k}: {v}" for k, v in cc.most_common(20)))
    if code == "QRK" and orgs:
        q = sorted([o for o in orgs if o["scores"]["q"] >= 3], key=lambda o: o["name"])
        lines.append("DATASET: unusual student groups with their verbatim missions:")
        lines += [f"  ORG[{o['name']}]: {(o['mission'] or '')[:260]}" for o in q[:45]]
    if code == "SOC" and orgs:
        s = sorted([o for o in orgs if o["scores"]["s"] >= 3], key=lambda o: o["name"])
        lines.append(f"DATASET: {sum(1 for o in orgs if o['scores']['s'] >= 2)} directory groups have a service/advocacy mission; examples:")
        lines += [f"  ORG[{o['name']}]: {(o['mission'] or '')[:220]}" for o in s[:30]]
    if code == "DIV" and orgs:
        d = sorted([o for o in orgs if o["scores"]["d"] >= 3], key=lambda o: o["name"])
        lines.append(f"DATASET: {sum(1 for o in orgs if o['scores']['d'] >= 2)} directory groups are cultural, identity, international or faith communities; examples:")
        lines += [f"  ORG[{o['name']}]: {(o['mission'] or '')[:200]}" for o in d[:30]]
    if code == "ACA" and courses:
        ug = [c for c in courses if c["number"][:1] in CFG["undergrad_course_first_digits"]]
        terms = sorted({c["term"] for c in courses})
        ins = {n for c in ug for n in c["instructors"]}
        ge = sum(1 for c in ug if c.get("general_education"))
        lines.append(f"DATASET: {CFG['schedule_name']} ({', '.join(terms)}): {len(courses)} courses, {len(ug)} undergraduate "
                     f"(100-499), {sum(len(c['sections']) for c in courses)} sections, {len(ins)} named instructors of "
                     f"undergraduate courses, {ge} undergraduate courses flagged as satisfying general education.")
        def seats(c):
            """Students registered, counted once: sum only the main section type (lectures if the course has them),
            because labs, discussions and quizzes re-count the same students."""
            by_type = collections.Counter()
            for sec in c["sections"]:
                m = re.match(r"(\d+) / (\d+)", sec.get("registered") or "")
                if m and "CANCEL" not in (sec.get("status") or ""):
                    by_type[(sec.get("type") or "").split("-")[0]] += int(m.group(1))
            if not by_type:
                return 0
            return by_type["Lecture"] if by_type.get("Lecture") else max(by_type.values())
        big = sorted([c for c in ug if c["instructors"] and seats(c) > 0], key=lambda c: -seats(c))[:45]
        lines.append("DATASET: largest undergraduate courses this term by students registered in lecture sections at retrieval time (code, title, instructors, registered, GE):")
        lines += [f"  COURSE[{c['code']} {c['title']}]: taught by {', '.join(c['instructors'][:4])}; {seats(c)} students registered"
                  f"{'; satisfies general education' if c.get('general_education') else ''}" for c in big]
    return "\n".join(lines)


SYSTEM = f"""You write the {SCHOOL} section of a concise research digest for prospective undergraduates and their advisors.
You are given numbered VERIFIED FACTS (each already checked against an official or reputable source) and sometimes DATASET
lines. Write from these only.

HARD RULES
- Every bullet must end with the IDs of the facts that support it, like [F12, F40]. Anything taken from DATASET or ORG
  lines must be cited as [DATA]. Never write [F?] — if you cannot cite it, leave it out.
- Use only names, numbers, dates and claims that appear in the cited facts. Never add outside knowledge, superlatives
  ("best", "top", "unique") or interpretation unless a cited fact states it. When unsure, leave it out.
- Keep periods/dates exactly as the facts give them. Never present a page's publication date as the date an event
  happened; only state an event date if the fact itself says the event happened then.
- Prefer the most recent information; avoid items older than 2022 unless they are history or traditions.
- Prefer specific, concrete, student-relevant items: named programs, clubs, traditions, courses, professors,
  opportunities, eligibility, numbers.

Return json: {{"overview": "<2 sentences describing what this section covers, no factual claims>",
 "at_a_glance": ["<short numeric or key fact bullet> [F..]", ...up to 6],
 "highlights": [{{"heading": "<3-6 word theme>", "bullets": ["<1-2 sentence bullet> [F..]", ...3-6]}}, ...4-7 themes]}}"""

CAP_WORD = re.compile(r"\b([A-Z][A-Za-z&'\.\-]+(?:\s+(?:of|and|for|the|in|&|de|la|at)?\s*[A-Z][A-Za-z&'\.\-]+)*)")
NUM = re.compile(r"\d[\d,\.]*%?")
STOP = {"The", "A", "An", "In", "On", "At", "For", "And", "Of", "Its", "Their", "This", "These", "Students", "Each",
        "University", "Through", "With", "As", "From", "Since",
        "Undergraduates", "Undergraduate", "Many", "More", "Over", "All", "Both", "Also", "It", "They", "Some", "One",
        "Fall", "Spring", "Summer", "Winter", "January", "February", "March", "April", "May", "June", "July", "August",
        "September", "October", "November", "December"} | set(CFG["digest_stop_words"])


def check_bullet(text, id_map, data_blob):
    ids = re.findall(r"F(\d+)", " ".join(re.findall(r"\[([^\]]+)\]", text)))
    uses_data = "[DATA" in text or "DATA]" in text
    if not ids and not uses_data:
        return False, "no citation"
    support = " ".join((id_map[int(i)]["fact"] + " " + id_map[int(i)]["evidence"] + " " +
                        str(id_map[int(i)].get("period") or "") + " " +
                        str((id_map[int(i)].get("entity") or {}).get("name") or ""))
                       for i in ids if int(i) in id_map)
    if uses_data:
        support += " " + data_blob
    if ids and not all(int(i) in id_map for i in ids):
        return False, "unknown fact id"
    body = re.sub(r"\[[^\]]*\]", "", text)
    low = support.lower().replace(",", "")
    for n in NUM.findall(body):
        n2 = n.rstrip(".").replace(",", "")
        if n2 and n2 not in low:
            return False, f"number {n} not in cited facts"
    for m in CAP_WORD.findall(body):
        words = [w for w in re.split(r"\s+", m) if w not in STOP and len(w) > 2]
        for w in words:
            if w.lower().strip(".'") not in support.lower():
                return False, f"name '{w}' not in cited facts"
    return True, ""


def call(user):
    for attempt in range(4):
        try:
            r = httpx.post("https://api.deepseek.com/chat/completions", timeout=600, headers={"Authorization": f"Bearer {KEY}"},
                           json={"model": MODEL, "thinking": {"type": "disabled"}, "temperature": 0.2, "max_tokens": 6000,
                                 "response_format": {"type": "json_object"},
                                 "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]})
            j = r.json()
            return json.loads(j["choices"][0]["message"]["content"]), j.get("usage", {})
        except Exception:
            time.sleep(5 * (attempt + 1))
    return None, {}


sections, audit, all_sources = [], [], collections.OrderedDict()
CACHE = college / "output" / "digest_sections.json"
cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
usage_total = collections.Counter()
ONLY = [c for c in os.environ.get("DIGEST_ONLY", "").split(",") if c]
for num, code, name, sub in CATS:
    if ONLY and code not in ONLY and code in cache:
        c = cache[code]
        ids = {int(k): v for k, v in c["id_map"].items()}
        sections.append({**c, "id_map": ids})
        continue
    cand = candidates(code)
    id_map = {i + 1: fa for i, fa in enumerate(cand)}
    data_blob = extra_context(code)
    user = f"SECTION {num}: {name} — {sub}\n\n{data_blob}\n\nVERIFIED FACTS:\n" + "\n".join(
        f"F{i}: {fa['fact']}" + (f" ({fa['period']})" if fa.get("period") else "") +
        f" <entity: {(fa.get('entity') or {}).get('name')}; source: {urlparse(fa['url']).hostname}>"
        for i, fa in id_map.items())
    res, u = call(user)
    for k, v in u.items():
        if isinstance(v, int):
            usage_total[k] += v
    if not res:
        audit.append({"section": name, "error": "model call failed"})
        continue

    def keep(b):
        ok, why = check_bullet(b, id_map, data_blob)
        audit.append({"section": name, "bullet": b, "kept": ok, "reason": why})
        return ok

    sec = {"num": num, "name": name, "sub": sub, "overview": res.get("overview", ""),
           "glance": [b for b in res.get("at_a_glance", []) if keep(b)],
           "themes": []}
    for th in res.get("highlights", []):
        bl = [b for b in th.get("bullets", []) if keep(b)]
        if bl:
            sec["themes"].append({"heading": th.get("heading", ""), "bullets": bl})
    sec["id_map"] = id_map
    ds = []
    if code in ("EXT", "QRK", "SOC", "DIV") and orgs:
        ds.append(orgs[0]["source_url"])
    if code == "ACA" and courses:
        ds.append(CFG["schedule_index_url"] or (courses[0]["source_url"].split("/school/")[0] + "/catalogue/school"
                                                if "/school/" in courses[0]["source_url"] else courses[0]["source_url"]))
    sec["data_sources"] = ds
    cache[code] = {**{k: v for k, v in sec.items() if k != "id_map"},
                   "id_map": {str(i): {"fact": f["fact"], "sources": f["sources"], "evidence": f["evidence"],
                                       "period": f.get("period"), "entity": f.get("entity")} for i, f in id_map.items()}}
    CACHE.write_text(json.dumps(cache))
    sections.append(sec)
    print(f"{num} {name}: glance {len(sec['glance'])}, themes {len(sec['themes'])}, "
          f"bullets kept {sum(1 for a in audit if a.get('section') == name and a.get('kept'))}/"
          f"{sum(1 for a in audit if a.get('section') == name and 'kept' in a)}", flush=True)


def render_cites(text, id_map, source_ids, data_sources=()):
    def repl(m):
        refs = []
        for part in re.split(r",\s*", m.group(1)):
            mm = re.match(r"F(\d+)", part.strip())
            if mm and int(mm.group(1)) in id_map:
                for u in id_map[int(mm.group(1))]["sources"][:2]:
                    if u not in source_ids:
                        source_ids[u] = len(source_ids) + 1
                    refs.append(source_ids[u])
            elif "DATA" in part:
                for u in data_sources:
                    if u not in source_ids:
                        source_ids[u] = len(source_ids) + 1
                    refs.append(source_ids[u])
        uniq = []
        for r in refs:
            if r not in uniq:
                uniq.append(r)
        return "[" + ", ".join(f"S{r}" if isinstance(r, int) else "directory/schedule" for r in uniq) + "]"
    return re.sub(r"\[([^\]]*F\d+[^\]]*|DATA[^\]]*)\]", repl, text)


stats = json.loads((college / "output" / "run_stats.json").read_text()) if (college / "output" / "run_stats.json").exists() else {}
now = datetime.now(timezone.utc).strftime("%B %d, %Y")
md = [f"# {SCHOOL}: Deep Research Digest", f"*The 10-category student-fit profile — {now}*\n"]
if stats:
    md.append("> **Evidence base:** " + stats.get("headline", "") + "\n")
md.append("Every bullet in this digest is drawn from verified facts: each fact's verbatim quote was checked against the "
          "source page, and each bullet was machine-checked so that every name and number it contains appears in the "
          "facts it cites. Source codes like [S12] refer to the Sources list at the end. Full detail for each category is "
          "in the companion category documents.\n")
md.append("## Contents\n" + "\n".join(f"{s['num']}. {s['name']} — *{s['sub']}*" + "  " for s in sections))
source_ids = collections.OrderedDict()
for s in sections:
    md.append(f"\n## {s['num']}. {s['name']}\n*{s['sub']}*\n")
    if s["overview"]:
        md.append(s["overview"] + "\n")
    if s["glance"]:
        md.append("**At a glance**\n")
        md += [f"- {render_cites(b, s['id_map'], source_ids, s['data_sources'])}" for b in s["glance"]]
        md.append("")
    for th in s["themes"]:
        md.append(f"### {th['heading']}\n")
        md += [f"- {render_cites(b, s['id_map'], source_ids, s['data_sources'])}" for b in th["bullets"]]
        md.append("")
md.append("\n## Sources\n")
for u, n in source_ids.items():
    m = page_meta.get(u, {})
    md.append(f"- **S{n}** {m.get('title') or u} — <{u}>")
out = college / "output" / "digest.md"
out.write_text("\n".join(md) + "\n")
(college / "output" / "digest_audit.jsonl").write_text("\n".join(json.dumps(a) for a in audit) + "\n")
print("digest written", out, "| model usage", dict(usage_total))
print("dropped bullets:", sum(1 for a in audit if a.get("kept") is False), "of", sum(1 for a in audit if "kept" in a))
