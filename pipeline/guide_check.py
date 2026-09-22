"""Check a student-guide chapter against its source pack.

Every sentence, bullet or table row that states a specific name or number must cite pack IDs like [RES-12, DRES-3],
and every number and capitalised name in it must appear in the cited facts. Sentences with no specifics (advice,
interpretation) need no citation. Headings may only use names found somewhere in the chapter's cited facts.
usage: python pipeline/guide_check.py colleges/usc RES      (exit code 1 when anything fails)
"""
import json
import re
import sys
from pathlib import Path

college, code = Path(sys.argv[1]), sys.argv[2]
pack = json.loads((college / "guide" / "packs" / f"{code}.json").read_text())
text = (college / "guide" / "chapters" / f"{code}.md").read_text()

CITE = re.compile(r"\[((?:D?[A-Z]{3}-\d+)(?:\s*,\s*D?[A-Z]{3}-\d+)*)\]")
NUM = re.compile(r"(?<![A-Za-z\-])\d[\d,\.]*%?")
CAP = re.compile(r"(?<![\w'’])[A-Z][A-Za-z&'’\.\-]*[A-Za-z]")
STOP = set("""A An And As At Be But By For From How If In Into Is It Its Not Of On Or Our So That The Their Then There These
They This Those To Was We What When Where Which Who Why Will With You Your Yours Here Each Every Many Most More Some Few
Also Both All Any One Two Three Four Five Six Seven Eight Nine Ten First Second Third Students Student Undergraduates
Undergraduate University Fall Spring Summer Winter January February March April May June July August September October
November December Monday Tuesday Wednesday Thursday Friday Saturday Sunday USC USC's Southern California Trojan Trojans
Los Angeles LA I Ask Look Check Consider Explore Visit Talk Read Think Try Note Keep Start Plan Find Worth What's
Who's Don't Can't It's You'll You're If you're Good Great Best Key Quick At-a-glance Glance Why How Around Nearly About
Over Under Across Beyond Beyond Within Before After During Since Until Unlike Like Among Between Through Along Instead
Rather Whether Whatever However Although Though While Because Even Only Just Still Yet Already Often Usually Sometimes
Take Join Apply Use Get See Expect Meet Build Learn Pick Choose Compare Bring Watch Remember""".split())
IGNORE_WORDS = {"Q&A", "AI", "STEM", "GPA", "PhD", "Ph.D", "Ph.D.", "B.A.", "B.S.", "Mon", "Wed", "Tue", "Thu", "Fri",
                "Section", "Sections", "Chapter", "Category"}


def support_for(ids):
    parts = []
    for i in ids:
        f = pack[i]
        parts += [f["fact"], f.get("evidence") or "", str(f.get("period") or ""), str(f.get("entity") or "")]
    return re.sub(r"[’‘]", "'", " ".join(parts)).lower().replace(",", "")


def specifics(body, skip_first=True):
    body = re.sub(r"[’‘]", "'", body)
    nums = [n.rstrip(".").replace(",", "") for n in NUM.findall(body)]
    nums = [n for n in nums if n and n.rstrip("%")]
    words = CAP.findall(body)
    if skip_first and words and body.lstrip("*_>-• \"'(").startswith(words[0]):
        words = words[1:]
    names = []
    for w in words:
        w2 = re.sub(r"('s|s')$", "", w).strip(".'")
        if w2 in STOP or w in STOP or w2 in IGNORE_WORDS or len(w2) < 3:
            continue
        names.append(w2)
    return nums, names


units, headings = [], []
in_table_header = False
for ln, raw in enumerate(text.splitlines(), 1):
    line = raw.strip()
    if not line or re.fullmatch(r"[\|\-\s:]+", line):
        continue
    if line.startswith("#"):
        headings.append((ln, line.lstrip("# ").strip()))
        continue
    if line.startswith("|"):
        units.append((ln, line))  # each table row is one unit
        continue
    body = re.sub(r"^(>\s*)?([-*]|\d+\.)\s+", "", line)
    # sentence split that keeps citations with their sentence
    parts = re.split(r"(?<=[\.\!\?\]])\s+(?=[A-Z\"“(*])", body)
    for p in parts:
        units.append((ln, p))

fails, cited_all = [], set()
for ln, u in units:
    ids = [i.strip() for m in CITE.findall(u) for i in m.split(",")]
    unknown = [i for i in ids if i not in pack]
    if unknown:
        fails.append((ln, u, f"unknown ID {unknown[0]}"))
        continue
    cited_all.update(ids)
    body = CITE.sub("", u)
    nums, names = specifics(body, skip_first=not u.startswith("|"))
    if u.startswith("|") and not ids:
        if nums or names:
            fails.append((ln, u, "table row with specifics has no citation"))
        continue
    if not ids:
        if nums or names:
            fails.append((ln, u, f"needs a citation (specifics: {(nums + names)[:3]})"))
        continue
    sup = support_for(ids)
    for n in nums:
        if n.rstrip("%") not in sup:
            fails.append((ln, u, f"number {n} not in cited facts"))
            break
    else:
        for w in names:
            if w.lower() not in sup:
                fails.append((ln, u, f"name '{w}' not in cited facts"))
                break

chapter_sup = support_for(sorted(cited_all)) if cited_all else ""
for ln, h in headings[1:]:
    _, names = specifics(h, skip_first=False)
    for w in names:
        if w.lower() not in chapter_sup:
            fails.append((ln, "# " + h, f"heading name '{w}' not in any cited fact"))
            break

words = len(re.findall(r"\w+", CITE.sub("", text)))
print(f"{code}: {len(units)} sentences/rows, {len(cited_all)} distinct facts cited, ~{words:,} words, {len(fails)} problems")
for ln, u, why in fails[:80]:
    print(f"  line {ln}: {why} :: {u[:160]}")
sys.exit(1 if fails else 0)
