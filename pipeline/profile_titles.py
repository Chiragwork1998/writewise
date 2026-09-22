"""Deterministic name + job title facts for faculty profile pages too thin for AI extraction
(e.g. template pages with only a heading, title and email). Evidence is the verbatim title line."""
import glob, json, re, sys
from pathlib import Path
college = Path(sys.argv[1])
sys.path.insert(0, str(Path(__file__).resolve().parent))
from college_config import load as load_config  # noqa: E402
SHORT = load_config(college)["short"] or "the university"
done = {json.loads(l)["key"].rsplit("#", 1)[0] for l in (college / "extract" / "extracted_pages.jsonl").read_text().splitlines()}
have = {json.loads(l)["url"] for l in (college / "extract" / "facts_raw.jsonl").read_text().splitlines()}
out = []
for f in glob.glob(str(college / "pages" / "*.json")):
    r = json.load(open(f))
    if r.get("status") != "ok" or (r.get("meta") or {}).get("bucket") != "faculty_profile" or r["url"] in have:
        continue
    lines = [l.strip() for l in (r.get("markdown") or "").splitlines() if l.strip()]
    for i, l in enumerate(lines[:-1]):
        m = re.match(r"^#\s+([A-Z][\w.'\- ]{2,60})$", l)
        nxt = lines[i + 1]
        if m and not nxt.startswith(("!", "[", "#", "*", "-")) and 3 < len(nxt) < 160 and "@" not in nxt:
            name, title = m.group(1).strip(), nxt
            out.append({"url": r["url"], "title": r.get("title"), "fetched_at": r.get("fetched_at"), "page_published": None,
                        "page_type": "faculty profile", "page_date": None, "bucket": "faculty_profile", "category": "ACA",
                        "fact": f"{name} is {title} at {SHORT}.", "evidence": f"{name} {title}", "period": None,
                        "entity": {"name": name, "type": "professor"}, "relations": [], "verification": "exact", "extractor": "profile_title"})
            break
with (college / "extract" / "facts_raw.jsonl").open("a") as fh:
    for x in out:
        fh.write(json.dumps(x) + "\n")
print("title facts added:", len(out)); print([x["fact"] for x in out[:5]])
