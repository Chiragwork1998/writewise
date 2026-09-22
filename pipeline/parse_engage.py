"""Deterministic parser for the EngageSC (CampusGroups) directory page: every student group with its type,
categories, mission and membership benefits, copied verbatim. Output: extract/organizations.jsonl"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import college_config  # noqa: E402
from firecrawl_client import url_id  # noqa: E402

college = Path(sys.argv[1])
cfg = college_config.load(college)
URL = cfg.get("org_directory_url")
if not URL:
    raise SystemExit(f"set org_directory_url in {college}/config/college.json "
                     f"(the CampusGroups/Engage directory page listing every group)")
rec = json.loads((college / "pages" / f"{url_id(URL)}.json").read_text())
md = rec["markdown"]


# Category tags used by the directory's own filter list; a trailing " - <tags>" is only a category list when every
# comma-separated part is one of these (otherwise the dash belongs to the group type, e.g. "FSLD - Panhellenic Council").
CATEGORY_TAGS = {"Academic", "Asian Greek Council", "Campus Department", "Career", "Civic Engagement", "Club Sport",
                 "Design Team", "Environmental/Sustainability", "Ethnic/Cultural", "Health & Wellness",
                 "Interfaith Council", "Interfraternity Council", "Media/Publications",
                 "Multicultural Greek Council", "National Pan-Hellenic Council", "Order of Omega", "Panhellenic Council",
                 "Political", "Pre-Professional", "Recreation", "Service", "Social", "Social Awareness", "Spiritual",
                 "Undergraduate Student Government", "Visual & Performing Arts"}
# schools name their own org categories ("Viterbi Student Organization"); those come from the college's settings
CATEGORY_TAGS |= set(cfg.get("org_category_tags_extra") or [])


def normalize_markdown(text):
    """Self-hosted Firecrawl uses a different HTML->markdown converter (setext headings, indented bullets).
    Rewrite to the cloud style so one parser handles both."""
    lines = [l.strip() for l in text.splitlines()]
    out = []
    i = 0
    while i < len(lines):
        l = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if l.startswith("["):
            j = i + 1
            while j < len(lines) and not lines[j]:
                j += 1
            if j < len(lines) and re.fullmatch(r"-{5,}", lines[j]):
                out.append("## " + l)
                i = j + 1
                continue
        if " - " in l and len(l) < 300:
            gtype, cats = l.rsplit(" - ", 1)
            parts = [c.strip() for c in cats.split(",") if c.strip()]
            # a council name can also be a category tag ("FSLD - Panhellenic Council"), so only treat the tail as
            # categories when what remains is a real group-type name rather than a short code
            if parts and all(c in CATEGORY_TAGS for c in parts) and len(gtype.strip()) >= 8:
                out += [gtype.strip(), "\\- " + cats.strip()]
                i += 1
                continue
        out.append(re.sub(r"^\*\s{2,}", "- ", l))
        i += 1
    return "\n".join(out)


if "## [" not in md:
    md = normalize_markdown(md)

LINK = re.compile(r"\[([^\]]*)\]\(([^)]*)\)")
blocks = re.split(r"\n(?=## \[)", md)
orgs = []
names = [re.match(r"## \[([^\]]+)\]", b).group(1).replace("\\", "").strip() if re.match(r"## \[([^\]]+)\]", b) else ""
         for b in blocks]
for bi, b in enumerate(blocks[1:], start=1):
    next_name = names[bi + 1] if bi + 1 < len(names) else None
    head = re.match(r"## \[([^\]]+)\]\(([^)]+)\)", b)
    if not head:
        continue
    name = head.group(1).replace("\\", "").strip()
    link = head.group(2).strip()
    lines = [l.strip() for l in b.splitlines() if l.strip()]
    gtype, cats = None, []
    if len(lines) > 1 and not lines[1].startswith("["):
        gtype = lines[1].replace("\\", "").strip()
    for l in lines[2:5]:
        if l.startswith("\\-") or l.startswith("-"):
            cats = [c.strip() for c in l.lstrip("\\-").split(",") if c.strip()]
            break

    def section(label):
        m = re.search(r"\*\*" + label + r"\*\*\s*\n(.*?)(?=\n\*\*[A-Z][^*]{2,40}\*\*|\nLifetime membership|\n[-*] [^\n]*\n\n\[!\[|\Z)",
                      b, re.S)
        if not m:
            return None
        txt = re.sub(r"\n\s*\n+", "\n", m.group(1)).strip()
        txt = LINK.sub(r"\1", txt).replace("\\", "")
        return txt or None

    mission = section("Mission")
    benefits = section("Membership Benefits")
    # the next group's name can leak in as a trailing "- Name" bullet; strip it only if it IS the next name
    if next_name:
        tail = re.compile(r"\n?-\s+" + re.escape(next_name) + r"\s*$")
        benefits = tail.sub("", benefits).strip() if benefits else benefits
        mission = tail.sub("", mission).strip() if mission else mission
    duration = re.search(r"\n(Lifetime membership|Annual membership|Semester membership|[^\n]*membership)\s*\n", b)
    orgs.append({"name": name, "link": link, "group_type": gtype, "categories": cats, "mission": mission,
                 "membership_benefits": benefits, "membership_duration": duration.group(1) if duration else None,
                 "source_url": URL, "fetched_at": rec.get("fetched_at")})

out = college / "extract" / "organizations.jsonl"
with out.open("w") as f:
    for o in orgs:
        f.write(json.dumps(o) + "\n")
print(f"organizations={len(orgs)} with_mission={sum(1 for o in orgs if o['mission'])} "
      f"types={len({o['group_type'] for o in orgs})}")
