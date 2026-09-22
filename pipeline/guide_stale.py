"""Flag guide sentences that may be out of date: leadership titles or 'current' wording backed only by old/undated facts.
usage: python pipeline/guide_stale.py colleges/usc CODE [CODE ...]"""
import json, re, sys
from pathlib import Path
college = Path(sys.argv[1])
CITE = re.compile(r"\[((?:D?[A-Z]{3}-\d+)(?:\s*,\s*D?[A-Z]{3}-\d+)*)\]")
ROLE = re.compile(r"\b(President|Provost|Chancellor|Dean|Vice President|Chair|Director|CEO|head coach|interim)\b")
NOWISH = re.compile(r"\b(currently|now|this year|new|recently|latest|today|upcoming|will)\b", re.I)
for code in sys.argv[2:]:
    pack = json.loads((college / "guide" / "packs" / f"{code}.json").read_text())
    text = (college / "guide" / "chapters" / f"{code}.md").read_text()
    for ln, line in enumerate(text.splitlines(), 1):
        for sent in re.split(r"(?<=[\.\!\?\]])\s+(?=[A-Z\"“(*])", line):
            ids = [i.strip() for m in CITE.findall(sent) for i in m.split(",")]
            if not ids or not (ROLE.search(sent) or NOWISH.search(sent)):
                continue
            years = [int(y) for i in ids if i in pack for y in re.findall(r"20[0-2]\d", str(pack[i].get("period") or "") + " " + pack[i]["fact"])]
            newest = max(years) if years else None
            if newest is None or newest < 2025:
                print(f"{code}:{ln} [{newest or 'undated'}] {CITE.sub('', sent).strip()[:170]}")
