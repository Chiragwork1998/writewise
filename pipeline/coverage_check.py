"""Is this college ready to ship? One deterministic go/no-go before any student sees a report.

USC was tuned by hand over weeks. The other colleges will not be, so the question "does this
college have enough material to answer all ten categories" has to be answerable without reading
a report and without spending a cent on the API.

Thresholds are read off USC, which is the only college known to produce usable reports, and are
set below its weakest category rather than at it -- the gate is meant to catch a college that
failed to crawl, not to insist every college be as rich as USC. USC's floor is QRK: 465 units,
64 hosts, 311 entities. A category under 200/20/100 cannot support a chapter.

usage: python pipeline/coverage_check.py colleges/usc wwrag/index-v3/usc
       exit 0 = ship, 1 = at least one FAIL
"""
import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent))
import college_config  # noqa: E402

CATEGORIES = {
    "CUL": "Culture", "EXT": "Extracurriculars", "QRK": "Quirks", "ACA": "Academics",
    "RES": "Research", "SOC": "Social Impact", "INN": "Innovative Programs",
    "INT": "Intellectual Alignment", "DIV": "Diversity", "NEW": "External Articles",
}
# (fail below, warn below) -- a category under the fail line cannot support a chapter
MIN_UNITS = (200, 400)
MIN_HOSTS = (20, 50)
MIN_ENTITIES = (100, 250)
MIN_RECALL = 0.80
MIN_GRAPH_EDGES = 2000


def verdict(value, fail_warn):
    fail, warn = fail_warn
    return "FAIL" if value < fail else ("warn" if value < warn else "ok")


def recall(db, cfg):
    """How many of the things a researcher expects to find are actually in the index."""
    items = cfg.get("recall_items") or []
    if not items:
        return None, []
    ctx = cfg.get("recall_context") or {}
    cased = set(cfg.get("recall_case_sensitive") or [])
    missing = []
    for item in items:
        if item in ctx:
            rx = re.compile(ctx[item], 0 if item in cased else re.I)
            hit = any(rx.search(t or "") for (t,) in db.execute(
                "SELECT text FROM units WHERE text LIKE ?", (f"%{item.split()[0]}%",)))
        elif item in cased:
            hit = db.execute("SELECT 1 FROM units WHERE text GLOB ? LIMIT 1",
                             (f"*{item}*",)).fetchone() is not None
        else:
            hit = db.execute("SELECT 1 FROM units WHERE text LIKE ? LIMIT 1",
                             (f"%{item}%",)).fetchone() is not None
        if not hit:
            missing.append(item)
    return (len(items) - len(missing)) / len(items), missing


def main(argv):
    if len(argv) < 3:
        raise SystemExit(__doc__)
    college, index = Path(argv[1]), Path(argv[2])
    cfg = college_config.load(college)
    sqlite = index / "chunks.sqlite"
    if not sqlite.exists():
        raise SystemExit(f"no index at {sqlite}")
    db = sqlite3.connect(sqlite)

    print(f"{cfg.get('name', college.name)}  --  {index}\n")
    print(f"{'':5s} {'category':24s} {'units':>7s} {'hosts':>6s} {'entities':>9s}   verdict")
    failures = []
    for code, label in CATEGORIES.items():
        units = db.execute("SELECT COUNT(*) FROM units WHERE category_code=?", (code,)).fetchone()[0]
        hosts = len({urlparse(u or "").netloc for (u,) in db.execute(
            "SELECT source_url FROM units WHERE category_code=?", (code,)) if u})
        ents = db.execute("SELECT COUNT(DISTINCT entity_name) FROM units "
                          "WHERE category_code=? AND entity_name IS NOT NULL", (code,)).fetchone()[0]
        marks = [verdict(units, MIN_UNITS), verdict(hosts, MIN_HOSTS), verdict(ents, MIN_ENTITIES)]
        worst = "FAIL" if "FAIL" in marks else ("warn" if "warn" in marks else "ok")
        if worst == "FAIL":
            failures.append(f"{code} ({label}): units={units} hosts={hosts} entities={ents}")
        print(f"{code:5s} {label:24s} {units:7d} {hosts:6d} {ents:9d}   {worst}")

    print()
    nodes = db.execute("SELECT COUNT(*) FROM graph_nodes").fetchone()[0] \
        if db.execute("SELECT 1 FROM sqlite_master WHERE name='graph_nodes'").fetchone() else 0
    edges = db.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0] if nodes else 0
    g = "ok" if edges >= MIN_GRAPH_EDGES else "FAIL"
    if g == "FAIL":
        failures.append(f"entity graph too small: {edges} edges (need {MIN_GRAPH_EDGES})")
    print(f"entity graph            {nodes:7d} nodes {edges:7d} edges   {g}")

    rate, missing = recall(db, cfg)
    if rate is None:
        print("recall_items            not set in config/college.json   warn "
              "(list 40-50 things a researcher expects, so this gate can mean something)")
    else:
        r = "ok" if rate >= MIN_RECALL else "FAIL"
        if r == "FAIL":
            failures.append(f"recall {rate:.0%} of expected items (need {MIN_RECALL:.0%})")
        print(f"expected items found    {rate:6.0%}                       {r}")
        if missing:
            print(f"  missing: {', '.join(missing[:10])}{' ...' if len(missing) > 10 else ''}")

    print()
    if failures:
        print("NOT READY TO SHIP:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("READY: every category has enough material to answer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
