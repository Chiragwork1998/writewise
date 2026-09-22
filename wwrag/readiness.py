"""wwrag/readiness.py -- can this college be shipped?

One question: for EVERY kind of applicant this product sells to, does the index hold
findable, specific, division-appropriate material in all ten categories -- and does
retrieval actually return it?

USC passed every existing check (recall 50/50, index verify OK, reports rendered) while a
finance applicant's Research chapter was yeast genetics and transportation engineering, and
the index held Elissa Grossman on "social networks in new venturing", Dan Wadhwani on
entrepreneurial processes and the Lloyd Greif Center ("the oldest entrepreneurship program
in the United States") the whole time. Nothing measured that, because every existing check
measures the CORPUS and none measures what a STUDENT gets.

This module measures both, per academic division, and returns a ship / no-ship verdict.

    python wwrag/readiness.py --index wwrag/index-v2/usc --college colleges/usc
    python wwrag/readiness.py --index ... --college ... --panel wwrag/panel/*.json   # adds part D

Part A  structure        every kind and category present above a floor
Part B  division matrix  research/teaching depth per division, and the spread between them
Part C  named people     distinct people with a research fact, per division
Part D  retrievability   run an archetype panel: personalisation, on-division rate, named-thing rate
Part E  contamination    facts about OTHER institutions sitting on this college's pages
Part F  freshness        share of units that carry a date at all

Thresholds live in GATES and are deliberately visible: they were set from USC's measured
numbers, and the first two or three colleges will move them.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import re
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

CATEGORY_CODES = ("CUL", "EXT", "QRK", "ACA", "RES", "SOC", "INN", "INT", "DIV", "NEW")

GATES: dict[str, Any] = {
    # --- A structure
    "min_facts_per_category": 400,        # USC's floor is QRK at 461
    "min_orgs": 200,
    "min_courses": 1500,
    "min_distinct_urls": 4000,            # USC: 9,028
    # --- B division matrix
    # A student applying to the worst-served degree-granting division must not be more than
    # this many times worse off than one applying to the best-served. USC: 2622/416 = 6.3x.
    "max_division_res_ratio": 3.0,
    "min_res_facts_per_division": 150,
    # --- C named people
    "min_named_researchers_per_division": 40,
    # --- D retrievability (needs --panel)
    "max_mean_jaccard": 0.30,             # USC measured 0.50 over 10 categories, 0.45 in RES
    "min_on_division_rate": 0.25,         # share of a panel member's RES+ACA evidence from a
                                          # division plausibly matching their intended field
    "min_named_thing_rate": 0.70,         # share of selected units naming a proper noun
    # --- E contamination
    "max_foreign_entity_fact_rate": 0.002,  # USC: 935/128,166 = 0.0073
    # --- F freshness
    "min_dated_unit_rate": 0.55,          # USC: 61,866/158,476 = 0.39
}

# entity_name values that name nothing a student could look up
GENERIC_ENTITY = re.compile(
    r"^(the\s+)?(major|minor|programme?s?|courses?|student organi[sz]ations?|"
    r"clubs?\s*&?\s*organi[sz]ations?|undergraduate research|research|funding resources|"
    r"student support|faculty|staff|students|academics?|admissions?|university|campus|"
    r"college|school|department|resources?|services?|events?|opportunit\w*|community|"
    r"diversity|equity|inclusion|leadership( and innovation)?)s?$", re.I)


def die(msg: str) -> None:
    print(f"readiness: {msg}", file=sys.stderr)
    raise SystemExit(2)


# --------------------------------------------------------------------------------------
# divisions: the dimension the index does not have and every report needs
# --------------------------------------------------------------------------------------

def load_divisions(college_dir: Path | None) -> dict[str, re.Pattern[str]]:
    """division name -> host pattern.

    Read from config/college.json. `divisions` is preferred; `school_hosts` (which already
    exists for faculty matching) is the fallback so USC needs no new config to be measured.
    A college with neither cannot be gated: divisions are the whole point of part B.
    """
    if college_dir is None:
        return {}
    p = Path(college_dir) / "config" / "college.json"
    if not p.is_file():
        return {}
    cfg = json.loads(p.read_text())
    divs = cfg.get("divisions")
    if isinstance(divs, dict) and divs:
        return {k: re.compile(v, re.I) for k, v in divs.items()}
    hosts = cfg.get("school_hosts") or {}
    by_host: dict[str, set[str]] = collections.defaultdict(set)
    for code, host in hosts.items():
        if host:
            by_host[host].add(code)
    # host -> a readable division name (its first label)
    return {h.split(".")[0]: re.compile(re.escape(h.split(".")[0]), re.I) for h in by_host}


def degree_granting(college_dir: Path | None, divisions: dict) -> list[str]:
    """Divisions that award undergraduate degrees -- the only ones a gate may fail on.

    A medical or law school with thin undergraduate research is correct, not broken.
    Config key `undergraduate_divisions`; default is every division, which is strict and
    will produce false failures until a college fills the key in. That is the right default:
    a noisy gate gets filled in, a silent one ships engineering labs to a finance applicant.
    """
    if college_dir:
        p = Path(college_dir) / "config" / "college.json"
        if p.is_file():
            ug = json.loads(p.read_text()).get("undergraduate_divisions")
            if isinstance(ug, list) and ug:
                return [d for d in ug if d in divisions]
    return sorted(divisions)


def division_of(url: str, divisions: dict[str, re.Pattern[str]]) -> str | None:
    host = re.match(r"https?://([^/]+)", url or "")
    host = host.group(1) if host else ""
    for name, rx in divisions.items():
        if rx.search(host):
            return name
    return None


# --------------------------------------------------------------------------------------
# the checks
# --------------------------------------------------------------------------------------

class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, bool, str]] = []

    def add(self, part: str, name: str, ok: bool, detail: str) -> None:
        self.rows.append((part, name, ok, detail))

    @property
    def failures(self) -> list[tuple[str, str, bool, str]]:
        return [r for r in self.rows if not r[2]]

    def print(self) -> None:
        part = None
        for p, name, ok, detail in self.rows:
            if p != part:
                print(f"\n{p}")
                part = p
            print(f"  [{'PASS' if ok else 'FAIL'}] {name:38s} {detail}")
        print(f"\n{'=' * 78}")
        if self.failures:
            print(f"NOT READY TO SHIP -- {len(self.failures)} gate(s) failed:")
            for _, name, _, detail in self.failures:
                print(f"   - {name}: {detail}")
        else:
            print("READY TO SHIP")


def part_a_structure(conn: sqlite3.Connection, rep: Report) -> None:
    kinds = dict(conn.execute("SELECT kind, count(*) FROM units GROUP BY kind"))
    for kind, floor in (("fact", 1), ("chunk", 1), ("org", GATES["min_orgs"]),
                        ("course", GATES["min_courses"])):
        n = kinds.get(kind, 0)
        rep.add("A. structure", f"units of kind {kind!r}", n >= floor, f"{n:,} (floor {floor:,})")
    cats = dict(conn.execute(
        "SELECT category_code, count(*) FROM units WHERE kind='fact' GROUP BY category_code"))
    thin = {c: cats.get(c, 0) for c in CATEGORY_CODES if cats.get(c, 0) < GATES["min_facts_per_category"]}
    rep.add("A. structure", "every category above fact floor", not thin,
            f"thin: {thin}" if thin else f"min {min(cats.get(c, 0) for c in CATEGORY_CODES):,}")
    n_urls = conn.execute("SELECT count(DISTINCT source_url) FROM units").fetchone()[0]
    rep.add("A. structure", "distinct source URLs", n_urls >= GATES["min_distinct_urls"],
            f"{n_urls:,} (floor {GATES['min_distinct_urls']:,})")
    ents = [e for (e,) in conn.execute("SELECT entity_name FROM units WHERE kind='fact'")]
    bad = sum(1 for e in ents if not e or len(e.strip()) < 3 or GENERIC_ENTITY.match(e.strip()))
    rate = 1 - bad / max(len(ents), 1)
    rep.add("A. structure", "facts naming a findable entity", rate >= 0.95,
            f"{rate:.1%} ({bad:,} generic or empty)")


def division_matrix(conn: sqlite3.Connection, divisions: dict) -> dict[str, collections.Counter]:
    agg: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    people: dict[str, set[str]] = collections.defaultdict(set)
    for cat, extra, url, ent, kind in conn.execute(
            "SELECT category_code, extra, source_url, entity_name, kind FROM units"):
        d = division_of(url, divisions)
        if not d:
            continue
        a = agg[d]
        a["units"] += 1
        if kind != "fact":
            continue
        a["facts"] += 1
        a[f"cat:{cat or 'NONE'}"] += 1
        etype = (json.loads(extra) or {}).get("entity_type")
        if etype in ("professor", "person"):
            a["person"] += 1
            if cat == "RES":
                a["person_res"] += 1
                if ent:
                    people[d].add(ent)
        if etype in ("lab", "center", "research institute"):
            a["lab_center"] += 1
    for d, names in people.items():
        agg[d]["named_researchers"] = len(names)
    return agg


def part_bc_divisions(agg: dict, ug: list[str], rep: Report) -> None:
    if not agg:
        rep.add("B. divisions", "division map present", False,
                "no `divisions` or `school_hosts` in config/college.json -- part B cannot run")
        return
    print("\nDivision matrix (facts, by category):")
    hdr = f"  {'division':18s} {'units':>7s} {'facts':>7s} {'pRES':>6s} {'people':>7s} {'lab/ctr':>7s} " + \
          " ".join(f"{c:>5s}" for c in CATEGORY_CODES)
    print(hdr)
    for d, a in sorted(agg.items(), key=lambda kv: -kv[1]["units"]):
        mark = "*" if d in ug else " "
        print(f" {mark}{d:18s} {a['units']:7d} {a['facts']:7d} {a['person_res']:6d} "
              f"{a['named_researchers']:7d} {a['lab_center']:7d} " +
              " ".join(f"{a['cat:' + c]:5d}" for c in CATEGORY_CODES))
    print("  (* = degree-granting; gates apply only to these)")

    scored = {d: agg[d]["person_res"] for d in ug if d in agg}
    if scored:
        best, worst = max(scored.values()), min(scored.values())
        ratio = best / max(worst, 1)
        rep.add("B. divisions", "research-depth spread across divisions",
                ratio <= GATES["max_division_res_ratio"],
                f"{ratio:.1f}x  best={max(scored, key=scored.get)}:{best} "
                f"worst={min(scored, key=scored.get)}:{worst} (max {GATES['max_division_res_ratio']}x)")
        under = {d: n for d, n in scored.items() if n < GATES["min_res_facts_per_division"]}
        rep.add("B. divisions", "every division above research floor", not under,
                f"below {GATES['min_res_facts_per_division']}: {under}" if under else "all above floor")
    named = {d: agg[d]["named_researchers"] for d in ug if d in agg}
    if named:
        under = {d: n for d, n in named.items() if n < GATES["min_named_researchers_per_division"]}
        rep.add("C. named people", "named researchers per division", not under,
                f"below {GATES['min_named_researchers_per_division']}: {under}" if under
                else f"min {min(named.values())}")


def part_e_contamination(conn: sqlite3.Connection, college_names: list[str], rep: Report) -> None:
    try:
        rows = [r[0] for r in conn.execute("SELECT name FROM graph_nodes WHERE type='university'")]
    except sqlite3.Error:
        rep.add("E. contamination", "university node list", False, "no graph_nodes table")
        return
    own = re.compile("|".join(re.escape(n) for n in college_names), re.I) if college_names else None
    foreign = [n for n in rows if not (own and own.search(n))]
    if not foreign:
        rep.add("E. contamination", "facts about other institutions", True, "none")
        return
    q = ("SELECT text FROM units WHERE kind='fact' AND entity_name IN (%s)"
         % ",".join("?" * len(foreign)))
    texts = [t for (t,) in conn.execute(q, foreign)]
    orphan = [t for t in texts if own and not own.search(t or "")]
    total = conn.execute("SELECT count(*) FROM units WHERE kind='fact'").fetchone()[0]
    rate = len(orphan) / max(total, 1)
    rep.add("E. contamination", "facts about other institutions", rate <= GATES["max_foreign_entity_fact_rate"],
            f"{len(orphan):,} facts ({rate:.2%}) name another university and never name this one; "
            f"{len(foreign)} foreign university nodes")


def part_f_freshness(conn: sqlite3.Connection, rep: Report) -> None:
    total = conn.execute("SELECT count(*) FROM units").fetchone()[0]
    dated = conn.execute("SELECT count(*) FROM units WHERE year IS NOT NULL").fetchone()[0]
    rate = dated / max(total, 1)
    rep.add("F. freshness", "units carrying a date", rate >= GATES["min_dated_unit_rate"],
            f"{rate:.1%} dated ({dated:,}/{total:,}); recency scoring is blind to the rest")


def part_d_panel(index: Path, panel: list[Path], divisions: dict, rep: Report,
                 python: str, per_category: int, workdir: Path) -> None:
    """Run the archetype panel through the real retriever and measure what a student gets."""
    results: dict[str, dict[str, list[dict]]] = {}
    for p in panel:
        out = workdir / f"panel_{p.stem}.json"
        cmd = [python, str(Path(__file__).with_name("retrieve.py")), "--index", str(index),
               "--profile", str(p), "--out", str(out), "--per-category", str(per_category)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            rep.add("D. retrievability", f"panel run {p.stem}", False, r.stderr.strip()[-200:])
            continue
        results[p.stem] = json.loads(out.read_text())
    if len(results) < 2:
        rep.add("D. retrievability", "panel size", False, "need at least two archetype profiles")
        return

    # D1 personalisation: two different applicants must not get the same report
    per_cat: dict[str, list[float]] = collections.defaultdict(list)
    for a, b in itertools.combinations(results, 2):
        for cat in CATEGORY_CODES:
            A = {u["unit_id"] for u in results[a].get(cat, [])}
            B = {u["unit_id"] for u in results[b].get(cat, [])}
            if A or B:
                per_cat[cat].append(len(A & B) / len(A | B))
    means = {c: sum(v) / len(v) for c, v in per_cat.items() if v}
    overall = sum(means.values()) / len(means)
    worst = max(means, key=means.get)
    rep.add("D. retrievability", "personalisation (mean Jaccard)", overall <= GATES["max_mean_jaccard"],
            f"{overall:.2f} overall, worst {worst}={means[worst]:.2f} "
            f"(max {GATES['max_mean_jaccard']}); 1.00 = every applicant gets the same report")

    # D2 named-thing rate: is the evidence made of things a student can look up?
    bad = tot = 0
    for r in results.values():
        for cat in CATEGORY_CODES:
            for u in r.get(cat, []):
                tot += 1
                e = (u.get("entity_name") or "").strip()
                if not e or len(e) < 3 or GENERIC_ENTITY.match(e):
                    bad += 1
    rate = 1 - bad / max(tot, 1)
    rep.add("D. retrievability", "selected evidence names a findable thing",
            rate >= GATES["min_named_thing_rate"],
            f"{rate:.1%} ({bad}/{tot} generic or unnamed)")

    # D3 on-division rate: does an applicant's Research and Academics evidence come from a
    # division that teaches their field? Needs `field_divisions` on each panel profile.
    scored = []
    for name, r in results.items():
        want = _panel_divisions(panel, name)
        if not want:
            continue
        hit = tot2 = 0
        for cat in ("RES", "ACA"):
            for u in r.get(cat, []):
                tot2 += 1
                if division_of(u.get("source_url") or "", divisions) in want:
                    hit += 1
        if tot2:
            scored.append((name, hit / tot2, hit, tot2))
    if scored:
        worst_name, worst_rate, h, t = min(scored, key=lambda x: x[1])
        rep.add("D. retrievability", "on-division RES+ACA evidence",
                worst_rate >= GATES["min_on_division_rate"],
                f"worst archetype {worst_name}: {h}/{t} = {worst_rate:.0%} "
                f"(min {GATES['min_on_division_rate']:.0%}); "
                + ", ".join(f"{n}:{r:.0%}" for n, r, _, _ in sorted(scored, key=lambda x: x[1])))
    else:
        rep.add("D. retrievability", "on-division RES+ACA evidence", False,
                "no panel profile carries `field_divisions`; add it to measure this")


def _panel_divisions(panel: list[Path], stem: str) -> set[str]:
    for p in panel:
        if p.stem == stem:
            try:
                return set(json.loads(p.read_text()).get("field_divisions") or [])
            except Exception:  # noqa: BLE001
                return set()
    return set()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", required=True, type=Path)
    ap.add_argument("--college", type=Path, help="colleges/<slug>, for config/college.json")
    ap.add_argument("--panel", nargs="*", type=Path, default=[],
                    help="archetype profile.json files; enables part D")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--per-category", type=int, default=24)
    ap.add_argument("--workdir", type=Path, default=Path("/tmp"))
    args = ap.parse_args(argv)

    db = args.index / "chunks.sqlite"
    if not db.is_file():
        die(f"no index at {db}")
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)

    divisions = load_divisions(args.college)
    ug = degree_granting(args.college, divisions)
    names: list[str] = []
    if args.college and (args.college / "config" / "college.json").is_file():
        cfg = json.loads((args.college / "config" / "college.json").read_text())
        names = [n for n in ([cfg.get("name"), cfg.get("short")] + (cfg.get("name_variants") or [])) if n]

    rep = Report()
    part_a_structure(conn, rep)
    part_bc_divisions(division_matrix(conn, divisions), ug, rep)
    part_e_contamination(conn, names, rep)
    part_f_freshness(conn, rep)
    if args.panel:
        part_d_panel(args.index, args.panel, divisions, rep, args.python,
                     args.per_category, args.workdir)
    else:
        rep.add("D. retrievability", "archetype panel", False,
                "not run: pass --panel. Part D is the only part that measures what a STUDENT "
                "gets rather than what the corpus holds.")
    rep.print()
    return 1 if rep.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
