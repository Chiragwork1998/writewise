"""wwrag/score_run.py -- score a finished run, section by section, without a model.

Measures what a reader would notice: does each section name real, specific things; is every
claim cited; does it say why it matters to THIS student; is it honest about what it does not
know. Deterministic, so two runs are comparable and a regression is visible.

These are mechanical proxies, not a judgement of whether the advice is good -- a section can
score well here and still be dull. Blind judges answer that. This catches the failures that
do not need a judge: uncited claims, sections with nothing specific in them, padding.

    python wwrag/score_run.py --run wwrag/runs/anika_v1
    python wwrag/score_run.py --run A --compare B        # two runs side by side
    python wwrag/score_run.py --run A --json             # machine-readable
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

CATEGORY_ORDER = ["CUL", "EXT", "QRK", "ACA", "RES", "SOC", "INN", "INT", "DIV", "NEW"]
CATEGORY_NAMES = {
    "CUL": "Culture", "EXT": "Extracurriculars", "QRK": "Quirks", "ACA": "Academics",
    "RES": "Research", "SOC": "Social Impact", "INN": "Innovative Programs",
    "INT": "Intellectual Alignment", "DIV": "Diversity", "NEW": "External Articles",
}

# A section is only useful if it names things a student can go and find. These are the shapes
# a nameable thing takes in this domain.
COURSE_CODE = re.compile(r"\b[A-Z]{2,5}[\s-]?\d{3,5}[A-Za-z]?\b|\b\d{1,2}\.\d{2,4}\b")
PROPER_NOUN = re.compile(r"\b(?:[A-Z][a-z]{2,}\s+){1,5}(?:Lab|Laboratory|Center|Centre|Institute|"
                         r"Program|Programme|Society|Club|Council|Department|School|Group|Initiative|"
                         r"Fellowship|Scholarship|Association|Collective|Ensemble|Team)\b")
PERSON = re.compile(r"\b(?:Professor|Prof\.|Dr\.)\s+[A-Z][a-z]+")
HEDGE_OK = re.compile(r"\b(could|can apply|may|students say|the group says|according to)\b", re.I)

# Sentences ABOUT the source rather than about the university. A neutral panel of four
# reviewers -- a counsellor, a student, a parent and a fact-checker -- read four of these
# reports with no knowledge of how they were made and scored them 3.75-4.25 out of 10 while
# this scorer was reporting 7.4-8.2. Nearly every line they quoted as padding was of this
# shape: "The article is titled X. This reference shows X." "The photo is credited to
# Venice Tang." "This is a blog post about getting started in research."
#
# Those satisfy every other component here -- they are cited, they name a specific thing,
# they carry a caveat -- which is why the score went up while the documents did not get
# better. A metric that cannot see its own blind spot will be optimised into it.
METADATA_PROSE = re.compile(
    r"\b(the (?:article|page|post|blog|piece|announcement|press release) (?:is )?(?:titled|is called)"
    r"|is titled\b"
    r"|photo (?:is )?credited to"
    r"|(?:this|the) (?:external )?(?:reference|article|source) (?:shows|describes|states|indicates)"
    r"|this is a (?:blog post|news article|press release|announcement)"
    r"|the (?:title|headline) (?:describes|says|reads)"
    r"|published (?:an article|a post|a blog)"
    r"|the page'?s title is)\b", re.I)
PROMISE = re.compile(r"\b(will be admitted|guarantee[sd]?|ensures? (?:your )?(?:admission|acceptance)|"
                     r"you will get (?:in|accepted))\b", re.I)


def _restates_itself(text: str) -> bool:
    """True when an item pads by saying the same thing twice.

    Observed shape: "The IDM lets students design programs crossing traditional majors.
    Directed by X and Y, the IDM lets students design an individual program of study that
    crosses the lines between traditional majors." Two sentences, one fact. A reader notices
    immediately, and it is invisible to every other check here because both halves are true
    and both are cited.
    """
    sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", text) if len(x.strip()) > 40]
    for i, a in enumerate(sentences):
        wa = set(re.findall(r"[a-z]{4,}", a.lower()))
        if len(wa) < 6:
            continue
        for b in sentences[i + 1:]:
            wb = set(re.findall(r"[a-z]{4,}", b.lower()))
            if len(wb) < 6:
                continue
            overlap = len(wa & wb) / min(len(wa), len(wb))
            if overlap >= 0.72:
                return True
    return False


def load(run: Path) -> dict[str, Any]:
    def read(name: str, *alts: str) -> Any:
        for n in (name, *alts):
            p = run / n
            if p.is_file():
                return json.loads(p.read_text())
        return None

    items = read("report_items.json", "verified/report_verified.json")
    if isinstance(items, dict):
        items = items.get("items") or items.get("report") or []
    evidence = read("evidence_units.json", "evidence.json") or {}
    units = evidence.get("units") if isinstance(evidence, dict) else None
    if units is None and isinstance(evidence, dict):
        units = [u for v in evidence.values() if isinstance(v, list) for u in v]
    return {
        "items": items or [],
        "units": {u["unit_id"]: u for u in (units or []) if isinstance(u, dict) and u.get("unit_id")},
        "ledger": read("verified/ledger.json") or {},
        "run": read("run.json") or {},
        "build": read("report_build.json") or {},
        "profile": read("profile.json") or {},
    }


def score_section(code: str, items: list[dict], units: dict[str, dict]) -> dict[str, Any]:
    n = len(items)
    if not n:
        return {"code": code, "name": CATEGORY_NAMES.get(code, code), "items": 0, "score": 0.0,
                "empty": True}

    prose = [" ".join(str(i.get(f) or "") for f in ("headline", "body", "why_it_matters"))
             for i in items]
    # caveats were carrying much of the padding ("Worth knowing: this is a blog post about
    # getting started in research"), and scanning only the body missed all of it
    all_text = [" ".join(str(i.get(f) or "")
                         for f in ("headline", "body", "why_it_matters", "caveat"))
                for i in items]
    specific = sum(1 for t in prose
                   if COURSE_CODE.search(t) or PROPER_NOUN.search(t) or PERSON.search(t))
    cited = sum(1 for i in items if i.get("evidence_ids"))
    why = sum(1 for i in items if (i.get("why_it_matters") or "").strip())
    basis = sum(1 for i in items if i.get("profile_basis"))
    caveat = sum(1 for i in items if (i.get("caveat") or "").strip())
    promises = sum(1 for t in prose if PROMISE.search(t))
    metadata = sum(1 for t in all_text if METADATA_PROSE.search(t))
    restated = sum(1 for t in all_text if _restates_itself(t))

    cited_ids = {e for i in items for e in (i.get("evidence_ids") or [])}
    dangling = [e for e in cited_ids if e not in units]
    relations = sum(1 for e in cited_ids if (units.get(e) or {}).get("kind") == "relation")
    distinct_sources = len({(units.get(e) or {}).get("source_url") for e in cited_ids} - {None})
    words = statistics.mean(len(t.split()) for t in prose) if prose else 0

    # 0-10. Citation integrity and specificity dominate: an uncited or vague section fails
    # regardless of how well it reads. Promises are a hard penalty -- the product must never
    # make one, so a single occurrence costs more than any other factor can earn back.
    parts = {
        "cited": 2.5 * (cited / n),
        "specific": 2.5 * (specific / n),
        "why_for_you": 1.5 * (why / n),
        "resume_basis": 1.0 * (basis / n),
        "honest_caveat": 1.0 * min(caveat / n, 1.0),
        "source_spread": 1.0 * min(distinct_sources / max(n * 1.5, 1), 1.0),
        "uses_relations": 0.5 * min(relations / max(n, 1), 1.0),
    }
    score = sum(parts.values())
    if dangling:
        score -= 3.0
    if promises:
        score -= 5.0
    # each item that talks about its source instead of the university costs more than the
    # specificity point it was earning by naming that source
    score -= 1.2 * (metadata / n)
    score -= 1.0 * (restated / n)
    score = max(0.0, min(10.0, score))

    return {
        "code": code, "name": CATEGORY_NAMES.get(code, code), "items": n,
        "score": round(score, 2), "specific": specific, "cited": cited, "why": why,
        "basis": basis, "caveat": caveat, "relations_cited": relations,
        "distinct_sources": distinct_sources, "avg_words": round(words),
        "dangling_citations": len(dangling), "promises": promises,
        "metadata_prose": metadata, "restated": restated, "empty": False,
        "parts": {k: round(v, 2) for k, v in parts.items()},
    }


def score_run(run: Path) -> dict[str, Any]:
    data = load(run)
    by_cat: dict[str, list] = {c: [] for c in CATEGORY_ORDER}
    for item in data["items"]:
        by_cat.setdefault(item.get("category_code", "?"), []).append(item)

    sections = [score_section(c, by_cat.get(c) or [], data["units"]) for c in CATEGORY_ORDER]
    live = [s for s in sections if not s["empty"]]
    overall = round(statistics.mean([s["score"] for s in sections]), 2) if sections else 0.0

    led = data["ledger"].get("counts") or data["ledger"] or {}
    return {
        "run": str(run),
        "overall": overall,
        "sections_populated": len(live),
        "items_total": sum(s["items"] for s in sections),
        "weakest": min(live, key=lambda s: s["score"])["code"] if live else None,
        "strongest": max(live, key=lambda s: s["score"])["code"] if live else None,
        "verification": {k: led.get(k) for k in
                         ("claims_total", "supported", "corrected", "removed", "items_dropped")
                         if k in led},
        "pages": data["build"].get("pages"),
        "cost_usd": (lambda c: c.get("total") if isinstance(c, dict) else c)(
            data["run"].get("cost_usd") or data["run"].get("cost") or {}),
        "sections": sections,
    }


def table(result: dict[str, Any], label: str = "") -> str:
    out = [f"{'cat':5} {'name':22} {'items':>5} {'spec':>5} {'cite':>5} {'why':>4} "
           f"{'cav':>4} {'rel':>4} {'src':>4} {'meta':>5} {'SCORE':>6}"]
    out.append("-" * 76)
    for s in result["sections"]:
        if s["empty"]:
            out.append(f"{s['code']:5} {s['name']:22} {'-':>5} {'':>5} {'':>5} {'':>4} "
                       f"{'':>4} {'':>4} {'':>4} {'EMPTY':>6}")
            continue
        n = s["items"]
        out.append(f"{s['code']:5} {s['name']:22} {n:>5} {s['specific']}/{n:<3} "
                   f"{s['cited']}/{n:<3} {s['why']:>4} {s['caveat']:>4} "
                   f"{s['relations_cited']:>4} {s['distinct_sources']:>4} "
                   f"{s.get('metadata_prose', 0) + s.get('restated', 0):>5} {s['score']:>6.2f}")
    out.append("-" * 76)
    head = f"OVERALL {result['overall']:.2f}/10"
    if label:
        head = f"{label}: {head}"
    extra = []
    if result.get("pages"):
        extra.append(f"{result['pages']}pp")
    if result.get("cost_usd"):
        extra.append(f"${float(result['cost_usd']):.4f}")
    v = result.get("verification") or {}
    if v.get("removed") is not None:
        extra.append(f"{v.get('removed')} claims removed")
    out.append(head + ("   (" + ", ".join(extra) + ")" if extra else ""))
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Score a finished run section by section.")
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--compare", type=Path, default=None, help="a second run to diff against")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    a = score_run(args.run.resolve())
    if args.json and not args.compare:
        print(json.dumps(a, indent=2))
        return 0
    print(table(a, args.run.name))

    if args.compare:
        b = score_run(args.compare.resolve())
        print()
        print(table(b, args.compare.name))
        print()
        print(f"{'cat':5} {'name':22} {args.run.name[:12]:>13} {args.compare.name[:12]:>13} {'delta':>8}")
        print("-" * 66)
        for sa, sb in zip(a["sections"], b["sections"]):
            d = sb["score"] - sa["score"]
            mark = "  <<<" if abs(d) >= 1.0 else ""
            print(f"{sa['code']:5} {sa['name']:22} {sa['score']:>13.2f} {sb['score']:>13.2f} "
                  f"{d:>+8.2f}{mark}")
        print("-" * 66)
        print(f"{'':5} {'OVERALL':22} {a['overall']:>13.2f} {b['overall']:>13.2f} "
              f"{b['overall'] - a['overall']:>+8.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
