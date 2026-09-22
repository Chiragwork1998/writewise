"""MOSAIC probe -- see what a report WOULD say, before paying to write it.

Retrieval costs nothing and takes under a minute; generation costs money and takes fifteen.
So the loop that matters is: change something, probe, look, change again. Only generate when
the probe looks right.

Prints, per chapter: how many units, how many name something findable, which named things, and
which of the student's own themes they answer. Nothing here calls a writing model.

usage:
  .venv-crawl4ai/bin/python wwrag/probe.py --profile <profile.json> --index wwrag/index-v3 \
      --college-id usc [--per-category 24] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import retrieve as R  # noqa: E402

NAMED = re.compile(
    r"\b[A-Z]{2,5}\s?-?\s?\d{3,5}[A-Za-z]?\b"
    r"|\b(?:[A-Z][\w&'.-]+\s+){1,5}"
    r"(?:Lab|Laboratory|Center|Centre|Institute|Program|Programme|Project|Society|Club|Council|"
    r"Department|School|Group|Initiative|Fellowship|Scholarship|Association|Academy|Team|"
    r"Competition|Challenge|Award|Prize|Hub|Office|Fund|Symposium|Conference|Fair|Showcase)s?\b"
)


def named_things(text: str) -> list[str]:
    return [" ".join(m.split()) for m in NAMED.findall(text or "")]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--college-id", required=True)
    ap.add_argument("--per-category", type=int, default=24)
    ap.add_argument("--json", default=None, help="also write the full result here")
    ap.add_argument("--top", type=int, default=6, help="named things to print per chapter")
    args = ap.parse_args(argv)

    profile = json.loads(Path(args.profile).read_text())
    index = R.CollegeIndex(Path(args.index) / args.college_id)
    embedder = R.Embedder(index.model_name, dims=index.dims)

    # what the student's own file is about, strongest first -- the same ranking the writer sees
    facets = R.profile_facets(profile)
    ranker = R.FacetRanker(embedder, facets, R.CATEGORIES)
    themes = R.theme_strength_semantic(facets, ranker._vec) if ranker._vec else R.theme_strength(facets)
    top_themes = sorted(themes.items(), key=lambda kv: -kv[1])[:8]

    print("=" * 78)
    print(f"WHAT THE FILE IS ABOUT  ({Path(args.profile).parent.name})")
    print("=" * 78)
    declared = profile.get("declared_fields") or []
    if declared:
        print(f"  declared areas : {', '.join(declared)}")
    print(f"  fields in file : {', '.join(profile.get('intended_fields') or []) or '-'}")
    print(f"  level          : {profile.get('level')}")
    print()
    for value, score in top_themes:
        print(f"   {score:5.2f}  {' '.join(str(value).split())[:86]}")

    results, _ = R.retrieve(index, profile, per_category=args.per_category,
                            embedder=embedder, gapfill_enabled=True)

    print()
    print("=" * 78)
    print("WHAT THE COLLEGE OFFERS BACK")
    print("=" * 78)
    print(f"{'':5s} {'units':>6s} {'named':>6s} {'distinct':>9s}   top named things")
    out = {}
    for cat in R.CATEGORIES:
        code = cat["code"]
        units = results.get(code) or []
        names, per_unit = [], 0
        for u in units:
            found = named_things((u.get("entity_name") or "") + " " + (u.get("text") or ""))
            if found:
                per_unit += 1
            names += found
        seen, ordered = set(), []
        for n in names:
            k = n.lower()
            if k not in seen and len(n) > 3:
                seen.add(k)
                ordered.append(n)
        out[code] = {"units": len(units), "named_units": per_unit, "distinct": len(ordered),
                     "things": ordered}
        print(f"{code:5s} {len(units):6d} {per_unit:6d} {len(ordered):9d}   "
              f"{'; '.join(ordered[:args.top])[:96]}")

    tot_u = sum(v["units"] for v in out.values())
    tot_n = sum(v["named_units"] for v in out.values())
    tot_d = sum(v["distinct"] for v in out.values())
    print()
    print(f"TOTAL  {tot_u} units, {tot_n} naming something ({tot_n/max(tot_u,1):.0%}), "
          f"{tot_d} distinct named things")
    empty = [c for c, v in out.items() if v["distinct"] == 0]
    if empty:
        print(f"CHAPTERS WITH NOTHING NAMED: {', '.join(empty)}  <- these will read as filler")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"themes": top_themes, "chapters": out}, indent=1, ensure_ascii=False))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
