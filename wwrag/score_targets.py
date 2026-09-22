"""Score a retrieval against a pre-registered target list. The ONE number every change is judged by.

The targets are things human counsellors said should have matched a student and did not. They
were chosen by them, not by us, before any fix -- so a change cannot game them. Retrieval costs
nothing, so this runs in ~30s and can be run after every edit.

usage:
  .venv-crawl4ai/bin/python wwrag/score_targets.py wwrag/targets/aadya_aggarwal_usc.json [--per-category 24]
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import retrieve as R

def score(spec: dict, per_category: int, index_root: str = "wwrag/index-v3") -> dict:
    prof = json.loads(Path(spec["profile"]).read_text())
    if spec.get("declared_fields"):
        prof["intended_fields"] = list(spec["declared_fields"])
        prof["declared_fields"] = list(spec["declared_fields"])
    idx = R.CollegeIndex(Path(index_root) / spec["college_id"])
    emb = R.Embedder(idx.model_name, dims=idx.dims)
    res, _ = R.retrieve(idx, prof, per_category=per_category, embedder=emb, gapfill_enabled=True)
    # SECOND NUMBER, independent of anyone's list: how many units name a specific thing AND
    # are about what this student declared or what their file is most about. The terms come
    # from the profile itself (declared fields + the strongest theme_strength phrases), so it is
    # the same for any student at any college. Exact target lists are brittle -- a genuinely
    # relevant course nobody happened to name scores zero on them -- and four separate fixes
    # each measured "no change" on the list while visibly changing what the chapters held.
    named_rx = re.compile(
        r"\b[A-Z]{2,5}\s?-?\s?\d{3,5}[A-Za-z]?\b|\b(?:[A-Z][\w&'.-]+\s+){1,5}"
        r"(?:Lab|Laboratory|Center|Centre|Institute|Program|Programme|Project|Society|Club|Council|"
        r"Department|School|Group|Initiative|Fellowship|Association|Academy|Team|Competition|"
        r"Challenge|Award|Prize|Hub|Office|Fund|Symposium|Conference|Fair|Showcase)s?\b"
        r"|\b(?:Professor|Prof\.|Dr\.)\s+[A-Z][a-z]+")
    facets = R.profile_facets(prof)
    strength = R.theme_strength(facets)
    top = [v for v, _ in sorted(strength.items(), key=lambda kv: -kv[1])[:10]]
    stems: set[str] = set()
    for phrase in list(prof.get("intended_fields") or []) + top:
        for w in re.findall(r"[A-Za-z]{5,}", str(phrase)):
            wl = w.lower()
            if wl in R.STOPWORDS or wl in R.RESUME_FILLER:
                continue
            stems.add(wl[:6])
    onbrief_rx = re.compile(r"\b(?:" + "|".join(sorted(map(re.escape, stems))) + r")", re.I) if stems else None
    specific_relevant = 0
    for code, units in res.items():
        for u in units:
            t = (u.get("text") or "") + " " + (u.get("entity_name") or "")
            if named_rx.search(t) and onbrief_rx is not None and onbrief_rx.search(t):
                specific_relevant += 1
    where: dict[str, list[str]] = {}
    for code, units in res.items():
        blob = " ".join((u.get("text") or "") + " " + (u.get("entity_name") or "") for u in units)
        for name, rx in spec["targets"].items():
            if re.search(rx, blob, re.I):
                where.setdefault(name, []).append(code)
    hit = sorted(where)
    miss = [n for n in spec["targets"] if n not in where]
    return {"hit": hit, "miss": miss, "n": len(spec["targets"]), "where": where,
            "units": sum(len(u) for u in res.values()),
            "specific_relevant": specific_relevant, "brief_stems": sorted(stems)}

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--per-category", type=int, default=24)
    ap.add_argument("--json", default=None)
    ap.add_argument("--index-root", default="wwrag/index-v3")
    ap.add_argument("--profile", default=None, help="override the profile path in the spec")
    a = ap.parse_args(argv)
    spec = json.loads(Path(a.spec).read_text())
    if a.profile:
        spec["profile"] = a.profile
    s = score(spec, a.per_category, index_root=a.index_root)
    print(f"{spec['student']} @ {spec['college_id']}:  targets {len(s['hit'])}/{s['n']}   "
          f"specific+relevant {s['specific_relevant']}/{s['units']}")
    print(f"  brief stems: {', '.join(s['brief_stems'])}")
    print("  HIT : " + ", ".join(f"{h} [{','.join(s['where'][h])}]" for h in s["hit"]))
    print("  MISS: " + ", ".join(s["miss"]))
    if a.json:
        Path(a.json).write_text(json.dumps(s, indent=1))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
