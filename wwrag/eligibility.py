"""wwrag/eligibility.py -- who is actually allowed to do the thing we are recommending.

The pipeline could prove that something exists and never asked whether this student could
have it. A reviewer found the consequence in a finished report: the top match in the Research
chapter was USC's Robotics and Autonomous Systems REU, which is funded by the National Science
Foundation. NSF REU awards are, as a rule, restricted to US citizens and permanent residents.
The student was an Indian national. Every sentence about it was correctly sourced, every quote
verbatim, and the recommendation was still wrong.

That is a different failure from a fabricated quote and the verification gate cannot see it:
"this programme exists" is true, and "this fits you" is the claim that is false.

DESIGN DECISION, deliberately conservative: this module does NOT infer a student's citizenship
from their resume and does NOT silently drop restricted material. Guessing nationality from a
document is unreliable and the failure mode is ugly. Instead it DETECTS the restriction a
source states and makes sure the reader is told. A student who is eligible loses nothing by
being told the rule; a student who is not is saved from wasting an application.

Two things it does do:
  1. tag any unit whose own text states a restriction, with the restriction in plain words
  2. flag material that is simply the wrong life stage for the applicant -- a pre-college
     summer course is not something an incoming undergraduate can enrol in, and that IS safe
     to filter, because the profile states the applicant's level and the source states whose
     programme it is
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

# Restrictions a source states about itself. Each is (key, human sentence, pattern).
# The human sentence is what a report should tell the reader; it never asserts anything the
# source did not, and it says "generally" where the rule has exceptions.
RESTRICTIONS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    (
        "nsf_funded",
        "This is funded by the National Science Foundation, and NSF programmes are generally "
        "open only to US citizens and permanent residents.",
        re.compile(r"\b(National Science Foundation|\bNSF\b)\b", re.I),
    ),
    (
        "us_citizen_only",
        "The source states this is limited to US citizens or permanent residents.",
        re.compile(r"\b(U\.?S\.?\s+citizens?|United States citizens?|permanent residents?)\b", re.I),
    ),
    (
        "pre_college",
        "This is a pre-college programme for high-school students, not something an enrolled "
        "undergraduate takes.",
        re.compile(r"\b(pre-?college|high[- ]school (?:students?|programme?s?|summer)"
                   r"|summer programs? for high[- ]school)\b", re.I),
    ),
    (
        "graduate_only",
        "The source describes this as being for graduate students.",
        re.compile(r"\b(graduate students only|open (?:only )?to graduate students"
                   r"|master'?s or doctoral students|PhD students only)\b", re.I),
    ),
    (
        "transfer_only",
        "The source describes this as being for transfer students.",
        re.compile(r"\b(transfer students only|open (?:only )?to transfer students)\b", re.I),
    ),
    (
        "by_application",
        "Places are limited and selected by application, so this is not open to everyone "
        "who wants it.",
        re.compile(r"\b(selected candidates|competitive (?:selection|application)|limited "
                   r"(?:spots|places|number of)|by application only|apply by)\b", re.I),
    ),
)

# Which restrictions make a unit simply WRONG for an applicant at a given level, as opposed to
# merely worth mentioning. Only life stage is filtered; nothing here guesses at nationality.
WRONG_FOR_LEVEL: dict[str, tuple[str, ...]] = {
    "undergraduate": ("pre_college", "graduate_only"),
    "graduate": ("pre_college",),
}


def unit_text(unit: dict[str, Any]) -> str:
    return " ".join(
        str(unit.get(k) or "")
        for k in ("text", "quote", "entity_name", "source_title", "source_url")
    )


def restrictions_for(unit: dict[str, Any]) -> list[dict[str, str]]:
    """Every restriction this unit's own text states, as {key, sentence, matched}."""
    blob = unit_text(unit)
    out: list[dict[str, str]] = []
    for key, sentence, pattern in RESTRICTIONS:
        m = pattern.search(blob)
        if m:
            out.append({"key": key, "sentence": sentence, "matched": m.group(0)})
    return out


def wrong_for_level(unit: dict[str, Any], level: str) -> str | None:
    """The restriction key that makes this unit wrong for an applicant at `level`, if any."""
    bad = WRONG_FOR_LEVEL.get((level or "undergraduate").lower(), ())
    if not bad:
        return None
    for r in restrictions_for(unit):
        if r["key"] in bad:
            return r["key"]
    return None


def annotate(units: Sequence[dict[str, Any]], level: str = "undergraduate"
             ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Tag units with their stated restrictions and drop the ones wrong for this level.

    Returns (kept units, stats). A dropped unit is recorded with the reason, because a
    category that quietly shrinks is the failure mode this pipeline exists to avoid.
    """
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, str]] = []
    tagged = 0
    by_key: dict[str, int] = {}
    for unit in units:
        bad = wrong_for_level(unit, level)
        if bad:
            dropped.append({"unit_id": str(unit.get("unit_id") or ""),
                            "entity": str(unit.get("entity_name") or ""),
                            "reason": bad})
            by_key[bad] = by_key.get(bad, 0) + 1
            continue
        found = restrictions_for(unit)
        if found:
            unit = dict(unit)
            unit["eligibility"] = found
            tagged += 1
            for r in found:
                by_key[r["key"]] = by_key.get(r["key"], 0) + 1
        kept.append(unit)
    return kept, {"level": level, "units_in": len(units), "units_kept": len(kept),
                  "dropped_wrong_level": dropped, "tagged": tagged, "by_restriction": by_key}


def annotate_buckets(buckets: dict[str, list[dict[str, Any]]], level: str = "undergraduate"
                     ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    out: dict[str, list[dict[str, Any]]] = {}
    stats: dict[str, Any] = {"level": level, "by_category": {}, "dropped_total": 0,
                             "tagged_total": 0}
    for code, units in buckets.items():
        if not isinstance(units, list):
            out[code] = units
            continue
        kept, s = annotate(units, level)
        out[code] = kept
        stats["by_category"][code] = {"kept": len(kept), "dropped": len(s["dropped_wrong_level"]),
                                      "tagged": s["tagged"]}
        stats["dropped_total"] += len(s["dropped_wrong_level"])
        stats["tagged_total"] += s["tagged"]
    return out, stats


def prompt_block(units: Iterable[dict[str, Any]]) -> str:
    """The instruction a writer needs when any supplied unit carries a restriction."""
    seen: dict[str, str] = {}
    for unit in units:
        for r in unit.get("eligibility") or []:
            seen.setdefault(r["key"], r["sentence"])
    if not seen:
        return ""
    lines = ["ELIGIBILITY -- some evidence below carries a restriction on WHO MAY DO IT.",
             "If you write about one of those units, the caveat MUST state the restriction.",
             "A reader who is not eligible has to learn that from you, not after applying.",
             "The restrictions present in this evidence:"]
    lines += [f"  - {s}" for s in seen.values()]
    return "\n".join(lines)
