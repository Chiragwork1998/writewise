"""Regression tests for wwrag/report.py -- the last stage, the one the student actually sees.

Every test here is named for the defect it prevents coming back. Two themes run through
them:

  * report.py is the LAST stage. Anything it refuses to render costs a full run, including
    both paid model stages, so it must refuse only what it genuinely cannot draw.
  * the output is a document a family reads. Stray superscripts, placeholder names and
    headings that end in a full stop are not cosmetic here; they are the whole product.

Offline: no model, no Chrome, no index. Run:
  /Users/chirag/college-intel/.venv-crawl4ai/bin/python -m pytest \
      /Users/chirag/college-intel/wwrag/tests/test_report_document.py -q
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import report  # noqa: E402


# --------------------------------------------------------------------------------------
# fixtures: the smallest thing that is still a real document
# --------------------------------------------------------------------------------------

UNITS = {
    "u-club": {
        "unit_id": "u-club",
        "source_url": "https://example.edu/clubs/robotics",
        "source_title": "Robotics Society",
        "source_kind": "official",
        "year": 2025,
    },
    "u-res": {
        "unit_id": "u-res",
        "source_url": "https://example.edu/research/undergrad",
        "source_title": "Undergraduate Research",
        "source_kind": "official",
    },
    "u-news": {
        "unit_id": "u-news",
        "source_url": "https://news.example.com/story",
        "source_title": "A campus story",
        "source_kind": "external",
    },
}

RESUME_LINE = "Robotics Club - Team captain (2 years). Built the drivetrain."

PROFILE_NAMED = {
    "student_id": "stu_test",
    "first_name": "Anika",
    "level": "undergraduate",
    "intended_fields": ["Electrical Engineering"],
    "activities": [{"name": "Robotics Club", "evidence_line": RESUME_LINE}],
    "projects": [],
    "achievements": [],
}

# profile.py documents first_name as `"" if absent` and flags it instead of failing, so a
# profile in exactly this shape is a supported input, not a corrupt file.
PROFILE_UNNAMED = dict(PROFILE_NAMED, first_name="")

ITEMS = [
    {
        "category_code": "EXT",
        "headline": "The Robotics Society runs undergraduate build teams.",
        "body": "The society lists undergraduate build teams [u-club].",
        "why_it_matters": "It lines up with the team you already captained.",
        "profile_basis": [RESUME_LINE],
        "evidence_ids": ["u-club"],
        "caveat": None,
    },
    {
        "category_code": "RES",
        "headline": "Undergraduate research placements",
        "body": "Undergraduates may apply to faculty projects [u-res].",
        "why_it_matters": "",
        "profile_basis": [],
        "evidence_ids": ["u-res"],
        "caveat": None,
    },
]

CATEGORIES = [("EXT", "Extracurriculars"), ("RES", "Research"), ("DIV", "Diversity of Community")]


def fresh_stats():
    """The same stats dict main() builds, so tests exercise the real code path."""
    return {"items_in": len(ITEMS), "items_rendered": 0,
            "items_dropped_no_evidence": 0, "items_dropped_empty": 0,
            "items_dropped_unknown_code": 0, "items_dropped_context_only": 0,
            "dangling_evidence_ids": 0, "profile_basis_dropped": 0,
            "citations": 0, "sources": 0,
            "categories_rendered": [], "categories_empty": [], "fit_summary": ""}


@pytest.fixture(autouse=True)
def _clean_module_globals():
    """report.py keeps LOG and STRICT at module scope; main() resets them, direct calls do not."""
    report.STRICT = False
    del report.LOG[:]
    yield
    report.STRICT = False
    del report.LOG[:]


def write_run(tmp_path, profile, items=ITEMS, meta=None):
    """Lay out the three input files main() expects and return them."""
    report_blob = {"items": items}
    report_blob.update(meta or {})
    (tmp_path / "report_items.json").write_text(json.dumps(report_blob), encoding="utf-8")
    (tmp_path / "evidence.json").write_text(
        json.dumps({"units": list(UNITS.values())}), encoding="utf-8")
    (tmp_path / "profile.json").write_text(json.dumps(profile), encoding="utf-8")
    return [
        "--report", str(tmp_path / "report_items.json"),
        "--evidence", str(tmp_path / "evidence.json"),
        "--profile", str(tmp_path / "profile.json"),
        "--out-dir", str(tmp_path / "out"),
        "--college-name", "Example University",
        "--as-of", "2026-01-01",
        "--no-pdf",
    ]


# --------------------------------------------------------------------------------------
# fix 1: a missing first name must not destroy a paid run at the last stage
# --------------------------------------------------------------------------------------

def test_profile_without_a_first_name_is_accepted_not_fatal(tmp_path):
    """profile.py may legitimately leave first_name empty.

    This used to raise, which killed the run at the LAST stage -- after the profile and
    generate model calls had already been billed -- over a field the document does not
    need. If this ever raises again, every resume whose given name we cannot split costs
    a full run.
    """
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(PROFILE_UNNAMED), encoding="utf-8")
    prof = report.load_profile(path)
    assert prof["student_id"] == "stu_test"
    assert any("no first name" in n for n in report.LOG), \
        "the gap must still be recorded, just not fatal"


def test_profile_without_a_student_id_or_level_is_still_fatal(tmp_path):
    """Relaxing first_name must not relax the two fields we truly cannot render without."""
    for missing in ("student_id", "level"):
        broken = {k: v for k, v in PROFILE_NAMED.items() if k != missing}
        path = tmp_path / f"profile_{missing}.json"
        path.write_text(json.dumps(broken), encoding="utf-8")
        with pytest.raises(ValueError, match=missing):
            report.load_profile(path)


def test_missing_first_name_does_not_fail_even_under_strict(tmp_path):
    """--strict escalates integrity notes into failures.

    A resume that never printed a given name is a gap in the INPUT, not a defect in our
    own work, so it must not be routed through note() -- doing so would reinstate the
    "dies at the last stage" bug for every strict run.
    """
    report.STRICT = True
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(PROFILE_UNNAMED), encoding="utf-8")
    assert report.load_profile(path)["level"] == "undergraduate"


def test_document_never_addresses_the_student_as_a_placeholder(tmp_path):
    """Without a name the title is just the college and the prose switches to "your file".

    "Example University for this student" is a failed mail merge, and that is the first
    thing a paying family notices.
    """
    md = report.build_markdown(ITEMS, {}, UNITS, PROFILE_UNNAMED, "Example University",
                               CATEGORIES, "2026-01-01", fresh_stats())
    assert md.splitlines()[0] == "# Example University"
    assert not md.splitlines()[0].rstrip().endswith("for")
    # the title block and the fit summary are the two places a name would be substituted
    head = md.split("## 01 ·", 1)[0]
    assert "this student" not in head
    assert "’s file" not in head
    assert "reads your file" in head

    named = report.build_markdown(ITEMS, {}, UNITS, PROFILE_NAMED, "Example University",
                                  CATEGORIES, "2026-01-01", fresh_stats())
    assert named.splitlines()[0] == "# Example University for Anika"


def test_unnamed_run_writes_a_document_with_no_dangling_for_in_its_title(tmp_path):
    """End to end through main(): the HTML <title> must not end in "for "."""
    assert report.main(write_run(tmp_path, PROFILE_UNNAMED)) == 0
    html = (tmp_path / "out" / "report.html").read_text(encoding="utf-8")
    title = re.search(r"<title>(.*?)</title>", html, re.S).group(1)
    assert title == "Example University"
    assert not title.rstrip().endswith("for")


# --------------------------------------------------------------------------------------
# presentation: what a student sees on the page
# --------------------------------------------------------------------------------------

def test_citation_superscripts_sit_tight_against_the_word_they_follow(tmp_path):
    """The generator writes "... projects [u-club], the ...". The space before the marker
    printed as a visible gap between a word and its raised number, which is the single
    clearest tell that a page was assembled rather than typeset."""
    assert report.main(write_run(tmp_path, PROFILE_NAMED)) == 0
    md = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    html = (tmp_path / "out" / "report.html").read_text(encoding="utf-8")
    assert ' <sup class="c">' not in md
    assert ' <sup class="c">' not in html
    assert '<sup class="c">' in md, "the fixture must actually produce citations"


def test_every_citation_superscript_resolves_to_a_source_note(tmp_path):
    """A raised number that links nowhere is worse than no number at all: it claims a
    source the reader cannot check."""
    assert report.main(write_run(tmp_path, PROFILE_NAMED)) == 0
    html = (tmp_path / "out" / "report.html").read_text(encoding="utf-8")
    anchors = set(re.findall(r'id="([^"]+)"', html))
    refs = set(re.findall(r'href="#([^"]+)"', html))
    assert refs, "the fixture must actually produce citations"
    assert refs <= anchors, f"unresolved citation targets: {sorted(refs - anchors)}"


def test_headings_do_not_end_in_a_full_stop(tmp_path):
    """The generator sometimes hands us a whole sentence as a headline. A heading that
    ends in a period reads as prose pasted into a title slot -- on every page."""
    assert report.heading_text("Intro to EE covers circuits, signals, and more.") \
        == "Intro to EE covers circuits, signals, and more"
    # ... and abbreviations, ellipses and real heading punctuation survive untouched
    assert report.heading_text("A programme in the U.S.") == "A programme in the U.S."
    assert report.heading_text("More to come...") == "More to come..."
    assert report.heading_text("Why here?") == "Why here?"

    assert report.main(write_run(tmp_path, PROFILE_NAMED)) == 0
    md = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    headings = [ln.lstrip("# ").strip() for ln in md.splitlines() if ln.startswith("###")]
    assert headings
    for h in headings:
        assert not (h.endswith(".") and not h.endswith("..")), f"heading ends in a period: {h!r}"


def test_summary_does_not_staple_bullet_sources_onto_the_opening_paragraph(tmp_path):
    """The summary's evidence_ids are the union of every surviving summary sentence's
    sources (run.py carries them because the verifier sometimes drops an inline marker).

    Working out what is still unattributed BEFORE the bullets resolve dumped every
    bullet's source onto the end of the opening paragraph as well, so the reader saw the
    same number twice: once in a superscript pile after the paragraph and again on a
    bullet two inches below. Only genuinely orphaned ids may trail the paragraph.
    """
    meta = {"fit_summary": {
        "fit_summary": "There are build teams here [u-club].",
        "strongest_matches": ["Undergraduate research placements [u-res]"],
        "open_questions": [],
        # u-news is cited by nothing inline; u-res belongs to the bullet, not the paragraph
        "evidence_ids": ["u-club", "u-res", "u-news"],
    }}
    md = report.build_markdown(ITEMS, meta, UNITS, PROFILE_NAMED, "Example University",
                               CATEGORIES, "2026-01-01", fresh_stats())
    paragraph = next(ln for ln in md.splitlines() if ln.startswith("There are build teams"))
    bullet = next(ln for ln in md.splitlines() if ln.startswith("- Undergraduate research"))
    para_nums = set(re.findall(r'href="#c00-s(\d+)"', paragraph))
    bullet_nums = set(re.findall(r'href="#c00-s(\d+)"', bullet))
    assert bullet_nums, "the bullet must carry its own citation"
    assert not (para_nums & bullet_nums), \
        f"the same source is printed on the paragraph and on the bullet: {para_nums & bullet_nums}"
    # the genuinely orphaned id is still shown -- dropping it would hide a source
    assert len(para_nums) == 2, f"expected the paragraph's own id plus the orphan, got {para_nums}"


def test_an_empty_category_says_so_and_is_given_no_source_list(tmp_path):
    """An empty chapter must read as an honest finding, never as a rendered-but-blank
    section with a "Sources for this chapter" heading under it."""
    stats = fresh_stats()
    md = report.build_markdown(ITEMS, {}, UNITS, PROFILE_NAMED, "Example University",
                               CATEGORIES, "2026-01-01", stats)
    assert stats["categories_empty"] == ["DIV"]
    tail = md.split("## 03 · Diversity of Community", 1)[1].split("---", 1)[0]
    assert "Not enough verified evidence" in tail
    assert "Sources for this chapter" not in tail
    assert "1." not in tail, "an empty chapter must not print a numbered source list"


# --------------------------------------------------------------------------------------
# fix 3: the page budget is an advisory range, not a licence to cut
# --------------------------------------------------------------------------------------

def test_page_budget_ceiling_fits_a_full_ten_chapter_report():
    """The old ceiling of 14 was set before the ten-chapter shape carried per-chapter
    source notes. A real run (36 items, 46 sources) is 17 pages at the top of the type
    ladder, so a ceiling below that makes the ladder grind the body type down to the 9.5pt
    floor chasing a number it can never reach -- and then still report false.

    The ceiling must clear a full report, and the floor must stay meaningful.
    """
    lo, hi = report.CONFIG["page_budget"]
    assert hi >= 17, "the ceiling must clear a full ten-chapter report at full body size"
    assert lo >= 1 and lo < hi


def test_a_full_report_renders_once_at_the_most_legible_body_size():
    """Nothing in the ladder may fire for a report the budget is meant to accommodate.

    Shrinking type to make page_budget_ok go green is the same dishonesty as truncating,
    just harder to see -- so the first, largest size on the ladder must be inside budget.
    """
    lo, hi = report.CONFIG["page_budget"]
    assert report.CONFIG["body_pt_ladder"][0] >= report.CONFIG["min_body_pt"]
    # a 17-page render at the top of the ladder must be accepted without tightening
    assert lo <= 17 <= hi


def test_build_summary_states_the_budget_policy_and_any_type_tightening(tmp_path):
    """page_budget_ok on its own cannot distinguish "fitted at full size" from "fitted only
    after shrinking the type", and a false there must not read as "the report is broken"."""
    assert report.main(write_run(tmp_path, PROFILE_NAMED)) == 0
    build = json.loads((tmp_path / "out" / "report_build.json").read_text(encoding="utf-8"))
    assert build["page_budget"] == report.CONFIG["page_budget"]
    assert "truncat" in build["page_budget_policy"]
    assert build["body_pt_tightened"] is False
    assert build["body_pt_full"] == report.CONFIG["body_pt_ladder"][0]


# --------------------------------------------------------------------------------------
# the product ships 45 colleges on this one file
# --------------------------------------------------------------------------------------

def test_no_college_specific_literals_in_report_py():
    """45 colleges ship on this module. A single leaked name makes 44 reports wrong."""
    source = Path(report.__file__).read_text(encoding="utf-8")
    body = source.split('"""', 2)[2]          # skip the module docstring's example paths
    for literal in ("usc", "USC", "Southern California", "Trojan"):
        assert literal not in body, f"college-specific literal {literal!r} leaked into the logic"
