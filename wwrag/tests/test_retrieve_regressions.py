"""Regression tests for wwrag/retrieve.py -- one per defect that shipped and was fixed.

Each test is named for the thing it protects, and each one FAILED against the build that had the
defect. They are deliberately blunt: most assert on the product harm (what came back for a
student) rather than on an implementation detail, because the implementation is allowed to change
and the harm is not.

The tiny-index fixtures come from test_retrieve.py, which owns them; this file only adds cases.

Run:
  /Users/chirag/college-intel/.venv-crawl4ai/bin/python -m pytest \
      /Users/chirag/college-intel/wwrag/tests/test_retrieve_regressions.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import re  # noqa: E402
import retrieve  # noqa: E402
from test_retrieve import StubEmbedder, profile, unit, write_index  # noqa: E402


# --------------------------------------------------------------------------------------
# 1. the undergraduate course gate (HIGH) -- flag spellings, and the silence
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "flag",
    [True, 1, 1.0, "1", "true", "True", "TRUE", "yes", "Y", "undergraduate", "bachelor"],
)
def test_undergraduate_course_flag_is_read_however_the_bundle_spells_it(tmp_path, flag):
    """45 colleges, 45 bundles, and no agreement on how to write a boolean.

    Reading only (1, True, "1", "true") meant any other spelling read as "not undergraduate",
    which drops EVERY course for an undergraduate applicant -- the entire course pool, with the
    course kind floors silently unmet.
    """
    units = [
        unit(
            f"alpha-course-{i}",
            "course",
            f"CSCI 2{i}0: Undergraduate seminar in computer science, four units, open to all "
            f"students in the college of letters arts and sciences.",
            entity_name=f"CSCI 2{i}0",
            source_url=f"https://example.edu/classes/csci2{i}0",
            extra={"is_undergraduate": flag, "dept": "CSCI", "number": f"2{i}0"},
            year=2026,
        )
        for i in range(3)
    ]
    index = retrieve.CollegeIndex(write_index(tmp_path / "index", "alpha", units))
    try:
        assert int(index.is_undergrad_course.sum()) == 3, f"flag {flag!r} was not understood"
        rows = retrieve.candidate_rows(index, retrieve.CATEGORY_BY_CODE["ACA"], undergraduate=True)
        assert len(rows) == 3, "the undergraduate gate dropped courses that ARE undergraduate"
    finally:
        index.close()


def test_a_graduate_flag_still_keeps_the_course_out(tmp_path):
    """The other half of the same fix: a readable false must stay false."""
    units = [
        unit(
            "alpha-course-grad",
            "course",
            "CSCI 555: Advanced systems seminar, four units, for students in the department.",
            entity_name="CSCI 555",
            source_url="https://example.edu/classes/csci555",
            extra={"is_undergraduate": "False", "dept": "CSCI"},
            year=2026,
        ),
        unit(
            "alpha-course-ug",
            "course",
            "CSCI 104: Data structures, four units, open to all students in the college.",
            entity_name="CSCI 104",
            source_url="https://example.edu/classes/csci104",
            extra={"is_undergraduate": "yes", "dept": "CSCI"},
            year=2026,
        ),
    ]
    index = retrieve.CollegeIndex(write_index(tmp_path / "index", "alpha", units))
    try:
        rows = retrieve.candidate_rows(index, retrieve.CATEGORY_BY_CODE["ACA"], undergraduate=True)
        assert [index.unit_id[r] for r in rows] == ["alpha-course-ug"]
    finally:
        index.close()


def test_emptying_the_whole_course_pool_is_loud_not_silent(tmp_path, capsys):
    """Silence WAS the bug. A category that loses its entire pool has to say so.

    A bundle whose is_undergraduate value we cannot read loses every course at the gate. That is
    survivable; doing it without a word is not, because the report just comes back as prose and
    nobody downstream can tell the difference between "this college has no courses" and "we could
    not read this bundle".
    """
    units = [
        unit(
            f"alpha-course-{i}",
            "course",
            f"CSCI 3{i}0: Seminar in computer science, four units, open to students in the college.",
            entity_name=f"CSCI 3{i}0",
            source_url=f"https://example.edu/classes/csci3{i}0",
            extra={"is_undergraduate": "level-01-unknown", "dept": "CSCI"},
            year=2026,
        )
        for i in range(4)
    ]
    index_dir = write_index(tmp_path / "index", "alpha", units)
    capsys.readouterr()
    index = retrieve.CollegeIndex(index_dir)
    try:
        opened = capsys.readouterr().out
        assert "WARNING" in opened and "NOT ONE" in opened, opened
        assert index.unreadable_course_flags == 4

        warnings: list[dict] = []
        rows = retrieve.candidate_rows(
            index, retrieve.CATEGORY_BY_CODE["ACA"], undergraduate=True, warnings=warnings
        )
        said = capsys.readouterr().out
        assert rows.size == 0
        assert "WARNING" in said and "removed ALL" in said, said
        # the floor is whatever the CATEGORIES table says (now a share of the section); the
        # warning must name it, whatever it is, rather than a number frozen in this test
        aca = retrieve.CATEGORY_BY_CODE["ACA"]
        want = f"floor of {float(aca['kind_floor']['course'])}"
        assert want in said, f"the unmet course floor was not named: wanted {want!r}"
        assert warnings and warnings[0]["category_code"] == "ACA"
        assert warnings[0]["candidates_before_gate"] == 4
    finally:
        index.close()


def test_the_course_gate_stays_quiet_when_it_is_merely_doing_its_job(tmp_path, capsys):
    """Loud on a lost pool, silent on a normal drop -- or the warning becomes noise."""
    units = [
        unit(
            "alpha-course-ug",
            "course",
            "CSCI 104: Data structures, four units, open to all students in the college.",
            entity_name="CSCI 104",
            source_url="https://example.edu/classes/csci104",
            extra={"is_undergraduate": True, "dept": "CSCI"},
        ),
        unit(
            "alpha-course-grad",
            "course",
            "CSCI 555: Advanced systems seminar, four units, for students in the department.",
            entity_name="CSCI 555",
            source_url="https://example.edu/classes/csci555",
            extra={"is_undergraduate": False, "dept": "CSCI"},
        ),
    ]
    index = retrieve.CollegeIndex(write_index(tmp_path / "index", "alpha", units))
    try:
        capsys.readouterr()
        retrieve.candidate_rows(index, retrieve.CATEGORY_BY_CODE["ACA"], undergraduate=True)
        assert "WARNING" not in capsys.readouterr().out
    finally:
        index.close()


# --------------------------------------------------------------------------------------
# 2. the own-code quota (HIGH) -- a thin category's own material must reach the scorer
# --------------------------------------------------------------------------------------


def thin_category_index(tmp_path) -> Path:
    """One thin category (QRK: 3 own facts) drowned in its support code (200 GEN facts).

    The GEN facts carry every word the QRK queries use, so an unquota'd top-80 is all GEN and the
    three facts the category is actually about are never scored at all.
    """
    units = [
        unit(
            f"alpha-fact-gen-{i}",
            "fact",
            f"General fact {i}: unusual traditions, odd rituals and surprising campus lore, quirky "
            f"and offbeat student clubs for an undergraduate interested in assistive technology "
            f"and renewable energy on this campus.",
            category_code="GEN",
            entity_name=f"General {i}",
            source_url=f"https://example.edu/general/{i}",
        )
        for i in range(200)
    ]
    units += [
        unit(
            f"alpha-fact-qrk-{i}",
            "fact",
            f"Fnorble Night {i}: every midwinter, first-years hurl tregs at the wibbit tower until "
            f"a mork answers back, a ritual nobody outside this place has heard of.",
            category_code="QRK",
            entity_name=f"Fnorble {i}",
            source_url=f"https://lore{i}.example.org/fnorble",
        )
        for i in range(3)
    ]
    return write_index(tmp_path / "index", "alpha", units)


def test_thin_category_own_fact_codes_are_not_swamped_by_support_codes(tmp_path, monkeypatch):
    """The category's OWN material must be in the scored pool, not just allowed into it.

    max_support_fraction runs at selection time and can only drop support rows that got in; it
    cannot put back own-code rows that never made any query's top-k. Without a quota reserved
    BEFORE scoring, this student's Quirks section is three general facts about the campus and not
    one of the three quirks the college actually has.
    """
    index = retrieve.CollegeIndex(thin_category_index(tmp_path))
    try:
        # the arm without the quota: exactly the old behaviour, and it loses every own-code fact
        monkeypatch.setitem(retrieve.CONFIG, "own_code_min_fraction", 0.0)
        before, _ = retrieve.retrieve(index, profile(), per_category=8, embedder=StubEmbedder())
        assert [u for u in before["QRK"] if u["category_code"] == "QRK"] == []

        monkeypatch.setitem(retrieve.CONFIG, "own_code_min_fraction", 0.5)
        after, explain = retrieve.retrieve(index, profile(), per_category=8, embedder=StubEmbedder())
        own = [u for u in after["QRK"] if u["category_code"] == "QRK"]
        assert len(own) == 3, f"the category's own facts are still missing: {after['QRK']}"

        quota = explain["categories"]["QRK"]["own_code_quota"]
        assert quota["second_pass"] is True
        assert quota["own_rows"] == 3 and quota["pool_rows"] == 203
    finally:
        index.close()


def test_the_own_code_quota_does_not_run_when_it_is_not_needed(tmp_path):
    """It is a floor, not a tax: a category whose own material already dominates pays nothing."""
    units = [
        unit(
            f"alpha-fact-cul-{i}",
            "fact",
            f"Campus tradition {i}: students gather for a ritual about campus culture and "
            f"traditions and student life every single year without fail.",
            category_code="CUL",
            entity_name=f"Tradition {i}",
            source_url=f"https://example.edu/tradition/{i}",
        )
        for i in range(12)
    ]
    index = retrieve.CollegeIndex(write_index(tmp_path / "index", "alpha", units))
    try:
        _, explain = retrieve.retrieve(index, profile(), per_category=6, embedder=StubEmbedder())
        quota = explain["categories"]["CUL"]["own_code_quota"]
        assert quota["second_pass"] is False
        assert quota["queries_short"] == 0
    finally:
        index.close()


def test_merge_with_own_quota_reserves_own_rows_without_reordering_the_rest():
    """The merge is by score, so the ranks handed to RRF still mean what they say."""
    main = [(10, 0.9), (11, 0.8), (12, 0.7)]
    own = [(20, 0.75), (21, 0.4)]
    merged = retrieve.merge_with_own_quota(
        main, own, own_set={20, 21}, top_k=3, min_own=2, descending=True
    )
    assert merged == [(10, 0.9), (20, 0.75), (21, 0.4)], merged
    assert [score for _, score in merged] == sorted((s for _, s in merged), reverse=True)

    # bm25 runs the other way: smaller is better
    merged = retrieve.merge_with_own_quota(
        [(10, -9.0), (11, -8.0)], [(20, -5.0)], own_set={20}, top_k=2, min_own=1, descending=False
    )
    assert merged == [(10, -9.0), (20, -5.0)], merged


# --------------------------------------------------------------------------------------
# 3. the diversity caps (MEDIUM) -- one site, one bucket; one page, one bucket
# --------------------------------------------------------------------------------------


def test_one_site_gets_one_host_cap_bucket_not_two(tmp_path):
    """Facts carry extra.source_host (no "www."), everything else fell back to host_of(url)
    (keeps it), so one site held two cap buckets and quietly supplied 2 x max_per_host."""
    units = [
        unit(
            f"alpha-fact-cul-{i}",
            "fact",
            f"Campus culture fact {i}: students keep a tradition about campus culture and "
            f"traditions and student life on this campus every year.",
            category_code="CUL",
            entity_name=f"Tradition {i}",
            source_url=f"https://www.example.edu/traditions/{i}",
            extra={"source_host": "example.edu"},
        )
        for i in range(6)
    ]
    units += [
        unit(
            f"alpha-chunk-cul-{i}",
            "chunk",
            f"Page {i} about campus culture and traditions and student life, describing how "
            f"students spend their time on this campus during the year.",
            source_url=f"https://www.example.edu/life/{i}",
            extra={"page_id": f"alpha-page-life-{i}"},
        )
        for i in range(6)
    ]
    index = retrieve.CollegeIndex(write_index(tmp_path / "index", "alpha", units))
    try:
        hosts = {index.source_host[i] for i in range(index.n_units)}
        assert hosts == {"example.edu"}, f"one site, two cap buckets: {hosts}"

        results, _ = retrieve.retrieve(index, profile(), per_category=8, embedder=StubEmbedder())
        from_site = [u for u in results["CUL"] if "example.edu" in u["source_url"]]
        assert len(from_site) <= retrieve.CONFIG["max_per_host"], (
            f"{len(from_site)} units from one site, cap is {retrieve.CONFIG['max_per_host']}"
        )
    finally:
        index.close()


def test_normalise_host_collapses_www_and_ports():
    assert retrieve.normalise_host("www.Example.edu") == "example.edu"
    assert retrieve.normalise_host("example.edu:443") == "example.edu"
    assert retrieve.normalise_host("news.example.edu") == "news.example.edu"
    assert retrieve.normalise_host("") == ""


def test_one_page_gets_one_page_cap_bucket_not_two(tmp_path):
    """Facts key on the URL, chunks keyed on extra.page_id: the same page paid twice.

    Four passages off one page read as four findings in the report and are one.
    """
    page = "https://example.edu/student-life/traditions"
    units = [
        unit(
            f"alpha-fact-cul-{i}",
            "fact",
            f"Distinct campus culture fact {i} about traditions and student life, describing a "
            f"different ritual that the students hold during the academic year.",
            category_code="CUL",
            entity_name=f"Tradition {i}",
            source_url=page,
        )
        for i in range(3)
    ]
    units += [
        unit(
            f"alpha-chunk-cul-{i}",
            "chunk",
            f"Passage {i} of the same page about campus culture and traditions and student life, "
            f"with wording that differs from every other passage on it.",
            source_url=page,
            extra={"page_id": "alpha-page-traditions"},
        )
        for i in range(3)
    ]
    index = retrieve.CollegeIndex(write_index(tmp_path / "index", "alpha", units))
    try:
        keys = {index.page_key[i] for i in range(index.n_units)}
        assert len(keys) == 1, f"one page, {len(keys)} cap buckets: {keys}"

        results, _ = retrieve.retrieve(index, profile(), per_category=6, embedder=StubEmbedder())
        assert len(results["CUL"]) <= retrieve.CONFIG["max_per_source_page"], results["CUL"]
    finally:
        index.close()


def test_canonical_page_url_ignores_scheme_www_and_a_trailing_slash():
    same = {
        retrieve.canonical_page_url("https://www.example.edu/a/b/"),
        retrieve.canonical_page_url("http://example.edu/a/b"),
        retrieve.canonical_page_url("https://EXAMPLE.edu/a/b#section"),
    }
    assert len(same) == 1, same
    # pages that differ by query string really are different pages
    assert retrieve.canonical_page_url("https://example.edu/a?p=2") != retrieve.canonical_page_url(
        "https://example.edu/a?p=3"
    )


# --------------------------------------------------------------------------------------
# 4. the graduate gate and typography (MEDIUM)
# --------------------------------------------------------------------------------------


def test_graduate_gate_sees_typographic_apostrophes(tmp_path):
    """The markers are ASCII-apostrophe and crawled prose is not.

    A page that writes "master’s programme" with real typography sailed straight through the
    gate and was offerable to a 16-year-old. In the first shipped index that was 121 rows.
    """
    units = [
        unit(
            "alpha-fact-curly",
            "fact",
            "The department offers a Master’s programme in data science, taught over two "
            "years, with evening classes for working professionals.",
            category_code="ACA",
            entity_name="MS Data Science",
            source_url="https://example.edu/programs/ms-curly",
        ),
        unit(
            "alpha-fact-ascii",
            "fact",
            "The department offers a Master's programme in applied physics, taught over two "
            "years, with evening classes for working professionals.",
            category_code="ACA",
            entity_name="MS Applied Physics",
            source_url="https://example.edu/programs/ms-ascii",
        ),
        unit(
            "alpha-fact-keep",
            "fact",
            "Undergraduates take small seminars taught by tenured faculty in computer science and "
            "may join a laboratory in their second year.",
            category_code="ACA",
            entity_name="Undergraduate teaching",
            source_url="https://example.edu/academics",
        ),
    ]
    index = retrieve.CollegeIndex(write_index(tmp_path / "index", "alpha", units))
    try:
        curly = index.unit_id.index("alpha-fact-curly")
        ascii_row = index.unit_id.index("alpha-fact-ascii")
        assert index.grad_only[ascii_row] == 1, "fixture no longer proves anything"
        assert index.grad_only[curly] == 1, "a typographic apostrophe walked past the graduate gate"

        rows = set(
            retrieve.candidate_rows(
                index, retrieve.CATEGORY_BY_CODE["ACA"], undergraduate=True
            ).tolist()
        )
        assert curly not in rows and ascii_row not in rows
        assert index.unit_id.index("alpha-fact-keep") in rows
    finally:
        index.close()


def test_apostrophe_fold_does_not_swallow_undergraduate_material(tmp_path):
    """Conservative as documented: graduate-looking AND nothing undergraduate."""
    units = [
        unit(
            "alpha-fact-rescued",
            "fact",
            "Undergraduates in the progressive degree program begin a Master’s degree in "
            "their senior year while finishing the bachelor of science.",
            category_code="ACA",
            entity_name="Progressive degree",
            source_url="https://example.edu/programs/progressive",
        )
    ]
    index = retrieve.CollegeIndex(write_index(tmp_path / "index", "alpha", units))
    try:
        assert index.grad_only[0] == 0, "the undergraduate rescue markers stopped working"
    finally:
        index.close()


# --------------------------------------------------------------------------------------
# 5. organisation fit (MEDIUM)
# --------------------------------------------------------------------------------------


def zero_fit_index(tmp_path) -> Path:
    """A fit-keyed category (QRK / "quirky") whose only organisations score 0 on that key."""
    units = [
        unit(
            f"alpha-org-{i}",
            "org",
            f"Engineering Society {i} (Recognized Student Organization). Unusual traditions, odd "
            f"rituals and surprising campus lore, quirky and offbeat student clubs.",
            entity_name=f"Engineering Society {i}",
            source_url=f"https://soc{i}.example.org/about",
            extra={"fit_scores": {"quirky": 0, "research": 3}},
        )
        for i in range(4)
    ]
    units += [
        unit(
            f"alpha-fact-qrk-{i}",
            "fact",
            f"Campus lore {i}: an odd ritual nobody can explain, one of the unusual traditions "
            f"and surprising campus legends students keep up every year.",
            category_code="QRK",
            entity_name=f"Lore {i}",
            source_url=f"https://example.edu/lore/{i}",
        )
        for i in range(6)
    ]
    return write_index(tmp_path / "index", "alpha", units)


def test_zero_fit_organisations_are_penalised_not_treated_as_neutral(tmp_path):
    """A club scoring 0 on the category's fit key is the WRONG club, not weak evidence."""
    index = retrieve.CollegeIndex(zero_fit_index(tmp_path))
    try:
        cat = retrieve.CATEGORY_BY_CODE["QRK"]
        org_row = index.unit_id.index("alpha-org-0")
        hit = retrieve.Hit(org_row)
        hit.rrf = 1.0
        scored = retrieve.rescore(index, cat, {org_row: hit})
        assert scored[org_row]["boosts"]["org_fit"] < 1.0, scored[org_row]["boosts"]
    finally:
        index.close()


def test_the_kind_floor_never_injects_a_zero_fit_organisation(tmp_path, capsys):
    """The floor stops a category coming back as twelve prose facts; it does not promise clubs
    at any price. Force-injecting the top-ranked org regardless of fit is how a quirky-clubs
    section filled up with ordinary engineering clubs."""
    index = retrieve.CollegeIndex(zero_fit_index(tmp_path))
    try:
        results, _ = retrieve.retrieve(index, profile(), per_category=6, embedder=StubEmbedder())
        orgs = [u for u in results["QRK"] if u["kind"] == "org"]
        assert orgs == [], f"zero-fit clubs were floored into a fit-keyed category: {orgs}"
        assert results["QRK"], "the category should still be filled by its own facts"
        assert "floor is left unmet" in capsys.readouterr().out
    finally:
        index.close()


def test_the_fit_gate_turns_itself_off_when_the_bundle_scores_no_organisation(tmp_path, capsys):
    """Colleges 2-45. A bundle that never computed fit scores must not lose every club in every
    fit-keyed category -- that is the same silent-emptying failure as the course gate."""
    units = [
        unit(
            f"alpha-org-{i}",
            "org",
            f"Society {i} (Recognized Student Organization). Unusual traditions, odd rituals and "
            f"surprising campus lore, quirky and offbeat student clubs.",
            entity_name=f"Society {i}",
            source_url=f"https://soc{i}.example.org/about",
            extra={},  # this bundle does not compute fit_scores at all
        )
        for i in range(4)
    ]
    index = retrieve.CollegeIndex(write_index(tmp_path / "index", "alpha", units))
    try:
        assert index.fit_scored_orgs == 0
        results, _ = retrieve.retrieve(index, profile(), per_category=4, embedder=StubEmbedder())
        assert [u for u in results["QRK"] if u["kind"] == "org"], "the gate emptied the org pool"
        assert "fit gate is OFF" in capsys.readouterr().out
    finally:
        index.close()


def test_a_scoring_organisation_is_still_floored_in(tmp_path):
    """The other half: fit that IS there must still be preferred and still fill the floor."""
    units = [
        unit(
            "alpha-org-quirky",
            "org",
            "The Midnight Kazoo Brigade (Recognized Student Organization). Unusual traditions, odd "
            "rituals and surprising campus lore, quirky and offbeat student clubs.",
            entity_name="Midnight Kazoo Brigade",
            source_url="https://kazoo.example.org/about",
            extra={"fit_scores": {"quirky": 3}},
        ),
        unit(
            "alpha-org-plain",
            "org",
            "Engineering Society (Recognized Student Organization). Unusual traditions, odd "
            "rituals and surprising campus lore, quirky and offbeat student clubs.",
            entity_name="Engineering Society",
            source_url="https://eng.example.org/about",
            extra={"fit_scores": {"quirky": 0}},
        ),
    ] + [
        unit(
            f"alpha-fact-qrk-{i}",
            "fact",
            f"Campus lore {i}: an odd ritual nobody can explain, one of the unusual traditions "
            f"and surprising campus legends students keep up every year.",
            category_code="QRK",
            entity_name=f"Lore {i}",
            source_url=f"https://example.edu/lore/{i}",
        )
        for i in range(6)
    ]
    index = retrieve.CollegeIndex(write_index(tmp_path / "index", "alpha", units))
    try:
        results, _ = retrieve.retrieve(index, profile(), per_category=6, embedder=StubEmbedder())
        orgs = [u["entity_name"] for u in results["QRK"] if u["kind"] == "org"]
        assert orgs == ["Midnight Kazoo Brigade"], orgs
    finally:
        index.close()


# --------------------------------------------------------------------------------------
# 6. recency (LOW)
# --------------------------------------------------------------------------------------


def test_future_dated_rows_do_not_outrank_current_material(tmp_path):
    """A year in the future is a mention ("apply by fall 2031"), not freshness.

    Handing it a perfect 1.0 put a stale page that merely names a future year above correctly
    dated current material, in exactly the categories that asked for recency.
    """
    units = [
        unit(
            f"alpha-fact-res-{i}",
            "fact",
            f"Research note {i}: undergraduates in the laboratory publish work on computer "
            f"science and engineering problems every year.",
            category_code="RES",
            entity_name=f"Lab {i}",
            source_url=f"https://example.edu/lab/{i}",
            year=2026,
        )
        for i in range(3)
    ]
    index = retrieve.CollegeIndex(write_index(tmp_path / "index", "alpha", units))
    try:
        index.reference_year = 2026
        current, next_year, far_future = 0, 1, 2
        index.year[current] = 2026
        index.year[next_year] = 2027  # a catalogue legitimately runs one year ahead
        index.year[far_future] = 2031  # a page that merely mentions a future year

        assert retrieve.recency_factor(index, current) == 1.0
        assert retrieve.recency_factor(index, next_year) == 1.0
        assert retrieve.recency_factor(index, far_future) < retrieve.recency_factor(index, current)
        undated = retrieve.CONFIG["recency_unknown_year_factor"]
        assert retrieve.recency_factor(index, far_future) == pytest.approx(undated)
    finally:
        index.close()


# --------------------------------------------------------------------------------------
# 7. PART B -- per-category query weighting
# --------------------------------------------------------------------------------------

# The blind measurement the CATEGORIES table cites: 3 independent judges per category, same
# student, same index, scored 0-10. This copy is here so that a weight cannot be changed without
# either changing the evidence or failing this test.
BLIND_MEASUREMENT: dict[str, tuple[float, float]] = {
    # code: (student-led score, category-led score)
    "CUL": (5.0, 6.0),
    "EXT": (7.0, 6.3),
    "QRK": (2.3, 3.3),
    "ACA": (4.7, 3.0),
    "RES": (7.0, 4.0),
    "SOC": (6.3, 7.0),
    "INN": (5.7, 6.0),
    "INT": (4.0, 3.0),
    "DIV": (3.3, 5.7),
    "NEW": (5.7, 2.0),
}


def losing_arm_weight(margin: float) -> float:
    """The rule the table documents: 1.00 - 0.15 x margin, rounded to 0.05, floored at 0.45."""
    return max(0.45, round(round((1.0 - 0.15 * margin) / 0.05) * 0.05, 2))


def test_query_weighting_is_per_category_and_matches_the_blind_measurement():
    """One global intent-vs-student knob was wrong for ten different questions.

    This is the test that stops somebody "simplifying" the table back to a single setting: it
    re-derives every weight from the measurement the table cites.
    """
    for cat in retrieve.CATEGORIES:
        code = cat["code"]
        student_score, category_score = BLIND_MEASUREMENT[code]
        weights = cat["query_weights"]
        margin = abs(student_score - category_score)
        loser = losing_arm_weight(margin)

        if student_score > category_score:
            assert cat["query_lead"] == "student", code
            assert weights["student"] == 1.00, code
            assert weights["intent"] == pytest.approx(loser), (code, weights, loser)
        else:
            assert cat["query_lead"] == "category", code
            assert weights["intent"] == 1.00, code
            assert weights["student"] == pytest.approx(loser), (code, weights, loser)

    # and it really is per-category, not one number wearing ten hats
    balances = {c["code"]: (c["query_lead"], c["query_weights"]["intent"]) for c in retrieve.CATEGORIES}
    assert len(set(balances.values())) > 1, balances
    assert balances["RES"][0] == "student" and balances["DIV"][0] == "category"


def test_the_measured_balance_reaches_the_planned_queries(tmp_path):
    """The table is only worth anything if the weights arrive in the queries that run."""
    plans = retrieve.plan_queries(profile())
    for code, queries in plans.items():
        cat = retrieve.CATEGORY_BY_CODE[code]
        # "intended_fields".startswith("intent") is True, which is precisely the trap the
        # comment in plan_queries warns about. Match the whole label, not a prefix.
        def is_intent(q):
            return q.facet == "intent" or q.facet.startswith("intent:")
        intent = [q for q in queries if is_intent(q)]
        student = [q for q in queries if not is_intent(q)]
        assert intent, code
        assert all(q.weight == pytest.approx(cat["query_weights"]["intent"]) for q in intent), code
        # A DECLARED field is the client's authoritative brief and now outweighs the rest of the
        # student's facets. It used to carry the same weight as a scraped hobby list, and lost:
        # six off-topic queries agree with each other, one on-topic query agrees with nobody.
        base = cat["query_weights"]["student"]
        boost = retrieve.CONFIG["declared_field_boost"]
        for q in student:
            want = base * boost if q.facet.startswith("intended_fields") else base
            assert q.weight == pytest.approx(want), f"{code} {q.facet}"

    # the two widest margins in the table, in opposite directions
    def _is_intent(q):
        return q.facet == "intent" or q.facet.startswith("intent:")
    assert max(q.weight for q in plans["RES"] if not _is_intent(q)) > max(
        q.weight for q in plans["RES"] if _is_intent(q)
    )
    # DIV is category-led: its own lens outweighs the student's ORDINARY facets. A declared
    # field is excluded from the comparison because it is not an ordinary facet -- it is the
    # client's authoritative brief, and it is meant to outrank the lens.
    def _ordinary_student(q):
        return not _is_intent(q) and not q.facet.startswith("intended_fields")
    assert max(q.weight for q in plans["DIV"] if _is_intent(q)) > max(
        q.weight for q in plans["DIV"] if _ordinary_student(q)
    )


def test_student_led_categories_lead_with_the_students_phrase():
    """Where the measurement said student-led, the query text has to lead with her words too --
    leading with the category's lens returns the college's generic page about the topic."""
    plans = retrieve.plan_queries(profile(intended_fields=["marine biology"]))

    res = [q for q in plans["RES"] if q.facet.startswith("intended_fields")]
    assert res and res[0].text.lower().startswith("marine biology"), res[0].text

    div = [q for q in plans["DIV"] if not q.facet.startswith("intent")]
    lens = retrieve.CATEGORY_BY_CODE["DIV"]["lens"]
    assert div and div[0].text.lower().startswith(lens.split(",")[0].lower()), div[0].text


def test_the_keyword_half_keeps_category_context_in_both_arms():
    """Not what the table measures, and right in both arms: the keyword half used to search her
    resume text alone, with no idea which section it was filling."""
    plans = retrieve.plan_queries(profile(intended_fields=["marine biology"]))
    for code in ("RES", "DIV"):
        cat = retrieve.CATEGORY_BY_CODE[code]
        head = cat["lens"].split(",")[0].split(" and ")[0].lower()
        for query in plans[code]:
            if query.facet.startswith("intent"):
                continue
            assert head in query.keyword_text.lower(), (code, query.keyword_text)


def test_every_category_declares_its_own_balance():
    """A category added for college 2-45 without these keys would silently fall back to one
    global setting -- the exact thing the measurement disproved."""
    for cat in retrieve.CATEGORIES:
        assert cat.get("query_lead") in ("student", "category"), cat["code"]
        weights = cat.get("query_weights") or {}
        assert set(weights) == {"intent", "student"}, cat["code"]
        assert all(0.4 <= float(v) <= 1.0 for v in weights.values()), cat["code"]
        assert max(weights.values()) == 1.00, cat["code"]


def test_weighting_does_not_disturb_determinism(tmp_path):
    """Same profile, same index, same evidence -- weights are settings, not randomness."""
    units = [
        unit(
            f"alpha-fact-res-{i}",
            "fact",
            f"Research note {i}: undergraduate researchers in the robotics laboratory publish "
            f"work on computer science problems with faculty mentors.",
            category_code="RES",
            entity_name=f"Lab {i}",
            source_url=f"https://example.edu/lab/{i}",
            year=2025,
        )
        for i in range(8)
    ]
    index = retrieve.CollegeIndex(write_index(tmp_path / "index", "alpha", units))
    try:
        first, _ = retrieve.retrieve(index, profile(), per_category=5, embedder=StubEmbedder())
        second, _ = retrieve.retrieve(index, profile(), per_category=5, embedder=StubEmbedder())
        assert [u["unit_id"] for u in first["RES"]] == [u["unit_id"] for u in second["RES"]]
    finally:
        index.close()


# --------------------------------------------------------------------------------------
# Enrolment machinery must not crowd out what a student can actually study
# --------------------------------------------------------------------------------------

@pytest.fixture()
def boilerplate_index(tmp_path):
    """One university-wide scheme republished by every school, plus real subject material.

    This is the shape that defeated every existing cap: the same topic arriving under a
    different entity name from a different hostname each time, so max_per_entity and
    max_per_host both saw one unit each and let all six through.
    """
    units = []
    for i, school in enumerate(["engineering", "business", "letters", "music", "policy", "law"]):
        units.append(unit(
            f"alpha-fact-pdp-{i}", "fact",
            "Progressive degree students must make completion of the undergraduate degree a "
            "priority and stay enrolled full time in their undergraduate major courses.",
            category_code="ACA",
            entity_name=f"Progressive Degree Program ({school})",
            source_url=f"https://{school}.example.edu/progressive-degree-{i}",
            source_kind="official",
        ))
    for i, (code, title) in enumerate([
        ("ACCT 370", "External Financial Reporting Issues"),
        ("ACCT 416", "Financial Statement Analysis"),
        ("BUAD 280", "Introduction to Financial Accounting"),
        ("MATH 118", "Calculus for Business and Economics"),
        ("FBE 421", "Financial Analysis and Valuation"),
        ("BUAD 310", "Applied Business Statistics"),
    ]):
        units.append(unit(
            f"alpha-course-{i}", "course",
            f"{code}: {title} (4.0 Units). An undergraduate course in the accounting and finance "
            f"degree requirements for business majors.",
            category_code="ACA", entity_name=code,
            source_url=f"https://classes.example.edu/{code.replace(' ', '').lower()}",
            source_kind="official",
            extra={"is_undergraduate": True, "dept": code.split()[0], "number": code.split()[1]},
        ))
    root = tmp_path / "index"
    write_index(root, "alpha", units)
    return root


def test_enrolment_machinery_is_capped_per_category(boilerplate_index):
    """Six units of combined-degree mechanics took 6 of an accountancy applicant's 29 Academics
    slots, because each copy came from a different school's own site under its own entity name."""
    index = retrieve.CollegeIndex(boilerplate_index / "alpha")
    results, _ = retrieve.retrieve(index, profile(), per_category=10, embedder=StubEmbedder())
    units = results["ACA"]
    boilerplate = [u for u in units if "progressive degree" in (u.get("text") or "").lower()]
    assert len(boilerplate) <= retrieve.CONFIG["max_process_boilerplate"], \
        f"{len(boilerplate)} units of enrolment machinery got through the cap"
    # and the cap must not starve the section -- the real subject material takes those slots
    assert len(units) >= 5, f"cap starved the category: only {len(units)} units"
    named = [u for u in units if any(c in (u.get("text") or "") for c in ("ACCT", "BUAD", "MATH", "FBE"))]
    assert len(named) > len(boilerplate), (
        f"{len(named)} units name something a student could enrol in, against "
        f"{len(boilerplate)} on enrolment machinery")


# --------------------------------------------------------------------------------------
# Ranking a student's own phrases against a category
# --------------------------------------------------------------------------------------

def test_word_overlap_confuses_car_servicing_with_community_service():
    """The failure this replaced: five-character stems scored a car dealership's "Heads of
    Sales, Service, HR" and a chatbot's "user engagement" above an award-winning e-waste
    recycling venture, for the Social Impact section. The venture scored zero and never
    became a query."""
    soc = next(c for c in retrieve.CATEGORIES if c["code"] == "SOC")
    values = [
        "Topsel Toyota Intern Collaborated with Heads of Sales, Service, HR",
        "Greenbyte Founder Conceptualized and led a large-scale e-waste management initiative",
    ]
    # with no ranker, overlap still picks the dealership first -- documented, not endorsed
    picked = retrieve.rank_for_category(values, soc, 1, None)
    assert "Toyota" in picked[0]


def test_social_impact_asks_about_sustainability():
    """The corpus tags sustainability, equity, literacy and K-12 outreach as Social Impact.
    The queries only asked about volunteering, so a university's own monthly e-waste drive
    was unreachable by the one applicant who had run one."""
    soc = next(c for c in retrieve.CATEGORIES if c["code"] == "SOC")
    blob = " ".join(soc["intents"]) + " " + soc["lens"]
    assert re.search(r"sustainab|environment|recycl", blob, re.I), \
        "Social Impact never asks about the environment, but the corpus files it here"


def test_facet_ranker_falls_back_rather_than_failing():
    """A worse ranking is recoverable; a failed run is not."""
    class Broken:
        def embed_queries(self, texts):
            raise RuntimeError("provider down")
    soc = next(c for c in retrieve.CATEGORIES if c["code"] == "SOC")
    ranker = retrieve.FacetRanker(Broken(), {"activities": ["a b c", "d e f"]}, [soc])
    assert ranker.rank(["a b c", "d e f"], soc) is None
    assert retrieve.rank_for_category(["a b c", "d e f"], soc, 1, ranker) == ["a b c"]


# --------------------------------------------------------------------------------------
# the anchor pass: one thing the student DID, run on its own, seated with its artefact
# --------------------------------------------------------------------------------------


@pytest.fixture()
def anchor_index(tmp_path):
    """A professor whose research IS the student's paper, a campus thing that IS the
    student's venture, a six-word fact that overlaps the file but says nothing, and filler."""
    from wwrag.tests.test_retrieve import unit, write_index

    units = [
        unit("alpha-fact-prof", "fact",
             "Professor Ada Lovelace researches the economic cost of menopause productivity loss "
             "among working women.",
             category_code="RES", entity_name="Ada Lovelace",
             source_url="https://example.edu/econ/lovelace"),
        unit("alpha-fact-hub", "fact",
             "The Campus Sustainability Hub collected 4,000 pounds of e-waste from homes in its "
             "first e-waste management and awareness drive.",
             category_code="SOC", entity_name="Campus Sustainability Hub",
             source_url="https://example.edu/sustainability/hub"),
        unit("alpha-fact-thin", "fact", "Ravi Kumar is an alumni mentor.",
             category_code="EXT", entity_name="Ravi Kumar",
             source_url="https://example.edu/alumni/mentors"),
    ]
    for code in ("RES", "SOC", "EXT", "ACA", "CUL"):
        for i in range(4):
            units.append(unit(f"alpha-fact-{code.lower()}-{i}", "fact",
                              f"Filler sentence number {i} for the {code} category about campus life "
                              f"and programs that every student can join here.",
                              category_code=code, entity_name=f"Thing {code} {i}",
                              source_url=f"https://example.edu/{code.lower()}/{i}"))
    return write_index(tmp_path / "index", "alpha", units)


def test_anchor_pass_seats_the_person_level_join_with_its_artefact(anchor_index):
    from wwrag.tests.test_retrieve import StubEmbedder, profile

    index = retrieve.CollegeIndex(anchor_index)
    saved = dict(retrieve.CONFIG)
    retrieve.CONFIG["anchor_min_sim"] = 0.2   # the stub embedder is bag-of-words; test the mechanism
    try:
        paper = ("Economic cost of menopause-related productivity loss published in a journal: "
                 "the estimated economic cost of menopause productivity loss among working women")
        venture = ("Greenbyte Founder led a large-scale e-waste management and awareness initiative "
                   "collecting e-waste from homes and schools")
        council = "Student Council Head of Alumni Relations connected with each alumni mentor for students"
        prof = profile(projects=[paper], activities=[venture, council],
                       achievements=["Quiz: 1st place: school science quiz"],
                       intended_fields=["Economics"])
        artefacts = [t for t, _ in retrieve.anchor_artefacts(prof, StubEmbedder())]
        assert paper in artefacts and venture in artefacts
        assert not any(a.startswith("Quiz") for a in artefacts), "a prize line is not an artefact"

        results, explain = retrieve.retrieve(index, prof, per_category=4, embedder=StubEmbedder())
        anchors = explain["anchors"]
        assert anchors, "no anchor seated"
        by_unit = {a["unit_id"]: a for a in anchors}
        assert by_unit["alpha-fact-prof"]["chapter"] == "RES"
        assert by_unit["alpha-fact-prof"]["why"].startswith("faculty")
        assert by_unit["alpha-fact-hub"]["chapter"] == "SOC"
        assert "alpha-fact-thin" not in by_unit, "a six-word fact is not an anchor"

        # the faculty join is seated FIRST, carrying the line of the file it answers to
        assert results["RES"][0]["unit_id"] == "alpha-fact-prof"
        assert results["RES"][0]["anchor_for"] == paper
        hub = next(u for u in results["SOC"] if u["unit_id"] == "alpha-fact-hub")
        assert hub["anchor_for"] in artefacts
        # anchors keep their seating order: every anchored row precedes every blended row
        for units in results.values():
            flags = [bool(u["anchor_for"]) for u in units]
            assert flags == sorted(flags, reverse=True), flags
        # and chapters did not grow
        assert all(len(v) <= 4 for v in results.values())
        assert all("anchor_for" in u for units in results.values() for u in units)
    finally:
        retrieve.CONFIG.clear(); retrieve.CONFIG.update(saved)
        index.close()
