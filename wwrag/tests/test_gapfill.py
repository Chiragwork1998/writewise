"""Tests for wwrag/gapfill.py and its wiring into wwrag/retrieve.py.

Offline and fast, in the same style as test_retrieve.py: a tiny index in the layout
wwrag/index_build.py documents, PLUS the graph_nodes table wwrag/graph.py writes, and the same
deterministic bag-of-words stub embedder. No model, no network, no other wwrag module.

What is asserted:
  * the pass closes a gap the first pass structurally could not see -- material that exists in
    the candidate set, that the category's own facet list makes unreachable;
  * the filter still runs BEFORE the search: a gap query can never return a row outside the
    category's pre-filtered candidate set, even when that row is the best keyword match in the
    whole index;
  * the change is BOUNDED: at most `gap_slots` units differ from a run with the pass off, and
    every diversity cap still holds;
  * `gapfill_enabled=False` reproduces the pass-off output exactly, so the before/after stays
    measurable;
  * determinism: same profile, same index, same bytes;
  * an index with no entity graph disables the pass LOUDLY instead of falling back to picking
    the rarest word on the resume;
  * a resume is data: FTS5 operators in it cannot reshape a gap query;
  * no college-specific literal has leaked into either module.

Run:
  /Users/chirag/college-intel/.venv-crawl4ai/bin/python -m pytest /Users/chirag/college-intel/wwrag/tests/test_gapfill.py -q
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gapfill  # noqa: E402
import retrieve  # noqa: E402

from test_retrieve import StubEmbedder, unit, write_index  # noqa: E402


GRAPH_SCHEMA = """
CREATE TABLE IF NOT EXISTS graph_nodes (
    id            TEXT PRIMARY KEY,
    type          TEXT NOT NULL,
    name          TEXT NOT NULL,
    name_key      TEXT NOT NULL,
    fact_count    INTEGER,
    source_count  INTEGER,
    categories    TEXT,
    degree        INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_graph_nodes_key  ON graph_nodes(name_key);
CREATE INDEX IF NOT EXISTS idx_graph_nodes_type ON graph_nodes(type);
"""


def add_graph(index_dir: Path, nodes: list[tuple[str, str]]) -> None:
    """Write the graph_nodes table wwrag/graph.py builds. `nodes` is [(type, name), ...]."""
    conn = sqlite3.connect(str(index_dir / "chunks.sqlite"))
    conn.executescript(GRAPH_SCHEMA)
    for node_type, name in nodes:
        key = " ".join(name.lower().split())
        conn.execute(
            "INSERT OR REPLACE INTO graph_nodes(id, type, name, name_key, fact_count,"
            " source_count, categories, degree) VALUES (?,?,?,?,1,1,'',1)",
            (f"{node_type}:{key}", node_type, name, key),
        )
    conn.commit()
    conn.close()


# --------------------------------------------------------------------------------------
# A college whose Research material is reachable only through a facet Research never reads.
# --------------------------------------------------------------------------------------
#
# Research's `facets` tuple is (intended_fields, projects, skills, interests) -- `activities` is
# not in it. So a student whose sensor work sits in `activities` cannot have a Research query
# built from it, at any depth. That is the reproduced production failure, in miniature.

ANCHOR_WORD = "thermistor"  # rare, and the graph names entity nodes after it
LAB_NAMES = [f"{ANCHOR_WORD.title()} Networks Group {i}" for i in range(1, 13)]

# Deliberately NOT in the lab text, the lab names or the quotes: the first pass's Research
# queries carry the category's lens ("research labs and undergraduate research") in BOTH
# halves, so any lab that repeats a lens word would be found by the first pass and there
# would be no gap to close.
LENS_WORDS = "research laboratory laboratories undergraduate undergraduates labs"


def gap_units() -> list[dict]:
    units: list[dict] = []
    # the material the gap pass should find: NAMED, inside the RES candidate set, and
    # describable only in the student's own vocabulary
    for i, name in enumerate(LAB_NAMES[:4]):
        units.append(
            unit(
                f"lab-{i}",
                "fact",
                f"The {name} operates {ANCHOR_WORD} humidity telemetry instrumentation.",
                category_code="RES",
                entity_name=name,
                quote=f"The {name} takes students each summer.",
                source_url=f"https://g{i}.example.edu/groups/page",
            )
        )
    # what the first pass actually retrieves for this student: true, cited and unnamed. Each
    # one repeats the lens AND her READ facets so it wins both halves, each is textually
    # distinct so the near-duplicate guard does not collapse them, and each sits on its own
    # host so the per-host cap does not either.
    topics = [
        "ballot measures", "zoning hearings", "transit budgets", "housing vouchers",
        "school boards", "civic technology", "census tracts", "municipal bonds",
        "open records", "participatory budgets", "city charters", "county audits",
        "voter turnout", "public comment", "permit backlogs", "utility rates",
        "park districts", "sanitation routes", "fire districts", "library levies",
    ]
    for i, topic in enumerate(topics):
        units.append(
            unit(
                f"prose-{i}",
                "chunk",
                f"{LENS_WORDS} on {topic} in public policy and local government, a "
                f"writing-intensive track for students interested in {topic}.",
                source_url=f"https://p{i}.example.edu/research/getting-started",
            )
        )
    # named FACTS the first pass legitimately prefers, so the per-kind cap on page prose does
    # not hand a slot to the very material this fixture says the first pass cannot reach
    for i, name in enumerate(("Policy Futures Center", "Civic Data Institute",
                              "Local Government Clinic", "Records Access Project",
                              "Charter Review Center", "Ballot Design Institute")):
        units.append(
            unit(
                f"decoyfact-{i}",
                "fact",
                f"The {name} offers {LENS_WORDS} in public policy, local government and "
                f"writing for students.",
                category_code="RES",
                entity_name=name,
                quote=f"{name} works with students on public policy.",
                source_url=f"https://f{i}.example.edu/centers/page",
            )
        )
    # a row OUTSIDE the Research pre-filter that is the best keyword match in the whole index:
    # if a gap query could ever widen the filter, this is the row that would prove it
    units.append(
        unit(
            "outside-filter",
            "fact",
            f"{ANCHOR_WORD} {ANCHOR_WORD} {ANCHOR_WORD} telemetry humidity instrumentation.",
            category_code="QRK",  # not in RES's fact_codes or support_codes
            entity_name=f"{ANCHOR_WORD.title()} Society",
            quote="a quirky club",
            source_url="https://example.edu/quirks",
        )
    )
    return units


def gap_profile(**overrides) -> dict:
    base = {
        "student_id": "gap-student",
        "first_name": "Rae",
        "level": "undergraduate",
        # the facets Research DOES read say nothing about the anchor material
        "intended_fields": ["public policy"],
        "skills": ["writing"],
        "interests": ["local government"],
        "values": ["service"],
        # ...and the anchor material sits in the one facet Research never reads
        "activities": [
            {
                "name": "Weather station build",
                "role": "builder",
                "detail": f"Built a {ANCHOR_WORD} humidity telemetry instrumentation rig.",
                "evidence_line": f"Built a {ANCHOR_WORD} humidity telemetry rig.",
            }
        ],
        "projects": [],
        "achievements": [],
        "raw_text_sha256": "0" * 64,
        "flags": [],
    }
    base.update(overrides)
    return base


@pytest.fixture()
def gap_index(tmp_path: Path):
    index_dir = write_index(tmp_path / "idx", "demo", gap_units())
    add_graph(index_dir, [("lab", name) for name in LAB_NAMES])
    index = retrieve.CollegeIndex(index_dir)
    try:
        yield index
    finally:
        index.close()


def run(index, profile, per_category=6, **kwargs):
    return retrieve.retrieve(
        index, profile, per_category=per_category, embedder=StubEmbedder(),
        categories=(retrieve.CATEGORY_BY_CODE["RES"],), **kwargs,
    )


# --------------------------------------------------------------------------------------
# the failure this pass exists to close
# --------------------------------------------------------------------------------------


def test_first_pass_cannot_reach_material_behind_an_unread_facet(gap_index):
    """Baseline. The labs exist, they are inside the candidate set, and the first pass misses
    them -- because Research never reads `activities`."""
    assert "activities" not in retrieve.CATEGORY_BY_CODE["RES"]["facets"]
    off, _ = run(gap_index, gap_profile(), gapfill_enabled=False)
    names = {u["entity_name"] for u in off["RES"]}
    assert not (names & set(LAB_NAMES)), f"expected the first pass to miss the labs, got {names}"


def test_gap_pass_closes_it(gap_index):
    on, explain = run(gap_index, gap_profile())
    names = {u["entity_name"] for u in on["RES"] if u["entity_name"]}
    assert names & set(LAB_NAMES), f"the gap pass did not surface the named labs: {names}"

    block = explain["categories"]["RES"]["gapfill"]
    assert block["ran"] and block["queries"], block
    query = block["queries"][0]
    assert query["lead_term"] == ANCHOR_WORD
    assert query["facet"] == "activities"
    assert query["facet_unread_by_first_pass"] is True
    # the resume line the query was built from is recorded verbatim, so the file is auditable
    assert ANCHOR_WORD in query["basis"]
    assert block["filled"] >= 1


def test_every_gap_unit_keeps_its_quote_and_source_url(gap_index):
    on, explain = run(gap_index, gap_profile())
    chosen = {s["unit_id"]: s for s in explain["categories"]["RES"]["selected"]}
    gap_units_shipped = [
        u for u in on["RES"] if chosen[u["unit_id"]]["selected_because"].startswith("gap slot")
    ]
    assert gap_units_shipped
    for u in gap_units_shipped:
        assert u["quote"] and u["quote"].strip()
        assert u["source_url"].startswith("https://")
        assert u["text"] and u["retrieval"]["keyword"] > 0.0


# --------------------------------------------------------------------------------------
# FILTER FIRST -- the invariant the whole pipeline was rebuilt for
# --------------------------------------------------------------------------------------


def test_a_gap_query_never_widens_the_pre_filter(gap_index):
    """The best keyword match for the gap query in the whole index is tagged for another
    category. It must be unreachable, exactly as it is for the first pass."""
    on, _ = run(gap_index, gap_profile())
    assert all(u["unit_id"] != "outside-filter" for u in on["RES"])
    assert all(u["category_code"] in (None, "RES", "GEN", "ACA") for u in on["RES"])


def test_gap_candidates_all_come_from_the_category_candidate_rows(gap_index):
    cat = retrieve.CATEGORY_BY_CODE["RES"]
    rows = set(retrieve.candidate_rows(gap_index, cat, True).tolist())
    _, explain = run(gap_index, gap_profile())
    ids = {gap_index.unit_id[r] for r in rows}
    for candidate in explain["categories"]["RES"]["gapfill"]["candidates"]:
        assert candidate["unit_id"] in ids


# --------------------------------------------------------------------------------------
# the change is BOUNDED and the caps still hold
# --------------------------------------------------------------------------------------


def test_the_change_is_bounded_by_the_reserved_slot_count(gap_index):
    off, _ = run(gap_index, gap_profile(), per_category=6, gapfill_enabled=False)
    on, _ = run(gap_index, gap_profile(), per_category=6)
    before = [u["unit_id"] for u in off["RES"]]
    after = [u["unit_id"] for u in on["RES"]]
    assert len(before) == len(after)
    added = set(after) - set(before)
    assert 0 < len(added) <= gapfill.gap_slots(6)


def test_gap_slots_scale_with_the_section_length():
    # measured at 2 of 12; production runs 24, where a fixed 2 would be half the measured share
    assert gapfill.gap_slots(12) == 2
    assert gapfill.gap_slots(24) == 4
    assert gapfill.gap_slots(2) == 0
    assert gapfill.gap_slots(60) == gapfill.CONFIG["gap_slots_max"]


def test_diversity_caps_still_hold_over_gap_units(gap_index):
    on, _ = run(gap_index, gap_profile(), per_category=12)
    entities: dict[str, int] = {}
    pages: dict[str, int] = {}
    for u in on["RES"]:
        key = (u["entity_name"] or u["unit_id"]).lower()
        entities[key] = entities.get(key, 0) + 1
        pages[u["source_url"]] = pages.get(u["source_url"], 0) + 1
    assert max(entities.values()) <= retrieve.CONFIG["max_per_entity"]
    assert max(pages.values()) <= retrieve.CONFIG["max_per_source_page"]


def test_no_gapfill_reproduces_the_pass_off_output_exactly(gap_index):
    first, _ = run(gap_index, gap_profile(), gapfill_enabled=False)
    second, _ = run(gap_index, gap_profile(), gapfill_enabled=False)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_determinism_with_the_pass_on(gap_index):
    first, _ = run(gap_index, gap_profile())
    second, _ = run(gap_index, gap_profile())
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


# --------------------------------------------------------------------------------------
# gates: what must NOT produce a gap query
# --------------------------------------------------------------------------------------


def test_an_index_with_no_entity_graph_disables_the_pass_loudly(tmp_path, capsys):
    """Colleges 2-45 may ship without a graph. The anchor gate must switch OFF and SAY SO --
    never fall back to ungated rarity, which picks nonsense anchors."""
    index_dir = write_index(tmp_path / "idx", "demo", gap_units())  # no add_graph
    index = retrieve.CollegeIndex(index_dir)
    try:
        with_graph_off, explain = run(index, gap_profile())
    finally:
        index.close()
    printed = capsys.readouterr().out
    assert "no entity graph" in printed
    assert explain["gapfill"]["graph_available"] is False
    assert explain["categories"]["RES"]["gapfill"]["ran"] is False
    names = {u["entity_name"] for u in with_graph_off["RES"]}
    assert not (names & set(LAB_NAMES))


def test_a_term_the_graph_knows_nothing_about_cannot_anchor(gap_index):
    """Rarity alone is not salience. A word the college has no named material for scores zero
    on the graph gate and must not become a query, however rare it is."""
    hobby = gap_profile(
        activities=[{"name": "Bagging", "role": "clerk",
                     "detail": "Bagged quinceanera supplies at a suburban grocery cooperative.",
                     "evidence_line": "Grocery bagger, weekends."}]
    )
    _, explain = run(gap_index, hobby)
    assert explain["categories"]["RES"]["gapfill"]["queries"] == []


def test_people_are_never_an_anchor_type_but_are_always_an_answer(gap_index):
    """A professor node's NAME is a personal name, so a resume word matching one says only
    that the word is somebody's surname."""
    for code, types in gapfill.CATEGORY_ANCHOR_TYPES.items():
        assert "professor" not in types and "person" not in types, code
    assert "professor" in gapfill.answer_types("RES")
    assert gapfill.answer_types("NEW") == ()


def test_a_rephrase_of_a_first_pass_query_is_refused():
    first_pass = ["undergraduate research opportunities with named professors and labs"]
    assert gapfill.is_rephrase("research opportunities with professors and labs", first_pass)
    assert not gapfill.is_rephrase(f"{ANCHOR_WORD} humidity telemetry", first_pass)


def test_a_category_led_category_gets_no_gap_pass(gap_index):
    """A gap query is nothing but the student's own words, and the CATEGORIES table's blind
    measurement says her words must not lead an abstract question."""
    quirks = retrieve.CATEGORY_BY_CODE["QRK"]
    assert quirks["query_lead"] == "category"
    _, explain = retrieve.retrieve(
        gap_index, gap_profile(), per_category=6, embedder=StubEmbedder(),
        categories=(quirks,),
    )
    block = explain["categories"]["QRK"]["gapfill"]
    assert block["queries"] == []
    assert "category-led" in block["reason"]


def test_a_gap_hit_that_names_nothing_may_not_take_a_slot():
    stop: set[str] = set()
    assert not gapfill.is_nameable(None, "undergraduate research is a cornerstone", stop)
    assert not gapfill.is_nameable("The University Program", "prose", stop)
    # a course UNIT is one named course by construction; a course-shaped code in loose prose
    # is not -- measured, that pattern also matches a standards number in a reference list
    assert gapfill.is_nameable(None, "CSCI 350 Operating Systems", stop, kind="course")
    assert not gapfill.is_nameable(None, "see IEEE 802 in the bibliography", stop, kind="chunk")
    assert gapfill.is_nameable("Forsburg Lab", "prose", stop)


def test_bm25_floor_drops_a_query_with_no_real_match():
    hits = [(1, -20.0), (2, -12.0), (3, -4.0)]
    kept = [row for row, _ in gapfill.apply_bm25_floor(hits)]
    assert kept == [1, 2]  # -4.0 is under 50% of the best -20.0
    assert gapfill.apply_bm25_floor([]) == []


# --------------------------------------------------------------------------------------
# the profile is DATA
# --------------------------------------------------------------------------------------


def test_fts_operators_in_a_resume_cannot_reshape_a_gap_query(gap_index):
    hostile = gap_profile(
        activities=[{
            "name": 'thermistor" OR unit_id:*',
            "role": "NEAR(x y)",
            "detail": f'{ANCHOR_WORD} humidity telemetry" OR "*"',
            "evidence_line": "AND NOT OR *",
        }]
    )
    on, explain = run(gap_index, hostile)
    assert all(u["unit_id"] != "outside-filter" for u in on["RES"])
    for query in explain["categories"]["RES"]["gapfill"]["queries"]:
        # every clause fts_query emits is a quoted string, so nothing here is an operator
        assert '"' not in query["keyword_text"]
        assert "*" not in query["keyword_text"]


def test_gap_anchors_read_evidence_line_which_the_first_pass_never_sees():
    """The sharpest word on a resume often appears only in evidence_line, which
    retrieve.profile_facets does not read. The gap reader does -- and reading it HERE, rather
    than widening profile_facets, is what keeps the first pass byte-identical."""
    profile = gap_profile(
        activities=[{"name": "Build", "role": "lead", "detail": "a rig",
                     "evidence_line": "wired an mqtt broker to a humidity probe"}]
    )
    facets = retrieve.profile_facets(profile)
    assert not any("mqtt" in v.lower() for v in facets["activities"])
    anchors = gapfill.anchor_entries(profile)
    assert any("mqtt" in a.basis.lower() for a in anchors)


# --------------------------------------------------------------------------------------
# 45 colleges ship on this one file
# --------------------------------------------------------------------------------------


def test_no_college_specific_literals_in_gapfill_py():
    """A single leaked name makes 44 colleges wrong. Note the substring trap: 'usc' hides
    inside ordinary English words such as 'muscle' and 'onusual' typos, so this check will
    fire on innocent prose too -- that is deliberate."""
    source = Path(gapfill.__file__).read_text(encoding="utf-8")
    body = source.split('"""', 2)[2]  # skip the module docstring
    for literal in ("usc", "USC", "Southern California", "Trojan"):
        assert literal not in body, f"college-specific literal {literal!r} leaked into gapfill.py"


def test_gapfill_thresholds_are_config_entries_not_magic_numbers():
    """Every threshold is tuned on one college and must be re-derived on the next one, so it
    has to be visible in one place and recorded in the explain file."""
    for key in (
        "anchor_min_df", "anchor_max_df_frac", "anchor_min_graph_fit", "query_max_df_frac",
        "bm25_floor_frac", "gap_slot_fraction", "gap_slots_max", "max_rephrase_overlap",
    ):
        assert key in gapfill.CONFIG


def test_the_explain_file_records_the_whole_gap_plan(gap_index):
    _, explain = run(gap_index, gap_profile())
    top = explain["gapfill"]
    assert top["enabled"] is True and top["graph_available"] is True
    assert top["model_calls"] == 0 and top["cost_usd"] == 0.0
    assert top["queries"] >= 1 and top["slots_filled"] >= 1
    assert top["config"]["anchor_min_graph_fit"] == gapfill.CONFIG["anchor_min_graph_fit"]
    assert "gapfill" in explain["timings_seconds"]
