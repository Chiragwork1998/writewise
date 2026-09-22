"""Edge direction: a relation stated the wrong way round ships a false sentence to a student.

"MKT 486 teaches Joseph Nunes" reached a real report. The extractor reads a relation off one
sentence and a sentence often names the object first ("MKT 486, led by Professor Nunes"). Node
types settle the direction without reading the sentence.

Run:
  /Users/chirag/college-intel/.venv-crawl4ai/bin/python -m pytest \
      /Users/chirag/college-intel/wwrag/tests/test_graph_direction.py -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import graph  # noqa: E402

TYPES = {
    "course:mkt 486": "course", "professor:joseph nunes": "professor",
    "lab:analog group": "lab", "professor:mike chen": "professor",
    "event:heritage month": "event", "center:la casa": "center",
    "event:trojan talks ep5": "event", "person:eileen crimmins": "person",
    "professor:molisch": "professor", "lab:wides": "lab",
    "office:provost": "office", "service:career center": "service",
    "program:good neighbors": "program", "event:after cool": "event",
}


def test_a_course_cannot_teach_a_professor():
    assert graph.reorient("course:mkt 486", "TEACHES", "professor:joseph nunes", TYPES) == \
        ("professor:joseph nunes", "course:mkt 486")


def test_a_lab_cannot_direct_a_professor():
    assert graph.reorient("lab:analog group", "DIRECTS", "professor:mike chen", TYPES) == \
        ("professor:mike chen", "lab:analog group")


def test_an_event_cannot_host_an_institution():
    assert graph.reorient("event:heritage month", "HOSTS", "center:la casa", TYPES) == \
        ("center:la casa", "event:heritage month")


def test_an_event_hosting_a_person_is_dropped_not_guessed():
    """A podcast guest is not the host. Turning it round would ship the opposite falsehood."""
    assert graph.reorient("event:trojan talks ep5", "HOSTS", "person:eileen crimmins", TYPES) is None


def test_valid_directions_are_left_alone():
    assert graph.reorient("professor:molisch", "DIRECTS", "lab:wides", TYPES) == \
        ("professor:molisch", "lab:wides")
    assert graph.reorient("office:provost", "FUNDS", "service:career center", TYPES) == \
        ("office:provost", "service:career center")
    assert graph.reorient("program:good neighbors", "FUNDS", "event:after cool", TYPES) == \
        ("program:good neighbors", "event:after cool")


def test_unknown_types_are_left_alone():
    """A college whose extractor emits a type we have no rule for must not lose edges."""
    assert graph.reorient("widget:a", "TEACHES", "widget:b", {}) == ("widget:a", "widget:b")


# --------------------------------------------------------------------------------------
# Hubs: the threshold has to scale, because colleges are not all USC's size
# --------------------------------------------------------------------------------------

def _graph_with_degrees(tmp_path, degrees):
    import sqlite3
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE graph_nodes (id TEXT, type TEXT, name TEXT, degree INT)")
    db.executemany("INSERT INTO graph_nodes VALUES (?,?,?,?)",
                   [(f"n{i}", "lab", f"n{i}", d) for i, d in enumerate(degrees)])
    return db


def test_hub_degree_scales_with_the_graph(tmp_path):
    """A count tuned on USC would never fire on a college with a tenth of the corpus."""
    big = _graph_with_degrees(tmp_path, [1] * 990 + [13] * 9 + [1300])
    small = _graph_with_degrees(tmp_path, [1] * 99 + [2])
    assert graph.hub_degree(big, graph.CONFIG) > graph.hub_degree(small, graph.CONFIG)
    assert graph.hub_degree(small, graph.CONFIG) == graph.CONFIG["hub_degree_floor"]


def test_a_small_specialised_school_is_not_a_hub():
    """USC Leventhal School of Accounting has degree 41 and anchors 15 units of real student
    evidence. Treating every node typed 'school' as a hub would have silenced it."""
    assert not graph.is_hub({"type": "school", "degree": 41}, 104)
    assert not graph.is_hub({"type": "school", "degree": 16}, 104)
    assert graph.is_hub({"type": "university", "degree": 1308}, 104)
