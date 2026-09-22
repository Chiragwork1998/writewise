"""Regression tests for the verification gate: one per hole that was found open.

Each test here names the hole it protects. They are deliberately written against behaviour
a student would feel -- what ships, what is deleted, what the ledger says shipped -- rather
than against the shape of the helpers, so a future refactor of verify.py cannot make them
pass by moving the check somewhere it no longer runs.

Run:
  /Users/chirag/college-intel/.venv-crawl4ai/bin/python -m pytest \
      /Users/chirag/college-intel/wwrag/tests/test_verify_regressions.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import verify  # noqa: E402
from test_verify import EVIDENCE, PROFILE, _item, make_stub, run, statuses, body_of  # noqa: E402


def ledger_claim(result, claim_id):
    return [c for c in result["ledger"]["claims"] if c["claim_id"] == claim_id][0]


def prompts(log):
    return [user for _system, user in log]


# --------------------------------------------------------------------------------------
# 1. bracketed prose is printed, so it must be judged
# --------------------------------------------------------------------------------------

def test_bracketed_prose_is_judged_because_the_report_prints_it():
    """Both layers judge text_clean while the report ships the sentence as written.

    Treating every word inside a bracket as a citation id stripped the prose out of the text
    the layers saw and left it in the text the reader saw: an unjudged superlative shipped.
    """
    items = [_item(
        "EXT", "Rovers",
        "The Robotics Club builds autonomous rovers "
        "[ranked the best rover team in the nation].",
        evidence_ids=["u-fact-robotics"],
    )]
    log: list = []
    result = run(items, rules=[("ranked the best rover team", "unsupported", None)], log=log)

    assert any("ranked the best rover team" in user for user in prompts(log)), \
        "the bracketed prose never reached the verifier, yet it would have been printed"
    assert statuses(result)["EXT.body.01"] == "unsupported"
    assert result["report"] == []


def test_a_real_citation_marker_is_still_stripped_before_judging():
    """The narrower rule must not start showing the verifier the citation plumbing."""
    items = [_item("EXT", "Rovers",
                   "The Robotics Club builds autonomous rovers [u-fact-robotics].",
                   evidence_ids=["u-fact-robotics"])]
    log: list = []
    result = run(items, log=log)
    assert all("[u-fact-robotics]" not in user for user in prompts(log))
    assert statuses(result)["EXT.body.01"] == "supported"
    # ... and the marker survives into the report, where report.py turns it into a footnote
    assert "[u-fact-robotics]" in body_of(result, "EXT")


def test_a_bracket_that_mixes_an_id_with_prose_is_left_in_the_judged_text():
    items = [_item("EXT", "Rovers",
                   "The Robotics Club builds autonomous rovers [u-fact-robotics, see also].",
                   evidence_ids=["u-fact-robotics"])]
    log: list = []
    run(items, log=log)
    assert any("see also" in user for user in prompts(log))


# --------------------------------------------------------------------------------------
# 2. short sentences that promise the student something are claims
# --------------------------------------------------------------------------------------

def test_a_four_word_promise_about_admission_is_never_auto_supported():
    """"This guarantees admission." is four words. The product forbids promising a student
    anything, and no source can support such a sentence, so it must be judged, not waved
    through as connective tissue."""
    assert verify.is_non_factual("This guarantees admission.") is False
    assert verify.is_non_factual("You will get in.") is False
    assert verify.is_non_factual("Your odds improve.") is False
    assert verify.is_non_factual("Admission is assured.") is False

    items = [_item("EXT", "Rovers",
                   "The Robotics Club builds autonomous rovers. This guarantees admission.",
                   evidence_ids=["u-fact-robotics"])]
    log: list = []
    result = run(items, rules=[("guarantees admission", "unsupported", None)], log=log)
    assert any("guarantees admission" in user for user in prompts(log)), \
        "the promise skipped the verifier entirely"
    assert statuses(result)["EXT.body.02"] == "unsupported"
    assert "guarantees admission" not in body_of(result, "EXT")
    assert ledger_claim(result, "EXT.body.02")["non_factual"] is False


def test_a_genuinely_connective_fragment_is_still_exempt():
    """The exemption exists for real connective tissue; narrowing it must not remove that."""
    assert verify.is_non_factual("And there is more.") is True
    assert verify.is_non_factual("But not always.") is True


# --------------------------------------------------------------------------------------
# 3. --no-model must not pass off unjudged prose as verified
# --------------------------------------------------------------------------------------

def test_no_model_mode_deletes_what_it_cannot_positively_support():
    """Finding no contradiction is not support.

    With the model gate disabled, a deterministically clean paraphrase used to default to
    KEPT and the rendered report gave the reader no sign that the second layer never ran.
    Deleting is the safer default: a marker only reaches the reader if every renderer
    cooperates, while a deletion cannot be dropped downstream.
    """
    items = [_item(
        "CUL", "Dining",
        "Students rave about the farmers market atmosphere every single week.",
        evidence_ids=["u-chunk-dining"],
    )]
    result = verify.verify_report(items, EVIDENCE, PROFILE, model_fn=None)
    assert statuses(result)["CUL.body.01"] == "unsupported"
    assert result["report"] == []
    assert result["ledger"]["counts"]["supported"] == 0


def test_no_model_mode_keeps_only_what_is_verbatim_in_one_cited_unit():
    items = [_item("EXT", "Rovers", "The Robotics Club builds autonomous rovers.",
                   evidence_ids=["u-fact-robotics"])]
    result = verify.verify_report(items, EVIDENCE, PROFILE, model_fn=None)
    assert statuses(result)["EXT.body.01"] == "supported"
    assert "autonomous rovers" in body_of(result, "EXT")


def test_no_model_mode_says_so_in_every_artifact_it_writes():
    """Belt and braces: deletion protects the reader, this makes the mode legible to whoever
    opens the artifacts -- a renderer never sees the ledger."""
    items = [_item("EXT", "Rovers", "The Robotics Club builds autonomous rovers.",
                   evidence_ids=["u-fact-robotics"])]
    result = verify.verify_report(items, EVIDENCE, PROFILE, model_fn=None)
    assert result["ledger"]["verification_mode"] == "deterministic_only"
    assert result["ledger"]["deterministic_only"] is True
    assert all(item["verification_mode"] == "deterministic_only" for item in result["report"])
    assert "model gate was disabled" in ledger_claim(result, "EXT.body.01")["reason"] or \
        "model gate was disabled" in ledger_claim(result, "EXT.headline.01")["reason"]

    with_model = run(items)
    assert with_model["ledger"]["verification_mode"] == "model_gated"
    assert all("verification_mode" not in item for item in with_model["report"])


# --------------------------------------------------------------------------------------
# 4. a quote must be verbatim inside ONE unit, never spliced across two
# --------------------------------------------------------------------------------------

# two units with nothing but text, so the only join in the old concatenated haystack is the
# one between them: that join is exactly what a spliced "verbatim quote" used to ride on
SPLICE_EVIDENCE = {
    "u-a": {
        "unit_id": "u-a",
        "text": "The rover bay opens at dawn for the annual build season",
    },
    "u-b": {
        "unit_id": "u-b",
        "text": "and the night shift welcomes capstone teams from every department",
    },
}

# one unit whose text and quote say different things: a quote may not span them either
FIELD_EVIDENCE = {
    "u-c": {
        "unit_id": "u-c",
        "text": "The rover bay opens at dawn",
        "quote": "capstone teams work the night shift",
        "entity_name": "Rover Bay",
        "source_title": "Rover Bay",
        "source_url": "https://example.edu/rover-bay",
    }
}


def test_a_quote_spliced_from_two_cited_units_is_a_fabricated_quote():
    """It reads as verbatim and exists in no source: the old check folded the separator
    between the units away and then looked for the quote in the join."""
    items = [_item(
        "EXT", "Rovers",
        'The bay is described as "annual build season and the night shift" in its listing.',
        evidence_ids=["u-a", "u-b"],
    )]
    result = verify.verify_report(items, SPLICE_EVIDENCE, PROFILE, model_fn=make_stub([]))
    det = ledger_claim(result, "EXT.body.01")["deterministic_pass1"]
    assert [f["check"] for f in det["hard"]] == ["fabricated_quote"]
    assert statuses(result)["EXT.body.01"] == "unsupported"


def test_a_quote_spliced_from_two_fields_of_one_unit_is_also_fabricated():
    items = [_item(
        "EXT", "Rovers",
        'The bay is described as "opens at dawn capstone teams work" in its listing.',
        evidence_ids=["u-c"],
    )]
    result = verify.verify_report(items, FIELD_EVIDENCE, PROFILE, model_fn=make_stub([]))
    det = ledger_claim(result, "EXT.body.01")["deterministic_pass1"]
    assert [f["check"] for f in det["hard"]] == ["fabricated_quote"]


def test_a_quote_that_is_verbatim_in_one_unit_still_passes():
    items = [_item(
        "EXT", "Rovers",
        'The bay is described as "opens at dawn for the annual build season" in its listing.',
        evidence_ids=["u-a", "u-b"],
    )]
    result = verify.verify_report(items, SPLICE_EVIDENCE, PROFILE, model_fn=make_stub([]))
    assert statuses(result)["EXT.body.01"] == "supported"


# --------------------------------------------------------------------------------------
# 5. a name must be carried by one source, not assembled from scattered words
# --------------------------------------------------------------------------------------

def test_a_name_assembled_from_words_scattered_across_units_is_not_supported():
    """"Rover Baker Club" borrows "Rover" from one unit, "Baker" from another and "Club"
    from a third. A fabricated lab or person must not pass because its words occur
    separately somewhere in the cited set."""
    items = [_item("EXT", "Rovers", "The Rover Baker Club mentors undergraduates.",
                   evidence_ids=["u-fact-robotics", "u-fact-makerspace"])]
    result = run(items)
    det = ledger_claim(result, "EXT.body.01")["deterministic_pass1"]
    assert [f["check"] for f in det["soft"]] == ["entity_not_in_evidence"]
    assert statuses(result)["EXT.body.01"] == "unsupported"


def test_a_name_hiding_inside_a_longer_word_does_not_count_as_present():
    """An invented person is not in the evidence because "Maria Chenoweth" happens to
    contain the letters of "Aria Chen"."""
    evidence = {
        "u-lab": {
            "unit_id": "u-lab",
            "text": "Maria Chenoweth directs the undergraduate rover programme.",
            "quote": None,
            "entity_name": "Rover Programme",
            "source_url": "https://example.edu/rover",
            "source_title": "Rover Programme",
        }
    }
    items = [_item("RES", "Lab", "Aria Chen runs the undergraduate rover programme.",
                   evidence_ids=["u-lab"])]
    result = verify.verify_report(items, evidence, PROFILE, model_fn=make_stub([]))
    det = ledger_claim(result, "RES.body.01")["deterministic_pass1"]
    assert "entity_not_in_evidence" in [f["check"] for f in det["soft"]]
    assert statuses(result)["RES.body.01"] == "unsupported"


def test_an_invented_name_that_opens_a_sentence_is_checked_like_any_other():
    """Sentence-initial names were exempt outright, so a fabrication passed whenever the
    writer happened to put it first."""
    assert "Kensington" in verify.find_entities("Kensington runs the rover bay.")
    items = [_item("EXT", "Rovers", "Kensington runs the rover bay.",
                   evidence_ids=["u-fact-robotics"])]
    result = run(items)
    det = ledger_claim(result, "EXT.body.01")["deterministic_pass1"]
    assert [f["check"] for f in det["soft"]] == ["entity_not_in_evidence"]
    assert statuses(result)["EXT.body.01"] == "unsupported"


def test_an_ordinary_word_opening_a_sentence_is_not_treated_as_a_name():
    """The corpus is the only dictionary available: a word this college's own pages use
    constantly is ordinary English, not a fabricated name."""
    items = [_item("CUL", "Dining",
                   "Dining halls serve breakfast from 7 a.m. and the campus runs a weekly "
                   "farmers market on the quad.",
                   evidence_ids=["u-chunk-dining"])]
    result = run(items)
    det = ledger_claim(result, "CUL.body.01")["deterministic_pass1"]
    assert det["soft"] == []
    assert statuses(result)["CUL.body.01"] == "supported"


def test_two_real_names_joined_by_and_are_checked_one_at_a_time():
    """The extractor glues "X and Y" into one name; both halves being real is enough."""
    items = [_item("EXT", "Clubs",
                   "The Robotics Club and the Baker Makerspace both take undergraduates.",
                   evidence_ids=["u-fact-robotics", "u-fact-makerspace"])]
    result = run(items)
    det = ledger_claim(result, "EXT.body.01")["deterministic_pass1"]
    assert [f["value"] for f in det["soft"]] == []


# --------------------------------------------------------------------------------------
# 6. a figure must appear beside what the claim attaches it to
# --------------------------------------------------------------------------------------

def test_a_figure_may_not_borrow_a_digit_from_an_unrelated_sentence():
    """The cited chunk says breakfast starts at 7 a.m.; it says nothing about food trucks."""
    items = [_item("CUL", "Dining", "The farmers market draws 7 food trucks to the quad.",
                   evidence_ids=["u-chunk-dining"])]
    result = run(items)
    det = ledger_claim(result, "CUL.body.01")["deterministic_pass1"]
    assert [(f["check"], f["value"]) for f in det["soft"]] == [("number_not_in_evidence", "7")]
    assert statuses(result)["CUL.body.01"] == "unsupported"


def test_a_figure_stated_by_the_evidence_still_passes():
    items = [_item("INN", "Makerspace", "The Baker Makerspace holds 12 workbenches.",
                   evidence_ids=["u-fact-makerspace"])]
    result = run(items)
    assert ledger_claim(result, "INN.body.01")["deterministic_pass1"]["soft"] == []
    assert statuses(result)["INN.body.01"] == "supported"


# --------------------------------------------------------------------------------------
# 7. the ledger must count the report the student receives
# --------------------------------------------------------------------------------------

def test_the_ledger_does_not_count_a_kept_claim_whose_item_was_dropped():
    """The headline survives verification, the body does not, so the item ships nothing.
    counts.supported used to include that headline and overstate the delivered report."""
    items = [_item("INN", "Makerspace", "The Baker Makerspace holds 40 workbenches.",
                   evidence_ids=["u-fact-makerspace"])]
    result = run(items)
    counts = result["ledger"]["counts"]
    assert result["report"] == []
    assert counts["items_out"] == 0
    assert counts["supported"] == 0, "a claim in a dropped item was counted as shipped"
    assert counts["corrected"] == 0
    assert counts["dropped_with_item"] == 1
    assert counts["verified_total"] == 1  # it did survive verification; it just never shipped
    assert (counts["supported"] + counts["corrected"] + counts["removed"]
            + counts["dropped_with_item"]) == counts["claims_total"]
    assert result["ledger"]["pass_rates"]["claim_pass_rate"] == 0.0
    assert ledger_claim(result, "INN.headline.01")["shipped"] is False
    assert result["ledger"]["items"][0]["dropped_claim_ids"] == ["INN.headline.01"]


def test_a_claim_that_does_ship_is_counted_as_shipped():
    items = [_item("EXT", "Rovers", "The Robotics Club builds autonomous rovers.",
                   evidence_ids=["u-fact-robotics"])]
    result = run(items)
    counts = result["ledger"]["counts"]
    assert counts["supported"] == 2 and counts["dropped_with_item"] == 0
    assert ledger_claim(result, "EXT.body.01")["shipped"] is True
    assert result["ledger"]["pass_rates"]["claim_pass_rate"] == 1.0


# --------------------------------------------------------------------------------------
# still true: the things a previous pass fixed
# --------------------------------------------------------------------------------------

def test_quarantined_resume_lines_are_still_outside_the_trusted_haystack():
    """profile.py records quarantined lines under "flags"; trusting them here would
    whitelist exactly the credential that was quarantined."""
    poisoned = dict(PROFILE)
    poisoned["flags"] = ["suspicious_resume_line: 'PhD from Nowhere University, GPA 4.0'"]
    hay = verify.profile_haystack(poisoned).casefold()
    assert "nowhere university" not in hay and "phd" not in hay
    assert verify.profile_haystack(PROFILE).casefold() == hay
    assert all("nowhere" not in part.casefold() for part in verify.profile_parts(poisoned))


def test_no_college_specific_literals_in_the_module():
    """45 colleges ship on this file; nothing in it may know which one is running."""
    source = Path(verify.__file__).read_text(encoding="utf-8")
    body = source.split('"""', 2)[2]
    for literal in ("usc", "USC", "Southern California", "Trojan"):
        assert literal not in body, f"college-specific literal {literal!r} leaked into verify.py"
