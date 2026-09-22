"""Regression tests for wwrag/generate.py -- one per defect that shipped and was fixed.

Run:
  /Users/chirag/college-intel/.venv-crawl4ai/bin/python -m pytest \
      /Users/chirag/college-intel/wwrag/tests/test_generate.py -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import generate  # noqa: E402

# --------------------------------------------------------------------------------------
# Course codes: both numbering styles must unpack the same way
# --------------------------------------------------------------------------------------

def test_course_codes_handles_both_numbering_styles():
    """COURSE_CODE has two alternations, so findall yields 4-tuples; every caller must go
    through course_codes(). Unpacking the raw findall into two names crashed a whole run."""
    assert generate.course_codes("CSCI 102L and MATH 125g") == [("CSCI", "102L"), ("MATH", "125g")]
    assert generate.course_codes("take 18.06 Linear Algebra") == [("18", "06")]
    assert generate.course_codes("BUAD 280 plus 6.006") == [("BUAD", "280"), ("6", "006")]
    assert generate.course_codes("no codes here") == []


def test_course_level_tolerates_trailing_letters():
    assert generate.course_level("102L") == 102
    assert generate.course_level("599") == 599
    assert generate.course_level("06") == 6
    assert generate.course_level("L") is None


def test_graduate_only_uses_lettered_codes_only():
    assert generate.is_graduate_only({"text": "CSCI 599 Advanced Topics"}) is True
    assert generate.is_graduate_only({"text": "CSCI 102L Fundamentals"}) is False
    # a dotted number is a sequence within a department, never a year of study
    assert generate.is_graduate_only({"text": "18.06 Linear Algebra"}) is False


def test_repeatable_tokens_survives_dotted_codes():
    """The second unpack site: it never ran, because is_graduate_only crashed first."""
    toks = generate.specific_tokens({"text": "18.06 and CSCI 102L", "quote": "", "entity_name": ""})
    assert "CSCI 102L" in toks


# --------------------------------------------------------------------------------------
# The "names a findable thing" gate deletes items, so what it fails to recognise is lost
# --------------------------------------------------------------------------------------

def test_hyphenated_course_codes_count_as_naming_something():
    """TAC-449 was binned because the pattern allowed "TAC 449" but not "TAC-449"."""
    assert generate.names_a_findable_thing("TAC-449: Applications of Machine Learning builds models.")
    assert generate.names_a_findable_thing("CSCI 270 Introduction to Algorithms.")
    assert generate.names_a_findable_thing("Take 6.036 in the spring.")


def test_degrees_count_as_naming_something():
    """"Bachelor of Science in Artificial Intelligence" named no recognised noun and was binned."""
    assert generate.names_a_findable_thing("Bachelor of Science in Artificial Intelligence spans CS.")
    assert generate.names_a_findable_thing("She is doing a B.S. in Accounting alongside it.")


def test_ordinary_institution_nouns_count():
    for t in ["Undergraduate Research Matching List shows which faculty are taking students.",
              "USC Sustainability Hub hosts an e-waste drop-off each month.",
              "The Maseeh Entrepreneurship Prize Competition awards $100,000.",
              "Apply through the Office of Undergraduate Programs."]:
        assert generate.names_a_findable_thing(t), t


def test_genuinely_empty_prose_is_still_caught():
    """The gate must keep doing its job -- these are what it exists to delete."""
    for t in ["Students can get involved in many ways on campus.",
              "The university is committed to a positive campus culture.",
              "There are lots of opportunities here.",
              "The page lists an Undergraduate section."]:
        assert not generate.names_a_findable_thing(t), t
