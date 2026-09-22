"""Applicant level: getting it wrong writes the whole report for the wrong person.

`level` decides whether graduate-only material is filtered out of every category. Two real
client resumes were misclassified as graduate -- one because he had won a golf tournament
called the Bengal Junior Masters, one because her school clubs carried date ranges longer
than eighteen months. Both got reports full of Progressive Degree and M.S. programme content.

Run:
  /Users/chirag/college-intel/.venv-crawl4ai/bin/python -m pytest \
      /Users/chirag/college-intel/wwrag/tests/test_profile_level.py -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import profile as P  # noqa: E402


def level(text):
    return P.level_from_signals(P.grad_level_signals(text))[0]


# --------------------------------------------------------------------------------------
# "Masters" the tournament vs "Master's" the degree
# --------------------------------------------------------------------------------------

def test_a_tournament_called_masters_is_not_a_degree():
    assert not P.GRADUATE_DEGREE_RE.search("Bengal Junior Masters - Winner")
    assert not P.GRADUATE_DEGREE_RE.search("Interschool Masters Championship - 3rd")
    assert not P.GRADUATE_DEGREE_RE.search("Scrum Master certification")
    assert not P.GRADUATE_DEGREE_RE.search("Master of Ceremonies at the annual gala")


def test_real_graduate_credentials_still_read():
    for t in ["Master's degree in Economics", "Master's, Economics, 2019",
              "Master of Science in Computer Science", "MBA, Wharton",
              "Pursuing masters in finance", "Masters degree, LSE",
              "M.S. Statistics", "Ph.D. in Astrophysics"]:
        assert P.GRADUATE_DEGREE_RE.search(t), t


def test_a_golfers_resume_is_undergraduate():
    """Two tournament names used to be two corroborating 'signals', defeating the two-line rule."""
    resume = (
        "Aditya\nICSE: 87.4%\nAP Macroeconomics: 5\n"
        "Bengal Junior Masters - Winner\n"
        "Interschool Masters Championship - Over 13, Overall 3rd\n"
    )
    assert level(resume) == "undergraduate"


# --------------------------------------------------------------------------------------
# School activities are not employment
# --------------------------------------------------------------------------------------

def test_school_clubs_with_long_date_ranges_are_not_work_history():
    resume = (
        "Aadya\nCalcutta International School, ICSE\n"
        "President: CIS Math Honour Society  June 2024 - Current\n"
        "Led a club of 30+ individuals and founded a Tutoring Program.\n"
        "The Third Eye  Aug 2023 - Current\n"
        "Organised bi-monthly seminars on women's health with NGOs and schools.\n"
        "Student Council: Head of Alumni Relations  Aug 2024 - July 2025\n"
    )
    assert level(resume) == "undergraduate"
    assert not [s for s in P.grad_level_signals(resume) if s["kind"] == "work_history"]


def test_a_genuine_graduate_is_still_graduate():
    """The veto must not simply switch graduate detection off."""
    resume = (
        "Bachelor of Technology, 2014 - 2018\n"
        "Master's degree in Computer Science, 2019\n"
        "Software Engineer, Acme Corp  Jan 2019 - Dec 2022\n"
        "Senior Engineer, Globex  Jan 2023 - Present\n"
    )
    assert level(resume) == "graduate"


def test_school_markers_veto_work_runs_even_without_activity_words():
    resume = (
        "Grade 12, Delhi Public School\n"
        "Family Business Support  Mar 2022 - Dec 2024\n"
        "Assisted with day-to-day operations.\n"
        "Research Assistant  Jan 2023 - Present\n"
    )
    assert level(resume) == "undergraduate"


# --------------------------------------------------------------------------------------
# Short evidence lines: the most precise facts on a CV are the shortest
# --------------------------------------------------------------------------------------

def test_a_short_evidence_line_still_has_to_match_a_whole_word():
    """The length rule guarded against a tiny needle landing inside an unrelated word.
    Word-boundary matching keeps that guarantee without discarding the fact."""
    hay = P.match_key("Skills Computer Java Sports Rowing, Table-Tennis, Football, Golf")
    assert P.whole_token_match("java", hay)
    assert P.whole_token_match("golf", hay)
    assert not P.whole_token_match("row", hay)          # inside "Rowing"
    assert not P.whole_token_match("avas", hay)         # inside "Java"
    assert not P.whole_token_match("xyz", hay)
    assert not P.whole_token_match("", hay)


def test_java_is_not_mistaken_for_javascript():
    assert not P.whole_token_match("java", P.match_key("Proficient in javascript and typescript"))


def test_a_one_word_skill_survives_validation():
    """A real applicant's only stated technical skill was the single word "Java" on its own
    line. It was dropped while "Football" and "Golf" were kept, because those sat in a
    comma-separated list long enough to clear eight characters."""
    resume = "Skills & Hobbies\nComputer\nJava\nSports\nRowing, Table-Tennis, Football, Golf\n"
    flags = []
    kept = P.validate_labels(
        [{"label": "Java", "evidence_line": "Java"}],
        "skills", P.match_key(resume), [], flags,
    )
    assert kept == ["Java"], f"Java was dropped: {flags}"


def test_an_unquotable_short_line_is_still_dropped():
    flags = []
    kept = P.validate_labels(
        [{"label": "Rust", "evidence_line": "Rust"}],
        "skills", P.match_key("Skills: Java, Python"), [], flags,
    )
    assert kept == []
    assert any("Rust" in f for f in flags)
