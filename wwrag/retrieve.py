"""Hybrid retrieval: a StudentProfile + one college index -> EvidenceUnits per category.

This is the part the previous build got wrong, so the order of operations is the point:

    1. PLAN      several deterministic, template-built queries per category, derived from the
                 student's own activities / projects / interests / intended fields. How much of
                 a query is the category's intent and how much is the student is a PER-CATEGORY
                 setting on the CATEGORIES table (query_lead / query_weights), measured, not
                 guessed -- the table carries the measurement.
    2. FILTER    build the candidate row set FIRST: this college's index, the category's own
                 fact codes (+ GEN as supporting context), the kinds that category may use, and
                 -- for an undergraduate applicant -- undergraduate courses only, with no
                 graduate-only prose. Nothing outside that set can be returned, because nothing
                 outside it is ever scored.
    3. SEARCH    for every query, run both halves over the SAME pre-filtered candidate rows, with
                 a share of each query's depth reserved for the category's OWN fact codes (see
                 apply_own_code_quota: a thin category's own material is otherwise outnumbered
                 ~35:1 by its support codes and never reaches the scorer at all):
                 (a) dense vector search: the candidate rows are sliced out of vectors.npy first
                     and the dot product runs on that submatrix, so an excluded row is never
                     scored at all -- exact, so no approximate index and no post-filter,
                 (b) SQLite FTS5 BM25 keyword search, JOINed against a temp table holding the
                     candidate rows, so the LIMIT applies after the filter, never before.
                 Exact identifiers (course codes like "CSCI 350", professor names) survive
                 because the keyword half indexes them verbatim and course codes are turned into
                 adjacency phrases.
    4. FUSE      Reciprocal Rank Fusion across every query and both halves.
    5. RESCORE   multi-source and official-source preference, per-kind weights, organisation
                 fit scores, and recency decay for the categories where staleness matters.
    6. SELECT    greedy pick with diversity caps so one page, one entity or one host cannot
                 own a category.

The index is the one wwrag/index_build.py writes:

    <index-dir>/<college_id>/chunks.sqlite   units + units_fts   (FTS5, columns keyword_text, entity_name)
    <index-dir>/<college_id>/vectors.npy     float32 [n_units, dims], L2-normalised, row == units.vec_row
    <index-dir>/<college_id>/meta.json       counts, model, dims, layout

Nothing about a particular college lives in this file. College id, index path and the category
vocabulary come from CLI arguments or the CONFIG / CATEGORIES tables at the top.

The student profile and every retrieved string are DATA, never instructions. This module never
sends either to a model: queries are built from the profile by template, and profile text is
stripped of control characters and has every FTS5 operator neutralised before it is used.
Downstream modules that do put this text in a prompt must delimit it and say so.

Run (search):

    /Users/chirag/college-intel/.venv-crawl4ai/bin/python /Users/chirag/college-intel/wwrag/retrieve.py \
        --profile /tmp/profile.json \
        --index /Users/chirag/college-intel/wwrag/index \
        --out /tmp/evidence.json \
        --per-category 12 --explain

--index may be the index root (the college sub-directory is found from --college-id, or
auto-detected when the root holds exactly one index) or the college index directory itself.

Output is exactly the shared contract: {category_code: [EvidenceUnit, ...]} in the fixed category
order, every unit carrying score and retrieval={vector, keyword, rrf}. --explain additionally
writes <out>.explain.json and prints, per selected unit, which queries found it, its vector and
keyword ranks, the boosts applied, and which units the diversity caps pushed out.

Decisions the shared contract left open
---------------------------------------
* `retrieval.vector` is the best cosine similarity across the queries that retrieved the unit,
  0.0 when only the keyword half found it; `retrieval.keyword` is the best BM25 strength
  (negated, so larger is better), 0.0 when only the dense half found it; `retrieval.rrf` is the
  fused score before boosts and `score` is the same after them.
* A category whose pre-filter matches nothing comes back as [], loudly: a warning on stdout and
  an entry in explain["empty_categories"]. Generation must then write nothing for it rather than
  fill the gap. An index where EVERY category is empty raises instead -- that is a wrong index,
  not a thin college.
* GEN (and the other support_codes) may supply at most max_support_fraction of a category, so
  general facts can support a section but never become it.
* Undergraduate-appropriateness: courses use the index's is_undergraduate flag; facts, chunks and
  organisations use the GRADUATE_ONLY_MARKERS / UNDERGRADUATE_RESCUE_MARKERS heuristic below,
  evaluated in SQL at load time so it is a column, not a post-filter. It is deliberately
  conservative -- a row is dropped only when it looks graduate-only AND says nothing
  undergraduate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

import numpy as np

# The second (gap-filling) retrieval pass lives in its own module. It is imported by plain name
# because this file is run as a script and imported by tests with its own directory on the path;
# the guard covers the case where neither put it there.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import gapfill  # noqa: E402

# --------------------------------------------------------------------------------------
# Configuration. Nothing here names a college.
# --------------------------------------------------------------------------------------

RETRIEVE_SCHEMA_VERSION = 1

CONFIG: dict[str, Any] = {
    "embedding_model": "BAAI/bge-small-en-v1.5",  # must match the index; local + free, never paid
    "default_per_category": 12,
    # query planning
    "queries_min": 3,
    # HELD AT 6, and this number has been earned. An audit found a real bug: the facet lists
    # arrive alphabetically sorted, so slicing them fed queries whatever began with an early
    # letter -- a state-championship rower's interests read "Badminton, Chess, Cooking,
    # Football" and two of his four declared fields of study were never searched.
    #
    # Raising the cap to let every facet seed a query was the obvious fix and it was measured
    # and REVERTED TWICE. At 12 queries, finance evidence fell from 13 mentions to 2 and
    # entrepreneurship from 30 to 15; at 8 it was still worse than 6 on three measures of
    # four. The cause is the fusion: RRF sums 1/(k+rank) across queries, so a unit found
    # STRONGLY BY ONE query loses to generic material touched weakly by many. Adding queries
    # does not add coverage, it thins every slice.
    #
    # The truncation bug is therefore still open, and the fix is NOT more queries -- it is
    # per-query reserved slots, the mechanism gapfill.py already uses, so a query that finds
    # one excellent thing can seat it without out-voting the field.
    # 8, chosen by measurement, not intuition. An audit found the facet lists arrive
    # alphabetically sorted, so slicing them fed queries whatever began with an early letter:
    # a state-champion rower's interests read "Badminton, Chess, Cooking, Football" and two of
    # his four declared fields of study were never searched at all.
    #
    # Fixing it needs MORE query slots, but not unboundedly: at 12 the fusion dilutes, because
    # RRF sums 1/(k+rank) and a unit found strongly by ONE query loses to generic material
    # touched weakly by many. Measured across three real applicants, 8 queries with every
    # declared field searched gives 6-12 MORE distinct named entities per student than 6
    # queries with two fields, at the same nameability. 12 gives fewer than either.
    "queries_max": 8,
    "query_max_chars": 220,
    "profile_field_max_chars": 240,
    "profile_list_cap": 8,
    # per-query candidate depth for each half
    "vector_top_k": 80,
    # For every kind a category has a floor for, how many rows of that kind each query is
    # guaranteed in its vector list before fusion. See apply_kind_quota().
    "kind_quota_per_query": 6,
    # ANCHOR PASS. For each of the student's most substantive artefacts -- a paper, a founded
    # organisation, an internship -- the single closest things at the college, found by running
    # the artefact ITSELF as the query, unblended. See anchor_pass().
    "anchor_artefacts": 6,        # the student's most substantive lines (by length, not recurrence)
    "anchor_min_words": 12,       # a prize line is not an artefact; a described piece of work is
    "anchor_duplicate_sim": 0.80, # the same paper listed as project AND achievement is one artefact
    "anchor_per_query": 3,        # candidates per (artefact, frame); the best joins then compete
    "anchor_min_sim": 0.45,       # measured: every lexical coincidence sat at <= 0.43, every real join >= 0.46
    "anchor_min_hit_words": 8,    # "X is an alumni mentor" is not an anchor; a described fact is
    "anchor_max_per_chapter": 4,
    "anchor_max_total": 10,
    "keyword_top_k": 80,
    "fts_overfetch": 4,  # ask FTS for k*this before trimming: cheap insurance against ties
    # OWN-CODE QUOTA. A category's pool is its own fact codes PLUS its support codes PLUS every
    # untagged chunk/org/course, and those outnumber the own material badly: in the first shipped
    # index the thin categories hold 305, 1,088 and 1,934 facts against one category's 21,266, so
    # a whole-pool top-80 could contain nothing the category is actually about. The only quota we
    # had (max_support_fraction) runs during SELECTION, long after the pool was scored -- it can
    # drop support material that got in, but it cannot conjure back own-code rows that never made
    # the top-80. So each query's candidate depth reserves this share for the category's own fact
    # codes, taken from a second pass over the own-code rows alone (the "widened pool" a thin
    # category needs), merged by score. Lower it and thin categories quietly go generic again.
    "own_code_min_fraction": 0.5,
    "own_code_top_k": 80,  # depth of the own-code-only pass
    # fusion
    "rrf_k": 60,
    # How much a unit's score may come from queries OTHER than its best one. See fuse().
    "fusion_agreement_weight": 0.20,
    # A field the student (or their counsellor) DECLARED is the authoritative brief for the whole
    # report. A scraped hobby list is not. They carried identical query weight, so "Gender Studies"
    # and "Lawn Tennis, Oil painting, Basketball, Golf" were worth the same vote.
    "declared_field_boost": 2.0,
    "vector_weight": 1.0,
    "keyword_weight": 1.0,
    "bm25_column_weights": (1.0, 1.5),  # keyword_text, entity_name
    # rescoring
    "multi_source_step": 0.07,  # per extra corroborating source, capped below
    "multi_source_cap": 4,
    "org_fit_step": 0.10,  # per fit_score point (0-3) on the category's fit key
    # A category keyed on organisation fit is ABOUT that fit. An org scoring 0 on the key is not
    # weak evidence, it is the wrong club, and at multiplier 1.0 it tied with a scored one and the
    # kind floor then force-injected it anyway. So zero fit is a penalty, and the floor (below)
    # refuses to inject an org under org_fit_floor_min_score into a fit-keyed category.
    "org_fit_zero_penalty": 0.55,
    "org_fit_floor_min_score": 1,
    "recency_half_life_years": 5.0,
    "recency_floor": 0.55,
    "recency_unknown_year_factor": 0.90,
    # A year in the future is not freshness, it is a mention ("applications open for fall 2031").
    # Course catalogues legitimately run one year ahead of the build, so that much is allowed;
    # anything further is treated as undated rather than perfect, or a stale page that merely
    # names a future year outranks correctly dated current material.
    "recency_future_tolerance_years": 1,
    "reference_year": None,  # None -> the index's build year
    # diversity caps (per category)
    "max_per_source_page": 2,
    "max_per_entity": 2,
    # A diversity cap on hostnames, expressed as a FRACTION of the section, not a count.
    # As an absolute 4 it never bound here -- this college spreads across 550+ hostnames and
    # the largest holds under 10% of the index -- but a college that publishes everything on
    # one domain, which is the norm for liberal-arts and many mid-size universities, would
    # have had every one of its ten categories capped at four units out of twenty-four. The
    # cap exists to stop one SITE dominating, and at a single-site college there is no such
    # thing, so a fraction degrades correctly to "no constraint" instead of "throttle
    # everything".
    "placeholder_course_weight": 0.80,
    "process_boilerplate_weight": 0.85,
    "theme_weight_power": 0.5,
    # Measured against 109 targets named by independent reviewers: depth 1 scored 24 on-target,
    # depth 2 scored 23, depth 4 scored 23. Reaching deeper into one query costs another query its
    # seat, so breadth at rank 1 is the whole of the benefit.
    "query_slot_depth": 1,
    # Down-weighting alone did not hold. A large college republishes its university-wide
    # machinery -- combined bachelor's/master's schemes, change-of-major rules, advising -- on
    # every one of its schools' own sites. One such scheme arrived as six units from six
    # hostnames under six entity names, defeating max_per_host and max_per_entity together, and
    # took 6 of an accountancy applicant's 29 Academics slots. So the cap has to be on what the
    # unit is, not on where it came from.
    "max_process_boilerplate": 4,
    "max_per_host": 4,
    # KNOWN LIMIT, not yet fixed. As an absolute count this never binds here -- this college
    # publishes on 559 hostnames and its largest holds 9% of the index -- but a college that
    # serves everything from www.college.edu, which is normal for liberal-arts and many
    # mid-size universities, has ONE hostname and would have all ten categories capped at four
    # units out of twenty-four while the cap prevents nothing.
    #
    # Standing the cap down on a single-host index was tried and reverted: it breaks two
    # regression tests that guard REAL behaviour here, for a college we do not yet have. The
    # measurement it needs is already computed (index.distinct_hosts), so the fix is a few
    # lines -- but it should land alongside a real second college, with those tests reworked
    # deliberately rather than bent to fit a speculative change.
    "max_per_kind": {"chunk": 5},  # page prose must not crowd out named facts, clubs and courses
    "max_support_fraction": 0.25,  # share of a category that may be supporting-code context
    # a unit shorter than this proves nothing; excluded in the pre-filter
    "min_unit_chars": 40,
    # undergraduate gate
    "undergraduate_levels": ("undergraduate",),
}

# Graduate-only material must never be offered to a 16-18 year old applicant. Courses have an
# explicit flag; for facts, page chunks and organisations we use these markers, applied as part of
# the pre-filter. A row is dropped only when it looks graduate-only AND says nothing undergraduate.
#
# The markers are written with the ASCII apostrophe, and crawled prose mostly is not: publishing
# systems emit U+2019 ("master’s degree"), so the stored text is folded onto ASCII in SQL before
# these are matched. Without that fold the gate is decorative on any page with typography turned
# on -- in the first shipped index 121 graduate-only rows walked straight through it and were
# offerable to a 16-year-old. Keep the fold list short: each extra codepoint costs about a second
# of index-load time on a 100k-unit index, and these two are the ones crawled pages actually use.
APOSTROPHE_CODEPOINTS: tuple[int, ...] = (0x2019, 0x02BC)


def apostrophe_folded_sql(column: str) -> str:
    """SQL expression: `column` lowercased with every typographic apostrophe folded onto ASCII '.

    Used for the graduate gate, so a marker written "master's" also matches "master’s". If this
    ever goes back to a bare lower(column), the gate stops seeing most real pages.
    """
    expr = f"lower({column})"
    for codepoint in APOSTROPHE_CODEPOINTS:
        expr = f"replace({expr}, char({codepoint}), '''')"
    return expr


GRADUATE_ONLY_MARKERS: tuple[str, ...] = (
    "%master's program%",
    "%master's degree%",
    "%master of science%",
    "%master of arts%",
    "%master of business%",
    "%mba program%",
    "%m.b.a.%",
    "%ph.d. program%",
    "%phd program%",
    "%doctoral program%",
    "%doctoral students%",
    "%doctor of philosophy%",
    "%graduate program%",
    "%graduate students only%",
    "%graduate-only%",
    "%graduate certificate%",
    "%graduate admission%",
    "%postdoctoral%",
    "%post-doctoral%",
    "%executive education%",
    "%professional degree program%",
)

UNDERGRADUATE_RESCUE_MARKERS: tuple[str, ...] = (
    "%undergraduate%",
    "%bachelor%",
    "%b.s.%",
    "%b.a.%",
    "%first-year student%",
    "%freshman%",
    "%sophomore%",
    "%progressive degree%",
    "%4+1%",
)

# extra.is_undergraduate is written by whichever bundle built the index, and bundles disagree
# about how to spell a boolean: 1, true, "True", "yes". Reading only a couple of spellings meant
# an unrecognised one read as "not undergraduate", which drops EVERY course for an undergraduate
# applicant -- the whole course pool gone, no error, and the course kind floors quietly unmet.
# Accept the spellings a bundle plausibly writes, and return None (rather than False) when the
# value is unreadable, so the loader can say out loud that it did not understand the flag.
UNDERGRADUATE_FLAG_TRUE: frozenset[str] = frozenset(
    {"1", "true", "t", "yes", "y", "undergraduate", "undergrad", "ug", "bachelor", "bachelors"}
)
UNDERGRADUATE_FLAG_FALSE: frozenset[str] = frozenset(
    {"0", "false", "f", "no", "n", "graduate", "grad", "postgraduate", "none", ""}
)

UNIT_KINDS = ("fact", "chunk", "org", "course")

# The ten report categories, in fixed order. `fact_codes` are the category's own facts;
# `support_codes` may appear as supporting context but are capped. `kinds` is the pre-filter on
# unit kind. `intents` are static query stems; `facets` says which profile fields feed queries.
#
# query_lead / query_weights: HOW MUCH OF THE QUERY IS THE STUDENT. DO NOT COLLAPSE THESE INTO
# ONE GLOBAL KNOB. Every query blends the category's own intent with a phrase from the student's
# resume. Which of the two should lead was a single global setting (intent 1.0, student 0.9)
# until 60 blind judgments said one setting cannot be right for ten different questions. Same
# student, same index, two builds -- "student-led" (the student's words dominate the query) and
# "category-led" (the category's intent leads) -- scored 0-10 by 3 independent blind judges per
# category:
#
#     category                  student-led   category-led   margin   winner
#     CUL Culture                   5.0           6.0          1.0    category
#     EXT Extracurriculars          7.0           6.3          0.7    student
#     QRK Quirks                    2.3           3.3          1.0    category
#     ACA Academics                 4.7           3.0          1.7    student
#     RES Research                  7.0           4.0          3.0    student
#     SOC Social Impact             6.3           7.0          0.7    category
#     INN Innovative Programs       5.7           6.0          0.3    category
#     INT Intellectual Alignment    4.0           3.0          1.0    student
#     DIV Diversity                 3.3           5.7          2.4    category
#     NEW External Articles         5.7           2.0          3.7    student
#
# The split is CONCRETE vs ABSTRACT. Where the student's actual interests are what makes an
# answer useful -- which labs, which courses, which clubs, which press -- her words must steer
# the search, and leading with the category's lens instead returns the college's generic page
# about that topic. Where the category is abstract -- diversity, quirks, culture, service -- her
# words drag in wrong-topic material (a robotics resume pulls engineering pages into Diversity),
# so the category has to lead and her words only tilt the ranking.
#
# The numbers are mechanical, so they can be re-derived when the next measurement lands: the
# winning arm gets 1.00, and the losing arm gets 1.00 - 0.15 x margin, rounded to 0.05 and never
# below 0.45 (below that the other half of the query stops contributing anything at all).
#
# `query_lead` also decides the WORDING: a student-led category puts her phrase at the front of
# the dense query and the lens behind it, a category-led one does the reverse. The keyword half
# keeps the category's words in BOTH arms -- that fix (it used to search her resume text alone,
# with no idea which section it was filling) was right in both builds and is not what this table
# measures.
CATEGORIES: tuple[dict[str, Any], ...] = (
    {
        "code": "CUL",
        "name": "Culture",
        "fact_codes": ("CUL",),
        "support_codes": ("GEN",),
        "kinds": ("fact", "chunk", "org"),
        "intents": (
            "campus culture, traditions and the values students live by",
            "student life, community spirit and what the place feels like day to day",
            "school motto, mission and institutional character",
        ),
        # EVERY facet, every category. This used to be a short hand-picked list per category,
        # and it was acting as a hard filter when the ranking function below already exists to
        # choose. The cost was severe and invisible: `achievements` was read by exactly ONE of
        # the ten categories, so a student's awards, competition results and measured outcomes
        # were unreachable from nine sections; `activities` was missing from Research, so an
        # applicant whose strongest technical work was logged as an activity rather than a
        # project could never seed a Research query with it, and the lab that matched it was
        # never retrieved.
        #
        # rank_for_category() picks the two items most relevant to this category out of
        # whatever it is given, so a wider list costs nothing in query count -- it only stops
        # the pipeline discarding part of the student before the ranking ever runs. The facet
        # ORDER below still sets the tie-break, so each category keeps its emphasis.
        "facets": ("values", "interests", "activities", "intended_fields", "projects", "achievements", "skills"),
        "lens": "campus culture and traditions",
        # category-led 6.0 vs 5.0: an abstract question, and her words pull in wrong-topic pages
        "query_lead": "category",
        "query_weights": {"intent": 1.00, "student": 0.85},
        "kind_weights": {"fact": 1.00, "chunk": 0.90, "org": 0.92},
        "kind_floor": {},
        "source_kind_weights": {"official": 1.05, "affiliated": 1.00, "external": 0.95},
        "recency": False,
        "org_fit_key": None,
    },
    {
        "code": "EXT",
        "name": "Extracurriculars",
        "fact_codes": ("EXT",),
        "support_codes": ("GEN",),
        "kinds": ("fact", "chunk", "org"),
        "intents": (
            "student clubs and organizations undergraduates can join",
            "competition teams, performing groups and student-run projects",
            "how students get involved outside class",
        ),
        "facets": ("activities", "interests", "projects", "skills", "intended_fields", "achievements", "values"),
        "lens": "student clubs and organizations",
        # student-led 7.0 vs 6.3: which clubs suit HER is the whole answer here
        "query_lead": "student",
        "query_weights": {"intent": 0.90, "student": 1.00},
        "kind_weights": {"fact": 1.00, "chunk": 0.85, "org": 1.10},
        # Half the section, because this category IS the club list. A college's organisation
        # directory is the one source that names groups a student can actually join, and those
        # records carry no category code -- so they clear the pre-filter but count towards no
        # quota. A floor of three left the other twenty-one slots to facts that merely assert
        # clubs exist ("... has student-run clubs and organizations.", "The program offers
        # Student Organizations."). Measured on three real applicants at 3 vs 8 vs 12 named
        # organisations went 5/4/6 -> 14/15/13, and the best-matching group in the whole corpus
        # for an AI applicant -- the student branch of the campus AI-in-society centre -- does
        # not appear at all below 8.
        "kind_floor": {"org": 0.5},
        "source_kind_weights": {"official": 1.02, "affiliated": 1.00, "external": 0.98},
        "recency": False,
        "org_fit_key": None,
    },
    {
        "code": "QRK",
        "name": "Quirks",
        "fact_codes": ("QRK",),
        "support_codes": ("GEN", "CUL"),
        "kinds": ("fact", "chunk", "org"),
        "intents": (
            "unusual traditions, odd rituals and surprising campus lore",
            "quirky and offbeat student clubs",
            "strange superstitions, mascots and campus legends",
        ),
        "facets": ("interests", "activities", "intended_fields", "projects", "achievements", "skills", "values"),
        "lens": "quirky traditions and unusual clubs",
        # category-led 3.3 vs 2.3: quirk is the college's, not hers; her words only tilt the ranking
        "query_lead": "category",
        "query_weights": {"intent": 1.00, "student": 0.85},
        "kind_weights": {"fact": 1.00, "chunk": 0.88, "org": 1.05},
        "kind_floor": {"org": 2},
        "source_kind_weights": {"official": 1.00, "affiliated": 1.00, "external": 1.02},
        "recency": False,
        "org_fit_key": "quirky",
    },
    {
        "code": "ACA",
        "name": "Academics",
        "fact_codes": ("ACA",),
        "support_codes": ("GEN", "INT"),
        "kinds": ("fact", "chunk", "course"),
        "intents": (
            "undergraduate courses and the professors who teach them",
            "majors, minors and degree requirements for undergraduates",
            "small classes, seminars and how undergraduates are taught",
        ),
        "facets": ("intended_fields", "skills", "interests", "projects", "activities", "achievements", "values"),
        "lens": "undergraduate courses, majors and degree requirements",
        # student-led 4.7 vs 3.0: leading with the lens returned the generic teaching page, not her courses
        "query_lead": "student",
        "query_weights": {"intent": 0.75, "student": 1.00},
        "kind_weights": {"fact": 1.00, "chunk": 0.90, "course": 1.08},
        # A share of the section, not a count. Academics IS the course list, and an absolute
        # floor of 3 was the same trap Extracurriculars was in with clubs: 3 of 24 slots for
        # the one kind of thing the chapter exists to name, the other 21 to prose about degree
        # requirements. Raising the club floor to half the section took named clubs per student
        # from 5 to 14 -- the largest single gain measured on this pipeline.
        # Measured on two students: 0.25 seats 6 of 24 and loses nothing; 0.4 seats 10 but drops
        # a declared-field degree fact (the rank pass shrinks from 11 picks to 4).
        "kind_floor": {"course": 0.25},
        "source_kind_weights": {"official": 1.05, "affiliated": 1.00, "external": 0.92},
        "recency": True,
        "org_fit_key": None,
    },
    {
        "code": "RES",
        "name": "Research",
        "fact_codes": ("RES",),
        "support_codes": ("GEN", "ACA"),
        "kinds": ("fact", "chunk", "course", "org"),
        "intents": (
            "undergraduate research opportunities with named professors and labs",
            "laboratories, centers and institutes taking undergraduate researchers",
            "funded summer research programs for undergraduates",
        ),
        "facets": ("intended_fields", "projects", "skills", "interests", "activities", "achievements", "values"),
        "lens": "research labs and undergraduate research",
        # student-led 7.0 vs 4.0, the second-widest margin: her field is what makes a lab relevant
        "query_lead": "student",
        "query_weights": {"intent": 0.55, "student": 1.00},
        "kind_weights": {"fact": 1.00, "chunk": 0.92, "course": 0.95, "org": 0.95},
        # TRIED AND REVERTED: an entity-type floor of 0.3 for professor/person/lab/center.
        # Measured on two applicants it cost one of them three named targets (SURF and the Min
        # Family Challenge -- programmes, displaced by seven lab seats) and gained the other
        # nothing, because the economist she needed is not a graph node at all: 35% of graph
        # nodes have no edges and many people are missing outright. A type floor can only seat
        # what the graph has typed. Research needs its people found by the student's queries
        # (see reserved seats), not reserved by type.
        "kind_floor": {},
        "source_kind_weights": {"official": 1.05, "affiliated": 1.02, "external": 0.92},
        "recency": True,
        "org_fit_key": "research",
    },
    {
        "code": "SOC",
        "name": "Social Impact",
        "fact_codes": ("SOC",),
        "support_codes": ("GEN",),
        "kinds": ("fact", "chunk", "org"),
        # The corpus tags this category with sustainability, equity, literacy, mentoring and
        # K-12 outreach as well as volunteering -- and then the queries only ever asked about
        # volunteering. An applicant who founded an e-waste recycling venture and won an award
        # for it ranked his own venture FOURTH against this lens, behind a nature club and a
        # mentoring stint, because nothing here said "environment". The university's monthly
        # e-waste drive, sitting in the index under this very code, reached him zero times.
        "intents": (
            "community service, volunteering and local partnerships",
            "nonprofit work, social justice and civic engagement by students",
            "programs serving the surrounding neighborhoods",
            "sustainability, environmental action and recycling on campus",
        ),
        "facets": ("values", "activities", "interests", "achievements", "intended_fields", "projects", "skills"),
        "lens": "community service, sustainability and social impact",
        # category-led 7.0 vs 6.3, but narrowly: her service record still counts
        "query_lead": "category",
        "query_weights": {"intent": 1.00, "student": 0.90},
        "kind_weights": {"fact": 1.00, "chunk": 0.90, "org": 1.05},
        "kind_floor": {"org": 2},
        "source_kind_weights": {"official": 1.02, "affiliated": 1.02, "external": 1.00},
        "recency": False,
        "org_fit_key": "social_impact",
    },
    {
        "code": "INN",
        "name": "Innovative Programs",
        "fact_codes": ("INN",),
        "support_codes": ("GEN", "ACA"),
        "kinds": ("fact", "chunk", "course", "org"),
        "intents": (
            "signature and unusual programs found at few other universities",
            "interdisciplinary institutes, accelerators and maker spaces for undergraduates",
            "new initiatives, pilot programs and recent launches",
        ),
        "facets": ("intended_fields", "projects", "interests", "skills", "activities", "achievements", "values"),
        "lens": "distinctive and innovative programs",
        # category-led 6.0 vs 5.7, the narrowest margin in the table: nearly balanced
        "query_lead": "category",
        "query_weights": {"intent": 1.00, "student": 0.95},
        "kind_weights": {"fact": 1.00, "chunk": 0.92, "course": 0.98, "org": 0.95},
        "kind_floor": {},
        "source_kind_weights": {"official": 1.04, "affiliated": 1.00, "external": 0.96},
        "recency": True,
        "org_fit_key": None,
    },
    {
        "code": "INT",
        "name": "Intellectual Alignment",
        "fact_codes": ("INT",),
        "support_codes": ("GEN", "ACA", "CUL"),
        "kinds": ("fact", "chunk", "course"),
        "intents": (
            "academic philosophy, core curriculum and how the school thinks about learning",
            "interdisciplinary thinking, great books and general education requirements",
            "intellectual debate, writing and the life of the mind on campus",
        ),
        "facets": ("interests", "values", "intended_fields", "activities", "projects", "achievements", "skills"),
        "lens": "academic philosophy and intellectual life",
        # student-led 4.0 vs 3.0: alignment is with HER mind, so her words lead
        "query_lead": "student",
        "query_weights": {"intent": 0.85, "student": 1.00},
        "kind_weights": {"fact": 1.00, "chunk": 0.95, "course": 1.00},
        "kind_floor": {},
        "source_kind_weights": {"official": 1.04, "affiliated": 1.00, "external": 0.96},
        "recency": False,
        "org_fit_key": None,
    },
    {
        "code": "DIV",
        "name": "Diversity of Community",
        "fact_codes": ("DIV",),
        "support_codes": ("GEN", "CUL"),
        # courses too: a gender-and-sexuality department is mostly COURSES, and
        # excluding that kind hid 49 of its 59 units from this chapter before
        # any search ran.
        "kinds": ("fact", "chunk", "org", "course"),
        "intents": (
            "support for international students and students from different backgrounds",
            "cultural centers, affinity groups and first-generation student support",
            "how the university builds an inclusive undergraduate community",
        ),
        "facets": ("values", "interests", "activities", "intended_fields", "projects", "achievements", "skills"),
        "lens": "diversity, belonging and support for students from every background",
        # category-led 5.7 vs 3.3: her resume dragged her field into a question that is not about it
        "query_lead": "category",
        "query_weights": {"intent": 1.00, "student": 0.65},
        "kind_weights": {"fact": 1.00, "chunk": 0.92, "org": 1.05},
        "kind_floor": {"org": 2},
        "source_kind_weights": {"official": 1.03, "affiliated": 1.00, "external": 0.98},
        "recency": False,
        "org_fit_key": "diversity_community",
    },
    {
        "code": "NEW",
        "name": "External Articles and References",
        "fact_codes": ("NEW",),
        "support_codes": (),
        "kinds": ("fact", "chunk"),
        "intents": (
            "recent news coverage of the university",
            "independent reporting, rankings and outside assessments",
            "awards, discoveries and controversies reported by the press",
        ),
        "facets": ("intended_fields", "interests", "values", "activities", "projects", "achievements", "skills"),
        "lens": "news coverage and outside reporting",
        # student-led 5.7 vs 2.0, the widest margin: generic press coverage is what nobody wanted
        "query_lead": "student",
        "query_weights": {"intent": 0.45, "student": 1.00},
        "kind_weights": {"fact": 1.00, "chunk": 0.90},
        "kind_floor": {},
        # the point of this category is the outside voice
        "source_kind_weights": {"official": 0.85, "affiliated": 1.00, "external": 1.20},
        "recency": True,
        "org_fit_key": None,
        # extra pre-filter: either the fact is tagged NEW, or it comes from a non-official source
        "require_external_or_own_code": True,
    },
)

CATEGORY_BY_CODE: dict[str, dict[str, Any]] = {c["code"]: c for c in CATEGORIES}
CATEGORY_ORDER: tuple[str, ...] = tuple(c["code"] for c in CATEGORIES)

# Dropped from keyword queries. Deliberately small: FTS5 already uses IDF, and over-stripping
# hurts phrase matching.
STOPWORDS = frozenset(
    """a an and are as at be been but by for from had has have how i in into is it its of on or our
    that the their them there these they this to was were what when where which who will with you your
    about also can could do does my me we us""".split()
)

# 3-5 digits, and a dotted numeric form. Three digits is one college's convention and
# nobody else's:
# Cornell CS 4780, Georgia Tech CS 1301, Penn CIS 5200 and Northeastern CS 3000 have four;
# Purdue CS 18000 has five; Brown CSCI 0111 and Swarthmore CPSC 021 start with a zero; MIT
# writes 6.036, where the department itself is a number. Under the old pattern none of them
# is a course code at all, so at those colleges nothing names a course and every check that
# asks 'did this item name something specific' silently answers no.
# Courses that are a MECHANISM rather than a subject: "Cooperative Education Work Experience",
# "Directed Research", "Special Topics", "Internship". Their descriptions are deliberately
# generic ("Supervised work experience", "Individual research and readings"), which is exactly
# what makes them match any student's query -- ENGR 395A surfaced for an AI-and-finance
# applicant and for an accountancy applicant in the same run, and told neither of them
# anything. They are 8.6% of the undergraduate course pool and were taking 1-2 of the 3-4
# course slots a student actually gets.
#
# Down-weighted, not banned: "CSCI 490 Directed Research" is a real and useful answer in a
# Research chapter, where the mechanism IS the point. It just must not out-rank a course with
# actual content in Academics.
PLACEHOLDER_COURSE = re.compile(
    r"\b(cooperative education|directed research|independent study|special topics"
    r"|internship|senior thesis|honors thesis|honou?rs research|undergraduate research"
    r"|research seminar|work experience|directed reading|field ?work|practicum)\b", re.I)

# The machinery of being enrolled, as opposed to what you can actually study. Measured across the
# four client resumes, this was 25-59% of every student's Academics slots: six units on Progressive
# Degree mechanics, change-of-major advising, "the best source is the course catalogue", "the page
# lists an Undergraduate section". It scores well because it uses the same words the category's
# queries do -- degree, course, requirement, undergraduate -- while naming nothing you can choose.
# Down-weighted rather than dropped: at a college whose pages are mostly process, something has to
# fill the section.
PROCESS_BOILERPLATE = re.compile(
    r"\b(progressive degree|chang(?:e|ing)\s+(?:of\s+)?major|academic advis(?:or|ors|ing|ement)"
    r"|advising appointment|degree progress|transcript request|d-?clearance"
    r"|registration (?:process|period|opens)|best source for|refer to the"
    r"|see the catalogu?e|for more information|learn more about"
    r"|the page lists|this page (?:lists|shows|contains)"
    # sentences that assert a listing exists and name nothing inside it
    r"|the program offers (?:student organi[sz]ations|undergraduate research)"
    r"|has student-run clubs|are listed as (?:a resource|other resources)"
    r"|may hire undergraduates as research assistants)\b", re.I)

# A line that is a piece of intellectual work -- the only kind a faculty search makes sense for.
# Genre words, not subject words: a paper, a study, a seminar; never "economics" or "AI".
RESEARCH_MARKER = re.compile(
    r"\b(?:paper|published|publication|research|study|studies|conference|journal|presented|"
    r"seminars?|thesis|experiment|analysis|survey|abstract|findings|hypothesis|dataset)\b", re.I)

PERSON_MARKER = re.compile(
    r"\b(?:professor|Professor|researcher|economist|scientist|Dr\.|PhD|Ph\.D)\b"
    r"|\b(?:conducts|researches|studies|investigates|examines|directs|leads)\b")

COURSE_CODE_RE = re.compile(r"\b([A-Za-z]{2,5})\s*[-– ]?\s*(\d{3,5}[A-Za-z]?)\b")
WORD_RE = re.compile(r"[A-Za-z0-9']+")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------


def die(msg: str) -> None:
    """Fail loudly. Never return a silently empty result set."""
    raise SystemExit(f"retrieve: ERROR: {msg}")


def log(msg: str) -> None:
    print(msg, flush=True)


def clean_text(value: Any, max_chars: int) -> str:
    """Profile / query text is DATA. Normalise it, strip control characters, cap the length."""
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    value = unicodedata.normalize("NFKC", value)
    value = CONTROL_RE.sub(" ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value[:max_chars].strip()


def read_json(path: Path) -> Any:
    if not path.exists():
        die(f"missing file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        die(f"{path} is not valid JSON: {exc}")


def as_int(value: Any) -> int | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def host_of(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").lower()
    except ValueError:
        return ""


def normalise_host(host: str) -> str:
    """One site, one cap bucket.

    max_per_host only means anything if a site has exactly one key. Facts carry
    extra.source_host, which the bundle already stripped of "www.", while every other kind falls
    back to host_of(url), which keeps it -- so one site held two buckets and quietly supplied
    twice its cap. Strip "www." (and any port) on both paths so the two agree.
    """
    host = (host or "").strip().lower()
    if "@" in host:  # userinfo, should never appear, but it must not become part of the key
        host = host.rsplit("@", 1)[1]
    if host.startswith("www."):
        host = host[4:]
    if ":" in host and not host.startswith("["):
        host = host.split(":", 1)[0]
    return host


def canonical_page_url(url: str) -> str:
    """Canonical key for "the same page": host without www., path without a trailing slash.

    Also one page, one bucket. Facts carry no extra.page_id and chunks do, so keying facts on the
    URL and chunks on the page id gave the same page TWO buckets and 2 x max_per_source_page
    slots -- four passages off one page, which reads as four findings and is one. Both kinds key
    on this now. Scheme and fragment are dropped; the query string is kept, because pages that
    differ only by query really are different pages.
    """
    url = (url or "").strip()
    if not url:
        return ""
    try:
        parts = urlparse(url)
    except ValueError:
        return url.lower()
    host = normalise_host(parts.netloc)
    if not host:  # not an absolute URL: keep it whole rather than invent a key
        return url.lower()
    path = (parts.path or "").rstrip("/")
    query = f"?{parts.query}" if parts.query else ""
    return f"{host}{path}{query}".lower()


def read_bool_flag(value: Any) -> bool | None:
    """Read a bundle's boolean however that bundle spelled it. None = unreadable / not stated.

    The undergraduate course gate turns on this. A value it cannot read must NOT silently read as
    False: that drops every course for an undergraduate applicant, which is the whole pool for
    the categories with a course floor. None is how the loader knows to say so out loud.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in UNDERGRADUATE_FLAG_TRUE:
            return True
        if text in UNDERGRADUATE_FLAG_FALSE:
            return False
    return None


# --------------------------------------------------------------------------------------
# the index
# --------------------------------------------------------------------------------------


def resolve_index_dir(index_arg: Path, college_id: str | None) -> Path:
    """Accept either the index root (<root>/<college_id>/) or a college index directory."""
    index_arg = Path(index_arg).expanduser()
    if not index_arg.exists():
        die(f"index directory does not exist: {index_arg}")
    if (index_arg / "meta.json").exists():
        return index_arg
    if college_id:
        candidate = index_arg / college_id
        if (candidate / "meta.json").exists():
            return candidate
        die(f"no index for college {college_id!r} under {index_arg} (expected {candidate}/meta.json)")
    if not index_arg.is_dir():
        die(f"{index_arg} is not a directory")
    children = sorted(p for p in index_arg.iterdir() if p.is_dir() and (p / "meta.json").exists())
    if len(children) == 1:
        return children[0]
    if not children:
        # A build in flight leaves vectors.npy.building behind; say so rather than "no index".
        building = sorted(
            p.parent for p in index_arg.glob("**/vectors.npy.building") if p.is_file()
        )
        if building:
            names = ", ".join(str(p) for p in building[:3])
            die(
                f"{index_arg} has no finished index yet: an embedding stage is still running in "
                f"{names} (vectors.npy.building). Wait for index_build.py to write meta.json."
            )
        die(
            f"{index_arg} holds no index (no meta.json here or one level down). "
            f"Build one with wwrag/index_build.py first."
        )
    names = ", ".join(p.name for p in children)
    die(f"{index_arg} holds several indexes ({names}); pass --college-id to choose one")
    raise AssertionError("unreachable")


class CollegeIndex:
    """Read-only view of one college's index, with the light per-unit columns held in memory.

    Unit text is NOT loaded up front (page chunks are large); it is fetched for the few hundred
    units that survive selection.
    """

    def __init__(self, index_dir: Path, expect_model: str | None = None) -> None:
        self.dir = Path(index_dir)
        self.meta: dict[str, Any] = read_json(self.dir / "meta.json")
        self.college_id: str = str(self.meta.get("college_id") or "")
        if not self.college_id:
            die(f"{self.dir}/meta.json has no college_id")
        if self.meta.get("complete") is False:
            die(f"index {self.dir} is incomplete (meta.complete is false); finish index_build.py first")

        layout = self.meta.get("layout") or {}
        db_name = layout.get("sqlite", "chunks.sqlite")
        vec_name = layout.get("vectors", "vectors.npy")
        self.fts_table = layout.get("fts_table", "units_fts")
        self.units_table = layout.get("units_table", "units")

        db_path = self.dir / db_name
        vec_path = self.dir / vec_name
        for path in (db_path, vec_path):
            if not path.exists():
                die(
                    f"index {self.dir} is missing {path.name}; run wwrag/index_build.py "
                    f"(a vectors.npy.building file means the embedding stage is still running)"
                )

        embedding = self.meta.get("embedding") or {}
        self.model_name: str = str(embedding.get("model") or CONFIG["embedding_model"])
        if expect_model and self.model_name != expect_model:
            die(
                f"index was built with embedding model {self.model_name!r} but this run wants "
                f"{expect_model!r}; vectors from different models are not comparable"
            )
        self.dims = int(embedding.get("dims") or 0)

        self.conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA temp_store=MEMORY")

        self.vectors = np.load(vec_path, mmap_mode="r")
        if self.vectors.ndim != 2:
            die(f"{vec_path} is not a 2-D array")
        if self.dims and self.vectors.shape[1] != self.dims:
            die(f"{vec_path} has {self.vectors.shape[1]} dims, meta.json says {self.dims}")
        self.dims = int(self.vectors.shape[1])

        self._load_columns()

        n_meta = as_int((self.meta.get("counts") or {}).get("units"))
        if n_meta is not None and n_meta != self.n_units:
            die(f"meta.json says {n_meta} units but the units table holds {self.n_units}")
        if self.vectors.shape[0] != self.n_units:
            die(f"vectors.npy has {self.vectors.shape[0]} rows but the units table holds {self.n_units}")

        ref = CONFIG["reference_year"]
        self.reference_year = int(ref) if ref else self._default_reference_year()

    # -- loading ------------------------------------------------------------------------

    def _load_columns(self) -> None:
        """One pass over the units table: light columns + the undergraduate gate, computed in SQL.

        The graduate-only test and the length gate run in SQL so that page-chunk text -- tens of
        megabytes of it -- never has to come into Python.
        """
        # the markers are ASCII-apostrophe; the stored text is folded onto ASCII first, or a page
        # that writes "master’s degree" with real typography sails through the graduate gate
        gate_text = apostrophe_folded_sql("text")
        grad_expr = " OR ".join([f"{gate_text} LIKE ?"] * len(GRADUATE_ONLY_MARKERS))
        rescue_expr = " OR ".join([f"{gate_text} LIKE ?"] * len(UNDERGRADUATE_RESCUE_MARKERS))
        sql = f"""
            SELECT vec_row, unit_id, kind, category_code, entity_name, source_url, source_kind, year,
                   length(text) AS text_len,
                   json_extract(extra, '$.source_count')     AS source_count,
                   json_extract(extra, '$.source_host')      AS source_host,
                   json_extract(extra, '$.page_id')          AS page_id,
                   json_extract(extra, '$.dept')             AS dept,
                   json_extract(extra, '$.is_undergraduate') AS is_undergraduate,
                   json_extract(extra, '$.fit_scores')       AS fit_scores,
                   CASE WHEN ({grad_expr}) AND NOT ({rescue_expr}) THEN 1 ELSE 0 END AS grad_only
              FROM {self.units_table}
             ORDER BY vec_row
        """
        params = list(GRADUATE_ONLY_MARKERS) + list(UNDERGRADUATE_RESCUE_MARKERS)
        rows = self.conn.execute(sql, params).fetchall()
        if not rows:
            die(f"index {self.dir} has no units; refusing to search an empty index")

        n = len(rows)
        self.n_units = n
        self.unit_id: list[str] = [""] * n
        self.kind: list[str] = [""] * n
        self.category_code: list[str | None] = [None] * n
        self.entity_name: list[str | None] = [None] * n
        self.distinct_hosts: int = 0           # how many hostnames this college publishes on
        self.source_url: list[str] = [""] * n
        self.source_kind: list[str] = [""] * n
        self.source_host: list[str] = [""] * n
        self.page_key: list[str] = [""] * n
        self.entity_key: list[str] = [""] * n
        self.year = np.full(n, -1, dtype=np.int32)
        self.text_len = np.zeros(n, dtype=np.int32)
        self.source_count = np.ones(n, dtype=np.int16)
        self.grad_only = np.zeros(n, dtype=np.uint8)
        self.is_undergrad_course = np.zeros(n, dtype=np.uint8)
        self.fit_scores: dict[int, dict[str, int]] = {}
        n_courses = 0
        unreadable_course_flags = 0

        for i, row in enumerate(rows):
            vec_row = int(row["vec_row"])
            if vec_row != i:
                die(f"units.vec_row is not dense/0-based at row {i} (saw {vec_row}); rebuild the index")
            self.unit_id[i] = row["unit_id"]
            kind = row["kind"]
            self.kind[i] = kind
            self.category_code[i] = row["category_code"]
            entity = row["entity_name"]
            self.entity_name[i] = entity
            url = row["source_url"] or ""
            self.source_url[i] = url
            self.source_kind[i] = row["source_kind"] or "official"
            # normalised on BOTH paths: extra.source_host has no "www.", host_of(url) keeps it,
            # and a site with two host keys gets two helpings of max_per_host
            self.source_host[i] = normalise_host(row["source_host"] or host_of(url))
            self.text_len[i] = int(row["text_len"] or 0)
            year = as_int(row["year"])
            if year is not None:
                self.year[i] = year
            count = as_int(row["source_count"])
            if count is not None and count > 0:
                self.source_count[i] = min(count, 32767)
            self.grad_only[i] = 1 if row["grad_only"] else 0
            if kind == "course":
                n_courses += 1
                # whatever this bundle spells the flag as; an unreadable value is counted and
                # reported below rather than read as "not undergraduate" in silence
                flag = read_bool_flag(row["is_undergraduate"])
                if flag is None:
                    unreadable_course_flags += 1
                elif flag:
                    self.is_undergrad_course[i] = 1

            # diversity keys. Facts and the chunks of the same page must share one page key --
            # see canonical_page_url: keying them differently handed one page four slots.
            page_id = row["page_id"]
            if url:
                self.page_key[i] = f"url:{canonical_page_url(url)}"
            elif page_id:
                self.page_key[i] = f"page:{page_id}"
            else:
                self.page_key[i] = f"unit:{row['unit_id']}"
            if kind == "course":
                dept = row["dept"]
                self.entity_key[i] = f"dept:{str(dept).lower()}" if dept else f"unit:{row['unit_id']}"
            elif entity:
                self.entity_key[i] = f"ent:{entity.strip().lower()}"
            else:
                self.entity_key[i] = f"unit:{row['unit_id']}"

            if kind == "org" and row["fit_scores"]:
                try:
                    scores = json.loads(row["fit_scores"])
                except (TypeError, ValueError, json.JSONDecodeError):
                    scores = None
                if isinstance(scores, dict):
                    self.fit_scores[i] = {k: as_int(v) or 0 for k, v in scores.items()}

        # Does this bundle score organisation fit AT ALL? A 0 means "not this kind of club" only
        # when the bundle computes fit; in a bundle that never computed any, every org scores 0
        # and a fit gate would silently empty the org pool of every fit-keyed category. So the
        # gate in select_units turns itself off (loudly) when nothing here is scored.
        # Courses that are a mechanism rather than a subject, resolved once per index so the
        # scoring loop is a set lookup rather than a text fetch per hit.
        self.placeholder_courses: set[int] = {
            int(r[0]) for r in self.conn.execute(
                "SELECT vec_row, text FROM units WHERE kind = 'course'"
            ) if PLACEHOLDER_COURSE.search(r[1] or "")
        }

        # Units about the process of being enrolled rather than about anything a student could
        # choose. Resolved once here for the same reason as the placeholder courses above.
        self.process_boilerplate: set[int] = {
            int(r[0]) for r in self.conn.execute("SELECT vec_row, text FROM units")
            if PROCESS_BOILERPLATE.search(r[1] or "")
        }

        # What kind of thing each unit's entity is, from the graph (professor, lab, org...), so a
        # category can reserve seats for a TYPE of entity -- Research for people and labs -- the
        # way Academics does for courses and Extracurriculars for clubs. Empty when the index has
        # no graph; nothing below depends on it being present.
        self.entity_type: list[str] = [""] * n
        self.foreign_institution: set[int] = set()
        self.home_institution: set[str] = set()
        try:
            own = {r[0] for r in self.conn.execute(
                "SELECT name_key FROM graph_nodes WHERE type = 'university' "
                "ORDER BY degree DESC LIMIT 3")}
            self.home_institution = set(own)
            foreign = {r[0] for r in self.conn.execute(
                "SELECT name_key FROM graph_nodes WHERE type = 'university'")} - own
            type_of: dict[str, str] = {}
            for name_key, typ in self.conn.execute("SELECT name_key, type FROM graph_nodes"):
                type_of.setdefault(name_key, typ)
            for i, name in enumerate(self.entity_name):
                if not name:
                    continue
                k = str(name).strip().lower()
                self.entity_type[i] = type_of.get(k, "")
                # A fact whose subject is another university is in this index by accident of the
                # crawl (a professor's previous employer, a news comparison). It must never take a
                # reserved seat: one about the University of Michigan did, in a real report.
                if k in foreign:
                    self.foreign_institution.add(i)
        except Exception:  # noqa: BLE001 - no graph, no types; floors on types simply find nothing
            pass

        # Facts that describe a PERSON'S research or role -- the rows an "anchor" search for
        # faculty runs against. Marked by the grammar of such a sentence, never by a name.
        by_grammar = {int(r[0]) for r in self.conn.execute(
            "SELECT vec_row, text FROM units WHERE kind = 'fact'")
            if PERSON_MARKER.search(r[1] or "")}
        by_graph: set[int] = set()
        graph_people: set[int] = set()
        try:  # the graph sharpens both masks; an index without one still gets the grammar
            by_graph = {int(r[0]) for r in self.conn.execute(
                "SELECT u.vec_row FROM units u JOIN graph_nodes g ON lower(u.entity_name) = g.name_key "
                "WHERE u.kind = 'fact' AND g.type IN ('person', 'professor')")}
            graph_people = {int(r[0]) for r in self.conn.execute(
                "SELECT u.vec_row FROM units u JOIN graph_nodes g ON lower(u.entity_name) = g.name_key "
                "WHERE g.type IN ('person', 'professor', 'university')")}
        except Exception:  # noqa: BLE001
            pass
        self.person_rows: np.ndarray = np.array(sorted(by_grammar | by_graph), dtype=np.int64)
        # rows that NAME a thing -- an anchor hit must be nameable or the writer cannot use it.
        # People belong to the faculty search above; the college itself names nothing.
        self.nameable_rows: np.ndarray = np.array(sorted(
            int(r[0]) for r in self.conn.execute(
                "SELECT vec_row, entity_name FROM units WHERE kind IN ('org', 'course') "
                "OR (entity_name IS NOT NULL AND entity_name <> '')")
            if int(r[0]) not in graph_people
            and str(r[1] or "").strip().lower() not in self.home_institution), dtype=np.int64)

        # How many hostnames this college publishes on. One means a per-hostname cap cannot
        # discriminate between sites and must not be applied.
        self.distinct_hosts = len({h for h in self.source_host if h})

        self.fit_scored_orgs = sum(
            1 for scores in self.fit_scores.values() if any(int(v) > 0 for v in scores.values())
        )

        self.kind_arr = np.array(self.kind, dtype=object)
        self.entity_type_arr = np.array(self.entity_type, dtype=object)
        self.category_arr = np.array([c or "" for c in self.category_code], dtype=object)
        self.source_kind_arr = np.array(self.source_kind, dtype=object)

        # SAY IT OUT LOUD. An index whose courses carry no flag this build can read loses its
        # entire course pool at the undergraduate gate, and until this warning existed that
        # happened in perfect silence: the course floors for the course-using categories simply
        # went unmet and the report came back all prose. 45 colleges ship on this code and their
        # bundles will not all agree on how to write a boolean.
        self.n_courses = n_courses
        self.n_undergraduate_courses = int(self.is_undergrad_course.sum())
        self.unreadable_course_flags = unreadable_course_flags
        if n_courses and not self.n_undergraduate_courses:
            log(
                f"  WARNING index {self.dir} holds {n_courses:,} courses and NOT ONE is flagged "
                f"undergraduate ({unreadable_course_flags:,} carried an extra.is_undergraduate "
                f"value this build could not read). Every course will be filtered out for an "
                f"undergraduate applicant -- check how this bundle spells the flag."
            )
        elif unreadable_course_flags:
            log(
                f"  WARNING index {self.dir}: {unreadable_course_flags:,} of {n_courses:,} courses "
                f"carry an unreadable extra.is_undergraduate value and are treated as graduate"
            )

    def _default_reference_year(self) -> int:
        built = str(self.meta.get("built_at") or "")
        match = re.match(r"(\d{4})", built)
        if match:
            return int(match.group(1))
        observed = int(self.year.max()) if self.n_units else 0
        return observed if observed > 0 else 2000

    def close(self) -> None:
        self.conn.close()

    # -- text for the selected few -------------------------------------------------------

    def fetch_text(self, vec_rows: Sequence[int]) -> dict[int, sqlite3.Row]:
        out: dict[int, sqlite3.Row] = {}
        rows = [int(r) for r in vec_rows]
        for start in range(0, len(rows), 500):
            block = rows[start : start + 500]
            marks = ",".join("?" * len(block))
            sql = (
                f"SELECT vec_row, text, quote, source_title FROM {self.units_table} "
                f"WHERE vec_row IN ({marks})"
            )
            for row in self.conn.execute(sql, block):
                out[int(row["vec_row"])] = row
        missing = [r for r in rows if r not in out]
        if missing:
            die(f"units table is missing vec_rows {missing[:5]}; the index is inconsistent")
        return out


# --------------------------------------------------------------------------------------
# the student profile (DATA)
# --------------------------------------------------------------------------------------


def load_profile(path: Path) -> dict[str, Any]:
    """Load a StudentProfile and validate the fields retrieval depends on. Fails loudly."""
    profile = read_json(Path(path))
    if not isinstance(profile, dict):
        die(f"{path} does not hold a StudentProfile object")
    student_id = clean_text(profile.get("student_id"), 120)
    if not student_id:
        die(f"{path}: StudentProfile has no student_id")
    level = clean_text(profile.get("level"), 40).lower()
    if level not in ("undergraduate", "graduate"):
        die(f"{path}: StudentProfile.level must be 'undergraduate' or 'graduate', got {level!r}")
    return profile


def _corpus_stopwords(index: "CollegeIndex | None") -> set[str]:
    """Words that identify the college itself, and so cannot discriminate inside its corpus.

    A resume often names the college -- a summer programme, a research internship, an alum
    parent. Those words then lead the query, and searching one college's own pages for its
    own name matches everything equally: the distinctive part of the phrase gets drowned.
    Measured case: an applicant whose strongest technical work was an MQTT sensor network had
    that activity stored under the name "University of Southern California". The query led
    with the college's name, and the top hits became outreach pages and a high-school summer
    programme, while the lab that actually does wireless sensor networking -- the top hit for
    the same idea phrased cleanly -- was never retrieved at all.

    The names come from the index's own metadata, so nothing here is specific to a college.
    """
    if index is None:
        return set()
    words: set[str] = set()
    meta = getattr(index, "meta", {}) or {}
    candidates = [meta.get("college_name"), meta.get("college_id")]
    aliases = meta.get("aliases") or meta.get("college_aliases") or []
    if isinstance(aliases, (list, tuple)):
        candidates.extend(aliases)
    # The index's own graph names the institution better than its metadata does: the
    # highest-degree university/school nodes ARE the college and its parts. Deriving the
    # stop-list from the data keeps this module free of any college-specific literal.
    try:
        conn = getattr(index, "conn", None)
        if conn is not None:
            # ONLY the university itself. Its schools must NOT be stripped: measured on a
            # real run, removing "viterbi", "marshall", "dornsife" and "rossier" cost
            # Social Impact 2.42 points and Diversity 1.83, because a school name is the
            # sharpest discriminator the corpus has -- Viterbi means engineering, Marshall
            # means business. Only the institution's own name is uninformative inside its
            # own pages, because every page carries it.
            rows = conn.execute(
                "SELECT name FROM graph_nodes WHERE type = 'university'"
                " ORDER BY degree DESC LIMIT 3"
            ).fetchall()
            candidates.extend(r[0] for r in rows)
    except Exception:  # noqa: BLE001 - an index without a graph simply gets a smaller list
        pass

    # Generic words are kept: dropping "engineering" or "medicine" would gut real queries.
    GENERIC = {
        "the", "and", "for", "college", "school", "university", "institute", "of",
        "engineering", "medicine", "business", "arts", "sciences", "letters", "dance",
        "music", "education", "journalism", "communication", "policy", "public", "health",
        "cinematic", "law", "pharmacy", "dentistry", "social", "work", "academy",
    }
    for value in candidates:
        for token in re.findall(r"[A-Za-z]{3,}", str(value or "")):
            low = token.lower()
            if low not in GENERIC:
                words.add(low)
    return words


def strip_corpus_noise(text: str, stop: set[str]) -> str:
    """Drop college-identifying words from a query phrase, keeping everything else intact."""
    if not stop or not text:
        return text
    kept = [w for w in text.split() if re.sub(r"[^A-Za-z]", "", w).lower() not in stop]
    out = " ".join(kept).strip()
    # never return an empty query: if the phrase was nothing but the college's name, the
    # caller is better off with the original than with nothing to search for
    return out if len(out) >= 3 else text


def profile_facets(profile: dict[str, Any]) -> dict[str, list[str]]:
    """Flatten the profile into short, cleaned phrases per facet. Order is the profile's order."""
    cap = int(CONFIG["profile_field_max_chars"])
    list_cap = int(CONFIG["profile_list_cap"])

    # profile.py emits some facets as bare strings ("interests") and others as objects
    # ("activities"). Both readers below accept BOTH shapes on purpose: a reader that
    # understands only one shape returns an empty list when the producer changes, and an
    # empty facet is invisible -- retrieval just gets quieter and the report gets thinner,
    # with no error anywhere. That failure mode is the reason this pipeline was rebuilt.

    def _phrase(entry: Any, fields: Sequence[str]) -> str:
        if isinstance(entry, str):
            return clean_text(entry, cap)
        if isinstance(entry, dict):
            parts = [clean_text(entry.get(f), cap) for f in fields]
            phrase = clean_text(" ".join(p for p in parts if p), cap)
            if phrase:
                return phrase
            # unknown object shape: fall back to any label-ish field it does carry
            for f in ("label", "name", "detail", "title", "text"):
                fallback = clean_text(entry.get(f), cap)
                if fallback:
                    return fallback
        return ""

    def strings(key: str) -> list[str]:
        raw = profile.get(key) or []
        if not isinstance(raw, list):
            return []
        out = [_phrase(item, ("label", "name", "detail")) for item in raw]
        return [s for s in out if s][:list_cap]

    def items(key: str, fields: Sequence[str]) -> list[str]:
        raw = profile.get(key) or []
        if not isinstance(raw, list):
            return []
        out = [_phrase(entry, fields) for entry in raw]
        return [s for s in out if s][:list_cap]

    return {
        "intended_fields": strings("intended_fields"),
        "skills": strings("skills"),
        "interests": strings("interests"),
        "values": strings("values"),
        "activities": items("activities", ("name", "role", "detail")),
        "projects": items("projects", ("name", "detail")),
        "achievements": items("achievements", ("detail",)),
    }


# How much of the resume stands behind a phrase, by the section it came from. Something the
# student DID, or declared they want to study, is worth more than a word in a hobbies list.
FACET_EVIDENCE_WEIGHT = {
    "achievements": 1.0, "projects": 1.0, "activities": 1.0, "intended_fields": 1.0,
    "skills": 0.7, "values": 0.7, "interests": 0.4,
}


# The verbs and nouns every resume uses. Two entries sharing "collaborated" are not on a theme
# together, and counting them as one put a car-dealership internship above a founded company.
RESUME_FILLER = frozenset("""
collaborated conducted designed developed created managed led organised organized supported
assisted worked helped built made ran participated received awarded honored honoured achieving
achieved completed presented gained using used through across various several members member
experience skills understanding knowledge opportunity opportunities role roles intern internship
student students program programme project projects team teams group groups annual current
first second third overall total more most other others including include includes well also
""".split())


def _theme_words(text: str) -> set[str]:
    """Distinctive whole words. Whole, not stemmed: the 5-character stems used for ranking merge
    "service" with "servicing", which is exactly the confusion this scoring exists to avoid."""
    return {w for w in WORD_RE.findall(text.lower())
            if len(w) > 3 and w not in STOPWORDS and w not in RESUME_FILLER}


def theme_strength(facets: dict[str, list[str]]) -> dict[str, float]:
    """Per phrase: how much of this resume is actually about it.

    Not everything a student writes down carries equal weight, and matching treated it as if it
    did. One applicant's report offered him a chess club: chess appeared exactly once, as one
    word in a comma-separated hobbies line at the end of the page. In the same document
    artificial intelligence appeared in his declared field, his published paper, two internships
    and four skills, and rowing appeared twice -- once as a hobby and once as a state gold medal.
    A theme is what recurs, so a phrase scores for every other entry in the document that shares
    a distinctive word with it, weighted by the section each entry came from.

    Deterministic, no model, no network.
    """
    entries: list[tuple[str, str, set[str]]] = []
    for facet, values in facets.items():
        for value in values:
            entries.append((facet, value, _theme_words(value)))
    strength: dict[str, float] = {}
    for facet, value, words in entries:
        own = FACET_EVIDENCE_WEIGHT.get(facet, 0.7)
        total = own
        if words:
            for other_facet, other_value, other_words in entries:
                if other_value == value and other_facet == facet:
                    continue
                if words & other_words:
                    total += FACET_EVIDENCE_WEIGHT.get(other_facet, 0.7)
        strength[value] = max(strength.get(value, 0.0), total)
    return strength


def theme_strength_semantic(facets: dict[str, list[str]], vectors: dict[str, Any],
                            floor: float = 0.45) -> dict[str, float]:
    """Same idea as theme_strength, but by meaning, so abbreviations count.

    Shared words cannot see that "AI-driven analytics", "GPT-based conversational models" and a
    declared field of "Artificial Intelligence" are one theme -- "AI" is two characters and never
    matches the spelt-out phrase -- so the applicant's central subject scored as low as his
    hobbies. Similarity catches it. Falls back to the word version when a vector is missing.
    """
    entries = [(facet, value) for facet, values in facets.items() for value in values]
    if any(vectors.get(v) is None for _, v in entries):
        return theme_strength(facets)
    # The two readings fail in opposite directions and neither is sufficient alone. Shared words
    # cannot see that "AI-driven analytics" and "Artificial Intelligence" are one theme; meaning
    # cannot see that a lone technical term like "Portfolio Optimization" is central when it
    # appears verbatim in an internship description and nowhere else. Whichever finds more
    # support for a phrase is the one to believe.
    lexical = theme_strength(facets)
    strength: dict[str, float] = {}
    for facet, value in entries:
        total = FACET_EVIDENCE_WEIGHT.get(facet, 0.7)
        for other_facet, other_value in entries:
            if other_value == value:
                continue
            sim = float(np.dot(vectors[value], vectors[other_value]))
            if sim >= floor:
                total += FACET_EVIDENCE_WEIGHT.get(other_facet, 0.7) * sim
        strength[value] = max(strength.get(value, 0.0), total, lexical.get(value, 0.0))
    return strength


# --------------------------------------------------------------------------------------
# query planning: deterministic, template-based, several facets per category
# --------------------------------------------------------------------------------------


class Query:
    """One query, with a different string for each half of the search.

    `text` is what the dense half embeds: the student's item PLUS the category's lens phrase,
    because the lens is what steers the embedding toward the right kind of material.

    `keyword_text` is what BM25 searches: the student's item ALONE. The lens is boilerplate --
    put "undergraduate courses and teaching" into BM25 and it faithfully returns every page that
    says "teaches undergraduate courses", which is true of hundreds of professors and tells the
    student nothing. The keyword half exists to find the student's own specifics (CSCI 350,
    a professor's name, "Thematic Option"), so it gets only those.
    """

    __slots__ = ("qid", "category_code", "facet", "text", "keyword_text", "weight")

    def __init__(
        self,
        qid: str,
        category_code: str,
        facet: str,
        text: str,
        keyword_text: str,
        weight: float,
    ) -> None:
        self.qid = qid
        self.category_code = category_code
        self.facet = facet
        self.text = text
        self.keyword_text = keyword_text
        self.weight = weight

    def as_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "category_code": self.category_code,
            "facet": self.facet,
            "text": self.text,
            "keyword_text": self.keyword_text,
            "weight": round(self.weight, 4),
        }


def _terms(text: str) -> set[str]:
    """Content words truncated to a 5-character stem, so volunteer/volunteering/volunteers match.

    Crude on purpose: this only ranks which of the student's own items to build a query from,
    and a real stemmer would be a dependency for no gain at this stake.
    """
    return {
        w[:5] for w in WORD_RE.findall(text.lower()) if len(w) > 2 and w not in STOPWORDS
    }


def rank_for_category(values: Sequence[str], category: dict[str, Any], take: int,
                      ranker: "FacetRanker | None" = None) -> list[str]:
    """Pick the `take` profile items most relevant to this category, deterministically.

    Meaning where it is available, word overlap where it is not. Overlap alone compares
    five-character stems, and on a real applicant that read a car dealership's "Heads of Sales,
    Service, HR" as community SERVIce and a chatbot's "user engagement" as civic ENGAGement,
    scoring both above a founder's award-winning e-waste recycling venture, which scored zero
    because it never uses the words "community", "service" or "volunteer". Those two took the
    Social Impact slots and the venture never became a query at all.

    Embeddings are deterministic, so the same profile still produces the same queries. The
    student's own words are only ever compared, never executed.
    """
    if ranker is not None:
        ordered = ranker.rank(values, category)
        if ordered is not None:
            return ordered[:take]
    target = _terms(" ".join(category["intents"]) + " " + category["lens"])
    scored = [
        (-len(_terms(value) & target), position, value)
        for position, value in enumerate(values)
    ]
    scored.sort()
    return [value for _, _, value in scored[:take]]


class FacetRanker:
    """Ranks a student's own phrases against a category by meaning, in one batch of embeddings.

    Every facet value and every category lens is embedded once per run -- a few dozen short
    strings -- and every ranking after that is a dot product. Falls back to word overlap by
    returning None if anything goes wrong, because a worse ranking is recoverable and a failed
    run is not.
    """

    def __init__(self, embedder: Any, facets: dict[str, list[str]],
                 categories: Sequence[dict[str, Any]]) -> None:
        self._vec: dict[str, np.ndarray] = {}
        self._facets = facets
        self._theme: dict[str, float] = {}
        texts: list[str] = []
        for values in facets.values():
            texts.extend(v for v in values if v)
        for cat in categories:
            texts.append(self._target(cat))
        texts = sorted({t for t in texts if t})
        if not texts:
            return
        try:
            vectors = embedder.embed_queries(texts)
        except Exception as exc:                                  # noqa: BLE001
            log(f"  note: facet ranking fell back to word overlap ({type(exc).__name__}: {exc})")
            return
        if vectors is None or len(vectors) != len(texts):
            return
        self._vec = {t: vectors[i] for i, t in enumerate(texts)}
        self._theme = theme_strength_semantic(facets, self._vec)

    @staticmethod
    def _target(category: dict[str, Any]) -> str:
        return " ".join(category["intents"]) + " " + category["lens"]

    def rank(self, values: Sequence[str], category: dict[str, Any]) -> list[str] | None:
        target = self._vec.get(self._target(category))
        if target is None:
            return None
        scored = []
        for position, value in enumerate(values):
            vec = self._vec.get(value)
            if vec is None:
                return None
            # Fit to the category, scaled by how much of the resume stands behind the phrase.
            # Without this a one-word hobby competes on equal terms with a founded company: a
            # report offered an applicant the chess club off a single word in a hobbies line,
            # while the venture he had run for a year, and been given an award for, went unused.
            # Damped, not linear -- a strong theme should lead its category, not crowd out
            # everything a weaker one would have found.
            weight = self._theme.get(value, 1.0) ** float(CONFIG["theme_weight_power"])
            scored.append((-float(np.dot(vec, target)) * weight, position, value))
        scored.sort()
        return [value for _, _, value in scored]


def plan_queries(
    profile: dict[str, Any], categories: Sequence[dict[str, Any]] = CATEGORIES,
    index: "CollegeIndex | None" = None, embedder: Any | None = None,
) -> dict[str, list[Query]]:
    """3-6 queries per category, each aimed at a different facet of the student.

    No generative model is called. The embedder, when given, is used only to rank the student's
    own phrases against each category; everything else is template and profile order, so the
    same profile always produces the same queries.
    """
    facets = profile_facets(profile)
    # words naming the college itself cannot discriminate within that college's own corpus
    stop = _corpus_stopwords(index)
    if stop:
        facets = {k: [strip_corpus_noise(v, stop) for v in vs] for k, vs in facets.items()}
    ranker = FacetRanker(embedder, facets, categories) if embedder is not None else None
    max_chars = int(CONFIG["query_max_chars"])
    q_min, q_max = int(CONFIG["queries_min"]), int(CONFIG["queries_max"])
    plans: dict[str, list[Query]] = {}

    for cat in categories:
        code = cat["code"]
        lens = cat["lens"]
        intents = list(cat["intents"])
        # PER-CATEGORY, from the blind measurement documented on the CATEGORIES table. One global
        # setting was wrong for ten different questions; do not fold these back into one.
        weights = cat.get("query_weights") or {}
        intent_weight = float(weights.get("intent", 1.0))
        student_weight = float(weights.get("student", 0.9))
        student_led = str(cat.get("query_lead") or "category") == "student"
        # (facet label, dense text, keyword text, is_intent)
        built: list[tuple[str, str, str, bool]] = []

        # facet 1: the category's own intent, unpersonalised. Always first; its weight relative
        # to the student's queries is this category's own setting, not a global one.
        built.append(("intent", intents[0], intents[0], True))

        # facets 2..n: one query per profile facet the category cares about, most specific first.
        for field in cat["facets"]:
            values = facets.get(field) or []
            if not values:
                continue
            # How a student phrase is worded into a query matters more than it looks.
            #
            # Writing "<her resume text> <lens>" makes her words the subject, so the search
            # finds university material ABOUT her topics instead of opportunities FOR someone
            # with her background: a student who tutors maths matched the university's own
            # pedagogy research and education minors rather than the courses she would enrol
            # in -- it matched her as a teacher, not as a student. A stronger
            # embedding model makes that worse, not better, because it follows the subject
            # more faithfully.
            #
            # That is why a CATEGORY-LED category writes "<lens> ... <her words>". But it is not
            # true everywhere: the blind measurement on the CATEGORIES table found that for the
            # concrete categories -- which labs, which courses, which clubs, which press -- the
            # subject SHOULD be her, because leading with the lens returns the college's generic
            # page on the topic. Those categories are marked query_lead="student" and put her
            # phrase in front. Either way the keyword half keeps the category's words too -- it
            # used to search her resume text alone, with no idea which section it was filling.
            # RANK, NEVER SLICE. Every one of these lists arrives alphabetically sorted from
            # profile.py, so taking the first N hands the student's queries to whatever begins
            # with an early letter. Measured on a real applicant: a founder of an e-waste
            # recycling company who had published a finance paper declared four fields --
            # Artificial Intelligence, Computer Science, Environmental Sustainability,
            # Finance -- and `values[:2]` searched the first two and silently dropped the two
            # that made him distinctive. His interests read "Badminton, Chess, Cooking,
            # Football" in every query in every category, while the State Rowing gold sat at
            # position six and reached nothing. His skills queried "Java" and dropped natural
            # language processing, portfolio optimisation and predictive modelling.
            #
            # rank_for_category() already exists to choose which items speak to THIS category.
            # It was being used for activities and projects and not for anything else.
            if field == "intended_fields":
                # EVERY declared field gets its own query. A student declares two to four of
                # these and each one is a deliberate statement about what they want to study;
                # dropping half of them is dropping half the brief. Ranking cannot save this
                # either -- none of "Finance", "Computer Science" or "Environmental
                # Sustainability" lexically overlaps "research labs", so they all tie at zero
                # and the alphabet decides.
                for value in values[:6]:
                    built.append((
                        f"{field}:{value[:40]}",
                        f"{value} for undergraduates: {lens}"
                        if student_led
                        else f"{lens} in {value} for undergraduates",
                        f"{value} {lens}",
                        False,
                    ))
            elif field in ("activities", "projects", "achievements"):
                # the two items that actually speak to THIS category, not just the first two
                picked = rank_for_category(values, cat, 2, ranker)
                for rank, value in enumerate(picked, start=1):
                    label = f"{field}{'' if rank == 1 else rank}:{value[:40]}"
                    built.append((
                        label,
                        f"{value}: {lens} for an undergraduate with that background"
                        if student_led
                        else f"{lens} for an undergraduate whose background includes {value}",
                        f"{value} {lens}",
                        False,
                    ))
            else:
                # ranked, not alphabetical: for a Research section that is the technical
                # skills, for Quirks it is the unusual hobby, and neither is decided by
                # where the word falls in the alphabet
                # ranked first so the most category-relevant lead, but keep more of the tail:
                # at four, a student's eight interests became "Badminton, Chess, Cooking,
                # Football" and his State Rowing gold was unreachable in every category.
                # The slice below used to run on the alphabetical list, which is what the
                # comment above was promising and the code was not doing.
                joined = ", ".join(rank_for_category(values, cat, 4, ranker))
                built.append((
                    field,
                    f"{joined}: {lens} for an undergraduate"
                    if student_led
                    else f"{lens} for an undergraduate interested in {joined}",
                    f"{joined} {lens}",
                    False,
                ))

        # pad with the category's other static intents so a thin profile still gets q_min queries
        idx = 1
        while len(built) < q_min and idx < len(intents):
            built.append((f"intent{idx + 1}", intents[idx], intents[idx], True))
            idx += 1

        # dedupe on normalised dense text, keep first occurrence, cap
        seen: set[str] = set()
        queries: list[Query] = []
        for facet, text, keyword_text, is_intent in built:
            text = clean_text(text, max_chars)
            keyword_text = clean_text(keyword_text, max_chars)
            key = text.lower()
            if not text or key in seen:
                continue
            seen.add(key)
            # this category's own balance -- and carried by a flag, not by sniffing the facet
            # label, because "intended_fields" is a student facet whose name starts like "intent"
            weight = intent_weight if is_intent else student_weight
            # A DECLARED field outweighs a scraped hobby list. Both were student facets carrying
            # the same weight, so an applicant's stated subject competed on equal terms with
            # "Lawn Tennis, Oil painting, Basketball, Golf" -- and lost, because six off-topic
            # queries agree with each other and one on-topic query agrees with nobody.
            if facet.startswith("intended_fields"):
                weight *= float(CONFIG["declared_field_boost"])
            queries.append(
                Query(f"{code}-q{len(queries) + 1}", code, facet, text, keyword_text or text, weight)
            )
            if len(queries) >= q_max:
                break

        if not queries:
            die(f"category {code}: no queries could be planned (empty intents?)")
        plans[code] = queries

    return plans


# --------------------------------------------------------------------------------------
# FILTER FIRST: the candidate row set for a category
# --------------------------------------------------------------------------------------


def candidate_rows(
    index: CollegeIndex,
    category: dict[str, Any],
    undergraduate: bool,
    warnings: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    """Row ids this category is allowed to see. Computed BEFORE any search runs.

    Restricts to: this index (one college per index directory), the kinds the category may use,
    the category's own fact codes plus its supporting codes, units long enough to prove anything,
    and -- for an undergraduate applicant -- undergraduate courses only with no graduate-only
    prose. This is the only place rows are excluded; both search halves are handed this set.

    `warnings` collects the structured form of anything this function says out loud, so the
    explain file carries it too; callers that only want the rows may leave it None.
    """
    n = index.n_units
    allowed_kinds = set(category["kinds"])
    allowed_codes = set(category["fact_codes"]) | set(category["support_codes"])

    mask = np.zeros(n, dtype=bool)
    for kind in allowed_kinds:
        mask |= index.kind_arr == kind

    # facts must carry one of the category's codes; other kinds are untagged by design
    is_fact = index.kind_arr == "fact"
    code_ok = np.zeros(n, dtype=bool)
    for code in allowed_codes:
        code_ok |= index.category_arr == code
    mask &= ~is_fact | code_ok

    if category.get("require_external_or_own_code"):
        own = np.zeros(n, dtype=bool)
        for code in category["fact_codes"]:
            own |= index.category_arr == code
        non_official = (index.source_kind_arr == "external") | (index.source_kind_arr == "affiliated")
        mask &= own | non_official

    mask &= index.text_len >= int(CONFIG["min_unit_chars"])

    if undergraduate:
        is_course = index.kind_arr == "course"
        courses_before = int(np.count_nonzero(mask & is_course))
        mask &= ~is_course | (index.is_undergrad_course == 1)
        mask &= index.grad_only == 0
        courses_after = int(np.count_nonzero(mask & is_course))
        if courses_before and not courses_after:
            # SILENCE IS THE BUG. Losing a few graduate courses is the gate working; losing the
            # whole course pool is a bundle whose flag we could not read, and it is invisible
            # downstream -- the category just comes back as prose and nobody knows a kind floor
            # went unmet. Warn on the way past, every time, for all 45 colleges.
            raw_floor = float((category.get("kind_floor") or {}).get("course") or 0)
            # int(0.4) is 0, which silenced this warning the moment the floor became a share of
            # the section. The section size is not known here, so name the share as written.
            floor = raw_floor
            detail = (
                f"category {category['code']}: the undergraduate gate removed ALL "
                f"{courses_before:,} candidate courses (none flagged undergraduate, or all read "
                f"as graduate-only prose)"
            )
            if floor:
                detail += f"; this category's floor of {floor} course(s) cannot be met"
            log(f"  WARNING {detail}")
            if warnings is not None:
                warnings.append(
                    {
                        "category_code": category["code"],
                        "kind": "course",
                        "candidates_before_gate": courses_before,
                        "kind_floor": floor,
                        "detail": detail,
                    }
                )

    return np.flatnonzero(mask).astype(np.int64)


def own_code_rows(index: CollegeIndex, category: dict[str, Any], rows: np.ndarray) -> np.ndarray:
    """The subset of `rows` carrying one of the category's OWN fact codes.

    Its support codes and every untagged chunk/org/course are deliberately NOT in here: this is
    the material the category is actually about, and the quota below exists to make sure some of
    it reaches the scorer.
    """
    if rows.size == 0:
        return rows
    codes = set(category["fact_codes"])
    cats = index.category_arr[rows]
    keep = np.zeros(rows.size, dtype=bool)
    for code in codes:
        keep |= cats == code
    return rows[keep]


# --------------------------------------------------------------------------------------
# SEARCH: both halves, over the same pre-filtered rows
# --------------------------------------------------------------------------------------


class Embedder:
    """Embeds queries with WHATEVER MODEL THE INDEX WAS BUILT WITH.

    The model name always comes from the index's own meta.json, never from a default here.
    A query embedded by a different model than the documents lands in a different vector
    space: every similarity is meaningless, and nothing errors -- the reports just quietly
    get worse. So the index decides, and this class follows.

    Two providers, chosen by the model's name: a local fastembed model, or OpenAI's paid
    embedding API for any 'text-embedding-*' model.
    """

    OPENAI_PREFIX = "text-embedding-"

    def __init__(self, model_name: str, threads: int | None = None, dims: int | None = None) -> None:
        self.model_name = model_name
        self.dims = int(dims) if dims else None
        self.is_openai = model_name.startswith(self.OPENAI_PREFIX)
        if self.is_openai:
            self._init_openai()
        else:
            from fastembed import TextEmbedding  # late import: only search needs the model

            self.model = TextEmbedding(model_name, threads=threads)

    def _init_openai(self) -> None:
        import httpx

        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not key:
            # try the project .env, which is where this key is kept (chmod 600)
            env_path = Path(__file__).resolve().parent.parent / ".env"
            if env_path.is_file():
                for line in env_path.read_text(encoding="utf-8").splitlines():
                    if line.startswith("OPENAI_API_KEY="):
                        key = line.split("=", 1)[1].strip()
                        break
        if not key:
            die("index was built with an OpenAI embedding model but OPENAI_API_KEY is not set")
        self._key = key
        self._client = httpx.Client(timeout=120.0)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        """Embed queries exactly the way index_build.py embedded the documents.

        For the local provider that means .embed(), not .query_embed(): fastembed's query
        form may prepend a retrieval instruction, and a query built differently from the
        index is not comparable to it. Matching the index is what matters.
        """
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        texts = list(texts)
        if self.is_openai:
            vecs = self._embed_openai(texts)
        else:
            vecs = np.asarray(list(self.model.embed(texts)), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return vecs / norms

    def _embed_openai(self, texts: list[str]) -> np.ndarray:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "input": [t if t.strip() else " " for t in texts],
        }
        if self.dims:
            payload["dimensions"] = int(self.dims)
        delay = 2.0
        last: Exception | None = None
        for attempt in range(5):
            try:
                resp = self._client.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={"Authorization": f"Bearer {self._key}"},
                    json=payload,
                )
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"HTTP {resp.status_code}")
                resp.raise_for_status()
                rows = sorted(resp.json()["data"], key=lambda d: d["index"])
                return np.asarray([r["embedding"] for r in rows], dtype=np.float32)
            except Exception as exc:  # noqa: BLE001 - retried; re-raised when exhausted
                last = exc
                if attempt == 4:
                    break
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
        die(f"query embedding failed after 5 attempts: {last}")


def vector_search(
    index: CollegeIndex,
    rows: np.ndarray,
    query_vecs: np.ndarray,
    top_k: int,
) -> list[list[tuple[int, float]]]:
    """Exact dot-product search over the pre-filtered submatrix only.

    The submatrix is sliced out of vectors.npy from `rows` first, so a row outside the filter is
    never scored at all. Exact, so there is no approximate index and no post-filtering step that
    could quietly empty a category.
    """
    n_queries = int(query_vecs.shape[0]) if query_vecs.ndim == 2 else 0
    if rows.size == 0 or n_queries == 0:
        return [[] for _ in range(n_queries)]
    sub = np.asarray(index.vectors[rows], dtype=np.float32)  # (n_cand, dims), filtered rows only
    sims = sub @ query_vecs.T  # (n_cand, n_queries)
    k = int(min(top_k, sub.shape[0]))
    results: list[list[tuple[int, float]]] = []
    for qi in range(sims.shape[1]):
        column = sims[:, qi]
        if k < column.shape[0]:
            part = np.argpartition(-column, k - 1)[:k]
        else:
            part = np.arange(column.shape[0])
        order = part[np.argsort(-column[part], kind="stable")]
        results.append([(int(rows[i]), float(column[i])) for i in order])
    return results


def fts_query(text: str) -> str:
    """Turn query text into an FTS5 MATCH expression.

    Exact identifiers survive: course codes become adjacency phrases ("csci 350" and "csci350"),
    so a student asking about operating systems can still be handed CSCI 350 by name. Every
    clause is a quoted string, which also neutralises the FTS5 operators -- the profile is data,
    and a resume containing NEAR(x y) or `" OR unit_id:*` cannot change the query's shape.
    """
    lowered = text.lower()
    clauses: list[str] = []
    seen: set[str] = set()

    for dept, number in COURSE_CODE_RE.findall(lowered):
        for clause in (f'"{dept} {number}"', f'"{dept}{number}"'):
            if clause not in seen:
                seen.add(clause)
                clauses.append(clause)

    tokens = 0
    for word in WORD_RE.findall(lowered):
        word = word.replace("'", "")
        if len(word) < 2 or word in STOPWORDS:
            continue
        clause = f'"{word}"'
        if clause in seen:
            continue
        seen.add(clause)
        clauses.append(clause)
        tokens += 1
        if tokens >= 24:
            break

    return " OR ".join(clauses)


def load_candidate_table(index: CollegeIndex, rows: np.ndarray, table: str = "cand_rows") -> str:
    """(Re)fill the temp table the keyword half joins against. Returns its qualified name."""
    conn = index.conn
    conn.execute(f"DROP TABLE IF EXISTS temp.{table}")
    conn.execute(f"CREATE TEMP TABLE {table} (vec_row INTEGER PRIMARY KEY)")
    conn.executemany(f"INSERT INTO temp.{table}(vec_row) VALUES (?)", ((int(r),) for r in rows.tolist()))
    return f"temp.{table}"


def keyword_search(
    index: CollegeIndex,
    rows: np.ndarray,
    query_text: str,
    top_k: int,
    cand_table: str = "temp.cand_rows",
) -> list[tuple[int, float]]:
    """BM25 keyword search restricted to the same candidate rows.

    The candidate rows are materialised in a temp table and JOINed, so SQLite drops everything
    outside the filter before LIMIT is applied -- a pre-filter, not a trim of an already-truncated
    list. The Python-side membership check afterwards asserts that this held.
    """
    match = fts_query(query_text)
    if not match or rows.size == 0:
        return []
    weights = CONFIG["bm25_column_weights"]
    limit = int(top_k * int(CONFIG["fts_overfetch"]))
    sql = f"""
        SELECT u.vec_row AS vec_row, bm25({index.fts_table}, ?, ?) AS score
          FROM {index.fts_table} f
          JOIN {index.units_table} u ON u.rowid = f.rowid
          JOIN {cand_table} c ON c.vec_row = u.vec_row
         WHERE {index.fts_table} MATCH ?
         ORDER BY score ASC, u.vec_row ASC
         LIMIT ?
    """
    try:
        raw = index.conn.execute(sql, (weights[0], weights[1], match, limit)).fetchall()
    except sqlite3.OperationalError as exc:
        # a malformed MATCH must never take the run down; report it and fall back to no keyword hits
        log(f"  keyword search skipped for one query ({exc}); match was: {match[:120]}")
        return []
    allowed = set(rows.tolist())
    out: list[tuple[int, float]] = []
    for row in raw:
        vec_row = int(row["vec_row"])
        if vec_row not in allowed:  # impossible via the join; assert it anyway
            die(f"keyword search returned row {vec_row} outside the candidate set")
        out.append((vec_row, float(row["score"])))
        if len(out) >= top_k:
            break
    return out


# --------------------------------------------------------------------------------------
# OWN-CODE QUOTA: a thin category's own material must reach the scorer
# --------------------------------------------------------------------------------------


def own_quota(top_k: int, n_own: int) -> int:
    """How many of a query's `top_k` candidates are reserved for the category's own fact codes."""
    if n_own <= 0 or top_k <= 0:
        return 0
    return min(int(n_own), int(math.floor(top_k * float(CONFIG["own_code_min_fraction"]))))


def merge_with_own_quota(
    main_hits: Sequence[tuple[int, float]],
    own_hits: Sequence[tuple[int, float]],
    own_set: set[int],
    top_k: int,
    min_own: int,
    descending: bool,
) -> list[tuple[int, float]]:
    """Merge the whole-pool ranked list with the own-code-only list, keeping `min_own` own rows.

    The two passes score the same way over the same rows -- the same cosine similarity, the same
    bm25() over the same FTS table -- so merging them by score is exactly the ranking the
    whole-pool pass would have produced at a much greater depth, without paying for that depth.
    The reserved own-code rows come first, then the rest by score, and the result is re-sorted by
    score so the ranks handed to RRF still mean what they say.

    `descending` says which way is better: True for cosine similarity, False for bm25 (negative,
    smaller is better).
    """
    best: dict[int, float] = {}
    first_seen: dict[int, int] = {}
    for position, (vec_row, score) in enumerate(list(main_hits) + list(own_hits)):
        if vec_row not in best or (score > best[vec_row] if descending else score < best[vec_row]):
            best[vec_row] = score
        first_seen.setdefault(vec_row, position)

    def order_key(pair: tuple[int, float]) -> tuple[float, int]:
        # ties keep the order the searches produced them in, so the merge stays deterministic
        return (-pair[1] if descending else pair[1], first_seen[pair[0]])

    ordered = sorted(best.items(), key=order_key)
    if len(ordered) <= top_k:
        return ordered

    kept: list[tuple[int, float]] = []
    taken: set[int] = set()
    for pair in ordered:
        if len(kept) >= min_own:
            break
        if pair[0] in own_set:
            kept.append(pair)
            taken.add(pair[0])
    for pair in ordered:
        if len(kept) >= top_k:
            break
        if pair[0] in taken:
            continue
        kept.append(pair)
        taken.add(pair[0])
    kept.sort(key=order_key)
    return kept


def apply_own_code_quota(
    index: CollegeIndex,
    category: dict[str, Any],
    rows: np.ndarray,
    own_rows: np.ndarray,
    query_vecs: np.ndarray,
    queries: Sequence[Query],
    vector_hits: list[list[tuple[int, float]]],
    keyword_hits: list[list[tuple[int, float]]],
) -> tuple[list[list[tuple[int, float]]], list[list[tuple[int, float]]], dict[str, Any]]:
    """Guarantee each query's candidate depth a minimum share of the category's OWN fact codes.

    WHY THIS EXISTS. A category's pool is its own facts plus its support codes plus every
    untagged chunk, org and course, and the own material can be a rounding error inside it -- in
    the first shipped index one category held 305 facts against another's 21,266, roughly 35 to 1
    against it once the support codes and untagged rows are counted. The top-80 of such a pool
    can contain none of the category's own material at all, and once a row is outside every
    query's list it is not merely ranked low, it does not exist: fusion never sees it, rescoring
    never sees it, and max_support_fraction -- which runs at SELECTION time, after all of this --
    can only drop support rows that did get in. It cannot put back what was never scored.

    So when the own-code rows fall short of the quota, the same queries run again over the
    own-code rows ALONE (the widened pool: a thin category's whole own pool now fits inside one
    top-k) and the two rankings are merged by score. Categories whose own material already fills
    the quota pay nothing -- the second pass does not run.
    """
    info: dict[str, Any] = {
        "own_rows": int(own_rows.size),
        "pool_rows": int(rows.size),
        "quota_per_query": 0,
        "second_pass": False,
        "queries_short": 0,
    }
    if own_rows.size == 0 or own_rows.size == rows.size or not queries:
        return vector_hits, keyword_hits, info

    own_set = set(own_rows.tolist())
    v_k = int(CONFIG["vector_top_k"])
    k_k = int(CONFIG["keyword_top_k"])
    v_quota = own_quota(v_k, own_rows.size)
    k_quota = own_quota(k_k, own_rows.size)
    info["quota_per_query"] = max(v_quota, k_quota)
    if not v_quota and not k_quota:
        return vector_hits, keyword_hits, info

    def short(hits: list[tuple[int, float]], quota: int) -> bool:
        return sum(1 for vec_row, _ in hits if vec_row in own_set) < quota

    short_v = {i for i, hits in enumerate(vector_hits) if short(hits, v_quota)}
    short_k = {i for i, hits in enumerate(keyword_hits) if short(hits, k_quota)}
    info["queries_short"] = len(short_v | short_k)
    if not short_v and not short_k:
        return vector_hits, keyword_hits, info

    info["second_pass"] = True
    own_k = int(CONFIG["own_code_top_k"])
    if short_v:
        own_vector = vector_search(index, own_rows, query_vecs, own_k)
        vector_hits = [
            merge_with_own_quota(hits, own_vector[i], own_set, v_k, v_quota, descending=True)
            if i in short_v
            else hits
            for i, hits in enumerate(vector_hits)
        ]
    if short_k:
        own_table = load_candidate_table(index, own_rows, table="own_code_rows")
        keyword_hits = [
            merge_with_own_quota(
                hits,
                keyword_search(index, own_rows, queries[i].keyword_text, own_k, own_table),
                own_set,
                k_k,
                k_quota,
                descending=False,
            )
            if i in short_k
            else hits
            for i, hits in enumerate(keyword_hits)
        ]
    return vector_hits, keyword_hits, info


# --------------------------------------------------------------------------------------
# FUSE + RESCORE
# --------------------------------------------------------------------------------------


class Hit:
    __slots__ = ("vec_row", "rrf", "best_vector", "best_keyword", "queries", "parts")

    def __init__(self, vec_row: int) -> None:
        self.vec_row = vec_row
        self.rrf = 0.0
        self.parts: list[float] = []
        self.best_vector = 0.0
        self.best_keyword = 0.0
        self.queries: dict[str, dict[str, Any]] = {}

    def note(self, qid: str, half: str, rank: int, raw: float) -> None:
        entry = self.queries.setdefault(qid, {"qid": qid, "vector_rank": None, "keyword_rank": None})
        entry[f"{half}_rank"] = rank
        entry[f"{half}_score"] = round(raw, 6)


def rows_of_kind(index: CollegeIndex, rows: np.ndarray, kind: str) -> np.ndarray:
    """The subset of `rows` a floor key describes: a unit kind, or "entity:<type,type>"."""
    if kind.startswith("entity:"):
        types = {t.strip() for t in kind[len("entity:"):].split(",") if t.strip()}
        return rows[np.isin(index.entity_type_arr[rows], list(types))]
    return rows[index.kind_arr[rows] == kind]


def row_is_kind(index: CollegeIndex, vec_row: int, kind: str) -> bool:
    if kind.startswith("entity:"):
        types = {t.strip() for t in kind[len("entity:"):].split(",") if t.strip()}
        return index.entity_type[vec_row] in types
    return index.kind[vec_row] == kind


def apply_kind_quota(
    index: CollegeIndex,
    category: dict[str, Any],
    rows: np.ndarray,
    query_vecs: np.ndarray,
    vector_hits: list[list[tuple[int, float]]],
    queries: Sequence["Query"] | None = None,
) -> tuple[list[list[tuple[int, float]]], dict[str, Any]]:
    """A kind the category has a floor for is guaranteed rows in each query's list BEFORE fusion.

    A floor can only seat rows that were fetched, and each query keeps its top 80. For an
    applicant who declared Economics, the query "Economics for undergraduates: undergraduate
    courses, majors and degree requirements" ranked the economics COURSES at #146-169 -- behind
    about 140 facts ABOUT the major ("learning objectives include ...") -- so not one economics
    course was ever in the pool, and the course floor had nothing to seat. Prose about a subject
    always outscores the subject's own catalogue entries on a query that uses the prose's words.

    So each floored kind gets its own exact search over just its rows, and its best few rows are
    merged into every query's list that lacks them. Courses compete with courses for the floor.
    Same mechanism as the own-code quota above it, for the same reason. Generic: any kind, any
    college, any student.
    """
    floors = category.get("kind_floor") or {}
    info: dict[str, Any] = {}
    if not floors or rows.size == 0 or not len(vector_hits):
        return vector_hits, info
    quota = int(CONFIG["kind_quota_per_query"])
    v_k = int(CONFIG["vector_top_k"])
    for kind in floors:
        kind_rows = rows_of_kind(index, rows, kind)
        if kind_rows.size == 0:
            info[kind] = {"rows": 0, "queries_topped_up": 0}
            continue
        kind_set = set(kind_rows.tolist())
        short = [i for i, hits in enumerate(vector_hits)
                 if sum(1 for r, _ in hits if r in kind_set) < quota]
        if not short:
            info[kind] = {"rows": int(kind_rows.size), "queries_topped_up": 0}
            continue
        kind_hits = vector_search(index, kind_rows, query_vecs, quota)
        vector_hits = [
            merge_with_own_quota(hits, kind_hits[i], kind_set, v_k, quota, descending=True)
            if i in short else hits
            for i, hits in enumerate(vector_hits)
        ]
        # The order the floor should seat this kind in: by the STUDENT'S questions, never the
        # category's generic intent. Fused order is led by the intent query, and for Research
        # that query loves any lab page saying "welcomes undergraduates" -- a floor walking fused
        # order seated a yeast lab, a materials lab and a software lab for an AI-and-finance
        # applicant, displacing the programmes he had. Type says what may sit here; the
        # student's own words say who.
        student_cols = [i for i, q in enumerate(queries or [])
                        if not (q.facet == "intent" or q.facet.startswith("intent:"))]
        cols = student_cols if student_cols else list(range(int(query_vecs.shape[0])))
        sims = np.asarray(index.vectors[kind_rows], dtype=np.float32) @ query_vecs[cols].T
        best = sims.max(axis=1)
        order = np.argsort(-best, kind="stable")
        info[kind] = {"rows": int(kind_rows.size), "queries_topped_up": len(short),
                      "ordered": [int(kind_rows[i]) for i in order[: 4 * quota * max(1, len(cols))]]}
    return vector_hits, info


def anchor_artefacts(profile: dict[str, Any], embedder: Any) -> list[tuple[str, np.ndarray]]:
    """The student's most substantive lines: a described piece of work, not a prize line.

    Ordered by length within facet weight (a project outranks an activity outranks an award),
    never by recurrence -- recurrence is what made three small quiz prizes outrank one published
    paper. The same work listed twice (paper as project AND as achievement) is merged by embedding.
    """
    facets = profile_facets(profile)
    min_words = int(CONFIG["anchor_min_words"])
    cands: list[tuple[float, str]] = []
    for facet in ("projects", "activities", "achievements"):
        w = float(FACET_EVIDENCE_WEIGHT.get(facet, 0.5))
        for v in facets.get(facet) or []:
            n = len(v.split())
            if n >= min_words:
                cands.append((w * min(n, 40), v))
    cands.sort(key=lambda t: -t[0])
    if not cands:
        return []
    texts = [v for _, v in cands]
    vecs = embedder.embed_queries(texts)
    dup = float(CONFIG["anchor_duplicate_sim"])
    keep: list[tuple[str, np.ndarray]] = []
    for text, vec in zip(texts, vecs):
        if any(float(vec @ kv) >= dup for _, kv in keep):
            continue
        keep.append((text, vec))
        if len(keep) >= int(CONFIG["anchor_artefacts"]):
            break
    return keep


def anchor_pass(
    index: CollegeIndex,
    profile: dict[str, Any],
    embedder: Any,
    results: dict[str, list[dict[str, Any]]],
    per_category: int,
) -> list[dict[str, Any]]:
    """What a counsellor does first: take one thing the student DID and find its closest match.

    Every chapter blends ~8 questions, and blending is exactly what buries the person-level
    match. A published paper on the economic cost of menopause, run through Research as one of
    eight facets, lost to a dementia-cost study; run on its own, framed as a faculty search in the
    student's declared fields and restricted to rows about people, its top hits were a labour
    economist who studies violence against women and the professor whose college description is
    "how our health interacts with the labor economy". An e-waste venture, run on its own, found
    the college's monthly e-waste drive at #1. Neither reached any chapter through the blend.

    So: each substantive artefact is run VERBATIM, twice -- once against rows that name a thing,
    once against rows about people, framed as a faculty search in the declared fields -- and the
    best joins across all artefacts compete for a capped number of seats at the front of the
    chapter their kind belongs to, each carrying the line of the file it answers to. Chapters do
    not grow: an anchor replaces the weakest blended row. Generic in every part: the frame words
    come from the student's own declared fields, and it costs only the embeddings.
    """
    out: list[dict[str, Any]] = []
    if embedder is None or index.n_units == 0 or not results:
        return out
    artefacts = anchor_artefacts(profile, embedder)
    if not artefacts:
        return out
    fields = [str(x) for x in (profile.get("declared_fields") or profile.get("intended_fields") or []) if x]
    frame = "faculty whose research is on" + (" " + ", ".join(fields[:4]) if fields else "") + ": "
    person_rows = getattr(index, "person_rows", np.array([], dtype=np.int64))
    nameable = getattr(index, "nameable_rows", np.array([], dtype=np.int64))
    if not nameable.size:
        nameable = np.arange(index.n_units, dtype=np.int64)
    k = int(CONFIG["anchor_per_query"])
    min_sim = float(CONFIG["anchor_min_sim"])
    skip = index.process_boilerplate | index.foreign_institution

    # Per artefact, per frame, the ranked candidates. Seats are then given ROUND-ROBIN in order
    # of substance: every artefact's best join first, then the seconds. Never by a global score
    # -- similarity is not comparable across artefacts (a short generic line scores 0.54
    # against "X is an alumni mentor"; a paper's true faculty match scores 0.45).
    ranked: list[tuple[str, str, list[tuple[int, float]]]] = []
    research = [t for t, _ in artefacts if RESEARCH_MARKER.search(t)]
    framed = dict(zip(research, embedder.embed_queries([frame + t for t in research]))) \
        if research and person_rows.size else {}
    # Faculty joins first, for every research artefact, then the named-thing joins: a person
    # whose research is the student's research is the most valuable seat a chapter has.
    for text, vec in artefacts:
        if text in framed:
            ranked.append((text, "faculty whose research is closest to this",
                           [(int(r), float(x)) for r, x in
                            vector_search(index, person_rows, framed[text].reshape(1, -1), k)[0]]))
    for text, vec in artefacts:
        ranked.append((text, "closest named thing at the college to this",
                       [(int(r), float(x)) for r, x in vector_search(index, nameable, vec.reshape(1, -1), k)[0]]))
    min_hit_words = int(CONFIG["anchor_min_hit_words"])

    cur = index.conn.cursor()
    cur.row_factory = sqlite3.Row
    seen: set[int] = set()
    seen_entities: set[str] = set()
    per_chapter: dict[str, int] = {}     # doubles as the next seat: anchors keep their seating order
    max_chapter = int(CONFIG["anchor_max_per_chapter"])
    max_total = int(CONFIG["anchor_max_total"])
    for rnd in range(k):
        for art, why, hits in ranked:
            if len(out) >= max_total:
                return out
            # one seat per (artefact, frame) per round: the first hit not yet considered
            for vec_row, sim in hits:
                if sim < min_sim or vec_row in seen or vec_row in skip:
                    continue
                kind = index.kind[vec_row]
                if why.startswith("faculty"):
                    code = "RES"
                elif kind == "org":
                    code = "EXT"
                elif kind == "course":
                    code = "ACA"
                else:
                    code = index.category_code[vec_row] or ""
                if code not in results or per_chapter.get(code, 0) >= max_chapter:
                    seen.add(vec_row)
                    continue
                row = cur.execute("SELECT * FROM units WHERE vec_row = ?", (int(vec_row),)).fetchone()
                seen.add(vec_row)
                if row is None or len(str(row["text"] or "").split()) < min_hit_words:
                    continue
                ekey = (row["entity_name"] or "").strip().lower()
                if ekey and ekey in seen_entities:   # one seat per named thing
                    continue
                if ekey:
                    seen_entities.add(ekey)
                per_chapter[code] = per_chapter.get(code, 0) + 1
                hit = Hit(int(vec_row)); hit.best_vector = sim; hit.rrf = sim; hit.parts = [sim]
                entry = {"hit": hit, "score": sim,
                         "selected_because": f"anchor ({why}): {art[:70]}", "anchor_for": art}
                unit = evidence_unit(index, int(vec_row), entry, row)
                chapter = results[code]
                present = next((j for j, u in enumerate(chapter) if u["unit_id"] == unit["unit_id"]), None)
                if present is not None:
                    chapter.pop(present)          # already there: promote it, now carrying its artefact
                elif len(chapter) >= per_category:
                    chapter.pop()                 # chapters do not grow: the weakest blended row yields
                chapter.insert(per_chapter[code] - 1, unit)
                out.append({"artefact": art[:90], "unit_id": unit["unit_id"], "chapter": code,
                            "sim": round(sim, 3), "why": why, "promoted": present is not None})
                break
    return out

def fuse(
    vector_hits: list[list[tuple[int, float]]],
    keyword_hits: list[list[tuple[int, float]]],
    queries: Sequence[Query],
) -> dict[int, Hit]:
    """Reciprocal Rank Fusion across every query and both halves."""
    k = float(CONFIG["rrf_k"])
    vw = float(CONFIG["vector_weight"])
    kw = float(CONFIG["keyword_weight"])
    hits: dict[int, Hit] = {}

    for qi, query in enumerate(queries):
        for rank, (vec_row, sim) in enumerate(vector_hits[qi], start=1):
            hit = hits.setdefault(vec_row, Hit(vec_row))
            hit.parts.append(query.weight * vw / (k + rank))
            hit.best_vector = max(hit.best_vector, sim)
            hit.note(query.qid, "vector", rank, sim)
        for rank, (vec_row, bm25) in enumerate(keyword_hits[qi], start=1):
            hit = hits.setdefault(vec_row, Hit(vec_row))
            hit.parts.append(query.weight * kw / (k + rank))
            # bm25() is negative and smaller is better; report it as a positive "how strong"
            hit.best_keyword = max(hit.best_keyword, -bm25)
            hit.note(query.qid, "keyword", rank, -bm25)

    # AGREEMENT IS A TIE-BREAKER, NOT THE SIGNAL.
    #
    # Plain Reciprocal Rank Fusion sums 1/(k+rank) over every query. With k=60 the gap between
    # being a query's #1 and its #50 is only 1.8x, while eight queries mildly agreeing multiplies
    # by eight. The arithmetic therefore says eight shrugs beat one emphatic yes: a unit ranked
    # #1 by one query and ignored by seven scores 0.0164, and a bland unit ranked #50 by all
    # eight scores 0.0727 -- the bland one wins by 4.4x.
    #
    # That is not a tuning problem, it is the shape of the sum, and lowering k does not fix it:
    # even at k=5 breadth still wins. It is why a Gender Studies applicant's own department never
    # reached her report while "the university has student organizations." did, and why the best
    # matching AI society in the corpus lost to generic club prose for an AI applicant. Only one
    # or two of a student's eight queries ever ask about what makes them unusual.
    #
    # So score each unit by its BEST single query, and let the rest contribute a fraction. The
    # question becomes "how well did the best question match this?" rather than "how many
    # questions had a mild opinion?". alpha=0 would ignore corroboration entirely; alpha=1 is the
    # old behaviour.
    alpha = float(CONFIG["fusion_agreement_weight"])
    for hit in hits.values():
        if not hit.parts:
            continue
        best = max(hit.parts)
        hit.rrf = best + alpha * (sum(hit.parts) - best)

    return hits


def recency_factor(index: CollegeIndex, vec_row: int) -> float:
    """Half-life decay on `year`, floored. Undated material gets a mild, fixed discount."""
    year = int(index.year[vec_row])
    if year <= 0:
        return float(CONFIG["recency_unknown_year_factor"])
    age = index.reference_year - year
    if age < -int(CONFIG["recency_future_tolerance_years"]):
        # A row dated after the index was built is not fresher than one dated correctly -- it is a
        # row that MENTIONS a future year ("apply by fall 2031", a naming-rights gift through
        # 2040). Handing it a perfect 1.0 put stale prose above current material in exactly the
        # categories that asked for recency, so treat it as undated instead.
        return float(CONFIG["recency_unknown_year_factor"])
    if age <= 0:
        return 1.0
    factor = 0.5 ** (age / float(CONFIG["recency_half_life_years"]))
    return max(float(CONFIG["recency_floor"]), factor)


def org_fit_score(index: CollegeIndex, vec_row: int, fit_key: str | None) -> int:
    """This organisation's 0-3 fit on the category's fit key; 0 when it has none."""
    if not fit_key or index.kind[vec_row] != "org":
        return 0
    return max(0, int((index.fit_scores.get(vec_row) or {}).get(fit_key) or 0))


def rescore(
    index: CollegeIndex, category: dict[str, Any], hits: dict[int, Hit]
) -> dict[int, dict[str, Any]]:
    """Apply kind, source, corroboration, organisation-fit and recency multipliers."""
    kind_weights: dict[str, float] = category["kind_weights"]
    source_weights: dict[str, float] = category["source_kind_weights"]
    fit_key = category.get("org_fit_key")
    use_recency = bool(category.get("recency"))
    step = float(CONFIG["multi_source_step"])
    cap = int(CONFIG["multi_source_cap"])
    fit_step = float(CONFIG["org_fit_step"])

    scored: dict[int, dict[str, Any]] = {}
    for vec_row, hit in hits.items():
        kind = index.kind[vec_row]
        boosts: dict[str, float] = {
            "kind": float(kind_weights.get(kind, 1.0)),
            "placeholder_course": (
                float(CONFIG["placeholder_course_weight"])
                if vec_row in index.placeholder_courses
                else 1.0
            ),
            "process_boilerplate": (
                float(CONFIG["process_boilerplate_weight"])
                if vec_row in index.process_boilerplate
                else 1.0
            ),
            "source_kind": float(source_weights.get(index.source_kind[vec_row], 1.0)),
            "multi_source": 1.0 + step * min(max(0, int(index.source_count[vec_row]) - 1), cap),
        }
        if fit_key and kind == "org":
            score = org_fit_score(index, vec_row, fit_key)
            # A fit-keyed category IS the fit: a club scoring 0 on it is not weak evidence, it is
            # the wrong club in the right-shaped slot. At 1.0 it tied with a club that actually
            # scored, and the kind floor then injected it anyway -- a quirky-clubs section full of
            # ordinary clubs. Zero fit is now a penalty, and select_units refuses to floor it in.
            boosts["org_fit"] = (
                float(CONFIG["org_fit_zero_penalty"])
                if score <= 0
                else 1.0 + fit_step * min(score, 3)
            )
        boosts["recency"] = recency_factor(index, vec_row) if use_recency else 1.0

        multiplier = 1.0
        for value in boosts.values():
            multiplier *= value
        scored[vec_row] = {
            "hit": hit,
            "boosts": {k: round(v, 4) for k, v in boosts.items()},
            "multiplier": multiplier,
            "score": hit.rrf * multiplier,
        }
    return scored


# --------------------------------------------------------------------------------------
# SELECT: diversity caps
# --------------------------------------------------------------------------------------


def dedupe_key(text: str) -> str:
    """Normalised prefix of a unit's text, used to spot the same passage served twice.

    Real bundles carry the same page under more than one URL (…/research/x and
    …/researchandinnovation/x), which gives two units with different ids, different page ids and
    identical prose. Without this they take two slots and the report says the same thing twice.
    """
    key = re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()
    # A cross-listed course is served under two codes with identical prose (PSYC 210 / POSC 210,
    # ENST 150 / IR 150). Keyed on the full text they took two seats and said the same thing
    # twice -- three of ten course seats in one real chapter. The code is not the course.
    key = re.sub(r"^[a-z]{2,5} ?\d{3,5}[a-z]? ?", "", key)
    return key[:200]


def reserved_by_rank(vhits: Sequence[Sequence[tuple]], depth: int,
                     skip: set[int] | frozenset[int] = frozenset()) -> list[int]:
    """Each query's top `depth` vector hits, interleaved: all the #1s, then all the #2s.

    Breadth before depth, so a query with a weak leader still gets its best row seated, and no
    single query can take two slots before another has taken one.
    """
    # A seat is reserved because the fused score is known to discard a query's best answer.
    # It must not be spent on filler: "The page lists an Undergraduate section." was a query's
    # #1 vector hit and took a reserved seat in a real report. Each query's list is walked past
    # anything in `skip` to its best hit that actually says something.
    usable = [[int(r) for r, _ in hits if int(r) not in skip] for hits in vhits]
    out: list[int] = []
    seen: set[int] = set()
    for rank in range(max(1, depth)):
        for hits in usable:
            if rank >= len(hits):
                continue
            row = hits[rank]
            if row not in seen:
                seen.add(row)
                out.append(row)
    return out


def select_units(
    index: CollegeIndex,
    category: dict[str, Any],
    scored: dict[int, dict[str, Any]],
    per_category: int,
    gap_candidates: Sequence[dict[str, Any]] | None = None,
    gap_scored: dict[int, dict[str, Any]] | None = None,
    query_best: Sequence[int] | None = None,
    kind_order: dict[str, Sequence[int]] | None = None,
) -> tuple[list[int], list[dict[str, Any]], dict[int, sqlite3.Row]]:
    """Greedy top-down pick, subject to per-page, per-entity, per-host, per-kind and
    supporting-code caps.

    A kind floor (e.g. three undergraduate courses for Academics) is filled first from the ranked
    list, so a category that should name clubs or courses does not come back as twelve prose
    facts. Returns (chosen vec_rows in score order, drop log).

    `query_best` is each query's single strongest vector hit, seated BEFORE the fused ranking.
    Reciprocal Rank Fusion rewards agreement: a unit found strongly by one query and not at all
    by the others is out-voted by generic material that every query touches weakly. That is
    fine when the queries all ask the same thing and fatal when they deliberately do not --
    ours are built one per facet precisely so that a student's minority interests get asked
    about. Measured case: a centre named "Responsible AI and Decision Making in Finance" scored
    0.58 on an applicant's own finance-and-AI project query, and never appeared in his report,
    because his other seven queries had no opinion about it. Letting each query seat its best
    find costs at most one slot per query and is the difference between asking a question and
    acting on the answer.

    `gap_candidates` (wwrag/gapfill.py) are second-pass hits that close a gap the first pass
    left. They take a small number of RESERVED slots ahead of everything else, in the order
    gapfill put them in -- (gap query, BM25 rank), never fused score, because a gap-closer is by
    definition found by exactly one query and fusion punishes precisely that. They are NOT added
    to `ranked`, so the first pass's own ordering is byte-identical to a run with this pass off;
    the change is exactly bounded to the reserved slots. Every one of them still goes through
    `blocked_by`, so no diversity, entity, host, kind, fit or support cap is bypassed.
    """
    gap_scored = gap_scored or {}
    ranked = sorted(scored.items(), key=lambda kv: (-kv[1]["score"], index.unit_id[kv[0]]))

    def entry_for(vec_row: int) -> dict[str, Any]:
        entry = scored.get(vec_row)
        return entry if entry is not None else gap_scored[vec_row]

    # this function may run twice for one category (once to establish what the first pass
    # selected, once with the gap candidates), so nothing may carry over from the first call
    for entry in list(scored.values()) + list(gap_scored.values()):
        entry.pop("selected_because", None)
    max_page = int(CONFIG["max_per_source_page"])
    max_entity = int(CONFIG["max_per_entity"])
    max_boilerplate = int(CONFIG["max_process_boilerplate"])
    max_host = int(CONFIG["max_per_host"])
    # A per-hostname cap on an index with exactly ONE hostname is not a diversity control, it
    # is just a throttle: it would cap every category of a college that publishes everything on
    # www.college.edu at four units out of twenty-four while preventing nothing. Two hostnames
    # is enough for the cap to mean something, so the stand-down is deliberately the narrowest
    # possible condition. Per-page and per-entity caps are unaffected and still protect a
    # single-domain college.
    max_kind: dict[str, int] = dict(CONFIG["max_per_kind"])
    support_codes = set(category["support_codes"])
    max_support = max(0, int(math.floor(per_category * float(CONFIG["max_support_fraction"]))))

    # A category keyed on organisation fit is ABOUT that fit, so an org scoring 0 on the key is
    # the wrong club and may not be selected for it -- not by the kind floor, which used to
    # force-inject the top-ranked org regardless, and not on rank either, which is how a
    # quirky-clubs section filled with ordinary engineering clubs. The gate needs fit data to
    # mean anything, so a bundle that scored no organisation at all turns it off out loud rather
    # than emptying the org pool of every fit-keyed category in silence (colleges 2-45).
    fit_key = category.get("org_fit_key")
    min_floor_fit = int(CONFIG["org_fit_floor_min_score"])
    fit_gate = bool(fit_key) and index.fit_scored_orgs > 0
    if fit_key and not fit_gate:
        log(
            f"  note category {category['code']}: no organisation in this index carries a fit "
            f"score, so the {fit_key!r} fit gate is OFF and clubs are ranked on text alone"
        )

    # text for the rows selection could plausibly reach, in one query, so the near-duplicate
    # guard can compare prose without pulling all 99k units into memory
    considered = [row for row, _ in ranked[: max(per_category * 8, 64)]]
    considered += [c["vec_row"] for c in (gap_candidates or []) if c["vec_row"] not in scored]
    texts = index.fetch_text(sorted(set(considered)))

    chosen: list[int] = []
    chosen_set: set[int] = set()
    page_count: dict[str, int] = {}
    entity_count: dict[str, int] = {}
    host_count: dict[str, int] = {}
    kind_count: dict[str, int] = {}
    seen_text: dict[str, str] = {}
    boilerplate_used = 0
    support_used = 0
    drops: list[dict[str, Any]] = []

    def blocked_by(vec_row: int, seating_floor: bool = False) -> str | None:
        row = texts.get(vec_row)
        if row is not None:
            key = dedupe_key(row["text"])
            if key and key in seen_text:
                return f"near-duplicate of {seen_text[key]}"
        page = index.page_key[vec_row]
        if page_count.get(page, 0) >= max_page:
            return f"source page cap ({max_page}) for {page}"
        entity = index.entity_key[vec_row]
        if entity_count.get(entity, 0) >= max_entity:
            return f"entity cap ({max_entity}) for {entity}"
        host = index.source_host[vec_row]
        # The host cap stops one SITE dominating a chapter. A kind floor is a deliberate
        # reservation for one kind of thing, and every course at a college is published on a
        # single host -- the catalogue -- so the cap silently limited every course floor to four,
        # whatever the floor said. Four separate floor implementations measured zero gain before
        # this was found. Seating a floor bypasses the host cap; nothing else does.
        if not seating_floor and host_count.get(host, 0) >= max_host:
            return f"host cap ({max_host}) for {host}"
        kind = index.kind[vec_row]
        if kind in max_kind and kind_count.get(kind, 0) >= max_kind[kind]:
            return f"kind cap ({max_kind[kind]}) for {kind}"
        if fit_gate and kind == "org" and org_fit_score(index, vec_row, fit_key) < min_floor_fit:
            return f"organisation fit {org_fit_score(index, vec_row, fit_key)} on {fit_key}"
        code = index.category_code[vec_row]
        if code in support_codes and support_used >= max_support:
            return f"supporting-context cap ({max_support}) for {code}"
        if vec_row in index.process_boilerplate and boilerplate_used >= max_boilerplate:
            return f"enrolment-process cap ({max_boilerplate})"
        return None

    def take(vec_row: int, reason: str) -> None:
        nonlocal support_used, boilerplate_used
        chosen.append(vec_row)
        chosen_set.add(vec_row)
        page = index.page_key[vec_row]
        entity = index.entity_key[vec_row]
        host = index.source_host[vec_row]
        kind = index.kind[vec_row]
        page_count[page] = page_count.get(page, 0) + 1
        entity_count[entity] = entity_count.get(entity, 0) + 1
        host_count[host] = host_count.get(host, 0) + 1
        kind_count[kind] = kind_count.get(kind, 0) + 1
        if vec_row in index.process_boilerplate:
            boilerplate_used += 1
        row = texts.get(vec_row)
        if row is not None:
            key = dedupe_key(row["text"])
            if key:
                seen_text.setdefault(key, index.unit_id[vec_row])
        if index.category_code[vec_row] in support_codes:
            support_used += 1
        entry_for(vec_row)["selected_because"] = reason

    # pass 0: RESERVED GAP SLOTS. Bounded, ordered by the gap query's own BM25 rank, and still
    # subject to every cap below -- max_per_entity=2 means one gap query can contribute at most
    # two units of one entity, so a gap-closer can never become the section.
    gap_taken = 0
    if gap_candidates:
        reserved = gapfill.gap_slots(per_category)
        for candidate in gap_candidates:
            if gap_taken >= reserved or len(chosen) >= per_category:
                break
            vec_row = int(candidate["vec_row"])
            if vec_row in chosen_set:
                continue
            reason = blocked_by(vec_row)
            if reason:
                if len(drops) < 40:
                    drops.append(
                        {
                            "unit_id": index.unit_id[vec_row],
                            "score": round(entry_for(vec_row)["score"], 6),
                            "reason": f"gap slot blocked: {reason}",
                        }
                    )
                continue
            take(vec_row, f"gap slot: {category['code']}-gap{candidate['gap_query'] + 1}")
            gap_taken += 1

    # pass 0b: ONE RESERVED SLOT PER QUERY. Each query's strongest vector hit is seated before
    # the fused ranking, because fusion systematically discards exactly the hits these queries
    # exist to find. Our queries are deliberately different from each other -- one per facet,
    # so a student's minority interests get asked about at all -- and RRF then rewards whatever
    # they AGREE on, which is the generic material. A centre called "Responsible AI and Decision
    # Making in Finance" scored 0.58 on an applicant's own finance-and-AI project query and
    # reached no report, because his seven other queries had no view on it.
    #
    # Bounded and safe: at most one row per query, never more than a third of the section, each
    # still passes every cap in blocked_by, and the fused ordering below is untouched.
    query_taken = 0
    reserved_rows: list[int] = []
    if query_best:
        max_query_slots = max(1, (per_category * int(CONFIG["query_slot_depth"])) // 3)
        for vec_row in query_best:
            if query_taken >= max_query_slots or len(chosen) >= per_category:
                break
            vec_row = int(vec_row)
            if vec_row in chosen_set or vec_row not in scored:
                continue
            if blocked_by(vec_row):
                continue
            take(vec_row, "query slot: this query's strongest hit")
            reserved_rows.append(vec_row)
            query_taken += 1

    # pass 1: kind floors
    for kind, floor in sorted((category.get("kind_floor") or {}).items()):
        # A floor below 1 is a SHARE of the section rather than a count. An absolute count tuned
        # against a 24-unit section silently becomes "the whole section" at 10 and "a rounding
        # error" at 60, and it cannot be right at both.
        floor = max(1, round(float(floor) * per_category)) if 0 < float(floor) < 1 else int(floor)
        taken = 0
        preferred = [int(r) for r in (kind_order or {}).get(kind, []) if int(r) in scored]
        seen_pref = set(preferred)
        candidates = preferred + [r for r, _ in ranked if r not in seen_pref]
        for vec_row in candidates:
            if taken >= floor or len(chosen) >= per_category:
                break
            if vec_row in chosen_set or not row_is_kind(index, vec_row, kind):
                continue
            # blocked_by carries the fit gate: the floor is here to stop a fit-keyed category
            # coming back as twelve prose facts, NOT to promise clubs at any price. An unmet
            # floor is honest; a wrong club is not.
            if blocked_by(vec_row, seating_floor=True):
                continue
            take(vec_row, f"kind floor: {kind}")
            taken += 1
        if taken < floor and kind == "org" and fit_gate:
            log(
                f"  note category {category['code']}: only {taken} of {floor} floor "
                f"organisation(s) scored at least {min_floor_fit} on fit key {fit_key!r}; "
                f"the floor is left unmet rather than filled with zero-fit ones"
            )

    # pass 2: the rest, in rank order
    for vec_row, entry in ranked:
        if len(chosen) >= per_category:
            break
        if vec_row in chosen_set:
            continue
        reason = blocked_by(vec_row)
        if reason:
            if len(drops) < 40:
                drops.append(
                    {
                        "unit_id": index.unit_id[vec_row],
                        "score": round(entry["score"], 6),
                        "reason": reason,
                    }
                )
            continue
        take(vec_row, "rank")

    # keep the final list in score order even after the kind-floor and gap passes jumped the queue
    # RESCUED, THEN BURIED. Pass 0b seats each query's strongest hit precisely because the fused
    # score is known to discard it -- and then this sort put it back where the fused score said it
    # belonged. Measured on a real applicant: her declared field's best match, rank 1 of 89,181
    # rows at cosine 0.657, was correctly reserved and then handed to the writer as evidence item
    # 17 of 24. The rescue worked and the sort undid it. Reserved rows keep the front of the
    # section, in the order they were seated; everything else sorts by score as before.
    reserved = {r: i for i, r in enumerate(reserved_rows)}
    chosen.sort(key=lambda r: (0, reserved[r]) if r in reserved
                else (1, -entry_for(r)["score"]))
    return chosen, drops, texts


# --------------------------------------------------------------------------------------
# EvidenceUnit assembly
# --------------------------------------------------------------------------------------


def evidence_unit(
    index: CollegeIndex, vec_row: int, entry: dict[str, Any], text_row: sqlite3.Row
) -> dict[str, Any]:
    """Exactly the shared EvidenceUnit contract -- no extra fields."""
    hit: Hit = entry["hit"]
    year = int(index.year[vec_row])
    return {
        "unit_id": index.unit_id[vec_row],
        "kind": index.kind[vec_row],
        "category_code": index.category_code[vec_row],
        "text": text_row["text"],
        "quote": text_row["quote"],
        "entity_name": index.entity_name[vec_row],
        "source_url": index.source_url[vec_row],
        "source_title": text_row["source_title"],
        "source_kind": index.source_kind[vec_row],
        "year": year if year > 0 else None,
        "score": round(float(entry["score"]), 6),
        # Why this row is here. A row seated by the reserved-query-slot pass is the single best
        # answer to one of the questions the student's own profile raised, and it keeps the front
        # of the section -- the fused score is exactly what would have discarded it.
        "selected_because": str(entry.get("selected_because") or ""),
        # For an anchor row: the exact line of the student's file this is the closest thing to.
        # The writer gets the pair, so "take your survey data to Barcellos" can be written.
        "anchor_for": str(entry.get("anchor_for") or ""),
        "retrieval": {
            "vector": round(float(hit.best_vector), 6),
            "keyword": round(float(hit.best_keyword), 6),
            "rrf": round(float(hit.rrf), 6),
        },
    }


# --------------------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------------------


GAP_EXPLAIN_OFF: dict[str, Any] = {"enabled": False, "ran": False, "reason": "disabled (--no-gapfill)"}


def gap_pass(
    index: CollegeIndex,
    profile: dict[str, Any],
    category: dict[str, Any],
    rows: np.ndarray,
    queries: Sequence[Query],
    chosen: Sequence[int],
    texts: dict[int, sqlite3.Row],
    scored: dict[int, dict[str, Any]],
    stats: "gapfill.IndexStats",
    cand_table: str,
    corpus_stop: set[str],
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], dict[str, Any]]:
    """The second retrieval pass for ONE category.

    Everything here searches the SAME pre-filtered `rows` and the SAME temp table the first pass
    built. The filter is never rebuilt and never widened, so nothing can reach the output that
    was not already in this category's candidate set -- `keyword_search` asserts that on the way
    back out, exactly as it does for the first pass.

    Returns (ordered reserved candidates, their scored entries, explain block).
    """
    selected: list[tuple[str, str, str]] = []
    for vec_row in chosen:
        row = texts.get(vec_row)
        selected.append(
            (
                index.entity_name[vec_row] or "",
                (row["text"] or "") if row is not None else "",
                (row["quote"] or "") if row is not None else "",
            )
        )

    first_pass_texts = [q.keyword_text for q in queries] + [q.text for q in queries]
    gap_queries, explain = gapfill.plan_gap_queries(
        profile, category, stats, selected, first_pass_texts, corpus_stop
    )
    explain["enabled"] = True
    if not gap_queries:
        return [], {}, explain

    top_k = int(gapfill.CONFIG["gap_top_k"])
    raw_hits = [
        keyword_search(index, rows, gq.keyword_text, top_k, cand_table) for gq in gap_queries
    ]
    # text for the few rows a reserved slot could plausibly reach, so the nameable and
    # actionability tests can read prose without pulling the index into memory
    wanted = sorted({row for hits in raw_hits for row, _ in hits})
    gap_texts = index.fetch_text(wanted) if wanted else {}

    def text_of(vec_row: int) -> str:
        row = gap_texts.get(vec_row) or texts.get(vec_row)
        if row is None:
            return ""
        return " ".join(str(row[k] or "") for k in ("text", "quote"))

    candidates = gapfill.reserved_candidates(
        raw_hits,
        set(chosen),
        lambda r: index.entity_name[r],
        text_of,
        corpus_stop,
        stats,
        gapfill.answer_types(category["code"]),
        lambda r: index.kind[r],
    )

    # Score the gap hits the same way everything else is scored, so a gap unit carries a real
    # score, real boosts and a real retrieval block. Fused over the GAP queries alone: adding
    # them to the first pass's fusion perturbs its ordering far beyond the reserved slots, which
    # would make the before/after unattributable.
    wrapped = [
        Query(gq.qid, category["code"], f"gap:{gq.facet}", gq.keyword_text, gq.keyword_text, 1.0)
        for gq in gap_queries
    ]
    gap_hits = fuse([[] for _ in wrapped], raw_hits, wrapped)
    gap_scored = rescore(index, category, gap_hits)
    # A row the first pass also found keeps its first-pass SCORE -- the gap pass must not
    # perturb pass-1 ordering -- but its explain entry and its reported keyword strength are
    # updated, so the artifact says the second pass found it too. Neither field feeds the
    # score, which is rrf x multiplier.
    for vec_row, hit in gap_hits.items():
        if vec_row in scored:
            scored[vec_row]["hit"].queries.update(hit.queries)
            scored[vec_row]["hit"].best_keyword = max(
                scored[vec_row]["hit"].best_keyword, hit.best_keyword
            )
    gap_scored = {r: e for r, e in gap_scored.items() if r not in scored}

    explain["searched"] = [
        {"qid": gq.qid, "hits": len(raw_hits[i])} for i, gq in enumerate(gap_queries)
    ]
    explain["candidates"] = [
        {
            "unit_id": index.unit_id[c["vec_row"]],
            "entity_name": index.entity_name[c["vec_row"]],
            "gap_query": gap_queries[c["gap_query"]].qid,
            "bm25_rank": c["rank"],
            "bm25": round(c["bm25"], 4),
            "reads_as_bibliography": bool(c["band"]),
        }
        for c in candidates[:12]
    ]
    return candidates, gap_scored, explain


def retrieve(
    index: CollegeIndex,
    profile: dict[str, Any],
    per_category: int,
    embedder: Any | None = None,
    categories: Sequence[dict[str, Any]] = CATEGORIES,
    gapfill_enabled: bool = True,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Run the whole pipeline. Returns ({category_code: [EvidenceUnit]}, explain).

    `gapfill_enabled` (default on, `--no-gapfill` off) runs the second, gap-closing retrieval
    pass described in wwrag/gapfill.py. Turning it off reproduces the previous output exactly,
    which is what keeps the before/after measurable.
    """
    if per_category < 1:
        die(f"--per-category must be at least 1, got {per_category}")

    level = clean_text(profile.get("level"), 40).lower()
    if level not in ("undergraduate", "graduate"):
        die(f"StudentProfile.level must be 'undergraduate' or 'graduate', got {level!r}")
    undergraduate = level in CONFIG["undergraduate_levels"]
    if not undergraduate:
        log(f"  note: profile level is {level!r}; the undergraduate-only pre-filter is OFF")

    plans = plan_queries(profile, categories, index=index, embedder=embedder)
    all_queries = [q for cat in categories for q in plans[cat["code"]]]

    # The gap pass's corpus statistics are a property of the INDEX, not of the student, so they
    # are built once here and shared by all ten categories -- and, across a batch, warm for
    # every student after the first.
    corpus_stop = _corpus_stopwords(index)
    gap_stats: gapfill.IndexStats | None = None
    if gapfill_enabled:
        gap_stats = gapfill.IndexStats(index.conn, index.fts_table, index.n_units)
        if not gap_stats.has_graph:
            # LOUD, never a silent fallback to ungated rarity: that picks nonsense anchors.
            log(gapfill.missing_graph_note("ALL"))
            gap_stats = None

    started = time.time()
    if embedder is None:
        embedder = Embedder(index.model_name, dims=index.dims)
    t_embed = time.time()
    query_vecs = embedder.embed_queries([q.text for q in all_queries])
    embed_seconds = time.time() - t_embed
    if query_vecs.shape[0] != len(all_queries):
        die(f"embedder returned {query_vecs.shape[0]} vectors for {len(all_queries)} queries")
    if query_vecs.shape[1] != index.dims:
        die(f"query vectors have {query_vecs.shape[1]} dims, the index has {index.dims}")

    offset = 0
    results: dict[str, list[dict[str, Any]]] = {}
    empty_categories: list[dict[str, str]] = []
    # anything the pre-filter says out loud is also recorded, so an operator reading the explain
    # file for college 23 sees the emptied course pool without scrolling back through stdout
    filter_warnings: list[dict[str, Any]] = []
    explain: dict[str, Any] = {
        "schema_version": RETRIEVE_SCHEMA_VERSION,
        "college_id": index.college_id,
        "student_id": clean_text(profile.get("student_id"), 120),
        "level": level,
        "undergraduate_filter": undergraduate,
        "per_category": per_category,
        "embedding_model": index.model_name,
        "reference_year": index.reference_year,
        "index_dir": str(index.dir),
        "index_units": index.n_units,
        "config": {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in CONFIG.items()
            if k != "embedding_model"
        },
        "categories": {},
        "timings_seconds": {},
    }

    for cat in categories:
        code = cat["code"]
        queries = plans[code]
        block = query_vecs[offset : offset + len(queries)]
        offset += len(queries)

        t0 = time.time()
        rows = candidate_rows(index, cat, undergraduate, filter_warnings)
        t_filter = time.time() - t0
        if rows.size == 0:
            # Loud, recorded, and never invented: downstream must write nothing for this
            # category rather than fill the gap. A whole index with nothing anywhere is a
            # different problem and raises below.
            log(
                f"  WARNING category {code}: the pre-filter left no candidate rows in "
                f"{index.dir}; this category will be empty"
            )
            empty_categories.append({"category_code": code, "reason": "pre-filter matched no rows"})
            results[code] = []
            explain["categories"][code] = {
                "name": cat["name"],
                "queries": [q.as_dict() for q in queries],
                "candidate_rows": 0,
                "candidates_by_kind": {kind: 0 for kind in UNIT_KINDS},
                "own_code_quota": {
                    "own_rows": 0,
                    "pool_rows": 0,
                    "quota_per_query": 0,
                    "second_pass": False,
                    "queries_short": 0,
                },
                "gapfill": {"enabled": gapfill_enabled, "ran": False, "reason": "no candidate rows"},
                "fused_units": 0,
                "returned": 0,
                "seconds": {
                    "filter": round(t_filter, 4), "vector": 0.0, "keyword": 0.0, "gapfill": 0.0
                },
                "selected": [],
                "dropped_by_caps": [],
                "empty_reason": "pre-filter matched no rows",
            }
            continue

        t0 = time.time()
        vhits = vector_search(index, rows, block, int(CONFIG["vector_top_k"]))
        t_vector = time.time() - t0

        t0 = time.time()
        table = load_candidate_table(index, rows)
        khits = [
            keyword_search(index, rows, q.keyword_text, int(CONFIG["keyword_top_k"]), table)
            for q in queries
        ]
        t_keyword = time.time() - t0

        # before fusion, not after: a row that is in no query's list cannot be fused, rescored or
        # selected, so the category's own material has to be guaranteed a place in those lists
        own_rows = own_code_rows(index, cat, rows)
        vhits, khits, quota_info = apply_own_code_quota(
            index, cat, rows, own_rows, block, queries, vhits, khits
        )
        # and the kinds this chapter exists to name -- courses for Academics, clubs for
        # Extracurriculars -- are guaranteed a place in the lists too, or the floor is empty
        vhits, kind_quota_info = apply_kind_quota(index, cat, rows, block, vhits, queries)
        kind_order = {k: v.get("ordered", []) for k, v in kind_quota_info.items()}

        hits = fuse(vhits, khits, queries)
        if not hits:
            die(
                f"category {code}: {rows.size:,} rows survived the pre-filter but both search "
                f"halves came back empty for {len(queries)} queries -- the index or the "
                f"embedding model is wrong, not the data"
            )
        scored = rescore(index, cat, hits)
        # Each query's strongest vector hits, interleaved by rank: every query's #1 first, then
        # every query's #2. Taking only #1 per query was not enough. For one applicant the
        # "artificial intelligence ... student clubs" query ranked the single best-matching
        # organisation in the whole corpus at #2 and a sentence saying a page HAS a club list at
        # #1, so the reserved slot went to the sentence and the organisation was never seen.
        # Interleaving by rank keeps breadth first and still reaches past a weak leader.
        query_best = reserved_by_rank(vhits, int(CONFIG["query_slot_depth"]),
                                      skip=index.process_boilerplate | index.foreign_institution)
        chosen, drops, texts = select_units(index, cat, scored, per_category,
                                            query_best=query_best, kind_order=kind_order)

        # SECOND PASS. What did this category miss? Same rows, same temp table, BM25 only.
        t0 = time.time()
        gap_explain: dict[str, Any] = dict(GAP_EXPLAIN_OFF)
        gap_scored: dict[int, dict[str, Any]] = {}
        if gapfill_enabled and gap_stats is not None:
            candidates, gap_scored, gap_explain = gap_pass(
                index, profile, cat, rows, queries, chosen, texts, scored,
                gap_stats, table, corpus_stop,
            )
            gap_explain["slots"] = gapfill.gap_slots(per_category)
            if candidates:
                chosen, drops, texts = select_units(
                    index, cat, scored, per_category, candidates, gap_scored,
                    query_best=query_best, kind_order=kind_order,
                )
                gap_explain["filled"] = sum(
                    1
                    for r in chosen
                    if str((scored.get(r) or gap_scored.get(r) or {}).get("selected_because", ""))
                    .startswith("gap slot")
                )
        elif gapfill_enabled:
            gap_explain = {"enabled": True, "ran": False, "reason": "index has no entity graph"}
        t_gap = time.time() - t0

        if not chosen:
            log(
                f"  WARNING category {code}: {len(hits):,} units were found but every one was "
                f"blocked by a diversity cap; this category will be empty"
            )
            empty_categories.append({"category_code": code, "reason": "every candidate hit a diversity cap"})
        missing = [r for r in chosen if r not in texts]
        if missing:
            texts.update(index.fetch_text(missing))
        entries = {**gap_scored, **scored}
        results[code] = [evidence_unit(index, r, entries[r], texts[r]) for r in chosen]

        explain["categories"][code] = {
            "name": cat["name"],
            "queries": [q.as_dict() for q in queries],
            "candidate_rows": int(rows.size),
            "candidates_by_kind": {
                kind: int(np.count_nonzero(index.kind_arr[rows] == kind)) for kind in UNIT_KINDS
            },
            "own_code_quota": quota_info,
            "gapfill": gap_explain,
            "fused_units": len(hits),
            "returned": len(results[code]),
            "seconds": {
                "filter": round(t_filter, 4),
                "vector": round(t_vector, 4),
                "keyword": round(t_keyword, 4),
                "gapfill": round(t_gap, 4),
            },
            "selected": [
                {
                    "rank": i + 1,
                    "unit_id": index.unit_id[r],
                    "kind": index.kind[r],
                    "category_code": index.category_code[r],
                    "entity_name": index.entity_name[r],
                    "source_url": index.source_url[r],
                    "source_kind": index.source_kind[r],
                    "year": int(index.year[r]) if int(index.year[r]) > 0 else None,
                    "score": round(entries[r]["score"], 6),
                    "rrf": round(entries[r]["hit"].rrf, 6),
                    "multiplier": round(entries[r]["multiplier"], 4),
                    "boosts": entries[r]["boosts"],
                    "selected_because": entries[r].get("selected_because", "rank"),
                    "found_by": sorted(
                        entries[r]["hit"].queries.values(),
                        key=lambda q: (
                            q["vector_rank"] if q.get("vector_rank") is not None else 10**6,
                            q["keyword_rank"] if q.get("keyword_rank") is not None else 10**6,
                            q["qid"],
                        ),
                    ),
                }
                for i, r in enumerate(chosen)
            ],
            "dropped_by_caps": drops,
        }

    explain["empty_categories"] = empty_categories
    explain["filter_warnings"] = filter_warnings
    explain["gapfill"] = {
        "enabled": gapfill_enabled,
        "graph_available": gap_stats is not None,
        "config": {
            k: (list(v) if isinstance(v, tuple) else v) for k, v in gapfill.CONFIG.items()
        },
        "queries": sum(
            len((explain["categories"].get(c) or {}).get("gapfill", {}).get("queries") or [])
            for c in explain["categories"]
        ),
        "slots_filled": sum(
            int((explain["categories"].get(c) or {}).get("gapfill", {}).get("filled") or 0)
            for c in explain["categories"]
        ),
        "df_lookups": gap_stats.df_lookups if gap_stats else 0,
        "graph_lookups": gap_stats.graph_lookups if gap_stats else 0,
        "model_calls": 0,
        "cost_usd": 0.0,
    }
    explain["timings_seconds"] = {
        "query_embedding": round(embed_seconds, 3),
        "gapfill": round(
            sum(
                float((explain["categories"].get(c) or {}).get("seconds", {}).get("gapfill") or 0.0)
                for c in explain["categories"]
            ),
            3,
        ),
        "total": round(time.time() - started, 3),
        "queries": len(all_queries),
    }
    if not any(results.values()):
        die(
            f"every category came back empty from {index.dir} ({index.n_units:,} units). "
            f"That is a wrong index or a category vocabulary that does not match it, not a "
            f"thin college -- refusing to hand generation an empty evidence set."
        )
    # LAST: the person-level joins, unblended, seated at the front of their chapters
    explain["anchors"] = anchor_pass(index, profile, embedder, results, per_category)
    if explain["anchors"]:
        log(f"  anchors: {len(explain['anchors'])} seated -- "
            + "; ".join(f"{a['chapter']}: {a['artefact'][:34]}" for a in explain["anchors"][:4]))
    return results, explain


def print_explain(explain: dict[str, Any]) -> None:
    log("")
    log(
        f"retrieval for student {explain['student_id']} against {explain['college_id']} "
        f"({explain['index_units']:,} indexed units, undergraduate filter "
        f"{'ON' if explain['undergraduate_filter'] else 'OFF'})"
    )
    for code in CATEGORY_ORDER:
        block = explain["categories"].get(code)
        if not block:
            continue
        log("")
        log(f"=== {code} {block['name']} ===")
        kinds = ", ".join(f"{k} {v:,}" for k, v in block["candidates_by_kind"].items() if v)
        log(f"  pre-filter: {block['candidate_rows']:,} of {explain['index_units']:,} rows ({kinds})")
        quota = block.get("own_code_quota") or {}
        if quota.get("second_pass"):
            log(
                f"  own-code quota: {quota['own_rows']:,} of {quota['pool_rows']:,} rows carry this "
                f"category's codes; {quota['queries_short']} quer(ies) fell short of "
                f"{quota['quota_per_query']} and were topped up from an own-code-only pass"
            )
        for query in block["queries"]:
            log(f"  query {query['qid']} [{query['facet']}] w={query['weight']}: {query['text']}")
        gap = block.get("gapfill") or {}
        for query in gap.get("queries") or []:
            blind = "facet UNREAD by this category" if query["facet_unread_by_first_pass"] else "uncovered"
            log(
                f"  GAP query {query['qid']} [{query['facet']}, {blind}] anchor "
                f"{query['lead_term']!r} (graph fit {query['graph_fit']}): {query['keyword_text']}"
            )
            log(f"      from resume line: {query['basis'][:110]}")
        if gap.get("enabled") and not (gap.get("queries") or []) and gap.get("reason"):
            log(f"  GAP: none ({gap['reason']})")
        log(
            f"  fused {block['fused_units']:,} units in "
            f"{block['seconds']['vector']:.2f}s vector + {block['seconds']['keyword']:.2f}s keyword"
        )
        for item in block["selected"]:
            found = ", ".join(
                f"{q['qid']}(v{q.get('vector_rank') or '-'}/k{q.get('keyword_rank') or '-'})"
                for q in item["found_by"]
            )
            boosts = " ".join(f"{k}x{v}" for k, v in sorted(item["boosts"].items()) if v != 1.0)
            name = item["entity_name"] or item["source_url"]
            log(f"  {item['rank']:>2}. [{item['kind']}] {name[:70]}")
            log(f"      score {item['score']:.5f} = rrf {item['rrf']:.5f} x {item['multiplier']:.3f}  {boosts}")
            log(f"      found by {found}  ({item['selected_because']})")
        if block["dropped_by_caps"]:
            log("  dropped by diversity caps:")
            for drop in block["dropped_by_caps"][:5]:
                log(f"      {drop['unit_id']} ({drop['score']:.5f}) - {drop['reason']}")


def write_results(out_path: Path, results: dict[str, list[dict[str, Any]]]) -> None:
    """Write exactly {category_code: [EvidenceUnit, ...]}, categories in the fixed order."""
    ordered = {code: results[code] for code in CATEGORY_ORDER if code in results}
    for code in results:
        if code not in ordered:
            ordered[code] = results[code]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(ordered, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Hybrid retrieval: StudentProfile + college index -> EvidenceUnits per category."
    )
    parser.add_argument("--profile", type=Path, required=True, help="StudentProfile JSON (wwrag/profile.py)")
    parser.add_argument(
        "--index",
        type=Path,
        required=True,
        help="index root (<root>/<college-id>/) or a college index directory",
    )
    parser.add_argument("--out", type=Path, required=True, help="where to write {category_code: [EvidenceUnit]}")
    parser.add_argument(
        "--per-category",
        type=int,
        default=int(CONFIG["default_per_category"]),
        help="evidence units per category (default %(default)s)",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="print why each unit was chosen and write <out>.explain.json",
    )
    parser.add_argument(
        "--no-gapfill",
        action="store_true",
        help="turn OFF the second, gap-closing retrieval pass (wwrag/gapfill.py). The pass is on "
             "by default; this flag reproduces the output the pipeline produced before it existed, "
             "so a before/after stays measurable.",
    )
    parser.add_argument("--college-id", default=None, help="pick one index when --index holds several")
    parser.add_argument("--threads", type=int, default=None, help="onnxruntime threads for query embedding")
    parser.add_argument(
        "--categories",
        default=None,
        help="comma-separated category codes to run (default: all ten, in order)",
    )
    args = parser.parse_args(argv)

    index_dir = resolve_index_dir(args.index, args.college_id)
    profile = load_profile(args.profile)

    selected = CATEGORIES
    if args.categories:
        codes = {c.strip().upper() for c in args.categories.split(",") if c.strip()}
        unknown = sorted(c for c in codes if c not in CATEGORY_BY_CODE)
        if unknown:
            die(f"unknown category codes: {', '.join(unknown)}")
        selected = tuple(CATEGORY_BY_CODE[c] for c in CATEGORY_ORDER if c in codes)

    t0 = time.time()
    index = CollegeIndex(index_dir)
    log(f"opened index {index_dir} ({index.n_units:,} units, {index.dims} dims) in {time.time() - t0:.2f}s")

    try:
        embedder = Embedder(index.model_name, threads=args.threads, dims=index.dims)
        results, explain = retrieve(
            index, profile, args.per_category, embedder, selected,
            gapfill_enabled=not args.no_gapfill,
        )
    finally:
        index.close()

    write_results(args.out, results)
    total = sum(len(v) for v in results.values())
    log(
        f"wrote {total} evidence units across {len(results)} categories to {args.out} "
        f"in {explain['timings_seconds']['total']:.2f}s "
        f"({explain['timings_seconds']['queries']} queries, "
        f"{explain['timings_seconds']['query_embedding']:.2f}s embedding)"
    )

    if args.explain:
        explain_path = Path(str(args.out) + ".explain.json")
        explain_path.write_text(
            json.dumps(explain, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
        )
        print_explain(explain)
        log("")
        log(f"explain written to {explain_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
