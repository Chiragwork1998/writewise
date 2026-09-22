"""Anchor-gap retrieval: a deterministic SECOND search pass over the SAME candidate rows.

WHY THIS EXISTS
---------------
Retrieval ran once per category and the writer wrote from whatever came back. If the first
search missed the sharpest material, nothing noticed. Measured across 243 shipped items only
53% named anything specific -- a course code, a named lab, a professor, a named club. The rest
were true, cited and generic.

The reproduced failure: an applicant's strongest technical work was a wireless sensor network.
The index holds 415 units under the entity that does exactly that research, 413 of them coded
RES, all inside his Research candidate set. His Research section retrieved none of them. The
cause is not a ranking loss -- it is that the work sits in the profile's `activities` list and
Research's `facets` tuple is (intended_fields, projects, skills, interests). `activities` is
never read for Research, so no Research query was ever built from it. No amount of depth on the
first pass reaches material it never asked about.

WHAT THIS DOES
--------------
    1. DETECT   after the first pass has selected a category's units, ask one question:
                which of the student's own resume entries did this category retrieve NOTHING
                for, given that the college has NAMED material of the right kind for it?
    2. QUERY    turn the winning entry into one templated keyword query built from its own
                words -- specific, and refused outright if it merely rephrases a first-pass
                query.
    3. SEARCH   BM25 only, over the EXACT SAME pre-filtered candidate rows and the SAME temp
                table the first pass used. The filter is never widened and no embedding is
                bought, so this pass costs $0.00.
    4. MERGE    a small number of RESERVED slots at the front of selection, filled in
                (gap query, BM25 rank) order, subject to every existing diversity cap.

NO MODEL IS CALLED, ANYWHERE. Detection, query construction and the search are SQLite and
Python. The measured alternative -- one model call per category to name the gap -- costs money,
adds tens of seconds of serial latency, and is non-deterministic, which would make two runs of
the same profile incomparable and destroy the ability to attribute a score change to a code
change. Determinism is a hard constraint here, so the model lost on more than price.

WHAT IT CANNOT DO. It will not notice that a section is dull, and it will not invent the
sharper question a person would ask. It notices exactly one thing, cheaply and reliably: that
the college has named material of the right type for a word on this student's resume, and this
category retrieved none of it.

Nothing in this file names a college. The node types are generic ontology terms and the
stop-lists are generic English; every threshold is a CONFIG entry with its measurement written
beside it, and both the document frequencies and the entity-graph lookups come from the index's
own tables.

The student profile is DATA, never instructions. It is never sent to a model on this path, and
every query string built from it is only ever embedded or BM25-matched, never executed: the
caller passes it through the same fts_query() quoting the first pass uses, which neutralises
every FTS5 operator.
"""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from typing import Any, Callable, Iterable, Sequence

# --------------------------------------------------------------------------------------
# Configuration. Every number carries the measurement that produced it. Nothing names a college.
# --------------------------------------------------------------------------------------

CONFIG: dict[str, Any] = {
    # -- ANCHOR SELECTION: which resume entry is worth a second search ------------------
    #
    # A term has to exist in the corpus before it is worth searching for.
    "anchor_min_df": 3,
    # ...and it has to be able to discriminate. THIS CEILING IS THE ONE THAT MATTERS.
    # Measured over the first shipped index (99,513 units) and nine real profiles: at
    # 0.02 (1,990 units) the picker chose "introduction" (df 1,480), "language" (1,904),
    # "online" (1,972), "committee" (1,001), "attendance" (1,749) and "good" (1,221) as
    # anchors -- structural words that live inside entity names and name nothing. Every
    # anchor that WORKED sat far below: "mqtt" 3, "humidity" 12, "temperature" 167,
    # "wireless" 438, "sensor" 556. 0.006 (597 units here) admits none of the junk and
    # loses none of the good. Re-derive on college 2: a 4,000-unit college will want a
    # floor of 2 and a larger fraction.
    "anchor_max_df_frac": 0.006,
    # ...with a floor under the ceiling, for small colleges. The fraction is meaningless on a
    # thin index -- on a few thousand units it lands below the minimum document frequency and
    # NOTHING can anchor, silently. Below this many units the graph gate is the whole filter,
    # which is the right answer there: in a small corpus a word matching ten named entities of
    # the right type is a strong signal whatever its document frequency.
    "anchor_min_df_ceiling": 25,
    # A term shorter than this is noise in a word-boundary graph match.
    "anchor_min_term_chars": 4,
    # The term must word-boundary-match the NAMES of at least this many graph nodes whose
    # type this category wants. That second gate is what separates "sensor" (86 research
    # nodes) from "grocery", "hindi", "accenture", "tableau" and "power bi" (0 each). It
    # costs one indexed-name scan per term, ~10ms, and reads only the index's own graph.
    #
    # THE FLOOR IS 10, NOT 3, AND THE DIFFERENCE IS MOST OF THE PASS'S QUALITY. Measured on
    # nine real profiles: every anchor that produced the right material scored in double or
    # triple figures -- sensor 86, math 83, chemistry 23, cinema 18, literary 15, spanish 13,
    # decision making 13, business analytics 11, basketball 10, project management 10. Every
    # anchor that produced junk scraped the bottom -- beach 4, church 5 (a professor whose
    # surname it is), lawn 5 (a patch of grass), partial 5, fast 6, dashboard 3, league 3,
    # gallery 6, monthly 3. At a floor of 3 a Research section anchored on "beach" and
    # returned two lakes while losing a robotics lab; at 10 it does not run at all, which is
    # the right answer. Lowering this is the fastest way to make the pass worse.
    "anchor_min_graph_fit": 10,
    # At most this many gap queries per category.
    "max_gap_queries": 2,
    # WHICH CATEGORIES GET A GAP PASS AT ALL. A gap query is built from the student's own
    # words with no category lens in it -- BM25 on a lens phrase faithfully returns every page
    # that repeats the lens, which is why the first pass keeps the lens out of its keyword half
    # too. So a gap query is a maximally student-led query, and the CATEGORIES table already
    # carries 60 blind judgments on exactly that question: for the five CATEGORY-led
    # categories, leading with the student drags wrong-topic material in (Diversity 3.3 vs 5.7,
    # Quirks 2.3 vs 3.3, Culture 5.0 vs 6.0). Measured here on a real profile, that prediction
    # held: with the pass on everywhere, Quirks and Social Impact filled reserved slots with
    # construction-management degree programmes, because the student's project tracked
    # construction workers. Restricting the pass to the student-led categories keeps every
    # gain on the concrete questions -- which labs, which courses, which clubs, which press --
    # and gives up the noise on the abstract ones. Set False to run it everywhere.
    "student_led_only": True,
    # -- QUERY CONSTRUCTION -------------------------------------------------------------
    #
    # The query is built from ALL of the entry's corpus-present words, not just the ones
    # that won the anchor gate. Measured: the nine-term string put the target research
    # group at BM25 ranks 0,1,2,4,5; trimming it to the five rarest pushed the target to
    # rank 9 and put a wrong lab at rank 0. The graph gate picks WHICH entry, not which
    # words. So query terms get a much looser ceiling than anchor terms -- they only have
    # to not be corpus-wide boilerplate.
    "query_max_df_frac": 0.05,
    "query_max_terms": 12,
    # A rephrase of a first-pass query is not a gap query. Measured: a deliberate rephrase
    # put the target at rank ~149 with nothing in the top 80. Refuse, do not merely
    # discourage: if this share of the gap query's content words already appear in any one
    # first-pass query for this category, the gap query is dropped.
    "max_rephrase_overlap": 0.6,
    # -- SEARCH -------------------------------------------------------------------------
    "gap_top_k": 40,
    # A gap query with no real match must contribute nothing. BM25 is negative and larger
    # magnitude is better; a candidate is kept only at this share of its own query's best.
    "bm25_floor_frac": 0.50,
    # -- MERGE --------------------------------------------------------------------------
    #
    # Reserved slots, as a share of the section. The design was measured at 2 of 12; the
    # production pipeline runs --per-category 24, where a fixed 2 would be 8% of a section
    # instead of the 17% that was measured. So it scales, and it is capped so a long
    # section cannot be taken over.
    "gap_slot_fraction": 1.0 / 6.0,
    "gap_slots_max": 4,
    # Within a window of this many BM25 ranks, prefer a fact that reads as an opportunity
    # over one that reads as a bibliography entry. A sharply targeted gap query lands on the
    # densest topical material, and for a research group that is its publication list.
    #
    # Swept over nine real profiles (120 reserved slots filled, the named-unit count identical
    # at every setting, so this only changes WHICH named unit ships):
    #     window   slots reading as bibliography   slots on the target group's own host
    #        3                  9                                7
    #        8                  8                                6
    #       20                  5                                7
    #       40 (no window)      0                                4
    # 20 halves the bibliography share without costing a single hit on the entity the pass was
    # built to find; dropping the window entirely clears the bibliography but starts trading
    # away the target. It stays a TIE-BREAK inside a window, never a hard preference, because
    # a hard one lets a rank-39 row outrank a rank-0 one.
    "actionability_window": 20,
}

# Generic heads: words that name a KIND of thing rather than a particular one. A resume word
# that is one of these cannot anchor a search, however rare it happens to be in this corpus.
GENERIC_TERMS = frozenset(
    """
    university college school schools student students undergraduate graduate program programs
    programme research centre center institute department departments faculty course courses
    class classes campus community organization organizations association society club clubs
    engineering science sciences technology studies studying academic academics education
    national international american global general office division project projects laboratory
    work working worked team teams member members group groups intern internship experience
    skills skill leadership management managed manager developed development create created
    creating design designed designing build building built system systems product products
    data analysis analytics tools tool platform platforms process processes improve improved
    increase increased reduce reduced support supported provide provided help helped lead led
    role detail details name based using used various several around including include included
    people person year years month months time times first second third best good great
    online offline remote local national regional introduction introductory advanced beginner
    committee council board chapter volunteer volunteering award awards honor honors
    the and for with from that this they their our his her its into over under about
    """.split()
)

_WORD_RE = re.compile(r"[A-Za-z0-9']+")
# UPPERCASE, deliberately. A case-insensitive version matches a bibliography's page numbers
# ("pp. 205") and an untitled publications chunk then counts as naming a course. This is the
# same shape wwrag/score_run.py counts as specific, so the filter and the metric agree.
_COURSE_CODE_RE = re.compile(r"\b[A-Z]{2,5}[\s–-]?\d{3,5}[A-Za-z]?\b|\b\d{1,2}\.\d{2,4}\b")
_ALNUM_RE = re.compile(r"[^a-z0-9]+")

# Prose that reads as a bibliography entry rather than as something an applicant can do. Used
# ONLY as a tie-break inside a narrow rank window (see actionability_window).
PUBLICATION_SHAPE = re.compile(
    r"\b(published|publication|proceedings|preprint|co-authored|coauthored|cited by|"
    r"in the journal|conference paper|was presented at|the paper |the bibliography|"
    r"a \d{4} paper|vol\.|pp\.|doi:)\b",
    re.IGNORECASE,
)

# Graph node types each category is ABOUT. This is the second half of the anchor gate: a resume
# word only counts when the college's own entity graph holds that many things of the right KIND
# whose NAME contains it. Generic ontology terms, written next to the category they serve; an
# index whose graph uses other type names simply scores 0 and the gate turns itself off out loud
# (see missing_graph_note).
#
# KEEP THESE NARROW. Measured over nine real profiles: with broad lists, incidental proper-noun
# fragments qualified as anchors -- a Research section anchored on "beach" (a place in an
# activity's name) and returned two lakes and an artist while losing a robotics lab; another
# anchored on "church" and returned a professor whose SURNAME is Church. Every anchor that
# worked -- "sensor", "math", "cinema", "spanish", "literary", "basketball", "data analysis" --
# matched the kind of thing its category is actually about. Narrowing the lists is what
# separates the two, and it costs nothing.
#
# An EMPTY tuple means "no gap pass for this category", and NEW is empty on measurement, not on
# principle: its pre-filter deliberately admits outside sources, which in a real index includes
# pages about OTHER universities, and every gap query built from a student's own words landed on
# them -- another university's debating society, a rival team, a conference in another state.
# The material is real, named and cited, and belongs to the wrong college.
CATEGORY_ANCHOR_TYPES: dict[str, tuple[str, ...]] = {
    "CUL": ("tradition", "organization", "event", "program", "center"),
    "EXT": ("organization", "program", "event", "facility", "service", "award"),
    "QRK": ("tradition", "organization", "event", "award", "facility"),
    "ACA": ("course", "program", "department", "school", "center"),
    "RES": ("lab", "center", "publication", "facility", "department", "program"),
    "SOC": ("organization", "service", "program", "partner", "center", "event"),
    "INN": ("program", "center", "lab", "facility", "organization", "course"),
    "INT": ("course", "program", "department", "center", "school"),
    "DIV": ("organization", "program", "center", "service", "event"),
    "NEW": (),
}

# PEOPLE ARE NEVER AN ANCHOR TYPE, AND ALWAYS AN ANSWER TYPE.
#
# A `professor` or `person` node's NAME is a personal name, so a resume word matching one says
# only that the word is somebody's surname -- it carries no topical signal at all. Measured:
# "douglas" matched 10 professors and 6 people and became a top-ranked anchor; "church" matched
# 3 professors. Both produced sections about the wrong things.
#
# The opposite is true of the ANSWER: a named professor is the single most useful thing a
# retrieved unit can carry, so a gap hit naming one always counts as nameable.
ANSWER_ONLY_TYPES: tuple[str, ...] = ("professor", "person")

# ...but only where a named person is what the section is FOR. Academics is about "courses and
# the professors who teach them", Research about "named professors and labs", and Intellectual
# Alignment about who a student would think alongside -- those three intents say so in the
# CATEGORIES table already. Extracurriculars is about clubs, and measured, letting people
# answer there filled a clubs section with four professors.
PEOPLE_ANSWER_CATEGORIES: frozenset[str] = frozenset({"ACA", "RES", "INT"})


def answer_types(category_code: str) -> tuple[str, ...]:
    """Node types a gap hit may name in order to take a reserved slot."""
    anchors = CATEGORY_ANCHOR_TYPES.get(category_code, ())
    if not anchors:
        return ()
    if category_code in PEOPLE_ANSWER_CATEGORIES:
        return anchors + ANSWER_ONLY_TYPES
    return anchors

# Profile facets a gap anchor may be drawn from, and the order they are preferred in when two
# entries tie. `activities` leads because that is where the reproduced miss lived.
ANCHOR_FACETS: tuple[str, ...] = (
    "activities",
    "projects",
    "achievements",
    "intended_fields",
    "skills",
    "interests",
    "values",
)


def log(msg: str) -> None:
    print(msg, flush=True)


def _clean(value: Any, cap: int = 400) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]", " ", text)
    return re.sub(r"\s+", " ", text).strip()[:cap].strip()


# --------------------------------------------------------------------------------------
# Profile entries a gap may anchor on
# --------------------------------------------------------------------------------------


class Anchor:
    """One resume entry, with the words a search could actually use."""

    __slots__ = ("facet", "position", "basis", "terms", "fit", "lead_term", "blind")

    def __init__(self, facet: str, position: int, basis: str) -> None:
        self.facet = facet
        self.position = position
        self.basis = basis
        self.terms: list[str] = []
        self.fit = 0
        self.lead_term = ""
        self.blind = False


def anchor_entries(profile: dict[str, Any]) -> list[Anchor]:
    """Every profile entry a gap could anchor on, from EVERY facet, in profile order.

    This deliberately reads more of the profile than `retrieve.profile_facets` does, in two
    ways, and both are the point:

      * EVERY facet, including the ones a given category's `facets` tuple never consults --
        that omission is the reproduced failure itself;
      * `evidence_line` as well as name/role/detail. The sharpest word on the reproduced
        resume ("mqtt", document frequency 3 in a 99,513-unit index) appears ONLY in the
        activity's evidence_line, which the first pass never reads, so the most
        discriminating token the student owns was invisible to retrieval.

    Reading it HERE rather than widening `profile_facets` is deliberate: the first pass's
    queries stay byte-identical, so a run with this pass off reproduces the old output exactly
    and the before/after stays attributable.
    """
    out: list[Anchor] = []
    for facet in ANCHOR_FACETS:
        raw = profile.get(facet)
        if not isinstance(raw, list):
            continue
        for position, entry in enumerate(raw):
            if isinstance(entry, str):
                basis = _clean(entry)
            elif isinstance(entry, dict):
                # each field capped separately, then the whole thing capped: a long
                # evidence_line must not be able to push the fields after it out of range,
                # and an evidence_line must not be truncated away by the fields before it
                parts = [
                    _clean(entry.get(field), 240)
                    for field in ("name", "role", "detail", "label", "title", "evidence_line")
                ]
                basis = _clean(" ".join(dict.fromkeys(p for p in parts if p)), 900)
            else:
                continue
            if basis:
                out.append(Anchor(facet, position, basis))
    return out


def candidate_terms(text: str, stop: Iterable[str]) -> list[str]:
    """Content unigrams (>= min chars) and adjacent bigrams from one resume entry, in order.

    Bigrams matter: a two-word technique is a far better anchor than either half, and the
    corpus counts it as a PHRASE, not as an OR of its words. De-duplicated, order preserved,
    so the same entry always yields the same list.
    """
    stop_set = {s.lower() for s in stop}
    min_chars = int(CONFIG["anchor_min_term_chars"])
    words = [w.replace("'", "").lower() for w in _WORD_RE.findall(text)]
    kept = [
        w
        for w in words
        if len(w) >= min_chars and w not in stop_set and w not in GENERIC_TERMS and not w.isdigit()
    ]
    terms: list[str] = []
    seen: set[str] = set()
    for word in kept:
        if word not in seen:
            seen.add(word)
            terms.append(word)
    # adjacent bigrams over the ORIGINAL word order, so "after effects" survives even though
    # neither half is distinctive on its own
    for left, right in zip(words, words[1:]):
        if len(left) < min_chars or len(right) < min_chars:
            continue
        if left in stop_set or right in stop_set:
            continue
        bigram = f"{left} {right}"
        if bigram not in seen:
            seen.add(bigram)
            terms.append(bigram)
    return terms


# --------------------------------------------------------------------------------------
# The index's own statistics: document frequency, and what the entity graph is named after
# --------------------------------------------------------------------------------------


class IndexStats:
    """Cached per-term corpus statistics for one index. Student-independent, so it is built
    once per college and reused by every student and every category after that.

    Both lookups read the index's own tables, so nothing here knows which college is running.
    """

    def __init__(self, conn: sqlite3.Connection, fts_table: str, n_units: int) -> None:
        self.conn = conn
        self.fts_table = fts_table
        self.n_units = max(1, int(n_units))
        self._df: dict[str, int] = {}
        self._fit: dict[str, dict[str, int]] = {}
        self._entity: dict[str, set[str]] = {}
        self.df_lookups = 0
        self.graph_lookups = 0
        self.has_graph = False
        try:
            row = conn.execute(
                "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='graph_nodes'"
            ).fetchone()
            if row and int(row[0]):
                node_count = conn.execute("SELECT count(*) FROM graph_nodes").fetchone()
                self.has_graph = bool(node_count and int(node_count[0]))
        except sqlite3.Error:
            self.has_graph = False

    def df(self, term: str) -> int:
        """How many units contain this term, as a PHRASE.

        A phrase, not an OR: asking the FTS index for `"after effects"` as two words reports
        the document frequency of "after" OR "effects", which is thousands, and a two-word
        anchor then fails a ceiling it should sail through.
        """
        cached = self._df.get(term)
        if cached is not None:
            return cached
        match = '"' + term.replace('"', "") + '"'
        try:
            row = self.conn.execute(
                f"SELECT count(*) FROM {self.fts_table} WHERE {self.fts_table} MATCH ?", (match,)
            ).fetchone()
            value = int(row[0]) if row else 0
        except sqlite3.Error:
            value = 0
        self.df_lookups += 1
        self._df[term] = value
        return value

    def graph_fit(self, term: str) -> dict[str, int]:
        """{node type: how many graph node NAMES contain this term as a whole word}.

        Word-boundary matching is load-bearing, not polish. With a bare substring match the
        picker chose "mar" (matching seminar and every school with those letters), "min" and
        "era" -- all of which score in the hundreds and mean nothing. Four SQL patterns cover
        the four whole-word positions.
        """
        cached = self._fit.get(term)
        if cached is not None:
            return cached
        if not self.has_graph:
            self._fit[term] = {}
            return {}
        # LIKE has two wildcards; a term carrying either would silently widen the match
        if "%" in term or "_" in term:
            self._fit[term] = {}
            return {}
        try:
            rows = self.conn.execute(
                "SELECT type, count(*) FROM graph_nodes"
                " WHERE name_key = ? OR name_key LIKE ? OR name_key LIKE ? OR name_key LIKE ?"
                " GROUP BY type",
                (term, term + " %", "% " + term, "% " + term + " %"),
            ).fetchall()
            out = {str(r[0]): int(r[1]) for r in rows}
        except sqlite3.Error:
            out = {}
        self.graph_lookups += 1
        self._fit[term] = out
        return out

    def fit_for(self, term: str, node_types: Sequence[str]) -> int:
        fit = self.graph_fit(term)
        wanted = set(node_types)
        return sum(count for node_type, count in fit.items() if node_type in wanted)

    def entity_types(self, name: str | None) -> set[str]:
        """Which graph node types this entity name resolves to, exactly. Empty when the graph
        has never heard of it."""
        if not name or not self.has_graph:
            return set()
        key = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", name)).strip().casefold()
        cached = self._entity.get(key)
        if cached is not None:
            return cached
        try:
            rows = self.conn.execute(
                "SELECT DISTINCT type FROM graph_nodes WHERE name_key = ?", (key,)
            ).fetchall()
            value = {str(r[0]) for r in rows}
        except sqlite3.Error:
            value = set()
        self._entity[key] = value
        return value

    def is_publication_entity(self, name: str | None) -> bool:
        return "publication" in self.entity_types(name)


# --------------------------------------------------------------------------------------
# DETECT: which resume entry did this category retrieve nothing for?
# --------------------------------------------------------------------------------------


def covered_texts(selected: Iterable[tuple[str, str, str]]) -> list[str]:
    """The part of the first pass's output that counts as COVERAGE, as (entity, text, quote).

    Only NAMED material counts. The question this pass asks is "does the college have named
    things of the right type for this resume word, and did this category retrieve none of
    them" -- so an unnamed page chunk that happens to contain the word in passing is not an
    answer to it.

    Measured, and it is not a corner case: the reproduced miss was cancelled outright by a
    single untitled faculty-directory chunk whose prose mentioned the word once, while the
    research group that actually does that work -- 415 units of it, all inside the same
    candidate set -- was still nowhere in the section. Counting that chunk as coverage made
    the pass silently decide there was no gap.
    """
    out: list[str] = []
    for entity, text, quote in selected:
        if entity:
            out.append(entity)
            out.append(text)
            out.append(quote)
    return [t for t in out if t]


def covered_words(texts: Iterable[str]) -> set[str]:
    """Whole words already present in what the first pass selected.

    Whole words, not stems. A five-character stem collides badly on real data -- "shortlisted"
    stems to "short", which then matches a person's surname -- and a false "already covered"
    silently cancels the gap this pass exists to find.
    """
    out: set[str] = set()
    for text in texts:
        if not text:
            continue
        for word in _WORD_RE.findall(str(text).lower()):
            word = word.replace("'", "")
            if len(word) < int(CONFIG["anchor_min_term_chars"]):
                continue
            out.add(word)
            if word.endswith("s") and len(word) > 4:
                out.add(word[:-1])
    return out


def _term_covered(term: str, covered: set[str], haystack: str) -> bool:
    if " " in term:
        return term in haystack
    return term in covered or (term + "s") in covered


class GapQuery:
    """One gap query: where it came from, what it searches for, and why it was allowed."""

    __slots__ = ("qid", "category_code", "facet", "basis", "terms", "keyword_text", "lead_term",
                 "graph_fit", "blind")

    def __init__(self, qid: str, category_code: str, anchor: Anchor, terms: Sequence[str]) -> None:
        self.qid = qid
        self.category_code = category_code
        self.facet = anchor.facet
        self.basis = anchor.basis
        self.terms = list(terms)
        self.keyword_text = " ".join(terms)
        self.lead_term = anchor.lead_term
        self.graph_fit = anchor.fit
        self.blind = anchor.blind

    def as_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "category_code": self.category_code,
            "facet": self.facet,
            "basis": self.basis,
            "lead_term": self.lead_term,
            "graph_fit": self.graph_fit,
            "facet_unread_by_first_pass": self.blind,
            "terms": list(self.terms),
            "keyword_text": self.keyword_text,
        }


def _content_tokens(text: str) -> set[str]:
    return {
        w.replace("'", "").lower()
        for w in _WORD_RE.findall(text or "")
        if len(w) >= int(CONFIG["anchor_min_term_chars"])
        and w.lower() not in GENERIC_TERMS
    }


def is_rephrase(keyword_text: str, first_pass_texts: Sequence[str]) -> bool:
    """True when this query is a rewording of a query the first pass already ran.

    "Specific, not a rephrasing of the first" is the whole point of a second pass: a measured
    rephrase control put the target at rank ~149 with nothing in the top 80, i.e. it
    regenerated exactly the generic material that was already shipping.
    """
    mine = _content_tokens(keyword_text)
    if not mine:
        return True
    limit = float(CONFIG["max_rephrase_overlap"])
    for other in first_pass_texts:
        theirs = _content_tokens(other)
        if not theirs:
            continue
        overlap = len(mine & theirs) / len(mine)
        if overlap > limit:
            return True
    return False


def plan_gap_queries(
    profile: dict[str, Any],
    category: dict[str, Any],
    stats: IndexStats,
    selected: Sequence[tuple[str, str, str]],
    first_pass_query_texts: Sequence[str],
    corpus_stopwords: Iterable[str] = (),
) -> tuple[list[GapQuery], dict[str, Any]]:
    """The whole gap-finder. No model, no randomness, no network.

    An entry is a GAP for this category when
      (a) the college's own entity graph holds at least `anchor_min_graph_fit` NAMED things of
          a type this category wants whose name contains one of the entry's corpus-rare words,
          and
      (b) not one of the entry's searchable words appears anywhere in what the first pass
          selected -- not in an entity name, not in a quote, not in the prose.

    Entries whose facet this category's first pass never reads are ranked FIRST, because for
    those the gap is structural: no first-pass query could have found the material, whatever
    its depth. That is the exact shape of the reproduced miss.
    """
    explain: dict[str, Any] = {
        "ran": False,
        "reason": "",
        "anchors_considered": 0,
        "anchors_qualified": 0,
        "rejected_as_rephrase": 0,
        "queries": [],
    }
    node_types = CATEGORY_ANCHOR_TYPES.get(category["code"], ())
    if CONFIG["student_led_only"] and str(category.get("query_lead") or "") != "student":
        explain["reason"] = (
            "category-led category: the CATEGORIES table's blind measurement says the student's "
            "words must not lead this question, and a gap query is nothing but her words"
        )
        return [], explain
    if not stats.has_graph:
        explain["reason"] = "index has no entity graph; the anchor gate cannot run"
        return [], explain
    if not node_types:
        explain["reason"] = (
            f"no gap pass for category {category['code']}: its anchor node types are empty"
        )
        return [], explain

    read_facets = set(category.get("facets") or ())
    named = covered_texts(selected)
    covered = covered_words(named)
    haystack = " ".join(str(t or "").lower() for t in named)

    min_df = int(CONFIG["anchor_min_df"])
    anchor_ceiling = max(
        int(CONFIG["anchor_min_df_ceiling"]),
        int(stats.n_units * float(CONFIG["anchor_max_df_frac"])),
    )
    query_ceiling = max(anchor_ceiling, int(stats.n_units * float(CONFIG["query_max_df_frac"])))
    min_fit = int(CONFIG["anchor_min_graph_fit"])

    qualified: list[Anchor] = []
    for anchor in anchor_entries(profile):
        explain["anchors_considered"] += 1
        terms = candidate_terms(anchor.basis, corpus_stopwords)
        if not terms:
            continue
        # every corpus-present word of the entry is a QUERY term; only the rare ones may
        # ANCHOR it. Trimming the query to the rare words alone was measured to lose the
        # target entity outright.
        present = [(t, stats.df(t)) for t in terms]
        anchor.terms = [t for t, d in present if min_df <= d <= query_ceiling]
        if not anchor.terms:
            continue
        # The ANCHOR term -- the best-fitting rare word of this entry -- is what has to be
        # missing. Testing the WHOLE entry for coverage does not work and is worth spelling
        # out: a one-line resume entry carries a dozen ordinary words ("network", "planning",
        # "real", "transfer") and at least one of them appears somewhere in twenty-four
        # retrieved passages essentially always, so the entry looks covered when the thing it
        # is actually about was never retrieved. Measured: it cancelled the exact miss this
        # pass was built to close.
        best_fit, best_term = 0, ""
        for term, frequency in present:
            if not (min_df <= frequency <= anchor_ceiling):
                continue
            if _term_covered(term, covered, haystack):
                continue
            fit = stats.fit_for(term, node_types)
            if fit > best_fit or (fit == best_fit and fit and term < best_term):
                best_fit, best_term = fit, term
        if best_fit < min_fit:
            continue
        anchor.fit = best_fit
        anchor.lead_term = best_term
        anchor.blind = anchor.facet not in read_facets
        qualified.append(anchor)

    explain["anchors_qualified"] = len(qualified)
    if not qualified:
        explain["reason"] = "no uncovered resume entry has named material of the right type"
        explain["ran"] = True
        return [], explain

    # Rank: structurally-unreachable facets first (no first-pass query could have found this),
    # then strongest graph fit, then the resume's own order. Fully deterministic.
    facet_rank = {facet: i for i, facet in enumerate(ANCHOR_FACETS)}
    qualified.sort(
        key=lambda a: (0 if a.blind else 1, -a.fit, facet_rank.get(a.facet, 99), a.position)
    )

    max_terms = int(CONFIG["query_max_terms"])
    out: list[GapQuery] = []
    for anchor in qualified:
        if len(out) >= int(CONFIG["max_gap_queries"]):
            break
        # rarest first: BM25's IDF is what makes a rare token worth having, and the cap keeps
        # one long resume line from becoming a 40-clause MATCH
        ordered = sorted(anchor.terms, key=lambda t: (stats.df(t), t))[:max_terms]
        # keep the entry's own word order in the emitted string, rarest-first only decides
        # which words survive the cap
        keep = set(ordered)
        terms = [t for t in anchor.terms if t in keep]
        keyword_text = " ".join(terms)
        if is_rephrase(keyword_text, first_pass_query_texts):
            explain["rejected_as_rephrase"] += 1
            continue
        out.append(GapQuery(f"{category['code']}-gap{len(out) + 1}", category["code"], anchor, terms))

    explain["ran"] = True
    explain["queries"] = [q.as_dict() for q in out]
    if not out and not explain["rejected_as_rephrase"]:
        explain["reason"] = "no gap query survived construction"
    return out, explain


# --------------------------------------------------------------------------------------
# MERGE: which gap hits may take a reserved slot
# --------------------------------------------------------------------------------------


def gap_slots(per_category: int) -> int:
    """How many of a section's slots are reserved for gap hits."""
    if per_category <= 2:
        return 0
    want = int(round(per_category * float(CONFIG["gap_slot_fraction"])))
    return max(1, min(int(CONFIG["gap_slots_max"]), want))


def apply_bm25_floor(hits: Sequence[tuple[int, float]]) -> list[tuple[int, float]]:
    """Drop everything below a share of this query's OWN best score.

    bm25() is negative and more negative is better. A gap query whose corpus match is weak
    should contribute nothing at all rather than its least-bad row.
    """
    if not hits:
        return []
    best = min(score for _, score in hits)  # most negative == strongest
    if best >= 0:
        return []
    floor = best * float(CONFIG["bm25_floor_frac"])
    return [(row, score) for row, score in hits if score <= floor]


def is_nameable(
    entity_name: str | None,
    text: str,
    stop: Iterable[str],
    stats: "IndexStats | None" = None,
    node_types: Sequence[str] = (),
    kind: str = "",
) -> bool:
    """A gap hit may only take a reserved slot if it NAMES a thing OF THE KIND THIS CATEGORY
    IS ABOUT.

    Two gates, and both were needed.

    (1) It must name something at all. Without that, gap slots displaced named units and the
        supply of named material FELL across students instead of rising. Specificity is the
        number this pass exists to move, so a gap-closer that names nothing is strictly worse
        than the unit it would displace.

    (2) The name must resolve to an entity the college's own graph holds, of a type this
        category wants -- the SAME type list the anchor gate used. Without this second gate a
        gap slot could name a real thing of the wrong kind, and measured on a real profile it
        did exactly that: a Quirks section took two master's-degree programmes in construction
        management, and an External-Articles section took a ballot initiative and another
        university. Both are proper names; neither is a quirk or a press reference. Asking
        the anchor gate's question of the ANSWER as well as of the question closes it.

    A unit whose KIND is a course is nameable by construction -- it is one named course. But a
    course code found in loose prose is NOT enough on its own, and that is measured: the course
    pattern also matches a standards number and a catalogue heading, so untitled page chunks
    took 7 of 120 reserved slots on the strength of "IEEE 802" inside a reference list, a
    stray "CTXA 470" on a video-hosting page and a raw "Course Descriptions (200-299)" dump.
    None of them names a thing a student can go and find, which is the entire test.

    An entity name made entirely of the college's own name plus generic heads never counts
    either: that is the institution, not a thing a student can go and find.
    """
    if kind == "course":
        return True
    name = (entity_name or "").strip()
    if not name:
        return False
    stop_set = {s.lower() for s in stop}
    distinctive = False
    for word in _WORD_RE.findall(name.lower()):
        word = word.replace("'", "")
        if len(word) < 3 or word in stop_set or word in GENERIC_TERMS:
            continue
        distinctive = True
        break
    if not distinctive:
        return False
    if stats is None or not stats.has_graph or not node_types:
        return True
    return bool(stats.entity_types(name) & set(node_types))


def actionability_band(text: str, entity_name: str | None, stats: IndexStats | None) -> int:
    """0 = reads as something to do, 1 = reads as a bibliography entry.

    A sharply targeted gap query lands on the densest topical material, and for a research
    group that is its publication list. Those rows are specific and citable but read as a
    bibliography rather than "this lab takes undergraduates and here is who runs it". Used
    ONLY as a tie-break inside a narrow rank window: a hard preference was measured to lose
    the right entity altogether in favour of generic outreach prose.
    """
    if PUBLICATION_SHAPE.search(text or ""):
        return 1
    if stats is not None and stats.is_publication_entity(entity_name):
        return 1
    return 0


def reserved_candidates(
    gap_hits: Sequence[Sequence[tuple[int, float]]],
    already: set[int],
    entity_name_of: Callable[[int], str | None],
    text_of: Callable[[int], str],
    stop: Iterable[str],
    stats: IndexStats | None = None,
    node_types: Sequence[str] = (),
    kind_of: Callable[[int], str] | None = None,
) -> list[dict[str, Any]]:
    """Order the gap hits that may compete for a reserved slot.

    ORDER BY (gap query, BM25 rank), NOT by fused score, and this is the part that decides
    whether the pass works at all. A gap-closer is by definition found by exactly ONE query,
    which is precisely what rank fusion punishes: its fused mass is one term against the first
    pass's twelve. Ordering the reserved list by fused score hands the slots to rows the first
    pass had already half-found -- measured, it returned a plausible but wrong lab instead of
    the right one. Ordering by the gap query's own BM25 rank returns the right one.
    """
    window = max(1, int(CONFIG["actionability_window"]))
    out: list[dict[str, Any]] = []
    seen: set[int] = set(already)
    for query_index, hits in enumerate(gap_hits):
        for rank, (vec_row, score) in enumerate(apply_bm25_floor(hits)):
            if vec_row in seen:
                continue
            text = text_of(vec_row)
            name = entity_name_of(vec_row)
            kind = kind_of(vec_row) if kind_of is not None else ""
            if not is_nameable(name, text, stop, stats, node_types, kind):
                continue
            seen.add(vec_row)
            out.append(
                {
                    "vec_row": vec_row,
                    "gap_query": query_index,
                    "rank": rank,
                    "bm25": float(score),
                    "band": actionability_band(text, name, stats),
                }
            )
    out.sort(key=lambda c: (c["gap_query"], c["rank"] // window, c["band"], c["rank"]))
    return out


def missing_graph_note(category_code: str) -> str:
    """What to say out loud when an index ships no entity graph.

    It must be LOUD and it must disable the anchor gate, never quietly fall back to picking
    the rarest word on the resume: ungated rarity was measured choosing a supermarket job, a
    family party and a negation as research anchors. 45 colleges ship on this code and not all
    of their bundles will carry a graph.
    """
    return (
        f"  WARNING category {category_code}: this index has no entity graph (graph_nodes), so "
        f"gap-filling is OFF for it. Build the graph with wwrag/graph.py to enable it; falling "
        f"back to unGATED rarity is not done, because it picks nonsense anchors."
    )
