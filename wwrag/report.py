"""Build the final student-college fit report: report.md + report.html + report.pdf.

Takes verified ReportItems, the StudentProfile and the EvidenceUnits produced by the
rest of the wwrag pipeline and renders one document: a title block, a short "how to
read this" note, a fit summary, then the fixed ten categories in order. Every factual
sentence carries a small superscript number that links to a per-chapter numbered source
note with the full URL. A category with no surviving items says so honestly.

Nothing college-specific is hardcoded: brand colours, page budget and the category
vocabulary live in CONFIG at the top; the college name arrives by argument or from the
report file's own metadata.

Run:
  python wwrag/report.py \
      --report   out/report_items.json \
      --evidence out/evidence.json \
      --profile  out/profile.json \
      --out-dir  out/final
Options: --college-name, --categories FILE, --as-of YYYY-MM-DD, --strict,
         --allow-empty, --no-pdf, --chrome PATH.
Prints a JSON summary (page count and file sizes) and writes it to report_build.json.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# --------------------------------------------------------------------------------------
# CONFIG - everything tunable, nothing college-specific.
# --------------------------------------------------------------------------------------

CONFIG = {
    # WriteWise brand, deliberately NOT any college's colours.
    "brand_navy": "#1B2A4A",
    "brand_gold": "#B08D57",
    "ink": "#1d2433",
    "muted": "#5b6475",
    "line": "#e3e6ec",
    "link": "#1f4e9a",
    "wash": "#FBF9F5",
    "serif": '"Charter", "Georgia", serif',
    "sans": '"Avenir Next", "Helvetica Neue", Arial, sans-serif',
    "page_size": "Letter",
    "page_margin": "16mm 15mm 18mm 15mm",
    # Body type ladder. We may tighten to fit the page budget but never below the floor.
    "body_pt_ladder": [10.2, 9.9, 9.5],
    "min_body_pt": 9.5,
    "notes_pt": 8.6,          # source notes are notes, not body type
    # Page budget, [floor, ceiling]. This is a SANITY RANGE, not a content limit: nothing
    # is ever truncated to land inside it. The old ceiling of 14 was wishful - it was set
    # before the ten-chapter shape had per-chapter source notes, and the first real run
    # (36 items, 46 sources) came in at 17 pages, so the ladder ground the body type down
    # to the 9.5pt floor chasing a number it could never reach and still reported
    # page_budget_ok: false. Size it from the shape instead: ten chapters at up to four
    # items each is ~1.2pp of prose per chapter plus ~0.3pp of numbered sources, plus a
    # title page and a summary page -> ~17. Eighteen leaves one page of headroom, so a
    # full report renders once, at the most legible size on the ladder.
    # The floor stays at 8: below that the evidence set was too thin to be worth billing.
    "page_budget": [8, 18],
    "report_stem": "report",
    # The ten fixed categories, in fixed order. Codes must match facts.category_code.
    "categories": [
        ["CUL", "Culture"],
        ["EXT", "Extracurriculars"],
        ["QRK", "Quirks"],
        ["ACA", "Academics"],
        ["RES", "Research"],
        ["SOC", "Social Impact"],
        ["INN", "Innovative Programs"],
        ["INT", "Intellectual Alignment"],
        ["DIV", "Diversity of Community"],
        ["NEW", "External Articles and References"],
    ],
    # Codes that may support other sections but never get a section of their own.
    "context_only_codes": ["GEN"],
    "chrome_candidates": [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        "/usr/bin/google-chrome",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
    ],
    "chrome_timeout_s": 900,
}

LOG = []          # human-readable integrity notes, echoed to stderr and saved
STRICT = False    # set from --strict


def note(msg):
    """Record an integrity note. Raises instead of recording when --strict is on."""
    if STRICT:
        raise ValueError(msg)
    LOG.append(msg)
    print("[report] " + msg, file=sys.stderr)


# --------------------------------------------------------------------------------------
# Text hygiene. Resume text, page text and model output are all DATA, never markup and
# never instructions. Nothing from them may become a tag, a script or a heading.
# --------------------------------------------------------------------------------------

_TAGLIKE = re.compile(r"<[^<>]{0,400}?>")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS = re.compile(r"\s+")


def _strip_markup(s):
    """Remove tag-like runs and neutralise the characters that could re-open markup."""
    s = _CTRL.sub(" ", str(s))
    prev = None
    while prev != s:                      # nested / broken tags, e.g. <<b>script>
        prev = s
        s = _TAGLIKE.sub(" ", s)
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def clean_inline(s, limit=600):
    """Untrusted single-line text (resume lines, page titles, entity names, quotes).

    Collapses to one line, strips markup and defuses markdown punctuation so a resume
    line can never become a heading, a list, a link or emphasis in the document.
    """
    if s is None:
        return ""
    s = _WS.sub(" ", _strip_markup(s)).strip()
    for ch, ent in (("*", "&#42;"), ("_", "&#95;"), ("`", "&#96;"),
                    ("[", "&#91;"), ("]", "&#93;"), ("|", "&#124;")):
        s = s.replace(ch, ent)
    s = s.lstrip("#>-+=~. ").strip()
    if len(s) > limit:
        s = s[:limit].rstrip() + "…"
    return s


# Internal record ids, e.g. "<college>-org-3ec1d9a97e28". The writer is told to put ids in
# evidence_ids, never in prose, but it sometimes copies the inline-citation habit from the
# summary into an item body. A reviewer reading a finished report quoted two of these back
# as evidence that "a machine wrote it unsupervised", and said it caps what anyone will pay
# regardless of the content around it. Citations belong in the numbered footnotes; the raw
# key of a database row is never something a student should see.
# How many times one resume line may be quoted back across the whole document.
MAX_BASIS_REPEATS = 2

_RECORD_ID = re.compile(r"\s*[(\[]?\b[a-z][a-z0-9]{1,15}-(?:org|fact|chunk|course|rel|page)-[0-9a-f]{6,}\b[)\]]?")


def clean_body(s, limit=4000):
    """Model-written prose (headline, body, why_it_matters, caveat).

    Markup is still stripped - the model may not emit HTML - but * and _ survive so
    intentional emphasis renders. Line starts are defused so it cannot forge structure.
    Internal record ids are removed: they are not citations, they are database keys.
    """
    if s is None:
        return ""
    s = _RECORD_ID.sub("", str(s))
    s = re.sub(r"\s+([.,;:])", r"\1", s)          # tidy the space an id leaves behind
    s = _strip_markup(s)
    out = []
    for raw in s.split("\n"):
        ln = raw.strip()
        if ln:
            ln = re.sub(r"^([#>]+|[-*+]\s|\d+\.\s|={2,}|-{2,})", "", ln).strip()
        out.append(ln)
    s = "\n".join(out).strip()
    s = re.sub(r"\n{3,}", "\n\n", s)
    if len(s) > limit:
        s = s[:limit].rstrip() + "…"
    return s


def safe_url(u):
    """Return an http(s) URL safe to place in an href, or None."""
    if not u:
        return None
    u = _CTRL.sub("", str(u)).strip()
    u = u.split()[0] if u.split() else ""
    if not re.match(r"^https?://[^\s<>\"']+$", u, re.I):
        return None
    return u.replace("&", "&amp;")


# --------------------------------------------------------------------------------------
# Loading. Fail loudly on anything missing or malformed.
# --------------------------------------------------------------------------------------

def _read_json_or_jsonl(path, what):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError("%s file not found: %s" % (what, p))
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("%s file is empty: %s" % (what, p))
    if p.suffix.lower() == ".jsonl" or text.startswith("{\"") and "\n{" in text:
        rows = []
        for i, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError("%s: bad JSON on line %d of %s: %s" % (what, i, p, e))
        return rows
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError("%s: bad JSON in %s: %s" % (what, p, e))


def _pick_list(obj, keys, what, path):
    """Accept either a bare list or an object carrying the list under a known key."""
    if isinstance(obj, list):
        return obj, {}
    if isinstance(obj, dict):
        for k in keys:
            if isinstance(obj.get(k), list):
                meta = {kk: vv for kk, vv in obj.items() if kk != k}
                return obj[k], meta
    raise ValueError("%s: expected a list (or an object with one of %s) in %s"
                     % (what, ", ".join(keys), path))


def load_report(path):
    raw = _read_json_or_jsonl(path, "report")
    items, meta = _pick_list(raw, ["items", "report_items", "reportItems", "sections",
                                   "verified_items", "results"], "report", path)
    clean = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            note("report item %d is not an object; dropped" % i)
            continue
        code = str(it.get("category_code") or "").strip().upper()
        if not code:
            note("report item %d has no category_code; dropped" % i)
            continue
        clean.append(it)
    return clean, meta


def load_evidence(path):
    raw = _read_json_or_jsonl(path, "evidence")
    units, _ = _pick_list(raw, ["units", "evidence", "evidence_units", "results"],
                          "evidence", path)
    by_id = {}
    for u in units:
        if not isinstance(u, dict):
            continue
        uid = str(u.get("unit_id") or u.get("id") or "").strip()
        if not uid:
            note("evidence unit without unit_id skipped")
            continue
        by_id[uid] = u
    if not by_id:
        raise ValueError("evidence file %s produced no usable units" % path)
    return by_id


# profile.py documents first_name as `"" if absent` and flags it rather than failing:
# plenty of real resumes print a full name we will not split, or none at all. This module
# is the LAST stage of the pipeline, so demanding a name here turned a cosmetic gap into a
# total loss of the run *after* both paid model stages had already been billed. Require
# only the two fields we genuinely cannot render without; the name is optional and every
# place that uses it below has an unnamed form that still reads like a finished document.
REQUIRED_PROFILE_FIELDS = ("student_id", "level")


def load_profile(path):
    prof = _read_json_or_jsonl(path, "profile")
    if not isinstance(prof, dict):
        raise ValueError("profile file %s must hold a StudentProfile object" % path)
    for key in REQUIRED_PROFILE_FIELDS:
        if not prof.get(key):
            raise ValueError("profile is missing required field %r (%s)" % (key, path))
    if not str(prof.get("first_name") or "").strip():
        # Deliberately not note(): --strict turns integrity notes into hard failures, and
        # a resume that never printed a given name is a gap in the INPUT, not a defect in
        # our own work. Escalating it here would put the "dies at the last stage" bug
        # straight back for every strict run. Record it and carry on.
        msg = ("the profile carries no first name; the report is written without one "
               "rather than addressing the student as a placeholder")
        LOG.append(msg)
        print("[report] " + msg, file=sys.stderr)
    return prof


def student_name(profile):
    """The student's given name, cleaned, or "" when the resume never printed one.

    Callers must branch on the empty string instead of substituting a placeholder: a
    document titled "... for this student" reads like a mail merge that lost its
    variable, which is exactly what a paying family notices first.
    """
    return clean_inline(profile.get("first_name") or "", 60)


def profile_evidence_lines(profile):
    """Every verbatim resume line the profile is allowed to quote back."""
    lines = {}
    for bucket in ("activities", "projects", "achievements"):
        for row in profile.get(bucket) or []:
            if isinstance(row, dict):
                ln = row.get("evidence_line")
                if isinstance(ln, str) and ln.strip():
                    lines[_WS.sub(" ", ln).strip().casefold()] = ln.strip()
    return lines


# --------------------------------------------------------------------------------------
# Citations: unit -> per-chapter number -> source note with the full URL.
# --------------------------------------------------------------------------------------

def new_cites(slug):
    return {"slug": slug, "order": [], "by_key": {}, "by_unit": {}}


def _source_key(unit):
    return safe_url(unit.get("source_url")) or ("unit:" + str(unit.get("unit_id")))


def cite_number(cites, unit):
    """Number for this unit inside this chapter; units sharing a URL share a number."""
    uid = str(unit.get("unit_id"))
    if uid in cites["by_unit"]:
        return cites["by_unit"][uid]
    key = _source_key(unit)
    if key not in cites["by_key"]:
        cites["order"].append(unit)
        cites["by_key"][key] = len(cites["order"])
    n = cites["by_key"][key]
    cites["by_unit"][uid] = n
    return n


def sup(cites, numbers):
    if not numbers:
        return ""
    slug = cites["slug"]
    links = ",".join('<a href="#%s-s%d">%d</a>' % (slug, n, n) for n in numbers)
    return '<sup class="c">%s</sup>' % links


# The generator writes "... community projects [E:u-123], the Association ..." - a space
# before the marker. Once the marker becomes a raised number that space prints as a visible
# gap floating between the word and its citation ("projects  ¹, the"), which is the first
# thing that makes a typeset page look unfinished. Close it up at the seam.
_SPACE_BEFORE_SUP = re.compile(r"[ \t]+(?=<sup class=\"c\">)")

_DOUBLE = re.compile(r"\[\[\s*([^\[\]]{2,300}?)\s*\]\]")
_SINGLE = re.compile(r"\[\s*\^?(?:E:|EV:|ev:|e:)?\s*([A-Za-z0-9][A-Za-z0-9_\-.:]{2,120}"
                     r"(?:\s*[,;]\s*[A-Za-z0-9][A-Za-z0-9_\-.:]{2,120})*)\s*\]")


def resolve_inline(text, units, cites, used):
    """Turn [id], [[id]], [E:id], [id1, id2] markers into superscript citations.

    A marker whose tokens are not all known evidence ids is left alone - it is prose,
    not a citation.
    """
    def repl(m):
        tokens = [t.strip() for t in re.split(r"[,;]", m.group(1)) if t.strip()]
        if not tokens or not all(t in units for t in tokens):
            return m.group(0)
        nums = []
        for t in tokens:
            nums.append(cite_number(cites, units[t]))
            used.add(t)
        return sup(cites, sorted(set(nums)))

    text = _DOUBLE.sub(repl, text)
    return _SPACE_BEFORE_SUP.sub("", _SINGLE.sub(repl, text))


def source_note(unit, number, slug):
    """One numbered source line: title, host, kind, year, then the full URL."""
    title = clean_inline(unit.get("source_title") or unit.get("entity_name") or "Source", 180)
    url = safe_url(unit.get("source_url"))
    bits = []
    host = ""
    if url:
        host = re.sub(r"^https?://", "", url).split("/")[0]
    if host:
        bits.append(clean_inline(host, 80))
    kind = clean_inline(unit.get("source_kind") or "", 30)
    if kind:
        bits.append(kind)
    year = unit.get("year")
    if isinstance(year, int) and 1800 < year < 2200:
        bits.append(str(year))
    tail = " · ".join(bits)
    head = '<span class="anchor" id="%s-s%d"></span>**%s**' % (slug, number, title or "Source")
    if tail:
        head += " — " + tail
    if url:
        return "%s<br>\n<%s>" % (head, url)
    # No usable http(s) address. Never print the raw value: it may be a javascript:
    # or data: string planted in a crawled page. Say so instead, and log the defect.
    note("evidence unit %r has no usable http(s) source_url (%r); the source note says so"
         % (unit.get("unit_id"), str(unit.get("source_url"))[:60]))
    return "%s<br>\n*the source address on this record is not a usable web link*" % head


# --------------------------------------------------------------------------------------
# Markdown assembly.
# --------------------------------------------------------------------------------------

def heading_text(s):
    """Normalise a model-written headline into something that can sit on a heading line.

    The generator sometimes hands us a whole sentence ("Introduction to Electrical
    Engineering covers circuits, components, signals, and microelectronics."). A heading
    that ends in a full stop is the classic tell that prose was pasted into a title slot,
    and a student sees it on every page. Fold it to one line and drop a single trailing
    period, keeping "..." and initialisms like "U.S." intact - and keeping ? and !, which
    are legitimate in a heading.
    """
    s = _WS.sub(" ", str(s or "")).strip()
    if s.endswith(".") and not s.endswith("..") and not re.search(r"[A-Z]\.$", s):
        s = s[:-1].rstrip()
    return s


def _dedupe_keep_order(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _first_named_thing(text):
    """The first specific, nameable thing in some prose: a course code or a named entity."""
    if not text:
        return ""
    m = re.search(r"\b[A-Z]{2,5}\s?\d{3,5}[A-Za-z]?(?::\s*[^.]{3,60})?"
                  r"|\b\d{1,2}\.\d{2,4}\b", str(text))
    if m:
        return _WS.sub(" ", m.group(0)).strip(" :,.")
    m = re.search(
        r"\b(?:[A-Z][\w&'-]+\s+){1,5}"
        r"(?:Lab|Laboratory|Center|Centre|Institute|Program|Programme|Society|Club|Council|"
        r"Department|School|Group|Initiative|Fellowship|Scholarship|Association|Academy|Minor|Major)\b",
        str(text))
    return _WS.sub(" ", m.group(0)).strip() if m else ""


def render_item(item, units, cites, allowed_lines, stats, chapter_title=None):
    """One ReportItem as markdown, or None if nothing of it survives."""
    ev_ids = [str(e).strip() for e in (item.get("evidence_ids") or []) if str(e).strip()]
    ev_ids = _dedupe_keep_order(ev_ids)
    known = [e for e in ev_ids if e in units]
    for missing in [e for e in ev_ids if e not in units]:
        note("evidence id %r cited by a %s item is not in the evidence set; citation dropped"
             % (missing, item.get("category_code")))
        stats["dangling_evidence_ids"] += 1
    if not known:
        note("dropped a %s item (%r): no citable evidence"
             % (item.get("category_code"), (item.get("headline") or "")[:60]))
        stats["items_dropped_no_evidence"] += 1
        return None

    used = set()
    headline = heading_text(clean_body(item.get("headline") or "", 200))
    body = clean_body(item.get("body") or "")
    why = clean_body(item.get("why_it_matters") or "", 1500)
    do_next = clean_body(item.get("what_you_would_do") or "", 1200)
    caveat = clean_body(item.get("caveat") or "", 500).replace("\n", " ").strip()
    if not (body or why):
        note("dropped a %s item (%r): empty body" % (item.get("category_code"), headline[:60]))
        stats["items_dropped_empty"] += 1
        return None

    body = resolve_inline(body, units, cites, used)
    why = resolve_inline(why, units, cites, used)
    do_next = resolve_inline(do_next, units, cites, used)

    trailing = [cite_number(cites, units[e]) for e in known if e not in used]
    if trailing:
        marker = sup(cites, sorted(set(trailing)))
        if body:
            body = body.rstrip() + marker
        else:
            why = why.rstrip() + marker
    for e in known:
        used.add(e)
    stats["citations"] += len(used)

    out = []
    # A headline that just repeats its own chapter -- "### Academics" sitting inside
    # "## 04 · Academics" -- is a placeholder the writer fell back on, and a reviewer
    # counted three of them. It tells the reader nothing and looks unfinished, so it is
    # replaced by the first specific thing the item actually names.
    if chapter_title and headline and _WS.sub(" ", headline).strip().casefold() == \
            _WS.sub(" ", str(chapter_title)).strip().casefold():
        replacement = _first_named_thing(body) or _first_named_thing(headline)
        stats["headings_deduped"] = stats.get("headings_deduped", 0) + 1
        note("item headline repeated its chapter title (%r); using %r"
             % (headline, replacement or "no heading"))
        headline = replacement or ""
    if headline:
        # Source pages sometimes set a name in full capitals; the report should not shout at
        # the reader because a web page did. Only recase when it is really all-caps prose --
        # an acronym or a course code on its own is left alone.
        letters = [c for c in headline if c.isalpha()]
        if len(letters) > 12 and all(c.isupper() for c in letters) and " " in headline.strip():
            headline = " ".join(
                w if (len(w) <= 5 and w.isupper()) or any(ch.isdigit() for ch in w)
                else w.capitalize()
                for w in headline.split()
            )
            stats["headlines_recased"] = stats.get("headlines_recased", 0) + 1
        out.append("### " + headline)
    out.append("")
    if body:
        out.append(body)
        out.append("")
    if why:
        out.append("**Why this matters for you.** " + why.replace("\n\n", " "))
        out.append("")
    # The one line a family can act on: not why it suits them, but what to do about it.
    if do_next:
        out.append("**What you could do here.** " + do_next.replace("\n\n", " "))
        out.append("")

    basis = []
    for line in item.get("profile_basis") or []:
        if not isinstance(line, str) or not line.strip():
            continue
        key = _WS.sub(" ", line).strip().casefold()
        # The same resume line was being quoted back at the student up to thirteen times in
        # one document -- the air-quality project appeared under thirteen separate items. A
        # reviewer called it "the same five resume blocks pasted back at me around thirty
        # times", and read it as machinery rather than attention. Seen twice, a quote is
        # evidence; seen thirteen times it is wallpaper, so a line is shown at most
        # MAX_BASIS_REPEATS times across the whole report and the connection is simply
        # stated in prose after that.
        uses = stats.setdefault("basis_line_uses", {})
        if uses.get(key, 0) >= MAX_BASIS_REPEATS:
            stats["basis_repeats_suppressed"] = stats.get("basis_repeats_suppressed", 0) + 1
            continue
        if allowed_lines and key not in allowed_lines:
            note("profile_basis line not found verbatim in the profile; dropped: %r"
                 % line.strip()[:70])
            stats["profile_basis_dropped"] += 1
            continue
        uses[key] = uses.get(key, 0) + 1
        basis.append(clean_inline(line, 300))
    basis = _dedupe_keep_order([b for b in basis if b])
    if basis:
        out.append("**From your own file:**")
        out.append("")
        for b in basis:
            out.append("> “" + b + "”")
            out.append(">")
        out.pop()
        out.append("")
    if caveat:
        out.append("*Worth knowing: " + caveat + "*")
        out.append("")
    stats["items_rendered"] += 1
    return "\n".join(out)


def default_fit_summary(profile, counts, college, categories_with_items):
    """A summary built only from our own process and the student's own file.

    Used when the report file carries no fit_summary. It states nothing about the
    college that is not a count of our own evidence.
    """
    first = student_name(profile)
    n_cat = len(categories_with_items)
    total = len(CONFIG["categories"])
    fields = [clean_inline(f, 60) for f in (profile.get("intended_fields") or [])][:4]
    acts = len(profile.get("activities") or [])
    projs = len(profile.get("projects") or [])
    n_empty = total - n_cat
    n_points = counts["items_rendered"]
    second = ("%d of the %d categories %s enough verified evidence to say something "
              "useful here" % (n_cat, total, "carries" if n_cat == 1 else "carry"))
    if n_empty:
        second += ("; %d %s not, and %s marked as such rather than filled with guesswork"
                   % (n_empty, "does" if n_empty == 1 else "do",
                      "is" if n_empty == 1 else "are"))
    second += (". Across them %s %d checked %s, each one traceable to the source note at "
               "the end of its chapter."
               % ("sits" if n_points == 1 else "sit", n_points,
                  "point" if n_points == 1 else "points"))
    # No name is a supported state (profile.py leaves it ""), so the unnamed form is a
    # real second-person sentence rather than "this student’s file" - the rest of the
    # document already speaks to the reader as "you", so this is the consistent voice.
    opening = ("This report reads %s’s file against a verified evidence set for %s and "
               "reports back in ten fixed categories." % (first, college) if first else
               "This report reads your file against a verified evidence set for %s and "
               "reports back in ten fixed categories." % college)
    lines = [
        opening,
        "",
        second,
    ]
    if acts or projs:
        lines += ["", "The connections below rest on %d %s and %d %s listed in the file, "
                      "and on nothing that was not written there."
                  % (acts, "activity" if acts == 1 else "activities",
                     projs, "project" if projs == 1 else "projects")]
    if fields:
        lines += ["", "Intended fields, as stated in the file: " + ", ".join(fields) + "."]
    return "\n".join(lines)


def distinctive_elements(profile, limit=7):
    """The handful of things that make this student unusual, ranked by how much of their own
    document stands behind each. A reader -- and a counsellor writing the essay -- needs to see
    what the report thought the student was about before reading what it matched.

    The ranking is the same one the writer works to, so the section and the chapters cannot
    disagree. No model call: it is computed from the profile.
    """
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        from retrieve import profile_facets, theme_strength
        facets = profile_facets(profile)
        strength = theme_strength(facets)
    except Exception:
        return []
    if not strength:
        return []
    # one line per thing, strongest first, and never two lines about the same thing
    seen, rows = set(), []
    for value, score in sorted(strength.items(), key=lambda kv: (-kv[1], kv[0])):
        head = " ".join(str(value).split())
        key = head.lower()[:40]
        if key in seen:
            continue
        seen.add(key)
        rows.append((score, clean_body(head, 260)))
        if len(rows) >= limit:
            break
    if not rows:
        return []
    out = ["## What stands out in this file", "",
           "Ranked by how much of the document stands behind each, strongest first. "
           "The chapters below are weighted the same way.", ""]
    out += [f"{i}. {text}" for i, (_score, text) in enumerate(rows, start=1)]
    return out


def header_card(profile, college, as_of):
    """Who this is for, in four lines. Reads the profile; says nothing it cannot support.

    `declared_fields` is what the student or counsellor asked for; `intended_fields` is what the
    document alone suggests. Saying which is which matters -- a reader needs to know whether the
    report followed a stated choice or inferred one.
    """
    name = clean_body(str(profile.get("first_name") or "").strip(), 80)
    level = str(profile.get("level") or "undergraduate")
    declared = [str(f) for f in (profile.get("declared_fields") or []) if str(f).strip()]
    inferred = [str(f) for f in (profile.get("intended_fields") or []) if str(f).strip()]
    rows = []
    if name:
        rows.append(("Student", name))
    rows.append(("Applying to", college))
    rows.append(("Level", f"{level} applicant"))
    if declared:
        rows.append(("Areas chosen", ", ".join(declared)))
        if inferred and sorted(inferred) != sorted(declared):
            rows.append(("Suggested by the file", ", ".join(inferred)))
    elif inferred:
        rows.append(("Areas suggested by the file", ", ".join(inferred) + "  *(none stated)*"))
    else:
        rows.append(("Areas", "*none stated, and none the file could evidence*"))
    rows.append(("Evidence checked", as_of))
    return ["| | |", "|---|---|"] + [f"| **{k}** | {v} |" for k, v in rows]


# Kept, but at the END: a reader opening the report wants the student and the college, not a
# lecture on footnotes. The client asked for the top of page one back.
HOW_TO_READ = [
    "**How to read this.** Every factual sentence carries a small raised number. "
    "That number points to the numbered source note at the end of the same chapter, "
    "where you will find the full web address it came from. Nothing here is written "
    "from memory or from reputation.",
    "",
    "- The ten chapters are always the same ten, in the same order, so two reports can "
    "be laid side by side.",
    "- Everything described is open to undergraduate students. Graduate-only programmes "
    "are left out on purpose.",
    "- Where a chapter says there is not enough verified evidence, that is the honest "
    "finding, not an oversight. Nothing has been invented to fill the space.",
    "- Quoted lines under “From your own file” are copied word for word from the "
    "material you supplied.",
    "- This is a description of fit, not a prediction about admission.",
]


def render_summary(meta, units, cites, profile, college, rendered_codes, stats):
    """The fit summary, from the generator's own summary object when there is one.

    Accepts either a plain string or the object the generator emits
    ({"fit_summary": str, "strongest_matches": [str], "open_questions": [str]}).
    Falls back to a summary built only from our own counts and the student's file.
    """
    raw = meta.get("fit_summary")
    if raw is None:
        raw = meta.get("summary")
    matches, questions, ev_ids = [], [], []
    if isinstance(raw, dict):
        ev_ids = [str(e) for e in (raw.get("evidence_ids") or [])]
        matches = [x for x in (raw.get("strongest_matches") or []) if isinstance(x, str)]
        questions = [x for x in (raw.get("open_questions") or []) if isinstance(x, str)]
        raw = raw.get("fit_summary") or raw.get("body") or raw.get("text") or ""
    if not (isinstance(raw, str) and raw.strip()):
        return default_fit_summary(profile, stats, college, rendered_codes), "generated"

    used = set()
    out = [resolve_inline(clean_body(raw, 3000), units, cites, used)]
    if matches:
        out += ["", "**Strongest matches**", ""]
        out += ["- " + resolve_inline(clean_body(m, 600).replace(chr(10), " "),
                                      units, cites, used) for m in matches]
    if questions:
        out += ["", "**Worth asking on a visit or in an interview**", ""]
        out += ["- " + resolve_inline(clean_body(q, 600).replace(chr(10), " "),
                                      units, cites, used) for q in questions]
    # evidence_ids on the summary is the UNION of every surviving summary sentence's
    # sources - the opening paragraph's and the bullets' - because the verifier's rewriter
    # sometimes keeps a claim while dropping its inline [id] marker. Work out what is
    # genuinely still unattributed only AFTER the bullets have claimed their own ids.
    # Computing it first stapled every bullet's source onto the opening paragraph too, so
    # the reader saw the same number twice: once in a six-deep superscript pile at the end
    # of the paragraph and again on the bullet two inches below it.
    extra = [cite_number(cites, units[e]) for e in ev_ids if e in units and e not in used]
    if extra:
        out[0] = out[0].rstrip() + sup(cites, sorted(set(extra)))
    return "\n".join(out), "provided"


def build_markdown(report_items, report_meta, units, profile, college, categories,
                   as_of, stats):
    allowed_lines = profile_evidence_lines(profile)
    first = student_name(profile)
    level = clean_inline(profile.get("level") or "", 40)
    if level and level.lower() != "undergraduate":
        note("profile level is %r, not undergraduate; the report is written for "
             "undergraduate applicants" % level)

    by_code = {}
    for it in report_items:
        code = str(it.get("category_code") or "").strip().upper()
        by_code.setdefault(code, []).append(it)

    known_codes = {c for c, _ in categories}
    for code, rows in sorted(by_code.items()):
        if code in CONFIG["context_only_codes"]:
            note("%d item(s) with context-only code %s were not given a section"
                 % (len(rows), code))
            stats["items_dropped_context_only"] += len(rows)
        elif code not in known_codes:
            note("%d item(s) with unknown category_code %r were dropped" % (len(rows), code))
            stats["items_dropped_unknown_code"] += len(rows)

    md = []
    # Without a name the title is just the college: "<College> for this student" reads as
    # a failed mail merge, and the subtitle on the next line already says who it is for.
    md.append("# %s for %s" % (college, first) if first else "# %s" % college)
    md.append("")
    # A reader opening this wants to know, in one glance, who it is about and what was asked
    # for -- not a paragraph on how to read footnotes. The apparatus moved to the end.
    md += header_card(profile, college, as_of)
    md.append("")
    prof_summary = ""
    if isinstance(report_meta, dict):
        prof_summary = clean_body(
            str((report_meta.get("summary") or {}).get("profile_summary") or ""), 900)
    if prof_summary:
        md.append("**" + prof_summary.strip().rstrip(".") + ".**")
        md.append("")
    distinctive = distinctive_elements(profile)
    if distinctive:
        md += distinctive
        md.append("")

    # ---- fit summary -------------------------------------------------------------
    chapters = []   # (code, title, body_md, cites)
    rendered_codes = []
    body_chunks = {}
    # One thing, recommended once. A reader who meets "Bachelor of Science in Artificial
    # Intelligence" as a fresh discovery in two different chapters stops believing the
    # document was assembled by anyone. Categories are retrieved independently, so the same
    # strong match legitimately wins in two of them; the second one has to give way.
    seen_headlines: set[str] = set()
    for idx, (code, title) in enumerate(categories, 1):
        slug = "c%02d" % idx
        cites = new_cites(slug)
        pieces = []
        for it in by_code.get(code, []):
            key = _WS.sub(" ", str(it.get("headline") or "")).strip().casefold()
            if key and key in seen_headlines:
                stats["duplicate_items"] = stats.get("duplicate_items", 0) + 1
                note("item %r already appeared in an earlier chapter; dropped" % it.get("headline"))
                continue
            chunk = render_item(it, units, cites, allowed_lines, stats, title)
            if chunk:
                if key:
                    seen_headlines.add(key)
                pieces.append(chunk)
        if pieces:
            rendered_codes.append(code)
        body_chunks[code] = (slug, cites, pieces)

    summary_cites = new_cites("c00")
    summary_md, stats["fit_summary"] = render_summary(
        report_meta, units, summary_cites, profile, college, rendered_codes, stats)

    md.append("## Fit summary")
    md.append("")
    md.append(summary_md)
    md.append("")
    if summary_cites["order"]:
        md.append("#### Sources for this summary")
        md.append("")
        for n, unit in enumerate(summary_cites["order"], 1):
            md.append("%d. %s" % (n, source_note(unit, n, summary_cites["slug"])))
        md.append("")

    # ---- the ten chapters --------------------------------------------------------
    for idx, (code, title) in enumerate(categories, 1):
        slug, cites, pieces = body_chunks[code]
        md.append("## %02d · %s" % (idx, title))
        md.append("")
        if not pieces:
            md.append("*Not enough verified evidence in this dataset to report on "
                      "%s for this student. Nothing has been written here rather than "
                      "filling the space.*" % title.lower())
            md.append("")
            stats["categories_empty"].append(code)
            continue
        stats["categories_rendered"].append(code)
        md.extend(pieces)
        md.append("#### Sources for this chapter")
        md.append("")
        for n, unit in enumerate(cites["order"], 1):
            md.append("%d. %s" % (n, source_note(unit, n, slug)))
        md.append("")
        stats["sources"] += len(cites["order"])

    md.append("---")
    md.append("")
    md.append("### How to read this report")
    md.append("")
    md.append("\n".join(l if l else "" for l in HOW_TO_READ))
    md.append("")
    md.append("---")
    md.append("")
    md.append("*Prepared by WriteWise. Every numbered source above is a live web address "
              "that can be opened and checked. Where a chapter is empty, the evidence set "
              "did not support a claim worth making.*")
    md.append("")
    return "\n".join(md).replace("\n\n\n", "\n\n") + "\n"


# --------------------------------------------------------------------------------------
# Minimal markdown -> HTML. We author the markdown ourselves, so only the constructs we
# emit need supporting; this keeps the module on the standard library.
# --------------------------------------------------------------------------------------

_AUTOLINK = re.compile(r"<(https?://[^>\s]+)>")
_BOLD = re.compile(r"\*\*(.+?)\*\*", re.S)
_ITAL = re.compile(r"(?<![\w*])\*([^*\n]+?)\*(?!\w)")


def _inline(s):
    s = _AUTOLINK.sub(lambda m: '<a href="%s">%s</a>' % (m.group(1), m.group(1)), s)
    s = _BOLD.sub(r"<strong>\1</strong>", s)
    s = _ITAL.sub(r"<em>\1</em>", s)
    return s.replace("  \n", "<br>\n")


def _blocks(text):
    lines = text.split("\n")
    out, para, i, last_head = [], [], 0, ""

    def flush():
        if para:
            out.append("<p>" + _inline(" ".join(para).strip()) + "</p>")
            para.clear()

    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if not s:
            flush()
            i += 1
            continue
        m = re.match(r"^(#{1,4})\s+(.*)$", s)
        if m:
            flush()
            lvl, txt = len(m.group(1)), m.group(2).strip()
            last_head = re.sub(r"<[^>]+>", "", txt)
            out.append("<h%d>%s</h%d>" % (lvl, _inline(txt), lvl))
            i += 1
            continue
        if s == "---":
            flush()
            out.append("<hr>")
            i += 1
            continue
        if s.startswith("> "):
            flush()
            inner = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                inner.append(re.sub(r"^\s*>\s?", "", lines[i]))
                i += 1
            out.append("<blockquote>" + _blocks("\n".join(inner)) + "</blockquote>")
            continue
        if re.match(r"^\d+\.\s+", s):
            flush()
            items = []
            while i < len(lines):
                cur = lines[i].strip()
                if re.match(r"^\d+\.\s+", cur):
                    items.append([re.sub(r"^\d+\.\s+", "", cur)])
                    i += 1
                elif cur and items and not re.match(r"^(#{1,4}\s|[-*]\s|>\s)", cur) \
                        and cur != "---":
                    items[-1].append(cur)
                    i += 1
                else:
                    break
            cls = ' class="sources"' if last_head.lower().startswith("sources") else ""
            out.append("<ol%s>" % cls
                       + "".join("<li>" + _inline("\n".join(it).strip()) + "</li>"
                                 for it in items) + "</ol>")
            continue
        if re.match(r"^[-*]\s+", s):
            flush()
            items = []
            while i < len(lines) and re.match(r"^[-*]\s+", lines[i].strip()):
                items.append(re.sub(r"^[-*]\s+", "", lines[i].strip()))
                i += 1
            out.append("<ul>" + "".join("<li>" + _inline(x) + "</li>" for x in items) + "</ul>")
            continue
        para.append(s)
        i += 1
    flush()
    return "\n".join(out)


CSS_TEMPLATE = """
@page { size: __PAGESIZE__; margin: __MARGIN__; }
:root { --navy:__NAVY__; --gold:__GOLD__; --ink:__INK__; --muted:__MUTED__;
        --line:__LINE__; --link:__LINK__; --wash:__WASH__; }
* { box-sizing: border-box; }
body { font-family: __SERIF__; color: var(--ink); font-size: __BODYPT__pt;
       line-height: 1.46; margin: 0; }
h1 { font-family: __SANS__; font-size: 23pt; color: var(--navy); letter-spacing: -0.3px;
     border-bottom: 3px solid var(--gold); padding-bottom: 7px; margin: 0 0 8px; }
h1 + p em { color: var(--muted); font-size: 10.5pt; font-style: normal;
            letter-spacing: 0.2px; }
h2 { font-family: __SANS__; font-size: 15pt; color: var(--navy); margin: 24px 0 8px;
     padding-top: 8px; border-top: 2px solid var(--gold); break-after: avoid;
     break-before: auto; }
h3 { font-family: __SANS__; font-size: 11.4pt; color: #24304a; margin: 15px 0 5px;
     break-after: avoid; }
h4 { font-family: __SANS__; font-size: 9.4pt; color: var(--muted); margin: 14px 0 4px;
     text-transform: uppercase; letter-spacing: 0.8px; break-after: avoid; }
p { margin: 4px 0 8px; orphans: 2; widows: 2; }
ul { margin: 3px 0 9px; padding-left: 18px; }
li { margin: 2px 0; break-inside: avoid; }
a { color: var(--link); text-decoration: none; word-break: break-all; }
strong { color: #111827; }
em { color: #33405a; }
blockquote { margin: 9px 0 11px; padding: 7px 13px; border-left: 3px solid var(--gold);
             background: var(--wash); color: #33384a; break-inside: avoid; }
blockquote p { margin: 3px 0; }
blockquote ul { margin: 3px 0 2px; }
sup.c { font-family: __SANS__; font-size: 0.62em; line-height: 0; vertical-align: super;
        white-space: nowrap; margin-left: 1px; }
sup.c a { color: var(--gold); text-decoration: none; padding: 0 0.5px; }
ol.sources { font-size: __NOTESPT__pt; color: var(--muted); padding-left: 20px;
             margin: 2px 0 6px; }
ol.sources li { margin: 3px 0; break-inside: avoid; }
ol.sources strong { color: #3a4560; font-weight: 600; }
ol.sources a { color: var(--muted); }
.anchor { display: inline; }
hr { border: 0; border-top: 1px solid var(--line); margin: 18px 0 10px; }
hr + p em { color: var(--muted); font-size: 8.8pt; }
"""


def to_html(md_text, title, body_pt):
    css = CSS_TEMPLATE
    for token, key in (("__PAGESIZE__", "page_size"), ("__MARGIN__", "page_margin"),
                       ("__NAVY__", "brand_navy"), ("__GOLD__", "brand_gold"),
                       ("__INK__", "ink"), ("__MUTED__", "muted"), ("__LINE__", "line"),
                       ("__LINK__", "link"), ("__WASH__", "wash"),
                       ("__SERIF__", "serif"), ("__SANS__", "sans")):
        css = css.replace(token, str(CONFIG[key]))
    css = css.replace("__BODYPT__", "%.2f" % body_pt)
    css = css.replace("__NOTESPT__", "%.2f" % CONFIG["notes_pt"])
    body = _blocks(md_text)
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<title>%s</title><style>%s</style></head><body>%s</body></html>"
            % (clean_inline(title, 200), css, body))


# --------------------------------------------------------------------------------------
# PDF, the same way pipeline/render_pdf.py does it: headless Chrome via subprocess.
# --------------------------------------------------------------------------------------

def find_chrome(explicit=None):
    for cand in [explicit, os.environ.get("WWRAG_CHROME")] + CONFIG["chrome_candidates"]:
        if cand and Path(cand).exists():
            return cand
    raise FileNotFoundError(
        "no headless Chrome found. Pass --chrome /path/to/chrome, set WWRAG_CHROME, "
        "or run with --no-pdf. Looked in: " + ", ".join(CONFIG["chrome_candidates"]))


def chrome_pdf(chrome, html_path, pdf_path):
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--no-sandbox", "--print-to-pdf=%s" % pdf_path,
                    "--virtual-time-budget=20000", Path(html_path).resolve().as_uri()],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   timeout=CONFIG["chrome_timeout_s"])
    if not Path(pdf_path).exists():
        raise RuntimeError("Chrome produced no PDF at %s" % pdf_path)
    return Path(pdf_path)


def page_count(pdf_path):
    try:
        import pymupdf
    except ImportError:
        try:
            import fitz as pymupdf  # noqa: N813
        except ImportError:
            note("pymupdf not available; page count not measured")
            return None
    with pymupdf.open(str(pdf_path)) as doc:
        return doc.page_count


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def load_categories(path):
    if not path:
        return [(c, t) for c, t in CONFIG["categories"]]
    rows = _read_json_or_jsonl(path, "categories")
    if isinstance(rows, dict):
        rows = rows.get("categories") or []
    out = []
    for r in rows:
        if isinstance(r, dict):
            out.append((str(r["code"]).upper(), str(r["title"])))
        elif isinstance(r, (list, tuple)) and len(r) >= 2:
            out.append((str(r[0]).upper(), str(r[1])))
    if not out:
        raise ValueError("no categories found in %s" % path)
    return out


def resolve_college(arg, report_meta, profile):
    for src in (arg, report_meta.get("college"), report_meta.get("college_name"),
                (report_meta.get("meta") or {}).get("college")
                if isinstance(report_meta.get("meta"), dict) else None,
                profile.get("college"), profile.get("college_name")):
        if isinstance(src, str) and src.strip():
            return clean_inline(src, 120)
    cid = report_meta.get("college_id") or profile.get("college_id")
    if isinstance(cid, str) and cid.strip():
        return clean_inline(cid.strip().upper(), 120)
    raise ValueError("college name unknown: pass --college-name, or put 'college' or "
                     "'college_id' in the report file's metadata")


def main(argv=None):
    global STRICT
    ap = argparse.ArgumentParser(description="Render the final fit report (markdown + PDF).")
    ap.add_argument("--report", required=True, help="JSON of verified ReportItems")
    ap.add_argument("--evidence", required=True, help="JSON/JSONL of EvidenceUnits")
    ap.add_argument("--profile", required=True, help="StudentProfile JSON")
    ap.add_argument("--out-dir", required=True, help="directory for report.md/.html/.pdf")
    ap.add_argument("--college-name", default=None)
    ap.add_argument("--summary", default=None,
                    help="optional JSON holding the fit summary, for when the report "
                         "file carries items only (the verifier's report_verified.json). "
                         "Either the summary object itself or a file containing one.")
    ap.add_argument("--categories", default=None, help="optional category vocabulary JSON")
    ap.add_argument("--as-of", default=None, help="date stamp, YYYY-MM-DD (default: today)")
    ap.add_argument("--strict", action="store_true",
                    help="raise on any integrity problem instead of logging it")
    ap.add_argument("--allow-empty", action="store_true",
                    help="allow a report where no item survives")
    ap.add_argument("--no-pdf", action="store_true", help="markdown and HTML only")
    ap.add_argument("--chrome", default=None, help="path to a Chrome/Chromium binary")
    args = ap.parse_args(argv)

    STRICT = args.strict
    del LOG[:]

    report_items, report_meta = load_report(args.report)
    if args.summary:
        blob = _read_json_or_jsonl(args.summary, "summary")
        if isinstance(blob, dict):
            report_meta["fit_summary"] = blob.get("summary") or blob.get("fit_summary") or blob
        else:
            raise ValueError("--summary must hold a JSON object, not a list: %s" % args.summary)
    units = load_evidence(args.evidence)
    profile = load_profile(args.profile)
    categories = load_categories(args.categories)
    college = resolve_college(args.college_name, report_meta, profile)
    as_of = args.as_of or _dt.date.today().isoformat()
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", as_of):
        raise ValueError("--as-of must look like YYYY-MM-DD, got %r" % as_of)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {"items_in": len(report_items), "items_rendered": 0,
             "items_dropped_no_evidence": 0, "items_dropped_empty": 0,
             "items_dropped_unknown_code": 0, "items_dropped_context_only": 0,
             "dangling_evidence_ids": 0, "profile_basis_dropped": 0,
             "basis_line_uses": {}, "basis_repeats_suppressed": 0,
             "headings_deduped": 0,
             "citations": 0, "sources": 0,
             "categories_rendered": [], "categories_empty": [], "fit_summary": ""}

    md_text = build_markdown(report_items, report_meta, units, profile, college,
                             categories, as_of, stats)
    if stats["items_rendered"] == 0 and not args.allow_empty:
        raise ValueError("no report item survived rendering (%d came in). Refusing to "
                         "write an empty report; pass --allow-empty to override. Notes: %s"
                         % (stats["items_in"], " | ".join(LOG[-5:]) or "none"))

    stem = CONFIG["report_stem"]
    md_path = out_dir / (stem + ".md")
    html_path = out_dir / (stem + ".html")
    pdf_path = out_dir / (stem + ".pdf")
    md_path.write_text(md_text, encoding="utf-8")

    # Same rule as the H1: no name means no dangling "for " in the browser tab / PDF title.
    _first = student_name(profile)
    title = "%s for %s" % (college, _first) if _first else college
    body_pt = CONFIG["body_pt_ladder"][0]
    pages = None
    lo, hi = CONFIG["page_budget"]

    if args.no_pdf:
        html_path.write_text(to_html(md_text, title, body_pt), encoding="utf-8")
    else:
        chrome = find_chrome(args.chrome)
        for attempt, size in enumerate(CONFIG["body_pt_ladder"]):
            if size < CONFIG["min_body_pt"]:
                break
            body_pt = size
            html_path.write_text(to_html(md_text, title, body_pt), encoding="utf-8")
            chrome_pdf(chrome, html_path, pdf_path)
            pages = page_count(pdf_path)
            if pages is None or pages <= hi:
                break
            note("%d pages at %.2fpt exceeds the %d page budget; tightening"
                 % (pages, body_pt, hi))
        if pages is not None and pages > hi:
            note("FINAL: %d pages at the %.2fpt floor, over the %d page budget. Left "
                 "long on purpose - nothing was truncated." % (pages, body_pt, hi))
        if pages is not None and pages < lo:
            note("FINAL: %d pages, under the %d page minimum. Not padded." % (pages, lo))

    # Report the legibility cost as well as the page count. page_budget_ok alone hides the
    # difference between "fitted at full size" and "fitted only after shrinking the body
    # type towards the floor" - and shrinking type to make a number go green is the same
    # dishonesty as truncating, just harder to see. Not a note(): --strict escalates notes
    # into failures, and a long report is a legitimate outcome, not an integrity problem.
    tightened = body_pt < CONFIG["body_pt_ladder"][0]
    if tightened:
        msg = ("body type was tightened from %.2fpt to %.2fpt to fit the %d page budget; "
               "no content was removed" % (CONFIG["body_pt_ladder"][0], body_pt, hi))
        LOG.append(msg)
        print("[report] " + msg, file=sys.stderr)

    summary = {
        "college": college,
        "student_id": str(profile.get("student_id")),
        "as_of": as_of,
        "body_pt": body_pt,
        "body_pt_full": CONFIG["body_pt_ladder"][0],
        "body_pt_tightened": tightened,
        "pages": pages,
        "page_budget": CONFIG["page_budget"],
        "page_budget_ok": (pages is None) or (lo <= pages <= hi),
        # What the budget actually means, spelled out next to the verdict so nobody reads
        # page_budget_ok: false as "the report is broken" or as licence to cut content.
        "page_budget_policy": "advisory range; content is never truncated to fit it",
        "files": {p.name: (p.stat().st_size if p.exists() else None)
                  for p in (md_path, html_path, pdf_path)},
        "paths": {p.name: str(p.resolve()) for p in (md_path, html_path, pdf_path)
                  if p.exists()},
        "stats": stats,
        "notes": list(LOG),
    }
    (out_dir / "report_build.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
