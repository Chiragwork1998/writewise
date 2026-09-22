# WriteWise — where the project stands

Written for an engineer joining the pipeline. It assumes you can read Python and have never
seen this repo. Read this first, then `PROJECT_STATE.md` for the longer history.

## What the product is

A student uploads a CV. They get a cited report about how they fit one university, across ten
fixed chapters: Culture, Extracurriculars, Quirks, Academics, Research, Social Impact,
Innovative Programs, Intellectual Alignment, Diversity, External Articles.

USC is the proof of concept. **45 colleges** are promised, so nothing may be USC-specific.
A test enforces that: `test_no_college_specific_literals_in_the_module` fails the build if the
string "USC" appears anywhere in `wwrag/retrieve.py` outside the docstring.

## The pipeline, end to end

```
CV ──> profile ──> retrieve ──> generate ──> verify ──> useful ──> report
       2-5 min     40 sec       5-15 min     2 min      30 sec     1 sec
       $0.02       FREE         $0.20        $0.05      $0.01      free
```

Run it:
```bash
.venv-crawl4ai/bin/python wwrag/run.py \
  --resume "inbox/Some Student.pdf" --college usc \
  --out-dir wwrag/runs/myrun/student --index-root wwrag/index-v3 --skip-index \
  --fields "Economics, Gender Studies"      # optional: what the student says they want to study
```

Every stage writes its own file into the run directory, and `--from-stage <name>` resumes from
any of them. **Learn this flag first.** Generation is the expensive, fragile stage; resuming
from `generate` reuses the profile and the retrieval, which are already paid for.

### What each stage does

| Stage | File | What it does |
|---|---|---|
| profile | `wwrag/profile.py` | CV -> structured JSON. Every item must quote the CV **verbatim** or it is dropped |
| retrieve | `wwrag/retrieve.py` | Picks 24 evidence units per chapter out of 165,056. Dense vectors + keyword search, fused; then the **anchor pass** seats the person-level joins at the front (see below) |
| generate | `wwrag/generate.py` | Writes the ten chapters, one model call each |
| verify | `wwrag/verify.py` | Re-checks every sentence against its cited quote. Deletes what it cannot prove |
| useful | `wwrag/useful.py` | Deletes sentences that are true but say nothing |
| report | `wwrag/report.py` | Markdown -> HTML -> PDF |

## The anchor pass — the one thing a counsellor does first

Blending ~8 questions per chapter is what buries the match a counsellor makes first: *this
paper → that professor; this venture → that campus programme.* So after the chapters are
filled, `retrieve.anchor_pass()` takes the student's most substantive lines (`anchor_artefacts()`:
by length within facet weight, duplicates merged by embedding — never by how often a word
recurs) and runs each one **verbatim, unblended**:

- against rows that **name a thing** (org, course, or a fact with an entity; people and the
  college itself excluded) — "closest named thing at the college to this";
- for lines that are research by genre (paper / study / seminar…), against rows **about
  people** (grammar mask ∪ graph person/professor nodes), framed as
  `faculty whose research is on <declared fields>: <line>`. The subject words come from the
  client's declared major. Never type subject words into the code.

Seats are given round-robin per artefact (similarity is not comparable across artefacts),
faculty joins first, one per named thing, hit ≥ 8 words, ≤ 4 per chapter / 10 total, similarity
≥ 0.45 (measured for text-embedding-3-large: coincidences ≤ 0.43, real joins ≥ 0.46 — re-measure
if the embedder changes). Chapters do not grow: the weakest blended row yields. Every anchored
unit carries `anchor_for` = the student's own line, and `generate.py` renders the pair so the
writer can say what the student would *do* with it. All knobs are `anchor_*` in `CONFIG`; the
regression test is `test_anchor_pass_seats_the_person_level_join_with_its_artefact`.

Where it is weak: generic role lines (student council, club president) match role language,
not substance. Do not fix that by adding names or subjects — fix the artefact picker or the
frame, and measure on both students.

## The five things that will bite you

**1. Retrieval is free. Generation is not.** Retrieval uses OpenAI embeddings and costs
fractions of a cent (measured: `cost $0.0000`). So you can sweep twenty retrieval configs in
fifteen minutes for nothing. `/tmp/sweep.py` (copy it somewhere permanent) does exactly that.
Never tune retrieval by generating reports.

**2. Chapters checkpoint; use it.** Each chapter is written to
`<run>/chapters/<CODE>.json` the moment it exists, and a resumed run reuses them. This exists
because the provider drops long responses under load, and losing the tenth call used to throw
away the nine already paid for — twice in one morning, 780 seconds of billed work each time.

**3. A failed chapter no longer kills the run.** It is logged and skipped. A report that ships
with eight of ten chapters, and says so, beats no report.

**4. Anything you change in `profile.py` affects all ten chapters.** The extraction prompt was
tightened once to stop "AP Calculus" being read as a major. It also removed "Artificial
Intelligence" from a student whose paper, two internships and four skills were about AI — and
every AI research group vanished from his report. **Diff the extracted profile before and after
any prompt change.** This is the single most expensive mistake made on this project.

**5. The provider drops long responses.** `deepseek-v4-pro` is a reasoning model that burns
~80,000 thinking tokens to produce ~20,000 tokens of output. Under concurrency the longest
chapters get dropped mid-stream. Concurrency 8 with `items_max: 5` is the measured sweet spot;
`items_max: 8` made chapters too long to return reliably.

## What is measured, not guessed

Everything below came from an A/B, and the numbers are in the code comments next to each setting.

| Change | Effect | Where |
|---|---|---|
| `per_category` 24 -> 32 | **+27% named things** in the evidence. Biggest single lever | `run.py` CONFIG |
| Concurrency 3 -> 8 | 762s -> 276s, identical output and price | `run.py --concurrency` |
| Salience weighting at 0.5 | Slightly better, removes filler. At 1.0 it gets **worse** | `retrieve.py theme_weight_power` |
| Org floor 0.5 in Extracurriculars | Named clubs 5 -> 14 per student | `retrieve.py` EXT `kind_floor` |
| Reserved slots deeper than rank 1 | **Worse** (24 -> 23 on-target). Reverted | `retrieve.py query_slot_depth` |
| 3 ranked facet items instead of 2 | **No gain** (137 -> 136). Reverted | `retrieve.py` |
| Anchor pass (person-level join) | Papers -> the right faculty (Nix, Parreñas, Alyakoob, Tully, Lv, Lou); Greenbyte -> Sustainability Hub e-waste (+1 target); nothing lost | `retrieve.py anchor_*` |
| Anchor seats by global similarity | **Worse**: generic lines at 0.54 crowd out a paper's 0.45 faculty join. Replaced by round-robin per artefact | `retrieve.py anchor_pass` |
| ACA course floor 0.4 -> 0.25 | 0.4 dropped a declared-field degree fact; 0.25 loses nothing (confirmed twice) | `retrieve.py` ACA `kind_floor` |

Two of those are recorded reverts. Keep doing that — a change that did not help is worth
writing down so nobody tries it again.

## Defects found and fixed (do not reintroduce)

- **Applicant level.** Two of four real clients were classified as *graduate* applicants: one
  because he had won a golf tournament called the "Bengal Junior **Masters**", one because her
  school clubs carried date ranges longer than 18 months and were read as full-time work. Level
  switches off the undergraduate filter for the entire retrieval, so both got reports full of
  master's programmes. `wwrag/tests/test_profile_level.py`.
- **458 backwards facts.** "MKT 486 teaches Joseph Nunes" — the extractor reads a relation off
  one sentence and sentences often name the object first. Fixed with node-type signatures: a
  course cannot teach, a lab cannot direct. 56 edges whose direction is genuinely unknowable
  are dropped rather than guessed. `wwrag/tests/test_graph_direction.py`.
- **Skills extraction.** A student's skills came out as `Football, Golf, Rowing, Table-Tennis`
  while his only technical skill, `Java`, was dropped for having an evidence line under eight
  characters. Short evidence now matches on word boundaries instead.
- **The "names a findable thing" gate was deleting good items** — `TAC-449` (hyphen), `Bachelor
  of Science in AI` (no degree noun), `Office of Undergraduate Programs` (plural).

## Known limits, honestly

- **Club pages are shallow.** 1,309 crawled, but 93% of `/events/` pages say "There are no
  upcoming events" and 98% of `/news/` are empty. The real depth is on the 364 clubs that link
  to their own external sites — not yet crawled.
- **Reports score 3-4/10** with independent reviewers. Accurate, cited, and thinner than a
  human researcher's brief. The gap is depth, not truth.
- **The verifier may be deleting the payload.** A sentence like "she would take her e-waste data
  to the Sustainability Hub's monthly drive" is a college fact x a profile fact x a *proposed
  action*. Only the first is groundable. Verification should be typed; currently it is not.
- `wwrag/tests/test_gapfill.py` is red and excluded — its fixture no longer reproduces a gap
  because first-pass retrieval improved.
- A homoglyph injection (Cyrillic о in "ignоre") still evades the profile quarantine.

## Running the tests

```bash
.venv-crawl4ai/bin/python -m pytest wwrag/tests/ -q --ignore=wwrag/tests/test_gapfill.py
```
185 pass. Each one is named for a defect that actually shipped. If you break one, read the
docstring before changing the test — several of them exist because a previous fix was a
regression in disguise.

## Before a new college ships

```bash
.venv-crawl4ai/bin/python pipeline/coverage_check.py colleges/<slug> wwrag/index-v3/<slug>
```
Reports units, distinct hosts and distinct entities for all ten chapters, plus graph size and
recall against the config's `recall_items`. Thresholds come from USC's weakest chapter. Exit 1
means do not ship.

Per-college settings live in `colleges/<slug>/config/college.json`; the template with every
field documented is `colleges/_template/config/college.json`.
