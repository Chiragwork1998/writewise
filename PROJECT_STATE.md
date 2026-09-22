# College Intel — project state

_Last updated: 21 September 2026. Written so any session, or any engineer, can pick this up cold._

## 1. What exists today

| Thing | Where | State |
|---|---|---|
| USC dataset | `colleges/usc/` | 78,658 verified facts, 5,128 pages, 1,002 clubs, 4,835 courses, graph of 29,934 nodes |
| Category documents + digest (PDF) | `colleges/usc/output/` | Current |
| Student guide, 58 pages | `colleges/usc/guide/usc_student_guide.pdf` | Current; red branding, "Prepared by WriteWise" |
| Client coverage report | `colleges/usc/output/usc_coverage_report.pdf` | **Stale**: quotes 69,990 facts / 4,570 pages |
| RAG bundle for the engineer | `colleges/usc/export/usc_rag_v2.zip` (39.6 MB) | Current; includes `page_chunks.jsonl` |
| Junior's runbook | `docs/runbook.html`, published at https://claude.ai/artifact/UYz2GLMqn5eyxZPvKz8Qv4 | Current |
| Per-college settings | `colleges/<slug>/config/college.json`, loader `pipeline/college_config.py`, template `colleges/_template`, `pipeline/new_college.sh` | USC's outputs verified byte-identical after the refactor |

## 2. Decisions made (and why)

| Decision | Reason |
|---|---|
| **Crawl4AI** fetches all pages | Tested: keeps 95% of cloud's fact quotes, 4× faster, free. Blind AI test matched cloud within normal model randomness |
| **Self-hosted Firecrawl: rejected** | Slower, needs 5–6 GB + Docker, site maps return no page titles, 0/41 bot-protected pages |
| **Firecrawl cloud kept for two jobs only** | Site maps with titles (~73 calls) and news search (~52 queries) = ~200 credits ≈ $1 per college |
| **Extraction by an open model on a rented GPU** | ~$1 per college versus $8–10 on DeepSeek. Model: Qwen3.8-27B (LiveBench: instruction-following 72.7, beats DeepSeek/Kimi/GLM/Opus 5 on that axis) |
| **No always-on GPU** | $500–2,000/month. Rent per batch; crossover for a dedicated GPU is ~20,000 students/month |
| **Supabase Pro for the database** | Engineer already built the schema; includes auth, storage, backups. Self-hosting saves $25 but costs more in his time |
| **Vercel Pro for the web app** | $20/month; free tier forbids commercial use. Crawling and extraction stay off Vercel (time limits) |
| **Server: Hetzner CX33 (~$10, EU) or Netcup (~$11, US)** | OVH cheaper but weaker reputation; Hetzner's US boxes cost 4× its EU ones. Not needed during the build month — the Mac does it |
| **Multiple free Firecrawl accounts: refused** | Breaks their terms; a ban mid-build risks the client deadline to save $38 |
| **Retrieval design** | Search scoped to one college, binary-quantized index + re-check, hybrid keyword + vector, rerank. Keeps 4M vectors inside a 16 GB box |

## 3. Costs

**Build month (45 colleges)**
| Item | Cost |
|---|---:|
| Firecrawl Hobby × 2 months | $38 |
| GPU for all 45 colleges (~45 h at $0.73–1.19/h) | $35–55 |
| Server | $0 (runs on the Mac) |
| DeepSeek | $0 |
| **Total** | **~$75–95** |

**Per college afterwards:** ~$3 machine cost + 2–4 human hours (under 1 hour once platform detectors exist).
**Per student:** $0.05–0.15, almost all of it story writing.
**Client's monthly run cost:** ~$60–90 (server $11–25, Supabase $25–40, Vercel $20, backups $1–3).

## 4. The month plan

| Week | Work |
|---|---|
| 1 | Platform detectors (CampusGroups, Coursedog and similar) — the thing that removes the human bottleneck. GPU model test. Server setup. Pick and rank the 45 colleges |
| 2 | Build colleges 1–15; make the process repeatable |
| 3 | Colleges 16–35, 4–6 in parallel overnight |
| 4 | Colleges 36–45, quality checks, handover with the junior doing it live |

**Needed from Chirag:** ranked list of 45 colleges; $19 Firecrawl; $10 RunPod credit; junior available from week 2; decision on how deep the quality checks go (full checks on all 45 add 45–90 hours).

**Stated risk:** 45 in four weeks holds only if the detectors cover most colleges. Otherwise realistic output is 25–30, and the client should hear that number now rather than later.

## 5. Open items

**Next actions**
1. Patch every AI step (`rate_urls`, `extract_facts`, `rate_orgs`, `build_digest`) to accept any OpenAI-compatible endpoint, not just DeepSeek.
2. Run the 200-page model test (~$3). Pass mark: ≥90% of DeepSeek's verified facts, ≥97% quote pass rate.
3. Build the platform detectors.
4. Refresh the client coverage report with current numbers.

**Before launch (flagged, not built)**
- Consent handling for students under 18 (India's DPDP and equivalents).
- Spend caps and per-user limits on new-college requests and story generation.
- A crawler user-agent naming the product with a contact URL.
- A tested backup restore.
- Uptime and failure alerts.
- The retrieval quality test: 50 queries + 20 resumes, recall targets.
- No automated tests on the pipeline (~35 scripts).

## 6. Ground rules held throughout
- Every fact carries a verbatim quote, checked against the page; failures are discarded, never repaired.
- Structured sources (course schedules, club directories) are parsed by code, never by AI.
- robots.txt and crawl-delays are obeyed; bot protection is never bypassed — blocked pages are listed as known gaps.
- Nothing reaches the client before the quality checks pass.
- The data bundles are the source of truth; databases are rebuildable copies.

---

## 7. Night of 22 September 2026 — audit of the four client reports

Chirag asked for a neutral check that the four reports (Aadya Aggarwal, Aadya Saha, Aashrut
Almal, Aditya Khaitan) are actually built on the right material, and that the mechanism is
structurally sound enough to repeat for 45 colleges. It was not sound. Four defects were found
by measurement, three of them affecting reports already generated.

### 7.1 Applicant level was wrong for two of the four students

`level` decides whether graduate-only material is filtered out of every category, and it is set
before retrieval runs. Two of the four were classified `graduate`:

- **Aditya Khaitan** — the detector matched the word "Masters" inside **"Bengal Junior Masters"**,
  a golf tournament he had won, and again in a second tournament name. Two lines, so the
  two-corroborating-lines safeguard passed it.
- **Aadya Aggarwal** — her school clubs carry date ranges ("President: CIS Math Honour Society,
  June 2024 – Current"; "The Third Eye, Aug 2023 – Current"), and the detector read 27 and 37
  months of *full-time work history*.

Both are school students applying as undergraduates. With the level wrong, the undergraduate
pre-filter was off for their whole retrieval, which is why Aditya's Academics chapter filled up
with Progressive Degree Program mechanics, M.S. in Communication Data Science, MSECE and MSQIS
advising, and a course whose own description reads "For graduate students".

Fixed in `wwrag/profile.py`:
- `GRADUATE_DEGREE_RE` no longer accepts the bare word "Masters". "Master's" with the apostrophe
  and "Master of &lt;degree field&gt;" stand alone; the bare word needs a degree word beside it.
  Tournaments, "Scrum Master" and "Master of Ceremonies" no longer count.
- `SCHOOL_STAGE_RE` / `SCHOOL_BOARD_RE` / `STUDENT_ACTIVITY_RE`: a dated range is not employment
  when the document shows a school stage (ICSE, CBSE, IB, A-levels, Grade 11/12, high school) or
  when the line itself is a club, society, council, volunteer or mentorship post.

Verified: all four students now read `undergraduate`; a real graduate résumé (Chirag's own) still
reads `graduate`. Tests in `wwrag/tests/test_profile_level.py`.

### 7.2 Relations were stated backwards

The extractor reads a relation off one sentence, and a sentence often names the object first
("MKT 486, led by Professor Joseph Nunes"), so the edge went in the wrong direction and the
report stated it as fact: *"MKT 486 teaches Joseph Nunes."* Aditya's report carried *"Academic
Workshops hosts Viterbi Undergraduate Advising."*

Text parsing was tried first and rejected: at ~50% precision a swap-detector would have broken as
many edges as it fixed. Node types settle it without reading the sentence — a course cannot teach,
a lab cannot direct, an event cannot host. An edge is turned round only when its stated subject
**cannot** act and its object **can**. Where direction is genuinely unknowable (an event said to
host a person — a podcast guest is not the host) the edge is dropped rather than guessed.

Result: **458 edges turned round, 56 dropped**, 2% of the graph. `wwrag/graph.py`, tests in
`wwrag/tests/test_graph_direction.py`.

### 7.3 A widened regex crashed every run

`COURSE_CODE` was widened to cover dotted numbering ("18.06") for other colleges, which gave it
four capture groups; two callers still unpacked two. Every run died in `generate`. `int("102L")`
would have crashed at the same line regardless. Fixed with `course_codes()` and `course_level()`;
tests in `wwrag/tests/test_generate.py`.

### 7.4 Aditya's accountancy gap was mine, not the corpus's

Last night this was recorded as a coverage gap. It is not: the index holds **91 undergraduate
business courses**, including ACCT 370/410/416/451/456, BUAD 280/281 and FBE 421 — 149 units
across the seven most relevant alone. The material was there; the wrong applicant level was
keeping graduate content in front of it.

## 8. Modularity work for the other 44 colleges

- `parse_engage.py`, `parse_classes.py` and `map_search.py` no longer hardcode USC URLs, term
  codes or the 28-host search list. New settings: `org_directory_url`, `org_category_tags_extra`,
  `schedule_base_url`, `schedule_terms`, `search_sites`. `map_search.py` falls back to deriving
  hosts from `undergrad_hosts` + `root_domain` when `search_sites` is unset. Both parsers were
  re-run against USC and produce byte-identical output.
- `hub_degree` was the absolute count 120, hand-tuned on USC; on a college with a tenth of the
  corpus it would never fire and every chapter would fill with school-level noise. It is now
  derived from the graph's own 99th percentile (USC: 104, selecting the same 18 nodes as 120 did).
  Blocking every node typed "school" was tried and **rejected by measurement** — it would have
  silenced USC Leventhal School of Accounting, which anchors 15 units of real student evidence.
- **`pipeline/coverage_check.py`** — the per-college go/no-go. Reports units, distinct hosts and
  distinct entities for each of the ten categories, plus graph size and recall of the config's
  `recall_items`. Thresholds are read off USC's weakest category (QRK: 465/64/311) and set below
  it, so the gate catches a failed crawl rather than demanding every college match USC.
  USC: READY, 98% recall. `python pipeline/coverage_check.py colleges/<slug> wwrag/index-v3/<slug>`

### 8.1 Enrolment machinery was crowding out the subjects

With the level fixed, Aditya's Academics chapter was still 45% *process*: six units on the
Progressive Degree Program, change-of-major advising, "the best source is the course catalogue",
"the page lists an Undergraduate section". Every one of the four students showed the same thing —
**25% to 59% of Academics slots**, in both runs. It scores well because it uses the category's own
words (degree, course, requirement, undergraduate) while naming nothing a student can choose.

Every existing diversity cap was defeated: the six copies of the one programme came from **six
hostnames under six entity names**, so `max_per_host` and `max_per_entity` each saw a single unit.
Any large college republishes its university-wide machinery on every school's site, so the cap had
to be on *what the unit is*, not where it came from.

Two changes in `wwrag/retrieve.py`: a `PROCESS_BOILERPLATE` pattern with
`process_boilerplate_weight: 0.70`, and `max_process_boilerplate: 2` per category.

Measured, re-running retrieval for three students (retrieval costs nothing — it uses OpenAI
embeddings only, no DeepSeek):

| | boilerplate before | after weight | after weight + cap |
|---|---|---|---|
| Aditya — Academics | 13/29 (45%) | 8/27 (30%) | **3/28 (11%)** |
| Aadya Aggarwal — Academics | 9/30 (30%) | 5/30 (17%) | **3/30 (10%)** |
| Aashrut — Academics | 9/63 (14%) | 6/40 (15%) | **3/42 (7%)** |

No category anywhere got worse. Aditya's Academics went from 6 to 10 units of
business/accountancy material — the Leventhal accounting and finance major requirements with
ACCT 370, MATH 117g and MATH 118 (Business and Economics), the undergraduate business emphases.
Test: `test_enrolment_machinery_is_capped_per_category` in `test_retrieve_regressions.py`.

### 8.2 Passage overlap was cut mid-word — fixed forward, not backfilled

The carry-over between page passages was a raw character slice (`buf[-350:]`), so **13,267 of
USC's 26,960 passages (49%) begin mid-word** — "lifornia President Michael V. Drake said" for
California. `pipeline/export_rag.py:tail_overlap()` now moves the cut forward to a line or word
boundary; tests in `wwrag/tests/test_chunking.py`.

**Deliberately not backfilled.** Nine of the ten passages cited across the four existing reports
start mid-word, and the prose that came out of all nine is coherent and correct: the writer reads
past the fragment and the verifier catches anything ungrounded. So the harm is a wasted first
token and a slightly degraded embedding, not a wrong report. Re-chunking means re-embedding all
26,960 passages — about **$0.70 of OpenAI credit** — and rebuilding `vectors.npy`. That is worth
doing before the next colleges are indexed, and not worth risking the one working index
overnight with nobody awake to check it.

### 8.3 Retrieval is free, so retrieval quality can be iterated without credit

`wwrag/retrieve.py` makes **no DeepSeek calls** — it embeds queries with OpenAI
`text-embedding-3-large` and costs fractions of a cent per student (measured: `cost $0.0000`).
Only `profile`, `generate`, `verify` and `useful` spend DeepSeek credit. So evidence quality can
be measured and tuned end to end while the DeepSeek balance is empty; only turning evidence into
a written report needs topping up.

## 9. Where the four reports stand

**None of the four is regenerated.** DeepSeek credit ran out mid-run: the balance is **$0.15**
and the API now closes connections part-way through a call. Aashrut's run got through retrieval
and died in `generate` on the SOC category with "peer closed connection without sending complete
message body".

The reports the client would receive today are the ones in `wwrag/runs/final4/`, and **Aditya's
and Aadya Aggarwal's are written for a graduate applicant** (§7.1). They should not go out as
they stand.

About **$1.50 of credit** finishes all four. Everything else is already on disk — corrected
profiles, corrected evidence, corrected graph — so run:

```
./finish_reports.sh
```

It resumes each student from the first stage that still needs paying for (Aashrut from
`generate`, Aditya and Aggarwal from `retrieve`, Saha from `profile`) and prints the balance
before each one.

Test suite: 166 passing (`test_gapfill.py` still excluded — its fixture no longer reproduces a
gap, because first-pass retrieval improved).
