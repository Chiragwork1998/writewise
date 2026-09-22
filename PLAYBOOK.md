# College Deep-Research Pipeline — Playbook

How to run the pipeline for a new college, what each step does, and what USC taught us. Everything lives in
`~/college-intel/`: code in `pipeline/`, one folder per college in `colleges/<slug>/`, API keys in `.env`.

## What it produces per college

| Output | File |
|---|---|
| One document per category (every verified fact, cited) | `output/categories/01_culture.md` … `10_external_articles_and_references.md` (+ `.pdf`) |
| General facts appendix, full course catalog | `00_general_facts.md`, `04b_course_catalog.md` |
| Digestible summary PDF across all 10 categories | `output/digest.pdf` |
| Crawl & extraction statistics, recall test, cost | `output/crawl_report.md` (+ `.pdf`) |
| Raw data for the product | `pages/*.json` (every scraped page), `extract/facts_raw.jsonl`, `extract/courses_*.jsonl`, `extract/organizations_tagged.jsonl` |

## The process (same for every college)

| # | Step | Command | Firecrawl credits | Notes |
|---|---|---|---|---|
| 1 | **Find every hostname** | `discover_hosts.py` | 0 | Certificate-transparency names + seeds; probes which answer. |
| 2 | **Harvest sites + read robots/sitemaps** | `inventory.py <college> <domain>` | 0 | Follows homepage links (wildcard certs hide many sites). Records crawl-delay and disallow rules. |
| 3 | **Firecrawl map on key sites** | `map_sites.py` | 1 per site | Returns URLs *with titles*; catches sites without sitemaps. |
| 4 | **Score + AI-rate candidates** | `select_urls.py` → `rate_urls.py` → `finalize_selection.py` | 0 | Keyword pre-filter, then DeepSeek rates title+URL 0–3 per category. ~$0.40 for 13k URLs. |
| 5 | **Scrape** | `run_queue.py <queues…>` | 1 per page | Host-aware dispatcher honours crawl-delay; results saved immediately (Firecrawl deletes after 24h). |
| 6 | **Structured sources** | `parse_classes.py`, `parse_engage.py` | 1 per page | Parsed deterministically — no AI, no hallucination risk. |
| 7 | **Faculty profiles** | `match_faculty.py` | 1 per profile | Instructors of undergraduate courses → their profile pages. |
| 8 | **Outside coverage** | `search_news.py` | 2 per 10 results + 1 per article | Use `site:` queries for independent outlets; generic queries return mostly the college's own pages. |
| 9 | **Extract facts** | `extract_facts.py` | 0 | DeepSeek; every fact carries a verbatim quote. Run in DeepSeek off-peak hours (half price). |
| 10 | **Verify + build documents** | `build_docs.py` | 0 | Re-checks every quote against the page; unverifiable facts never reach documents. |
| 11 | **Digest** | `build_digest.py` | 0 | Every bullet machine-checked: names/numbers must appear in the facts it cites. |
| 12 | **Report + PDFs** | `build_stats.py`, `render_pdf.py` | 0 | |

## Accuracy safeguards (why the output can be trusted)

1. **Verbatim evidence.** The extraction model must copy an exact quote for each fact; a script confirms the quote is on
   the page (98%+ pass; failures are dropped, not "fixed").
2. **Deterministic parsing where structure exists.** Course schedules and the student-organization directory are parsed
   with code, not AI.
3. **Citations everywhere.** Each bullet cites its source page and retrieval date; official vs. external is labelled.
4. **Digest guardrail.** A summary bullet is kept only if every capitalised name and number in it appears in the facts it
   cites.
5. **Recall test.** Known items (from earlier research or a human checklist) are searched for in the dataset, with proof text.

## Lessons from USC (apply to the next college)

- **Look for the one page that lists everything.** USC's student-organization directory (CampusGroups/EngageSC,
  `club_signup?view=all`) lists all 1,008 groups *with missions* on a single page — 1 credit instead of 1,000. Many colleges
  use CampusGroups or Campus Labs Engage; check the "view all" listing first.
- **Schedules hide data behind "Expand All".** Firecrawl `actions` (JavaScript click + wait) reveal sections, instructors,
  times, enrollment and syllabus links for 1 credit per department.
- **Discovery needs three methods.** Certificate logs missed most school sites; homepage link-harvesting and Firecrawl map
  filled the gaps.
- **Respect crawl-delay.** Some sites ask for 10–120 s between pages (USC's catalogue: 120 s). Pace them and get the same data
  from faster sources (the schedule of classes instead of the catalogue).
- **Budget the AI step, not just scraping.** Dense pages yield 30–130 facts; cap facts per page for profiles and news.
- **Deduplicate URL variants** (`usc.edu/x` vs `www.usc.edu/x/`) before spending credits.
- **Generic news search returns the college's own site.** Use `site:` queries for student papers and major outlets.

## Starting a new college

1. `mkdir -p colleges/<slug>/{discovery,pages,extract,output,logs,queues}`; put the CT-log hostnames in
   `discovery/subdomains.txt` (crt.sh) and known entry points in `discovery/seed_hosts.txt`.
2. Run steps 1–4, review the rating summary, then scrape (step 5) highest-value pages first.
3. Identify the college's structured sources (schedule of classes, org directory, catalog) and adapt/extend the parsers.
4. Run steps 7–12. Check the recall test and a random sample of 50 facts by hand before sharing.
