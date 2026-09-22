"""Per-college settings.

Each college keeps its own values in colleges/<slug>/config/college.json. Anything a college file leaves out falls back to
DEFAULTS below, which are generic and suit most US universities. Start a new college by copying
colleges/_template/config/college.json (see docs/runbook.html, "College settings file").

Fields marked (regex) are regular-expression fragments; everything else is plain text.
"""
import datetime
import json
import re
from pathlib import Path

YEAR = datetime.date.today().year

GENERIC_NEWS_DOMAINS = [
    "wikipedia.org", "nytimes.com", "washingtonpost.com", "wsj.com", "apnews.com", "reuters.com", "insidehighered.com",
    "chronicle.com", "usnews.com", "pbs.org", "npr.org", "govtech.com", "universitybusiness.com", "bloomberg.com",
    "forbes.com", "cnbc.com", "axios.com", "theatlantic.com"]

DEFAULTS = {
    "name": None,                     # full official name, e.g. "Stanford University"
    "short": None,                    # short name used in sentences, e.g. "Stanford"
    "root_domain": None,              # main domain, e.g. "stanford.edu"
    "current_year": YEAR,             # drives "recent" windows in URL scoring and rating

    # discovery (inventory.py)
    "affiliated_domains": [],         # the college's own sites on other domains: student paper, athletics, hospital
    "skip_host_patterns": [],         # (regex) extra first-labels of hostnames never worth crawling (portals, tools)

    # URL selection (select_urls.py, rate_urls.py, finalize_selection.py)
    "sitemap_skip_hosts": [],         # huge archive sites whose sitemaps would flood the candidate pool
    "url_keywords": None,             # (regex) full per-category keyword patterns; None = generic set + url_keywords_extra
    "url_keywords_extra": {},         # (regex) college words added to the generic set, e.g. {"culture": "tommy|traveler"}
    "selection_host_caps": {},        # host -> candidate cap before AI rating (x3 is applied); others 45
    "undergrad_hosts": [],            # host labels mainly serving undergraduates (score bonus), e.g. "admission", "www.cs"
    "queue_host_caps": {},            # host -> max pages in the scrape queue; others 60

    # fetching (run_queue.py)
    "fetch_host_caps": {},            # host -> parallel requests (only matters for hosts without crawl-delay)
    "expand_button_text": "Expand All",  # text of a button to click before capture (cloud Firecrawl, queue "expand": true)

    # structured sources, faculty, news
    "structured_url_patterns": [],    # (regex) pages parsed by code (schedule, club directory): never sent to the AI
    "news_hosts_capped": [],          # the college's own newsroom hosts: fewer facts per article
    "school_hosts": {},               # schedule school code -> that school's website host (faculty matching)
    "trusted_news_domains": GENERIC_NEWS_DOMAINS,  # outlets whose search results get scraped (search_news.py)

    # documents and digest (build_docs.py, build_digest.py, profile_titles.py)
    "news_domains": GENERIC_NEWS_DOMAINS,  # independent outlets: their facts always go to category 10
    "core_subdomains": ["today", "news", "undergrad", "admission", "admissions", "studentaffairs", "research"],
    "name_variants": [],              # lowercase names stripped when merging entity names, e.g. ["usc", "university of southern california"]
    "identity_words": [],             # words marking the college's own lore; Quirks keeps facts matching these
    "digest_good_subdomains": ["today", "news", "undergrad", "admission", "admissions", "studentaffairs", "research"],
    "digest_stop_words": [],          # capitalised words that are not names (college name parts, city)
    "org_directory_name": "student organization directory",
    "org_directory_url": None,
    "org_category_tags_extra": [],
    "schedule_name": "Schedule of Classes",
    "schedule_base_url": None,
    "schedule_terms": {},
    "search_sites": [],
    "schedule_index_url": None,       # page to cite for the course dataset; None = first course's source page
    "schedule_bucket": None,          # queue bucket of schedule pages (crawl report candidate count)
    "undergrad_course_first_digits": "1234",  # course numbers starting with these digits count as undergraduate

    # crawl report (build_stats.py)
    "recall_items": [],               # things a good researcher expects to find (names, labs, clubs, traditions)
    "recall_context": {},             # (regex) item -> pattern requiring context, for ambiguous words
    "recall_case_sensitive": [],      # items matched case-sensitively (acronyms that are also words)
    "sitemap_note": "",               # optional note after the sitemap URL count
    "known_gaps": [],                 # college-specific limits, one sentence each

    # PDFs (render_pdf.py)
    "pdf_accent": "#1f3a5f",
    "pdf_gold": "#b08d57",
}

GENERIC_URL_KEYWORDS = {
    "culture": r"mission|values|history|histor|tradition|heritage|spirit|marching-band|commencement|convocation|"
               r"welcome-week|homecoming|rivalry|our-story|identity|culture|founding|milestone|facts|at-a-glance|"
               r"who-we-are|legacy|centennial|anniversary",
    "extracurriculars": r"student-organi|student-group|clubs?\b|club-|rso|greek|fraternit|sororit|assembl|"
                        r"club-sports|intramural|recreation|ensemble|a-cappella|publication|student-government|"
                        r"get-involved|involvement|design-team|competition|society|chapter|leadership|student-life|"
                        r"campus-activities|student-media|radio|theatre-company|improv|debate|model-un|esports",
    "quirks": r"tradition|quirk|unusual|only-at|secret|legend|mascot|ritual|things-to-do|bucket|weird|escape|esports|"
              r"trivia|prank|hidden|myth|superstition|fun-fact|oddit|lore|flagpole|shrine|"
              r"game-day|gameday|tailgat|dog|pet|cat-cafe|food-truck|midnight|streak",
    "academics": r"major|minor|degree|curricul|course|general-education|honors|advis|requirement|"
                 r"bachelor|\bba\b|\bbs\b|\bbfa\b|b-a-|b-s-|progressive-degree|undergraduate|catalogue|catalog|teaching|"
                 r"faculty|professor|department|academic|program|study|learning|seminar|class",
    "research": r"research|\blabs?\b|lab-|center|centre|institute|undergraduate-research|"
                r"provost.*fellow|fellowship|directed-research|symposium|grant|scholars?-program|"
                r"laborator|initiative|innovation|discover|breakthrough|study-finds|scientists",
    "social_impact": r"communit|service|service-learning|volunteer|civic|engagement|nonprofit|non-profit|"
                     r"outreach|k-12|neighborhood|sustainab|equity|justice|pro-bono|clinic|health-equity|social-impact|"
                     r"social-good|homeless|literacy|mentor|impact|philanthrop|partnership|underserved|access",
    "innovative_programs": r"signature|interdisciplinar|design-your-own|accelerator|"
                           r"incubator|maker|venture|startup|global-experience|"
                           r"first-of-its-kind|new-program|launch|artificial-intelligence|\bai\b|quantum|immersive|"
                           r"distinctive|unique|pioneer|entrepreneur|studio|residency|exchange-program|dual-degree|"
                           r"joint-degree|pathway|bootcamp|intensive|living-lab|fellows-program",
    "intellectual_alignment": r"philosoph|approach|pedagog|liberal-arts|our-approach|why-|vision|strategic|"
                              r"academic-freedom|open-dialogue|critical-thinking|interdisciplinar|inquiry|ethos|"
                              r"principles|dean.?s-message|message-from|welcome-from|framework|manifesto|purpose|"
                              r"whole-person|well-being|wellbeing|humanities|free-expression|dialogue",
    "diversity_international": r"international|visa|f-1|j-1|opt\b|cultural-center|lgbt|queer|first-gen|"
                               r"first-generation|veteran|disabilit|accessib|belonging|diversit|inclusi|"
                               r"latin|black|asian|native|indigenous|religio|spiritual|interfaith|chabad|hillel|"
                               r"muslim|language-institute|global|multicultural|heritage-month|undocumented|"
                               r"transfer-student",
    "news": r"/news/|/stories/|/story/|/press|/spotlight|/features?/|/blog/|" + f"({YEAR - 2}|{YEAR - 1}|{YEAR})",
}


def load(college_dir):
    """Settings for one college folder: its config/college.json on top of DEFAULTS."""
    cfg = dict(DEFAULTS)
    p = Path(college_dir) / "config" / "college.json"
    if p.exists():
        cfg.update({k: v for k, v in json.loads(p.read_text()).items() if not k.startswith("_")})
    return cfg


def for_path(path):
    """Settings for whichever college folder contains `path` (walks up to find config/college.json)."""
    p = Path(path).resolve()
    for d in [p] + list(p.parents):
        if (d / "config" / "college.json").exists():
            return load(d)
    return dict(DEFAULTS)


def url_keywords(cfg):
    if cfg.get("url_keywords"):
        return dict(cfg["url_keywords"])
    kw = dict(GENERIC_URL_KEYWORDS)
    for cat, extra in (cfg.get("url_keywords_extra") or {}).items():
        kw[cat] = f"{kw[cat]}|{extra}" if cat in kw else extra
    return kw


def alt(items, escape=True):
    """Regex alternation from a list; a list that is empty matches nothing."""
    parts = [re.escape(x) if escape else x for x in items]
    return "|".join(parts) if parts else r"(?!x)x"


def root_or_default(cfg, argv_value=None):
    root = argv_value or cfg.get("root_domain")
    if not root:
        raise SystemExit("root domain unknown: pass it on the command line or set root_domain in config/college.json")
    return root
