"""Round-2 URL selection: score every known URL (Firecrawl map + sitemaps) against the 10
research categories and pick the most valuable pages within per-category quotas.

usage: python pipeline/select_urls.py colleges/usc out_queue.jsonl
"""
import collections
import glob
import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

sys.path.insert(0, str(Path(__file__).parent))
from firecrawl_client import normalize, url_id  # noqa: E402

college = Path(sys.argv[1])
out_path = Path(sys.argv[2])
from college_config import alt, load as load_config, root_or_default, url_keywords  # noqa: E402
CFG = load_config(college)
ROOT = root_or_default(CFG)
Y = CFG["current_year"]
disc = college / "discovery"

# ---------------------------------------------------------------- candidate pool
pool = {}


def add(url, title="", desc="", lastmod=None, src=""):
    try:
        u = normalize(url)
    except Exception:
        return
    p = urlparse(u)
    if not p.hostname or p.scheme not in ("https",):
        return
    rec = pool.setdefault(u, {"url": u, "title": "", "desc": "", "lastmod": None, "src": set()})
    rec["title"] = rec["title"] or (title or "")
    rec["desc"] = rec["desc"] or (desc or "")
    rec["lastmod"] = rec["lastmod"] or lastmod
    rec["src"].add(src)


for f in glob.glob(str(disc / "maps" / "*.json")):
    for l in json.load(open(f)).get("links", []):
        add(l.get("url", ""), l.get("title"), l.get("description"), src="map")
for f in glob.glob(str(disc / "sitemaps" / "*.json")):
    site = Path(f).stem
    if site in CFG["sitemap_skip_hosts"]:
        continue
    for u, lm in json.load(open(f))["sitemap_urls"]:
        add(u, lastmod=lm, src="sitemap")

fetched = {json.loads(l)["url"] for l in (college / "logs" / "fetch_log.jsonl").read_text().splitlines()
           if json.loads(l)["status"] == "ok"}
print("pool", len(pool), "already fetched", len(fetched))

# ---------------------------------------------------------------- exclusions
EXCLUDE = re.compile(
    r"(/tags?/|/categor(y|ies)/|/author/|/page/\d+|/feed/?$|/wp-json|/wp-admin|/wp-login|/search\b|[?&]s=|/login|"
    r"sign-?in|/cart|/checkout|/donat|/giving|/give/|/make-a-gift|/job-bank|/jobs?/|/job-|/employment-opportunit|"
    r"/careers?-at-|/calendar|/events?/|/event-|/webinar|/print/|/share|/attachment/|/wp-content/|"
    r"\.(jpe?g|png|gif|svg|webp|mp4|mp3|mov|zip|docx?|xlsx?|pptx?|ics|xml|css|js)(\?|$)|mediacoverage|"
    r"media-coverage|/in-the-news|/concert_program|/newsletter|/payroll|/purchasing|/procurement|/forms?/|"
    r"/intranet|/portal|covid|coronavirus|/rfp|/vendor|/parking|/privacy|/terms-of|/accessibility-statement|"
    r"/cookie|/sitemap|/feedback|/contact-?us|/directions|/maps?/|/visit/|/hours|/staff-directory|/our-staff|"
    r"/alumni/(?!.*(tradition|history))|/obituar|/in-memoriam|/(zh|ar|br|es|ko|ja|vi|fr|de)/|/lang/|"
    r"/modules/|/test-|/demo|/draft|/preview|/node/\d+|/user/|/admin/|\?replytocom|/amp/?$|/embed/?$|"
    r"/conferences?/\d{4}|/cle-|/continuing-legal|/executive-education|/exec-ed|/registration|/apply-now|"
    r"/request-info|/rfi|/thank-you|/confirmation|/unsubscribe|/subscribe)", re.I)
LOW_HOSTS = CFG["selection_host_caps"]
DEFAULT_HOST_CAP = 45

# ---------------------------------------------------------------- category keywords
# ---------------------------------------------------------------- category keywords (config/college.json)
CATS = url_keywords(CFG)
CAT_RX = {k: re.compile(v, re.I) for k, v in CATS.items()}
QUOTA = {"academics": 700, "research": 650, "culture": 220, "extracurriculars": 260, "quirks": 120,
         "social_impact": 380, "innovative_programs": 330, "intellectual_alignment": 160,
         "diversity_international": 300, "news": 560}

UNDERGRAD_HOSTS = re.compile(r"^(" + alt([f"www.{ROOT}", ROOT] + CFG["undergrad_hosts"]) + r")\.", re.I)
YEAR = re.compile(r"(20[0-2]\d)")


def score(rec):
    u = rec["url"]
    p = urlparse(u)
    host = p.hostname
    path = unquote(p.path + ("?" + p.query if p.query else "")).lower()
    text_title = (rec["title"] or "").lower()
    text_desc = (rec["desc"] or "").lower()
    hay = f"{path} {text_title} {text_desc}"
    if EXCLUDE.search(path):
        return None
    if re.search(r"/profile/|/directory/faculty/|/faculty/profile|/lecturer/profile|/faculty/[a-z-]+/?$", path):
        return None  # faculty profiles are selected separately from the class schedule
    cats = {}
    for c, rx in CAT_RX.items():
        hits = len(rx.findall(path)) + 2 * len(rx.findall(text_title)) + 0.5 * len(rx.findall(text_desc))
        if hits:
            cats[c] = hits
    if not cats:
        return None
    s = sum(cats.values())
    if UNDERGRAD_HOSTS.search(host or ""):
        s += 2
    depth = len([x for x in p.path.split("/") if x])
    s -= max(0, depth - 3) * 0.7
    years = [int(y) for y in YEAR.findall(path + " " + (rec["lastmod"] or "")[:4])]
    newest = max(years) if years else None
    is_news = "news" in cats
    if is_news:
        if newest and newest >= Y:
            s += 2
        elif newest and newest >= Y - 1:
            s += 1
        elif newest and newest <= Y - 4:
            s -= 4
    if "undergrad" in hay:
        s += 1.5
    best = max((c for c in cats if c != "news"), key=lambda c: cats[c], default="news")
    bucket = "news" if (is_news and cats.get("news", 0) >= 1 and best in ("research", "academics")) else best
    return {"url": u, "host": host, "score": round(s, 2), "bucket": bucket, "cats": sorted(cats, key=lambda c: -cats[c])[:3],
            "title": rec["title"][:120]}


scored = [x for x in (score(r) for r in pool.values()) if x and x["url"] not in fetched]
scored.sort(key=lambda x: -x["score"])
# Loose cut: generous per-host caps; DeepSeek then rates relevance (pipeline/rate_urls.py).
host_used = collections.Counter()
cands = []
for x in scored:
    cap = 3 * LOW_HOSTS.get(x["host"], DEFAULT_HOST_CAP)
    if host_used[x["host"]] >= cap:
        continue
    host_used[x["host"]] += 1
    cands.append(x)
    if len(cands) >= 14000:
        break
with out_path.open("w") as f:
    for x in cands:
        f.write(json.dumps(x) + "\n")
print("scored", len(scored), "candidates", len(cands), "hosts", len(host_used))
