"""Stage 7a — collect every faculty/staff profile-looking URL we know about (Firecrawl maps, sitemaps, links on
scraped pages) into discovery/profile_urls.json, which match_faculty.py needs.

Check the printed host list: if a school's profiles use a URL shape not in PROFILE below, add it and re-run.
usage: python pipeline/build_profile_urls.py colleges/<slug> <root_domain>
"""
import collections
import glob
import json
import re
import sys
from pathlib import Path

college, root = Path(sys.argv[1]), sys.argv[2].lower()
PROFILE = re.compile(r"(/profile/|/directory/faculty/[^/]+/[^/]+|/faculty/profile|/lecturer/profile|/faculty-directory/|"
                     r"/personnel/|/faculty/[a-z0-9-]+/?$|/people/[a-z0-9-]+/?$|/faculty-research/directory/[a-z0-9-]+|"
                     r"/our-people/|/bio/|/directory/[a-z0-9-]+/?$|/staff/[a-z0-9-]+/?$)", re.I)
urls = set()
for f in glob.glob(str(college / "discovery" / "maps" / "*.json")):
    for l in json.load(open(f)).get("links", []) or []:
        urls.add(l["url"] if isinstance(l, dict) else l)
for f in glob.glob(str(college / "discovery" / "sitemaps" / "*.json")):
    for u, _ in json.load(open(f)).get("sitemap_urls", []):
        urls.add(u)
for f in glob.glob(str(college / "pages" / "*.json")):
    for l in json.load(open(f)).get("links", []) or []:
        if isinstance(l, str):
            urls.add(l)
prof = sorted({u.split("#")[0] for u in urls if PROFILE.search(u) and root in (re.sub(r"^https?://", "", u).split("/")[0])})
hosts = collections.Counter(re.sub(r"^https?://", "", u).split("/")[0] for u in prof)
(college / "discovery" / "profile_urls.json").write_text(json.dumps(prof))
print(f"profile-like URLs: {len(prof):,}")
for h, n in hosts.most_common(30):
    print(f"  {n:6,}  {h}")
