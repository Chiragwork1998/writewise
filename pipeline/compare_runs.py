"""Honest side-by-side comparison of the cloud run and the self-hosted run.

Compares: page fetching, page text, structured datasets (courses, organizations), discovery (map/search),
extracted facts per category, entities, graph, recall test, named-person checks, and speed.
usage: python pipeline/compare_runs.py colleges/usc colleges/usc_selfhost
"""
import collections
import difflib
import json
import re
import statistics
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent))
from firecrawl_client import url_id  # noqa: E402

cloud, self_dir = Path(sys.argv[1]), Path(sys.argv[2])
src = (Path(__file__).parent / "build_docs.py").read_text()
ns = {"re": re, "unicodedata": __import__("unicodedata")}
exec(src[src.index("def ascii_fold"): src.index("def key(")], ns)
norm, verify = ns["norm"], ns["verify"]


def load_pages(d):
    out = {}
    for f in (d / "pages").glob("*.json"):
        r = json.loads(f.read_text())
        if r.get("status") == "ok":
            out[r["url"]] = r
    return out


def load_log(d):
    return [json.loads(l) for l in (d / "logs" / "fetch_log.jsonl").read_text().splitlines()]


cp, sp = load_pages(cloud), load_pages(self_dir)
clog, slog = load_log(cloud), load_log(self_dir)
targets = {r["url"] for r in slog}  # URLs attempted in the self-hosted run
shared = [u for u in cp if u in targets]

facts_by_url = collections.defaultdict(list)
for l in (cloud / "extract" / "facts_raw.jsonl").read_text().splitlines():
    d = json.loads(l)
    if d["verification"] in ("exact", "near"):
        facts_by_url[d["url"]].append(d["evidence"])

rows = []
for u in shared:
    c, s = cp[u], sp.get(u)
    cmd = c.get("markdown") or ""
    smd = (s or {}).get("markdown") or ""
    cw, sw = norm(cmd).split(), norm(smd).split()
    sim = difflib.SequenceMatcher(None, cw[:5000], sw[:5000], autojunk=False).ratio() if cw and sw else 0.0
    ev = facts_by_url.get(u, [])
    pn = norm(smd)
    kept = sum(1 for e in ev if verify(e, pn) in ("exact", "exact_words", "near")) if ev else 0
    rows.append({"url": u, "bucket": (c.get("meta") or {}).get("bucket"), "ok": bool(s), "sim": round(sim, 3),
                 "cloud_chars": len(cmd), "self_chars": len(smd), "facts": len(ev), "facts_kept": kept,
                 "cloud_sec": c.get("elapsed"), "self_sec": (s or {}).get("elapsed")})
(self_dir / "page_comparison.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

fetched_ok = sum(1 for r in rows if r["ok"] and r["self_chars"] > 200)
cloud_ok = sum(1 for r in rows if r["cloud_chars"] > 200)
same_text = sum(1 for r in rows if r["sim"] >= 0.8)
ft, fk = sum(r["facts"] for r in rows), sum(r["facts_kept"] for r in rows)
csec = [r["cloud_sec"] for r in rows if r["cloud_sec"]]
ssec = [r["self_sec"] for r in rows if r["self_sec"]]


def courses(d, term="20263"):
    p = d / "extract" / f"courses_{term}.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


cc, sc = courses(cloud), courses(self_dir)
cmap = {c["code"]: c for c in cc}
smap = {c["code"]: c for c in sc}
both_codes = set(cmap) & set(smap)
same_instr = sum(1 for k in both_codes if set(cmap[k]["instructors"]) == set(smap[k]["instructors"]))
instr_cloud = {n for c in cc for n in c["instructors"]}
instr_self = {n for c in sc for n in c["instructors"]}


def orgs(d):
    p = d / "extract" / "organizations.jsonl"
    return {o["name"]: o for o in (json.loads(l) for l in p.read_text().splitlines())} if p.exists() else {}


co, so = orgs(cloud), orgs(self_dir)
org_both = set(co) & set(so)
same_mission = sum(1 for n in org_both if re.sub(r"\s+", " ", co[n]["mission"] or "").strip()
                   == re.sub(r"\s+", " ", so[n]["mission"] or "").strip())


def facts_stats(d):
    p = d / "extract" / "facts_raw.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    ok = [r for r in rows if r["verification"] in ("exact", "exact_words", "near")]
    cats = collections.Counter(r.get("category") for r in ok)
    ents = {(str((r.get("entity") or {}).get("type")), re.sub(r"\W+", " ", str((r.get("entity") or {}).get("name")).lower()).strip()) for r in ok}
    return {"raw": len(rows), "verified": len(ok), "cats": cats, "entities": len(ents),
            "verify_rate": len(ok) / max(len(rows), 1)}


cf, sf = facts_stats(cloud), facts_stats(self_dir)


def docs_stats(d):
    p = d / "output" / "build_stats.json"
    return json.loads(p.read_text()) if p.exists() else None


cd, sd = docs_stats(cloud), docs_stats(self_dir)


def graph_size(d):
    p = d / "output" / "graph" / "nodes.csv"
    if not p.exists():
        return None, None
    n = sum(1 for _ in p.open()) - 1
    e = sum(1 for _ in (d / "output" / "graph" / "edges.csv").open()) - 1
    return n, e


cn, ce = graph_size(cloud)
sn, se = graph_size(self_dir)

md = ["# Cloud Firecrawl vs self-hosted Firecrawl — same college, same pipeline", "",
      "Both runs used the same URL list, the same extraction model and prompts, the same verification and the same "
      "document builder. Only the fetching layer differs. The self-hosted run used Firecrawl v2.11.162 in Docker on a "
      "MacBook (Apple Silicon, 10 CPUs / 7.7 GB to Docker) over a home connection, with SearXNG for search and the "
      "Schedule of Classes' public data endpoint instead of browser clicks.", "",
      "## 1. Fetching the same pages", "", "| Measure | Cloud | Self-hosted |", "|---|---:|---:|",
      f"| Pages attempted | {len(shared):,} | {len(shared):,} |",
      f"| Pages returned with content (>200 chars) | {cloud_ok:,} | {fetched_ok:,} ({fetched_ok / max(len(shared),1) * 100:.1f}%) |",
      f"| Page text matching cloud (≥80%) | — | {same_text:,} ({same_text / max(len(shared),1) * 100:.1f}%) |",
      f"| Cloud's verified quotes found in self-hosted text | — | {fk:,} of {ft:,} ({fk / max(ft,1) * 100:.1f}%) |",
      f"| Median seconds per page | {statistics.median(csec) if csec else '—'} | {statistics.median(ssec) if ssec else '—'} |",
      ""]

by_bucket = collections.defaultdict(list)
for r in rows:
    by_bucket[r["bucket"] or "other"].append(r)
md += ["**By page type**", "", "| Page type | Pages | Returned | Text ≥80% match | Quotes kept |", "|---|---:|---:|---:|---:|"]
for b, g in sorted(by_bucket.items(), key=lambda kv: -len(kv[1])):
    t, k = sum(x["facts"] for x in g), sum(x["facts_kept"] for x in g)
    md.append(f"| {b} | {len(g):,} | {sum(1 for x in g if x['ok'] and x['self_chars'] > 200):,} | "
              f"{sum(1 for x in g if x['sim'] >= 0.8):,} | {k:,}/{t:,} ({k / max(t,1) * 100:.0f}%) |")

md += ["", "## 2. Course schedule (clicks vs public data endpoint)", "", "| Measure | Cloud (clicks) | Self-hosted (data endpoint) |",
       "|---|---:|---:|",
       f"| Departments with data | {len({c['source_url'] for c in cc}):,} | {len({c['source_url'] for c in sc}):,} |",
       f"| Courses | {len(cc):,} | {len(sc):,} |",
       f"| Undergraduate courses | {sum(1 for c in cc if c['number'][:1] in '1234'):,} | {sum(1 for c in sc if c['number'][:1] in '1234'):,} |",
       f"| Sections | {sum(len(c['sections']) for c in cc):,} | {sum(len(c['sections']) for c in sc):,} |",
       f"| Named instructors | {len(instr_cloud):,} | {len(instr_self):,} |",
       f"| Courses in both | {len(both_codes):,} | {len(both_codes):,} |",
       f"| Same instructor list for those courses | — | {same_instr:,} ({same_instr / max(len(both_codes),1) * 100:.0f}%) |",
       f"| Courses only in this run | {len(set(cmap) - set(smap)):,} | {len(set(smap) - set(cmap)):,} |", ""]

md += ["## 3. Student organization directory", "", "| Measure | Cloud | Self-hosted |", "|---|---:|---:|",
       f"| Groups parsed | {len(co):,} | {len(so):,} |",
       f"| Same group names | {len(org_both):,} | {len(org_both):,} |",
       f"| Identical mission text | — | {same_mission:,} of {len(org_both):,} |", ""]

mapcmp = self_dir / "discovery_selfhost" / "_map_comparison.json"
if mapcmp.exists():
    m = json.loads(mapcmp.read_text())
    tot = lambda k: sum(x[k] for x in m)
    md += ["## 4. Discovery (finding URLs) and search", "", "| Measure | Cloud | Self-hosted |", "|---|---:|---:|",
           f"| Sites mapped | {len(m)} | {len(m)} |",
           f"| URLs returned | {tot('cloud_urls'):,} | {tot('self_urls'):,} |",
           f"| URLs with titles (needed for AI page picking) | {tot('cloud_titles'):,} | {tot('self_titles'):,} |",
           f"| URLs found by both | {tot('overlap'):,} | {tot('overlap'):,} |",
           f"| URLs only this run found | {tot('cloud_urls') - tot('overlap'):,} | {tot('self_only'):,} |", ""]
scmp = self_dir / "discovery_selfhost" / "_search_comparison.json"
if scmp.exists():
    r = json.loads(scmp.read_text())
    s = lambda k: sum(x[k] for x in r)
    md += ["**Search** (cloud Firecrawl search vs self-hosted SearXNG), same queries", "",
           "| Measure | Cloud | Self-hosted |", "|---|---:|---:|",
           f"| Queries | {len(r)} | {len(r)} |",
           f"| Results returned | {s('cloud'):,} | {s('self'):,} |",
           f"| Results from outside the university | {s('cloud_external'):,} | {s('self_external'):,} |",
           f"| Same result URLs | — | {s('overlap'):,} |",
           f"| Queries returning nothing | — | {sum(1 for x in r if x['self'] == 0)} |", ""]

if sf:
    md += ["## 5. Extracted facts and documents", "", "| Measure | Cloud | Self-hosted |", "|---|---:|---:|",
           f"| Facts extracted | {cf['raw']:,} | {sf['raw']:,} |",
           f"| Quote verified against the page | {cf['verify_rate'] * 100:.1f}% | {sf['verify_rate'] * 100:.1f}% |",
           f"| Verified facts | {cf['verified']:,} | {sf['verified']:,} |",
           f"| Distinct entities | {cf['entities']:,} | {sf['entities']:,} |",
           f"| Graph nodes / relationships | {cn:,} / {ce:,} | {(sn or 0):,} / {(se or 0):,} |", ""]
    if cd and sd:
        cdoc = {d[1]: d[2] for d in cd["documents"]}
        sdoc = {d[1]: d[2] for d in sd["documents"]}
        md += ["**Facts per category document**", "", "| Category | Cloud | Self-hosted | Difference |", "|---|---:|---:|---:|"]
        for name in cdoc:
            c_, s_ = cdoc[name], sdoc.get(name, 0)
            md.append(f"| {name} | {c_:,} | {s_:,} | {(s_ - c_) / max(c_, 1) * 100:+.0f}% |")
        md.append("")


# ---------------- fetch failures: which pages self-hosted could not get, and did cloud get them?
last_self = {}
for r in slog:
    if r.get("status") in ("ok", "error", "robots_disallowed"):
        last_self[r["url"]] = r
self_failed = [u for u, r in last_self.items() if r["status"] == "error" and u not in sp]
fail_hosts = collections.Counter(urlparse(u).hostname for u in self_failed)
cloud_had = sum(1 for u in self_failed if u in cp)
lost_facts = sum(len(facts_by_url.get(u, [])) for u in self_failed)
md += ["## 6. Pages self-hosted could not fetch", "",
       f"{len(self_failed)} URLs failed after retries ({cloud_had} of them were fetched fine by cloud; those pages carried "
       f"{lost_facts:,} verified facts in the cloud run). Cause, checked by hand on keck.usc.edu, today.usc.edu and apnews.com: "
       "self-hosted Firecrawl reports `document_antibot` (\"Scrape aborted after exceeding retry limit\") and a plain "
       "request from the same machine gets HTTP 403. These sites sit behind bot protection that cloud Firecrawl's "
       "anti-bot layer gets through and the open-source version does not.", "",
       "| Site | Failed pages |", "|---|---:|"]
md += [f"| {h} | {n} |" for h, n in fail_hosts.most_common(15)]
md.append("")


# ---------------- time and machine load
def active_minutes(log, gap=600):
    from datetime import datetime
    ts = sorted(datetime.strptime(r["t"], "%Y-%m-%dT%H:%M:%SZ").timestamp() for r in log
                if r.get("status") in ("ok", "error") and r.get("t"))
    return sum(min(b - a, gap) for a, b in zip(ts, ts[1:])) / 60 if len(ts) > 1 else 0


def credits(log):
    return sum(r.get("credits") or 0 for r in log)


dstats = self_dir / "logs" / "docker_stats.csv"
load = ""
if dstats.exists():
    snap = collections.defaultdict(lambda: [0.0, 0.0])
    for line in dstats.read_text().splitlines():
        parts = line.split(",")
        if len(parts) != 4 or not parts[1].startswith("firecrawl"):
            continue
        cpu = float(parts[2].rstrip("%") or 0)
        m = re.match(r"([\d.]+)\s*([KMG]i?B)", parts[3])
        mem = float(m.group(1)) * {"KiB": 1 / 1024 / 1024, "MiB": 1 / 1024, "GiB": 1}.get(m.group(2), 0) if m else 0
        snap[parts[0]][0] += cpu
        snap[parts[0]][1] += mem
    if snap:
        cpus = [v[0] for v in snap.values()]
        mems = [v[1] for v in snap.values()]
        load = (f"| Self-hosted stack load (all containers) | — | CPU avg {statistics.mean(cpus) / 100:.1f} cores, peak "
                f"{max(cpus) / 100:.1f}; memory avg {statistics.mean(mems):.1f} GB, peak {max(mems):.1f} GB |")
md += ["## 7. Time, credits and machine load", "", "| Measure | Cloud | Self-hosted |", "|---|---:|---:|",
       f"| Active fetching time (pauses over 10 min excluded) | {active_minutes(clog):.0f} min | {active_minutes(slog):.0f} min |",
       f"| Firecrawl cloud credits used | {credits(clog):,} | 0 |"]
if load:
    md.append(load)
md += ["", "Cloud times include deliberate crawl-delay pacing and the discovery/triage rounds; self-hosted times are one "
       "pass over a fixed URL list on a laptop, so they are indicative only.", ""]


# ---------------- recall test and named people
def recall_table(d):
    p = d / "output" / "crawl_report.md"
    if not p.exists():
        return None, {}
    t = p.read_text()
    m = re.search(r"Recall test — (\d+)/(\d+)", t)
    items = {}
    sec = t[t.find("## 4. Recall test"):]
    sec = sec[: sec.find("\n## ", 5)]
    for line in sec.splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == 3 and cells[0] not in ("Item", "---"):
            items[cells[0]] = "not found" not in cells[1]
    return (m.group(1), m.group(2)) if m else None, items


rc, ri = recall_table(cloud)
rs, rsi = recall_table(self_dir)
if rc and rs:
    diff = [k for k in ri if ri[k] != rsi.get(k)]
    md += ["## 8. Recall test (50 items named in the earlier ChatGPT research)", "",
           f"Cloud found **{rc[0]}/{rc[1]}**, self-hosted found **{rs[0]}/{rs[1]}**.", ""]
    if diff:
        md += ["| Item | Cloud | Self-hosted |", "|---|---|---|"]
        md += [f"| {k} | {'found' if ri[k] else 'not found'} | {'found' if rsi.get(k) else 'not found'} |" for k in diff]
    else:
        md.append("Both runs found and missed exactly the same items.")
    md.append("")

PEOPLE = ["Bhaskar Krishnamachari", "Lowell Stott", "Wolinsky-Nahmias", "Cameron Egan", "Ruddell", "Agius Vallejo"]


def verified_rows(d):
    p = d / "extract" / "facts_raw.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


cr_rows = [r for r in verified_rows(cloud) if r["verification"] in ("exact", "exact_words", "near")]
sr_rows = [r for r in verified_rows(self_dir) if r["verification"] in ("exact", "exact_words", "near")]
md += ["## 9. Named people", "", "Verified facts that mention the person, and Fall 2026 courses listing them as instructor.", "",
       "| Person | Cloud facts | Self-hosted facts | Cloud courses | Self-hosted courses |", "|---|---:|---:|---:|---:|"]
for name in PEOPLE:
    last = name.split()[-1]
    rx = re.compile(r"(?<![A-Za-z])" + re.escape(last) + r"(?![A-Za-z])")
    fc_ = sum(1 for r in cr_rows if rx.search(r["fact"] + " " + r["evidence"]))
    fs_ = sum(1 for r in sr_rows if rx.search(r["fact"] + " " + r["evidence"]))
    cc_ = sum(1 for c in cc if any(rx.search(i) for i in c["instructors"]))
    sc_ = sum(1 for c in sc if any(rx.search(i) for i in c["instructors"]))
    md.append(f"| {name} | {fc_:,} | {fs_:,} | {cc_} | {sc_} |")
md.append("")

# ---------------- randomness control
ctl = json.loads((Path(cloud).parent / "usc_control" / "control_result.json").read_text()) \
    if (Path(cloud).parent / "usc_control" / "control_result.json").exists() else None
if ctl:
    o = ctl["overlap"]
    pct = lambda k: o[k][0] / max(o[k][1], 1) * 100
    noise = (pct("A1->A2") + pct("A2->A1")) / 2
    fetch = (pct("A1->S") + pct("S->A1") + pct("A2->S") + pct("S->A2")) / 4
    c = ctl["counts"]
    md += ["## 10. Is the difference self-hosting, or just AI randomness?", "",
           f"On {ctl['pages']} randomly chosen pages, the AI read the **cloud** page text twice (A1, A2) using the same "
           "code and prompt as the self-hosted run, and those were compared with the self-hosted extraction (S) of the "
           "same URLs. A1 vs A2 shows how much the AI varies on identical input; the extra gap between A and S is what "
           "self-hosted fetching changes.", "",
           "| Measure | Cloud text, run 1 | Cloud text, run 2 | Self-hosted text |", "|---|---:|---:|---:|",
           f"| Verified facts | {c['A1']:,} | {c['A2']:,} | {c['S']:,} |",
           f"| Pages with no facts | {ctl['zero_pages']['A1']} | {ctl['zero_pages']['A2']} | {ctl['zero_pages']['S']} |", "",
           f"- Two runs on **identical** cloud text share **{noise:.0f}%** of their facts (the natural randomness floor).",
           f"- Cloud-text runs vs self-hosted-text run share **{fetch:.0f}%** of their facts.",
           f"- So self-hosted fetching accounts for about **{max(0.0, noise - fetch):.0f} percentage points** of "
           "difference beyond normal randomness.",
           f"- For reference, the original cloud extraction (O, {c['O']:,} facts on these pages, earlier prompt) vs "
           f"today's re-run on the same text: {pct('O->A1'):.0f}% / {pct('A1->O'):.0f}% overlap.", ""]

worst = sorted([r for r in rows if not r["ok"] or r["sim"] < 0.6], key=lambda r: r["sim"])[:30]
md += ["## 11. Pages where self-hosted did worse", "", f"{len(worst)} shown of "
       f"{sum(1 for r in rows if not r['ok'] or r['sim'] < 0.6)} pages that failed or differ a lot.", "",
       "| Page | Type | Self chars / cloud chars | Text match | Quotes kept |", "|---|---|---|---:|---|"]
for r in worst:
    md.append(f"| {r['url'][:80]} | {r['bucket']} | {r['self_chars']:,} / {r['cloud_chars']:,} | {r['sim']} | "
              f"{r['facts_kept']}/{r['facts']} |")
(self_dir / "comparison_report.md").write_text("\n".join(md) + "\n")
print("\n".join(md[:60]))
print("\n... full report:", self_dir / "comparison_report.md")
