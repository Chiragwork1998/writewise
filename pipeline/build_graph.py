"""Knowledge graph export: nodes + edges from verified facts, the class schedule and the org directory.

Every edge keeps its provenance (source URL + evidence quote, or the dataset it came from).
Can be restricted to a subset of page URLs (used by the budget simulation).

usage: python pipeline/build_graph.py colleges/usc "University of Southern California" usc.edu
Outputs output/graph/nodes.csv, edges.csv, graph.json
"""
import collections
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def load_context(college, school, root):
    src = (Path(__file__).parent / "build_docs.py").read_text()
    ns = {"__name__": "graph_import"}
    saved = sys.argv
    sys.argv = [saved[0], str(college), school, "X", root]
    head = src[: src.index("# ------------------------------------------------------------------ rendering helpers")]
    exec(head, ns)
    # resolve() lives further down; pull just that block
    block = src[src.index("ALIASES = {}"): src.index("def entity_sections(")]
    exec(block, ns)
    sys.argv = saved
    return ns


NODE_TYPES = {"university", "school", "department", "program", "course", "center", "lab", "professor", "person",
              "organization", "tradition", "service", "office", "partner", "facility", "award", "event", "publication"}


def build(ns, page_filter=None, course_filter=None, include_orgs=True):
    facts, orgs, courses, key, resolve = ns["facts"], ns["orgs"], ns["courses"], ns["key"], ns["resolve"]
    nodes, edges = {}, []
    name_index = {}

    def node(t, name, **attrs):
        k, canon = resolve(t, name)
        nid = f"{t}:{k}"
        if nid not in nodes:
            nodes[nid] = {"id": nid, "type": t, "name": canon, "facts": 0, "sources": set()}
        for a, v in attrs.items():
            nodes[nid].setdefault(a, v)
        name_index.setdefault(key(name), nid)
        return nid

    for fa in facts:
        if page_filter is not None and not (set(fa["sources"]) & page_filter):
            continue
        e = fa.get("entity") or {}
        t = (e.get("type") or "other").lower()
        if t not in NODE_TYPES or not e.get("name"):
            continue
        nid = node(t, e["name"].strip())
        nodes[nid]["facts"] += 1
        nodes[nid]["sources"].update(fa["sources"])
        nodes[nid].setdefault("categories", set()).add(fa["code"])
        for rel in fa.get("relations") or []:
            if not (isinstance(rel, list) and len(rel) == 3 and all(isinstance(x, str) and x.strip() for x in rel)):
                continue
            s, p, o = (x.strip() for x in rel)
            edges.append({"src_name": s, "pred": re.sub(r"[^A-Z_]", "", p.upper()) or "RELATED_TO", "dst_name": o,
                          "source_url": fa["url"], "evidence": fa["evidence"][:300]})

    for c in courses:
        if course_filter is not None and c["source_url"] not in course_filter:
            continue
        cid = node("course", f"{c['code']} {c['title']}", code=c["code"], term=c["term"])
        nodes[cid]["sources"].add(c["source_url"])
        for n in c["instructors"]:
            pid = node("professor", n)
            nodes[pid]["sources"].add(c["source_url"])
            edges.append({"src": pid, "pred": "TEACHES", "dst": cid, "source_url": c["source_url"], "evidence": c["term"]})
    if include_orgs:
        for o in orgs:
            oid = node("organization", o["name"], group_type=o["group_type"])
            nodes[oid]["sources"].add(o["source_url"])
            for cat in o["categories"]:
                tid = node("tag", cat) if False else f"tag:{key(cat)}"
                nodes.setdefault(tid, {"id": tid, "type": "tag", "name": cat, "facts": 0, "sources": set()})
                edges.append({"src": oid, "pred": "TAGGED", "dst": tid, "source_url": o["source_url"], "evidence": "directory category"})

    # resolve relation endpoints to nodes by name; keep only edges whose both ends are known entities
    resolved = []
    for ed in edges:
        if "src" not in ed:
            s = name_index.get(key(ed["src_name"]))
            d = name_index.get(key(ed["dst_name"]))
            if not s or not d or s == d:
                continue
            ed = {"src": s, "pred": ed["pred"], "dst": d, "source_url": ed["source_url"], "evidence": ed["evidence"]}
        resolved.append(ed)
    seen, uniq = set(), []
    for ed in resolved:
        k = (ed["src"], ed["pred"], ed["dst"])
        if k not in seen:
            seen.add(k)
            uniq.append(ed)
    return nodes, uniq


def export(nodes, edges, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "nodes.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "type", "name", "fact_count", "source_count", "categories"])
        for n in nodes.values():
            w.writerow([n["id"], n["type"], n["name"], n["facts"], len(n["sources"]), ";".join(sorted(n.get("categories", [])))])
    with (out_dir / "edges.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src", "predicate", "dst", "source_url", "evidence"])
        for e in edges:
            w.writerow([e["src"], e["pred"], e["dst"], e["source_url"], e["evidence"]])
    (out_dir / "graph.json").write_text(json.dumps({
        "nodes": [{**{k: v for k, v in n.items() if k not in ("sources", "categories")}, "sources": sorted(n["sources"])[:5],
                   "categories": sorted(n.get("categories", []))} for n in nodes.values()],
        "edges": edges}))


if __name__ == "__main__":
    college, school, root = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    ns = load_context(college, school, root)
    nodes, edges = build(ns)
    export(nodes, edges, college / "output" / "graph")
    print("nodes by type:", collections.Counter(n["type"] for n in nodes.values()).most_common())
    print("edges by predicate:", collections.Counter(e["pred"] for e in edges).most_common(15))
    print(f"TOTAL nodes={len(nodes):,} edges={len(edges):,}")
