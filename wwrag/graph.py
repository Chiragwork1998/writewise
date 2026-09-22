"""wwrag/graph.py -- entity graph over an existing index: facts become chains.

The bundle ships a verified entity graph (professor DIRECTS lab, professor TEACHES course,
lab PART_OF department, ...) that the rest of the pipeline never opened. Without it a report
is a directory: each retrieved fact stands alone, and a student reads a list of things that
exist. With it a report can say who runs the lab, what they teach, and which department it
sits in -- the chain an adviser would draw, and the thing a student cannot get by searching
the college's own website.

Every edge carries its own source_url and a verbatim evidence quote, so a claim built from a
chain is exactly as checkable as a claim built from a fact. Nothing here invents a relation:
the graph was extracted from the same pages, with the same quote-or-discard rule.

The tables live in the index's chunks.sqlite but need no vectors, so adding them to an index
that already exists costs seconds and no embedding spend.

Build (once per index):
    python wwrag/graph.py build --index wwrag/index-openai/usc \
        --data-dir colleges/<slug>/export/<bundle>

Inspect:
    python wwrag/graph.py show --index wwrag/index-openai/usc --entity "Autonomous Networks Research Group"
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

CONFIG: dict[str, Any] = {
    "nodes_file": "graph_nodes.csv",
    "edges_file": "graph_edges.csv",
    # Relations worth putting in front of a student, best first. A chain is only useful if
    # the reader can act on it, so "who runs this and what do they teach" beats "this is
    # tagged with that".
    "relation_priority": (
        # who runs it and who teaches it first: those are the two things a student cannot
        # look up. PART_OF sits near the bottom deliberately -- every entity has a parent,
        # so ranking it high made 42% of all chains read "X is part of Y".
        "DIRECTS", "TEACHES", "RESEARCHES", "OFFERED_BY", "HOSTS",
        "FUNDS", "PARTNERS_WITH", "AWARDED", "MEMBER_OF",
        "PART_OF", "AFFILIATED_WITH", "LOCATED_IN",
    ),
    # Relations that say nothing a student can use.
    "relation_skip": ("TAGGED",),
    # At most this many of one relation per anchor. Measured on a real run: PART_OF took 73
    # of 173 chains because every entity has a parent, while TEACHES -- the single most useful
    # thing you can tell an applicant -- managed 5. Without a cap the always-available
    # relation crowds out the rare valuable one, and every chain reads "X is part of Y".
    "max_per_relation": 1,
    # Neighbour types that are administrative rather than something a student does.
    "type_skip": ("office",),
    # MEMBER_OF a named individual surfaces whoever happened to be listed on a club page --
    # a real student's name, printed in a stranger's report, telling the reader nothing.
    # A professor is different: that is who runs the thing.
    "member_of_person_types": ("person",),
    # Node types worth surfacing: a person, a place to work, a thing to take.
    "type_priority": (
        "professor", "person", "lab", "center", "program", "course",
        "department", "school", "organization", "facility", "tradition",
    ),
    "max_hops": 2,
    "neighbours_per_unit": 2,   # 4 let weak relations fill slots behind the strong one
    "max_evidence_chars": 300,
    # An entity wired to hundreds of things is a hub -- the university itself, or a whole
    # school -- and "X is part of the university" tells a student nothing. On USC's graph the
    # 18 nodes above degree 120 are 14 schools, 2 universities and one lab counted twice under
    # an alias, with nothing at all between 100 and 119. So what "hub" really means is a type,
    # not a number, and the type rule is the one that holds at a college whose whole graph is
    # smaller than USC's engineering school.
    #
    # Tempting, and wrong: blocking every node typed "school" as well. Measured against the
    # four students' real evidence, that would have blocked USC Leventhal School of Accounting
    # (degree 41, anchoring 15 units) and the Iovine and Young Academy (degree 16) -- small,
    # specific schools that are the most useful thing an applicant can be told about. A hub is
    # a node wired to everything, not a node that happens to be a school.
    #
    # So the rule stays a degree, and the degree scales to the graph in front of it: an absolute
    # count tuned on USC would never fire on a college with a tenth of the corpus. USC's 99th
    # percentile is 13, so this lands at 104 and selects the same 18 nodes that 120 did.
    "hub_degree_p99_multiple": 8,
    "hub_degree_floor": 60,
}

_WS = re.compile(r"\s+")


_HUB_DEGREE_CACHE: dict[int, int] = {}


def hub_degree(conn: sqlite3.Connection, cfg: dict[str, Any]) -> int:
    """The degree at which a node counts as a hub, read off this graph's own distribution."""
    key = id(conn)
    if key not in _HUB_DEGREE_CACHE:
        degrees = [d for (d,) in conn.execute(
            "SELECT degree FROM graph_nodes WHERE degree IS NOT NULL ORDER BY degree")]
        p99 = degrees[int(len(degrees) * 0.99)] if degrees else 0
        _HUB_DEGREE_CACHE[key] = max(int(cfg["hub_degree_floor"]),
                                     p99 * int(cfg["hub_degree_p99_multiple"]))
    return _HUB_DEGREE_CACHE[key]


def is_hub(row: dict[str, Any], threshold: int) -> bool:
    """Wired to so much that a chain from it says nothing an applicant can act on."""
    return (row.get("degree") or 0) >= threshold


def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(2)


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def norm_name(text: str | None) -> str:
    """Match the bundle's node-id convention: lowercase, whitespace-collapsed."""
    if not text:
        return ""
    out = unicodedata.normalize("NFKC", str(text)).casefold()
    return _WS.sub(" ", out).strip()


# --------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS graph_nodes (
    id            TEXT PRIMARY KEY,   -- "<type>:<normalised name>", as the bundle writes it
    type          TEXT NOT NULL,
    name          TEXT NOT NULL,
    name_key      TEXT NOT NULL,      -- normalised name, for matching a unit's entity_name
    fact_count    INTEGER,
    source_count  INTEGER,
    categories    TEXT,
    degree        INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_graph_nodes_key  ON graph_nodes(name_key);
CREATE INDEX IF NOT EXISTS idx_graph_nodes_type ON graph_nodes(type);

CREATE TABLE IF NOT EXISTS graph_edges (
    src        TEXT NOT NULL,
    predicate  TEXT NOT NULL,
    dst        TEXT NOT NULL,
    source_url TEXT,
    evidence   TEXT               -- verbatim quote that states the relation
);
CREATE INDEX IF NOT EXISTS idx_graph_edges_src ON graph_edges(src);
CREATE INDEX IF NOT EXISTS idx_graph_edges_dst ON graph_edges(dst);
"""


# --------------------------------------------------------------------------------------
# Edge direction
# --------------------------------------------------------------------------------------

# The extractor reads a relation off one sentence, and a sentence often names the object first
# ("MKT 486, led by Professor Joseph Nunes"). It then writes the edge the wrong way round, and the
# report states it as fact: "MKT 486 teaches Joseph Nunes". Node types settle this without reading
# the sentence at all -- a course cannot teach, a lab cannot direct, an event cannot host -- so an
# edge is turned round only when its stated subject CANNOT act and its object CAN.
CANNOT_BE_SUBJECT = {
    "TEACHES": {"course", "publication", "award", "tradition", "tag", "facility"},
    "DIRECTS": {"course", "publication", "award", "tradition", "tag", "facility",
                "lab", "center", "program", "event", "service"},
    "HOSTS": {"course", "publication", "award", "tradition", "tag", "event"},
}
CAN_BE_SUBJECT = {
    "TEACHES": {"person", "professor", "organization", "department", "center", "lab",
                "school", "program", "office"},
    "DIRECTS": {"person", "professor", "organization", "office", "school", "department"},
    "HOSTS": {"organization", "center", "lab", "school", "department", "university",
              "office", "program", "service", "facility", "partner"},
}
# "Ep 5 hosts Eileen Crimmins" is wrong, but so is turning it round: she is the guest, not the host.
# When an event or a publication is said to host a person, nobody can tell which way it runs, so the
# edge is dropped rather than guessed. Institutions are different -- they host, they are not hosted.
UNKNOWABLE_HOST = {"person", "professor"}


def reorient(src: str, pred: str, dst: str, node_types: dict[str, str]) -> tuple[str, str] | None:
    """(src, dst) the right way round, or None when the direction cannot be known."""
    st, dt = node_types.get(src), node_types.get(dst)
    if st not in CANNOT_BE_SUBJECT.get(pred, ()):
        return src, dst
    if pred == "HOSTS" and dt in UNKNOWABLE_HOST:
        return None
    if dt in CAN_BE_SUBJECT.get(pred, ()):
        return dst, src
    return src, dst


def build(index_dir: Path, data_dir: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    db_path = index_dir / "chunks.sqlite"
    if not db_path.is_file():
        die(f"{db_path} does not exist -- build the index first")
    nodes_csv = data_dir / cfg["nodes_file"]
    edges_csv = data_dir / cfg["edges_file"]
    for p in (nodes_csv, edges_csv):
        if not p.is_file():
            die(f"{p} is missing from the bundle")

    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA)
    conn.execute("DELETE FROM graph_nodes")
    conn.execute("DELETE FROM graph_edges")

    csv.field_size_limit(1 << 24)
    skip = set(cfg["relation_skip"])

    nodes: list[tuple] = []
    with nodes_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            nid = (row.get("id") or "").strip()
            if not nid:
                continue
            name = (row.get("name") or "").strip()
            nodes.append((
                nid, (row.get("type") or "").strip(), name, norm_name(name),
                int(row.get("fact_count") or 0), int(row.get("source_count") or 0),
                (row.get("categories") or "").strip(),
            ))
    conn.executemany(
        "INSERT OR REPLACE INTO graph_nodes(id,type,name,name_key,fact_count,source_count,categories)"
        " VALUES (?,?,?,?,?,?,?)", nodes,
    )

    edges: list[tuple] = []
    degree: dict[str, int] = defaultdict(int)
    skipped = 0
    turned = 0
    dropped_direction = 0
    node_types = {n[0]: n[1] for n in nodes}
    with edges_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            src = (row.get("src") or "").strip()
            dst = (row.get("dst") or "").strip()
            pred = (row.get("predicate") or "").strip().upper()
            if not src or not dst or not pred:
                continue
            if pred in skip:
                skipped += 1
                continue
            oriented = reorient(src, pred, dst, node_types)
            if oriented is None:
                dropped_direction += 1
                continue
            if oriented != (src, dst):
                turned += 1
            src, dst = oriented
            edges.append((src, pred, dst, (row.get("source_url") or "").strip(),
                          (row.get("evidence") or "").strip()))
            degree[src] += 1
            degree[dst] += 1
    conn.executemany(
        "INSERT INTO graph_edges(src,predicate,dst,source_url,evidence) VALUES (?,?,?,?,?)", edges,
    )
    conn.executemany("UPDATE graph_nodes SET degree=? WHERE id=?",
                     [(d, n) for n, d in degree.items()])
    conn.commit()

    stats = {
        "nodes": len(nodes),
        "edges": len(edges),
        "skipped_relations": skipped,
        "turned_round": turned,
        "dropped_unknown_direction": dropped_direction,
        "hubs": conn.execute("SELECT COUNT(*) FROM graph_nodes WHERE degree >= ?",
                             (hub_degree(conn, cfg),)).fetchone()[0],
    }
    conn.close()
    return stats


def has_graph(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='graph_edges'"
    ).fetchone()
    if not row or not row[0]:
        return False
    return bool(conn.execute("SELECT 1 FROM graph_edges LIMIT 1").fetchone())


# --------------------------------------------------------------------------------------
# Expand
# --------------------------------------------------------------------------------------

def _resolve(conn: sqlite3.Connection, entity_name: str | None) -> list[str]:
    """Node ids for a unit's entity_name. Several types can share a name."""
    key = norm_name(entity_name)
    if not key:
        return []
    rows = conn.execute("SELECT id FROM graph_nodes WHERE name_key = ?", (key,)).fetchall()
    return [r[0] for r in rows]


def _neighbours(conn: sqlite3.Connection, node_ids: Sequence[str], cfg: dict[str, Any]
                ) -> list[dict[str, Any]]:
    if not node_ids:
        return []
    marks = ",".join("?" for _ in node_ids)
    out: list[dict[str, Any]] = []
    for direction, sql in (
        ("out", f"SELECT src, predicate, dst, source_url, evidence FROM graph_edges WHERE src IN ({marks})"),
        ("in", f"SELECT dst, predicate, src, source_url, evidence FROM graph_edges WHERE dst IN ({marks})"),
    ):
        for anchor, pred, other, url, ev in conn.execute(sql, list(node_ids)):
            out.append({"anchor": anchor, "predicate": pred, "other": other,
                        "direction": direction, "source_url": url, "evidence": ev})
    return out


def _node_rows(conn: sqlite3.Connection, ids: Iterable[str]) -> dict[str, dict[str, Any]]:
    ids = list(dict.fromkeys(ids))
    out: dict[str, dict[str, Any]] = {}
    for start in range(0, len(ids), 900):
        batch = ids[start:start + 900]
        marks = ",".join("?" for _ in batch)
        for row in conn.execute(
            f"SELECT id,type,name,fact_count,source_count,categories,degree"
            f" FROM graph_nodes WHERE id IN ({marks})", batch
        ):
            out[row[0]] = {"id": row[0], "type": row[1], "name": row[2], "fact_count": row[3],
                           "source_count": row[4], "categories": row[5], "degree": row[6]}
    return out


def _rank(edge: dict[str, Any], node: dict[str, Any], cfg: dict[str, Any]) -> tuple:
    rel_pri = cfg["relation_priority"]
    typ_pri = cfg["type_priority"]
    rel_rank = rel_pri.index(edge["predicate"]) if edge["predicate"] in rel_pri else len(rel_pri)
    typ_rank = typ_pri.index(node["type"]) if node.get("type") in typ_pri else len(typ_pri)
    # a neighbour with evidence beats one without; a well-sourced entity beats a bare mention
    return (rel_rank, typ_rank, 0 if edge.get("evidence") else 1,
            -(node.get("source_count") or 0))


def expand_unit(conn: sqlite3.Connection, unit: dict[str, Any], cfg: dict[str, Any] | None = None
                ) -> list[dict[str, Any]]:
    """Chains hanging off one evidence unit's entity, best first.

    Each entry is independently citable: it carries the source_url and the verbatim quote
    that states the relation, so a sentence built from it can be verified like any other.
    """
    cfg = cfg or CONFIG
    node_ids = _resolve(conn, unit.get("entity_name"))
    if not node_ids:
        return []
    hub = hub_degree(conn, cfg)
    anchors = _node_rows(conn, node_ids)
    # One name can resolve to several nodes ("Autonomous Networks Research Group" exists as
    # both a lab and an organization, one of them with no edges at all). Expand from the most
    # connected sense, and never refuse to expand from the anchor itself: the anchor is what
    # the student's evidence is actually about, however well connected it happens to be.
    node_ids = sorted(node_ids, key=lambda n: -(anchors.get(n, {}).get("degree") or 0))
    # Do not expand FROM a hub. A fact about a whole school expands to every dean and every
    # department, which is true, cited, and useless to a student deciding where to apply --
    # "Maja Mataric directs USC" is not advice. Chains are only worth drawing between things
    # specific enough to act on.
    if is_hub(anchors.get(node_ids[0], {}), hub):
        return []

    edges = [e for e in _neighbours(conn, node_ids, cfg) if e["other"] not in set(node_ids)]
    if not edges:
        return []
    others = _node_rows(conn, (e["other"] for e in edges))

    scored: list[tuple[tuple, dict[str, Any]]] = []
    seen: set[tuple[str, str]] = set()
    for e in edges:
        node = others.get(e["other"])
        if not node:
            continue
        # ...and do not expand INTO one either: "this lab is part of the engineering school"
        # is shelf-filler next to "this professor directs this lab and teaches this course".
        if (node.get("degree") or 0) >= hub:
            continue
        key = (e["predicate"], e["other"])
        if key in seen:
            continue
        seen.add(key)
        # value filters, applied before ranking so a low-value chain cannot fill a slot
        if e["predicate"] == "MEMBER_OF" and node.get("type") in cfg.get("member_of_person_types", ()):
            continue
        if node.get("type") in cfg.get("type_skip", ()):
            continue
        scored.append((_rank(e, node, cfg), {
            "relation": e["predicate"],
            "direction": e["direction"],
            "entity": node["name"],
            "entity_type": node["type"],
            "anchor": anchors.get(e["anchor"], {}).get("name") or unit.get("entity_name"),
            "evidence": (e.get("evidence") or "")[: int(cfg["max_evidence_chars"])],
            "source_url": e.get("source_url") or "",
            "categories": node.get("categories") or "",
        }))
    scored.sort(key=lambda p: p[0])
    # Take the best, but cap how many can share one relation so the slots stay varied.
    cap = int(cfg.get("max_per_relation", 0) or 0)
    want = int(cfg["neighbours_per_unit"])
    picked: list[dict[str, Any]] = []
    used: dict[str, int] = {}
    overflow: list[dict[str, Any]] = []
    for _, item in scored:
        rel = item["relation"]
        if cap and used.get(rel, 0) >= cap:
            overflow.append(item)
            continue
        used[rel] = used.get(rel, 0) + 1
        picked.append(item)
        if len(picked) >= want:
            break
    # rather than return fewer chains than asked for, refill from what the cap held back
    for item in overflow:
        if len(picked) >= want:
            break
        picked.append(item)
    return picked[:want]


def expand_units(conn: sqlite3.Connection, units: Sequence[dict[str, Any]],
                 cfg: dict[str, Any] | None = None) -> dict[str, list[dict[str, Any]]]:
    """{unit_id: [chain, ...]} for every unit that anchors onto the graph."""
    cfg = cfg or CONFIG
    if not has_graph(conn):
        return {}
    out: dict[str, list[dict[str, Any]]] = {}
    for unit in units:
        uid = unit.get("unit_id")
        if not uid:
            continue
        chains = expand_unit(conn, unit, cfg)
        if chains:
            out[uid] = chains
    return out


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Entity graph over an existing index.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="load the bundle's graph into an existing index")
    b.add_argument("--index", required=True, type=Path, help="a college index directory")
    b.add_argument("--data-dir", required=True, type=Path, help="the bundle directory")

    s = sub.add_parser("show", help="print the chains hanging off one entity")
    s.add_argument("--index", required=True, type=Path)
    s.add_argument("--entity", required=True)

    args = ap.parse_args(argv)

    if args.cmd == "build":
        stats = build(args.index.resolve(), args.data_dir.resolve(), CONFIG)
        log(f"graph loaded into {args.index}/chunks.sqlite: "
            f"{stats['nodes']:,} nodes, {stats['edges']:,} edges "
            f"({stats['skipped_relations']:,} uninformative relations skipped, "
            f"{stats['hubs']:,} hubs will not be expanded through)")
        return 0

    conn = sqlite3.connect(f"file:{args.index / 'chunks.sqlite'}?mode=ro", uri=True)
    if not has_graph(conn):
        die("this index has no graph tables; run 'build' first")
    chains = expand_unit(conn, {"unit_id": "probe", "entity_name": args.entity}, CONFIG)
    if not chains:
        log(f"no chains found for {args.entity!r}")
        return 1
    print(json.dumps(chains, indent=2, ensure_ascii=False))
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# --------------------------------------------------------------------------------------
# Chains as evidence
# --------------------------------------------------------------------------------------

RELATION_PHRASING = {
    "DIRECTS":        ("directs", "is directed by"),
    "TEACHES":        ("teaches", "is taught by"),
    "RESEARCHES":     ("researches", "is researched by"),
    "OFFERED_BY":     ("is offered by", "offers"),
    "PART_OF":        ("is part of", "includes"),
    "HOSTS":          ("hosts", "is hosted by"),
    "FUNDS":          ("funds", "is funded by"),
    "PARTNERS_WITH":  ("partners with", "partners with"),
    "MEMBER_OF":      ("is a member of", "has as a member"),
    "AFFILIATED_WITH": ("is affiliated with", "is affiliated with"),
    "AWARDED":        ("was awarded", "was awarded to"),
    "LOCATED_IN":     ("is located in", "contains"),
}


def _sentence(chain: dict[str, Any]) -> str:
    """State the relation as one plain sentence, in the direction the edge runs."""
    forward, reverse = RELATION_PHRASING.get(
        chain["relation"], (chain["relation"].lower().replace("_", " "),) * 2
    )
    anchor, other = chain.get("anchor") or "", chain.get("entity") or ""
    if chain.get("direction") == "in":
        # the edge points AT the anchor: the neighbour is the subject
        return f"{other} {forward} {anchor}."
    return f"{anchor} {forward} {other}."


def chains_to_units(chains_by_unit: dict[str, list[dict[str, Any]]],
                    college_id: str = "", cfg: dict[str, Any] | None = None
                    ) -> list[dict[str, Any]]:
    """Turn graph chains into ordinary EvidenceUnits, so the rest of the pipeline can use them.

    A chain handed to the writer as loose context would be unusable: the verifier checks every
    sentence against the units it cites, so a sentence built from a relation nobody can cite
    gets deleted. Making each relation a unit -- with its own id, its verbatim edge quote and
    its source URL -- means a chain-derived claim is checked exactly like a fact-derived one,
    and a student can follow it back to the page that states it.

    Units are deduplicated across the whole run: the same professor-directs-lab edge reached
    from three different facts is one unit, cited three times.
    """
    cfg = cfg or CONFIG
    import hashlib

    out: dict[str, dict[str, Any]] = {}
    prefix = f"{college_id}-" if college_id else ""
    for anchor_unit_id, chains in (chains_by_unit or {}).items():
        for chain in chains:
            evidence = (chain.get("evidence") or "").strip()
            url = (chain.get("source_url") or "").strip()
            if not evidence or not url:
                # without a quote and a source it cannot be verified, so it cannot ship
                continue
            text = _sentence(chain)
            key = hashlib.sha256(f"{text}|{url}".encode("utf-8")).hexdigest()[:12]
            uid = f"{prefix}rel-{key}"
            if uid in out:
                out[uid]["anchored_by"].append(anchor_unit_id)
                continue
            out[uid] = {
                "unit_id": uid,
                "kind": "relation",
                "category_code": None,
                "text": text,
                "quote": evidence,
                "entity_name": chain.get("entity"),
                "source_url": url,
                "source_title": None,
                "source_kind": "official",
                "year": None,
                "relation_detail": {
                    "relation": chain.get("relation"),
                    "anchor": chain.get("anchor"),
                    "entity": chain.get("entity"),
                    "entity_type": chain.get("entity_type"),
                },
                "anchored_by": [anchor_unit_id],
            }
    return list(out.values())


def expand_evidence(index_dir: Path, buckets: dict[str, list[dict[str, Any]]],
                    college_id: str = "", cfg: dict[str, Any] | None = None
                    ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Add relation units to each category's evidence. Returns (buckets, stats).

    A category gets the chains hanging off its OWN units, so a relation only appears where
    the thing it relates was already judged relevant. No category gains evidence about
    something it never retrieved.
    """
    cfg = cfg or CONFIG
    db = Path(index_dir) / "chunks.sqlite"
    if not db.is_file():
        return buckets, {"available": False, "reason": f"{db} not found"}
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        if not has_graph(conn):
            return buckets, {"available": False, "reason": "index has no graph tables"}
        out: dict[str, list[dict[str, Any]]] = {}
        added_total = 0
        per_cat: dict[str, int] = {}
        for code, units in buckets.items():
            if not isinstance(units, list):
                out[code] = units
                continue
            chains = expand_units(conn, units, cfg)
            rel_units = chains_to_units(chains, college_id, cfg)
            have = {u.get("unit_id") for u in units}
            fresh = [u for u in rel_units if u["unit_id"] not in have]

            # Interleave: each relation goes immediately after the unit it hangs off, not in
            # a block at the end. Appended at the end they were never cited once in 62
            # citations -- the writer reads the rich facts first, fills its slots, and never
            # reaches a terse one-line relation. Next to its anchor, the chain reads as part
            # of that thing rather than as a separate, weaker fact.
            by_anchor: dict[str, list[dict[str, Any]]] = {}
            for rel in fresh:
                for anchor_id in rel.get("anchored_by") or []:
                    by_anchor.setdefault(anchor_id, []).append(rel)
            placed: set[str] = set()
            ordered: list[dict[str, Any]] = []
            for unit in units:
                ordered.append(unit)
                for rel in by_anchor.get(unit.get("unit_id"), []):
                    if rel["unit_id"] not in placed:
                        placed.add(rel["unit_id"])
                        ordered.append(rel)
            ordered.extend(r for r in fresh if r["unit_id"] not in placed)

            out[code] = ordered
            per_cat[code] = len(fresh)
            added_total += len(fresh)
        return out, {"available": True, "relations_added": added_total, "by_category": per_cat}
    finally:
        conn.close()
