"""Build the local hybrid-search index for one college's verified fact bundle.

Reads the four bundle files (facts, page chunks, organizations, courses) and writes a
self-contained index directory that the retrieval module can open read-only:

    <index-dir>/<college_id>/chunks.sqlite   units table (full metadata) + units_fts (FTS5 keyword search)
    <index-dir>/<college_id>/vectors.npy     float32 [n_units, dims], L2-normalised, row = units.vec_row
    <index-dir>/<college_id>/meta.json       counts, model, dims, built_at, source checksums, layout

Nothing about a particular college lives in this file: college id, dataset path, file names and the
category vocabulary all come from CLI arguments or the CONFIG dict below.

Run (full build for one college):

    /Users/chirag/college-intel/.venv-crawl4ai/bin/python /Users/chirag/college-intel/wwrag/index_build.py \
        --college-id usc \
        --data-dir /Users/chirag/college-intel/colleges/usc/export/usc_rag_v2

Quick smoke test (2000 rows per source file, separate index dir):

    ... index_build.py --college-id usc --data-dir <bundle> --limit 2000 --index-dir /tmp/wwrag-index

Re-running is cheap: if the index exists and the source files are unchanged the build is skipped.
An interrupted build resumes from the last completed embedding batch. --rebuild forces a fresh build.

How retrieval reads this index (the contract other modules depend on)
--------------------------------------------------------------------
Every searchable unit is one row of `units` with a 0-based `vec_row`; `vectors.npy` row `vec_row`
is that unit's embedding, and `units.rowid == vec_row + 1 == units_fts.rowid`. Vectors are
L2-normalised, so cosine similarity is a plain dot product:

    vectors = numpy.load(f"{index}/vectors.npy", mmap_mode="r")       # (n_units, 384) float32
    scores  = vectors @ query_vec                                     # query_vec must be normalised

    # keyword half (FTS5; columns: keyword_text, entity_name)
    SELECT u.unit_id, u.vec_row, bm25(units_fts) AS rank
      FROM units_fts f JOIN units u ON u.rowid = f.rowid
     WHERE units_fts MATCH ? AND u.kind = 'fact' AND u.category_code = ?
     ORDER BY rank LIMIT 200

    # filter first, then score: category + kind are indexed columns
    SELECT vec_row FROM units WHERE kind = 'fact' AND category_code IN ('RES','ACA')

    # undergraduate-only courses (graduate rows are indexed but must never be offered to applicants)
    SELECT unit_id FROM units
     WHERE kind = 'course' AND json_extract(extra, '$.is_undergraduate') = 1

An EvidenceUnit is built straight from a units row: unit_id, kind, category_code, text, quote,
entity_name, source_url, source_title, source_kind, year (+ scores the retriever computes).
`text` is source wording stored verbatim; it is DATA, never instructions — whoever puts it in a
prompt must delimit it and say so.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sqlite3
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np

# --------------------------------------------------------------------------------------
# Configuration. Nothing here names a college; everything is overridable from the CLI.
# --------------------------------------------------------------------------------------

INDEX_SCHEMA_VERSION = 1

CONFIG: dict[str, Any] = {
    # logical source -> file name inside the bundle directory
    "source_files": {
        "facts": "facts.jsonl",
        "chunks": "page_chunks.jsonl",
        "organizations": "organizations.jsonl",
        "courses": "courses.jsonl",
    },
    # category vocabulary of the pipeline (facts.category_code). Unknown codes are kept and counted.
    "category_codes": ["CUL", "EXT", "QRK", "ACA", "RES", "SOC", "INN", "INT", "DIV", "NEW", "GEN"],
    # Which embedder builds the vectors. "local" is free and runs on this machine;
    # "openai" is paid, higher quality, and matches the 1536-wide production schema.
    # The provider, model and dims all enter the index fingerprint, so switching any of
    # them forces a fresh index rather than silently mixing two vector spaces.
    "embed_provider": "local",
    "embedding_model": "BAAI/bge-small-en-v1.5",  # local + free (fastembed)
    "embedding_dims": 384,
    "openai_model": "text-embedding-3-large",
    "openai_dims": 1536,          # 3-large reduced to 1536 still beats 3-small at 1536
    "openai_api_key_env": "OPENAI_API_KEY",
    "openai_base_url": "https://api.openai.com/v1",
    "openai_texts_per_request": 128,
    "openai_concurrency": 8,
    "openai_max_retries": 6,
    "openai_timeout_s": 120.0,
    "max_embed_chars": 4000,   # model window is 512 tokens (~2k chars); trim input, keep full text stored
    "embed_batch_size": 128,   # texts per fastembed forward batch
    "db_read_batch": 1024,     # rows pulled from sqlite per embedding round trip (= resume granularity)
    "progress_every": 2000,    # print a progress line every N units
    "insert_batch": 2000,      # rows per executemany
    "fts_tokenizer": "porter unicode61 remove_diacritics 2",
}

UNIT_KINDS = ("fact", "chunk", "org", "course")

class OpenAIEmbedder:
    """Paid embeddings behind fastembed's interface: .embed(texts, batch_size=...) -> vectors.

    Presenting the same shape as TextEmbedding means the embedding loop, its checkpointing
    and its memmap writes stay exactly as they are; only the vector source changes.

    Requests go out concurrently because this stage is network-bound, not CPU-bound --
    one request at a time is what makes a paid embedding run take hours.
    """

    def __init__(self, model: str, dims: int, api_key: str, base_url: str,
                 texts_per_request: int, concurrency: int, max_retries: int, timeout_s: float):
        import httpx  # imported late; the local provider needs no HTTP client

        self.model = model
        self.dims = int(dims)
        self.texts_per_request = int(texts_per_request)
        self.concurrency = int(concurrency)
        self.max_retries = int(max_retries)
        self._url = base_url.rstrip("/") + "/embeddings"
        self._headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self._client = httpx.Client(timeout=timeout_s, http2=False)
        self.tokens_used = 0
        self._lock = threading.Lock()

    def close(self) -> None:
        self._client.close()

    def _one_request(self, texts: list[str]) -> list[list[float]]:
        # The API rejects empty input; a unit with no text cannot reach here, but a blank
        # line would fail the whole batch, so substitute a space rather than lose 128 rows.
        payload = {
            "model": self.model,
            "input": [t if t.strip() else " " for t in texts],
            "dimensions": self.dims,
        }
        delay = 2.0
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.post(self._url, headers=self._headers, json=payload)
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                resp.raise_for_status()
                body = resp.json()
                rows = sorted(body["data"], key=lambda d: d["index"])
                if len(rows) != len(texts):
                    raise RuntimeError(f"asked for {len(texts)} vectors, got {len(rows)}")
                with self._lock:
                    self.tokens_used += int((body.get("usage") or {}).get("total_tokens") or 0)
                return [r["embedding"] for r in rows]
            except Exception as exc:  # noqa: BLE001 - retried below, re-raised when exhausted
                last = exc
                if attempt == self.max_retries - 1:
                    break
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
        raise RuntimeError(f"embedding request failed after {self.max_retries} attempts: {last}")

    def embed(self, texts, batch_size: int | None = None):
        texts = list(texts)
        chunks = [texts[i : i + self.texts_per_request]
                  for i in range(0, len(texts), self.texts_per_request)]
        out: list[list[float] | None] = [None] * len(chunks)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = {pool.submit(self._one_request, c): i for i, c in enumerate(chunks)}
            for fut in concurrent.futures.as_completed(futures):
                out[futures[fut]] = fut.result()   # a failure here must stop the build
        vectors: list[list[float]] = []
        for part in out:
            vectors.extend(part or [])
        # Re-normalise: shortening 3-large to fewer dimensions leaves vectors slightly off unit
        # length, and meta.json promises retrieval that a dot product IS the cosine.
        arr = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return arr / norms


def effective_model(cfg: dict[str, Any]) -> tuple[str, int]:
    """The (model, dims) actually used, for the configured provider.

    The fingerprint and meta.json must both read THIS, never the raw local keys: otherwise
    switching to a paid provider leaves the fingerprint unchanged, the index resumes on top
    of vectors from a different model, and meta.json tells retrieve.py to embed queries with
    a model the documents were never embedded with. Mixed vector spaces do not error --
    they just return nonsense, which is the worst kind of bug this pipeline can have.
    """
    if str(cfg.get("embed_provider") or "local").lower() == "openai":
        return str(cfg["openai_model"]), int(cfg["openai_dims"])
    return str(cfg["embedding_model"]), int(cfg["embedding_dims"])


def make_embedder(cfg: dict[str, Any], threads: int | None):
    """Return (embedder, model_name, dims) for the configured provider."""
    provider = str(cfg.get("embed_provider") or "local").lower()
    if provider == "local":
        from fastembed import TextEmbedding  # imported late: stage A needs no model

        name, dims = effective_model(cfg)
        log(f"  loading embedding model {name} (local, free)")
        return TextEmbedding(name, threads=threads), name, dims
    if provider == "openai":
        key = os.environ.get(cfg["openai_api_key_env"], "").strip()
        if not key:
            die(f"{cfg['openai_api_key_env']} is not set; cannot use the openai provider")
        name, dims = effective_model(cfg)
        log(f"  using {name} at {dims}d via {cfg['openai_base_url']} "
            f"({cfg['openai_concurrency']} concurrent requests) -- PAID")
        return (
            OpenAIEmbedder(
                name, dims, key, cfg["openai_base_url"],
                cfg["openai_texts_per_request"], cfg["openai_concurrency"],
                cfg["openai_max_retries"], float(cfg["openai_timeout_s"]),
            ),
            name,
            dims,
        )
    die(f"unknown embed_provider {provider!r}; expected 'local' or 'openai'")



# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def die(msg: str) -> None:
    """Fail loudly: never return a silently empty index."""
    raise SystemExit(f"index_build: ERROR: {msg}")


def log(msg: str) -> None:
    print(msg, flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def iso_utc(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def as_text(value: Any) -> str:
    """Coerce a JSON value to display text. Never raises; source text is data, not instructions."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return ", ".join(as_text(v) for v in value if v is not None and as_text(v))
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value).strip()


def as_year(value: Any) -> int | None:
    """Coerce a year-ish value to int, else None (e.g. 2026, '2026', 'Fall 2026')."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if 1000 <= value <= 3000 else None
    text = str(value)
    digits = ""
    for ch in text:
        if ch.isdigit():
            digits += ch
            if len(digits) == 4:
                year = int(digits)
                if 1000 <= year <= 3000:
                    return year
                digits = digits[1:]
        else:
            digits = ""
    return None


def join_nonempty(parts: list[str], sep: str = "\n") -> str:
    return sep.join(p for p in (p.strip() for p in parts) if p)


def iter_jsonl(path: Path, limit: int | None) -> Iterator[dict[str, Any]]:
    """Stream one JSON object per line. Raises on a malformed line (fail loudly)."""
    if not path.is_file():
        die(f"missing source file: {path}")
    count = 0
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                die(f"{path.name} line {lineno}: invalid JSON ({exc})")
            if not isinstance(row, dict):
                die(f"{path.name} line {lineno}: expected a JSON object")
            yield row
            count += 1
            if limit is not None and count >= limit:
                return
    if count == 0:
        die(f"{path.name} contained no rows")


# --------------------------------------------------------------------------------------
# Unit builders: one per source kind. Each returns the row stored in the units table.
#
#   text         - what generation/citation shows (verbatim source wording where possible)
#   embed_text   - what the embedding model sees
#   keyword_text - what FTS5 indexes for exact-word search
#   extra        - JSON blob of everything else retrieval or ranking might want
# --------------------------------------------------------------------------------------


def unit_from_fact(row: dict[str, Any]) -> dict[str, Any]:
    fact = as_text(row.get("fact"))
    if not fact:
        die(f"fact {row.get('id')!r} has no 'fact' text")
    entity = as_text(row.get("entity_name"))
    quote = as_text(row.get("evidence_quote")) or None
    return {
        "unit_id": as_text(row.get("id")),
        "kind": "fact",
        "category_code": as_text(row.get("category_code")) or None,
        "text": fact,
        "quote": quote,
        "entity_name": entity or None,
        "source_url": as_text(row.get("source_url")),
        "source_title": as_text(row.get("source_title")) or None,
        "source_kind": as_text(row.get("source_kind")) or "official",
        "year": as_year(row.get("year")) or as_year(row.get("period")) or as_year(row.get("page_published")),
        "embed_text": f"{entity}: {fact}" if entity else fact,
        "keyword_text": join_nonempty([fact, quote or "", entity, as_text(row.get("entity_type"))], " "),
        "extra": {
            "category": as_text(row.get("category")) or None,
            "category_raw": row.get("category_raw"),
            "entity_type": row.get("entity_type"),
            "relations": row.get("relations") or [],
            "period": row.get("period"),
            "source_urls": row.get("source_urls") or ([row.get("source_url")] if row.get("source_url") else []),
            "source_count": row.get("source_count"),
            "source_host": row.get("source_host"),
            "page_published": row.get("page_published"),
            "retrieved_at": row.get("retrieved_at"),
            "verification": row.get("verification"),
        },
    }


def unit_from_chunk(row: dict[str, Any]) -> dict[str, Any]:
    text = as_text(row.get("text"))
    if not text:
        die(f"page chunk {row.get('id')!r} has no 'text'")
    title = as_text(row.get("title"))
    body = join_nonempty([title, text])
    return {
        "unit_id": as_text(row.get("id")),
        "kind": "chunk",
        "category_code": None,  # page chunks are not category-tagged in the bundle
        "text": text,
        "quote": None,
        "entity_name": None,
        "source_url": as_text(row.get("url")),
        "source_title": title or None,
        "source_kind": as_text(row.get("source_kind")) or "official",
        "year": as_year(row.get("page_published")),
        "embed_text": body,
        "keyword_text": body,
        "extra": {
            "page_id": row.get("page_id"),
            "chunk_index": row.get("chunk_index"),
            "chunk_count": row.get("chunk_count"),
            "retrieved_at": row.get("retrieved_at"),
        },
    }


def unit_from_org(row: dict[str, Any]) -> dict[str, Any]:
    name = as_text(row.get("name"))
    if not name:
        die(f"organization {row.get('id')!r} has no 'name'")
    group_type = as_text(row.get("group_type"))
    categories = as_text(row.get("categories"))
    mission = as_text(row.get("mission"))
    benefits = as_text(row.get("membership_benefits"))
    duration = as_text(row.get("membership_duration"))
    header = f"{name} ({group_type})" if group_type else name
    text = join_nonempty(
        [
            header,
            f"Categories: {categories}" if categories else "",
            f"Mission: {mission}" if mission else "",
            f"Membership benefits: {benefits}" if benefits else "",
        ]
    )
    return {
        "unit_id": as_text(row.get("id")),
        "kind": "org",
        "category_code": None,  # orgs serve several categories; use extra.fit_scores to steer
        "text": text,
        "quote": mission or None,
        "entity_name": name,
        "source_url": as_text(row.get("profile_url")) or as_text(row.get("source_url")),
        "source_title": name,
        "source_kind": "official",  # the bundle builds this from the official directory
        "year": None,
        "embed_text": join_nonempty([header, categories, mission], ". "),
        "keyword_text": join_nonempty([header, categories, mission, benefits, duration], " "),
        "extra": {
            "group_type": group_type or None,
            "categories": row.get("categories") or [],
            "mission": mission or None,
            "membership_benefits": benefits or None,
            "membership_duration": duration or None,
            "fit_scores": row.get("fit_scores") or {},
            "profile_url": row.get("profile_url"),
            "directory_url": row.get("source_url"),
            "retrieved_at": row.get("retrieved_at"),
        },
    }


def unit_from_course(row: dict[str, Any]) -> dict[str, Any]:
    code = as_text(row.get("code"))
    title = as_text(row.get("title"))
    if not code and not title:
        die(f"course {row.get('id')!r} has neither 'code' nor 'title'")
    description = as_text(row.get("description"))
    units_txt = as_text(row.get("units"))
    ge = as_text(row.get("general_education"))
    prereq = as_text(row.get("prerequisites_text"))
    instructors = as_text(row.get("instructors"))
    header = f"{code}: {title}" if code and title else (code or title)
    if units_txt:
        header = f"{header} ({units_txt})"
    is_ug = bool(row.get("is_undergraduate"))
    return {
        "unit_id": as_text(row.get("id")),
        "kind": "course",
        "category_code": None,
        "text": join_nonempty([header, description]),
        "quote": description or None,
        "entity_name": code or title,
        "source_url": as_text(row.get("source_url")),
        "source_title": title or code,
        "source_kind": "official",  # the bundle builds this from the official schedule of classes
        "year": as_year(row.get("term")),
        "embed_text": join_nonempty([f"{code} {title}".strip(), description], ". "),
        "keyword_text": join_nonempty(
            [code, title, description, as_text(row.get("dept")), ge, prereq, instructors], " "
        ),
        "extra": {
            "is_undergraduate": is_ug,
            "term": row.get("term"),
            "dept": row.get("dept"),
            "number": row.get("number"),
            "units": units_txt or None,
            "general_education": ge or None,
            "prerequisites_text": prereq or None,
            "corequisites_text": row.get("corequisites_text"),
            "recommended_prep": row.get("recommended_prep"),
            "cross_listed": row.get("cross_listed"),
            "instructors": row.get("instructors") or [],
            "section_count": len(row.get("sections") or []),
            "sections": row.get("sections") or [],
            "school_code": row.get("school_code"),
            "data_origin": row.get("data_origin"),
            "retrieved_at": row.get("retrieved_at"),
        },
    }


BUILDERS = {
    "facts": unit_from_fact,
    "chunks": unit_from_chunk,
    "organizations": unit_from_org,
    "courses": unit_from_course,
}


# --------------------------------------------------------------------------------------
# SQLite
# --------------------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE units (
    vec_row       INTEGER NOT NULL,          -- 0-based row in vectors.npy (== rowid - 1)
    unit_id       TEXT NOT NULL UNIQUE,
    kind          TEXT NOT NULL,             -- fact | chunk | org | course
    category_code TEXT,                      -- facts only; NULL elsewhere
    text          TEXT NOT NULL,             -- what the report may quote / paraphrase
    quote         TEXT,                      -- verbatim evidence quote where the source has one
    entity_name   TEXT,
    source_url    TEXT NOT NULL,
    source_title  TEXT,
    source_kind   TEXT NOT NULL,             -- official | affiliated | external
    year          INTEGER,
    embed_text    TEXT NOT NULL,             -- exact string that was embedded (kept for resume/debug)
    extra         TEXT NOT NULL              -- JSON, sorted keys
);
CREATE INDEX idx_units_kind ON units(kind);
CREATE INDEX idx_units_kind_category ON units(kind, category_code);
CREATE INDEX idx_units_vec_row ON units(vec_row);
CREATE INDEX idx_units_source_kind ON units(source_kind);
"""


def open_db(path: Path, write: bool) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    if write:
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA cache_size=-200000")  # ~200 MB page cache during build
    return conn


def build_units(
    data_dir: Path,
    college_id: str,
    cfg: dict[str, Any],
    limit: int | None,
    db_path: Path,
) -> dict[str, Any]:
    """Stage A: stream the four source files into chunks.sqlite. Returns build stats."""
    if db_path.exists():
        db_path.unlink()
    conn = open_db(db_path, write=True)
    conn.executescript(SCHEMA)
    conn.execute(
        f"CREATE VIRTUAL TABLE units_fts USING fts5("
        f"keyword_text, entity_name, tokenize='{cfg['fts_tokenizer']}')"
    )

    counts_by_kind: dict[str, int] = {k: 0 for k in UNIT_KINDS}
    counts_by_category: dict[str, int] = {}
    unknown_categories: dict[str, int] = {}
    rows_read: dict[str, int] = {}
    undergraduate_courses = 0
    seen_ids: set[str] = set()
    known_codes = set(cfg["category_codes"])
    max_chars = int(cfg["max_embed_chars"])
    insert_batch = int(cfg["insert_batch"])
    progress_every = int(cfg["progress_every"])

    unit_rows: list[tuple] = []
    fts_rows: list[tuple] = []
    vec_row = 0
    started = time.time()

    def flush() -> None:
        if not unit_rows:
            return
        conn.executemany(
            "INSERT INTO units(rowid, vec_row, unit_id, kind, category_code, text, quote, entity_name,"
            " source_url, source_title, source_kind, year, embed_text, extra)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            unit_rows,
        )
        conn.executemany("INSERT INTO units_fts(rowid, keyword_text, entity_name) VALUES (?,?,?)", fts_rows)
        unit_rows.clear()
        fts_rows.clear()

    for source_name, file_name in cfg["source_files"].items():
        path = data_dir / file_name
        builder = BUILDERS[source_name]
        read = 0
        for row in iter_jsonl(path, limit):
            read += 1
            row_college = as_text(row.get("college_id"))
            if row_college and row_college != college_id:
                die(
                    f"{file_name}: row {row.get('id')!r} has college_id {row_college!r},"
                    f" expected {college_id!r} — wrong dataset for --college-id"
                )
            unit = builder(row)
            unit_id = unit["unit_id"]
            if not unit_id:
                die(f"{file_name}: a row has no 'id'; citations need stable unit ids")
            if unit_id in seen_ids:
                die(f"{file_name}: duplicate unit id {unit_id!r}; citations would be ambiguous")
            seen_ids.add(unit_id)
            if not unit["source_url"]:
                die(f"{file_name}: unit {unit_id!r} has no source url; every claim must be citable")

            code = unit["category_code"]
            if code:
                counts_by_category[code] = counts_by_category.get(code, 0) + 1
                if code not in known_codes:
                    unknown_categories[code] = unknown_categories.get(code, 0) + 1
            if unit["kind"] == "course" and unit["extra"].get("is_undergraduate"):
                undergraduate_courses += 1

            embed_text = unit["embed_text"].strip()[:max_chars]
            if not embed_text:
                die(f"{file_name}: unit {unit_id!r} produced empty embedding text")

            unit_rows.append(
                (
                    vec_row + 1,  # rowid
                    vec_row,
                    unit_id,
                    unit["kind"],
                    code,
                    unit["text"],
                    unit["quote"],
                    unit["entity_name"],
                    unit["source_url"],
                    unit["source_title"],
                    unit["source_kind"],
                    unit["year"],
                    embed_text,
                    json.dumps(unit["extra"], sort_keys=True, ensure_ascii=False),
                )
            )
            fts_rows.append((vec_row + 1, unit["keyword_text"], unit["entity_name"] or ""))
            counts_by_kind[unit["kind"]] += 1
            vec_row += 1
            if len(unit_rows) >= insert_batch:
                flush()
            if vec_row % progress_every == 0:
                elapsed = time.time() - started
                log(f"  units {vec_row:>7,}  ({source_name})  {vec_row / max(elapsed, 1e-6):,.0f}/s")
        rows_read[file_name] = read
        flush()
        conn.commit()
        log(f"  read {file_name}: {read:,} rows")

    flush()
    conn.commit()
    conn.execute("PRAGMA optimize")
    conn.commit()
    conn.close()

    if vec_row == 0:
        die("no units were indexed; refusing to write an empty index")

    return {
        "total": vec_row,
        "by_kind": counts_by_kind,
        "by_category_code": dict(sorted(counts_by_category.items())),
        "unknown_category_codes": dict(sorted(unknown_categories.items())),
        "undergraduate_courses": undergraduate_courses,
        "rows_read": rows_read,
        "seconds": round(time.time() - started, 1),
    }


# --------------------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------------------


def embed_units(
    db_path: Path,
    vec_tmp: Path,
    n_units: int,
    cfg: dict[str, Any],
    start_row: int,
    threads: int | None,
    state_path: Path,
    state: dict[str, Any],
) -> float:
    """Stage B: fill vectors.npy.building from units.embed_text, resumable at `start_row`."""
    model, model_name, dims = make_embedder(cfg, threads)

    if start_row > 0 and not vec_tmp.exists():
        log("  partial vector file is gone; restarting the embedding stage from row 0")
        start_row = 0
    mode = "r+" if start_row > 0 else "w+"
    vectors = np.lib.format.open_memmap(vec_tmp, mode=mode, dtype=np.float32, shape=(n_units, dims))
    if vectors.shape != (n_units, dims):
        die(f"vector file {vec_tmp} has shape {vectors.shape}, expected {(n_units, dims)}")

    conn = open_db(db_path, write=False)
    read_batch = int(cfg["db_read_batch"])
    embed_batch = int(cfg["embed_batch_size"])
    progress_every = int(cfg["progress_every"])
    done = start_row
    started = time.time()
    last_progress = done

    while done < n_units:
        rows = conn.execute(
            "SELECT vec_row, embed_text FROM units WHERE vec_row >= ? ORDER BY vec_row LIMIT ?",
            (done, read_batch),
        ).fetchall()
        if not rows:
            die(f"units table ended at row {done} but {n_units} units were expected")
        first_row = rows[0][0]
        if first_row != done:
            die(f"expected vec_row {done}, got {first_row} — index is inconsistent")
        # Sort the batch by text length before embedding: the model pads every batch to its longest
        # member, so mixing a 10-token fact with a 500-token page chunk wastes most of the compute.
        # (length, position) keeps it deterministic; results are scattered back to source order.
        order = sorted(range(len(rows)), key=lambda i: (len(rows[i][1]), i))
        texts = [rows[i][1] for i in order]
        embedded = np.asarray(list(model.embed(texts, batch_size=embed_batch)), dtype=np.float32)
        if embedded.shape != (len(rows), dims):
            die(f"model returned {embedded.shape}, expected {(len(rows), dims)} — wrong model or dims")
        batch = np.empty((len(rows), dims), dtype=np.float32)
        batch[np.asarray(order, dtype=np.int64)] = embedded
        vectors[done : done + len(rows)] = batch
        done += len(rows)
        vectors.flush()
        state["embedded_rows"] = done
        write_state(state_path, state)
        if done - last_progress >= progress_every or done == n_units:
            elapsed = time.time() - started
            rate = (done - start_row) / max(elapsed, 1e-6)
            remaining = (n_units - done) / rate if rate > 0 else 0.0
            log(
                f"  embedded {done:>7,}/{n_units:,}  {rate:,.0f}/s  "
                f"elapsed {elapsed / 60:,.1f}m  eta {remaining / 60:,.1f}m"
            )
            last_progress = done

    conn.close()
    del vectors
    return round(time.time() - started, 1)


# --------------------------------------------------------------------------------------
# Build state / fingerprint
# --------------------------------------------------------------------------------------


def source_fingerprint(data_dir: Path, cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for file_name in sorted(cfg["source_files"].values()):
        path = data_dir / file_name
        if not path.is_file():
            die(f"missing source file: {path}")
        out[file_name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return out


def build_fingerprint(college_id: str, sources: dict[str, Any], cfg: dict[str, Any], limit: int | None) -> dict[str, Any]:
    return {
        "index_schema_version": INDEX_SCHEMA_VERSION,
        "college_id": college_id,
        "sources": sources,
        "embed_provider": str(cfg.get("embed_provider") or "local"),
        "embedding_model": effective_model(cfg)[0],
        "embedding_dims": effective_model(cfg)[1],
        "max_embed_chars": cfg["max_embed_chars"],
        "fts_tokenizer": cfg["fts_tokenizer"],
        "category_codes": list(cfg["category_codes"]),
        "limit": limit,
    }


def write_state(path: Path, state: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------------------
# Main build
# --------------------------------------------------------------------------------------


def build_index(
    college_id: str,
    data_dir: Path,
    index_root: Path,
    cfg: dict[str, Any],
    limit: int | None,
    rebuild: bool,
    threads: int | None,
    built_at_epoch: float,
    skip_vectors: bool,
) -> dict[str, Any]:
    if not data_dir.is_dir():
        die(f"data dir does not exist: {data_dir}")
    index_dir = index_root / college_id
    index_dir.mkdir(parents=True, exist_ok=True)
    db_path = index_dir / "chunks.sqlite"
    db_tmp = index_dir / "chunks.sqlite.building"
    vec_path = index_dir / "vectors.npy"
    vec_tmp = index_dir / "vectors.npy.building"
    meta_path = index_dir / "meta.json"
    state_path = index_dir / "build_state.json"

    total_started = time.time()
    log(f"index_build: college={college_id} data={data_dir}")
    log("  checksumming source files")
    sources = source_fingerprint(data_dir, cfg)
    fingerprint = build_fingerprint(college_id, sources, cfg, limit)

    meta = read_json(meta_path)
    if (
        not rebuild
        and meta
        and meta.get("complete")
        and meta.get("fingerprint") == fingerprint
        and db_path.is_file()
        and vec_path.is_file()
    ):
        log("  index is up to date (sources unchanged) — nothing to do. Use --rebuild to force.")
        return meta

    state = read_json(state_path) or {}
    resumable = (
        not rebuild
        and state.get("fingerprint") == fingerprint
        and state.get("stage") == "vectors"
        and db_path.is_file()
        and isinstance(state.get("units_total"), int)
        and isinstance(state.get("embedded_rows"), int)
        # either partial vectors are on disk, or nothing was embedded yet (units stage is still valid)
        and (vec_tmp.is_file() or int(state["embedded_rows"]) == 0)
    )

    if resumable:
        n_units = int(state["units_total"])
        start_row = min(int(state["embedded_rows"]), n_units)
        unit_stats = state.get("unit_stats") or {}
        log(f"  resuming: units already built ({n_units:,}); embeddings done {start_row:,}")
    else:
        for stale in (db_tmp, vec_tmp, vec_path, meta_path):
            if stale.exists():
                stale.unlink()
        log("  stage 1/2: building units + FTS5 index")
        unit_stats = build_units(data_dir, college_id, cfg, limit, db_tmp)
        os.replace(db_tmp, db_path)
        n_units = int(unit_stats["total"])
        start_row = 0
        state = {
            "fingerprint": fingerprint,
            "stage": "vectors",
            "units_total": n_units,
            "embedded_rows": 0,
            "unit_stats": unit_stats,
        }
        write_state(state_path, state)
        log(
            f"  units: {n_units:,} in {unit_stats['seconds']}s "
            f"({db_path.stat().st_size / 1e6:,.0f} MB sqlite)"
        )

    embed_seconds = 0.0
    if skip_vectors:
        log("  stage 2/2: SKIPPED (--skip-vectors); rerun without the flag to finish the index")
    else:
        log(f"  stage 2/2: embedding {n_units - start_row:,} units")
        embed_seconds = embed_units(
            db_path, vec_tmp, n_units, cfg, start_row, threads, state_path, state
        )
        os.replace(vec_tmp, vec_path)
        state["stage"] = "done"
        write_state(state_path, state)

    meta = {
        "college_id": college_id,
        "complete": not skip_vectors,
        "index_schema_version": INDEX_SCHEMA_VERSION,
        "built_at": iso_utc(built_at_epoch),
        "built_at_epoch": int(built_at_epoch),
        "data_dir": str(data_dir),
        "limit": limit,
        "embedding": {
            "model": effective_model(cfg)[0],
            # must name the provider that actually produced these vectors: retrieve.py reads this
            # block to decide how to embed the query, and a wrong label means a mixed vector space
            "provider": (
                "fastembed (local, free)"
                if str(cfg.get("embed_provider") or "local").lower() == "local"
                else f"{str(cfg['embed_provider']).lower()} api ({cfg['openai_base_url']}) — PAID"
            ),
            "dims": effective_model(cfg)[1],
            "dtype": "float32",
            "normalized": True,
            "max_embed_chars": cfg["max_embed_chars"],
            "batch_size": cfg["embed_batch_size"],
        },
        "counts": {
            "units": n_units,
            "by_kind": unit_stats.get("by_kind", {}),
            "by_category_code": unit_stats.get("by_category_code", {}),
            "unknown_category_codes": unit_stats.get("unknown_category_codes", {}),
            "undergraduate_courses": unit_stats.get("undergraduate_courses"),
            "rows_read": unit_stats.get("rows_read", {}),
        },
        "layout": {
            "sqlite": "chunks.sqlite",
            "vectors": "vectors.npy",
            "units_table": "units",
            "vector_row_column": "vec_row",
            "vector_row_is_rowid_minus_one": True,
            "fts_table": "units_fts",
            "fts_columns": ["keyword_text", "entity_name"],
            "fts_rowid_equals_units_rowid": True,
            "fts_tokenizer": cfg["fts_tokenizer"],
            "similarity": "cosine == dot product (rows are L2-normalised)",
        },
        "category_codes": list(cfg["category_codes"]),
        "sources": sources,
        "fingerprint": fingerprint,
        "timings_seconds": {
            "units": unit_stats.get("seconds"),
            "embedding": embed_seconds,
            "total": round(time.time() - total_started, 1),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=1, sort_keys=True), encoding="utf-8")
    return meta


def verify_index(index_root: Path, college_id: str) -> dict[str, Any]:
    """Reopen the finished index and sanity-check it (shapes, counts, one FTS query)."""
    index_dir = index_root / college_id
    db_path = index_dir / "chunks.sqlite"
    vec_path = index_dir / "vectors.npy"
    conn = open_db(db_path, write=False)
    n_units = conn.execute("SELECT COUNT(*) FROM units").fetchone()[0]
    n_fts = conn.execute("SELECT COUNT(*) FROM units_fts").fetchone()[0]
    by_kind = dict(conn.execute("SELECT kind, COUNT(*) FROM units GROUP BY kind ORDER BY kind").fetchall())
    conn.close()
    out = {"units": n_units, "fts_rows": n_fts, "by_kind": by_kind}
    if vec_path.is_file():
        vectors = np.load(vec_path, mmap_mode="r")
        out["vectors_shape"] = list(vectors.shape)
        out["vectors_dtype"] = str(vectors.dtype)
        sample = np.asarray(vectors[: min(1000, vectors.shape[0])], dtype=np.float32)
        norms = np.linalg.norm(sample, axis=1)
        out["vector_norm_min"] = round(float(norms.min()), 4)
        out["vector_norm_max"] = round(float(norms.max()), 4)
        if n_units != vectors.shape[0]:
            die(f"units ({n_units}) and vectors ({vectors.shape[0]}) disagree")
    if n_units != n_fts:
        die(f"units ({n_units}) and FTS rows ({n_fts}) disagree")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the local hybrid search index for one college bundle.")
    ap.add_argument("--college-id", required=True, help="college id, e.g. the bundle's college_id value")
    ap.add_argument("--data-dir", required=True, type=Path, help="directory holding the bundle jsonl files")
    ap.add_argument(
        "--index-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "index",
        help="index root; the index is written to <index-dir>/<college-id>/",
    )
    ap.add_argument("--limit", type=int, default=None, help="max rows read per source file (testing)")
    ap.add_argument("--rebuild", action="store_true", help="discard any existing index and rebuild")
    ap.add_argument("--skip-vectors", action="store_true", help="build units + FTS only (testing)")
    ap.add_argument("--threads", type=int, default=None, help="ONNX threads for the embedding model")
    ap.add_argument("--embed-provider", choices=("local", "openai"), default=None,
                    help="'local' (free, on this machine) or 'openai' (paid, higher quality)")
    ap.add_argument("--embed-model", default=None, help="override the model for the chosen provider")
    ap.add_argument("--embed-dims", type=int, default=None, help="override the vector width")
    ap.add_argument("--batch-size", type=int, default=None, help=f"embedding batch size (default {CONFIG['embed_batch_size']})")
    ap.add_argument("--db-read-batch", type=int, default=None, help=f"units per embedding round trip / resume point (default {CONFIG['db_read_batch']})")
    ap.add_argument("--progress-every", type=int, default=None, help=f"progress line every N units (default {CONFIG['progress_every']})")
    ap.add_argument("--max-embed-chars", type=int, default=None, help=f"trim embedding input (default {CONFIG['max_embed_chars']})")
    ap.add_argument("--built-at", type=str, default=None, help="build timestamp, unix seconds or ISO8601 (default: now)")
    args = ap.parse_args(argv)

    cfg = dict(CONFIG)
    if args.batch_size:
        cfg["embed_batch_size"] = args.batch_size
    if args.embed_provider:
        cfg["embed_provider"] = args.embed_provider
    if args.embed_model:
        key = "openai_model" if cfg.get("embed_provider") == "openai" else "embedding_model"
        cfg[key] = args.embed_model
    if args.embed_dims:
        key = "openai_dims" if cfg.get("embed_provider") == "openai" else "embedding_dims"
        cfg[key] = args.embed_dims
    if args.db_read_batch:
        cfg["db_read_batch"] = args.db_read_batch
    if args.progress_every:
        cfg["progress_every"] = args.progress_every
    if args.max_embed_chars:
        cfg["max_embed_chars"] = args.max_embed_chars

    if args.built_at is None:
        built_at_epoch = time.time()
    else:
        try:
            built_at_epoch = float(args.built_at)
        except ValueError:
            try:
                built_at_epoch = datetime.fromisoformat(args.built_at.replace("Z", "+00:00")).timestamp()
            except ValueError:
                die(f"--built-at is neither unix seconds nor ISO8601: {args.built_at!r}")

    if args.limit is not None and args.limit <= 0:
        die("--limit must be positive")

    meta = build_index(
        college_id=args.college_id,
        data_dir=args.data_dir.resolve(),
        index_root=args.index_dir.resolve(),
        cfg=cfg,
        limit=args.limit,
        rebuild=args.rebuild,
        threads=args.threads,
        built_at_epoch=built_at_epoch,
        skip_vectors=args.skip_vectors,
    )

    check = verify_index(args.index_dir.resolve(), args.college_id)
    index_dir = args.index_dir.resolve() / args.college_id
    log("")
    log("=" * 72)
    log(f"index ready: {index_dir}")
    log(f"  units      : {check['units']:,}  {json.dumps(check['by_kind'], sort_keys=True)}")
    if "vectors_shape" in check:
        log(
            f"  vectors    : {check['vectors_shape']} {check['vectors_dtype']} "
            f"(norms {check['vector_norm_min']}–{check['vector_norm_max']})"
        )
    else:
        log("  vectors    : NOT BUILT (--skip-vectors)")
    log(f"  categories : {json.dumps(meta['counts']['by_category_code'], sort_keys=True)}")
    if meta["counts"].get("unknown_category_codes"):
        log(f"  WARNING unknown category codes: {meta['counts']['unknown_category_codes']}")
    log(f"  undergrad courses: {meta['counts'].get('undergraduate_courses')}")
    log(
        f"  model      : {meta['embedding']['model']} ({meta['embedding']['dims']}d, "
        f"{meta['embedding']['provider']})"
    )
    log(f"  built_at   : {meta['built_at']}")
    log(f"  timings    : {json.dumps(meta['timings_seconds'], sort_keys=True)}")
    for name in ("chunks.sqlite", "vectors.npy", "meta.json"):
        p = index_dir / name
        if p.exists():
            log(f"  {name:<14} {p.stat().st_size / 1e6:,.1f} MB")
    log("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
