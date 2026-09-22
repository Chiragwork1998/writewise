"""wwrag/run.py -- one resume + one college -> a verified fit report, end to end.

Runs the six stages in order, each as its own process, writing every intermediate
artifact into one run directory:

    <out-dir>/
        profile.json            profile.py    resume -> StudentProfile, every item proved by a resume line
        profile.usage.json      profile.py    tokens spent extracting it
        evidence.json           retrieve.py   {category_code: [EvidenceUnit]} -- the native retrieval shape
        evidence_units.json     this file     the same units, flat and de-duplicated, for the two
                                              consumers that want a list rather than a mapping
        report_draft.json       generate.py   ReportItems + fit summary + token ledger
        verify_input.json       this file     the draft items PLUS the fit summary, broken into
                                              carrier items so the summary is verified too
        verified/               verify.py     report_verified.json, verified_claims.json, ledger.json
        report_items.json       this file     verified items, summary carriers removed
        summary.json            this file     the summary, rebuilt from the sentences that survived
        report.md/.html/.pdf    report.py     the document
        report_build.json       report.py     page count, citation count, integrity notes
        run.json                this file     inputs, models, timings, tokens, cost and counts per stage

Why an orchestrator and not a shell pipeline: the six modules were written independently
and disagree at two seams. This file normalises both, and fails loudly naming the stage
that broke rather than letting a consumer read a key its producer never wrote and quietly
default to empty -- the failure mode that makes every stage report success while the
student gets a thin report.

Run:
    /Users/chirag/college-intel/.venv-crawl4ai/bin/python /Users/chirag/college-intel/wwrag/run.py \
        --resume /Users/chirag/college-intel/inbox/ChiragTalwar.pdf \
        --college usc \
        --out-dir /Users/chirag/college-intel/runs/smoke01

Options: --skip-index, --per-category, --college-name, --data-dir, --index-root,
         --no-pdf, --no-model-verify, --from-stage, --to-stage, --dry-run
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# --------------------------------------------------------------------------------------
# config -- the only place a college, a path convention, a model or a price appears
# --------------------------------------------------------------------------------------

CONFIG: dict[str, Any] = {
    "index_root": HERE / "index",
    # {root} and {college_id} are filled in; --data-dir overrides it entirely
    "data_dir_template": "{root}/colleges/{college_id}/export/{college_id}_rag_v2",
    # display names for colleges we have bundles for; --college-name overrides
    "college_names": {"usc": "University of Southern California"},
    "api_base": "https://api.deepseek.com",
    "api_key_env": "DEEPSEEK_API_KEY",
    "writer_model": "deepseek-v4-pro",
    "cheap_model": "deepseek-flash",
    "embedding": "BAAI/bge-small-en-v1.5 (fastembed, local, free)",
    # 24, and the story is worth keeping. A free 7-config sweep showed 32 gives 27% more named
    # things in the EVIDENCE, so it was raised -- and the report got worse, not better: three
    # chapters failed outright because a third more evidence makes each response a third longer,
    # and the provider drops long responses. The sweep measured the half of the pipeline that
    # never fails. Do not raise this again without fixing the writer first.
    "per_category": 24,
    "category_order": ["CUL", "EXT", "QRK", "ACA", "RES", "SOC", "INN", "INT", "DIV", "NEW"],
    "supporting_code": "GEN",
    # DeepSeek published prices, USD per 1M tokens. Off-peak is half price.
    "prices_usd_per_mtok": {
        "deepseek-v4-pro": {
            "peak": {"cache_hit": 0.044, "cache_miss": 1.32, "output": 3.96},
            "offpeak": {"cache_hit": 0.022, "cache_miss": 0.66, "output": 1.98},
        },
        "deepseek-flash": {
            "peak": {"cache_hit": 0.006, "cache_miss": 0.30, "output": 1.20},
            "offpeak": {"cache_hit": 0.003, "cache_miss": 0.15, "output": 0.60},
        },
    },
    "peak_windows_utc": ((1, 4), (6, 10)),
}

# "useful" is a second quality gate, after verification and before rendering. verify.py
# answers "is this true?"; useful.py answers "is this worth the reader's attention?" -- a
# question nothing used to ask, which is how source metadata and self-restatement reached
# finished reports while every other check passed.
STAGES = ("index", "profile", "retrieve", "generate", "verify", "useful", "report")

# key carried on a synthetic item so we can pull the summary back out after verification
ROLE_KEY = "_wwrag_summary_role"
SUMMARY_ROLES = ("fit_summary", "strongest_match", "open_question")
CITATION = re.compile(r"\[([^\[\]]{1,400})\]")


class StageError(RuntimeError):
    """A stage failed, or produced something the next stage cannot read."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(f"[{stage}] {message}")
        self.stage = stage
        self.message = message


def log(msg: str = "") -> None:
    print(msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------------------
# cost
# --------------------------------------------------------------------------------------

def is_peak(when: dt.datetime) -> bool:
    when = when.astimezone(dt.timezone.utc)
    if when.weekday() >= 5:
        return False
    hour = when.hour + when.minute / 60.0
    return any(start <= hour < end for start, end in CONFIG["peak_windows_utc"])


def price_calls(calls: list[dict], when: dt.datetime) -> dict[str, Any]:
    """Total the tokens and the money for a list of {model, usage} records."""
    band = "peak" if is_peak(when) else "offpeak"
    totals = {"prompt_tokens": 0, "prompt_cache_hit_tokens": 0,
              "prompt_cache_miss_tokens": 0, "completion_tokens": 0}
    cost = 0.0
    unpriced: set[str] = set()
    for call in calls:
        usage = call.get("usage") or {}
        model = str(call.get("model") or "")
        prompt = int(usage.get("prompt_tokens") or 0)
        hit = int(usage.get("prompt_cache_hit_tokens") or 0)
        miss = usage.get("prompt_cache_miss_tokens")
        miss = int(miss) if miss is not None else max(prompt - hit, 0)
        out = int(usage.get("completion_tokens") or 0)
        totals["prompt_tokens"] += prompt
        totals["prompt_cache_hit_tokens"] += hit
        totals["prompt_cache_miss_tokens"] += miss
        totals["completion_tokens"] += out
        rate = (CONFIG["prices_usd_per_mtok"].get(model) or {}).get(band)
        if not rate:
            unpriced.add(model)
            continue
        cost += (hit * rate["cache_hit"] + miss * rate["cache_miss"]
                 + out * rate["output"]) / 1e6
    return {"calls": len(calls), "tokens": totals, "pricing_band": band,
            "cost_usd": round(cost, 6), "unpriced_models": sorted(unpriced)}


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------

def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, stage: str) -> Any:
    if not path.exists():
        raise StageError(stage, f"expected output {path} was not written")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StageError(stage, f"{path} is not valid JSON: {exc}") from exc


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def cited_ids(text: str) -> list[str]:
    out: list[str] = []
    for group in CITATION.findall(text or ""):
        out.extend(part.strip() for part in re.split(r"[,;]", group) if part.strip())
    return out


def run_stage(stage: str, cmd: list[Any], record: dict[str, Any],
              tail_lines: int = 25) -> dict[str, Any]:
    """Run one module as a subprocess. Raise StageError naming the stage if it fails."""
    argv = [str(c) for c in cmd]
    log("")
    log(f"=== {stage} " + "=" * max(0, 62 - len(stage)))
    log("    " + " ".join(argv))
    started = time.time()
    proc = subprocess.run(argv, cwd=str(ROOT), capture_output=True, text=True)
    seconds = round(time.time() - started, 2)

    err = (proc.stderr or "").strip().splitlines()
    out = (proc.stdout or "").strip().splitlines()
    for line in err[-tail_lines:]:
        log("    | " + line)
    entry = {
        "stage": stage,
        "command": argv,
        "seconds": seconds,
        "returncode": proc.returncode,
        "stderr_tail": err[-60:],
        "stdout_tail": out[-40:],
    }
    record["stages"].append(entry)
    if proc.returncode != 0:
        for line in out[-tail_lines:]:
            log("    > " + line)
        detail = " | ".join((err[-3:] or out[-3:]) or ["no output"])
        raise StageError(stage, f"exited {proc.returncode} after {seconds}s: {detail}")
    log(f"    ok in {seconds}s")
    return entry


# --------------------------------------------------------------------------------------
# seam 1: retrieve.py writes {category_code: [unit]}; verify.py and report.py each want a
# flat list under a known key. Derive one rather than rewriting either module.
# --------------------------------------------------------------------------------------

# Set by main() once the index directory is known; flatten_evidence reads it at the seam.
GRAPH_INDEX_DIR: dict[str, str] = {}

# The applicant's level, set by main() once the profile exists, read by the evidence seam.
# It used to reach flatten_evidence() by referencing `profile_path`, a LOCAL of main(), so
# every call raised NameError -- which the seam's own except clause caught and logged as
# "eligibility: skipped after an error". The gate written to stop an ineligible programme
# being named as a student's strongest match therefore never executed once, and said so in
# a line that read like a routine skip. Fail-open hid fail-always.
APPLICANT: dict[str, str] = {}


def graph_tables_present(index_dir: Path) -> bool:
    """True when this index already carries a populated entity graph."""
    db = Path(index_dir) / "chunks.sqlite"
    if not db.is_file():
        return False
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='graph_edges'"
            ).fetchone()
            if not row or not row[0]:
                return False
            return bool(conn.execute("SELECT 1 FROM graph_edges LIMIT 1").fetchone())
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001 - report it; a silent False is how this hid before
        log(f"    !! could not read {db} to check for an entity graph ({exc})")
        return False


def flatten_evidence(evidence_path: Path, flat_path: Path) -> dict[str, Any]:
    data = read_json(evidence_path, "retrieve")
    if not isinstance(data, dict):
        raise StageError("retrieve", f"{evidence_path}: expected an object keyed by category code")

    buckets = data.get("evidence") if isinstance(data.get("evidence"), dict) else None
    if buckets is None:
        buckets = data.get("by_category") if isinstance(data.get("by_category"), dict) else None
    if buckets is None:
        buckets = {k: v for k, v in data.items() if isinstance(v, list)}
    if not buckets:
        raise StageError("retrieve", f"{evidence_path}: no per-category evidence found")

    # Drop what this applicant cannot do, and tag what they must be told about. A source
    # saying "this exists" is not the same as "you may have this", and the verification gate
    # only ever checked the first. Life stage is filtered (a pre-college course is not open
    # to an enrolled undergraduate); citizenship is never inferred, only reported.
    # Add the entity graph's relationships as evidence, HERE at the seam rather than inside
    # the writer, so the writer and the verifier see exactly the same units. Expanding inside
    # generate.py alone would let it cite a relation that never reached evidence_units.json,
    # and every such citation would look fabricated to the verifier and be deleted.
    graph_stats: dict[str, Any] = {"available": False, "reason": "no index passed"}
    if GRAPH_INDEX_DIR.get("path"):
        try:
            sys.path.insert(0, str(HERE))
            import graph as _graph

            buckets, graph_stats = _graph.expand_evidence(
                Path(GRAPH_INDEX_DIR["path"]), buckets, GRAPH_INDEX_DIR.get("college_id", ""),
                level=APPLICANT.get("level") or "undergraduate",
            )
            if graph_stats.get("available"):
                log(f"    graph: +{graph_stats['relations_added']} relation units "
                    f"{graph_stats['by_category']}")
                if graph_stats.get("wrong_level_dropped"):
                    log(f"    graph: dropped {len(graph_stats['wrong_level_dropped'])} relation(s) "
                        f"naming a course of the wrong level: "
                        + ", ".join(d["entity"][:40] for d in graph_stats["wrong_level_dropped"][:4]))
            else:
                log(f"    graph: not used ({graph_stats.get('reason')})")
        except Exception as exc:  # noqa: BLE001 - enrichment must never fail a paid run
            log(f"    graph: skipped after an error ({exc})")
            graph_stats = {"available": False, "reason": str(exc)}

    # The level gate runs AFTER the graph, so a relation's sentence is judged like any other
    # unit. Before, relation units skipped it: a graduate course reached an undergraduate report.
    try:
        sys.path.insert(0, str(HERE))
        import eligibility as _elig

        level = APPLICANT.get("level") or "undergraduate"
        buckets, elig_stats = _elig.annotate_buckets(buckets, level)
        if elig_stats["dropped_total"] or elig_stats["tagged_total"]:
            log(f"    eligibility ({level}): dropped {elig_stats['dropped_total']} wrong-level "
                f"unit(s), tagged {elig_stats['tagged_total']} with a stated restriction")
    except Exception as exc:  # noqa: BLE001 - a run still completes, but this is never quiet
        log(f"    !! ELIGIBILITY GATE FAILED ({type(exc).__name__}: {exc}) -- restricted "
            f"material is NOT being filtered or flagged in this run")
        elig_stats = {"error": f"{type(exc).__name__}: {exc}", "available": False}

        # persist the expanded evidence so generate.py reads the same set
        data_out = dict(data)
        if isinstance(data_out.get("evidence"), dict):
            data_out["evidence"] = buckets
        elif isinstance(data_out.get("by_category"), dict):
            data_out["by_category"] = buckets
        else:
            data_out.update(buckets)
        data_out["graph"] = graph_stats
        write_json(evidence_path, data_out)

    flat: list[dict[str, Any]] = []
    seen: set[str] = set()
    per_category: dict[str, int] = {}
    for code, units in buckets.items():
        if not isinstance(units, list):
            continue
        per_category[code] = len(units)
        for unit in units:
            if not isinstance(unit, dict):
                continue
            unit_id = str(unit.get("unit_id") or unit.get("id") or "").strip()
            if not unit_id or unit_id in seen:
                continue
            seen.add(unit_id)
            flat.append(unit)

    if not flat:
        raise StageError(
            "retrieve",
            f"{evidence_path}: retrieval returned no evidence at all. Every downstream "
            f"stage would produce an empty report. Check the index and the category filters.")

    write_json(flat_path, {"units": flat})
    empty = sorted(code for code, n in per_category.items() if n == 0)
    log(f"    evidence: {len(flat)} unique units across {len(per_category)} categories")
    log(f"    evidence: {json.dumps(per_category, sort_keys=True)}")
    if empty:
        log(f"    evidence: EMPTY categories {empty} -- those sections will say so, not invent content")
    return {"units_unique": len(flat), "units_by_category": per_category, "empty_categories": empty}


# Which stages read the flat file rather than retrieve.py's native per-category shape.
FLAT_EVIDENCE_CONSUMERS = ("verify", "report")


def ensure_flat_evidence(evidence_path: Path, flat_path: Path,
                         stages: list[str], record: dict[str, Any]) -> None:
    """Make sure evidence_units.json exists for whoever is about to read it.

    The flat file is a property of the evidence FILE, not of the retrieve STAGE. It used
    to be derived only inside the retrieve branch of the driver loop, which meant
    `--from-stage generate` produced a run with evidence.json on disk and no
    evidence_units.json -- and verify.py was still handed `--evidence evidence_units.json`.
    The run then died at verify, i.e. *after* the generate stage had already been billed
    to a real card. Derive it from what is on disk, before the first stage starts, so a
    resumed run either works or fails for free.

    Deriving it every time also keeps the two files in step: an evidence.json rebuilt by
    hand between runs can no longer be silently paired with a stale flat file.
    """
    if "retrieve" in stages:
        return                              # the retrieve branch derives it from its own output
    if not any(stage in stages for stage in FLAT_EVIDENCE_CONSUMERS):
        return                              # nothing downstream is going to read it
    if not evidence_path.exists():
        raise StageError(
            "preflight",
            f"resuming at {stages[0]!r} needs retrieval's output, but {evidence_path} is "
            f"not there. Re-run with --from-stage retrieve, or point --out-dir at the run "
            f"directory that has it. Failing now rather than after a paid stage.")
    log("")
    log("=== preflight " + "=" * 53)
    log(f"    normalising {evidence_path.name} -> {flat_path.name} (retrieve is not in this run)")
    record["counts"]["retrieve"] = flatten_evidence(evidence_path, flat_path)


# --------------------------------------------------------------------------------------
# seam 2: generate.py writes a fit summary that verify.py never sees, because verify reads
# items only. Carry the summary through verification as items, then rebuild it from what
# survived, so no summary sentence reaches the student unchecked.
# --------------------------------------------------------------------------------------

def carrier(role: str, headline: str, text: str) -> dict[str, Any]:
    return {
        "category_code": CONFIG["supporting_code"],
        "headline": headline,
        "body": text,
        "why_it_matters": "",
        "evidence_ids": sorted(set(cited_ids(text))),
        "profile_basis": [],
        "caveat": None,
        ROLE_KEY: role,
    }


def build_verify_input(draft_path: Path, verify_input_path: Path) -> dict[str, Any]:
    draft = read_json(draft_path, "generate")
    items = draft.get("items") if isinstance(draft, dict) else draft
    if not isinstance(items, list) or not items:
        raise StageError("generate", f"{draft_path}: generation produced no items")

    uncited = [i for i in items if not (i.get("evidence_ids") or [])]
    if uncited:
        raise StageError(
            "generate",
            f"{draft_path}: {len(uncited)} item(s) cite no evidence. Nothing uncited may "
            f"reach a student; fix generation rather than letting verification delete them.")

    summary = draft.get("summary") if isinstance(draft, dict) else {}
    summary = summary if isinstance(summary, dict) else {}
    carriers: list[dict[str, Any]] = []
    fit = str(summary.get("fit_summary") or "").strip()
    if fit:
        carriers.append(carrier("fit_summary", "Fit summary", fit))
    for text in summary.get("strongest_matches") or []:
        if isinstance(text, str) and text.strip():
            carriers.append(carrier("strongest_match", "Strongest match", text.strip()))
    for text in summary.get("open_questions") or []:
        if isinstance(text, str) and text.strip():
            carriers.append(carrier("open_question", "Worth asking", text.strip()))

    write_json(verify_input_path, {"items": list(items) + carriers})

    by_category: dict[str, int] = {}
    for item in items:
        code = str(item.get("category_code") or "?")
        by_category[code] = by_category.get(code, 0) + 1
    log(f"    draft: {len(items)} items {json.dumps(by_category, sort_keys=True)}")
    log(f"    draft: {len(carriers)} summary fragment(s) added to the verification queue")
    return {
        "items": len(items),
        "items_by_category": by_category,
        "summary_carriers": len(carriers),
        "dropped_by_generation": len(draft.get("dropped") or []) if isinstance(draft, dict) else 0,
    }


def split_verified(verified_path: Path, items_path: Path,
                   summary_path: Path) -> dict[str, Any]:
    rows = read_json(verified_path, "verify")
    if isinstance(rows, dict):
        rows = rows.get("items") or rows.get("report") or []
    if not isinstance(rows, list):
        raise StageError("verify", f"{verified_path}: expected a list of verified items")

    items: list[dict[str, Any]] = []
    parts: dict[str, list[str]] = {role: [] for role in SUMMARY_ROLES}
    summary_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        role = row.get(ROLE_KEY)
        if role in parts:
            body = str(row.get("body") or "").strip()
            if body:
                parts[role].append(body)
                # The rewriter sometimes drops the inline [unit_id] marker while keeping
                # the claim. Carry the surviving ids so report.py can still show a source
                # for that sentence rather than leaving it looking unattributed.
                summary_ids.update(str(i) for i in (row.get("evidence_ids") or []))
        else:
            items.append(row)

    write_json(items_path, items)
    summary = {
        "fit_summary": " ".join(parts["fit_summary"]).strip(),
        "strongest_matches": parts["strongest_match"],
        "open_questions": parts["open_question"],
        "evidence_ids": sorted(summary_ids),
    }
    has_summary = bool(summary["fit_summary"])
    if has_summary:
        write_json(summary_path, {"summary": summary})
    return {
        "items_verified": len(items),
        "summary_survived": has_summary,
        "strongest_matches": len(summary["strongest_matches"]),
        "open_questions": len(summary["open_questions"]),
    }


# --------------------------------------------------------------------------------------
# per-stage checks
# --------------------------------------------------------------------------------------

def check_profile(path: Path) -> dict[str, Any]:
    profile = read_json(path, "profile")
    if not isinstance(profile, dict) or not profile.get("student_id"):
        raise StageError("profile", f"{path} is not a StudentProfile (no student_id)")
    counts = {key: len(profile.get(key) or [])
              for key in ("intended_fields", "activities", "projects", "skills",
                          "interests", "values", "achievements")}
    if sum(counts.values()) == 0:
        raise StageError(
            "profile",
            f"{path} extracted nothing from the resume. Every later stage would produce a "
            f"generic report. Flags: {profile.get('flags')}")
    log(f"    profile: level={profile.get('level')!r} first_name={profile.get('first_name')!r}")
    log(f"    profile: {json.dumps(counts, sort_keys=True)}")
    dropped = [f for f in (profile.get("flags") or []) if str(f).startswith("dropped")]
    if dropped:
        log(f"    profile: {len(dropped)} item(s) dropped by the verbatim gate")
    return {"level": profile.get("level"), "first_name": profile.get("first_name"),
            "counts": counts, "flags": profile.get("flags") or []}


def check_verified(out_dir: Path) -> dict[str, Any]:
    ledger = read_json(out_dir / "ledger.json", "verify")
    counts = ledger.get("counts") or {}
    rates = ledger.get("pass_rates") or {}
    items_in = int(counts.get("items_in") or 0)
    log(f"    verified: claims {counts.get('claims_total')} -> "
        f"supported {counts.get('supported')}, corrected {counts.get('corrected')}, "
        f"removed {counts.get('removed')}")
    log(f"    verified: items {counts.get('items_in')} in -> {counts.get('items_out')} out")
    if counts.get("items_out") == 0:
        raise StageError(
            "verify",
            "verification deleted every item. Generation wrote claims its own evidence "
            "does not support -- a real failure, not a formatting problem.")
    if items_in and (items_in - int(counts.get("items_out") or 0)) / items_in > 0.5:
        log(f"    ! verified: over half the items failed. Read {out_dir / 'ledger.json'} "
            f"before trusting this run.")
    return {"counts": counts, "pass_rates": rates,
            "deterministic_only": ledger.get("deterministic_only"),
            "usage": ledger.get("usage") or {"calls": []}}


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="One resume + one college -> a verified fit report.")
    parser.add_argument("--resume", required=True, help="resume .pdf/.txt/.md")
    parser.add_argument("--college", required=True, help="college id, e.g. the bundle's college_id")
    parser.add_argument("--out-dir", required=True, help="run directory for every artifact")
    parser.add_argument("--skip-index", action="store_true",
                        help="never build the index; fail if it is missing or unfinished")
    parser.add_argument("--data-dir", default=None,
                        help="the college bundle (default: the template in CONFIG)")
    parser.add_argument("--index-root", default=str(CONFIG["index_root"]),
                        help="index root; the college index lives in <root>/<college>/")
    parser.add_argument("--college-name", default=None, help="display name for the document")
    parser.add_argument("--per-category", type=int, default=CONFIG["per_category"])
    parser.add_argument(
        "--fields", default=None,
        help="comma-separated majors the student has declared, e.g. 'Gender Studies, Economics'",
    )
    parser.add_argument(
        "--concurrency", type=int, default=8,
        help="chapters written at once. Measured on a real run: 3 took 762s and 8 took 276s for "
             "the same output at the same price, because the bill is per token, not per second.",
    )
    parser.add_argument("--env", default=str(ROOT / ".env"))
    parser.add_argument("--from-stage", default="index", choices=STAGES,
                        help="resume a run from this stage, reusing earlier artifacts")
    parser.add_argument("--to-stage", default="report", choices=STAGES,
                        help="stop after this stage (checking its seam first)")
    parser.add_argument("--no-pdf", action="store_true")
    parser.add_argument("--no-model-verify", action="store_true",
                        help="deterministic verification only -- no API calls")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    args = parser.parse_args(argv)

    resume = Path(args.resume).expanduser().resolve()
    if not resume.exists():
        raise StageError("preflight", f"resume not found: {resume}")

    college_id = args.college.strip().lower()
    if not college_id:
        raise StageError("preflight", "--college must name a college id")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    index_root = Path(args.index_root).expanduser().resolve()
    index_dir = index_root / college_id
    # tell the evidence seam which index to pull entity relationships from
    GRAPH_INDEX_DIR["path"] = str(index_dir)
    GRAPH_INDEX_DIR["college_id"] = college_id
    data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir else Path(
        CONFIG["data_dir_template"].format(root=ROOT, college_id=college_id)).resolve()
    college_name = (args.college_name
                    or CONFIG["college_names"].get(college_id)
                    or college_id.upper())

    profile_path = out_dir / "profile.json"
    # A resumed run (--from-stage retrieve/generate/...) skips the profile stage, so read the
    # level here too; the profile stage sets it again once it has written a fresh one.
    if profile_path.exists():
        APPLICANT["level"] = str((read_json(profile_path, "profile") or {}).get("level")
                                 or "undergraduate")
    profile_usage_path = out_dir / "profile.usage.json"
    evidence_path = out_dir / "evidence.json"
    flat_path = out_dir / "evidence_units.json"
    useful_path = out_dir / "report_useful.json"
    useful_ledger_path = out_dir / "usefulness_cuts.json"
    draft_path = out_dir / "report_draft.json"
    verify_input_path = out_dir / "verify_input.json"
    verified_dir = out_dir / "verified"
    items_path = out_dir / "report_items.json"
    summary_path = out_dir / "summary.json"

    first, last = STAGES.index(args.from_stage), STAGES.index(args.to_stage)
    if last < first:
        raise StageError("preflight",
                         f"--to-stage {args.to_stage} comes before --from-stage {args.from_stage}")
    stages = list(STAGES[first:last + 1])
    # render from the gated file when the gate ran, otherwise from the verified items, so
    # --from-stage report still works on a run that predates this stage
    report_source = useful_path if ("useful" in stages or useful_path.exists()) else items_path

    record: dict[str, Any] = {
        "run_id": out_dir.name,
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "inputs": {
            "resume": str(resume),
            "resume_sha256": sha256_file(resume),
            "college_id": college_id,
            "college_name": college_name,
            "data_dir": str(data_dir),
            "index_dir": str(index_dir),
            "per_category": args.per_category,
        },
        "models": {
            "writer": CONFIG["writer_model"],
            "cheap": CONFIG["cheap_model"],
            "api_base": CONFIG["api_base"],
            "embeddings": CONFIG["embedding"],
        },
        "python": sys.executable,
        "stages": [],
        "counts": {},
        "usage": {},
        "cost_usd": {},
        "failed_stage": None,
        "error": None,
    }

    python = sys.executable
    commands: dict[str, list[Any]] = {
        "index": [python, HERE / "index_build.py", "--college-id", college_id,
                  "--data-dir", data_dir, "--index-dir", index_root],
        "graph": [python, HERE / "graph.py", "build",
                  "--index", index_dir, "--data-dir", data_dir],
        "profile": [python, HERE / "profile.py", "--resume", resume,
                    "--out", profile_path, "--env-file", args.env,
                    "--model", CONFIG["writer_model"]]
                   + (["--fields", args.fields] if args.fields else []),
        "retrieve": [python, HERE / "retrieve.py", "--profile", profile_path,
                     "--index", index_root, "--college-id", college_id,
                     "--out", evidence_path, "--per-category", str(args.per_category)],
        "generate": [python, HERE / "generate.py", "--profile", profile_path,
                     "--evidence", evidence_path, "--out", draft_path,
                     "--model", CONFIG["writer_model"], "--env-file", args.env,
                     "--api-base", CONFIG["api_base"], "--college-name", college_name,
                     "--concurrency", str(args.concurrency), "--quiet"],
        "verify": [python, HERE / "verify.py", "--report", verify_input_path,
                   "--evidence", flat_path, "--profile", profile_path,
                   "--out", verified_dir, "--env", args.env,
                   "--model", CONFIG["cheap_model"], "--api-base", CONFIG["api_base"]],
        "useful": [python, HERE / "useful.py", "--report", items_path,
                   "--out", useful_path, "--ledger", useful_ledger_path,
                   "--model", CONFIG["cheap_model"], "--api-base", CONFIG["api_base"],
                   "--env", args.env],
        "report": [python, HERE / "report.py", "--report", report_source,
                   "--evidence", flat_path, "--profile", profile_path,
                   "--out-dir", out_dir, "--college-name", college_name],
    }
    if args.no_model_verify:
        commands["verify"].append("--no-model")
    if args.no_pdf:
        commands["report"].append("--no-pdf")

    if args.dry_run:
        for stage in stages:
            print(f"{stage}: " + " ".join(str(c) for c in commands[stage]))
        return 0

    started = time.time()
    try:
        # Seam 1 is a property of the files, not of the stage list: do it before anything
        # runs so a resumed run that is missing retrieval's output fails here, for free,
        # instead of at verify with the generate stage already paid for.
        ensure_flat_evidence(evidence_path, flat_path, stages, record)
        for stage in stages:
            if stage == "index":
                meta_path = index_dir / "meta.json"
                complete = False
                if meta_path.exists():
                    try:
                        complete = bool(json.loads(meta_path.read_text()).get("complete"))
                    except json.JSONDecodeError:
                        complete = False
                if complete:
                    meta = json.loads(meta_path.read_text())
                    log(f"\n=== index " + "=" * 57)
                    log(f"    reusing {index_dir}: {meta['counts']['units']:,} units, "
                        f"{meta['embedding']['model']} ({meta['embedding']['provider']})")
                    record["stages"].append({"stage": "index", "seconds": 0.0,
                                             "returncode": 0, "skipped": "already complete"})
                elif args.skip_index:
                    raise StageError(
                        "index",
                        f"no finished index at {index_dir} and --skip-index was given. "
                        f"Drop --skip-index, or build it with index_build.py first.")
                else:
                    if not data_dir.exists():
                        raise StageError("index", f"college bundle not found: {data_dir}")
                    run_stage("index", commands["index"], record, tail_lines=20)
                # The entity graph is part of a usable index, not an optional extra. It was a
                # manual `graph.py build` step, so the natural command for a new college
                # produced an index with no graph_nodes/graph_edges tables at all -- and BOTH
                # features that depend on it fail quiet: graph relations simply add nothing,
                # and gapfill disables itself. College #2 would have run end to end, reported
                # success, and shipped reports missing the two things added to fix the worst
                # quality problem we had.
                # Build it only when it is actually absent, so reusing a finished index costs
                # nothing -- but NEVER skip the check, because an index without a graph is the
                # exact state college #2 lands in and both dependent features fail silently.
                if graph_tables_present(index_dir):
                    log("    graph: already built")
                elif data_dir.exists():
                    run_stage("graph", commands["graph"], record)
                else:
                    log(f"    !! NO ENTITY GRAPH at {index_dir} and no bundle at {data_dir} to "
                        f"build one from -- relation evidence and the gap-filling pass are BOTH "
                        f"disabled for this run")
                meta = read_json(meta_path, "index")
                record["counts"]["index"] = {
                    "units": meta["counts"]["units"],
                    "by_kind": meta["counts"]["by_kind"],
                    "by_category_code": meta["counts"]["by_category_code"],
                    "undergraduate_courses": meta["counts"].get("undergraduate_courses"),
                    "embedding_model": meta["embedding"]["model"],
                    "embedding_provider": meta["embedding"]["provider"],
                    "built_at": meta.get("built_at"),
                }

            elif stage == "profile":
                run_stage("profile", commands["profile"], record)
                record["counts"]["profile"] = check_profile(profile_path)
                # the evidence seam needs the applicant's level; set it as soon as it is known
                APPLICANT["level"] = str((read_json(profile_path, "profile") or {}).get("level")
                                         or "undergraduate")
                calls = (read_json(profile_usage_path, "profile").get("calls")
                         if profile_usage_path.exists() else [])
                record["usage"]["profile"] = price_calls(calls, dt.datetime.now(dt.timezone.utc))
                record["cost_usd"]["profile"] = record["usage"]["profile"]["cost_usd"]

            elif stage == "retrieve":
                run_stage("retrieve", commands["retrieve"], record, tail_lines=10)
                record["counts"]["retrieve"] = flatten_evidence(evidence_path, flat_path)
                record["cost_usd"]["retrieve"] = 0.0  # local embeddings, free

            elif stage == "generate":
                run_stage("generate", commands["generate"], record)
                record["counts"]["generate"] = build_verify_input(draft_path, verify_input_path)
                draft = read_json(draft_path, "generate")
                record["usage"]["generate"] = {
                    "calls": len(draft.get("usage", {}).get("calls") or []),
                    "tokens": {k: v for k, v in (draft.get("usage") or {}).items()
                               if k != "calls"},
                }
                record["cost_usd"]["generate"] = draft.get("cost_usd")

            elif stage == "verify":
                run_stage("verify", commands["verify"], record)
                verified = check_verified(verified_dir)
                split = split_verified(verified_dir / "report_verified.json",
                                       items_path, summary_path)
                verified.update(split)
                usage = verified.pop("usage")
                record["counts"]["verify"] = verified
                record["usage"]["verify"] = price_calls(usage.get("calls") or [],
                                                        dt.datetime.now(dt.timezone.utc))
                record["cost_usd"]["verify"] = record["usage"]["verify"]["cost_usd"]
                log(f"    verified: summary {'survived' if split['summary_survived'] else 'DELETED'}"
                    f", {split['strongest_matches']} strongest match(es), "
                    f"{split['open_questions']} open question(s)")

            elif stage == "useful":
                run_stage("useful", commands["useful"], record)
                gated = read_json(useful_path, "useful")
                st = (gated.get("usefulness") or {}) if isinstance(gated, dict) else {}
                record["counts"]["useful"] = st
                log(f"    usefulness: cut {st.get('sentences_cut', 0)}/"
                    f"{st.get('sentences_total', 0)} sentences "
                    f"({st.get('cut_deterministic', 0)} mechanical, "
                    f"{st.get('cut_by_model', 0)} judged); items "
                    f"{st.get('items_in', 0)} -> {st.get('items_out', 0)}")

            elif stage == "report":
                cmd = list(commands["report"])
                if summary_path.exists():
                    cmd += ["--summary", summary_path]
                else:
                    log("    report: no verified summary; report.py will build its own "
                        "from the surviving items only")
                run_stage("report", cmd, record)
                build = read_json(out_dir / "report_build.json", "report")
                record["counts"]["report"] = {
                    "pages": build.get("pages"),
                    "body_pt": build.get("body_pt"),
                    "page_budget_ok": build.get("page_budget_ok"),
                    "files": build.get("files"),
                    "items_rendered": (build.get("stats") or {}).get("items_rendered"),
                    "citations": (build.get("stats") or {}).get("citations"),
                    "sources": (build.get("stats") or {}).get("sources"),
                    "categories_rendered": (build.get("stats") or {}).get("categories_rendered"),
                    "categories_empty": (build.get("stats") or {}).get("categories_empty"),
                    "notes": build.get("notes") or [],
                }
                record["cost_usd"]["report"] = 0.0
    except StageError as exc:
        record["failed_stage"] = exc.stage
        record["error"] = exc.message
        raise
    finally:
        record["seconds"] = round(time.time() - started, 2)
        record["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        record["cost_usd"]["total"] = round(
            sum(v for v in record["cost_usd"].values() if isinstance(v, (int, float))), 6)
        record["seconds_by_stage"] = {s["stage"]: s.get("seconds", 0.0) for s in record["stages"]}
        write_json(out_dir / "run.json", record)

    log("")
    log("=" * 68)
    log(f"done in {record['seconds']}s -> {out_dir}")
    log(f"cost ${record['cost_usd']['total']:.4f} "
        f"({json.dumps({k: v for k, v in record['cost_usd'].items() if k != 'total'})})")
    for name in ("report.md", "report.html", "report.pdf"):
        path = out_dir / name
        if path.exists():
            log(f"    {path}  ({path.stat().st_size:,} B)")
    log("=" * 68)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except StageError as exc:
        log("")
        log(f"FAILED in stage '{exc.stage}': {exc.message}")
        sys.exit(2)
