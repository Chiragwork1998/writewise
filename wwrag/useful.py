"""wwrag/useful.py -- the usefulness gate: delete sentences that are true but not worth saying.

The pipeline had one quality dimension and needed two. verify.py answers "is this supported
by the evidence?" and answers it well. Nothing answered "is this worth a reader's attention?"

That gap has a measurable cost. Four independent reviewers -- a college counsellor, a
seventeen-year-old applicant, a parent and a fact-checking editor -- read finished reports
with no knowledge of how they were produced and scored them 3.75-4.25 out of 10, worth about
$20, while the project's own structural scorer reported 7.4-8.2. Almost every line they
quoted as padding was impeccably true and impeccably cited:

    "The article is titled 'Students presenting AI research at the poster session.'
     This external reference shows students presenting AI research at a poster session."
    "The photo is credited to Venice Tang."
    "Worth knowing: This is a blog post about getting started in research as an undergraduate."
    "The Interdisciplinary Major Program lets students design programs crossing traditional
     majors. Directed by X and Y, the Interdisciplinary Major Program lets students design an
     individual program of study that crosses the lines between traditional majors."

Every one passes verification, because every one is supported. They fail a different test:
they tell the reader nothing about the university that changes their picture of it or gives
them something to do.

This module applies that second test, sentence by sentence, and deletes what fails. It runs
AFTER verification, so everything it sees is already true; it only ever removes. It cannot
introduce an unsupported claim because it writes nothing.

Deliberately NOT a scoring pass. A score invites tuning toward the score, which is how the
structural scorer came to report 8/10 for documents a counsellor would not forward. This
returns a keep/cut decision per sentence and a reason.

Run:
  python wwrag/useful.py --report runs/x/report_items.json --out runs/x/report_useful.json
  python wwrag/useful.py --report r.json --out o.json --no-model    # deterministic layer only
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Sequence

import httpx

CONFIG: dict[str, Any] = {
    "api_base": "https://api.deepseek.com",
    "api_key_env": "DEEPSEEK_API_KEY",
    "model": "deepseek-flash",
    "temperature": 0.0,
    "timeout_s": 180.0,
    "max_retries": 4,
    "batch_size": 8,
    "max_workers": 4,
    # An item stripped below this many characters of body has nothing left worth printing.
    "min_body_chars_after": 90,
    # Never cut so much from one item that only a fragment remains; drop the item instead.
    "max_cut_fraction": 0.7,
}

# --------------------------------------------------------------------------------------
# Deterministic layer: shapes that are never worth printing, no model needed
# --------------------------------------------------------------------------------------

# Sentences about the SOURCE rather than the subject.
METADATA = re.compile(
    r"\b(the (?:article|page|post|blog|piece|announcement|press release|title|headline)\b[^.]{0,40}"
    r"(?:is )?(?:titled|is called|describes|says|reads)"
    r"|is titled\b"
    r"|photo (?:is )?credited to"
    r"|(?:this|the) (?:external )?(?:reference|article|source|page) (?:shows|describes|states|indicates|is from)"
    r"|this is a (?:blog post|news article|press release|announcement|photo|caption)"
    r"|published (?:an article|a post|a blog piece)"
    r"|the page'?s title is"
    r"|according to the (?:article|page|post)'?s title)\b", re.I)

# Sentences that assert the existence of something without saying anything about it.
EMPTY_ASSERTION = re.compile(
    r"^\s*(?:the )?[A-Z][\w\s&'.,-]{2,60}\s+"
    r"(?:offers|has|provides|includes|lists|maintains|features)\s+"
    r"(?:a|an|the|some|several|many|various|multiple)?\s*"
    r"(?:student )?(?:organizations?|clubs?|programs?|programmes?|opportunities|resources|services|majors?|options?)\s*\.?\s*$",
    re.I)

_WS = re.compile(r"\s+")

# Sentences that must never be cut, whatever else they look like.
#
# A neutral reviewer caught this gate deleting "The Research Experiences for Undergraduates
# site is supported by a grant from the National Science Foundation." It reads like trivia.
# It is the only signal in the document that the programme is NSF-funded and therefore, as a
# rule, closed to non-US citizens -- and the student it was written for applies from India.
# Cutting it did not remove padding; it removed the one line that stopped the report's
# top-ranked match from being useless to her.
#
# Anything carrying eligibility, funding, cost, deadlines, selectivity, prerequisites or a
# concrete requirement stays, because a reader cannot act correctly without it and its value
# is usually invisible in the sentence itself.
PROTECTED = re.compile(
    r"\b(eligib|eligibility|prerequisite|pre-requisite|required for|requirement"
    r"|must (?:be|have|complete|apply)|only open to|open only to|restricted to"
    r"|citizens?|permanent resident|visa|international students? (?:are|may|must|cannot)"
    r"|funded by|grant from|scholarship|stipend|tuition|fee|cost|\$[0-9]"
    r"|deadline|apply by|applications? (?:open|close|due)"
    r"|competitive|selective|acceptance rate|limited (?:to|spots|places)"
    r"|credits?|units?|semesters?|GPA|minimum grade"
    r"|contact (?:the|them)|email .*@|how to apply)\b", re.I)



def normalise(text: str | None) -> str:
    if not text:
        return ""
    return _WS.sub(" ", unicodedata.normalize("NFKC", str(text))).strip()


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(\[]|\d)", normalise(text))
    return [p.strip() for p in parts if p.strip()]


def content_words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]{4,}", text.lower()))


def restates(a: str, b: str, threshold: float = 0.72) -> bool:
    """True when two sentences carry essentially the same content."""
    wa, wb = content_words(a), content_words(b)
    if len(wa) < 6 or len(wb) < 6:
        return False
    return len(wa & wb) / min(len(wa), len(wb)) >= threshold


def deterministic_cut(sentences: Sequence[str]) -> dict[int, str]:
    """{index: reason} for sentences no model call is needed to reject."""
    cuts: dict[int, str] = {}
    for i, sent in enumerate(sentences):
        if PROTECTED.search(sent):
            continue
        if METADATA.search(sent):
            cuts[i] = "describes the source rather than the university"
        elif EMPTY_ASSERTION.match(sent):
            cuts[i] = "asserts that something exists without saying anything about it"
    for i, a in enumerate(sentences):
        if i in cuts:
            continue
        for j in range(i + 1, len(sentences)):
            if j in cuts or PROTECTED.search(sentences[j]):
                continue
            if restates(a, sentences[j]):
                cuts[j] = "repeats what an earlier sentence already said"
    return cuts


# --------------------------------------------------------------------------------------
# Model layer
# --------------------------------------------------------------------------------------

SYSTEM = """You decide whether sentences in a college-fit report earn their place.

Every sentence you see has ALREADY been verified as true and correctly cited. You are not
checking accuracy. You are answering one question per sentence:

  Does this tell the reader something about the university that changes their picture of it,
  or gives them something they could act on?

CUT a sentence when it:
  - describes a source: its title, author, date, photo credit, or that it is an article
  - states that something exists without saying anything about it ("the school has clubs")
  - repeats what another sentence in the same item already said
  - is a caveat that warns of nothing ("this is a blog post", "the recognition is for faculty")
  - is generic enough to be true of most universities

KEEP a sentence when it:
  - names something specific: a course code, a named lab, a person, a club, a tradition
  - gives a number, a requirement, a date, a process, an eligibility rule
  - says something a reader would not have assumed
  - honestly states a real limitation the reader needs ("selection is competitive",
    "the page does not say how to apply")

NEVER CUT, whatever else you think of the sentence:
  - eligibility, restrictions, or who a thing is open to (including citizenship or visa)
  - funding sources, cost, stipends, scholarships, fees
  - prerequisites, credits, units, grade requirements
  - deadlines, how to apply, who to contact
  - how selective or competitive something is
A sentence like "the site is supported by a grant from the National Science Foundation" looks
like trivia and is not: it tells an international reader the programme is probably closed to
them. Details that constrain who can do a thing are the most useful sentences in the document,
and they rarely look it.

Be strict about padding, generous about everything else. A report of six sentences that all
earn their place beats twenty that mostly do not -- but when genuinely unsure, KEEP. Deleting
something useful is worse than leaving one weak line.

Reply with JSON only: {"decisions":[{"id":"<id>","keep":true|false,"reason":"<short>"}]}
Return one decision for every id you were given."""


class UsefulError(RuntimeError):
    pass


def load_api_key(env_file: Path | None, env_name: str) -> str:
    key = os.environ.get(env_name, "").strip()
    if key:
        return key
    if env_file and env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith(f"{env_name}="):
                return line.split("=", 1)[1].strip()
    raise UsefulError(f"{env_name} is not set")


def judge_batch(client: httpx.Client, api_base: str, key: str, model: str,
                batch: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    payload = {
        "model": model,
        "temperature": CONFIG["temperature"],
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps({"sentences": batch}, ensure_ascii=False)},
        ],
    }
    delay = 2.0
    last: Exception | None = None
    for attempt in range(int(CONFIG["max_retries"])):
        try:
            r = client.post(f"{api_base.rstrip('/')}/chat/completions",
                            headers={"Authorization": f"Bearer {key}"}, json=payload)
            r.raise_for_status()
            body = json.loads(r.json()["choices"][0]["message"]["content"])
            out: dict[str, dict[str, Any]] = {}
            for d in body.get("decisions") or []:
                if isinstance(d, dict) and d.get("id"):
                    out[str(d["id"])] = {"keep": bool(d.get("keep", True)),
                                         "reason": str(d.get("reason") or "")}
            return out
        except Exception as exc:  # noqa: BLE001 - retried; on final failure we keep everything
            last = exc
            if attempt == int(CONFIG["max_retries"]) - 1:
                break
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
    # A gate that cannot reach its model must not silently delete a student's report. Failing
    # open here is the safe direction precisely because this pass only ever REMOVES: the worst
    # case is the padding we already ship today, not a missing section.
    print(f"  ! usefulness model unreachable ({last}); keeping everything in this batch",
          file=sys.stderr)
    return {}


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------

def gate(items: list[dict[str, Any]], *, use_model: bool, api_base: str, key: str,
         model: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fields = ("body", "why_it_matters", "caveat")
    units: list[dict[str, Any]] = []          # every sentence, flattened
    for idx, item in enumerate(items):
        for field in fields:
            for s_i, sent in enumerate(split_sentences(str(item.get(field) or ""))):
                units.append({"id": f"{idx}.{field}.{s_i}", "item": idx, "field": field,
                              "pos": s_i, "text": sent})

    decisions: dict[str, dict[str, Any]] = {}
    # deterministic first, per item and field, so restatement is judged within its own block
    by_block: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for u in units:
        by_block.setdefault((u["item"], u["field"]), []).append(u)
    for block in by_block.values():
        for i, reason in deterministic_cut([b["text"] for b in block]).items():
            decisions[block[i]["id"]] = {"keep": False, "reason": reason, "layer": "deterministic"}

    remaining = [u for u in units if u["id"] not in decisions]
    model_calls = 0
    if use_model and remaining:
        size = int(CONFIG["batch_size"])
        batches = [remaining[i:i + size] for i in range(0, len(remaining), size)]
        model_calls = len(batches)
        with httpx.Client(timeout=float(CONFIG["timeout_s"])) as client:
            with concurrent.futures.ThreadPoolExecutor(max_workers=int(CONFIG["max_workers"])) as pool:
                futs = [pool.submit(judge_batch, client, api_base, key, model,
                                    [{"id": u["id"], "text": u["text"]} for u in b])
                        for b in batches]
                for f in futs:
                    for uid, d in (f.result() or {}).items():
                        decisions[uid] = {**d, "layer": "model"}

    kept_items: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    dropped_items = 0
    for idx, item in enumerate(items):
        new_item = dict(item)
        removed_chars = 0
        original_chars = 0
        for field in fields:
            sents = split_sentences(str(item.get(field) or ""))
            if not sents:
                continue
            original_chars += sum(len(s) for s in sents)
            keep: list[str] = []
            for s_i, sent in enumerate(sents):
                d = decisions.get(f"{idx}.{field}.{s_i}")
                if d and not d.get("keep", True):
                    removed_chars += len(sent)
                    ledger.append({"item": idx, "category_code": item.get("category_code"),
                                   "headline": item.get("headline"), "field": field,
                                   "sentence": sent, "reason": d.get("reason"),
                                   "layer": d.get("layer")})
                else:
                    keep.append(sent)
            new_item[field] = " ".join(keep) if keep else (None if field == "caveat" else "")

        body_left = len(str(new_item.get("body") or ""))
        too_thin = body_left < int(CONFIG["min_body_chars_after"])
        gutted = original_chars and (removed_chars / original_chars) > float(CONFIG["max_cut_fraction"])
        # An item that still names a specific, findable thing is never dropped outright: the
        # gate deleted a whole "Undergraduate Microelectronics Commons Scholars Program" item
        # that a reviewer called a top-three match for this student. Losing a real lead costs
        # far more than keeping a thin paragraph about one.
        names_something = bool(re.search(
            r"\b[A-Z]{2,5}\s?\d{3,5}[A-Za-z]?\b|\b\d{1,2}\.\d{2,4}\b"   # a course code
            r"|\b(?:Lab|Laboratory|Center|Centre|Institute|Program|Programme|Society|Club|"
            r"Council|Department|Fellowship|Scholarship|Academy|Initiative)\b",
            f"{new_item.get('headline') or ''} {new_item.get('body') or ''}"))
        # Keep it even when almost nothing survived. A reviewer called the deleted
        # "Undergraduate Microelectronics Commons Scholars Program" a top-three match for
        # this student; its write-up was pure publication metadata, but the NAME and its
        # source link are the value. A reader can follow a name. They cannot follow a gap.
        if names_something:
            too_thin = gutted = False
        if too_thin or gutted:
            dropped_items += 1
            ledger.append({"item": idx, "category_code": item.get("category_code"),
                           "headline": item.get("headline"), "field": "*item*",
                           "sentence": None,
                           "reason": ("nothing substantial left after cutting"
                                      if too_thin else "most of the item was padding"),
                           "layer": "item"})
            continue
        kept_items.append(new_item)

    stats = {
        "items_in": len(items),
        "items_out": len(kept_items),
        "items_dropped": dropped_items,
        "sentences_total": len(units),
        "sentences_cut": sum(1 for d in decisions.values() if not d.get("keep", True)),
        "cut_deterministic": sum(1 for d in decisions.values()
                                 if not d.get("keep", True) and d.get("layer") == "deterministic"),
        "cut_by_model": sum(1 for d in decisions.values()
                            if not d.get("keep", True) and d.get("layer") == "model"),
        "model_calls": model_calls,
        "model_used": bool(use_model),
    }
    return kept_items, {"stats": stats, "ledger": ledger}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Delete report sentences that are true but not useful.")
    ap.add_argument("--report", required=True, type=Path, help="verified ReportItems JSON")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--ledger", type=Path, default=None, help="where to write the cut ledger")
    ap.add_argument("--no-model", action="store_true", help="deterministic layer only, no API calls")
    ap.add_argument("--model", default=CONFIG["model"])
    ap.add_argument("--api-base", default=CONFIG["api_base"])
    ap.add_argument("--env", default=str(Path(__file__).resolve().parent.parent / ".env"))
    args = ap.parse_args(argv)

    raw = json.loads(args.report.read_text(encoding="utf-8"))
    items = raw.get("items") if isinstance(raw, dict) else raw
    if not isinstance(items, list) or not items:
        raise UsefulError(f"{args.report}: no items")

    key = "" if args.no_model else load_api_key(Path(args.env), CONFIG["api_key_env"])
    kept, report = gate(items, use_model=not args.no_model, api_base=args.api_base,
                        key=key, model=args.model)

    out = {"items": kept, "usefulness": report["stats"]} if isinstance(raw, dict) else kept
    if isinstance(raw, dict):
        for k, v in raw.items():
            if k != "items":
                out.setdefault(k, v)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.ledger:
        args.ledger.write_text(json.dumps(report["ledger"], indent=2, ensure_ascii=False),
                               encoding="utf-8")

    s = report["stats"]
    print(f"usefulness: {s['sentences_cut']}/{s['sentences_total']} sentences cut "
          f"({s['cut_deterministic']} mechanical, {s['cut_by_model']} judged), "
          f"items {s['items_in']} -> {s['items_out']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except UsefulError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
