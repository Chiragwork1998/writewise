"""Passage overlap must start on a word.

The carry-over between passages was a raw character slice, so half of the 26,960 passages in
USC's index began mid-word -- "lifornia President Michael V. Drake said" for California. The
written reports survived it, but the passage's first token is wasted and its embedding is
computed on a non-word.

Run:
  /Users/chirag/college-intel/.venv-crawl4ai/bin/python -m pytest \
      /Users/chirag/college-intel/wwrag/tests/test_chunking.py -q
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))

OVERLAP = 350


def tail_overlap(buf, want=OVERLAP):
    """Mirror of pipeline/export_rag.py:tail_overlap (that module runs work at import)."""
    if len(buf) <= want:
        return buf
    tail = buf[-want:]
    line = tail.find("\n")
    if 0 <= line <= want // 3:
        return tail[line + 1:]
    m = re.search(r"\s", tail)
    return tail[m.end():] if m else tail


def test_source_and_test_copy_agree():
    """If export_rag's version changes, this file has to change with it."""
    src = (Path(__file__).resolve().parents[2] / "pipeline" / "export_rag.py").read_text()
    body = src.split("def tail_overlap(", 1)[1]
    for line in ('tail = buf[-want:]', 'line = tail.find', 'return tail[line + 1:]',
                 'm = re.search(r"\\s", tail)', 'return tail[m.end():] if m else tail'):
        assert line in body, f"export_rag.tail_overlap no longer contains {line!r}"


def test_overlap_never_starts_mid_word():
    text = ("California President Michael V. Drake said the university will expand "
            "undergraduate research across every school. ") * 20
    out = tail_overlap(text)
    assert out, "overlap came back empty"
    assert not out[0].isspace()
    # the first token must be a whole word from the source, not a fragment of one
    assert re.match(r"^[A-Za-z]", out)
    first = out.split()[0].strip(".,;:")
    assert first in text.split() or first + "." in text.split(), f"fragment {first!r}"


def test_a_nearby_line_start_is_preferred():
    buf = "x" * 300 + "\nrest of the text here " + "y" * 40
    assert tail_overlap(buf).startswith("rest of the text")


def test_short_buffers_pass_through():
    assert tail_overlap("abc") == "abc"
    assert tail_overlap("") == ""


def test_a_buffer_with_no_whitespace_is_not_destroyed():
    """One long unbroken token must not come back empty -- some pages are a single URL."""
    buf = "z" * 900
    assert len(tail_overlap(buf)) == OVERLAP
