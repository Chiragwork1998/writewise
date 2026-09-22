"""Regression tests for wwrag/run.py -- the orchestrator's seams on a RESUMED run.

run.py exists to normalise two places where the six modules disagree. Seam 1 turns
retrieve.py's `{category_code: [unit]}` into the flat `{"units": [...]}` that verify.py and
report.py are both handed on the command line.

That normalisation is a property of the evidence FILE, not of the retrieve STAGE. It used
to live inside the retrieve branch of the driver loop, so `--from-stage generate` produced
a run with evidence.json on disk, no evidence_units.json, and a verify command still
pointing at the file that was never written. The run died at verify -- after the generate
stage had been billed to a real card.

No subprocess is ever spawned here: run_stage is stubbed, so these tests cost nothing and
prove exactly the thing that used to cost money.

Run:
  /Users/chirag/college-intel/.venv-crawl4ai/bin/python -m pytest \
      /Users/chirag/college-intel/wwrag/tests/test_run_resume_seams.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run as runner  # noqa: E402


EVIDENCE = {
    "CUL": [{"unit_id": "u-cul", "source_url": "https://example.edu/culture"}],
    "RES": [{"unit_id": "u-res", "source_url": "https://example.edu/research"},
            {"unit_id": "u-cul", "source_url": "https://example.edu/culture"}],
    "DIV": [],
}

PROFILE = {"student_id": "stu_test", "first_name": "Anika", "level": "undergraduate",
           "activities": [{"name": "Robotics", "evidence_line": "Robotics Club - captain"}]}


@pytest.fixture
def run_dir(tmp_path):
    """A run directory shaped like one a previous run left behind."""
    out = tmp_path / "run"
    out.mkdir()
    (out / "profile.json").write_text(json.dumps(PROFILE), encoding="utf-8")
    (out / "evidence.json").write_text(json.dumps(EVIDENCE), encoding="utf-8")
    (tmp_path / "resume.txt").write_text("Anika\nRobotics Club - captain\n", encoding="utf-8")
    return out


@pytest.fixture
def stub_stages(monkeypatch):
    """Replace the subprocess runner. Records the stages reached, then stops the run.

    Stopping at the first stage is the point: it proves whether the seam was normalised
    BEFORE anything was spawned, rather than part-way through a loop that has already paid
    for generation.
    """
    reached: list[str] = []

    def fake_run_stage(stage, cmd, record, tail_lines=25):
        reached.append(stage)
        raise runner.StageError(stage, "stubbed: no subprocess in tests")

    monkeypatch.setattr(runner, "run_stage", fake_run_stage)
    return reached


def argv_for(run_dir, **over):
    args = {"--resume": str(run_dir.parent / "resume.txt"),
            "--college": "example",
            "--out-dir": str(run_dir),
            "--from-stage": "generate",
            "--to-stage": "verify"}
    args.update(over)
    out = ["--skip-index"]
    for k, v in args.items():
        out += [k, v]
    return out


# --------------------------------------------------------------------------------------

def test_resuming_after_retrieve_still_normalises_the_evidence_seam(run_dir, stub_stages):
    """--from-stage generate must still leave verify's --evidence file on disk.

    Before the fix this file simply did not exist, and the crash landed at verify: one
    stage too late, with the writer model already paid for.
    """
    flat = run_dir / "evidence_units.json"
    assert not flat.exists()

    with pytest.raises(runner.StageError):
        runner.main(argv_for(run_dir))

    assert flat.exists(), "verify.py is handed this path; it must exist before verify runs"
    units = json.loads(flat.read_text(encoding="utf-8"))["units"]
    assert [u["unit_id"] for u in units] == ["u-cul", "u-res"], "flattened and de-duplicated"
    assert stub_stages == ["generate"], "the seam must be closed before any stage is spawned"


def test_resuming_straight_at_report_normalises_the_evidence_seam(run_dir, stub_stages):
    """report.py is handed the same flat file, so resuming at report has the same need."""
    with pytest.raises(runner.StageError):
        runner.main(argv_for(run_dir, **{"--from-stage": "report", "--to-stage": "report"}))
    assert (run_dir / "evidence_units.json").exists()
    assert stub_stages == ["report"]


def test_a_resumed_run_with_no_retrieval_output_fails_before_any_paid_stage(run_dir,
                                                                           stub_stages):
    """If retrieval's output is not on disk either, say so at preflight.

    Discovering it at verify means the student has been charged for generation and has
    nothing to show for it.
    """
    (run_dir / "evidence.json").unlink()
    with pytest.raises(runner.StageError) as caught:
        runner.main(argv_for(run_dir))
    assert caught.value.stage == "preflight"
    assert "evidence.json" in caught.value.message
    assert stub_stages == [], "nothing may be spawned once the inputs are known to be missing"


def test_normalising_is_skipped_when_retrieve_will_produce_it_itself(run_dir, stub_stages):
    """A full run must not pre-normalise from a stale evidence.json.

    retrieve.py is about to overwrite it, and the retrieve branch flattens its own fresh
    output. Doing it twice would publish counts from the previous run.
    """
    with pytest.raises(runner.StageError):
        runner.main(argv_for(run_dir, **{"--from-stage": "retrieve", "--to-stage": "verify"}))
    assert not (run_dir / "evidence_units.json").exists()
    assert stub_stages == ["retrieve"]


def test_stage_ranges_that_never_read_the_flat_file_do_no_work(run_dir, stub_stages):
    """profile-only runs have nothing downstream to normalise for; leave the disk alone."""
    with pytest.raises(runner.StageError):
        runner.main(argv_for(run_dir, **{"--from-stage": "profile", "--to-stage": "profile"}))
    assert not (run_dir / "evidence_units.json").exists()
    assert stub_stages == ["profile"]


def test_ensure_flat_evidence_is_bound_to_the_file_not_to_the_stage_list(tmp_path):
    """The unit-level statement of the same rule, independent of the driver loop."""
    ev = tmp_path / "evidence.json"
    flat = tmp_path / "evidence_units.json"
    ev.write_text(json.dumps(EVIDENCE), encoding="utf-8")
    record = {"counts": {}}

    for stages in (["generate", "verify"], ["verify"], ["report"], ["generate", "verify", "report"]):
        flat.unlink(missing_ok=True)
        runner.ensure_flat_evidence(ev, flat, stages, record)
        assert flat.exists(), f"{stages} reads the flat file and must get one"

    flat.unlink(missing_ok=True)
    runner.ensure_flat_evidence(ev, flat, ["index", "profile"], record)
    assert not flat.exists(), "nothing downstream reads it; do not write it"
