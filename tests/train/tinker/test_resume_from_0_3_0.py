"""A run that platoon 0.3.0 interrupted resumes here exactly (RFC 0019, milestone M4).

``tests/fixtures/resume-0.3.0/`` holds, for three runs that platoon 0.3.0 interrupted in the middle of a step
(``resume_0_3_0.py``: a weighted mixture whose slow entry is capped, one environment, a ``uniform`` mixture whose rows
have nested mappings with different keys), the files a resume reads: ``checkpoints.jsonl`` and ``tasks/``, whose
dispatcher state is of 0.3.0's format 1.  Each resumes here through the task samplers, as ``platoon train`` builds
them:

- no change of the sampling inputs is reported: the fingerprint of a 0.3.0 config is the one 0.3.0 saved;
- the groups the checkpoint kept train once, and are not rolled out again;
- every position of the order is consumed exactly once across both starts, or is still outstanding at the end;
- every group the resume runs, those it draws again included, runs the row platoon 0.3.0's order holds at its
  position (``rows.json``), which under ``uniform`` is the row's dataset form, not the listing's form 0.3.0 drew again.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import resume_0_3_0

from platoon.train.task_stats import read_events
from platoon.train.tinker.task_records import CHANGED_KEY

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "resume-0.3.0"


def _state(run_dir: Path, row: dict) -> dict:
    return json.loads((run_dir / row["dispatch"]["file"]).read_bytes())


@pytest.mark.parametrize("name", sorted(resume_0_3_0.RUNS))
def test_a_run_that_platoon_0_3_0_interrupted_resumes_exactly(name, tmp_path, monkeypatch):
    fixture = FIXTURES / name
    trainer, workflow, steps = resume_0_3_0.start(monkeypatch, name, tmp_path)
    run_dir = Path(trainer.run_log_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(fixture / "checkpoints.jsonl", run_dir / "checkpoints.jsonl")
    shutil.copytree(fixture / "tasks", run_dir / "tasks")
    expected = json.loads((fixture / "rows.json").read_text())
    rows = [json.loads(line) for line in (run_dir / "checkpoints.jsonl").read_text().splitlines()]
    interrupted = rows[-1]
    saved = _state(run_dir, interrupted)
    assert interrupted["name"].endswith("_interrupted") and saved["format"] == 1
    kept_keys = {tuple(key) for key in saved["kept"]["groups"]}
    position_of = {(start, seq): position for start, seq, _task, position in saved["outstanding"]}
    kept = {position_of[key] for key in kept_keys}
    waiting = {position_of[tuple(key)] for keys in saved["waiting"].values() for key in keys}
    assert kept and set(position_of.values()) - kept - waiting  # it kept groups, and some draws were running

    resume_0_3_0.run(trainer, workflow)

    lock = run_dir / "lock.json"
    assert not lock.is_file() or CHANGED_KEY not in json.loads(lock.read_text())
    # The kept groups trained once, and were not rolled out again.
    trained = [position for step in steps for position in step]
    assert kept <= set(trained) and not kept & set(workflow.started)
    # Every position consumed exactly once across both starts, or still outstanding at the end.
    events = list(read_events(run_dir))
    drawn = {(event["start"], event["seq"]): event for event in events if event["event"] == "draw"}
    fated = [drawn[(event["start"], event["seq"])]["position"] for event in events if event["event"] == "fate"]
    assert len(fated) == len(set(fated))
    final = _state(run_dir, [json.loads(line) for line in (run_dir / "checkpoints.jsonl").read_text().splitlines()][-1])
    outstanding = {position for _start, _seq, _task, position in final["outstanding"]}
    assert set(fated) | outstanding == set(range(final["next_position"])) and not set(fated) & outstanding
    consumed = [drawn[(e["start"], e["seq"])]["position"] for e in events if e.get("fate") == "consumed"]
    first_start = [position for position in consumed if position not in trained]
    assert sorted(first_start + trained) == sorted(consumed)
    # Each group of the resume ran the row 0.3.0's order holds at its position, those drawn again included.
    assert workflow.rows and all(row == expected[position] for position, row in workflow.rows.items())
    # The draws that were running or waited for their label were drawn again; those that ran ran their own rows (a
    # draw that waited may still wait for its label when the run ends).
    reissued = {event["position"] for event in drawn.values() if event.get("reason") == "reissue"}
    assert reissued & set(workflow.rows)
    if name == "uniform-mixture":  # rows that 0.3.0 drew again in their listing's form, which differs
        assert any(
            '"meta": {"m": null' in workflow.rows[position] or '"n": null' in workflow.rows[position]
            for position in reissued & set(workflow.rows)
        )
