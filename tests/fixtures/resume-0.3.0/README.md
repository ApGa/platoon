# Runs that platoon 0.3.0 interrupted

Written by platoon 0.3.0 (commit 75685e8, tagged `v0.3.0`) running `tests/train/tinker/resume_0_3_0.py`, which
defines the three runs and how each was interrupted:

```
PYTHONPATH=<a checkout of platoon 0.3.0>:tests/train/tinker \
    python tests/train/tinker/resume_0_3_0.py tests/fixtures/resume-0.3.0
```

Each directory holds what a resume reads of one run's directory: `checkpoints.jsonl`, whose checkpoint paths are a
fake Tinker service's, and `tasks/` (the per-task statistics' snapshots, the dispatcher's states, of format 1, the
groups the interrupt checkpoint kept, and the event log). `rows.json` is the row 0.3.0's order holds at each of the
run's first 64 positions, as its dataloader draws it. `test_resume_from_0_3_0.py` resumes each run with this platoon.
