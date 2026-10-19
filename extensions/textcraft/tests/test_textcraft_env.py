"""TextCraft's environments without platoon: ``platoon_textcraft.env`` imports neither platoon nor IPython, and serves
the task files the package ships (``data/``, pinned by ``data/SHA256SUMS``), with their splits."""

from __future__ import annotations

import hashlib
import subprocess
import sys

from platoon_textcraft import env


def test_the_environments_import_neither_platoon_nor_ipython():
    code = (
        "import sys, platoon_textcraft.env\n"
        "print(sorted({name.split('.')[0] for name in sys.modules} & {'platoon', 'IPython'}))"
    )

    result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=300)

    assert (result.returncode, result.stdout.strip()) == (0, "[]"), result.stderr


def test_the_task_files_are_the_ones_sha256sums_pins():
    lines = (env.DATA_DIR / "SHA256SUMS").read_text().splitlines()
    pinned = {name: digest for digest, name in (line.split() for line in lines)}

    assert sorted(pinned) == sorted(path.name for path in env.DATA_DIR.glob("*.jsonl"))
    for name, digest in pinned.items():
        assert hashlib.sha256((env.DATA_DIR / name).read_bytes()).hexdigest() == digest, name


def test_the_environments_serve_their_splits():
    synth, original = env.TextCraftSynthEnvironment, env.TextCraftEnvironment

    assert (len(synth.list_tasks("train")), len(synth.list_tasks("val"))) == (2522, 632)
    assert len(synth.list_tasks("train-medium")) == 852  # the RAO training set
    assert (len(original.list_tasks("train")), len(original.list_tasks("val"))) == (1000, 100)
    # Every task keeps its gold trajectory: the Harbor export builds each task's reference solution from it.
    assert all(spec["misc"]["gold_trajectory"] for spec in (*synth.list_tasks("train"), *synth.list_tasks("val")))
