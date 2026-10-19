"""Node-local scratch (``platoon.utils.scratch``): the first variable that names it, in a fixed order, else ``/tmp``."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from platoon.utils.scratch import NODE_SCRATCH_VARIABLES, node_scratch

ORDER = ("PLATOON_NODE_SCRATCH", "PBS_LOCALDIR", "SLURM_TMPDIR", "TMPDIR")


@pytest.fixture
def unset(monkeypatch):
    """No variable names node-local scratch until a test sets one."""

    for variable in {*ORDER, *NODE_SCRATCH_VARIABLES}:
        monkeypatch.delenv(variable, raising=False)
    return monkeypatch


def test_node_scratch_is_the_first_of_its_variables_that_is_set_else_tmp(unset, tmp_path):
    for variable in ORDER:
        unset.setenv(variable, str(tmp_path / variable))

    for variable in ORDER:
        assert node_scratch() == tmp_path / variable
        unset.delenv(variable)

    assert node_scratch() == Path("/tmp")


def test_a_variable_set_to_the_empty_string_counts_as_unset(unset, tmp_path):
    unset.setenv("PLATOON_NODE_SCRATCH", "")
    unset.setenv("PBS_LOCALDIR", "")
    unset.setenv("TMPDIR", str(tmp_path))

    assert node_scratch() == tmp_path


def test_temporary_files_and_directories_go_to_node_scratch_which_is_created(unset, tmp_path):
    from platoon.utils import scratch

    unset.setenv("TMPDIR", str(tmp_path / "small-tmp"))  # what tempfile alone would use
    unset.setenv("PBS_LOCALDIR", str(tmp_path / "local"))  # not there yet: scratch makes it

    made = Path(scratch.mkdtemp("platoon-test-"))
    with scratch.temporary_directory() as directory, scratch.temporary_file() as unnamed:
        assert Path(directory).parent == tmp_path / "local"
        unnamed.write(b"x")
    with scratch.named_temporary_file("w", suffix=".txt") as named:
        assert Path(named.name).parent == tmp_path / "local"

    assert made.parent == tmp_path / "local" and made.name.startswith("platoon-test-")
    assert not (tmp_path / "small-tmp").exists()


def test_platoons_temporary_files_and_ipc_go_to_node_scratch(unset, tmp_path):
    """The rollout workers' IPC files, the process provider's sandboxes and the CLI harnesses' local state, which
    went to $TMPDIR alone (else /tmp), go to node-local scratch."""

    from platoon.harness.runner import HarnessRunner
    from platoon.sandbox.process import ProcessProvider
    from platoon.train.tinker.workers.protocol import IPC_DIRNAME, ipc_dir

    unset.setenv("TMPDIR", str(tmp_path / "tmp"))
    unset.setenv("PLATOON_NODE_SCRATCH", str(tmp_path / "scratch"))

    uid = os.getuid()
    assert ipc_dir() == str(tmp_path / "scratch" / f"{IPC_DIRNAME}-{uid}")
    assert ProcessProvider().root == tmp_path / "scratch" / f"platoon-sandboxes-{uid}"
    assert HarnessRunner(logs_dir=tmp_path / "logs")._state_root == tmp_path / "scratch" / f"platoon-harness-{uid}"


def test_the_ipc_directory_in_shared_scratch_is_this_users_alone(unset, tmp_path):
    """Node scratch may be shared by a node's users (/tmp): the rollout workers' files, which hold the gateway's key
    and which the trainer unpickles, go to a directory of this user's own, never to one that someone else made."""

    from platoon.train.tinker.workers.protocol import IPC_DIRNAME, ipc_dir

    unset.setenv("PLATOON_NODE_SCRATCH", str(tmp_path))
    mine = tmp_path / f"{IPC_DIRNAME}-{os.getuid()}"
    mine.mkdir(mode=0o777)
    mine.chmod(0o777)  # past the umask: as a directory made for others to write

    assert ipc_dir() == str(mine)
    assert stat.S_IMODE(mine.stat().st_mode) == 0o700  # made private

    mine.rmdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    mine.symlink_to(elsewhere)  # planted where this user's directory goes
    with pytest.raises(RuntimeError, match="is not a directory of this user's"):
        ipc_dir()


def test_no_module_makes_a_temporary_file_outside_node_scratch():
    """platoon makes its temporary files through platoon.utils.scratch, or names their directory: tempfile alone
    takes $TMPDIR, else /tmp, which a node may keep small."""

    import re

    package = Path(__file__).resolve().parents[2] / "platoon"
    calls = re.compile(r"tempfile\.(mkdtemp|mkstemp|TemporaryDirectory|TemporaryFile|NamedTemporaryFile|gettempdir)\(")
    found = []
    for path in sorted(package.rglob("*.py")):
        if path.name == "scratch.py" and path.parent.name == "utils":
            continue
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if calls.search(line) and "dir=" not in line:
                found.append(f"{path.relative_to(package.parent)}:{number}: {line.strip()}")
    assert found == []
