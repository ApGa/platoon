"""Named harness toolchains: the tree hash, lookup under the toolchain root, node-local staging, and the spec keys
that name a toolchain and the harness's memory."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from site_cache import use_site_cache

from platoon.harness import toolchains
from platoon.harness.spec import HarnessSpecError, parse_spec
from platoon.harness.toolchains import (
    MANIFEST,
    PreparedToolchain,
    Toolchain,
    ToolchainError,
    cache_root,
    prepare,
    resolve,
    stage,
    tree_hash,
)
from platoon.sandbox.base import Mount


@pytest.fixture(autouse=True)
def _a_fresh_process(monkeypatch):
    """Each test stages as a new process does: with no copy found current, or warned about, before."""

    monkeypatch.setattr(toolchains, "_SERVED", set())
    monkeypatch.setattr(toolchains, "_WARNED", set())


def make_toolchain(root: Path, name: str = "tc-1.0", *, build_path: str | None = None, manifest: bool = True) -> Path:
    """A small toolchain: an executable, a module, a symlink and a byte-compiled file."""

    path = root / name
    (path / "bin").mkdir(parents=True)
    (path / "lib" / "__pycache__").mkdir(parents=True)
    (path / "bin" / "tool").write_text("#!/bin/sh\necho tool\n")
    (path / "bin" / "tool").chmod(0o755)
    (path / "lib" / "mod.py").write_text("VALUE = 1\n")
    (path / "lib" / "__pycache__" / "mod.cpython-312.pyc").write_bytes(b"\x00compiled")
    (path / "bin" / "alias").symlink_to("tool")
    if manifest:
        data = {"name": name, "build_path": build_path or str(path.resolve()), "tree_hash": tree_hash(path)}
        (path / MANIFEST).write_text(json.dumps(data))
    return path


# --------------------------------------------------------------------------- the tree hash


def test_the_tree_hash_covers_contents_modes_and_links_but_not_the_manifest_or_byte_code(tmp_path):
    path = make_toolchain(tmp_path, manifest=False)
    digest = tree_hash(path)
    assert digest.startswith("sha256:") and len(digest) == len("sha256:") + 64

    (path / MANIFEST).write_text("{}")
    (path / "lib" / "__pycache__" / "other.pyc").write_bytes(b"more")
    assert tree_hash(path) == digest

    for change in (
        lambda: (path / "lib" / "mod.py").write_text("VALUE = 2\n"),
        lambda: (path / "bin" / "tool").chmod(0o644),
        lambda: ((path / "bin" / "alias").unlink(), (path / "bin" / "alias").symlink_to("mod")),
        lambda: (path / "lib" / "empty").mkdir(),
    ):
        change()
        assert tree_hash(path) != digest
        digest = tree_hash(path)


# --------------------------------------------------------------------------- lookup


def test_a_name_resolves_under_the_root_with_its_manifest(tmp_path, monkeypatch):
    path = make_toolchain(tmp_path / "root", build_path="/opt/toolchains/tc-1.0")

    toolchain = resolve("tc-1.0", root=tmp_path / "root")
    assert toolchain == Toolchain(
        name="tc-1.0", path=path, build_path="/opt/toolchains/tc-1.0", tree_hash=tree_hash(path)
    )

    use_site_cache(monkeypatch, tmp_path / "cache")
    in_site = make_toolchain(tmp_path / "cache" / "toolchains", build_path="/opt/toolchains/tc-1.0")
    assert resolve("tc-1.0").path == in_site  # the site's toolchain root when the caller names none


def test_a_toolchain_without_a_manifest_is_built_where_it_is(tmp_path):
    path = make_toolchain(tmp_path, manifest=False)

    assert resolve("tc-1.0", root=tmp_path) == Toolchain("tc-1.0", path, str(path.resolve()), None)


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        ("{not json", "cannot read"),
        ("[]", "JSON object"),
        ('{"build_path": "relative/path"}', "absolute path"),
        ('{"tree_hash": "md5:abc"}', "sha256"),
    ],
)
def test_a_bad_manifest_is_refused(tmp_path, manifest, message):
    path = make_toolchain(tmp_path, manifest=False)
    (path / MANIFEST).write_text(manifest)

    with pytest.raises(ToolchainError, match=message):
        resolve("tc-1.0", root=tmp_path)


def test_a_name_is_looked_up_under_the_toolchain_root(tmp_path, monkeypatch):
    """Without a root, a named toolchain is looked up under the cache root's toolchains, where platoon prepare builds
    it: the site profile's cache, else ~/.cache/platoon.  A root the caller names (platoon toolchain build --root)
    wins."""

    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    assert toolchains.toolchain_root() == tmp_path / "home" / ".cache" / "platoon" / "toolchains"
    with pytest.raises(ToolchainError, match=f"is not under the toolchain root {tmp_path / 'home'}/.cache/platoon"):
        resolve("tc-1.0")
    with pytest.raises(ToolchainError, match="is not under the toolchain root"):
        resolve("tc-1.0", root=tmp_path)
    use_site_cache(monkeypatch, tmp_path / "site")
    assert toolchains.toolchain_root() == tmp_path / "site" / "toolchains"
    assert toolchains.toolchain_root(tmp_path / "elsewhere") == tmp_path / "elsewhere"


# --------------------------------------------------------------------------- staging


def test_staging_copies_once_and_checks_the_tree_hash(tmp_path):
    path = make_toolchain(tmp_path / "root")
    toolchain = resolve("tc-1.0", root=tmp_path / "root")

    staged = stage(toolchain, cache=tmp_path / "cache")

    assert staged.parent == tmp_path / "cache" and staged.name.startswith("tc-1.0-")
    assert tree_hash(staged) == toolchain.tree_hash and os.access(staged / "bin" / "tool", os.X_OK)
    assert (staged / "bin" / "alias").is_symlink()
    marker = staged / "bin" / "tool"
    marker.write_text("#!/bin/sh\necho staged\n")
    assert stage(toolchain, cache=tmp_path / "cache") == staged  # already there: not copied again
    assert marker.read_text().endswith("staged\n")
    assert [entry.name for entry in (tmp_path / "cache").iterdir() if entry.name.endswith(".partial")] == []

    (path / "lib" / "mod.py").write_text("VALUE = 3\n")  # the toolchain changed after its manifest was written
    changed = Toolchain(toolchain.name, path, toolchain.build_path, toolchain.tree_hash)
    with pytest.raises(ToolchainError, match="does not match its manifest"):
        stage(changed, cache=tmp_path / "other-cache")
    assert [entry.name for entry in (tmp_path / "other-cache").iterdir() if not entry.name.endswith(".lock")] == []


def test_a_staged_copy_is_open_to_every_user(tmp_path):
    """A sandbox's other users run the copy too, such as an agent that runs as the user its Harbor task names: every
    user may read it, enter its directories and run its programs, whatever modes the build left (a umask of 077)."""

    path = make_toolchain(tmp_path / "root")
    for entry in [path, *path.rglob("*")]:
        if not entry.is_symlink():
            entry.chmod(stat.S_IMODE(entry.stat().st_mode) & 0o700)
    toolchain = resolve("tc-1.0", root=tmp_path / "root")

    staged = stage(toolchain, cache=tmp_path / "cache")

    modes = {
        entry.relative_to(staged).as_posix(): stat.S_IMODE(entry.stat().st_mode)
        for entry in [staged, *staged.rglob("*")]
        if not entry.is_symlink()
    }
    assert modes["."] & 0o555 == 0o555 and modes["bin"] & 0o555 == 0o555
    assert modes["bin/tool"] & 0o555 == 0o555 and modes["lib/mod.py"] & 0o444 == 0o444
    assert modes["lib/mod.py"] & 0o111 == 0  # what its owner may not run stays not runnable
    assert tree_hash(staged) == toolchain.tree_hash


def test_concurrent_stagings_copy_once(tmp_path, monkeypatch):
    make_toolchain(tmp_path / "root")
    toolchain = resolve("tc-1.0", root=tmp_path / "root")
    copies = []
    copytree = toolchains.shutil.copytree

    def counting_copytree(*args, **kwargs):
        if Path(args[0]) == toolchain.path:  # copytree calls itself for each subdirectory
            copies.append(args[0])
        return copytree(*args, **kwargs)

    monkeypatch.setattr(toolchains.shutil, "copytree", counting_copytree)
    results: list[Path] = []
    threads = [
        threading.Thread(target=lambda: results.append(stage(toolchain, cache=tmp_path / "cache"))) for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(copies) == 1 and len(set(results)) == 1


_STAGER = """
import os, sys, time
from pathlib import Path
from platoon.harness import toolchains

root, cache, copies, go = sys.argv[1:5]
toolchain = toolchains.resolve("tc-1.0", root=root)
copytree = toolchains.shutil.copytree


def counting_copytree(*args, **kwargs):
    if Path(args[0]) == toolchain.path:  # copytree calls itself for each subdirectory
        with open(copies, "a") as handle:
            handle.write(f"{os.getpid()}\\n")
    return copytree(*args, **kwargs)


toolchains.shutil.copytree = counting_copytree
while not os.path.exists(go):  # all start together
    time.sleep(0.005)
print(toolchains.stage(toolchain, cache=cache))
"""


def test_stagings_in_many_processes_at_once_copy_once_and_agree(tmp_path):
    """Processes that stage a toolchain at once (a node's first rollouts, rollouts of two jobs, a pre-stage beside
    them) take turns on its lock: one copies it, the others wait and find the copy, and every one gets the same
    checked copy."""

    make_toolchain(tmp_path / "root")
    toolchain = resolve("tc-1.0", root=tmp_path / "root")
    script = tmp_path / "stager.py"
    script.write_text(_STAGER)
    arguments = [str(tmp_path / name) for name in ("root", "cache", "copies", "go")]
    stagers = [
        subprocess.Popen([sys.executable, str(script), *arguments], stdout=subprocess.PIPE, text=True) for _ in range(8)
    ]
    (tmp_path / "go").touch()
    results = [stager.communicate(timeout=120)[0].strip() for stager in stagers]

    assert [stager.returncode for stager in stagers] == [0] * 8
    assert len(set(results)) == 1 and Path(results[0]).parent == tmp_path / "cache"
    assert len((tmp_path / "copies").read_text().split()) == 1  # one of them copied it
    assert tree_hash(results[0]) == toolchain.tree_hash


def test_a_current_copy_is_returned_without_its_lock_a_copy_or_a_hash(tmp_path, monkeypatch):
    """Once a copy is staged and checked in this staging session, a rollout's staging is a no-op: it neither takes
    the lock nor copies nor hashes anything."""

    make_toolchain(tmp_path / "root")
    toolchain = resolve("tc-1.0", root=tmp_path / "root")
    staged = stage(toolchain, cache=tmp_path / "cache")

    def refuse(*args, **kwargs):
        raise AssertionError("a current copy is neither copied, hashed nor locked again")

    monkeypatch.setattr(toolchains, "tree_digest", refuse)
    monkeypatch.setattr(toolchains.shutil, "copytree", refuse)
    monkeypatch.setattr(toolchains.fcntl, "flock", refuse)
    assert [stage(toolchain, cache=tmp_path / "cache") for _ in range(3)] == [staged] * 3


def _hashes(monkeypatch) -> list[str]:
    """Count the tree hashes staging computes (each a read of the whole copy)."""

    hashed: list[str] = []
    digest = toolchains.tree_digest

    def counting(directory, *args, **kwargs):
        hashed.append(Path(directory).name)
        return digest(directory, *args, **kwargs)

    monkeypatch.setattr(toolchains, "tree_digest", counting)
    return hashed


def test_a_copy_kept_from_an_earlier_job_is_checked_once_in_each_job_that_finds_it(tmp_path, monkeypatch):
    """A copy that outlived its job (node-local disk that persists) is kept by the next job on the node when its files
    still match the manifest's tree hash: the first staging of each job (staging session) hashes it again, under the
    lock, and stamps it with the session; the session's later stagings, and those of a job that checked it before,
    find it current.  So jobs that share the node and the copy each check it once."""

    for variable in toolchains.JOB_ID_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    make_toolchain(tmp_path / "root")
    toolchain = resolve("tc-1.0", root=tmp_path / "root")
    cache = tmp_path / "cache"
    monkeypatch.setenv("PBS_JOBID", "100.server")
    first = toolchains.staging_session()
    staged = stage(toolchain, cache=cache)
    hashed = _hashes(monkeypatch)

    monkeypatch.setenv("PBS_JOBID", "101.server")  # the next job on the node
    assert toolchains.staging_session() != first
    assert stage(toolchain, cache=cache) == staged
    assert hashed == [staged.name]  # checked, not copied
    assert stage(toolchain, cache=cache) == staged and hashed == [staged.name]  # once in the job
    monkeypatch.setenv("PBS_JOBID", "100.server")  # the first job, still running beside it
    assert stage(toolchain, cache=cache) == staged and hashed == [staged.name]
    stamp = json.loads((cache / f".{staged.name}.staged").read_text())
    assert stamp["tree_hash"] == toolchain.tree_hash and len(stamp["sessions"]) == 2


def test_a_kept_copy_that_no_longer_matches_its_manifest_is_copied_again(tmp_path, monkeypatch, caplog):
    """A node's cleanup may have removed or changed files of a copy kept from an earlier job: the next job's first
    staging finds that its tree hash no longer matches, and copies the toolchain again, beside it, then renames the
    fresh copy into its place; the old one is set aside whole, and a warning names it."""

    make_toolchain(tmp_path / "root")
    toolchain = resolve("tc-1.0", root=tmp_path / "root")
    cache = tmp_path / "cache"
    monkeypatch.setenv("PBS_JOBID", "200.server")
    staged = stage(toolchain, cache=cache)
    (staged / "lib" / "mod.py").unlink()

    monkeypatch.setenv("PBS_JOBID", "201.server")
    lines: list[str] = []
    with caplog.at_level("WARNING", logger=toolchains.__name__):
        results = toolchains.prestage(["tc-1.0"], root=tmp_path / "root", cache=cache, log=lines.append)

    assert results["tc-1.0"].staged == staged
    assert (staged / "lib" / "mod.py").read_text() == "VALUE = 1\n" and tree_hash(staged) == toolchain.tree_hash
    assert "no longer matched its manifest: copied again" in lines[0]
    (stale,) = [entry for entry in cache.iterdir() if entry.name.endswith(".stale")]
    assert stale.name.startswith(f".{staged.name}.") and (stale / "bin" / "tool").is_file()
    assert sorted(entry.name for entry in cache.iterdir()) == sorted(
        [staged.name, f".{staged.name}.lock", f".{staged.name}.staged", stale.name]
    )
    assert f"the old copy is set aside at {stale}" in caplog.text


def test_replacing_a_kept_copy_keeps_the_old_one_whole_for_the_rollouts_that_run_from_it(tmp_path, monkeypatch):
    """Jobs that share a node share its kept copy, and a sandbox's bind mount follows the directory it mounted.  When
    one job replaces the copy, the rollouts another job runs from it keep every file of it, not yet opened ones too;
    that job's next rollouts take the fresh copy, current for them without another check."""

    make_toolchain(tmp_path / "root")
    toolchain = resolve("tc-1.0", root=tmp_path / "root")
    cache = tmp_path / "cache"
    monkeypatch.setenv("PBS_JOBID", "300.server")
    staged = stage(toolchain, cache=cache)
    mounted = os.open(staged, os.O_RDONLY | os.O_DIRECTORY)  # what a running sandbox holds, as its bind mount does
    try:
        (staged / "lib" / "mod.py").unlink()  # a cleanup removes a file: the copy no longer matches its manifest
        monkeypatch.setenv("PBS_JOBID", "301.server")  # another job on the node replaces the copy
        assert toolchains._stage(toolchain, cache)[1] == "replaced"

        assert {"bin", "lib"} <= set(os.listdir(mounted))
        with open(os.open("bin/tool", os.O_RDONLY, dir_fd=mounted)) as tool:  # opened only now
            assert tool.read() == "#!/bin/sh\necho tool\n"
        assert (staged / "lib" / "mod.py").is_file() and tree_hash(staged) == toolchain.tree_hash
    finally:
        os.close(mounted)
    hashed = _hashes(monkeypatch)
    monkeypatch.setattr(toolchains, "_SERVED", set())  # a new rollout process of the first job
    monkeypatch.setenv("PBS_JOBID", "300.server")
    assert toolchains._stage(toolchain, cache) == (staged, "current") and hashed == []


def test_a_cache_that_cannot_record_a_check_serves_its_copy_unchecked_with_one_warning(tmp_path, monkeypatch, caplog):
    """A kept copy in a cache this process cannot write a stamp to (a read-only directory) cannot have its check
    recorded, so every staging would hash it again, one after another under the lock: it is used as it is, as before
    platoon checked kept copies, with one warning, and without the lock once the process has found it so.  What a
    killed staging left there, which no staging can remove, is no obstacle."""

    if os.geteuid() == 0:
        pytest.skip("root writes to a read-only directory")
    make_toolchain(tmp_path / "root")
    toolchain = resolve("tc-1.0", root=tmp_path / "root")
    cache = tmp_path / "cache"
    monkeypatch.setenv("PBS_JOBID", "400.server")
    staged = stage(toolchain, cache=cache)
    (cache / f".{staged.name}.staged.0badc0de.tmp").write_text("{")  # a stamp a killed staging half wrote, kept
    cache.chmod(0o555)
    try:
        monkeypatch.setenv("PBS_JOBID", "401.server")
        hashed = _hashes(monkeypatch)
        with caplog.at_level("WARNING", logger=toolchains.__name__):
            assert toolchains._stage(toolchain, cache) == (staged, "unchecked")
            monkeypatch.setattr(toolchains.fcntl, "flock", _refuse)
            assert [stage(toolchain, cache=cache) for _ in range(3)] == [staged] * 3
    finally:
        cache.chmod(0o755)
    assert hashed == []
    assert caplog.text.count("is used without being checked against its manifest: no stamp can be written") == 1


def test_a_read_only_cache_without_a_lock_file_serves_its_copy_and_cannot_stage_a_new_one(tmp_path, monkeypatch):
    """A cache filled ahead and made read-only, with no lock file: its copy is used as it is (as before platoon checked
    kept copies), and a toolchain it does not hold cannot be staged there."""

    if os.geteuid() == 0:
        pytest.skip("root writes to a read-only directory")
    make_toolchain(tmp_path / "root")
    make_toolchain(tmp_path / "root", "tc-2.0")
    toolchain = resolve("tc-1.0", root=tmp_path / "root")
    cache = tmp_path / "cache"
    monkeypatch.setenv("PBS_JOBID", "500.server")
    staged = stage(toolchain, cache=cache)
    (cache / f".{staged.name}.lock").unlink()
    cache.chmod(0o555)
    try:
        monkeypatch.setenv("PBS_JOBID", "501.server")
        assert toolchains._stage(toolchain, cache) == (staged, "unchecked")
        with pytest.raises(ToolchainError, match="cannot stage toolchain 'tc-2.0'"):
            prepare("tc-2.0", root=tmp_path / "root", cache=cache)
    finally:
        cache.chmod(0o755)


_FULL_DISK = """
import resource, sys
from platoon.harness import toolchains

root, cache = sys.argv[1:3]
toolchain = toolchains.resolve("tc-1.0", root=root)
resource.setrlimit(resource.RLIMIT_FSIZE, (0, resource.getrlimit(resource.RLIMIT_FSIZE)[1]))  # nothing more is written
print(toolchains._stage(toolchain, cache)[1], toolchains._stage(toolchain, cache)[1])
"""


def test_a_full_disk_serves_a_kept_copy_unchecked(tmp_path, monkeypatch):
    """A disk that takes no more bytes (a file size limit of 0 stands for it): the kept copy is used unchecked, once
    warned, instead of being hashed again by every staging."""

    make_toolchain(tmp_path / "root")
    toolchain = resolve("tc-1.0", root=tmp_path / "root")
    monkeypatch.setenv("PBS_JOBID", "600.server")
    stage(toolchain, cache=tmp_path / "cache")
    script = tmp_path / "full_disk.py"
    script.write_text(_FULL_DISK)
    env = {**os.environ, "PBS_JOBID": "601.server"}

    run = subprocess.run(
        [sys.executable, str(script), str(tmp_path / "root"), str(tmp_path / "cache")],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert run.returncode == 0, run.stderr
    assert run.stdout.split() == ["unchecked", "current"]
    assert run.stderr.count("is used without being checked against its manifest: no stamp can be written") == 1


def test_a_process_reads_the_stamp_once_in_a_session(tmp_path, monkeypatch):
    """A process that found the copy current in its session does not read the stamp again."""

    make_toolchain(tmp_path / "root")
    toolchain = resolve("tc-1.0", root=tmp_path / "root")
    staged = stage(toolchain, cache=tmp_path / "cache")
    monkeypatch.setattr(toolchains, "_SERVED", set())  # another process of the session
    assert stage(toolchain, cache=tmp_path / "cache") == staged  # reads the stamp
    monkeypatch.setattr(toolchains, "_read_stamp", _refuse)
    assert stage(toolchain, cache=tmp_path / "cache") == staged


def _refuse(*args, **kwargs):
    raise AssertionError("not called once the process found the copy current")


def test_a_copy_an_older_platoon_staged_is_checked_before_it_is_used(tmp_path, monkeypatch):
    """A copy with no stamp (staged by a platoon that wrote none) is hashed once, then stamped."""

    make_toolchain(tmp_path / "root")
    toolchain = resolve("tc-1.0", root=tmp_path / "root")
    staged = stage(toolchain, cache=tmp_path / "cache")
    (tmp_path / "cache" / f".{staged.name}.staged").unlink()
    monkeypatch.setattr(toolchains, "_SERVED", set())  # a process of the platoon that finds the copy
    hashed = _hashes(monkeypatch)

    assert stage(toolchain, cache=tmp_path / "cache") == staged
    assert stage(toolchain, cache=tmp_path / "cache") == staged
    assert hashed == [staged.name]


def test_prestaging_stages_each_toolchain_once_and_reports_what_it_did(tmp_path, monkeypatch):
    for variable in toolchains.JOB_ID_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    make_toolchain(tmp_path / "root")
    cache = tmp_path / "cache"
    lines: list[str] = []

    first = toolchains.prestage(
        ["tc-1.0", "missing-1.0", "tc-1.0"], root=tmp_path / "root", cache=cache, log=lines.append
    )
    again = toolchains.prestage(["tc-1.0"], root=tmp_path / "root", cache=cache, log=lines.append)

    assert list(first) == ["tc-1.0", "missing-1.0"]  # each name once
    assert isinstance(first["tc-1.0"], PreparedToolchain) and first["tc-1.0"] == again["tc-1.0"]
    assert isinstance(first["missing-1.0"], ToolchainError)
    assert lines[0].startswith(f"toolchain tc-1.0 at {first['tc-1.0'].staged}: copied (")
    assert "could not stage the toolchain missing-1.0 ahead of the rollouts; each rollout stages it itself" in lines[1]
    assert ": was staged already (" in lines[2]


def test_the_staging_session_is_the_boot_and_the_scheduler_job(monkeypatch):
    for variable in toolchains.JOB_ID_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    alone = toolchains.staging_session()
    assert alone.startswith("boot=") and alone.endswith(" job=")

    monkeypatch.setenv("SLURM_JOB_ID", "77")
    assert toolchains.staging_session() == alone + "77"
    monkeypatch.setenv("PBS_JOBID", "88.server")  # the first of the variables that is set
    assert toolchains.staging_session() == alone + "88.server"


def test_staging_removes_the_partial_copies_that_killed_stagings_left(tmp_path):
    make_toolchain(tmp_path / "root")
    toolchain = resolve("tc-1.0", root=tmp_path / "root")
    name = stage(toolchain, cache=tmp_path / "probe").name  # a staged copy has the same name in every cache
    cache = tmp_path / "cache"
    left = [cache / f".{name}.{suffix}.partial" for suffix in ("0badc0de", "5ca1ab1e")]
    for partial in left:
        (partial / "bin").mkdir(parents=True)
        (partial / "bin" / "tool").write_text("half copied")
    others = [
        cache / ".tc-2.0-0123456789abcdef.0badc0de.partial",  # another toolchain's, which its own stager may write
        cache / f".{name}.x-0123456789abcdef.0badc0de.partial",  # that of a toolchain named "<this copy's name>.x"
    ]
    for partial in others:
        partial.mkdir()

    staged = stage(toolchain, cache=cache)

    assert staged == cache / name and tree_hash(staged) == toolchain.tree_hash
    assert [partial for partial in left if partial.exists()] == []
    assert all(partial.is_dir() for partial in others)


def test_staged_copies_go_to_node_local_scratch_unless_a_cache_is_set(tmp_path, monkeypatch):
    for variable in (toolchains.TOOLCHAIN_CACHE_ENV, "PLATOON_NODE_SCRATCH", "PBS_LOCALDIR", "SLURM_TMPDIR"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    monkeypatch.setenv("PBS_LOCALDIR", str(tmp_path / "job-local"))
    copies = f"platoon-toolchains-{os.getuid()}"

    assert cache_root() == tmp_path / "job-local" / copies  # the job's node-local directory before $TMPDIR

    monkeypatch.setenv("PLATOON_NODE_SCRATCH", str(tmp_path / "node"))
    make_toolchain(tmp_path / "root")
    assert prepare("tc-1.0", root=tmp_path / "root").staged.parent == tmp_path / "node" / copies

    monkeypatch.setenv(toolchains.TOOLCHAIN_CACHE_ENV, str(tmp_path / "cache"))
    assert cache_root() == tmp_path / "cache" and cache_root(tmp_path / "given") == tmp_path / "given"


def test_a_toolchain_without_a_tree_hash_is_versioned_by_its_files(tmp_path):
    path = make_toolchain(tmp_path / "root", manifest=False)
    toolchain = resolve("tc-1.0", root=tmp_path / "root")

    first = stage(toolchain, cache=tmp_path / "cache")
    (path / "lib" / "mod.py").write_text("VALUE = 22\n")
    second = stage(toolchain, cache=tmp_path / "cache")

    assert first != second and (second / "lib" / "mod.py").read_text() == "VALUE = 22\n"


def test_a_prepared_toolchain_mounts_its_copy_at_its_build_path(tmp_path, monkeypatch):
    monkeypatch.setenv(toolchains.TOOLCHAIN_CACHE_ENV, str(tmp_path / "cache"))
    make_toolchain(tmp_path / "root", build_path="/opt/toolchains/tc-1.0")

    prepared = prepare("tc-1.0", root=tmp_path / "root")

    assert prepared.staged.parent == tmp_path / "cache"
    assert prepared.mount() == Mount(source=str(prepared.staged), target="/opt/toolchains/tc-1.0", read_only=True)
    assert prepared.sandbox_path == "/opt/toolchains/tc-1.0"
    assert prepared.host_path == str(prepared.staged)  # the build path is not on this host


def test_on_the_host_a_toolchain_runs_from_its_build_path_when_it_is_there(tmp_path):
    path = make_toolchain(tmp_path / "root")
    toolchain = resolve("tc-1.0", root=tmp_path / "root")

    prepared = PreparedToolchain(toolchain=toolchain, staged=tmp_path / "cache" / "tc-1.0-x")

    assert prepared.host_path == str(path.resolve())  # no relocation where nothing needs it


# --------------------------------------------------------------------------- spec keys


def _spec(**tables) -> dict:
    return {"name": "agent", "version": "1", "launch": {"argv": ["{toolchain}/bin/tool"]}, **tables}


def test_a_spec_names_its_toolchain_and_the_memory_its_processes_need():
    spec = parse_spec(_spec(install={"toolchain": "openhands-1.54.0"}, resources={"memory_mb": 2048}))

    assert spec.install.toolchain == "openhands-1.54.0" and spec.resources.memory_mb == 2048
    assert parse_spec(_spec(install={"toolchain": "/opt/tc"})).resources.memory_mb is None


@pytest.mark.parametrize(
    ("tables", "message"),
    [
        ({"install": {"toolchain": "../escape"}}, "absolute path or a name"),
        ({"install": {"toolchain": "has space"}}, "absolute path or a name"),
        ({"resources": {"memory_mb": 0}}, "positive number"),
        ({"resources": {"memory_mb": 1.5}}, "whole number"),
        ({"resources": {"cpus": 2}}, "unknown key"),
    ],
)
def test_bad_toolchain_names_and_resources_are_refused(tables, message):
    tables.setdefault("install", {"toolchain": "tc"})
    with pytest.raises(HarnessSpecError, match=message):
        parse_spec(_spec(**tables))
