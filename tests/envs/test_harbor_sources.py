"""Fetching Harbor datasets from the Harbor Hub (a fake Harbor CLI), Harbor's legacy registry, git repositories and
adapters (local git repositories), offline: what each locks, what moves, and what rollout workers never do."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from site_cache import use_site_cache

from platoon.envs.harbor import HarborEnvironmentService, references, sources
from platoon.envs.harbor.compose import COMPOSE_RULE
from platoon.envs.harbor.fetch import FetchError, ensure_lock, fetch, fetch_config, fetch_if_needed, is_pinned
from platoon.envs.harbor.hashing import HASH_RULE, content_hash
from platoon.envs.harbor.lock import LockError, read_lock, verify_lock
from platoon.envs.harbor.locked import LockedTasks
from platoon.envs.service import EnvSetupError
from platoon.sandbox.process import ProcessProvider

DIGEST = "sha256:" + "d" * 64

# A stand-in for the Harbor CLI's `--version`, `version show` and `download --export`, serving the packages of the
# JSON file $FAKE_HUB names and appending each command line to $FAKE_HUB_CALLS.
FAKE_CLI = r"""
import json, os, shutil, sys
from pathlib import Path

hub = json.loads(Path(os.environ["FAKE_HUB"]).read_text())
with open(os.environ["FAKE_HUB_CALLS"], "a") as calls:
    calls.write(json.dumps(sys.argv[1:]) + "\n")
args = sys.argv[1:]
if args == ["--version"]:
    print("0.24.0")
    sys.exit(0)


def resolve(reference):
    name, _, ref = reference.partition("@")
    package = hub["packages"].get(name)
    if package is None:
        print(f"Error: Package '{name}' was not found or you do not have access.")
        sys.exit(1)
    ref = ref or "latest"
    for version in package["versions"]:
        if ref in (str(version["revision"]), version["digest"]) or ref in version["tags"]:
            return name, package, version
    print(f"Error: Version not found: {reference}")
    sys.exit(1)


if args[:2] == ["version", "show"]:
    name, package, version = resolve(args[2])
    if "--tasks" in args and package["type"] != "dataset":
        print("Error: --tasks is only valid for dataset versions")
        sys.exit(1)
    shown = {"package": name, "type": package["type"], "revision": version["revision"],
             "content_hash": version["digest"], "tags": version["tags"], "yanked_at": None}
    if "--tasks" in args:
        shown["tasks"] = [
            {"task_version": {"content_hash": task["digest"].removeprefix("sha256:"),
                              "package": {"name": task["name"], "org": {"name": task["org"]}}}, "available": True}
            for task in version["tasks"]
        ]
    print(json.dumps(shown))
elif args[0] == "download" and "--export" in args:
    flaky = Path(os.environ["FAKE_HUB_CALLS"] + ".flaky")
    if flaky.exists():  # one transient failure, as a reset stream gives
        flaky.unlink()
        print("httpx.RemoteProtocolError: <StreamReset stream_id:1, error_code:2>")
        sys.exit(1)
    name, package, version = resolve(args[1])
    output = Path(args[args.index("-o") + 1])
    if package["type"] == "dataset":  # links stay links, as tarfile's data filter keeps those inside the export
        for task in version["tasks"]:
            shutil.copytree(hub["files"][task["digest"]], output / name.split("/")[1] / task["name"], symlinks=True)
    else:
        shutil.copytree(hub["files"][version["digest"]], output / name.split("/")[1], symlinks=True)
    print("Successfully downloaded")
else:
    print(f"fake harbor: unexpected arguments {args}")
    sys.exit(2)
"""


def _task(root: Path, name: str, *, reward: str = "1", extra: dict[str, str] | None = None) -> Path:
    files = {
        "task.toml": f'[task]\nname = "{name}"\n[environment]\nnetwork_mode = "no-network"\n',
        "instruction.md": f"Solve {name}.\n",
        "tests/test.sh": f"#!/bin/bash\necho {reward} > /logs/verifier/reward.txt\n",
        "solution/solve.sh": "#!/bin/bash\ntrue\n",
        **(extra or {}),
    }
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


class FakeHub:
    """Packages of a fake Harbor Hub, served by :data:`FAKE_CLI` through ``PLATOON_HARBOR_CLI``."""

    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.root = root
        root.mkdir(parents=True)
        self.state = {"packages": {}, "files": {}}
        self.file = root / "hub.json"
        self.calls_file = root / "calls.jsonl"
        script = root / "harbor.py"
        script.write_text(FAKE_CLI)
        monkeypatch.setenv("FAKE_HUB", str(self.file))
        monkeypatch.setenv("FAKE_HUB_CALLS", str(self.calls_file))
        monkeypatch.setenv(sources.HARBOR_CLI_ENV, shlex.join([sys.executable, str(script)]))
        monkeypatch.setattr(sources, "_RETRY_DELAY_S", 0.0)

    def task(self, name: str, **kwargs) -> tuple[Path, str]:
        directory = _task(self.root / "tasks" / name, name, **kwargs)
        digest = "sha256:" + content_hash(directory)
        self.state["files"][digest] = str(directory)
        return directory, digest

    def dataset(self, package: str, revision: int, tasks: dict[str, str], tags: list[str], digest: str) -> None:
        versions = self.state["packages"].setdefault(package, {"type": "dataset", "versions": []})["versions"]
        for version in versions:  # a tag names one version
            version["tags"] = [tag for tag in version["tags"] if tag not in tags]
        org = package.split("/")[0]
        listed = [{"org": org, "name": name, "digest": task_digest} for name, task_digest in tasks.items()]
        versions.append({"revision": revision, "digest": digest, "tags": tags, "tasks": listed})
        self.file.write_text(json.dumps(self.state))

    def single(self, package: str, digest: str) -> None:
        self.state["packages"][package] = {
            "type": "task",
            "versions": [{"revision": 1, "digest": digest, "tags": ["latest"], "tasks": []}],
        }
        self.file.write_text(json.dumps(self.state))

    def calls(self) -> list[list[str]]:
        if not self.calls_file.is_file():
            return []
        return [json.loads(line) for line in self.calls_file.read_text().splitlines()]


@pytest.fixture
def hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeHub:
    return FakeHub(tmp_path / "hub", monkeypatch)


def _published(hub: FakeHub) -> dict[str, str]:
    """``org/bench`` at revision 1 (tags v1) and revision 2 (tags latest): two tasks, then three."""

    alpha, beta, gamma = (hub.task(name)[1] for name in ("alpha", "beta", "gamma"))
    hub.dataset("org/bench", 1, {"alpha": alpha, "beta": beta}, ["v1"], "sha256:" + "1" * 64)
    hub.dataset("org/bench", 2, {"alpha": alpha, "beta": beta, "gamma": gamma}, ["latest"], "sha256:" + "2" * 64)
    return {"alpha": alpha, "beta": beta, "gamma": gamma}


def test_a_hub_dataset_at_a_revision_locks_with_the_hubs_digests(hub: FakeHub, tmp_path: Path):
    digests = _published(hub)
    allowlist = tmp_path / "keep.json"
    allowlist.write_text(json.dumps(["alpha", "gamma"]))
    config = fetch_config("harbor://org/bench@2", cache=tmp_path / "cache", tasks=[allowlist])
    assert is_pinned(config.dataset) and config.pinned and config.immutable
    lock = fetch(config)
    assert [task.name for task in lock.tasks] == ["alpha", "gamma"]  # the whole dataset downloaded, two selected
    assert {task.name: task.digest for task in lock.tasks} == {"alpha": digests["alpha"], "gamma": digests["gamma"]}
    assert lock.resolved == "sha256:" + "2" * 64 and lock.fetched_with == "harbor==0.24.0 (cli)"
    assert lock.source == {
        "kind": "harbor-hub",
        "package": "org/bench",
        "type": "dataset",
        "ref": "2",
        "digest": "sha256:" + "2" * 64,
        "revision": 2,
        "tags": ["latest"],
    }
    assert read_lock(config.lock_path()) == lock and list(verify_lock(lock, config.cache)) == []
    downloads = [call for call in hub.calls() if call[0] == "download"]
    assert downloads == [["download", "org/bench@" + "sha256:" + "2" * 64, "--export", "-o", downloads[0][-1]]]
    assert not (config.cache / ".tmp").exists() or not any((config.cache / ".tmp").iterdir())  # export removed
    # Pinned and current: nothing runs again, not even the CLI.
    before = len(hub.calls())
    assert fetch_if_needed(config) == (lock, False) and len(hub.calls()) == before


def test_a_single_task_package_is_a_dataset_of_one_task(hub: FakeHub, tmp_path: Path):
    _, digest = hub.task("hello-mcp")
    hub.single("harbor/hello-mcp", digest)
    lock = fetch(fetch_config("harbor://harbor/hello-mcp@1", cache=tmp_path / "cache"))
    assert [(task.name, task.digest) for task in lock.tasks] == [("hello-mcp", digest)]
    assert lock.resolved == digest and lock.source["type"] == "task"


def test_hub_tags_move_and_resolve_only_with_an_update(hub: FakeHub, tmp_path: Path):
    _published(hub)
    config = fetch_config("harbor://org/bench@latest", cache=tmp_path / "cache")
    assert not config.pinned and not is_pinned("harbor://org/bench") and is_pinned(f"harbor://org/bench@{DIGEST}")
    with pytest.raises(FetchError, match="moving reference"):
        fetch(config)
    with pytest.raises(LockError, match="moving reference"):
        ensure_lock(config, fetch_allowed=True)
    assert hub.calls() == []  # refused before the CLI ran
    lock = fetch(config, update=True)
    assert lock.resolved == "sha256:" + "2" * 64 and len(lock.tasks) == 3
    # The download is by the digest the tag resolved to, so a tag that moves meanwhile cannot change the tasks.
    assert ["download", "org/bench@sha256:" + "2" * 64] == [call for call in hub.calls() if call[0] == "download"][0][
        :2
    ]
    assert ensure_lock(config, fetch_allowed=False) == lock  # the lock pins it now
    # The tag moves: the lock still answers, until the next update resolves it again.
    _, delta = hub.task("delta")
    hub.dataset("org/bench", 3, {"delta": delta}, ["latest"], "sha256:" + "3" * 64)
    assert ensure_lock(config, fetch_allowed=False) == lock
    assert [task.name for task in fetch(config, update=True).tasks] == ["delta"]


def test_a_task_whose_files_do_not_hash_to_the_hubs_digest_fails_the_fetch(hub: FakeHub, tmp_path: Path):
    directory, digest = hub.task("alpha")
    hub.dataset("org/bench", 1, {"alpha": digest}, ["latest"], DIGEST)
    (directory / "tests" / "test.sh").write_text("#!/bin/bash\necho 0 > /logs/verifier/reward.txt\n")  # after publish
    with pytest.raises(FetchError, match=r"Harbor task 'alpha' .* hash to sha256:\w+, and the source gives it"):
        fetch(fetch_config("harbor://org/bench@1", cache=tmp_path / "cache"))


def test_a_transient_cli_failure_is_retried(hub: FakeHub, tmp_path: Path):
    _published(hub)
    Path(f"{hub.calls_file}.flaky").touch()
    lock = fetch(fetch_config("harbor://org/bench@1", cache=tmp_path / "cache"))
    assert [task.name for task in lock.tasks] == ["alpha", "beta"]
    assert sum(call[0] == "download" for call in hub.calls()) == 2


def test_the_harbor_cli_is_the_pinned_release_through_uvx(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(sources.HARBOR_CLI_ENV, raising=False)
    monkeypatch.setattr(sources.shutil, "which", lambda name: f"/opt/uv/{name}" if name in ("uvx", "uv") else None)
    pinned = [
        "--python",
        sys.executable,
        "--exclude-newer",  # its dependencies as they were at its release, in every new uv cache
        sources.HARBOR_CLI_EXCLUDE_NEWER,
        "--from",
        f"harbor=={sources.HARBOR_CLI_VERSION}",
        "harbor",
    ]
    assert sources.harbor_cli() == ["/opt/uv/uvx", *pinned]
    monkeypatch.setattr(sources.shutil, "which", lambda name: "/opt/uv/uv" if name == "uv" else None)
    assert sources.harbor_cli() == ["/opt/uv/uv", "tool", "run", *pinned]
    monkeypatch.setattr(sources.shutil, "which", lambda name: None)
    with pytest.raises(FetchError, match="install uv"):
        sources.harbor_cli()
    monkeypatch.setenv(sources.HARBOR_CLI_ENV, "harbor --quiet")
    assert sources.harbor_cli() == ["harbor", "--quiet"]


def test_cli_failures_name_the_command_and_its_output(hub: FakeHub, tmp_path: Path):
    _published(hub)
    with pytest.raises(FetchError, match=r"harbor version show org/nope@1 .*was not found"):
        fetch(fetch_config("harbor://org/nope@1", cache=tmp_path / "cache"))
    assert sum(call[:2] == ["version", "show"] for call in hub.calls()) == 1  # a permanent error: no retries
    with pytest.raises(FetchError, match="harbor://org/name@ref"):
        fetch_config("harbor://org/bench/extra@1")


def test_rollout_workers_never_run_the_harbor_cli(hub: FakeHub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _published(hub)
    use_site_cache(monkeypatch, tmp_path / "site")  # the Harbor cache these entries fetch into, never the home's
    settings = {"dataset": "harbor://org/bench@1", "lock": tmp_path / "bench.lock.json"}
    worker = LockedTasks(**settings)
    with pytest.raises(LockError, match="does not exist"):
        worker.dataset(fetch_allowed=False)
    assert hub.calls() == []
    trainer = LockedTasks(**settings)  # the trainer, which lists the tasks, fetches a pinned reference
    assert trainer.names("train") == ["alpha", "beta"]
    calls = len(hub.calls())
    worker = LockedTasks(**settings)
    assert worker.task_dir("beta").is_dir() and worker.index("alpha") == {}
    assert len(hub.calls()) == calls


# --- git repositories ------------------------------------------------------------------------------------------


def _git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repository(
    root: Path,
    files: dict[str, str],
    links: dict[str, str] | None = None,
    *,
    directories: dict[str, Path] | None = None,
) -> tuple[str, str]:
    """A git repository at ``root`` holding ``files``, the symbolic ``links`` (path: target) and copies of
    ``directories`` (path: directory, links kept), served as GitHub serves one (partial clones, any commit); its
    ``file://`` URL and the commit."""

    root.mkdir(parents=True)
    for relative, directory in (directories or {}).items():
        shutil.copytree(directory, root / relative, symlinks=True)
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "uploadpack.allowFilter", "true")
    _git(root, "config", "uploadpack.allowAnySHA1InWant", "true")
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    for relative, target in (links or {}).items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(target)
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "tasks")
    return root.as_uri(), _git(root, "rev-parse", "HEAD")


def _task_files(name: str, prefix: str) -> dict[str, str]:
    """The files :func:`_task` writes, below ``prefix``."""

    return {
        f"{prefix}{name}/task.toml": f'[task]\nname = "{name}"\n[environment]\nnetwork_mode = "no-network"\n',
        f"{prefix}{name}/instruction.md": f"Solve {name}.\n",
        f"{prefix}{name}/tests/test.sh": "#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n",
        f"{prefix}{name}/solution/solve.sh": "#!/bin/bash\ntrue\n",
    }


@pytest.fixture
def repository(tmp_path: Path) -> tuple[str, str]:
    files = {
        "README.md": "datasets\n",
        **_task_files("alpha", "datasets/bench/"),
        **_task_files("beta", "datasets/bench/"),
        **_task_files("gamma", "datasets/bench/hard/"),
        "datasets/bench/alpha/environment/data/task.toml": "not another task\n",
        **_task_files("other", "datasets/elsewhere/"),
    }
    return _repository(tmp_path / "upstream", files)


def test_a_git_directory_locks_only_the_selected_tasks_checked_out_sparsely(
    repository: tuple[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    url, commit = repository
    checked_out: list[list[str]] = []
    real = sources._Checkout.check_out

    def check_out(self, *, directories=(), patterns=()):
        checked_out.append(sorted(directories))
        return real(self, directories=directories, patterns=patterns)

    monkeypatch.setattr(sources._Checkout, "check_out", check_out)
    select = {"name": ["alpha", "hard/*"]}
    config = fetch_config(f"git+{url}@{commit}#datasets/bench", cache=tmp_path / "cache", select=select)
    assert config.pinned and config.immutable and not config.local
    lock = fetch(config)
    # Ids as a directory's scan gives them: a task directly below the dataset by its name, a deeper one by its path;
    # a task.toml inside a task's own files is not another task.  Only the selected ones are checked out.
    assert [task.name for task in lock.tasks] == ["alpha", "hard/gamma"]
    assert lock.selection["of"] == 3
    assert checked_out == [["datasets/bench/alpha", "datasets/bench/hard/gamma"]]
    assert lock.resolved == commit and lock.fetched_with.startswith("git ")
    assert lock.source == {"kind": "git", "url": url, "commit": commit, "path": "datasets/bench"}
    entry = config.cache / lock.get("alpha").path
    assert os.access(entry / "tests" / "test.sh", os.X_OK)
    assert lock.get("alpha").digest == "sha256:" + content_hash(tmp_path / "upstream" / "datasets" / "bench" / "alpha")
    assert list(verify_lock(lock, config.cache)) == []


def test_a_git_branch_moves_and_an_abbreviated_commit_cannot_be_fetched(repository: tuple[str, str], tmp_path: Path):
    url, commit = repository
    moving = fetch_config(f"git+{url}@main#datasets/elsewhere", cache=tmp_path / "cache")
    assert not moving.pinned
    with pytest.raises(FetchError, match="moving reference"):
        fetch(moving)
    lock = fetch(moving, update=True)
    assert lock.resolved == commit and lock.source["ref"] == "main" and [task.name for task in lock.tasks] == ["other"]
    with pytest.raises(FetchError, match="give all 40 characters"):
        fetch(fetch_config(f"git+{url}@{commit[:7]}#datasets/bench", cache=tmp_path / "cache"), update=True)
    with pytest.raises(FetchError, match="relative path inside the repository"):
        fetch_config(f"git+{url}@{commit}#../outside")
    with pytest.raises(FetchError, match="expected git"):
        fetch_config(f"git+{tmp_path}@{commit}")


def test_a_git_task_kept_in_git_lfs_needs_git_lfs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    files = _task_files("alpha", "")
    files["alpha/environment/model.bin"] = "version https://git-lfs.github.com/spec/v1\noid sha256:00\nsize 9\n"
    url, commit = _repository(tmp_path / "upstream", files)
    real_which = sources.shutil.which
    monkeypatch.setattr(sources.shutil, "which", lambda name: None if name == "git-lfs" else real_which(name))
    with pytest.raises(FetchError, match=r"kept in Git LFS, e\.g\. \['alpha/environment/model\.bin'\]"):
        fetch(fetch_config(f"git+{url}@{commit}", cache=tmp_path / "cache"))


# --- the legacy registry ---------------------------------------------------------------------------------------


def test_a_registry_dataset_takes_each_task_from_its_repository(repository: tuple[str, str], tmp_path: Path):
    url, commit = repository
    local = _task(tmp_path / "registry" / "local-task", "local-task")
    entry = lambda name, path, ref=commit: {"name": name, "git_url": url, "git_commit_id": ref, "path": path}  # noqa: E731
    registry = [
        {
            "name": "bench",
            "version": "1.0",
            "description": "",
            "tasks": [
                entry("a", "datasets/bench/alpha"),
                entry("g", "datasets/bench/hard/gamma"),
                entry("g", "datasets/bench/hard/gamma"),  # repeated rows are taken once, as Harbor takes them
                {"name": "here", "path": "local-task"},
            ],
        },
        {"name": "bench", "version": "2.0", "description": "", "tasks": [entry("b", "datasets/bench/beta", "HEAD")]},
    ]
    (tmp_path / "registry" / "registry.json").write_text(json.dumps(registry))
    config = fetch_config("harbor://bench@1.0", registry=tmp_path / "registry", cache=tmp_path / "cache")
    assert config.pinned and not is_pinned("harbor://bench@1.0")  # pinned by its registry file
    lock = fetch(config)
    assert [task.name for task in lock.tasks] == ["a", "g", "here"]
    assert lock.get("a").source_path == f"{url}@{commit}#datasets/bench/alpha"
    assert lock.get("here").digest == "sha256:" + content_hash(local)
    assert lock.source["kind"] == "harbor-registry" and lock.source["repositories"] == [{"url": url, "commit": commit}]
    # Editing the registry file is a new input: the lock no longer answers.
    assert ensure_lock(config, fetch_allowed=False) == lock
    (tmp_path / "registry" / "registry.json").write_text(json.dumps(registry[:1]))
    edited = fetch_config("harbor://bench@1.0", registry=tmp_path / "registry", cache=tmp_path / "cache")
    with pytest.raises(LockError, match="does not exist"):
        ensure_lock(edited, fetch_allowed=False)
    # A task at HEAD moves: an update resolves it, and the lock records the commit.
    (tmp_path / "registry" / "registry.json").write_text(json.dumps(registry))
    head = fetch_config("harbor://bench@2.0", registry=tmp_path / "registry", cache=tmp_path / "cache")
    with pytest.raises(FetchError, match="task 'b' is at HEAD"):
        fetch(head)
    resolved = fetch(head, update=True)
    assert resolved.get("b").source_path == f"{url}@{commit}#datasets/bench/beta"
    # No version: Harbor's newest, which moves.
    newest = fetch_config("harbor://bench", registry=tmp_path / "registry", cache=tmp_path / "cache")
    assert not newest.pinned and [task.name for task in fetch(newest, update=True).tasks] == ["b"]


def test_which_registries_pin_their_datasets():
    commit = "a" * 40
    pinned = references.legacy_registry_pinned
    assert pinned("harbor://kumo@1.0", "registry.json")
    assert pinned("harbor://kumo@1.0", f"https://example.com/org/repo/{commit}/registry.json")
    assert not pinned("harbor://kumo@1.0", sources.DEFAULT_REGISTRY)
    assert not pinned("harbor://kumo@1.0", None)
    assert not pinned("harbor://kumo", "registry.json")
    with pytest.raises(FetchError, match="registry names the legacy registry"):
        fetch_config("harbor://org/bench@1", registry="registry.json")


# --- adapters --------------------------------------------------------------------------------------------------

TOY_ADAPTER = """
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--count", type=int, default=1)
parser.add_argument("--source", type=Path, required=True)
args = parser.parse_args()
words = (args.source / "data" / "words.txt").read_text().split()
for number in range(args.count):
    task = args.output_dir / f"toy-{number}"
    (task / "tests").mkdir(parents=True)
    (task / "task.toml").write_text('[environment]\\nnetwork_mode = "no-network"\\n')
    (task / "instruction.md").write_text(f"Say {words[number]}.\\n")
    (task / "tests" / "test.sh").write_text("#!/bin/bash\\necho 1 > /logs/verifier/reward.txt\\n")
"""


@pytest.fixture
def adapter_repository(tmp_path: Path) -> tuple[str, str]:
    return _repository(
        tmp_path / "adapter-upstream",
        {"adapters/toy/generate.py": TOY_ADAPTER, "data/words.txt": "hello world again\n", "big/blob.txt": "x\n"},
    )


def test_an_adapter_generates_the_tasks_from_its_pinned_checkout(adapter_repository: tuple[str, str], tmp_path: Path):
    url, commit = adapter_repository
    adapter = {
        "repo": url,
        "commit": commit,
        "path": "adapters/toy",
        "command": [sys.executable, "generate.py"],
        "args": ["--count", "2", "--source", "{checkout}"],
        "sparse": ["/adapters/", "/data/"],
    }
    config = fetch_config(adapter=adapter, cache=tmp_path / "cache")
    assert config.dataset == f"adapter:{url}@{commit}#adapters/toy" and config.pinned and not config.local
    lock = fetch(config)
    assert [task.name for task in lock.tasks] == ["toy-0", "toy-1"]
    assert (config.cache / lock.get("toy-1").path / "instruction.md").read_text() == "Say world.\n"
    assert lock.resolved == commit
    assert lock.source == {
        "kind": "adapter",
        "repo": url,
        "commit": commit,
        "path": "adapters/toy",
        "command": [sys.executable, "generate.py"],
        "args": ["--count", "2", "--source", "{checkout}"],
        "env": {},
        "sparse": ["/adapters/", "/data/"],
        "uv_lock": None,
    }
    # Pinned and current: the adapter does not run again.
    assert fetch_if_needed(config) == (lock, False)
    # Its arguments are inputs of the lock.
    other = fetch_config(adapter={**adapter, "args": ["--count", "1", "--source", "{checkout}"]}, cache=config.cache)
    assert other.inputs_digest() != config.inputs_digest()
    assert [task.name for task in fetch(other).tasks] == ["toy-0"]


def test_an_adapter_that_fails_names_its_output_and_a_moving_one_needs_an_update(
    adapter_repository: tuple[str, str], tmp_path: Path
):
    url, commit = adapter_repository
    failing = {"repo": url, "commit": commit, "path": "adapters/toy", "command": [sys.executable, "generate.py"]}
    with pytest.raises(FetchError, match=r"(?s)the adapter failed \(exit 2\).*--source"):  # usage wraps on 3.13
        fetch(fetch_config(adapter=failing, cache=tmp_path / "cache"))
    assert not any((tmp_path / "cache" / ".tmp").iterdir())  # its checkout removed
    moving = fetch_config(adapter={**failing, "commit": "main"}, cache=tmp_path / "cache")
    with pytest.raises(FetchError, match="moving reference"):
        fetch(moving)
    with pytest.raises(FetchError, match="unknown keys"):
        fetch_config(adapter={**failing, "script": "x"})
    with pytest.raises(FetchError, match="not both"):
        fetch_config("harbor://org/bench@1", adapter=failing)


def test_an_adapters_default_command_is_its_projects_script(tmp_path: Path):
    project = tmp_path / "adapter"
    project.mkdir()
    (project / "pyproject.toml").write_text('[project]\nname = "x"\n[project.scripts]\ntoy = "toy.main:main"\n')
    assert sources._project_command(project, "adapter") == ["uv", "run", "--project", ".", "toy"]
    (project / "uv.lock").write_text("")
    assert sources._project_command(project, "adapter") == ["uv", "run", "--project", ".", "--locked", "toy"]
    (project / "pyproject.toml").write_text('[project]\nname = "x"\n')
    with pytest.raises(FetchError, match="name the adapter's command"):
        sources._project_command(project, "adapter")


# --- what a downloaded task may bring in -----------------------------------------------------------------------


def _host_files(tmp_path: Path) -> tuple[Path, Path]:
    """A file and a directory of the fetching host, which no task may bring into the cache."""

    secret = tmp_path / "host" / "netrc"
    secret.parent.mkdir(parents=True)
    secret.write_text("machine example.com password host-secret\n")
    keys = tmp_path / "host" / "ssh"
    keys.mkdir()
    (keys / "id_ed25519").write_text("host-secret key\n")
    _task(tmp_path / "host" / "tasks" / "beta", "beta")  # a task there, for a task path that goes through a link
    return secret, keys


def _host_links(secret: Path, keys: Path) -> dict[str, dict[str, str]]:
    """Links in task ``alpha`` that leave it for the host, by kind: the first three would put a file of the host into
    the cache, and the hash would leave out the directory the last one names, which a host may have or not."""

    return {
        "absolute": {"alpha/environment/netrc": str(secret)},
        "climbing": {"alpha/tests/netrc": "../" * 40 + str(secret).lstrip("/")},
        "directory": {"alpha/solution": str(keys)},
        "below": {"alpha/environment/keys": str(keys)},
    }


def _nothing_of_the_host(cache: Path) -> bool:
    return not cache.exists() or not any(
        "host-secret" in path.read_text(errors="replace") for path in cache.rglob("*") if path.is_file()
    )


@pytest.mark.parametrize("kind", ["absolute", "climbing", "directory", "below"])
def test_a_link_that_leaves_its_task_fails_a_git_or_registry_fetch(kind: str, tmp_path: Path):
    secret, keys = _host_files(tmp_path)
    files = {path: text for path, text in _task_files("alpha", "").items() if not path.startswith("alpha/solution/")}
    url, commit = _repository(tmp_path / "upstream", files, _host_links(secret, keys)[kind])
    name = next(iter(_host_links(secret, keys)[kind])).removeprefix("alpha/")  # the link itself, not a file below it
    with pytest.raises(FetchError, match=rf"Harbor task 'alpha': {name} is a symbolic link, or lies behind one, that"):
        fetch(fetch_config(f"git+{url}@{commit}", cache=tmp_path / "git-cache"))
    registry = tmp_path / "registry.json"
    rows = [{"name": "alpha", "git_url": url, "git_commit_id": commit, "path": "alpha"}]
    registry.write_text(json.dumps([{"name": "linked", "version": "1.0", "tasks": rows}]))
    with pytest.raises(FetchError, match=rf"{name} is a symbolic link, or lies behind one, that leaves the task"):
        fetch(fetch_config("harbor://linked@1.0", registry=registry, cache=tmp_path / "registry-cache"))
    for cache in (tmp_path / "git-cache", tmp_path / "registry-cache"):
        assert _nothing_of_the_host(cache) and not (cache / "locks").exists()


# Files of a repository beside its tasks, which a checkout of the selected tasks leaves out.
_SHARED = {
    "shared/privileged.yaml": "services:\n  main:\n    privileged: true\n",
    "shared/internal.yaml": "services:\n  main:\n    networks: [inside]\nnetworks:\n  inside:\n    internal: true\n",
    "shared/data.txt": "shared data\n",
    "shared/solution/solve.sh": "#!/bin/bash\ntrue\n",
}


@pytest.mark.parametrize(
    ("links", "name", "whole"),
    [
        (
            {"environment/docker-compose.yaml": "../../shared/privileged.yaml"},
            "environment/docker-compose.yaml",
            "compose: main.privileged: gives the service every privilege of the host",
        ),
        (
            {
                "environment/docker-compose.yaml": "../../shared/internal.yaml",
                "environment/data.txt": "../../shared/data.txt",
            },
            "environment/data.txt",
            "task_config: alpha: environment/docker-compose.yaml gives main networks (internal only)",
        ),
        ({"environment/shared": "../../shared"}, "environment/shared", None),
        ({"solution": "../shared/solution"}, "solution", None),
    ],
    ids=["compose-file", "compose-file-and-data", "data-directory", "solution-directory"],
)
def test_a_link_out_of_its_task_fails_the_fetch_where_the_checkout_lacks_what_it_names(
    links: dict[str, str], name: str, whole: str | None, hub: FakeHub, tmp_path: Path
):
    """A git or registry fetch checks out only the selected tasks, so there a link out of a task names nothing, which
    Harbor's hash skips: cached without it, the task would lose its compose file, its data or its solution unnoticed
    (and run as a task without a compose file, where a fetch of the whole repository refuses the file, ``whole``), so
    the fetch fails on it, as on such a link in a Hub export.  A link inside the task that names nothing is skipped, as
    in the hash."""

    files = {path: text for path, text in _task_files("alpha", "").items() if not path.startswith("alpha/solution/")}
    linked = {f"alpha/{link}": target for link, target in links.items()}
    upstream = tmp_path / "upstream"
    url, commit = _repository(
        upstream,
        {**files, **_SHARED, "alpha/environment/stale/keep.txt": "kept\n"},
        {**linked, "alpha/environment/stale/gone.txt": "nothing"},  # in the task, naming nothing
    )
    leaves = rf"Harbor task 'alpha': {re.escape(name)} is a symbolic link, or lies behind one, that leaves the task"
    with pytest.raises(FetchError, match=leaves):
        fetch(fetch_config(f"git+{url}@{commit}", cache=tmp_path / "git-cache"))
    registry = tmp_path / "registry.json"
    rows = [{"name": "alpha", "git_url": url, "git_commit_id": commit, "path": "alpha"}]
    registry.write_text(json.dumps([{"name": "linked", "version": "1.0", "tasks": rows}]))
    with pytest.raises(FetchError, match=leaves):
        fetch(fetch_config("harbor://linked@1.0", registry=registry, cache=tmp_path / "registry-cache"))
    published = tmp_path / "published" / "alpha"  # the task alone, its digest taken without what its links name
    shutil.copytree(upstream / "alpha", published, symlinks=True)
    digest = "sha256:" + content_hash(published)
    hub.state["files"][digest] = str(published)
    hub.dataset("org/linked", 1, {"alpha": digest}, ["latest"], DIGEST)
    with pytest.raises(FetchError, match=leaves):
        fetch(fetch_config("harbor://org/linked@1", cache=tmp_path / "hub-cache"))
    for cache in ("git-cache", "registry-cache", "hub-cache"):
        assert not (tmp_path / cache / "locks").exists()
    # The whole repository as a directory, whose links a fetch follows: its compose files are refused, the rest runs.
    refused = fetch(fetch_config(str(upstream), cache=tmp_path / "whole-cache")).get("alpha").refused
    assert (refused or "").startswith(whole) if whole else refused is None, refused
    for link in links:
        (upstream / "alpha" / link).unlink()
    _git(upstream, "add", "-A")
    _git(upstream, "commit", "-q", "-m", "no link out of the task")
    lock = fetch(fetch_config(f"git+{url}@{_git(upstream, 'rev-parse', 'HEAD')}", cache=tmp_path / "cache"))
    entry = tmp_path / "cache" / lock.get("alpha").path
    assert (entry / "environment" / "stale" / "keep.txt").read_text() == "kept\n"
    assert not os.path.lexists(entry / "environment" / "stale" / "gone.txt")
    assert lock.get("alpha").digest == "sha256:" + content_hash(upstream / "alpha")


def test_a_task_behind_a_link_fails_the_fetch(tmp_path: Path):
    _host_files(tmp_path)
    url, commit = _repository(
        tmp_path / "upstream", _task_files("alpha", "tasks/"), {"linked": str(tmp_path / "host" / "tasks")}
    )
    registry = tmp_path / "registry.json"
    rows = [
        {"name": name, "git_url": url, "git_commit_id": commit, "path": path}
        for name, path in (("alpha", "tasks/alpha"), ("beta", "linked/beta"))
    ]
    registry.write_text(json.dumps([{"name": "linked", "version": "1.0", "tasks": rows}]))
    config = fetch_config("harbor://linked@1.0", registry=registry, cache=tmp_path / "cache")
    with pytest.raises(FetchError, match=r"Harbor task 'beta' is a symbolic link, or lies behind one \(linked/beta\)"):
        fetch(config)
    assert _nothing_of_the_host(config.cache)


@pytest.mark.parametrize(
    ("target", "message"),
    [
        (
            "../../shared/alpha.toml",
            r"Harbor task 'alpha': task\.toml is a symbolic link, or lies behind one, that leaves the task",
        ),
        ("gone.toml", r"Harbor task 'alpha' has no task\.toml"),
    ],
    ids=["out-of-the-task", "inside-the-task"],
)
def test_a_task_toml_that_links_to_nothing_fails_a_hub_export_as_it_fails_a_checkout(
    target: str, message: str, hub: FakeHub, tmp_path: Path
):
    """A task.toml that is a symbolic link naming nothing (a file of the repository outside the task, which a
    checkout of the task leaves out, or no file at all) fails a fetch of a Hub export as it fails a git or registry
    fetch, naming the link, rather than as a task that the Harbor CLI's export lacks."""

    files = {path: text for path, text in _task_files("alpha", "tasks/").items() if path != "tasks/alpha/task.toml"}
    upstream = tmp_path / "upstream"
    url, commit = _repository(
        upstream, {**files, "shared/alpha.toml": '[task]\nname = "alpha"\n'}, {"tasks/alpha/task.toml": target}
    )
    with pytest.raises(FetchError, match=message):
        fetch(fetch_config(f"git+{url}@{commit}#tasks", cache=tmp_path / "git-cache"))
    registry = tmp_path / "registry.json"
    rows = [{"name": "alpha", "git_url": url, "git_commit_id": commit, "path": "tasks/alpha"}]
    registry.write_text(json.dumps([{"name": "linked", "version": "1.0", "tasks": rows}]))
    with pytest.raises(FetchError, match=message):
        fetch(fetch_config("harbor://linked@1.0", registry=registry, cache=tmp_path / "registry-cache"))
    digest = _publish(hub, upstream / "tasks" / "alpha", "alpha")  # its link kept, naming nothing in the export
    hub.dataset("org/linked", 1, {"alpha": digest}, ["latest"], DIGEST)
    with pytest.raises(FetchError, match=message):
        fetch(fetch_config("harbor://org/linked@1", cache=tmp_path / "hub-cache"))
    for cache in ("git-cache", "registry-cache", "hub-cache"):
        assert not (tmp_path / cache / "locks").exists()


def test_a_link_inside_its_task_is_cached_as_its_file(hub: FakeHub, tmp_path: Path):
    files = _task_files("alpha", "")
    url, commit = _repository(tmp_path / "upstream", files, {"alpha/tests/solve.sh": "../solution/solve.sh"})
    lock = fetch(fetch_config(f"git+{url}@{commit}", cache=tmp_path / "cache"))
    cached = tmp_path / "cache" / lock.get("alpha").path / "tests" / "solve.sh"
    assert not cached.is_symlink() and cached.read_text() == files["alpha/solution/solve.sh"]
    # Harbor's content hash takes a link as the file it names.
    assert lock.get("alpha").digest == "sha256:" + content_hash(tmp_path / "upstream" / "alpha")
    assert list(verify_lock(lock, tmp_path / "cache")) == []
    # A Hub export can hold links too (Harbor's own extraction keeps those that stay in the export, another CLI may
    # keep any), and one that leaves its task fails the fetch as well.
    secret, _ = _host_files(tmp_path)
    directory, _ = hub.task("gamma")
    (directory / "tests" / "netrc").symlink_to(secret)
    digest = "sha256:" + content_hash(directory)
    hub.state["files"][digest] = str(directory)
    hub.dataset("org/linked", 1, {"gamma": digest}, ["latest"], DIGEST)
    with pytest.raises(FetchError, match=r"Harbor task 'gamma': tests/netrc is a symbolic link, or lies behind one"):
        fetch(fetch_config("harbor://org/linked@1", cache=tmp_path / "hub-cache"))
    assert _nothing_of_the_host(tmp_path / "hub-cache")


def test_an_adapters_links_are_cached_as_their_files(tmp_path: Path):
    adapter = (
        "import pathlib, sys\n"
        "out = pathlib.Path(sys.argv[sys.argv.index('--output-dir') + 1]) / 'toy-0'\n"
        "(out / 'tests').mkdir(parents=True)\n"
        "(out / 'task.toml').write_text('[environment]\\nnetwork_mode = \"no-network\"\\n')\n"
        "(out / 'instruction.md').symlink_to(pathlib.Path.cwd() / 'instruction.md')  # into its checkout\n"
        "(out / 'tests' / 'test.sh').write_text('#!/bin/bash\\necho 1 > /logs/verifier/reward.txt\\n')\n"
    )
    url, commit = _repository(tmp_path / "upstream", {"gen.py": adapter, "instruction.md": "Say hi.\n"})
    config = fetch_config(
        adapter={"repo": url, "commit": commit, "command": [sys.executable, "gen.py"]}, cache=tmp_path / "cache"
    )
    lock = fetch(config)
    cached = config.cache / lock.get("toy-0").path / "instruction.md"
    assert not cached.is_symlink() and cached.read_text() == "Say hi.\n"  # its checkout is gone
    assert list(verify_lock(lock, config.cache)) == []


# --- what moves, and what is refused ---------------------------------------------------------------------------


def test_a_registry_files_own_tasks_are_fetched_again_as_a_directory_is(repository: tuple[str, str], tmp_path: Path):
    url, commit = repository
    local = _task(tmp_path / "registry" / "local-task", "local-task")
    registry = tmp_path / "registry" / "registry.json"
    remote = [{"name": "a", "git_url": url, "git_commit_id": commit, "path": "datasets/bench/alpha"}]
    registry.write_text(
        json.dumps(
            [
                {"name": "bench", "version": "1.0", "tasks": [{"name": "here", "path": "local-task"}]},
                {"name": "remote", "version": "1.0", "tasks": remote},
            ]
        )
    )
    config = fetch_config("harbor://bench@1.0", registry=registry, cache=tmp_path / "cache")
    assert config.pinned and config.local and not config.immutable
    first, fetched = fetch_if_needed(config)
    (local / "instruction.md").write_text("Changed after the fetch.\n")
    second, again = fetch_if_needed(config)
    assert fetched and again and first.get("here").digest != second.get("here").digest
    # A dataset whose tasks are all at commits cannot change: a current lock answers it, and nothing runs.
    pinned = fetch_config("harbor://remote@1.0", registry=registry, cache=tmp_path / "cache")
    assert pinned.pinned and not pinned.local and pinned.immutable
    lock, _ = fetch_if_needed(pinned)
    assert fetch_if_needed(pinned) == (lock, False)


def test_of_two_registry_entries_with_one_version_the_last_counts_as_in_harbor(tmp_path: Path):
    _task(tmp_path / "registry" / "first", "first")
    _task(tmp_path / "registry" / "second", "second")
    entries = [
        {"name": "bench", "version": "1.0", "tasks": [{"name": name, "path": name}]} for name in ("first", "second")
    ]
    (tmp_path / "registry" / "registry.json").write_text(json.dumps(entries))
    lock = fetch(fetch_config("harbor://bench@1.0", registry=tmp_path / "registry", cache=tmp_path / "cache"))
    assert [task.name for task in lock.tasks] == ["second"]


def test_hub_task_names_must_be_package_names(hub: FakeHub, tmp_path: Path):
    _, digest = hub.task("escape")
    hub.dataset("org/bench", 1, {"../escape": digest}, ["latest"], DIGEST)
    with pytest.raises(FetchError, match=r"listed a task named '\.\./escape', which is not a package name"):
        fetch(fetch_config("harbor://org/bench@1", cache=tmp_path / "cache"))
    with pytest.raises(FetchError, match="expected harbor://org/name@ref"):
        fetch_config("harbor://org/a..b@1")


def test_an_adapter_that_names_its_output_in_its_command_gets_no_output_dir(tmp_path: Path):
    adapter = (
        "import argparse, pathlib\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--out', type=pathlib.Path, required=True)\n"
        "task = parser.parse_args().out / 'toy-0'\n"
        "(task / 'tests').mkdir(parents=True)\n"
        "(task / 'task.toml').write_text('[environment]\\nnetwork_mode = \"no-network\"\\n')\n"
        "(task / 'instruction.md').write_text('Say hi.\\n')\n"
        "(task / 'tests' / 'test.sh').write_text('#!/bin/bash\\necho 1 > /logs/verifier/reward.txt\\n')\n"
    )
    url, commit = _repository(tmp_path / "upstream", {"gen.py": adapter})
    command = [sys.executable, "gen.py", "--out", "{output}"]
    lock = fetch(fetch_config(adapter={"repo": url, "commit": commit, "command": command}, cache=tmp_path / "cache"))
    assert [task.name for task in lock.tasks] == ["toy-0"]


def _with_credentials(url: str, userinfo: str) -> str:
    """``url`` with ``userinfo`` (``user:password``, or a token for a user name) in front of its host."""

    scheme, _, rest = url.partition("://")
    return f"{scheme}://{userinfo}@{rest}"


@pytest.mark.parametrize("userinfo", ["user:sekret-token", "ghp_sekret"])  # a password, a token as the user name
def test_a_url_with_credentials_is_refused_without_showing_them(userinfo: str, tmp_path: Path):
    dataset = _with_credentials("git+https://example.com/org/repo", userinfo) + "@" + "a" * 40
    with pytest.raises(FetchError, match=r"https://\*\*\*@example\.com/org/repo carries credentials") as refused:
        fetch_config(dataset)
    assert "sekret" not in str(refused.value)
    repo = _with_credentials("https://example.com/org/repo.git", userinfo)
    with pytest.raises(FetchError, match="carries credentials") as refused:
        fetch_config(adapter={"repo": repo, "commit": "a" * 40})
    assert "sekret" not in str(refused.value)
    with pytest.raises(FetchError, match="carries credentials"):
        fetch_config("harbor://kumo@1.0", registry=_with_credentials("https://example.com/registry.json", userinfo))
    registry = tmp_path / "registry.json"
    url = _with_credentials("https://example.com/org/repo", userinfo)
    row = {"name": "a", "git_url": url, "git_commit_id": "a" * 40, "path": "a"}
    registry.write_text(json.dumps([{"name": "bench", "version": "1.0", "tasks": [row]}]))
    with pytest.raises(
        FetchError, match=r"task 'a': https://\*\*\*@example\.com/org/repo carries credentials"
    ) as refused:
        fetch(fetch_config("harbor://bench@1.0", registry=registry, cache=tmp_path / "cache"))
    assert "sekret" not in str(refused.value)
    assert fetch_config(f"git+ssh://git@example.com/org/repo@{'a' * 40}").dataset.startswith("git+ssh://git@")  # a user


# --- retries ---------------------------------------------------------------------------------------------------


def test_a_transient_failure_is_retried_whatever_the_dataset_is_named(hub: FakeHub, tmp_path: Path):
    _, alpha = hub.task("alpha")
    hub.dataset("org/invalid-tasks", 1, {"alpha": alpha}, ["latest"], DIGEST)
    Path(f"{hub.calls_file}.flaky").touch()
    lock = fetch(fetch_config("harbor://org/invalid-tasks@1", cache=tmp_path / "cache"))
    assert [task.name for task in lock.tasks] == ["alpha"]
    assert sum(call[0] == "download" for call in hub.calls()) == 2


@pytest.mark.parametrize(
    ("output", "error_lines", "permanent"),
    [
        ("Error: Package not found: org/nope", True, True),
        ("Error: Task 'org/x@1' was not found or you do not have access.", True, True),
        ("Downloading task not-found-handler\nhttpx.RemoteProtocolError: <StreamReset stream_id:1>", True, False),
        ("  | httpx.ReadTimeout: The read operation timed out", True, False),
        ("Error: [Errno -3] Temporary failure in name resolution", True, False),
        ("httpx.HTTPStatusError: Client error '404 Not Found' for url 'https://hub.example/x'", True, True),
        ("remote: Repository not found.\nfatal: repository 'https://example.com/org/repo/' not found", False, True),
        ("fatal: couldn't find remote ref refs/heads/nope", False, True),
        ("fatal: Authentication failed for 'https://example.com/org/private/'", False, True),
        ("git@example.com: Permission denied (publickey).\nfatal: Could not read from remote repository.", False, True),
        (
            "fatal: unable to access 'https://example.com/org/unknown-repo/': Could not resolve host: x.invalid",
            False,
            False,
        ),
        (
            "error: RPC failed; curl 92 HTTP/2 stream 5 was not closed cleanly\nfatal: early EOF\n"
            "fatal: fetch-pack: invalid index-pack output",
            False,
            False,
        ),
        (
            "fatal: unable to access 'https://example.com/org/repo-not-found/': The requested URL returned error: 502",
            False,
            False,
        ),
    ],
)
def test_which_failures_are_permanent(output: str, error_lines: bool, permanent: bool):
    assert sources._permanent(output, error_lines=error_lines) is permanent


def test_a_registry_url_is_retried_on_a_server_error_and_not_on_a_missing_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, repository: tuple[str, str]
):
    import io
    import urllib.error

    url, commit = repository
    rows = [{"name": "a", "git_url": url, "git_commit_id": commit, "path": "datasets/bench/alpha"}]
    body = json.dumps([{"name": "bench", "version": "1.0", "tasks": rows}]).encode()
    answers: list[int] = [503, 200]
    calls: list[str] = []

    def urlopen(address: str, timeout: float):
        calls.append(address)
        code = answers.pop(0)
        if code != 200:
            raise urllib.error.HTTPError(address, code, "error", {}, None)
        return io.BytesIO(body)

    monkeypatch.setattr(sources, "_RETRY_DELAY_S", 0.0)
    monkeypatch.setattr(sources.urllib.request, "urlopen", urlopen)
    registry = f"https://example.com/org/repo/{commit}/registry.json"
    lock = fetch(fetch_config("harbor://bench@1.0", registry=registry, cache=tmp_path / "cache"))
    assert [task.name for task in lock.tasks] == ["a"] and len(calls) == 2
    answers[:] = [404]
    with pytest.raises(FetchError, match="HTTP 404"):
        fetch(fetch_config("harbor://bench@1.0", registry=registry, cache=tmp_path / "other"))
    assert len(calls) == 3


# --- compose files: a task from any source is planned as a local one is ---------------------------------------

COMPOSE_FIXTURES = Path(__file__).parent / "fixtures" / "harbor-compose"
STACK_FIXTURES = Path(__file__).parent / "fixtures" / "harbor-stacks" / "tasks"
# Why a fetch refuses a stack by default (refuse.sidecars), and why it did before stacks ran (a lock of then says so).
STACK_REFUSED = "stack: the compose file adds sidecar services (refuse: {sidecars: false} fetches it as a stack)"
STACK_REFUSED_THEN = "stack: the compose file adds sidecar services, which platoon cannot run yet"


def _publish(hub: FakeHub, task_dir: Path, name: str) -> str:
    """Publish a copy of the task directory ``task_dir`` on the fake Hub as ``name``; its digest."""

    directory = hub.root / "tasks" / name
    shutil.copytree(task_dir, directory, symlinks=True)
    digest = "sha256:" + content_hash(directory)
    hub.state["files"][digest] = str(directory)
    return digest


@pytest.mark.asyncio
async def test_a_hub_task_with_a_compose_file_runs_from_the_stack_plan_its_lock_records(
    hub: FakeHub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A Hub task's compose file is planned at fetch as a local task's is, and the lock records each plan beside how
    the source resolved: a rollout worker runs the main-only task from its lock's plan (its init serves through
    grading, so the oracle scores) and never runs the CLI, and Harbor's hello-mcp, whose MCP server runs beside main,
    is refused with its plan, as a fetch refuses a stack by default."""

    serve = _publish(hub, COMPOSE_FIXTURES / "tasks" / "main-only-serve", "main-only-serve")
    mcp = _publish(hub, COMPOSE_FIXTURES / "upstream" / "examples" / "hello-mcp", "hello-mcp")
    hub.dataset("org/stacks", 1, {"main-only-serve": serve, "hello-mcp": mcp}, ["latest"], DIGEST)
    use_site_cache(monkeypatch, tmp_path / "site")
    provider = ProcessProvider(root=tmp_path / "sandboxes")

    trainer = HarborEnvironmentService(dataset="harbor://org/stacks@1", provider=provider)
    assert await trainer.list_tasks("train") == ["main-only-serve"]  # the trainer fetches the pinned revision

    config = fetch_config("harbor://org/stacks@1")
    lock = read_lock(config.lock_path())
    assert (lock.source["kind"], lock.source["digest"], lock.compose_rule) == ("harbor-hub", DIGEST, COMPOSE_RULE)
    assert {task.name: task.digest for task in lock.tasks} == {"hello-mcp": mcp, "main-only-serve": serve}
    main_only, sidecars = lock.get("main-only-serve"), lock.get("hello-mcp")
    assert (main_only.flags["stack"], main_only.refused, main_only.compose["kind"]) == ("main_only", None, "main_only")
    assert "environment/data/serve.py" in main_only.input_files  # what main binds is an input of the task
    assert (sidecars.flags["stack"], sidecars.refused) == ("sidecars", STACK_REFUSED)
    assert set(sidecars.compose["services"]) == {"main", "mcp-server"}
    written = json.loads(config.lock_path().read_text())
    assert (written["source"], written["compose_rule"]) == (dict(lock.source), COMPOSE_RULE)

    calls = len(hub.calls())
    worker = HarborEnvironmentService(dataset="harbor://org/stacks@1", provider=provider)  # a rollout worker's
    session = await worker.open(await worker.get_task("main-only-serve"), rollout_id="rollout-1")
    try:
        assert session.sandbox.spec.env["GREETING"] == "hello from compose"  # from the lock's plan
        assert (await session.run_solution()).exit_code == 0
        outcome = await session.grade()
        assert (outcome.status, outcome.reward, outcome.metadata["stack"]) == ("scored", 1.0, "main_only")
    finally:
        await session.close()
    assert len(hub.calls()) == calls and list(provider.root.iterdir()) == []


def test_a_git_tasks_compose_file_is_refused_or_planned_as_an_overlay_edits_it(tmp_path: Path):
    compose = "services:\n  main:\n    environment: [MODE=git]\n    cap_add: [NET_ADMIN]\n"
    files = {
        **_task_files("alpha", "tasks/"),
        "tasks/alpha/environment/Dockerfile": "FROM python:3.12-slim\nWORKDIR /app\n",
        "tasks/alpha/environment/docker-compose.yaml": compose,
    }
    url, commit = _repository(tmp_path / "upstream", files)
    overlay = tmp_path / "overlays" / "fix"
    overlay.mkdir(parents=True)
    (overlay / "overlay.toml").write_text('[[compose]]\nservice = "main"\nunset = ["cap_add"]\n')
    dataset = f"git+{url}@{commit}#tasks"

    plain = fetch(fetch_config(dataset, cache=tmp_path / "cache", lock=tmp_path / "plain.json"))
    fixed = fetch(fetch_config(dataset, cache=tmp_path / "cache", lock=tmp_path / "fixed.json", overlays=[overlay]))

    assert (plain.get("alpha").refused or "").startswith("compose: main.cap_add: adds NET_ADMIN")
    task = fixed.get("alpha")
    assert (task.refused, task.flags["stack"], task.overlays) == (None, "main_only", ("fix",))
    assert task.compose["edits"] == ["fix: main: unset cap_add"] and "cap_add" not in task.compose["services"]["main"]
    for lock, path in ((plain, tmp_path / "plain.json"), (fixed, tmp_path / "fixed.json")):
        assert lock.source == {"kind": "git", "url": url, "commit": commit, "path": "tasks"}
        assert lock.compose_rule == COMPOSE_RULE and read_lock(path) == lock


def test_a_lock_platoon_0_4_0_wrote_reads_and_answers_without_a_source_or_plans(tmp_path: Path):
    """A lock from before the sources and the stack plans holds no ``source``, no ``compose_rule`` and no task's
    ``compose``: it reads as it did and still answers the config it was written for, and written again it gains only
    the rule."""

    config = fetch_config(f"hf://org/harbor-dataset@{'f' * 40}", cache=tmp_path / "cache")
    task = {
        "name": "serve",
        "digest": "sha256:" + "0" * 64,
        "path": "tasks/" + "0" * 64,
        "index": {},
        "flags": {"stack": "sidecars", "verifier": "shared"},
        "image": {"kind": "prebuilt", "ref": "python:3.12-slim"},
        "refused": "stack: the compose file adds sidecar services, which platoon cannot run yet",
        "input_files": {},
    }
    old = {  # the keys of a lock platoon 0.4.0 wrote, in its order
        "version": 1,
        "dataset": config.dataset,
        "resolved": "f" * 40,
        "fetched_with": "huggingface_hub==1.33.0",
        "hash_rule": HASH_RULE,
        "build_key_rule": "platoon-buildkey-v1",
        "overlay": None,
        "inputs": config.inputs_digest(),
        "selection": {"tasks": {}, "select": {}, "selected": 1, "of": 1},
        "modes": "executable bit restored on *.sh files and files starting with #!",
        "created_at": "2026-10-25T00:00:00+00:00",
        "tasks": [task],
    }
    config.lock_path().parent.mkdir(parents=True)
    config.lock_path().write_text(json.dumps(old))

    lock = ensure_lock(config, fetch_allowed=False)  # as a rollout worker reads it

    assert (lock.source, lock.compose_rule, lock.get("serve").compose) == ({}, "", None)
    assert lock.get("serve").refused == task["refused"]  # refused as 0.4.0 refused it, until it is fetched again
    assert lock.to_json() == {**old, "compose_rule": ""}


def _stacks(root: Path) -> Path:
    """A dataset directory at ``root`` with the stack fixture ``app-kv`` (main stores a value through a key-value
    sidecar) and a copy of it, ``commandless``, whose sidecar runs its registry image's own command, which platoon does
    not know yet."""

    shutil.copytree(STACK_FIXTURES / "app-kv", root / "app-kv")
    shutil.copytree(STACK_FIXTURES / "app-kv", root / "commandless")
    compose = root / "commandless" / "environment" / "docker-compose.yaml"
    compose.write_text(compose.read_text().replace('    command: ["python3", "/srv/kv/kv.py"]\n', ""))
    return root


@pytest.mark.parametrize("source", ["git", "hub"])
@pytest.mark.asyncio
async def test_a_fetched_stack_locks_as_a_local_one_and_runs_only_where_stacks_run(
    source: str, hub: FakeHub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A task whose compose file adds sidecars, from git or the Hub, is planned and locked as the same task fetched
    from a directory: refused by default (``stack``), and with ``refuse: {sidecars: false}`` a runnable stack, its plan
    the same either way.  A run lists it where the provider runs stacks and stops when it lists the tasks with
    ``stacks: refuse``, on a provider that runs none, and on a stack that no rollout could start; a rollout worker's
    session runs it from the lock, the no-op scoring 0 and the oracle 1, and never fetches."""

    monkeypatch.setenv("PLATOON_NODE_SCRATCH", str(tmp_path / "scratch"))
    tasks = _stacks(tmp_path / "tasks")
    if source == "git":
        url, commit = _repository(tmp_path / "upstream", {}, directories={"datasets/kv": tasks})
        dataset = f"git+{url}@{commit}#datasets/kv"
    else:
        digests = {name: _publish(hub, tasks / name, name) for name in ("app-kv", "commandless")}
        hub.dataset("org/kv", 1, digests, ["latest"], DIGEST)
        dataset = "harbor://org/kv@1"
    cache = use_site_cache(monkeypatch, tmp_path / "site")
    for refuse, refused in (({}, STACK_REFUSED), ({"sidecars": False}, None)):
        local = fetch(fetch_config(str(tasks), cache=cache, lock=tmp_path / "local.json", refuse=refuse))
        lock = fetch(fetch_config(dataset, refuse=refuse))
        assert (lock.source["kind"], lock.compose_rule) == ({"git": "git", "hub": "harbor-hub"}[source], COMPOSE_RULE)
        for name in ("app-kv", "commandless"):
            fetched, planned = lock.get(name), local.get(name)
            kinds = (fetched.flags["stack"], fetched.compose["kind"])
            assert (fetched.refused, kinds) == (refused, ("sidecars", "sidecars"))
            for field in ("digest", "flags", "compose", "image", "input_files"):
                assert getattr(fetched, field) == getattr(planned, field), field

    provider = ProcessProvider(root=tmp_path / "sandboxes")
    settings: dict[str, Any] = {"dataset": dataset, "refuse": {"sidecars": False}}
    runs = HarborEnvironmentService(provider=provider, **settings)
    assert sorted(await runs.list_tasks("train")) == ["app-kv", "commandless"]
    await runs.check_tasks("train", ["app-kv"])  # the process provider runs stacks
    unstartable = r"stacks that no rollout can start \(commandless\): .*the command of its image"
    with pytest.raises(EnvSetupError, match=unstartable) as error:
        await runs.check_tasks("train", ["app-kv", "commandless"])
    assert error.value.failure_kind == "env_config"
    refusing = {
        "the environment sets stacks: refuse": HarborEnvironmentService(provider=provider, stacks="refuse", **settings),
        "the 'docker' sandbox provider runs no": HarborEnvironmentService(provider="docker", **settings),
    }
    for why, service in refusing.items():
        with pytest.raises(EnvSetupError, match=rf"1 of the 1 tasks of split 'train' are multi-container .*{why}"):
            await service.check_tasks("train", ["app-kv"])

    calls = len(hub.calls())
    worker = HarborEnvironmentService(provider=provider, **settings)  # a rollout worker's: it reads the lock
    session = await worker.open(await worker.get_task("app-kv"), rollout_id="rollout-1")
    try:
        assert session.stack is not None and set(session.stack.services) == {"main", "kv"}
        noop = await session.grade()
        assert (noop.status, noop.reward, noop.metadata["stack"]) == ("scored", 0.0, "sidecars"), noop.metadata
        assert (await session.run_solution()).exit_code == 0
        oracle = await session.grade()
        assert (oracle.status, oracle.reward) == ("scored", 1.0), oracle.metadata.get("test_output")
    finally:
        await session.close()
    assert len(hub.calls()) == calls and list(provider.root.iterdir()) == []


@pytest.mark.parametrize("source", ["git", "registry", "hub"])
def test_a_link_out_of_a_stack_fails_the_fetch_before_its_compose_file_is_planned(
    source: str, hub: FakeHub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A downloaded task's links are checked before its compose file is planned: a stack whose sidecar's program
    links out of the task, to a file the checkout leaves out, fails the fetch, naming the link, whether the fetch would
    refuse the stack or lock it, and nothing of it is planned, cached or locked."""

    tasks = tmp_path / "tasks"
    shutil.copytree(STACK_FIXTURES / "app-kv", tasks / "app-kv")
    program = tasks / "app-kv" / "environment" / "kv" / "kv.py"
    shared = tmp_path / "shared"
    shared.mkdir()
    shutil.move(program, shared / "kv.py")
    program.symlink_to("../../../../shared/kv.py")  # tasks/app-kv/environment/kv/kv.py -> shared/kv.py
    url, commit = _repository(tmp_path / "upstream", {}, directories={"tasks": tasks, "shared": shared})
    registry = tmp_path / "registry.json"
    rows = [{"name": "app-kv", "git_url": url, "git_commit_id": commit, "path": "tasks/app-kv"}]
    registry.write_text(json.dumps([{"name": "kv", "version": "1.0", "tasks": rows}]))
    hub.dataset("org/kv", 1, {"app-kv": _publish(hub, tasks / "app-kv", "app-kv")}, ["latest"], DIGEST)
    dataset, extra = {
        "git": (f"git+{url}@{commit}#tasks", {}),
        "registry": ("harbor://kv@1.0", {"registry": registry}),
        "hub": ("harbor://org/kv@1", {}),
    }[source]

    def planned(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the compose file was planned")

    monkeypatch.setattr("platoon.envs.harbor.fetch._plan_compose", planned)
    leaves = r"Harbor task 'app-kv': environment/kv/kv\.py is a symbolic link, or lies behind one, that leaves the task"
    for refuse in ({}, {"sidecars": False}):
        cache = tmp_path / f"cache-{len(refuse)}"
        with pytest.raises(FetchError, match=leaves):
            fetch(fetch_config(dataset, cache=cache, refuse=refuse, **extra))
        assert not (cache / "locks").exists() and not any((cache / "tasks").glob("*"))


@pytest.mark.parametrize(
    ("refuse", "refused", "refetched"),
    [
        ({}, STACK_REFUSED_THEN, STACK_REFUSED),
        (
            {"sidecars": False},
            "task_config: app-kv: environment/docker-compose.yaml makes the environment a multi-container stack "
            "(sidecar services), which platoon cannot run yet (RFC 0010); the task is refused, never started without "
            "them",
            None,
        ),
    ],
    ids=["refused-stack", "sidecars-false"],
)
@pytest.mark.asyncio
async def test_a_lock_of_fetched_stacks_written_before_stacks_ran_still_reads_and_runs_its_main_only_task(
    refuse: dict[str, Any], refused: str, refetched: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A lock that a fetch from git wrote before platoon ran stacks reads and answers its config as it did: it differs
    from today's only by its stack's refusal, which said then that platoon could not run it (``stack``, or with
    ``refuse: {sidecars: false}`` ``task_config``).  Its main-only task runs from its plan, the stack stays refused and
    unlisted, and a fetch that repairs the lock locks the stack as today's fetch does."""

    monkeypatch.setenv("PLATOON_NODE_SCRATCH", str(tmp_path / "scratch"))
    url, commit = _repository(
        tmp_path / "upstream",
        {},
        directories={
            "tasks/main-only-serve": COMPOSE_FIXTURES / "tasks" / "main-only-serve",
            "tasks/app-kv": STACK_FIXTURES / "app-kv",
        },
    )
    use_site_cache(monkeypatch, tmp_path / "site")
    config = fetch_config(f"git+{url}@{commit}#tasks", refuse=refuse)
    fetch(config)
    then = json.loads(config.lock_path().read_text())  # what the fetch of the same commit wrote then
    [stack] = [task for task in then["tasks"] if task["name"] == "app-kv"]
    stack["refused"] = refused
    config.lock_path().write_text(json.dumps(then, indent=1))
    assert list(then) == [
        *("version", "dataset", "resolved", "fetched_with", "source", "hash_rule", "build_key_rule", "compose_rule"),
        *("overlay", "inputs", "selection", "modes", "created_at", "tasks"),
    ]

    lock = ensure_lock(config, fetch_allowed=False)  # as a rollout worker reads it
    assert lock.get("app-kv").refused == refused and lock.to_json() == then
    assert fetch_if_needed(config) == (lock, False)  # it answers its config: nothing is fetched again
    provider = ProcessProvider(root=tmp_path / "sandboxes")
    settings: dict[str, Any] = {"dataset": config.dataset, "refuse": refuse, "provider": provider}
    trainer = HarborEnvironmentService(**settings)
    assert await trainer.list_tasks("train") == ["main-only-serve"]
    await trainer.check_tasks("train", ["main-only-serve"])
    worker = HarborEnvironmentService(**settings)
    session = await worker.open(await worker.get_task("main-only-serve"), rollout_id="rollout-1")
    try:
        assert (await session.run_solution()).exit_code == 0
        outcome = await session.grade()
        assert (outcome.status, outcome.reward, outcome.metadata["stack"]) == ("scored", 1.0, "main_only")
    finally:
        await session.close()

    repaired, fetched = fetch_if_needed(config, repair=True)
    assert fetched and repaired.get("app-kv").refused == refetched
    assert repaired.get("main-only-serve") == lock.get("main-only-serve")
