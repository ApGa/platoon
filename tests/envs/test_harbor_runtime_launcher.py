"""The launcher of runtime Harbor tasks (``platoon/envs/harbor/launcher.py``), run as the verifier runs it but without
root: a wrapper replaces what it does as root (the seal's chown, the replay's switch of user and its sweep of that
user's processes), and it runs in a process of its own, over directories that stand for the verifier's ``/tests``,
``/logs/verifier``, working directory, ``/tmp`` and what the agents published.  What it reads of the task, which reward
and report it writes for each way a grader, a replay or a submission behaves, and what it leaves running; how it seals,
limits and sweeps; and that it parses as the oldest Python it runs on, with nothing beyond the standard library.
Where unprivileged user namespaces work, it also runs with its own operations, as root in one."""

from __future__ import annotations

import ast
import contextlib
import errno
import hashlib
import json
import os
import resource
import shutil
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any

import pytest

from platoon.envs.harbor import launcher

LAUNCHER = Path(launcher.__file__)
# Runs the launcher with the root-only operations replaced (unless ``ops`` names the launcher's own): the seal only
# takes the other users' permissions, hardening only strips the write bits it can without being root, the working
# directory, already this user's, is lent without a change of owner, the replay runs as this user with the launcher's
# limits on files, its sweep kills what is below the launcher, never this user's other processes, what the replay's
# user left where every user may write is this user's, in the verifier's own temporary directories, and it removes no
# System V IPC objects, since this user's other processes may have some.
WRAPPER = """
import os, sys
sys.path.insert(0, {directory!r})
import launcher

class WithoutRoot(launcher.Ops):
    def euid(self):
        return {euid}

    def seal(self, path):
        os.chmod(path, 0o700)

    def give(self, path, uid):
        pass

    def harden(self, path):
        launcher._strip_foreign_write(path)

    def lend(self, workdir):
        launcher._lend_tree(workdir, give=False)

    def demote(self, max_file_bytes, max_address_space_bytes=None):
        return lambda: launcher._limit_files(max_file_bytes)

    def processes_of(self, uid):
        return []

    def ipc_of(self, uid):
        return []

    def replays_own(self, info):
        return info.st_uid == os.getuid()

paths = launcher.Paths(
    tests={tests!r},
    logs={logs!r},
    workdir={workdir!r},
    scratch={scratch!r},
    published={published!r},
    temporary={temporary!r},
)
{extra}
sys.exit(launcher.run(paths, {ops}))
"""
HONEST_GRADER = """
import json, os
with open(os.path.join(os.environ["PLATOON_WORKDIR"], "answer.json")) as stream:
    passed = json.load(stream) == {"answer": 42}
with open(os.environ["PLATOON_VERDICT"], "w") as stream:
    json.dump({"passed": passed, "checks": [{"id": "answer", "passed": passed, "margin": 0.5}]}, stream)
"""
SOLVE = """
import json
with open("data.txt") as stream:
    value = 2 * int(stream.read())
with open("answer.json", "w") as stream:
    json.dump({"answer": value}, stream)
"""


class Verifier:
    """The verifier's directories in ``root`` (its ``/tmp`` in ``scratch`` when given), what the agents published
    (``published``: ``/logs/agent`` and ``/logs/artifacts``), where every user may write (``temporary``: its ``/tmp``,
    ``/var/tmp`` and ``/dev/shm``), a task's launcher settings and grader, and a run of the launcher."""

    def __init__(self, root: Path, scratch: Path | None = None) -> None:
        self.root = root
        self.tests, self.logs, self.workdir, self.scratch = (root / name for name in ("tests", "logs", "app", "tmp"))
        if scratch is not None:
            self.scratch = scratch / "tmp"
        self.published = [root / "published" / "agent", root / "published" / "artifacts"]
        self.temporary = [self.scratch, root / "var-tmp", root / "shm"]
        directories = (self.tests / "platoon", self.tests / "gen", self.logs, self.workdir, self.scratch)
        for directory in (*directories, *self.published, *self.temporary[1:]):
            directory.mkdir(parents=True)
        (self.logs / launcher.STDOUT).write_text("")
        self.config: dict[str, Any] = {
            "schema": launcher.SCHEMA,
            "grader_owner": "generated",
            "reward_key": "reward",
            "timeout_s": 120.0,
            "max_output_bytes": 1 << 20,
            "max_copy_bytes": 1 << 30,
            "artifacts": ["answer.json"],
            "grader": {"command": [sys.executable, "grade.py"]},
            "replay": None,
        }
        self.grader(HONEST_GRADER)

    def grader(self, source: str) -> None:
        (self.tests / "gen" / "grade.py").write_text(textwrap.dedent(source))

    def submit(self, answer: Any = 42, **files: str) -> None:
        if answer is not None:
            (self.workdir / "answer.json").write_text(json.dumps({"answer": answer}))
        for name, text in files.items():
            (self.workdir / name).write_text(textwrap.dedent(text))

    def replay(self, outputs: tuple[str, ...] = ("answer.json",)) -> None:
        self.config["replay"] = {"command": [sys.executable, "solve.py"], "outputs": list(outputs)}

    def run(
        self,
        *,
        euid: int = 0,
        timeout_s: float = 120.0,
        settings: str | None = "config",
        workdir: Path | None = None,
        extra: str = "",
        env: dict[str, str] | None = None,
        ops: str = "WithoutRoot()",
        prefix: tuple[str, ...] = (),
    ) -> tuple[int, dict[str, Any], dict[str, Any]]:
        """Run the launcher; its exit status, the reward file and the report it wrote.  ``settings``: the text of its
        settings file instead of ``config``'s, or None for none; ``workdir``: the working directory it is given, if
        not ``self.workdir``; ``extra``: code the wrapper runs before the launcher (it may replace the launcher's
        functions); ``env``: variables the launcher's process gets besides this one's; ``ops``: the operations it
        runs with (``launcher.Ops()`` for its own); ``prefix``: the command it runs under (``unshare -r``)."""

        if settings is not None:
            text = json.dumps(self.config) if settings == "config" else settings
            (self.tests / "platoon" / launcher.CONFIG).write_text(text)
        wrapper = self.root / "wrapper.py"
        wrapper.write_text(
            WRAPPER.format(
                directory=str(LAUNCHER.parent),
                euid=euid,
                tests=str(self.tests),
                logs=str(self.logs),
                workdir=str(workdir or self.workdir),
                scratch=str(self.scratch),
                published=[str(path) for path in self.published],
                temporary=[str(path) for path in self.temporary],
                extra=textwrap.dedent(extra),
                ops=ops,
            )
        )
        with open(self.logs / launcher.STDOUT, "ab") as stdout:
            done = subprocess.run(
                [*prefix, sys.executable, str(wrapper)],
                cwd=self.workdir,
                stdout=stdout,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
                env={**os.environ, **(env or {})},
            )
        return done.returncode, self.read(launcher.REWARD), self.read(launcher.REPORT)

    def read(self, name: str) -> dict[str, Any]:
        path = self.logs / name
        return json.loads(path.read_text()) if path.is_file() else {}


@pytest.fixture
def verifier(tmp_path: Path) -> Verifier:
    return Verifier(tmp_path)


# ----------------------------------------------------------------------------------------------------------- verdicts


def test_an_honest_grader_scores_the_answer_and_its_report_says_how(verifier):
    verifier.submit(42)

    code, reward, report = verifier.run()

    assert (code, reward) == (0, {"reward": 1})
    assert report["schema"] == launcher.SCHEMA and report["grader_owner"] == "generated"
    assert (report["failure_kind"], report["reason"]) == (None, None)
    assert report["verdict"] == {"passed": True, "checks": [{"id": "answer", "passed": True, "margin": 0.5}]}
    assert report["grader"]["exit_code"] == 0 and report["contract"]["violation"] is None
    assert "platoon launcher" in (verifier.logs / launcher.STDOUT).read_text()
    assert sorted(path.name for path in verifier.logs.iterdir()) == [
        "gen",
        "launcher.json",
        "reward.json",
        "test-stdout.txt",
    ]
    assert (
        oct(verifier.tests.stat().st_mode & 0o777) == "0o700" and oct(verifier.logs.stat().st_mode & 0o777) == "0o700"
    )


def test_the_reward_takes_the_name_the_environment_trains(verifier):
    verifier.submit(41)
    verifier.config["reward_key"] = "correctness"

    assert verifier.run()[1] == {"correctness": 0}


WRITE_VERDICT = 'import json, os\nwith open(os.environ["PLATOON_VERDICT"], "w") as stream:\n    {}\n'
# JSON a level deeper than the launcher reads, which any Python would parse, and JSON that opens more arrays than some
# Pythons parse (a RecursionError) while others reach its end (a ValueError): each is refused the same way everywhere.
DEEP_JSON = {
    "parsable": "[" * (launcher.MAX_JSON_DEPTH + 1) + "]" * (launcher.MAX_JSON_DEPTH + 1),
    "beyond-some-parsers": "[" * 100_000,
}


@pytest.mark.parametrize(
    ("grader", "problem"),
    [
        (WRITE_VERDICT.format('json.dump({"passed": False, "invalid": True}, stream)'), "holds invalid, beyond passed"),
        (WRITE_VERDICT.format('json.dump({"passed": 1}, stream)'), "the verdict's passed is not true or false"),
        (
            WRITE_VERDICT.format(
                'json.dump({"passed": True, "checks": [{"id": "a", "passed": True, "margin": 1e999}]}, stream)'
            ),
            "check 0's margin is not a finite number",
        ),
        (
            WRITE_VERDICT.format(
                'json.dump({"passed": True, "checks": [{"id": "a", "passed": True, "x": 1}]}, stream)'
            ),
            "check 0 is not an object of id, passed and margin",
        ),
        (
            WRITE_VERDICT.format('json.dump({"passed": True, "checks": [{"passed": True}]}, stream)'),
            "check 0 needs an id",
        ),
        (WRITE_VERDICT.format("json.dump([True], stream)"), "the verdict is not an object"),
        (WRITE_VERDICT.format('stream.write("{passed: true")'), "the verdict is not JSON"),
        ("pass\n", "the grader wrote no verdict"),
    ],
    ids=[
        "invalid",
        "not-bool",
        "infinite-margin",
        "unknown-check-key",
        "no-check-id",
        "not-object",
        "not-json",
        "none",
    ],
)
def test_a_verdict_beyond_the_schema_is_the_graders_protocol_fault_never_a_censor(verifier, grader, problem):
    """A generated grader cannot declare a measurement invalid, which would censor what it does not like: any verdict
    outside {passed, checks} is its own fault."""

    verifier.grader(grader)
    verifier.submit(42)

    code, reward, report = verifier.run()

    assert (code, reward) == (0, {"invalid": 1, "failure_kind": "verifier_generated_protocol"})
    assert problem in report["detail"]


@pytest.mark.parametrize("verdict", DEEP_JSON.values(), ids=DEEP_JSON.keys())
def test_a_verdict_nested_too_deeply_to_read_is_the_graders_protocol_fault(verifier, verdict):
    verifier.grader(WRITE_VERDICT.format(f"stream.write({verdict!r})"))
    verifier.submit(42)

    _, reward, report = verifier.run()

    assert reward == {"invalid": 1, "failure_kind": "verifier_generated_protocol"}
    assert report["detail"] == "the verdict is nested too deeply to read"


def test_a_check_id_longer_than_the_schema_allows_is_the_graders_protocol_fault(verifier):
    long_id = "a" * (launcher.MAX_CHECK_ID_CHARS + 1)
    verifier.grader(
        WRITE_VERDICT.format(
            f'json.dump({{"passed": True, "checks": [{{"id": "{long_id}", "passed": True}}]}}, stream)'
        )
    )
    verifier.submit(42)

    _, reward, report = verifier.run()

    assert reward == {"invalid": 1, "failure_kind": "verifier_generated_protocol"}
    assert report["detail"] == f"check 0's id holds more than {launcher.MAX_CHECK_ID_CHARS} characters"


def test_a_verdict_as_large_as_the_schema_allows_keeps_its_score_and_a_report_grading_reads_whole(verifier):
    """The hundred checks a report keeps, with ids as long as the schema allows in characters JSON escapes to twelve
    bytes each, would make a report of some 300 KB: it leaves the checks out and stays within its bound, far below what
    grading reads of a file (where a larger one would be no report at all), and the verdict's score stands."""

    from platoon.envs.harbor.verifier import MAX_REWARD_BYTES

    wide = "\U0001f600" * (launcher.MAX_CHECK_ID_CHARS - 3)
    checks = [{"id": f"{n:03d}{wide}", "passed": True} for n in range(launcher.REPORTED_CHECKS)]
    verdict = json.dumps({"passed": True, "checks": checks})
    verifier.grader(WRITE_VERDICT.format(f"stream.write({verdict!r})"))
    verifier.submit(42)

    code, reward, report = verifier.run()

    assert (code, reward) == (0, {"reward": 1})
    size = (verifier.logs / launcher.REPORT).stat().st_size
    assert size <= launcher.MAX_REPORT_BYTES + 1 < MAX_REWARD_BYTES
    assert report["truncated"] == ["verdict.checks"] and report["verdict"] == {"passed": True}


def test_a_launcher_that_fails_once_the_grader_ran_is_the_graders_fault(verifier):
    """As root beside the launcher, the grader may break what the launcher reads after it: a failure of the launcher's
    then is the grader's (``unfinished``), never the trusted side's."""

    verifier.submit(42)

    code, reward, report = verifier.run(
        extra="""
        def broken(path):
            raise RuntimeError("what the launcher reads is gone")

        launcher.read_verdict = broken
        """
    )

    assert (code, reward) == (0, {"invalid": 1, "failure_kind": "verifier_generated_unfinished"})
    assert "what the launcher reads is gone" in report["detail"] and "RuntimeError" in report["trace"]


def test_a_grader_that_exits_nonzero_after_a_verdict_is_its_fault_whatever_it_wrote(verifier):
    verifier.grader(
        """
        import json, os, sys
        with open(os.environ["PLATOON_VERDICT"], "w") as stream:
            json.dump({"passed": False}, stream)
        sys.exit(3)
        """
    )
    verifier.submit(42)

    _, reward, report = verifier.run()

    assert reward == {"invalid": 1, "failure_kind": "verifier_generated_exit"}
    assert report["detail"] == "the grader exited 3" and report["grader"]["exit_code"] == 3


@pytest.mark.parametrize(("owner", "prefix"), [("generated", "verifier_generated"), ("trusted", "verifier_trusted")])
def test_the_graders_owner_names_its_failure_kinds(verifier, owner, prefix):
    verifier.config["grader_owner"] = owner
    verifier.grader("import sys\nsys.exit(1)\n")
    verifier.submit(42)

    assert verifier.run()[1] == {"invalid": 1, "failure_kind": f"{prefix}_exit"}


def test_a_grader_command_that_cannot_start_is_its_exit_fault(verifier):
    verifier.config["grader"]["command"] = ["./no-such-grader"]
    verifier.submit(42)

    _, reward, report = verifier.run()

    assert reward == {"invalid": 1, "failure_kind": "verifier_generated_exit"}
    assert report["detail"].startswith("the grader's command cannot start")


def test_a_grader_that_outlives_its_time_is_killed_and_its_timeout_fault(verifier):
    verifier.grader("import time\ntime.sleep(60)\n")
    verifier.config["timeout_s"] = 7.0  # 2 s for the launcher's steps, then the margin
    verifier.submit(42)

    started = time.monotonic()
    _, reward, report = verifier.run()

    assert reward == {"invalid": 1, "failure_kind": "verifier_generated_timeout"}
    assert report["grader"]["timed_out"] and time.monotonic() - started < 30


def test_the_launcher_writes_the_only_reward_file(verifier):
    """What a grader writes in /logs/verifier besides its verdict goes: its reward files, its logs."""

    verifier.grader(
        """
        import json, os
        logs = os.path.dirname(os.path.dirname(os.environ["PLATOON_VERDICT"]))
        for name, text in (("reward.json", '{"reward": 1}'), ("reward.txt", "1"), ("stdout.txt", "x")):
            with open(os.path.join(logs, name), "w") as stream:
                stream.write(text)
        os.mkdir(os.path.join(logs, "more"))
        with open(os.environ["PLATOON_VERDICT"], "w") as stream:
            json.dump({"passed": False}, stream)
        """
    )
    verifier.submit(42)

    _, reward, _ = verifier.run()

    assert reward == {"reward": 0}
    assert sorted(path.name for path in verifier.logs.iterdir()) == [
        "gen",
        "launcher.json",
        "reward.json",
        "test-stdout.txt",
    ]
    assert (verifier.logs / "gen" / launcher.GRADER_LOG).is_file()


def test_what_a_grader_leaves_running_is_killed_before_the_reward_is_written(verifier):
    """A daemon that detached keeps writing a reward of 1: the launcher is its subreaper, so it kills it first."""

    marker = verifier.root / "writer.pid"
    verifier.grader(
        f"""
        import json, os, time
        logs = os.path.dirname(os.path.dirname(os.environ["PLATOON_VERDICT"]))
        if os.fork() == 0:
            os.setsid()
            if os.fork() == 0:
                with open({str(marker)!r}, "w") as stream:
                    stream.write(str(os.getpid()))
                while True:
                    with open(os.path.join(logs, "reward.json"), "w") as stream:
                        json.dump({{"reward": 1}}, stream)
                    time.sleep(0.01)
            os._exit(0)
        os.wait()
        while not os.path.exists({str(marker)!r}):
            time.sleep(0.01)
        with open(os.environ["PLATOON_VERDICT"], "w") as stream:
            json.dump({{"passed": False}}, stream)
        """
    )
    verifier.submit(42)

    try:
        _, reward, _ = verifier.run()
        time.sleep(0.2)  # long enough for a writer that survived to write again

        assert verifier.read(launcher.REWARD) == reward == {"reward": 0}
        with pytest.raises(ProcessLookupError):
            os.kill(int(marker.read_text()), 0)
    finally:  # a writer the launcher failed to kill must not outlive the test
        with contextlib.suppress(OSError, ValueError):
            os.kill(int(marker.read_text()), signal.SIGKILL)


def test_a_verdict_that_is_a_link_is_the_graders_protocol_fault(verifier):
    """A link in place of the verdict would have the launcher read whatever it points to."""

    verifier.grader(
        """
        import json, os
        elsewhere = os.path.join(os.path.dirname(os.environ["PLATOON_VERDICT"]), "elsewhere.json")
        with open(elsewhere, "w") as stream:
            json.dump({"passed": True}, stream)
        os.symlink(elsewhere, os.environ["PLATOON_VERDICT"])
        """
    )
    verifier.submit(42)

    _, reward, report = verifier.run()

    assert reward == {"invalid": 1, "failure_kind": "verifier_generated_protocol"}
    assert report["detail"] == "the verdict is a symbolic link"


def test_a_log_the_grader_replaced_with_a_fifo_never_blocks_the_launcher(verifier):
    """The grader, root in the verifier, may put a FIFO that nothing writes in place of its log: the launcher reads a
    log's tail only from a regular file, opened without waiting, and the verdict's score stands."""

    verifier.grader(
        HONEST_GRADER
        + textwrap.dedent(
            """
            log = os.path.join(os.path.dirname(os.environ["PLATOON_VERDICT"]), "grader.log")
            os.remove(log)
            os.mkfifo(log)
            """
        )
    )
    verifier.submit(42)

    code, reward, report = verifier.run(timeout_s=60)

    assert (code, reward) == (0, {"reward": 1})
    assert report["grader"]["log_tail"] == ""


# ---------------------------------------------------------------------------------------------- the output contract


@pytest.mark.parametrize(
    ("setup", "violation"),
    [
        (lambda v: None, "answer.json is missing"),
        (lambda v: (v.workdir / "answer.json").symlink_to(v.tests / "gen" / "grade.py"), "is a symbolic link"),
        (lambda v: (v.workdir / "answer.json").write_text("{not json"), "answer.json is not JSON"),
        (lambda v: (v.workdir / "answer.json").write_bytes(b" " * (2 << 20)), "more than an output may"),
        (lambda v: os.mkfifo(v.workdir / "answer.json"), "answer.json is not a regular file"),
    ],
    ids=["missing", "symlink", "not-json", "too-large", "fifo"],
)
def test_artifacts_that_break_the_output_contract_score_zero_as_the_solvers_fault(verifier, setup, violation):
    setup(verifier)

    _, reward, report = verifier.run()

    assert reward == {"reward": 0}
    assert report["reason"] == "solver_output_contract" and violation in report["detail"]
    assert "grader" not in report  # the grader never ran


@pytest.mark.parametrize("artifact", DEEP_JSON.values(), ids=DEEP_JSON.keys())
def test_an_artifact_nested_too_deeply_to_read_breaks_the_contract_never_the_launcher(verifier, artifact):
    """A solver's artifact that nests deeper than the launcher reads is its own contract violation, on every Python,
    whether or not its parser would take it: never a failure of the launcher's, which would be the trusted side's."""

    (verifier.workdir / "answer.json").write_text(artifact)

    code, reward, report = verifier.run()

    assert (code, reward) == (0, {"reward": 0})
    assert report["reason"] == "solver_output_contract" and "nested too deeply" in report["detail"]


@pytest.mark.parametrize(
    ("text", "depth"),
    [
        ("", 0),
        ("1", 0),
        ("[]", 1),
        ('{"a": [1, {"b": []}], "c": {}}', 4),
        ('["[[[", "{{"]', 1),  # brackets in strings are text
        (r'["\"[[", "\\", "\u005b"]', 1),  # escapes: a quote, a backslash, a bracket
        ("[" * 5, 5),  # what a text opens counts, closed or not
        ('[["]]', 2),  # and so does what a string that never ends leaves open
        ("]]][", 0),  # a parser stops at the first ]
    ],
)
def test_json_text_depth_counts_the_brackets_open_outside_strings(text, depth):
    assert launcher.json_text_depth(text) == depth


def test_load_json_refuses_json_nested_deeper_than_the_launcher_reads_before_it_parses():
    limit = launcher.MAX_JSON_DEPTH

    value = launcher.load_json("[" * limit + "]" * limit)
    for _ in range(limit - 1):
        (value,) = value
    assert value == []
    assert launcher.load_json('["' + "[" * 100_000 + '"]') == ["[" * 100_000]
    for text in (*DEEP_JSON.values(), '{"a": ' * (limit + 1)):
        with pytest.raises(launcher.NestedTooDeeply, match="nested too deeply") as refused:
            launcher.load_json(text)
        assert isinstance(refused.value, ValueError)  # what reads JSON refuses it as it refuses text that is no JSON


def test_a_directory_artifact_is_checked_file_by_file(verifier):
    verifier.config["artifacts"] = ["out"]
    (verifier.workdir / "out" / "deep").mkdir(parents=True)
    (verifier.workdir / "out" / "deep" / "a.json").write_text("[1")
    verifier.submit(42)

    report = verifier.run()[2]

    assert report["reason"] == "solver_output_contract" and "out/deep/a.json is not JSON" in report["detail"]


# ------------------------------------------------------------------------------------------------------- the replay


def test_a_replay_that_reproduces_the_outputs_is_what_the_grader_grades(verifier):
    """The submitted answer says 42, the code computes 2 × 21: the grader grades what the code wrote, and the
    submitted one waits in $PLATOON_SUBMITTED."""

    verifier.replay()
    verifier.config["artifacts"] = ["answer.json", "solve.py"]
    verifier.submit(42, **{"solve.py": SOLVE, "data.txt": "21"})
    verifier.grader(
        HONEST_GRADER
        + textwrap.dedent(
            """
            with open(os.path.join(os.environ["PLATOON_SUBMITTED"], "answer.json")) as stream:
                assert json.load(stream) == {"answer": 42}
            """
        )
    )

    _, reward, report = verifier.run()

    assert reward == {"reward": 1}, report
    assert report["replay"]["exit_code"] == 0 and report["replay"]["outputs"] == ["answer.json"]


def test_the_grader_grades_the_replayed_output_not_the_submitted_one(verifier):
    verifier.replay()
    verifier.submit(42, **{"solve.py": SOLVE, "data.txt": "20"})  # the code computes 40, whatever the answer says

    _, reward, report = verifier.run()

    assert reward == {"reward": 0} and report["reason"] is None
    assert json.loads((verifier.workdir / "answer.json").read_text()) == {"answer": 40}


@pytest.mark.parametrize(
    ("solve", "detail"),
    [
        ("import sys\nsys.exit(4)\n", "the replay exited 4"),
        ("pass\n", "after the replay, answer.json is missing"),
        ("import os\nos.symlink('/etc/passwd', 'answer.json')\n", "answer.json is a symbolic link"),
        ("open('answer.json', 'w').write('{nope')\n", "answer.json is not JSON"),
        ("import os\nos.mkdir('answer.json')\n", "answer.json is a directory"),
    ],
    ids=["exits", "no-output", "symlink", "not-json", "directory"],
)
def test_a_replay_that_does_not_reproduce_the_outputs_scores_zero_as_the_solvers_fault(verifier, solve, detail):
    verifier.replay()
    verifier.submit(42, **{"solve.py": solve})

    _, reward, report = verifier.run()

    assert reward == {"reward": 0}
    assert report["reason"] == "solver_nonreproducible" and detail in report["detail"], report
    assert "grader" not in report


def test_a_replay_reproduces_its_outputs_where_tmp_is_another_filesystem(tmp_path):
    """In a verifier's sandbox the working directory and /tmp are often filesystems apart: the submitted outputs move
    aside all the same."""

    shm = Path("/dev/shm")
    if not shm.is_dir() or os.stat(shm).st_dev == os.stat(tmp_path).st_dev:
        pytest.skip("no /dev/shm on another filesystem than the test's temporary directory")
    scratch = Path(tempfile.mkdtemp(dir=shm, prefix="platoon-launcher-test-"))
    try:
        verifier = Verifier(tmp_path, scratch=scratch)
        verifier.replay(outputs=("answer.json", "out/log.txt"))
        verifier.config["artifacts"] = ["answer.json", "out", "solve.py"]
        (verifier.workdir / "out").mkdir()
        (verifier.workdir / "out" / "log.txt").write_text("submitted")
        solve = SOLVE + "open('out/log.txt', 'w').write('replayed')\n"
        verifier.submit(42, **{"solve.py": solve, "data.txt": "21"})

        _, reward, report = verifier.run()

        assert reward == {"reward": 1}, report
        assert (verifier.workdir / "out" / "log.txt").read_text() == "replayed"
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def test_a_replay_puts_its_outputs_back_whatever_names_the_submission_holds(verifier):
    """The submission holds the name a launcher's temporary file once took beside an output: the replay's output still
    goes back, and the grader grades it."""

    verifier.replay()
    verifier.config["artifacts"] = ["answer.json", "solve.py", "answer.json.platoon-replay"]
    verifier.submit(42, **{"solve.py": SOLVE, "data.txt": "21", "answer.json.platoon-replay": "{}"})

    _, reward, report = verifier.run()

    assert reward == {"reward": 1}, report


def test_a_file_where_the_replay_writes_a_directory_breaks_the_contract(verifier):
    """The replay writes ``out/answer.json``, and the submission holds a file ``out``: its output could not go back,
    which the contract says before the replay runs, as the solver's fault."""

    verifier.replay(outputs=("out/answer.json",))
    verifier.config["artifacts"] = ["solve.py", "out"]
    (verifier.workdir / "out").write_text("not a directory")
    verifier.submit(
        None, **{"solve.py": "import os\nos.remove('out')\nos.mkdir('out')\nopen('out/answer.json', 'w').write('1')\n"}
    )

    _, reward, report = verifier.run()

    assert reward == {"reward": 0}
    assert report["reason"] == "solver_output_contract" and report["detail"] == (
        "out is a file, where the replay writes out/answer.json"
    )
    assert "replay" not in report and "grader" not in report


def test_a_working_directory_too_large_to_copy_scores_zero_as_the_solvers_fault(verifier):
    verifier.replay()
    verifier.config["artifacts"] = ["solve.py"]
    verifier.config["max_copy_bytes"] = 64
    verifier.submit(None, **{"solve.py": SOLVE, "data.txt": "21", "padding.txt": "x" * 100})

    _, reward, report = verifier.run()

    assert reward == {"reward": 0}
    assert report["reason"] == "solver_output_contract" and "more than 64 bytes for a replay" in report["detail"]


def test_a_copy_for_the_replay_past_its_time_scores_zero_as_the_solvers_fault(verifier):
    """With 5 s, the launcher's steps have no time left once its margin is kept: the copy of the working directory for
    the replay is past the replay's time before it starts."""

    verifier.replay()
    verifier.config.update(artifacts=["solve.py"], timeout_s=5.0)
    verifier.submit(None, **{"solve.py": SOLVE, "data.txt": "21"})

    _, reward, report = verifier.run()

    assert reward == {"reward": 0}
    assert report["reason"] == "solver_nonreproducible" and "outlived its time" in report["detail"]


def test_the_replay_starts_without_the_outputs_it_must_write_again(verifier):
    verifier.replay()
    verifier.submit(42, **{"solve.py": "import os\nassert not os.path.exists('answer.json')\nraise SystemExit(5)\n"})

    assert verifier.run()[2]["detail"] == "the replay exited 5"


def test_a_replay_that_outlives_its_time_scores_zero(verifier):
    verifier.replay()
    verifier.config["timeout_s"] = 9.0  # 4 s for the launcher's steps: 2 s for the replay
    verifier.submit(42, **{"solve.py": "import time\ntime.sleep(60)\n"})

    report = verifier.run()[2]

    assert report["reason"] == "solver_nonreproducible" and report["replay"]["timed_out"]
    assert report["replay"]["timeout_s"] <= 2.5


def test_the_replay_sees_none_of_the_verifiers_variables(verifier, monkeypatch):
    monkeypatch.setenv("PLATOON_TEST_SECRET", "s3cret")
    verifier.replay()
    probe = (
        "import os, sys\nsys.exit(6 if 'PLATOON_TEST_SECRET' in os.environ or 'PLATOON_VERDICT' in os.environ else 7)\n"
    )
    verifier.submit(42, **{"solve.py": probe})

    assert verifier.run()[2]["detail"] == "the replay exited 7"


def test_the_replay_keeps_the_variables_it_lists_and_the_templates_and_no_others(verifier, monkeypatch):
    """The replay's environment is a list of names, never a pattern: a thread count it names, and a variable of the
    template's that the settings' ``replay_env`` names, reach it; one that only looks like a thread count does not."""

    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    monkeypatch.setenv("SITE_DATA", "/data")
    monkeypatch.setenv("GRADER_THREADS", "s3cret")
    verifier.replay()
    verifier.config["replay_env"] = ["SITE_DATA"]
    probe = """
        import json, os
        names = ("OMP_NUM_THREADS", "SITE_DATA", "GRADER_THREADS", "HOME")
        with open("answer.json", "w") as stream:
            json.dump({name: os.environ.get(name) for name in names}, stream)
    """
    verifier.submit(42, **{"solve.py": probe})

    _, reward, report = verifier.run()

    assert reward == {"reward": 0} and report["reason"] is None  # the grader graded what the replay saw
    seen = json.loads((verifier.workdir / "answer.json").read_text())
    assert seen.pop("HOME").endswith("/home")  # a directory of the replay's own
    assert seen == {"OMP_NUM_THREADS": "3", "SITE_DATA": "/data", "GRADER_THREADS": None}


def test_a_replay_that_daemonizes_is_swept_before_its_outputs_are_read(verifier):
    """The replay's command exits and leaves a daemon, which left its session: the launcher, its subreaper, kills it
    before it reads the outputs, and says so."""

    marker = verifier.root / "daemon.pid"
    verifier.replay()
    daemonizing = SOLVE + textwrap.dedent(
        f"""
        import os, time
        if os.fork() == 0:
            os.setsid()
            if os.fork() == 0:
                with open({str(marker)!r}, "w") as stream:
                    stream.write(str(os.getpid()))
                time.sleep(120)
            os._exit(0)
        os.wait()
        while not os.path.exists({str(marker)!r}):
            time.sleep(0.01)
        """
    )
    verifier.submit(42, **{"solve.py": daemonizing, "data.txt": "21"})

    try:
        _, reward, report = verifier.run()

        assert reward == {"reward": 1}, report
        assert report["replay"]["swept"] == 1 and "survivors" not in report["replay"]
        with pytest.raises(ProcessLookupError):
            os.kill(int(marker.read_text()), 0)
    finally:  # a daemon the launcher failed to kill must not outlive the test
        with contextlib.suppress(OSError, ValueError):
            os.kill(int(marker.read_text()), signal.SIGKILL)


def test_a_replay_that_leaves_processes_that_cannot_be_killed_scores_zero_as_the_solvers_fault(verifier):
    verifier.replay()
    verifier.submit(42, **{"solve.py": SOLVE, "data.txt": "21"})

    _, reward, report = verifier.run(
        extra="launcher.kill_left = lambda also=None, window_s=0: (3, 1) if also is not None else (0, 0)\n"
    )

    assert reward == {"reward": 0}
    assert (report["reason"], report["detail"]) == (
        "solver_nonreproducible",
        "processes the replay left could not be killed (1)",
    )
    assert report["replay"]["survivors"] == 1 and "grader" not in report


def test_the_replay_runs_with_limits_on_its_files_processes_and_open_files():
    probe = textwrap.dedent(
        f"""
        import json, resource, sys
        sys.path.insert(0, {str(LAUNCHER.parent)!r})
        import launcher
        launcher._limit_replay(1 << 20)
        names = ("RLIMIT_CORE", "RLIMIT_FSIZE", "RLIMIT_NPROC", "RLIMIT_NOFILE")
        print(json.dumps({{name: resource.getrlimit(getattr(resource, name)) for name in names}}))
        """
    )

    done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True, timeout=60)

    def capped(name: str, value: int) -> list[int]:
        hard = resource.getrlimit(getattr(resource, name))[1]
        value = value if hard == resource.RLIM_INFINITY else min(value, hard)
        return [value, value]

    assert json.loads(done.stdout) == {
        "RLIMIT_CORE": [0, 0],
        "RLIMIT_FSIZE": capped("RLIMIT_FSIZE", (1 << 20) + 1),
        "RLIMIT_NPROC": capped("RLIMIT_NPROC", launcher.MAX_REPLAY_PROCESSES),
        "RLIMIT_NOFILE": capped("RLIMIT_NOFILE", launcher.MAX_REPLAY_OPEN_FILES),
    }


def test_a_replay_command_the_image_cannot_start_is_the_tasks_protocol_fault(verifier):
    verifier.replay()
    verifier.config["replay"]["command"] = ["platoon-no-such-program"]
    verifier.submit(42)

    _, reward, report = verifier.run()

    assert reward == {"invalid": 1, "failure_kind": "verifier_generated_protocol"}
    assert report["detail"].startswith("the replay's command cannot start")


@pytest.mark.parametrize(
    "setup", [lambda v: None, lambda v: (v.workdir / "run.sh").write_text("echo")], ids=["missing", "not-executable"]
)
def test_a_replay_program_the_submission_lacks_or_cannot_run_scores_zero_as_the_solvers_fault(verifier, setup):
    """``./run.sh`` is the submission's own program: a submission without it, or with one that cannot run, does not
    replay, which is the solver's fault, not the task's."""

    verifier.replay()
    verifier.config["replay"]["command"] = ["./run.sh"]
    verifier.submit(42)
    setup(verifier)

    _, reward, report = verifier.run()

    assert reward == {"reward": 0}
    assert report["reason"] == "solver_nonreproducible" and report["detail"].startswith("the replay's command cannot")


def test_a_replay_that_names_its_files_by_their_absolute_paths_reproduces_its_outputs(verifier):
    """The replay runs in the working directory itself, at its own path: a submission that reads its inputs and writes
    its outputs by their absolute paths, as a solver that ran there would, replays as it ran, and the grader grades
    what it wrote (42), not what was submitted (41)."""

    here = str(verifier.workdir)
    absolute = f"""
        import json
        with open({here + "/data.txt"!r}) as stream:
            value = 2 * int(stream.read())
        with open({here + "/answer.json"!r}, "w") as stream:
            json.dump({{"answer": value}}, stream)
    """
    verifier.replay()
    verifier.config["artifacts"] = ["answer.json", "solve.py"]
    verifier.submit(41, **{"solve.py": absolute, "data.txt": "21"})

    _, reward, report = verifier.run()

    assert reward == {"reward": 1}, report
    assert json.loads((verifier.workdir / "answer.json").read_text()) == {"answer": 42}


def test_the_replay_runs_at_the_working_directorys_path_with_a_home_of_its_own_that_goes_after_it(verifier):
    """Its working directory is the verifier's, by its own path; its home and temporary directory are one directory of
    its own, apart from the working directory and removed after it, with what it wrote there."""

    verifier.replay()
    verifier.config["artifacts"] = ["answer.json", "solve.py"]
    probe = """
        import json, os
        seen = {"cwd": os.getcwd(), **{name: os.environ.get(name) for name in ("PWD", "HOME", "TMPDIR")}}
        for name in ("HOME", "TMPDIR"):
            with open(os.path.join(os.environ[name], name + ".left"), "w") as stream:
                stream.write("x")
        with open("answer.json", "w") as stream:
            json.dump({"answer": 42, "seen": seen}, stream)
    """
    verifier.submit(42, **{"solve.py": probe})

    _, reward, report = verifier.run()

    assert reward == {"reward": 0} and report["reason"] is None, report  # the grader graded what the replay wrote
    seen = json.loads((verifier.workdir / "answer.json").read_text())["seen"]
    assert seen["cwd"] == seen["PWD"] == str(verifier.workdir)
    home = Path(seen["HOME"])
    assert seen["TMPDIR"] == str(home) and home.name == "home"
    assert home.parent.parent == verifier.scratch and home.parent.name.startswith("platoon-launcher-")
    assert not home.exists()
    assert sorted(path.name for path in verifier.workdir.iterdir()) == ["answer.json", "solve.py"]


def test_the_replay_ends_early_enough_to_put_the_working_directory_back_in_its_time(verifier):
    """The replay's share of the time covers copying the working directory aside and putting it back, which takes
    about twice as long (a copy back, and walks to remove what was lent and harden it again): the replay ends twice as
    long before its share is over as the copy took (here 0.5 s of a share of 2 s, so at most 0.5 s is left), and a copy
    that leaves it no time is the solver's 0."""

    verifier.replay()
    verifier.config.update(artifacts=["solve.py"], timeout_s=9.0)  # 4 s for the launcher's steps: 2 s for the replay
    verifier.submit(None, **{"solve.py": "import time\ntime.sleep(60)\n"})
    slow_copy = """
        import time
        _snapshot = launcher._snapshot

        def slow(*args, **kwargs):
            time.sleep({seconds})
            return _snapshot(*args, **kwargs)

        launcher._snapshot = slow
    """

    report = verifier.run(extra=slow_copy.format(seconds=0.5))[2]

    assert report["reason"] == "solver_nonreproducible" and report["replay"]["timed_out"], report
    assert report["replay"]["timeout_s"] <= 0.6

    report = verifier.run(extra=slow_copy.format(seconds=0.8))[2]

    assert (report["reward"], report["reason"]) == (0, "solver_nonreproducible"), report
    assert report["detail"] == "copying the working directory for the replay, and back, leaves it no time"
    assert "replay" not in report


def test_putting_the_working_directory_back_past_the_replays_time_scores_zero(verifier):
    """The copy of the working directory back from its pristine copy has the replay's time and the window for removing
    what it left: one that takes longer, as a working directory too large to copy in time does, is the solver's 0."""

    verifier.replay()
    verifier.config["artifacts"] = ["answer.json", "solve.py"]
    verifier.submit(42, **{"solve.py": SOLVE, "data.txt": "21"})
    late = """
        _clone_tree = launcher._clone_tree

        def late(source, target, *args, **kwargs):
            if os.path.realpath(target) == os.path.realpath(paths.workdir):
                kwargs["deadline"] = 0.0  # long past
            return _clone_tree(source, target, *args, **kwargs)

        launcher._clone_tree = late
    """

    _, reward, report = verifier.run(extra=late)

    assert reward == {"reward": 0}
    assert (report["reason"], report["detail"]) == (
        "solver_nonreproducible",
        "putting the working directory back outlived the replay's time",
    )
    assert "grader" not in report


# --------------------------------------------------------------------------------------- faults on the trusted side


def test_a_launcher_that_does_not_run_as_root_fails_closed_as_a_trusted_fault(verifier):
    verifier.submit(42)

    code, reward, report = verifier.run(euid=1000)

    assert (code, reward) == (0, {"invalid": 1, "failure_kind": "verifier_trusted_launcher"})
    assert "runs as uid 1000, not root" in report["detail"]
    assert not (verifier.logs / "gen").exists()  # nothing of the task ran


@pytest.mark.parametrize(
    "settings",
    [None, '{"schema": "something/v9"}', json.dumps({"schema": launcher.SCHEMA, "grader_owner": "someone"})],
    ids=["missing", "other-schema", "unusable"],
)
def test_settings_the_launcher_cannot_use_are_a_trusted_fault(verifier, settings):
    verifier.submit(42)

    code, reward, report = verifier.run(settings=settings)

    assert (code, reward) == (0, {"invalid": 1, "failure_kind": "verifier_trusted_launcher"})
    assert "its settings" in report["detail"]


def test_settings_nested_too_deeply_are_a_trusted_fault(verifier):
    verifier.submit(42)
    settings = json.dumps(verifier.config)[:-1] + f', "deep": {DEEP_JSON["parsable"]}}}'

    code, reward, report = verifier.run(settings=settings)

    assert (code, reward) == (0, {"invalid": 1, "failure_kind": "verifier_trusted_launcher"})
    assert "its settings" in report["detail"] and "nested too deeply" in report["detail"]


@pytest.mark.parametrize("artifact", ["/tests/gen", "../answer.json", "./answer.json"])
def test_artifacts_outside_the_working_directory_are_settings_the_launcher_cannot_use(verifier, artifact):
    """``materialize`` refuses them; settings that name one anyway grade nothing."""

    verifier.config["artifacts"] = [artifact]
    verifier.submit(42)

    _, reward, report = verifier.run()

    assert reward == {"invalid": 1, "failure_kind": "verifier_trusted_launcher"}
    assert "hold no usable artifacts" in report["detail"]


def test_a_working_directory_inside_the_verifiers_logs_is_a_trusted_fault(verifier):
    verifier.submit(42)
    inside = verifier.logs / "app"
    inside.mkdir()

    _, reward, report = verifier.run(workdir=inside)

    assert reward == {"invalid": 1, "failure_kind": "verifier_trusted_launcher"}
    assert "are one inside the other" in report["detail"]


def test_a_seal_that_does_not_take_fails_closed_as_a_trusted_fault(verifier):
    verifier.submit(42)
    verifier.tests.chmod(0o755)

    _, reward, report = verifier.run(extra="WithoutRoot.seal = lambda self, path: None\n")

    assert reward == {"invalid": 1, "failure_kind": "verifier_trusted_launcher"}
    assert (
        report["detail"] == f"{verifier.tests} is still open to other users once sealed (owner {os.getuid()}, mode 755)"
    )
    assert not (verifier.logs / "gen").exists()  # nothing of the task ran


def test_a_seal_that_cannot_be_made_fails_closed_as_a_trusted_fault(verifier):
    verifier.submit(42)
    extra = """
        import errno

        def read_only(self, path):
            raise OSError(errno.EROFS, "Read-only file system", path)

        WithoutRoot.seal = read_only
    """

    _, reward, report = verifier.run(extra=extra)

    assert reward == {"invalid": 1, "failure_kind": "verifier_trusted_launcher"}
    assert report["detail"].startswith(f"it cannot seal {verifier.tests}: ") and "Read-only" in report["detail"]


def test_what_the_agents_published_is_sealed_where_it_is(verifier):
    """A copy of the submitted outputs in ``/logs/artifacts`` would let a replay reproduce them without computing them;
    a missing ``/logs/agent`` has nothing to seal."""

    shutil.rmtree(verifier.published[0])
    (verifier.published[1] / "answer.json").write_text(json.dumps({"answer": 42}))
    verifier.published[1].chmod(0o755)
    verifier.submit(42)

    code, reward, _ = verifier.run()

    assert (code, reward) == (0, {"reward": 1})
    assert stat.S_IMODE(verifier.published[1].stat().st_mode) == 0o700 and not verifier.published[0].exists()


def test_a_published_directory_that_is_a_link_fails_closed(verifier):
    """Sealing a link would seal, and then trust, whatever it points to."""

    shutil.rmtree(verifier.published[1])
    (verifier.root / "elsewhere").mkdir()
    verifier.published[1].symlink_to(verifier.root / "elsewhere")
    verifier.submit(42)

    _, reward, report = verifier.run()

    assert reward == {"invalid": 1, "failure_kind": "verifier_trusted_launcher"}
    assert report["detail"] == f"{verifier.published[1]} is a symbolic link, which it cannot seal"


@pytest.mark.parametrize(
    ("mode", "sealed"), [(0o777, False), (0o1777, True), (0o775, True)], ids=["everyones", "sticky", "a-groups"]
)
def test_a_sealed_directory_the_replays_user_could_replace_fails_closed(verifier, mode, sealed):
    """A directory that every user may write in, without the sticky bit, would let the replay's user rename a sealed
    directory away and put one of its own in its place; a group's write permission is not the replay's."""

    holder = verifier.root  # it holds tests/ and logs/
    verifier.submit(42)
    original = stat.S_IMODE(holder.stat().st_mode)
    holder.chmod(mode)
    try:
        _, reward, report = verifier.run()
    finally:
        holder.chmod(original)

    if sealed:
        assert reward == {"reward": 1}, report
    else:
        assert reward == {"invalid": 1, "failure_kind": "verifier_trusted_launcher"}
        assert report["detail"].startswith(f"the directory that holds {verifier.tests} lets the replay's user replace")


@pytest.mark.parametrize(
    ("where", "mode", "exposed"),
    [(".", 0o777, True), (".", 0o1777, True), ("out", 0o777, True), (".", 0o775, False)],
    ids=["everyones", "sticky", "below", "a-groups"],
)
def test_a_working_directory_the_replays_user_may_write_in_fails_closed(verifier, where, mode, exposed):
    """Where the replay's user may write in the working directory, even with the sticky bit, it could put a file where
    an output goes back, or a link where its directory would be made, or change what the grader reads: an image's
    working directory that every user may write in is a fault on the trusted side, found before anything replays."""

    verifier.replay()
    verifier.config["artifacts"] = ["answer.json", "solve.py", "out"]
    (verifier.workdir / "out").mkdir()
    verifier.submit(42, **{"solve.py": SOLVE, "data.txt": "21"})
    directory = verifier.workdir / where
    original = stat.S_IMODE(directory.stat().st_mode)
    directory.chmod(mode)
    try:
        _, reward, report = verifier.run()
    finally:
        directory.chmod(original)

    if not exposed:
        assert reward == {"reward": 1}, report
        return
    assert reward == {"invalid": 1, "failure_kind": "verifier_trusted_launcher"}
    shown = os.path.normpath(directory)
    assert report["detail"].startswith(f"{shown}, in the verifier's working directory, lets the replay's user write")
    assert "replay" not in report and "grader" not in report


def test_the_replays_user_is_never_root(verifier):
    verifier.replay()
    verifier.submit(42, **{"solve.py": SOLVE, "data.txt": "21"})

    _, reward, report = verifier.run(extra="launcher.REPLAY_UID = 0\n")

    assert reward == {"invalid": 1, "failure_kind": "verifier_trusted_launcher"}
    assert report["detail"] == "the replay's user must be a uid above 0 (root's), not 0"


def test_settings_whose_replay_env_names_no_variables_are_unusable(verifier):
    verifier.config["replay_env"] = ["SITE_DATA", "1BAD"]
    verifier.submit(42)

    _, reward, report = verifier.run()

    assert reward == {"invalid": 1, "failure_kind": "verifier_trusted_launcher"}
    assert "hold no usable replay_env" in report["detail"]


def test_details_the_launchers_output_cannot_encode_never_stop_it(verifier):
    """The launcher prints its detail, which may hold a file name of the submission's, to an output whose encoding may
    take ASCII only: it escapes what the encoding cannot take."""

    verifier.config["artifacts"] = ["out"]
    (verifier.workdir / "out").mkdir()
    (verifier.workdir / "out" / "\u7b54\u3048.json").write_text("{")

    code, reward, report = verifier.run(env={"PYTHONIOENCODING": "ascii"})

    assert (code, reward) == (0, {"reward": 0})
    assert report["detail"].startswith("out/\u7b54\u3048.json is not JSON")
    assert "out/\\u7b54\\u3048.json is not JSON" in (verifier.logs / launcher.STDOUT).read_text()


def test_a_report_too_large_for_its_bound_keeps_what_the_outcome_needs():
    report = {
        "schema": launcher.SCHEMA,
        "grader_owner": "generated",
        "contract": {"artifacts": ["a" * 1000] * 10, "violation": None},
        "verdict": {"passed": True, "checks": [{"id": "x" * 200, "passed": True}] * 10},
        "reward": 1,
        "failure_kind": None,
        "reason": None,
        "detail": "d" * 5000,
        "seconds": 1.5,
    }

    assert json.loads(launcher.bounded_report(report)) == report  # it fits
    kept = json.loads(launcher.bounded_report(report, limit=7000))
    assert kept["truncated"] == ["verdict.checks", "contract.artifacts"] and kept["contract"] == {"violation": None}
    assert len(launcher.bounded_report(report, limit=7000)) <= 7000
    least = json.loads(launcher.bounded_report(report, limit=100))
    assert least == {
        "schema": launcher.SCHEMA,
        "grader_owner": "generated",
        "reward": 1,
        "failure_kind": None,
        "reason": None,
        "seconds": 1.5,
        "detail": "d" * launcher.TAIL_CHARS,
        "truncated": "all but the outcome",
    }


def test_the_launcher_parses_as_python_3_8_and_imports_only_the_standard_library():
    source = LAUNCHER.read_text()
    tree = ast.parse(source, feature_version=(3, 8))

    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in (node.names if isinstance(node, ast.Import) else [ast.alias(node.module or "")])
    }
    assert imported - {"__future__"} <= set(sys.stdlib_module_names), imported


def test_descendants_reads_the_process_tree_from_proc(tmp_path):
    for pid, parent, state in ((10, 1, "S"), (11, 10, "R"), (12, 11, "S"), (13, 10, "Z"), (14, 2, "S")):
        (tmp_path / str(pid)).mkdir()
        (tmp_path / str(pid) / "stat").write_text(f"{pid} (a) name) {state} {parent} 1 1\n")
    (tmp_path / "self").mkdir()

    assert sorted(launcher.descendants(10, proc=str(tmp_path))) == [11, 12]  # never the zombie, nor another tree


def test_user_processes_reads_every_processs_users_from_proc(tmp_path):
    rows = (
        (20, "65533\t65533\t65533\t65533", "S (sleeping)"),
        (21, "0\t65533\t0\t0", "R (running)"),  # one of its ids is the user's
        (22, "65533\t65533\t65533\t65533", "Z (zombie)"),
        (23, "1000\t1000\t1000\t1000", "S (sleeping)"),
        (os.getpid(), "65533\t65533\t65533\t65533", "R (running)"),  # the launcher itself
    )
    for pid, uids, state in rows:
        (tmp_path / str(pid)).mkdir()
        (tmp_path / str(pid) / "status").write_text(f"Name:\ta: b\nState:\t{state}\nUid:\t{uids}\n")
    (tmp_path / "self").mkdir()

    assert sorted(launcher.user_processes(65533, proc=str(tmp_path))) == [20, 21]


def test_kill_left_counts_what_outran_it_and_never_what_is_not_its_own(monkeypatch):
    impossible = 2**31 - 1  # above any pid_max: no real process is ever signalled
    monkeypatch.setattr(launcher, "descendants", lambda root, proc="/proc": [])
    monkeypatch.setattr(launcher, "_reap", lambda: None)
    monkeypatch.setattr(launcher, "_signal", lambda pid: None)  # it never ends

    assert launcher.kill_left(lambda: [impossible], window_s=0.05) == (1, 1)
    assert launcher.kill_left(lambda: [], window_s=0.05) == (0, 0)


def test_kill_left_never_excludes_a_replay_user_process_it_cannot_signal(monkeypatch):
    """A process below the launcher it may not signal is foreign (another user namespace's, in a view of /proc that
    holds more than the sandbox), counted as no survivor; but a process of the replay's user it may not signal is a
    survivor all the same (I1), since the replay's user is the sandbox's own and the launcher is root over it."""

    impossible = 2**31 - 1
    monkeypatch.setattr(launcher, "_reap", lambda: None)

    def foreign(pid: int) -> None:
        raise PermissionError(pid)

    monkeypatch.setattr(launcher, "_signal", foreign)
    # In the launcher's own descendants tree: foreign, excluded.
    monkeypatch.setattr(launcher, "descendants", lambda root, proc="/proc": [impossible])
    assert launcher.kill_left(window_s=0.05) == (0, 0)
    # Of the replay's user, named by ``also``: a survivor, the solver's 0, never silently excluded.
    monkeypatch.setattr(launcher, "descendants", lambda root, proc="/proc": [])
    assert launcher.kill_left(lambda: [impossible], window_s=0.05) == (0, 1)


def _namespaces_work(*flags: str) -> bool:
    if shutil.which("unshare") is None:
        return False
    return subprocess.run(["unshare", *flags, "true"], capture_output=True, timeout=60).returncode == 0


_USER_NAMESPACES = pytest.mark.skipif(not _namespaces_work("-r"), reason="unprivileged user namespaces are unavailable")
_AS_ROOT = {"ops": "launcher.Ops()", "prefix": ("unshare", "-r")}


@_USER_NAMESPACES
def test_its_own_operations_seal_and_grade_as_root_in_a_user_namespace(verifier):
    """``unshare -r`` makes the launcher root in a user namespace that maps this user alone: its own seal, which chowns
    each directory to root and checks that it took, seals, and a task without a replay grades."""

    sealed = (verifier.tests, verifier.logs, *verifier.published)
    for path in sealed:
        path.chmod(0o755)
    verifier.submit(42)

    code, reward, report = verifier.run(**_AS_ROOT)

    assert (code, reward) == (0, {"reward": 1}), report
    for path in sealed:
        assert (stat.S_IMODE(path.stat().st_mode), path.stat().st_uid) == (0o700, os.getuid())


@_USER_NAMESPACES
def test_its_own_operations_fail_closed_where_the_sandbox_has_no_replay_user(verifier):
    """A user namespace that maps root alone has no replay's user: the launcher replays nothing, a fault on the
    trusted side, never a score."""

    verifier.replay()
    verifier.submit(42, **{"solve.py": SOLVE, "data.txt": "21"})

    code, reward, report = verifier.run(**_AS_ROOT)

    assert (code, reward) == (0, {"invalid": 1, "failure_kind": "verifier_trusted_launcher"}), report
    assert "the replay's" in report["detail"] and "grader" not in report


# Run as PID 1 of a fresh user and PID namespace, where every process is the test's: a replay that daemonizes, whose
# daemon the sweep finds by its user alone (the process tree is left out).
_SWEEP_IN_A_NAMESPACE = """
import os, sys, time
sys.path.insert(0, {directory!r})
import launcher
assert os.getpid() == 1
read, write = os.pipe()
if os.fork() == 0:
    os.setsid()
    if os.fork() == 0:
        os.write(write, str(os.getpid()).encode())
        time.sleep(60)
        os._exit(0)
    os._exit(0)
os.waitpid(-1, 0)
daemon = int(os.read(read, 32))
launcher.descendants = lambda root, proc="/proc": []
found = launcher.user_processes(os.geteuid()) == [daemon]
print(found, *launcher.kill_left(lambda: launcher.user_processes(os.geteuid())), launcher.user_processes(os.geteuid()))
"""


@pytest.mark.skipif(
    not _namespaces_work("-r", "--pid", "--fork", "--mount-proc"), reason="unprivileged PID namespaces are unavailable"
)
def test_the_sweep_ends_a_daemon_of_the_replays_user_wherever_it_is():
    code = _SWEEP_IN_A_NAMESPACE.format(directory=str(LAUNCHER.parent))

    done = subprocess.run(
        ["unshare", "-r", "--pid", "--fork", "--mount-proc", sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["True", "1", "0", "[]"]


# ------------------------------------------------------------------------ hardening invariants (I1-I5)

# I1 has its sweep regression above (test_kill_left_never_excludes_a_replay_user_process_it_cannot_signal); the rest
# cover disk, inode and memory exhaustion by a replay (I2), the working directory before the replay (I3), the
# replay's PATH (I4), and the working directory after it (I5): whatever a replay did there, the grader finds the
# pristine copy taken before it and the replayed outputs.


def test_clear_left_removes_what_the_replays_user_left_and_nothing_else(tmp_path):
    """It removes the user's entries in each directory, and below it through directories that user may write in, but
    never the launcher's own, an entry of another user's, what a directory the user may not write in holds, or what a
    link names; a directory the sandbox lacks, and one named twice (through a link), are no fault."""

    tmp, shm, elsewhere = tmp_path / "tmp", tmp_path / "shm", tmp_path / "elsewhere"
    for directory in (tmp, shm, elsewhere, tmp / "public", tmp / "private", tmp / "launcher", tmp / "mine"):
        directory.mkdir()
    (tmp_path / "tmp-again").symlink_to(tmp)
    (elsewhere / "kept").write_text("kept")
    for path in (tmp / "public" / "mine", tmp / "private" / "mine", tmp / "launcher" / "mine", tmp / "theirs"):
        path.write_text("x")
    (tmp / "mine" / "deep").mkdir()
    (tmp / "mine" / "deep" / "f").write_text("x")
    (tmp / "mine-link").symlink_to(elsewhere)
    os.mkfifo(shm / "mine")
    inodes = {path.lstat().st_ino: path.name for path in tmp_path.rglob("*")}
    owned = {"mine", "mine-link", "launcher", "deep", "f"}

    removed = launcher.clear_left(
        [str(tmp), str(shm), str(tmp_path / "missing"), str(tmp_path / "tmp-again")],
        owned=lambda info: inodes.get(info.st_ino) in owned,
        writable=lambda info, path: inodes.get(info.st_ino) == "public",
        skip=str(tmp / "launcher"),
    )

    assert removed == 4  # tmp/mine (with all below it), tmp/mine-link, tmp/public/mine and shm/mine
    assert sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*")) == [
        "elsewhere",
        "elsewhere/kept",
        "shm",
        "tmp",
        "tmp-again",
        "tmp/launcher",
        "tmp/launcher/mine",
        "tmp/private",
        "tmp/private/mine",
        "tmp/public",
        "tmp/theirs",
    ]


def test_remove_tree_never_follows_a_link_and_reaches_paths_longer_than_any_the_system_takes(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "kept").write_text("kept")
    root = tmp_path / "tree"
    root.mkdir()
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for _ in range(launcher.MAX_TREE_DEPTH - 1):  # a name of 30 characters a level: 7,680 characters in all
            os.mkdir("n" * 30, dir_fd=descriptor)
            inner = os.open("n" * 30, os.O_RDONLY | os.O_DIRECTORY, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = inner
        os.symlink(str(outside), "link", dir_fd=descriptor)
        os.mkfifo("fifo", dir_fd=descriptor)
        os.close(os.open("file", os.O_WRONLY | os.O_CREAT, 0o600, dir_fd=descriptor))
    finally:
        os.close(descriptor)

    launcher.remove_tree(str(root))

    assert not root.exists() and (outside / "kept").read_text() == "kept"
    (tmp_path / "link").symlink_to(outside)
    launcher.remove_tree(str(tmp_path / "link"))  # a link is removed, never what it names
    assert not (tmp_path / "link").exists() and (outside / "kept").exists()


def test_remove_tree_stops_once_its_time_is_over(tmp_path):
    for n in range(launcher._CLEAR_BATCH + 1):
        (tmp_path / f"f{n}").touch()

    with pytest.raises(launcher.Leftover, match="its time to remove them was over"):
        launcher.remove_tree(str(tmp_path), until=time.monotonic() - 1)


def test_what_the_replays_user_left_where_every_user_may_write_is_removed_before_the_grader_runs(verifier):
    """I2: what the replay's user left in the verifier's /tmp, /var/tmp and /dev/shm is gone before the grader runs,
    so the grader neither finds it nor goes without the room it took; the launcher's own directory stays."""

    verifier.replay()
    verifier.config["artifacts"] = ["answer.json", "solve.py"]
    places = [str(path) for path in verifier.temporary]
    leaving = SOLVE + textwrap.dedent(
        f"""
        import os
        for place in {places!r}:
            os.makedirs(os.path.join(place, "left", "inner"))
            open(os.path.join(place, "left", "inner", "junk"), "w").write("x")
        """
    )
    verifier.submit(42, **{"solve.py": leaving, "data.txt": "21"})

    _, reward, report = verifier.run()

    assert reward == {"reward": 1}, report
    assert report["replay"]["cleared"] == 3
    assert not (verifier.temporary[0] / "left").exists()
    assert [sorted(path.name for path in place.iterdir()) for place in verifier.temporary[1:]] == [[], []]
    assert [path.name for path in verifier.scratch.iterdir()][0].startswith("platoon-launcher-")


def test_a_replay_that_leaves_more_than_the_launcher_removes_in_its_time_scores_zero(verifier):
    """I2: the launcher removes what a replay left for at most a while past its time; here none, and 600 files take
    more than a look at the clock allows, so the submission is the solver's 0, never a launcher fault."""

    verifier.replay()
    leaving = SOLVE + "\nfor n in range(600):\n    open('left-%d' % n, 'w').close()\n"
    verifier.submit(42, **{"solve.py": leaving, "data.txt": "21"})

    _, reward, report = verifier.run(extra="launcher.CLEAR_WINDOW_S = -1000.0\n")

    assert reward == {"reward": 0}
    assert report["reason"] == "solver_nonreproducible" and report["detail"].startswith(
        "the replay left what the launcher cannot remove in its time: its time to remove them was over"
    ), report["detail"]
    assert "grader" not in report


def test_a_replay_that_leaves_a_tree_too_deep_to_remove_scores_zero(verifier):
    """I2: a tree too deep for the launcher to remove is the solver's 0, whatever the launcher's open-file limit."""

    verifier.replay()
    deep = SOLVE + textwrap.dedent(
        f"""
        import os
        for _ in range({launcher.MAX_TREE_DEPTH + 1}):
            os.mkdir("d")
            os.chdir("d")
        """
    )
    verifier.submit(42, **{"solve.py": deep, "data.txt": "21"})

    _, reward, report = verifier.run()

    assert (reward, report["reason"]) == ({"reward": 0}, "solver_nonreproducible")
    assert f"d nests more than {launcher.MAX_TREE_DEPTH} levels deep" in report["detail"]


# A filesystem found full as the launcher copies into the working directory after the replay: as it puts the
# working directory back from its pristine copy, or as it puts the replay's outputs back.
_NO_ROOM_BACK = """
import errno
_copy_file = launcher._copy_file

def full(source, target, mode, times=None):
    if target.startswith({workdir!r}) and target.endswith("answer.json") == {outputs!r}:
        raise OSError(errno.ENOSPC, "No space left on device")
    return _copy_file(source, target, mode, times)

launcher._copy_file = full
"""
_NO_ROOM_STEPS = {
    "restore": (False, "the working directory finds no room as the launcher puts it back"),
    "outputs": (True, "the replay's outputs find no room in the working directory"),
}


@pytest.mark.parametrize("left", [True, False], ids=["the-replay-left-files", "nothing-left"])
@pytest.mark.parametrize("step", _NO_ROOM_STEPS)
def test_a_full_filesystem_putting_the_working_directory_back_is_the_solvers_zero_only_once_the_replay_left_files(
    verifier, left, step
):
    """I2, I5: a filesystem found full as the working directory is put back from its pristine copy, or as the replay's
    outputs go back into it, is the solver's 0 once the replay left files of its user where every user may write (the
    room it lacks is what the replay took); with nothing left, it is the trusted side's fault, as a full filesystem is
    before any replay ran, never a blame the solver can move."""

    shm = verifier.temporary[2]
    solve = SOLVE + (f"\nopen({str(shm / 'left')!r}, 'w').write('x')\n" if left else "")
    verifier.replay()
    verifier.config["artifacts"] = ["answer.json", "solve.py"]
    verifier.submit(42, **{"solve.py": solve, "data.txt": "21"})
    outputs, detail = _NO_ROOM_STEPS[step]

    _, reward, report = verifier.run(extra=_NO_ROOM_BACK.format(workdir=str(verifier.workdir), outputs=outputs))

    if left:
        assert (reward, report["reason"]) == ({"reward": 0}, "solver_nonreproducible"), report
        assert report["detail"].startswith(detail)
        assert report["replay"]["cleared"] >= 1
    else:
        assert reward == {"invalid": 1, "failure_kind": "verifier_trusted_launcher"}, report
        assert "No space left on device" in report["detail"]
    assert "grader" not in report


# A filesystem found full while the launcher holds what the replay wrote: its copy of an output, or its change of
# owner (a filesystem that copies on write may need room for either).
_NO_ROOM_HELD = {
    "copy": (
        """
        import errno
        _copy_file = launcher._copy_file

        def full(source, target, mode, times=None):
            if "/replayed/" in target:
                raise OSError({number}, os.strerror({number}))
            return _copy_file(source, target, mode, times)

        launcher._copy_file = full
        """,
        "WithoutRoot()",
    ),
    "seal": (
        """
        import errno

        class FullOnHold(WithoutRoot):
            def seal(self, path):
                if path.endswith("replayed"):
                    raise OSError({number}, os.strerror({number}))
                return super().seal(path)
        """,
        "FullOnHold()",
    ),
}


@pytest.mark.parametrize("left", [True, False], ids=["the-replay-left-files", "nothing-left"])
@pytest.mark.parametrize(("step", "number"), [("copy", errno.ENOSPC), ("seal", errno.EDQUOT)])
def test_a_full_filesystem_holding_the_outputs_is_judged_as_one_putting_them_back_is(verifier, left, step, number):
    """I2: a filesystem found full while the launcher holds what the replay wrote is judged as one found full putting
    the outputs back: the solver's 0 once the replay left files of its user where every user may write, else the
    trusted side's, never a launcher fault the solver can move its blame onto."""

    shm = verifier.temporary[2]
    solve = SOLVE + (f"\nopen({str(shm / 'left')!r}, 'w').write('x')\n" if left else "")
    verifier.replay()
    verifier.config["artifacts"] = ["answer.json", "solve.py"]
    verifier.submit(42, **{"solve.py": solve, "data.txt": "21"})
    extra, ops = _NO_ROOM_HELD[step]

    _, reward, report = verifier.run(extra=extra.format(number=number), ops=ops)

    if left:
        assert (reward, report["reason"]) == ({"reward": 0}, "solver_nonreproducible"), report
        assert report["detail"].startswith("the replay's outputs find no room where the launcher holds them")
        assert report["replay"]["cleared"] >= 1
    else:
        assert reward == {"invalid": 1, "failure_kind": "verifier_trusted_launcher"}, report
        assert os.strerror(number) in report["detail"]
    assert "grader" not in report


@pytest.mark.parametrize("step", ["copy", "seal"])
def test_holding_the_outputs_failing_otherwise_is_the_trusted_sides_fault_whatever_the_replay_left(verifier, step):
    """I2's line: only a full filesystem is judged by what the replay left; any other failure to hold what it wrote is
    the launcher's own, before the grader runs."""

    shm = verifier.temporary[2]
    verifier.replay()
    verifier.config["artifacts"] = ["answer.json", "solve.py"]
    verifier.submit(42, **{"solve.py": SOLVE + f"\nopen({str(shm / 'left')!r}, 'w').write('x')\n", "data.txt": "21"})
    extra, ops = _NO_ROOM_HELD[step]

    _, reward, report = verifier.run(extra=extra.format(number=errno.EIO), ops=ops)

    assert reward == {"invalid": 1, "failure_kind": "verifier_trusted_launcher"}, report
    assert report["detail"].startswith("it cannot hold what the replay wrote as root's"), report["detail"]


def test_putting_the_working_directory_back_failing_otherwise_is_the_trusted_sides_fault(verifier):
    """I2's line, as the working directory is put back: a failure other than a full filesystem is the launcher's own,
    whatever the replay left."""

    shm = verifier.temporary[2]
    verifier.replay()
    verifier.config["artifacts"] = ["answer.json", "solve.py"]
    verifier.submit(42, **{"solve.py": SOLVE + f"\nopen({str(shm / 'left')!r}, 'w').write('x')\n", "data.txt": "21"})
    extra = """
        import errno

        class BrokenHarden(WithoutRoot):
            hardened = 0

            def harden(self, path):
                BrokenHarden.hardened += 1
                if BrokenHarden.hardened > 1:  # the first is before the replay, the second as it is put back
                    raise OSError(errno.EIO, os.strerror(errno.EIO))
                return super().harden(path)
    """

    _, reward, report = verifier.run(extra=extra, ops="BrokenHarden()")

    assert reward == {"invalid": 1, "failure_kind": "verifier_trusted_launcher"}, report
    assert report["detail"].startswith("it cannot put the working directory back"), report["detail"]


def test_the_launcher_frees_what_the_replay_left_in_the_working_directory_before_it_holds_the_outputs(
    tmp_path, monkeypatch
):
    """I2: everything the replay left in the working directory but its outputs is gone before the hold's first step
    that may need room (its copy of an output), so the room the replay took there never decides whose fault a full
    filesystem is; what it holds is root's alone."""

    workdir, held = tmp_path / "app", tmp_path / "replayed"
    (workdir / "out" / "junk").mkdir(parents=True)
    (workdir / "out" / "answer.json").write_text("{}")
    (workdir / "out" / "junk" / "big").write_text("x" * 4096)
    (workdir / "data.txt").write_text("21")
    seen = []
    copy_file = launcher._copy_file

    def recording(source, target, mode, times=None):
        seen.append(sorted(str(path.relative_to(workdir)) for path in workdir.rglob("*")))
        copy_file(source, target, mode, times)

    class Seal:
        def seal(self, path):
            os.chmod(path, 0o700)

    monkeypatch.setattr(launcher, "_copy_file", recording)

    launcher._hold(str(workdir), str(held), ["out/answer.json"], Seal(), None)

    assert seen == [["out", "out/answer.json"]]
    assert sorted(str(path.relative_to(held)) for path in held.rglob("*")) == ["out", "out/answer.json"]
    assert stat.S_IMODE((held / "out" / "answer.json").stat().st_mode) == 0o600
    assert stat.S_IMODE((held / "out").stat().st_mode) == 0o700


def test_the_launcher_clears_every_place_the_sandbox_lets_every_user_write(monkeypatch):
    """I2: besides the temporary directories, the launcher clears ``/run/lock`` (every user's to write in Debian's
    images, ``/var/lock`` linking to it) and ``/dev/mqueue`` of what the replay's user left there."""

    monkeypatch.setenv("PLATOON_SANDBOX_ROOT", "/sandbox")

    temporary = launcher.Paths.here().temporary

    assert temporary == tuple(
        f"/sandbox{path}" for path in ("/tmp", "/var/tmp", "/dev/shm", "/run/lock", "/var/lock", "/dev/mqueue")
    )


def _oom_adj_writable() -> bool:
    path = getattr(launcher, "_OOM_SCORE_ADJ", "/proc/self/oom_score_adj")
    try:
        with open(path) as stream:
            original = stream.read().strip()
        with open(path, "w") as stream:
            stream.write(original + "\n")
        return True
    except OSError:
        return False


def test_the_replay_may_be_capped_in_address_space():
    """I2: a replay process may be given a hard address-space limit (RLIMIT_AS), so one process cannot reserve the
    sandbox's memory out from under the launcher."""

    probe = textwrap.dedent(
        f"""
        import json, resource, sys
        sys.path.insert(0, {str(LAUNCHER.parent)!r})
        import launcher
        launcher._limit_replay(1 << 20, 1 << 34)
        print(json.dumps(resource.getrlimit(resource.RLIMIT_AS)))
        """
    )

    done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True, timeout=60)

    hard = resource.getrlimit(resource.RLIMIT_AS)[1]
    want = (1 << 34) if hard == resource.RLIM_INFINITY else min(1 << 34, hard)
    assert json.loads(done.stdout) == [want, want]


@pytest.mark.skipif(not _oom_adj_writable(), reason="/proc/self/oom_score_adj is not writable here")
def test_the_replay_is_the_most_killable_to_the_oom_killer():
    """I2: the replay's process disfavours itself for the OOM killer (oom_score_adj 1000), so memory it runs the
    sandbox out of ends the replay, not the launcher or the grader."""

    probe = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(LAUNCHER.parent)!r})
        import launcher
        launcher._oom_disfavor()
        with open(launcher._OOM_SCORE_ADJ) as stream:
            sys.stdout.write(stream.read().strip())
        """
    )

    done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True, timeout=60)

    assert done.stdout == str(launcher.OOM_SCORE_ADJ)


def test_the_replay_disfavours_itself_for_oom_before_it_drops_privileges(monkeypatch):
    """I2, the ordering the Singularity end-to-end test caught: oom_score_adj must be written while still root.  After
    setuid to the replay's user the process is no longer dumpable, so /proc/self is root's and the write would fail;
    the demote preexec therefore writes it before setgroups/setgid/setuid."""

    calls = []
    monkeypatch.setattr(launcher, "_libc", lambda: None)
    monkeypatch.setattr(launcher, "_oom_disfavor", lambda: calls.append("oom"))
    monkeypatch.setattr(launcher, "_limit_replay", lambda *a, **k: calls.append("limit"))
    monkeypatch.setattr(launcher, "_prctl", lambda *a, **k: calls.append("prctl"))
    monkeypatch.setattr(os, "setgroups", lambda groups: calls.append("setgroups"))
    monkeypatch.setattr(os, "setgid", lambda gid: calls.append("setgid"))
    monkeypatch.setattr(os, "setuid", lambda uid: calls.append("setuid"))
    monkeypatch.setattr(os, "getresuid", lambda: (launcher.REPLAY_UID,) * 3)
    monkeypatch.setattr(os, "getresgid", lambda: (launcher.REPLAY_UID,) * 3)
    monkeypatch.setattr(os, "getgroups", lambda: [])

    launcher.Ops().demote(1 << 20)()

    assert calls.index("oom") < calls.index("setuid"), calls
    assert calls.index("setuid") < calls.index("limit") < calls.index("prctl"), calls


def test_nothing_in_the_working_directory_is_the_replay_users_to_write_before_the_replay(verifier):
    """I3: before the replay, the launcher makes the working directory root's with the group and other write bits
    stripped, files and directories at any depth; a submission file any user may write is normalized, never refused,
    so the solver cannot force a trusted fault, and the honest submission is still graded."""

    verifier.replay()
    verifier.config["artifacts"] = ["answer.json", "solve.py"]
    verifier.submit(42, **{"solve.py": SOLVE, "data.txt": "21"})
    (verifier.workdir / "sub").mkdir()
    loose = verifier.workdir / "sub" / "loose.txt"
    loose.write_text("x")
    loose.chmod(0o666)

    _, reward, report = verifier.run()

    assert reward == {"reward": 1}, report  # normalized, never refused
    assert not stat.S_IMODE(loose.stat().st_mode) & (stat.S_IWGRP | stat.S_IWOTH)


def test_the_replays_path_holds_absolute_entries_alone():
    """I4: the replay's PATH keeps only its absolute entries, so nothing in the replay's own directory stands in for a
    program; a PATH with none left falls back to the default."""

    assert launcher._absolute_path("/usr/bin:.:rel::/bin") == "/usr/bin:/bin"
    assert launcher._absolute_path("") == launcher.DEFAULT_PATH
    assert launcher._absolute_path(".::rel") == launcher.DEFAULT_PATH


_PATH_PROBE_GRADER = """
import json, os
with open(os.path.join(os.environ["PLATOON_WORKDIR"], "answer.json")) as stream:
    passed = json.load(stream).get("answer") == 42
with open(os.environ["PLATOON_VERDICT"], "w") as stream:
    json.dump({"passed": passed}, stream)
"""


def test_the_replay_runs_with_a_path_of_absolute_entries_alone(verifier):
    """I4, end to end: a verifier PATH with an empty, a ``.`` and a relative entry reaches the replay as its absolute
    entries alone."""

    verifier.replay()
    verifier.config["artifacts"] = ["answer.json", "solve.py"]
    verifier.grader(_PATH_PROBE_GRADER)
    probe = """
        import json, os
        with open("answer.json", "w") as stream:
            json.dump({"answer": 42, "path": os.environ.get("PATH", "")}, stream)
    """
    verifier.submit(42, **{"solve.py": probe})

    _, reward, report = verifier.run(env={"PATH": "/usr/bin:.:rel::/bin"})

    assert reward == {"reward": 1}, report
    assert json.loads((verifier.workdir / "answer.json").read_text())["path"] == "/usr/bin:/bin"


def _tree(root: Path) -> dict[str, Any]:
    """What a directory holds, by relative path: a directory, a file's bytes, or a link's text."""

    found: dict[str, Any] = {}
    for path in sorted(root.rglob("*")):
        name = str(path.relative_to(root))
        if path.is_symlink():
            found[name] = ("link", os.readlink(path))
        elif path.is_dir():
            found[name] = ("directory",)
        elif path.is_file():
            found[name] = ("file", path.read_bytes())
        else:
            found[name] = ("special",)
    return found


def test_whatever_the_replay_does_in_the_working_directory_the_grader_finds_it_pristine_with_the_outputs(verifier):
    """I5: the replay, in place, changes an input, removes another, moves the submission's code, leaves files, a
    directory tree, a link and a FIFO, and changes the modes of a file and of the working directory: once it is over,
    the working directory holds what it held before the replay but the submitted output, and the replayed output, each
    file root's to write alone, the directory with its mode from before, and that is what the grader grades."""

    verifier.replay()
    verifier.config["artifacts"] = ["answer.json", "solve.py", "inputs"]
    (verifier.workdir / "inputs").mkdir()
    (verifier.workdir / "inputs" / "table.csv").write_text("a,b\n1,2\n")
    (verifier.workdir / "run.sh").write_text("#!/bin/sh\n")
    (verifier.workdir / "run.sh").chmod(0o755)
    seen = verifier.root / "seen.txt"
    vandal = SOLVE + textwrap.dedent(
        f"""
        import os
        with open({str(seen)!r}, "w") as stream:
            stream.write(os.getcwd())
        with open("data.txt", "w") as stream:
            stream.write("20")
        os.remove("inputs/table.csv")
        os.rename("solve.py", "moved.py")
        os.makedirs("left/deep")
        with open("left/deep/junk", "w") as stream:
            stream.write("x")
        os.symlink("/", "root")
        os.mkfifo("pipe")
        os.chmod("run.sh", 0o600)
        os.chmod(".", 0o700)
        """
    )
    verifier.submit(41, **{"solve.py": vandal, "data.txt": "21"})
    before = _tree(verifier.workdir)
    del before["answer.json"]  # the submitted output, which moves aside
    mode = stat.S_IMODE(verifier.workdir.stat().st_mode)

    _, reward, report = verifier.run()

    assert reward == {"reward": 1}, report  # the grader graded the replayed 42
    assert seen.read_text() == str(verifier.workdir)  # it ran in place
    after = _tree(verifier.workdir)
    assert after.pop("answer.json") == ("file", b'{"answer": 42}')
    assert after == before
    assert stat.S_IMODE(verifier.workdir.stat().st_mode) == mode & ~0o022  # as hardened before the replay
    assert stat.S_IMODE((verifier.workdir / "run.sh").stat().st_mode) == 0o755
    for path in verifier.workdir.rglob("*"):
        assert not path.lstat().st_mode & (stat.S_IWGRP | stat.S_IWOTH), path


def test_the_working_directory_keeps_its_own_user_attributes_and_none_the_replay_gave_it(verifier):
    """I5: a user that may write in a directory may give it user extended attributes, and change or remove those it
    has, as the replay's user may the working directory's, the one entry of it that the launcher puts back without
    making it anew: it gives the directory back exactly those it had before the replay."""

    try:
        os.setxattr(verifier.workdir, "user.platoon-probe", b"x")
        os.removexattr(verifier.workdir, "user.platoon-probe")
    except OSError as exc:
        pytest.skip(f"no user extended attributes here: {exc}")
    os.setxattr(verifier.workdir, "user.image", b"kept")
    os.setxattr(verifier.workdir, "user.dropped", b"put back")
    verifier.replay()
    verifier.config["artifacts"] = ["answer.json", "solve.py"]
    marking = SOLVE + textwrap.dedent(
        """
        import os
        os.setxattr(".", "user.left-by-the-replay", b"x")
        os.setxattr(".", "user.image", b"changed")
        os.removexattr(".", "user.dropped")
        """
    )
    verifier.submit(42, **{"solve.py": marking, "data.txt": "21"})

    _, reward, report = verifier.run()

    assert reward == {"reward": 1}, report
    kept = {name: os.getxattr(verifier.workdir, name) for name in os.listxattr(verifier.workdir)}
    assert kept == {"user.image": b"kept", "user.dropped": b"put back"}


def test_a_link_in_the_working_directory_stays_a_link_that_the_replay_reads_through(verifier):
    """I5: a link the image's working directory holds (a submission brings none: the artifacts are files and
    directories) is in place for the replay, which reads through it, and is put back as a link, never what it names."""

    dataset = verifier.root / "dataset"
    dataset.mkdir()
    (dataset / "value.txt").write_text("21")
    (verifier.workdir / "data").symlink_to(dataset)
    verifier.replay()
    verifier.config["artifacts"] = ["answer.json", "solve.py"]
    verifier.submit(42, **{"solve.py": SOLVE.replace('"data.txt"', '"data/value.txt"')})

    _, reward, report = verifier.run()

    assert reward == {"reward": 1}, report
    assert os.readlink(verifier.workdir / "data") == str(dataset)
    assert _tree(dataset) == {"value.txt": ("file", b"21")}


@pytest.mark.parametrize(
    ("above", "mode", "refused"),
    [(1, 0o777, True), (2, 0o777, True), (1, 0o1777, False), (1, 0o775, False)],
    ids=["parent", "above-the-parent", "sticky", "a-groups"],
)
def test_a_directory_above_the_working_directory_the_replays_user_could_replace_fails_closed(
    verifier, above, mode, refused
):
    """I5: the replay's user writes in the working directory, which stays root's; a directory above it in which that
    user may rename or remove what it holds, the parent or one above it, would let it move the working directory away as
    it replays there and leave another in its place: a fault on the trusted side, found before anything replays.  With
    the sticky bit set, that user may move nothing of root's, as the working directory is; a group's write permission
    is not the replay's."""

    outer = verifier.root / "outer"
    workdir = outer / "inner" / "app" if above == 2 else outer / "app"
    workdir.mkdir(parents=True)
    for name, text in {"answer.json": json.dumps({"answer": 42}), "solve.py": SOLVE, "data.txt": "21"}.items():
        (workdir / name).write_text(text)
    verifier.replay()
    verifier.config["artifacts"] = ["answer.json", "solve.py"]
    original = stat.S_IMODE(outer.stat().st_mode)
    outer.chmod(mode)
    try:
        _, reward, report = verifier.run(workdir=workdir)
    finally:
        outer.chmod(original)

    if not refused:
        assert reward == {"reward": 1}, report
        return
    assert reward == {"invalid": 1, "failure_kind": "verifier_trusted_launcher"}, report
    info = outer.stat()
    assert report["detail"].startswith(
        f"the directory {outer}, above the verifier's working directory {workdir}, lets the replay's user replace what "
        f"it holds (owner {info.st_uid}, group {info.st_gid}, mode {mode:o})"
    ), report["detail"]
    assert "replay" not in report and "grader" not in report


def test_the_launchers_own_directory_inside_the_working_directory_fails_closed(tmp_path):
    """I5: the replay's user writes in the working directory, so the launcher's own directory (its pristine copy, the
    submitted outputs) must not be in it: a fault on the trusted side, found before anything replays."""

    verifier = Verifier(tmp_path, scratch=tmp_path / "app")  # the verifier's /tmp is the working directory's tmp/
    verifier.replay()
    verifier.config["artifacts"] = ["answer.json", "solve.py"]
    verifier.submit(42, **{"solve.py": SOLVE, "data.txt": "21"})

    _, reward, report = verifier.run()

    assert reward == {"invalid": 1, "failure_kind": "verifier_trusted_launcher"}, report
    assert report["detail"].startswith(f"the launcher's own directory {verifier.scratch}/platoon-launcher-")
    assert f"is inside the verifier's working directory {verifier.workdir}" in report["detail"]
    assert "replay" not in report and "grader" not in report


def test_a_working_directory_where_every_user_may_write_is_put_back_never_cleared(verifier):
    """I2, I5: a working directory in one of the places where every user may write (a template's /tmp/work) is left
    out when the launcher clears those places of what the replay's user left: it puts the working directory back
    itself, with the replayed outputs."""

    place = verifier.temporary[1]
    workdir = place / "work"
    workdir.mkdir()
    leaving = SOLVE + f"\nopen({str(place / 'left')!r}, 'w').write('x')\n"
    for name, text in {"answer.json": json.dumps({"answer": 41}), "solve.py": leaving, "data.txt": "21"}.items():
        (workdir / name).write_text(text)
    verifier.replay()
    verifier.config["artifacts"] = ["answer.json", "solve.py"]

    _, reward, report = verifier.run(workdir=workdir)

    assert reward == {"reward": 1}, report
    assert report["replay"]["cleared"] == 1  # what it left beside the working directory
    assert sorted(path.name for path in place.iterdir()) == ["work"]
    assert sorted(path.name for path in workdir.iterdir()) == ["answer.json", "data.txt", "solve.py"]


def _mounts_work() -> bool:
    if shutil.which("unshare") is None:
        return False
    with tempfile.TemporaryDirectory() as directory:
        probe = ["unshare", "-r", "-m", "sh", "-c", 'mount -t tmpfs -o mode=755 tmpfs "$0"', directory]
        return subprocess.run(probe, capture_output=True, timeout=60).returncode == 0


@pytest.mark.skipif(not _mounts_work(), reason="unprivileged mount namespaces are unavailable")
def test_a_mount_point_in_the_working_directory_fails_closed(verifier):
    """I5: a filesystem mounted below the working directory, which no submission can make: the replay would change what
    is on it in place, and the launcher could not remove the mount point to put the directory back, so it refuses
    before anything replays, a fault on the trusted side."""

    verifier.replay()
    verifier.config["artifacts"] = ["answer.json", "solve.py"]
    verifier.submit(42, **{"solve.py": SOLVE, "data.txt": "21"})
    mounted = verifier.workdir / "mounted"
    mounted.mkdir()
    mount = ("unshare", "-r", "-m", "sh", "-c", 'mount -t tmpfs -o mode=755 tmpfs "$0" && exec "$@"', str(mounted))

    _, reward, report = verifier.run(prefix=mount)

    assert reward == {"invalid": 1, "failure_kind": "verifier_trusted_launcher"}, report
    assert report["detail"].startswith(f"{mounted}, in the verifier's working directory, is a mount point")
    assert "replay" not in report and "grader" not in report


def test_lend_gives_the_replays_user_what_the_working_directory_holds_and_keeps_the_directory_roots(
    tmp_path, monkeypatch
):
    """I5: :meth:`Ops.lend` makes the directories and the regular files below the working directory the replay's
    user's, its owner's permissions added (read and write, and search for a directory: an image's read-only files are
    the replay's to change in place, as a copy of them was), and leaves the directory itself root's, of that user's
    group, mode 770: that user may add and remove entries there, but can neither rename the directory nor change its
    mode.  A file with another link (here one outside the working directory) is first replaced with a copy of its own,
    which is lent, so the other name stays root's as it was; a link and a FIFO, which only an image puts there, stay
    root's: lending a link would lend what it reaches beyond the working directory."""

    workdir, outside = tmp_path / "app", tmp_path / "outside"
    (workdir / "sub").mkdir(parents=True)
    (workdir / "sub" / "file").write_text("x")
    (workdir / "sub" / "file").chmod(0o640)
    (workdir / "sub").chmod(0o750)
    (workdir / "read-only").mkdir()
    (workdir / "read-only" / "table.csv").write_text("a,b\n")
    (workdir / "read-only" / "table.csv").chmod(0o444)
    (workdir / "read-only").chmod(0o555)
    outside.mkdir()
    (outside / "tool").write_text("tool")
    (outside / "tool").chmod(0o550)
    os.utime(outside / "tool", ns=(1_000_000_000_000_000_000, 1_000_000_000_000_000_000))
    (workdir / "link").symlink_to(outside)
    os.link(outside / "tool", workdir / "tool")
    os.mkfifo(workdir / "pipe")
    owners: list[tuple[str, int, int]] = []
    modes: list[tuple[str, int]] = []
    monkeypatch.setattr(os, "lchown", lambda path, uid, gid: owners.append((os.path.relpath(path, tmp_path), uid, gid)))
    monkeypatch.setattr(os, "chown", lambda *args, **kwargs: pytest.fail("chown follows links"))
    monkeypatch.setattr(os, "chmod", lambda path, mode: modes.append((os.path.relpath(path, tmp_path), mode)))

    try:
        launcher.Ops().lend(str(workdir))
    finally:
        monkeypatch.undo()
        (workdir / "read-only").chmod(0o755)

    user = launcher.REPLAY_UID
    lent = ["app/read-only", "app/read-only/table.csv", "app/sub", "app/sub/file", "app/tool"]
    assert sorted(owners) == sorted([("app", 0, user), *((name, user, user) for name in lent)])
    assert sorted(modes) == sorted(
        [
            ("app", 0o770),
            ("app/read-only", 0o755),
            ("app/read-only/table.csv", 0o644),
            ("app/sub", 0o750),
            ("app/sub/file", 0o640),
            ("app/tool", 0o750),
        ]
    )
    copied, original = (workdir / "tool").stat(), (outside / "tool").stat()
    assert copied.st_ino != original.st_ino and (copied.st_nlink, original.st_nlink) == (1, 1)
    assert (workdir / "tool").read_text() == "tool" and copied.st_mtime_ns == original.st_mtime_ns
    assert sorted(path.name for path in workdir.iterdir()) == ["link", "pipe", "read-only", "sub", "tool"]


# ------------------------------------------------------------------------- what the working directory gets back (I5)


def _described(root: Path) -> dict[str, Any]:
    """What a grader may find of what a directory holds, by relative path: each entry's kind, mode and modification
    time; a file's bytes (their digest), the names in the directory that name the same file, and, for one of a MiB or
    more, whether it keeps holes; a link's text."""

    found: dict[str, Any] = {}
    names: dict[int, list[str]] = {}
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        name = str(path.relative_to(root))
        shown: tuple[Any, ...] = (stat.S_IFMT(info.st_mode), stat.S_IMODE(info.st_mode), info.st_mtime_ns)
        if stat.S_ISREG(info.st_mode):
            names.setdefault(info.st_ino, []).append(name)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            shown += (digest, info.st_ino, info.st_size >= 1 << 20 and info.st_blocks * 512 < info.st_size // 2)
        elif stat.S_ISLNK(info.st_mode):
            shown += (os.readlink(path),)
        found[name] = shown
    for name, shown in found.items():  # a file's inode, which a copy changes, stands for the names that share it
        if stat.S_ISREG(shown[0]):
            found[name] = (*shown[:4], tuple(names[shown[4]]), shown[5])
    return found


def test_the_working_directory_is_put_back_with_its_modes_times_names_fifos_links_and_holes(verifier):
    """I5, what the pristine copy keeps: whatever the replay does to what an image left in the working directory (it
    changes modes, times and contents, writes in a directory and a file the image left read-only, writes through one
    name of a file of two and removes the other, removes a FIFO and a link, and fills a sparse file's hole), the grader
    finds each entry as it was before the replay: its kind, mode and modification time, its contents, the two names of
    one file, the FIFO, the link and the hole, and nothing the replay added."""

    workdir = verifier.workdir
    files = {"secret.key": 0o600, "tool": 0o750, "a.txt": 0o644, "old.txt": 0o644}
    for name, mode in files.items():
        (workdir / name).write_text(name)
        (workdir / name).chmod(mode)
    os.link(workdir / "a.txt", workdir / "b.txt")
    os.utime(workdir / "old.txt", ns=(1_000_000_000 * 10**9, 1_000_000_000 * 10**9))
    for name, mode in {"private": 0o700, "shared": 0o2755, "read-only": 0o555}.items():
        (workdir / name).mkdir()
        (workdir / name / "inner.txt").write_text(name)
        (workdir / name).chmod(mode)
    (workdir / "read-only" / "inner.txt").chmod(0o444)
    os.mkfifo(workdir / "pipe", 0o640)
    (workdir / "link").symlink_to("a.txt")
    with open(workdir / "sparse.img", "wb") as stream:
        stream.truncate(16 << 20)
        stream.seek(8 << 20)
        stream.write(b"x" * 4096)
    vandal = SOLVE + textwrap.dedent(
        """
        import os
        os.chmod("secret.key", 0o644)
        os.chmod("private", 0o755)
        with open("read-only/inner.txt", "a") as stream:
            stream.write("changed in place")
        open("read-only/added.txt", "w").write("x")
        open("shared/added.txt", "w").write("x")
        with open("a.txt", "a") as stream:
            stream.write("changed through one name")
        os.remove("b.txt")
        os.remove("pipe")
        os.remove("link")
        os.utime("old.txt", (2_000_000_000, 2_000_000_000))
        with open("sparse.img", "r+b") as stream:
            stream.write(b"y" * (1 << 20))
        """
    )
    verifier.replay()
    verifier.config["artifacts"] = ["answer.json", "solve.py"]
    verifier.submit(41, **{"solve.py": vandal, "data.txt": "21"})
    before = _described(workdir)
    del before["answer.json"]  # the submitted output, which moves aside

    try:
        _, reward, report = verifier.run()
        after = _described(workdir)
    finally:
        for name in ("read-only", "private"):  # for the test's own clean-up
            (workdir / name).chmod(0o755)

    assert reward == {"reward": 1}, report  # the replay wrote what the image left read-only, and 42
    assert after.pop("answer.json")[1] == 0o644
    assert after == before
    assert after["a.txt"][4] == ("a.txt", "b.txt") and stat.S_ISFIFO(after["pipe"][0])


def test_a_replay_in_place_never_writes_through_a_name_of_a_file_outside_the_working_directory(verifier):
    """A file of the image's working directory may be another name of a file outside it: the launcher gives the replay
    a copy of its own in its place, which the replay may change, and the file outside keeps what it held."""

    outside = verifier.root / "shared-tool.txt"
    outside.write_text("the image's")
    os.link(outside, verifier.workdir / "tool.txt")
    verifier.replay()
    verifier.config["artifacts"] = ["answer.json", "solve.py"]
    writing = SOLVE + "\nopen('tool.txt', 'w').write('the replay\\'s')\n"
    verifier.submit(41, **{"solve.py": writing, "data.txt": "21"})

    _, reward, report = verifier.run()

    assert reward == {"reward": 1}, report
    assert outside.read_text() == (verifier.workdir / "tool.txt").read_text() == "the image's"


def test_copy_file_keeps_a_files_holes_contents_mode_and_times(tmp_path):
    source, target = tmp_path / "sparse.img", tmp_path / "copy.img"
    with open(source, "wb") as stream:
        stream.truncate(16 << 20)
        stream.seek(8 << 20)
        stream.write(b"x" * 4096)
    times = (1_000_000_000 * 10**9, 1_100_000_000 * 10**9)

    launcher._copy_file(str(source), str(target), 0o640, times)

    copied, original = target.stat(), source.stat()
    assert target.read_bytes() == source.read_bytes()
    assert (stat.S_IMODE(copied.st_mode), copied.st_atime_ns, copied.st_mtime_ns) == (0o640, *times)
    if original.st_blocks * 512 < (8 << 20):  # a filesystem that keeps holes
        assert copied.st_blocks <= original.st_blocks


def test_a_copy_past_its_time_stops_between_directories_too(tmp_path):
    """A tree of directories alone takes time to copy too: each copy looks at its deadline before each directory as
    before each file."""

    source = tmp_path / "source"
    for number in range(50):
        (source / f"d{number}").mkdir(parents=True)
    for copy in (launcher._copy_tree, launcher._clone_tree):
        target = tmp_path / copy.__name__
        target.mkdir()

        with pytest.raises(launcher.SolverFault, match="outlived its time"):
            copy(str(source), str(target), deadline=time.monotonic() - 1)

        assert list(target.iterdir()) == []


def test_the_pristine_copy_refuses_what_another_filesystem_holds_and_what_it_cannot_copy(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    (source / "file").write_text("x")
    target.mkdir()

    with pytest.raises(launcher.LauncherFault, match="is a mount point"):
        launcher._clone_tree(str(source), str(target), device=-1)
    (source / "file").unlink()
    os.mknod(source / "socket", stat.S_IFSOCK | 0o600)
    with pytest.raises(launcher.LauncherFault, match="is a socket or a device"):
        launcher._clone_tree(str(source), str(target))


def test_a_socket_in_the_working_directory_fails_closed_before_anything_changes(verifier):
    """A socket or a device in the working directory, which only an image puts there: the replay could reach it there,
    and the launcher could not copy it and put it back, so it refuses before anything changes there (a file any user
    may write keeps its mode, which hardening would strip), a fault on the trusted side."""

    verifier.replay()
    verifier.config["artifacts"] = ["answer.json", "solve.py"]
    verifier.submit(42, **{"solve.py": SOLVE, "data.txt": "21"})
    (verifier.workdir / "data.txt").chmod(0o666)
    os.mknod(verifier.workdir / "service.sock", stat.S_IFSOCK | 0o600)

    _, reward, report = verifier.run()

    assert reward == {"invalid": 1, "failure_kind": "verifier_trusted_launcher"}, report
    assert report["detail"].startswith(
        f"{verifier.workdir / 'service.sock'}, in the verifier's working directory, is a socket or a device"
    )
    assert "replay" not in report and "grader" not in report
    assert stat.S_IMODE((verifier.workdir / "data.txt").stat().st_mode) == 0o666


@pytest.mark.parametrize("late", ["hardening", "outputs"])
def test_putting_the_working_directory_back_that_ends_past_its_time_scores_zero(verifier, late):
    """The copy back keeps to its time, but what follows it does not (here the window past the replay's time is long
    over): hardening what it copied, or putting the outputs back, that ends past the replay's time is the solver's 0,
    never the grader's time."""

    verifier.replay()
    verifier.config["artifacts"] = ["answer.json", "solve.py"]
    verifier.submit(42, **{"solve.py": SOLVE, "data.txt": "21"})
    unhurried = {
        "hardening": """
            _clone_tree = launcher._clone_tree

            def unhurried(source, target, deadline=None, limit=None, device=None):
                if os.path.realpath(target) == os.path.realpath(paths.workdir):
                    deadline = None
                return _clone_tree(source, target, deadline, limit, device)

            launcher._clone_tree = unhurried
        """,
        "outputs": """
            _restore = launcher._restore
            launcher._restore = lambda workdir, pristine, ops, kept, until: _restore(workdir, pristine, ops, kept, None)
        """,
    }[late]

    _, reward, report = verifier.run(extra="launcher.CLEAR_WINDOW_S = -1000.0\n" + textwrap.dedent(unhurried))

    assert reward == {"reward": 0}, report
    assert (report["reason"], report["detail"]) == (
        "solver_nonreproducible",
        "putting the working directory back outlived the replay's time",
    )
    assert "grader" not in report


def test_putting_the_working_directory_back_checks_its_time_once_it_is_done(tmp_path, monkeypatch):
    """Putting the directory back looks at its time once it is done, as well as while it copies, so that each caller
    (the replay, and each replay that follows it) finds a put back that ended too late, whatever took the time."""

    workdir, pristine = tmp_path / "app", tmp_path / "pristine"
    workdir.mkdir()
    pristine.mkdir()
    (pristine / "data.txt").write_text("21")
    kept = launcher._Directory(str(workdir))
    clone = launcher._clone_tree
    monkeypatch.setattr(launcher, "_clone_tree", lambda source, target, deadline=None: clone(source, target))

    class Hardening:
        def harden(self, path):
            pass

    with pytest.raises(launcher.SolverFault, match="putting the working directory back outlived the replay's time"):
        launcher._restore(str(workdir), str(pristine), Hardening(), kept, time.monotonic() - 1)
    assert (workdir / "data.txt").read_text() == "21"


def test_a_full_filesystem_putting_the_working_directory_back_is_tried_once_more_once_it_is_written_out(verifier):
    """I2: with nothing of the replay's left where every user may write, a filesystem found full as the working
    directory is put back may still be freeing what the launcher removed there (XFS and btrfs do so after a while): the
    launcher has it write out what it holds and puts the directory back once more before it judges a full filesystem
    the trusted side's, and the grade goes on once that is enough."""

    marker = verifier.root / "synced"
    verifier.replay()
    verifier.config["artifacts"] = ["answer.json", "solve.py"]
    verifier.submit(41, **{"solve.py": SOLVE, "data.txt": "21"})
    extra = f"""
        import errno
        _restore = launcher._restore

        def full_once(*args):
            if not os.path.exists({str(marker)!r}):
                raise OSError(errno.ENOSPC, "No space left on device")
            return _restore(*args)

        launcher._restore = full_once
        launcher._sync = lambda path: open({str(marker)!r}, "w").write(path)
    """

    _, reward, report = verifier.run(extra=extra)

    assert reward == {"reward": 1}, report
    assert marker.read_text() == os.path.realpath(verifier.workdir)


def _binds_work() -> bool:
    if shutil.which("unshare") is None:
        return False
    with tempfile.TemporaryDirectory() as directory:
        probe = ["unshare", "-r", "-m", "sh", "-c", 'mount --bind "$0" "$0"', directory]
        return subprocess.run(probe, capture_output=True, timeout=60).returncode == 0


@pytest.mark.skipif(not _binds_work(), reason="unprivileged bind mounts are unavailable")
@pytest.mark.parametrize("kind", ["directory", "file"])
def test_a_bind_mount_in_the_working_directory_fails_closed_before_anything_changes(verifier, kind):
    """I5: a directory or a file bound below the working directory from the same filesystem, which no submission can
    make: the launcher finds it among its mounts before it changes anything there, a fault on the trusted side, and
    what the mount shows elsewhere keeps its contents and its mode, which hardening, the lend and the put back would
    have changed or removed."""

    verifier.replay()
    verifier.config["artifacts"] = ["answer.json", "solve.py"]
    verifier.submit(42, **{"solve.py": SOLVE, "data.txt": "21"})
    source, target = verifier.root / "shared", verifier.workdir / "shared"
    if kind == "directory":
        source.mkdir()
        (source / "precious.txt").write_text("host data")
        target.mkdir()
        source.chmod(0o777)
    else:
        source.write_text("host data")
        target.write_text("")
        source.chmod(0o666)
    bind = ("unshare", "-r", "-m", "sh", "-c", 'mount --bind "$0" "$1" && shift && exec "$@"', str(source), str(target))

    _, reward, report = verifier.run(prefix=bind)

    assert reward == {"invalid": 1, "failure_kind": "verifier_trusted_launcher"}, report
    assert report["detail"].startswith(f"{target}, in the verifier's working directory, is a mount point")
    assert "replay" not in report and "grader" not in report
    precious = source / "precious.txt" if kind == "directory" else source
    assert precious.read_text() == "host data"
    assert stat.S_IMODE(source.stat().st_mode) == (0o777 if kind == "directory" else 0o666)


def test_mount_points_reads_the_kernels_escapes_back_and_finds_what_is_mounted_below(tmp_path):
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_bytes(
        b"22 1 8:4 / / rw,relatime shared:1 - xfs /dev/sda4 rw\n"
        b"30 22 8:4 /app /app rw - xfs /dev/sda4 rw\n"
        b"31 22 8:4 /src /application rw - xfs /dev/sda4 rw\n"
        b"32 30 8:4 /src /app/with\\040a\\011tab rw - xfs /dev/sda4 rw\n"
        b"33 30 8:4 /f /app/back\\134slash\\012line rw - xfs /dev/sda4 rw\n"
    )

    points = launcher.mount_points(mountinfo.read_bytes())

    assert points == ["/", "/app", "/application", "/app/with a\ttab", "/app/back\\slash\nline"]
    assert launcher._mounted_below("/app", mountinfo=str(mountinfo)) == "/app/with a\ttab"
    assert launcher._mounted_below("/application", mountinfo=str(mountinfo)) is None
    with pytest.raises(launcher.LauncherFault, match="cannot read its mounts"):
        launcher._mounted_below("/app", mountinfo=str(tmp_path / "missing"))


def _acl(*entries: tuple[int, int, int]) -> bytes:
    return struct.pack("<I", 2) + b"".join(struct.pack("<HHI", *entry) for entry in entries)


_OWNER, _GROUP, _OTHER = (0x01, 0o7, 0xFFFFFFFF), (0x04, 0o5, 0xFFFFFFFF), (0x20, 0o5, 0xFFFFFFFF)


@pytest.mark.parametrize(
    ("acl", "writable"),
    [
        (_acl(_OWNER, (0x02, 0o7, 65533), _GROUP, (0x10, 0o7, 0xFFFFFFFF), _OTHER), True),
        (_acl(_OWNER, (0x02, 0o7, 65533), _GROUP, (0x10, 0o5, 0xFFFFFFFF), _OTHER), False),
        (_acl(_OWNER, _GROUP, (0x08, 0o2, 65533), (0x10, 0o7, 0xFFFFFFFF), _OTHER), True),
        (_acl(_OWNER, (0x02, 0o7, 1000), _GROUP, (0x08, 0o7, 1000), (0x10, 0o7, 0xFFFFFFFF), _OTHER), False),
        (b"\x02\x00", True),
    ],
    ids=["its-user", "masked", "its-group", "another-user", "unreadable"],
)
def test_an_access_acl_that_lets_the_replays_user_write_counts_as_its_permission(monkeypatch, acl, writable):
    """An entry of the replay's user, or of its group, that grants writing, which the mask lets through, lets it
    write; an ACL the launcher cannot read lets it too, so a check that asks fails closed."""

    monkeypatch.setattr(os, "getxattr", lambda path, name, follow_symlinks=True: acl)

    assert launcher._acl_lets_replay_write("/somewhere") is writable


def test_a_directory_without_an_access_acl_lets_the_replay_write_by_its_mode_bits_alone(monkeypatch):
    def none(path, name, follow_symlinks=True):
        raise OSError(errno.ENODATA, os.strerror(errno.ENODATA))

    monkeypatch.setattr(os, "getxattr", none)

    assert launcher._acl_lets_replay_write("/somewhere") is False


def _acls_work() -> bool:
    if shutil.which("setfacl") is None:
        return False
    with tempfile.TemporaryDirectory() as directory:
        probe = ["setfacl", "-m", f"u:{launcher.REPLAY_UID}:rwx", directory]
        return subprocess.run(probe, capture_output=True, timeout=60).returncode == 0


@pytest.mark.skipif(not _acls_work(), reason="POSIX ACLs are unavailable here")
@pytest.mark.parametrize("sticky", [False, True], ids=["refused", "sticky"])
def test_a_directory_above_the_working_directory_whose_acl_lets_the_replays_user_replace_it_fails_closed(
    verifier, sticky
):
    """I5: the parent's access ACL, not its mode bits, lets the replay's user write there, and so rename the working
    directory away as it replays: a fault on the trusted side, found before anything replays; with the sticky bit set,
    that user may move nothing of root's there."""

    outer = verifier.root / "outer"
    workdir = outer / "app"
    workdir.mkdir(parents=True)
    for name, text in {"answer.json": json.dumps({"answer": 42}), "solve.py": SOLVE, "data.txt": "21"}.items():
        (workdir / name).write_text(text)
    subprocess.run(["setfacl", "-m", f"u:{launcher.REPLAY_UID}:rwx", str(outer)], check=True, timeout=60)
    if sticky:
        outer.chmod(outer.stat().st_mode | stat.S_ISVTX)
    verifier.replay()
    verifier.config["artifacts"] = ["answer.json", "solve.py"]

    _, reward, report = verifier.run(workdir=workdir)

    if sticky:
        assert reward == {"reward": 1}, report
        return
    assert reward == {"invalid": 1, "failure_kind": "verifier_trusted_launcher"}, report
    assert report["detail"].startswith(
        f"the directory {outer}, above the verifier's working directory {workdir}, lets the replay's user replace"
    ), report["detail"]


# --------------------------------------------------------------------- System V IPC objects the replay leaves (I2)


def test_user_ipc_reads_each_kinds_owner_and_creator_from_proc(tmp_path):
    (tmp_path / "sysvipc").mkdir()
    (tmp_path / "sysvipc" / "shm").write_text(
        "       key      shmid perms                  size  cpid  lpid nattch   uid   gid  cuid  cgid      atime      "
        "dtime      ctime                   rss                  swap\n"
        "         0          7  1600              67108864  1234  1234      0 65533 65533 65533 65533          0      "
        "    0 1700000000              67108864                     0\n"
        "         0          8  1600                  4096  1234  1234      0  1000  1000  1000  1000          0      "
        "    0 1700000000                  4096                     0\n"
    )
    (tmp_path / "sysvipc" / "sem").write_text(
        "       key      semid perms      nsems   uid   gid  cuid  cgid      otime      ctime\n"
        "         0          3   600          1     0     0 65533 65533          0 1700000000\n"  # given away, its own
    )

    assert sorted(launcher.user_ipc(65533, proc=str(tmp_path))) == [("sem", 3), ("shm", 7)]  # no msg: none of it


_IPC_NAMESPACES = pytest.mark.skipif(
    not _namespaces_work("-r", "-i"), reason="unprivileged IPC namespaces are unavailable"
)
# Run in a fresh user and IPC namespace, whose System V IPC objects are the test's alone: one of each kind, found by
# their user and removed.
_IPC_IN_A_NAMESPACE = """
import ctypes, os, sys
sys.path.insert(0, {directory!r})
import launcher
libc = ctypes.CDLL(None, use_errno=True)
assert min(libc.shmget(0, 1 << 20, 0o1600), libc.semget(0, 1, 0o1600), libc.msgget(0, 0o1600)) >= 0
found = launcher.user_ipc(os.getuid())
print(sorted(kind for kind, _ in found), launcher.remove_ipc(found), launcher.user_ipc(os.getuid()))
"""


@_IPC_NAMESPACES
def test_remove_ipc_removes_shared_memory_semaphores_and_message_queues():
    code = _IPC_IN_A_NAMESPACE.format(directory=str(LAUNCHER.parent))
    command = ["unshare", "-r", "-i", sys.executable, "-c", code]

    done = subprocess.run(command, capture_output=True, text=True, timeout=60)

    assert done.returncode == 0, done.stderr
    assert done.stdout.split("\n")[0] == "['msg', 'sem', 'shm'] 3 []"


# The replay leaves a touched shared memory segment, a semaphore set and a message queue; the grader passes only on the
# right answer with none of them left.
_LEAVES_IPC = """
import ctypes
libc = ctypes.CDLL(None, use_errno=True)
libc.shmat.restype = ctypes.c_void_p
segment = libc.shmget(0, 1 << 20, 0o1600)
address = libc.shmat(segment, None, 0)
ctypes.memset(address, 1, 1 << 20)
libc.shmdt(ctypes.c_void_p(address))
libc.semget(0, 1, 0o1600)
libc.msgget(0, 0o1600)
"""
_NO_IPC_GRADER = """
import json, os
with open(os.path.join(os.environ["PLATOON_WORKDIR"], "answer.json")) as stream:
    passed = json.load(stream) == {"answer": 42}
for kind in ("shm", "sem", "msg"):
    with open(os.path.join("/proc/sysvipc", kind)) as stream:
        passed = passed and len(stream.read().splitlines()) == 1
with open(os.environ["PLATOON_VERDICT"], "w") as stream:
    json.dump({"passed": passed}, stream)
"""
_IPC_OPS = """
class IpcOps(WithoutRoot):
    def ipc_of(self, uid):
        return launcher.user_ipc(os.getuid())
"""


@_IPC_NAMESPACES
@pytest.mark.parametrize("removed", [True, False], ids=["removed", "cannot-be-removed"])
def test_the_system_v_ipc_objects_the_replay_left_are_removed_before_the_grader_runs(verifier, removed):
    """I2: shared memory, a semaphore set and a message queue outlive the replay's processes, and the memory they hold
    the grader would go without: the launcher removes those of the replay's user before the grader runs, counted with
    what that user left where every user may write; one it cannot remove is the solver's 0, as a process that cannot
    be killed is."""

    verifier.replay()
    verifier.config["artifacts"] = ["answer.json", "solve.py"]
    verifier.grader(_NO_IPC_GRADER)
    verifier.submit(41, **{"solve.py": SOLVE + _LEAVES_IPC, "data.txt": "21"})
    extra = _IPC_OPS + ("" if removed else "launcher.remove_ipc = lambda objects: 0\n")

    _, reward, report = verifier.run(extra=extra, ops="IpcOps()", prefix=("unshare", "-r", "-i"))

    if removed:
        assert reward == {"reward": 1}, report
        assert report["replay"]["cleared"] == 3
        return
    assert (reward, report["reason"]) == ({"reward": 0}, "solver_nonreproducible"), report
    assert report["detail"] == (
        "the replay left what the launcher cannot remove in its time: 3 System V IPC object(s) of its user cannot be "
        "removed"
    )
