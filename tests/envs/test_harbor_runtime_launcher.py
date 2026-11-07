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
# takes the other users' permissions, hardening only strips the write bits it can without being root, the replay runs
# as this user with the launcher's limits on files, its sweep kills what is below the launcher, never this user's other
# processes, and what the replay's user left where every user may write is this user's, in the verifier's own
# temporary directories.
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

    def demote(self, max_file_bytes, max_address_space_bytes=None):
        return lambda: launcher._limit_files(max_file_bytes)

    def processes_of(self, uid):
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
# A submission whose code computes something else each time it runs: it draws from an unseeded generator.
UNSEEDED_SOLVE = """
import json, random
with open("answer.json", "w") as stream:
    json.dump({"answer": random.choice([41, 42]), "draw": random.random()}, stream)
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
    assert launcher.REPLAYED_SHA256 not in report  # no replay: the grader judged the artifacts as they were collected
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


def test_a_verdict_nested_too_deeply_to_read_is_the_graders_protocol_fault(verifier):
    verifier.grader(WRITE_VERDICT.format('stream.write("[" * 100000)'))
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


def test_an_artifact_nested_too_deeply_to_read_breaks_the_contract_never_the_launcher(verifier):
    """``json.loads`` gives up on deep nesting with a ``RecursionError``, no ``ValueError``: a solver's artifact that
    nests deeply is its own contract violation, not a failure of the launcher's, which would be the trusted side's."""

    (verifier.workdir / "answer.json").write_text("[" * 100_000)

    code, reward, report = verifier.run()

    assert (code, reward) == (0, {"reward": 0})
    assert report["reason"] == "solver_output_contract" and "nested too deeply" in report["detail"]


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


def test_the_report_holds_the_sha256_of_the_outputs_the_replay_wrote_before_the_grader_ran(verifier):
    """What the grader judged of a replay, its outputs' names and bytes, is in the report: the grader, root, may change
    them once it runs (here it empties the answer it judged), so they are taken before."""

    verifier.replay(outputs=("answer.json", "out/log.txt"))
    verifier.config["artifacts"] = ["answer.json", "solve.py"]
    solve = SOLVE + "import os\nos.mkdir('out')\nopen('out/log.txt', 'w').write('replayed')\n"
    verifier.submit(42, **{"solve.py": solve, "data.txt": "21"})
    verifier.grader(HONEST_GRADER + 'open(os.path.join(os.environ["PLATOON_WORKDIR"], "answer.json"), "w").close()\n')

    _, reward, report = verifier.run()

    assert reward == {"reward": 1}, report
    assert (verifier.workdir / "answer.json").read_text() == ""
    rule = launcher.OUTPUTS_RULE.encode()
    files = {"answer.json": b'{"answer": 42}', "out/log.txt": b"replayed"}
    named = b"".join(
        b"\0" + name.encode() + b"\0" + hashlib.sha256(data).hexdigest().encode() for name, data in files.items()
    )
    assert report[launcher.REPLAYED_SHA256] == hashlib.sha256(rule + named).hexdigest()


@pytest.mark.parametrize(("solve", "same"), [(SOLVE, True), (UNSEEDED_SOLVE, False)], ids=["deterministic", "unseeded"])
def test_replays_record_the_same_sha256_only_where_they_write_the_same_bytes(tmp_path, solve, same):
    """Two grades of one submission, each in a verifier of its own, as a regrade grades it."""

    digests = []
    for grade in ("first", "regrade"):
        verifier = Verifier(tmp_path / grade)
        verifier.replay()
        verifier.config["artifacts"] = ["answer.json", "solve.py"]
        verifier.submit(42, **{"solve.py": solve, "data.txt": "21"})
        digests.append(verifier.run()[2][launcher.REPLAYED_SHA256])

    assert (digests[0] == digests[1]) is same


def test_the_sha256_of_the_replayed_outputs_reads_each_as_the_launcher_reads_every_file(tmp_path):
    """Never through a link, and never waiting on a FIFO: the file is checked on the open descriptor."""

    os.mkfifo(tmp_path / "fifo.json")
    (tmp_path / "answer.json").write_text("{}")
    (tmp_path / "link.json").symlink_to(tmp_path / "answer.json")

    with pytest.raises(ValueError, match="fifo.json is not a regular file"):
        launcher.outputs_sha256(str(tmp_path), ["fifo.json"])
    with pytest.raises(OSError):
        launcher.outputs_sha256(str(tmp_path), ["link.json"])
    assert launcher.outputs_sha256(str(tmp_path), ["answer.json"]) != launcher.outputs_sha256(str(tmp_path), [])


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
    assert seen.pop("HOME").endswith("/replay")
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


def test_a_report_cut_to_the_outcome_keeps_the_sha256_of_the_replayed_outputs():
    """A regrade compares it to tell a submission that computes other outputs each time from a flaky grader."""

    digest = "ab" * 32
    report = {"schema": launcher.SCHEMA, "reward": 1, "detail": "d" * 5000, launcher.REPLAYED_SHA256: digest}

    least = json.loads(launcher.bounded_report(report, limit=100))

    assert least["truncated"] == "all but the outcome" and least[launcher.REPLAYED_SHA256] == digest


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


# ------------------------------------------------------------------------ hardening invariants (I1-I4)

# I1 has its sweep regression above (test_kill_left_never_excludes_a_replay_user_process_it_cannot_signal); the rest
# cover disk, inode and memory exhaustion by a replay (I2), the working directory before the replay (I3) and the
# replay's PATH (I4).


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
        writable=lambda info: inodes.get(info.st_ino) == "public",
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


_NO_ROOM_BACK = """
import errno
_copy_file = launcher._copy_file

def full(source, target, mode):
    if target.startswith({workdir!r}):
        raise OSError(errno.ENOSPC, "No space left on device")
    return _copy_file(source, target, mode)

launcher._copy_file = full
"""


@pytest.mark.parametrize("left", [True, False], ids=["the-replay-left-files", "nothing-left"])
def test_a_full_filesystem_putting_the_outputs_back_is_the_solvers_zero_only_once_the_replay_left_files(verifier, left):
    """I2: the replay's outputs put back where the filesystem is full is the solver's 0 once the replay left files of
    its user where every user may write (the room it lacks is what the replay took); with nothing left, it is the
    trusted side's fault, as a full filesystem is before any replay ran, never a blame the solver can move."""

    shm = verifier.temporary[2]
    solve = SOLVE + (f"\nopen({str(shm / 'left')!r}, 'w').write('x')\n" if left else "")
    verifier.replay()
    verifier.config["artifacts"] = ["answer.json", "solve.py"]
    verifier.submit(42, **{"solve.py": solve, "data.txt": "21"})

    _, reward, report = verifier.run(extra=_NO_ROOM_BACK.format(workdir=str(verifier.workdir)))

    if left:
        assert (reward, report["reason"]) == ({"reward": 0}, "solver_nonreproducible"), report
        assert report["detail"].startswith("the replay's outputs find no room in the working directory")
        assert report["replay"]["cleared"] >= 1
    else:
        assert reward == {"invalid": 1, "failure_kind": "verifier_trusted_launcher"}, report
        assert "No space left on device" in report["detail"]


# A filesystem found full while the launcher holds what the replay wrote: its rename, or its change of owner (a
# filesystem that copies on write may need room for either).
_NO_ROOM_HELD = {
    "rename": (
        """
        import errno
        _rename = os.rename

        def full(source, target):
            if target.endswith("replayed"):
                raise OSError({number}, os.strerror({number}))
            return _rename(source, target)

        os.rename = full
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
@pytest.mark.parametrize(("step", "number"), [("rename", errno.ENOSPC), ("seal", errno.EDQUOT)])
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


@pytest.mark.parametrize("step", ["rename", "seal"])
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


def test_the_launcher_frees_the_replays_copy_before_it_holds_the_outputs(tmp_path, monkeypatch):
    """I2: everything of the replay's copy but its outputs is gone before the hold's first step that may need room
    (the rename), so the room the replay took there never decides whose fault a full filesystem is."""

    stage, held = tmp_path / "replay", tmp_path / "replayed"
    (stage / "out" / "junk").mkdir(parents=True)
    (stage / "out" / "answer.json").write_text("{}")
    (stage / "out" / "junk" / "big").write_text("x" * 4096)
    (stage / "data.txt").write_text("21")
    seen = []
    rename = os.rename

    def recording(source, target):
        seen.append(sorted(str(path.relative_to(source)) for path in Path(source).rglob("*")))
        rename(source, target)

    class Seal:
        def seal(self, path):
            os.chmod(path, 0o700)

    monkeypatch.setattr(os, "rename", recording)

    launcher._hold(str(stage), str(held), ["out/answer.json"], Seal(), None)

    assert seen == [["out", "out/answer.json"]]
    assert sorted(str(path.relative_to(held)) for path in held.rglob("*")) == ["out", "out/answer.json"]
    assert stat.S_IMODE((held / "out" / "answer.json").stat().st_mode) == 0o600


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
