"""The launcher of runtime Harbor tasks (``platoon/envs/harbor/launcher.py``), run as the verifier runs it but without
root: a wrapper replaces what it does as root (the seal's chown, the replay's switch of user and its sweep of that
user's processes), and it runs in a process of its own, over directories that stand for the verifier's ``/tests``,
``/logs/verifier``, working directory, ``/tmp`` and what the agents published.  What it reads of the task, which reward
and report it writes for each way a grader, a replay or a submission behaves, and what it leaves running; how it seals,
limits and sweeps; and that it parses as the oldest Python it runs on, with nothing beyond the standard library.
Where unprivileged user namespaces work, it also runs with its own operations, as root in one."""

from __future__ import annotations

import ast
import codecs
import contextlib
import errno
import hashlib
import json
import os
import re
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
# System V IPC objects and clears no keyrings, since this user's other processes may have some.
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

    def clear_keys(self):
        pass

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

    def variant(
        self,
        name: str = "fifty",
        inputs: dict[str, str] | None = None,
        expected: Any = None,
        tolerance: dict[str, Any] | None = None,
        meta: Any = None,
    ) -> Path:
        """Give the task a replay variant whose data lie in ``tests/gen/variants/<name>``: ``inputs`` (by default a
        ``data.txt`` of 50) laid over the working inputs, and what it ``expected`` (by default an answer of 100);
        the directory of its data."""

        root = self.tests / "gen" / "variants" / name
        for relative, text in (inputs if inputs is not None else {"data.txt": "50"}).items():
            path = root / "inputs" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        (root / "inputs").mkdir(parents=True, exist_ok=True)
        wanted = expected if expected is not None else {"answer.json#/answer": 100}
        (root / "expected.json").write_text(wanted if isinstance(wanted, str) else json.dumps(wanted))
        self.checks()["variants"].append(
            {
                "name": name,
                "inputs": f"tests/gen/variants/{name}/inputs",
                "expected": f"tests/gen/variants/{name}/expected.json",
                "tolerance": tolerance or {"default": 0, "keys": {}},
                "meta": meta,
            }
        )
        return root

    def forbid(self, literals: Any, sources: tuple[str, ...] = ("solve.py",)) -> None:
        """Give the task forbidden literals, ``(value, tolerance)`` pairs (or the text of their file), for the scan of
        ``sources``."""

        text = (
            literals if isinstance(literals, str) else json.dumps([{"value": v, "tolerance": t} for v, t in literals])
        )
        (self.tests / "gen" / "forbidden.json").write_text(text)
        self.checks()["forbidden_literals"] = {
            "path": "tests/gen/forbidden.json",
            "sources": list(sources),
            "scan": launcher.LITERAL_SCAN,
        }

    def checks(self) -> dict[str, Any]:
        return self.config.setdefault("replay_checks", {"variants": [], "forbidden_literals": None})

    def condition(self, name: str = "fifty", inputs: dict[str, str] | None = None) -> Path:
        """Give the task's replay a hidden evaluation condition whose inputs, ``inputs`` (by default a ``data.txt`` of
        50), lie in ``tests/gen/conditions/<name>``; that directory."""

        root = self.tests / "gen" / "conditions" / name
        root.mkdir(parents=True, exist_ok=True)
        for relative, text in (inputs if inputs is not None else {"data.txt": "50"}).items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        self.config["replay"].setdefault("conditions", []).append(
            {"id": name, "inputs": f"tests/gen/conditions/{name}"}
        )
        return root

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


# ---------------------------------------------------------------------------------------------------- replay checks

# A submission that computes its answer without reading its inputs: its replay writes 42 again, on any data.
IGNORING_SOLVE = """
import json
with open("answer.json", "w") as stream:
    json.dump({"answer": 6 * 7}, stream)
"""
# The SHA-256 of the replayed outputs of SOLVE over a data.txt of 21: {"answer": 42}.
SOLVED_SHA256 = hashlib.sha256(
    launcher.OUTPUTS_RULE.encode() + b"\0answer.json\0" + hashlib.sha256(b'{"answer": 42}').hexdigest().encode()
).hexdigest()


def _checked(verifier: Verifier, solve: str = SOLVE) -> None:
    """A task with a replay, whose submission is ``solve`` (and its answer) over a working input data.txt of 21."""

    verifier.replay()
    verifier.config["artifacts"] = ["answer.json", "solve.py"]
    verifier.submit(42, **{"solve.py": solve, "data.txt": "21"})


def _in_variants(code: str) -> str:
    """Python that runs ``code`` only in the replay of a variant, whose data.txt holds 50."""

    return f"\nimport json, os, sys\nif open('data.txt').read().strip() == '50':\n    {code}\n"


def _leaving(verifier: Verifier, solve: str = SOLVE) -> str:
    """``solve``, then lines that leave a directory of the replay's user in each of the verifier's temporary
    directories (its /tmp, which holds the launcher's own directory, and the others), and note in the answer how many
    of those an earlier replay left there are still to be seen."""

    places = [str(path) for path in verifier.temporary[1:]]
    return solve + textwrap.dedent(
        f"""
        import json, os
        places = [os.path.abspath(os.path.join(os.environ["TMPDIR"], "..", "..")), *{places!r}]
        data = open("data.txt").read().strip()
        seen = sum(os.path.exists(os.path.join(place, "left-21")) for place in places)
        for place in places:
            os.makedirs(os.path.join(place, "left-" + data, "inner"))
            open(os.path.join(place, "left-" + data, "inner", "junk"), "w").write("x")
        answer = json.load(open("answer.json"))
        json.dump({{**answer, "seen": seen}}, open("answer.json", "w"))
        """
    )


_ANSWER_GRADER = HONEST_GRADER.replace('json.load(stream) == {"answer": 42}', 'json.load(stream)["answer"] == 42')


def test_what_a_replays_user_leaves_where_every_user_may_write_is_removed_after_each_replay(verifier):
    """What the replay's user left in /tmp, /var/tmp and /dev/shm is gone before the next replay starts, and before the
    grader runs: no variant's replay sees what the replay left there, and nothing a replay left takes room."""

    _checked(verifier, _leaving(verifier))
    verifier.variant(expected={"answer.json#/answer": 100, "answer.json#/seen": 0})
    verifier.grader(_ANSWER_GRADER)

    _, reward, report = verifier.run()

    assert reward == {"reward": 1}, report
    assert report["replay"]["cleared"] == 3 and report["replay_checks"]["variants"][0]["cleared"] == 3
    assert [sorted(path.name for path in place.iterdir()) for place in verifier.temporary[1:]] == [[], []]
    assert [path.name for path in verifier.scratch.iterdir()] == [report_directory(verifier)]


def report_directory(verifier: Verifier) -> str:
    """The name of the launcher's own directory in the verifier's /tmp, which it leaves there."""

    (directory,) = (path.name for path in verifier.scratch.iterdir() if path.name.startswith("platoon-launcher-"))
    return directory


def test_with_a_time_of_their_own_each_replay_takes_at_most_that_and_the_grader_the_rest(verifier):
    """``replay_timeout_s`` gives each replay its own time, here 2 s for the replay and each of two variants, rather
    than a share of half of the time left; the grader gets what they leave, most of 108 s."""

    _checked(verifier)
    verifier.variant()
    verifier.variant("seven", inputs={"data.txt": "7"}, expected={"answer.json#/answer": 14})
    verifier.config["replay_timeout_s"] = 2

    _, reward, report = verifier.run()

    assert reward == {"reward": 1}, report
    times = [report["replay"]["timeout_s"]] + [entry["timeout_s"] for entry in report["replay_checks"]["variants"]]
    assert all(1.5 < each <= 2.0 for each in times), times
    assert report["grader"]["timeout_s"] > 100.0


def test_a_replay_that_outlives_its_own_time_scores_zero(verifier):
    verifier.replay()
    verifier.submit(42, **{"solve.py": "import time\ntime.sleep(30)\n", "data.txt": "21"})
    verifier.config["replay_timeout_s"] = 1

    _, reward, report = verifier.run()

    assert (reward, report["reason"]) == ({"reward": 0}, "solver_nonreproducible")
    assert report["detail"].startswith("the replay outlived its 0.9") and 0.9 < report["replay"]["timeout_s"] <= 1.0


def test_replays_with_a_time_of_their_own_end_by_that_many_times_it_after_the_launcher_started(monkeypatch):
    """The launcher's checks before the replays take from the replays' time, never the grader's: replays of 3 s each,
    a replay and a variant, end 6 s after the launcher started, which was 5 s ago, so the replay has 1 s left."""

    monkeypatch.setattr(launcher.time, "monotonic", lambda: 1000.0)
    replays = launcher.Replays.of({"replay_timeout_s": 3}, started=995.0, deadline=1100.0, variants=1)

    assert (replays.deadline(), replays.deadline()) == (1001.0, 1001.0)
    shared = launcher.Replays.of({}, started=995.0, deadline=1100.0, variants=1)  # half of the 100 s left, in turn
    assert shared.deadline() == 1025.0 and shared.deadline() == 1050.0


@pytest.mark.parametrize(
    ("timeout", "problem"),
    [
        (0, "0 is no finite number of seconds above 0"),
        (-1, "-1 is no finite number of seconds above 0"),
        (True, "True is no finite number"),
        ("5", "'5' is no finite number"),
        (float("nan"), "nan is no finite number"),
        (54, "2 replays of 54 s leave the grader none of the 108 s the launcher's steps have"),
    ],
)
def test_a_replay_time_the_launcher_cannot_give_is_settings_it_cannot_use(verifier, timeout, problem):
    _checked(verifier)
    verifier.variant()
    verifier.config["replay_timeout_s"] = timeout

    _, reward, report = verifier.run()

    assert reward == {"invalid": 1, "failure_kind": "verifier_trusted_launcher"}
    assert f"hold no usable replay_timeout_s ({problem}" in report["detail"], report["detail"]


def test_a_replay_time_without_a_replay_is_settings_the_launcher_cannot_use(verifier):
    verifier.submit(42)
    verifier.config["replay_timeout_s"] = 5

    _, reward, report = verifier.run()

    assert reward == {"invalid": 1, "failure_kind": "verifier_trusted_launcher"}
    assert "hold no usable replay_timeout_s (the task has no replay)" in report["detail"]


def _tmpfs_works() -> bool:
    if shutil.which("unshare") is None:
        return False
    with tempfile.TemporaryDirectory() as directory:
        mount = 'mount -t tmpfs -o size=1m tmpfs "$0"'
        return (
            subprocess.run(["unshare", "-rm", "sh", "-c", mount, directory], capture_output=True, timeout=60).returncode
            == 0
        )


_TMPFS = pytest.mark.skipif(not _tmpfs_works(), reason="unprivileged mount namespaces with tmpfs are unavailable")


def _on_a_small_tmpfs(verifier: Verifier, size: str, workdir: bool = False) -> tuple[str, ...]:
    """The command prefix that runs the launcher as root of a user and mount namespace whose verifier's /tmp is a
    tmpfs of ``size``; with ``workdir``, its working directory too, where the replays run: a tmpfs of ``size`` holds
    both, each bound at its own path, the working directory with what it held."""

    if not workdir:
        return (
            "unshare",
            "-rm",
            "sh",
            "-c",
            f'mount -t tmpfs -o size={size} tmpfs "$0" && exec "$@"',
            str(verifier.scratch),
        )
    small = verifier.root / "small"
    small.mkdir()
    script = (
        f'mount -t tmpfs -o size={size} tmpfs "$0" && mkdir "$0/app" "$0/tmp" && cp -a "$1/." "$0/app/" && '
        'mount --bind "$0/app" "$1" && mount --bind "$0/tmp" "$2" && cd "$1" && shift 2 && exec "$@"'
    )
    return ("unshare", "-rm", "sh", "-c", script, str(small), str(verifier.workdir), str(verifier.scratch))


# A replay that fills its filesystem with files of 1 MiB until none fits, at most 64 of them: in the working directory,
# where it runs, or (``where`` its home's parent's parent) in the verifier's /tmp beside the launcher's directory.
_FILLING = """
import os
where = {where}
n = 0
try:
    while n < 64:
        with open(os.path.join(where, "junk-%d-%d" % (os.getpid(), n)), "wb") as stream:
            stream.write(b"x" * (1 << 20))
        n += 1
except OSError:
    pass
"""


@_TMPFS
@pytest.mark.parametrize("variants", [False, True], ids=["alone", "with-a-variant"])
def test_a_replay_that_fills_its_directorys_filesystem_is_graded_and_its_variants_replay(verifier, variants):
    """On a tmpfs of 16 MiB that holds the working directory and the verifier's /tmp, a replay that writes its outputs,
    then fills the working directory, as a variant's replay does too: what each left is removed, and the working
    directory put back, before any variant replays, and holding what the replay wrote takes no room, so the honest
    submission scores 1."""

    _checked(verifier, SOLVE + _FILLING.format(where='"."'))
    if variants:
        verifier.variant()

    _, reward, report = verifier.run(prefix=_on_a_small_tmpfs(verifier, "16m", workdir=True))

    assert reward == {"reward": 1}, report
    assert [entry["passed"] for entry in report.get("replay_checks", {}).get("variants", [])] == [True] * variants


@_TMPFS
@pytest.mark.parametrize("variants", [False, True], ids=["alone", "with-a-variant"])
def test_a_replay_that_fills_tmp_beside_its_directory_keeps_the_solvers_zero(verifier, variants):
    """On a /tmp of 3 MiB, a submission that ignores its inputs and whose replay fills /tmp beside its own directory:
    what it left there is removed before its variant replays, which then fails, the solver's 0, never the trusted
    side's fault; without the variant, it is graded."""

    leaving = _FILLING.format(where='os.path.join(os.environ["TMPDIR"], "..", "..")')
    _checked(verifier, IGNORING_SOLVE + leaving)
    if variants:
        verifier.variant()

    _, reward, report = verifier.run(prefix=_on_a_small_tmpfs(verifier, "3m"))

    assert report["failure_kind"] is None, report
    assert report["replay"]["cleared"] >= 1
    if variants:
        assert (reward, report["reason"]) == ({"reward": 0}, "solver_variant_failed"), report
    else:
        assert (reward, report["detail"]) == ({"reward": 1}, "the verdict passed"), report


# Fills the filesystem as a variant's replay is set up or the working directory put back after it (or with
# ``workdir``, as the replay's outputs go back in the working directory), as an earlier replay that filled it from
# somewhere the launcher does not clear would have: the variant's inputs laid over the working directory
# (``variant-inputs``), the file of its output (``variant-log``), its lending (``variant-lend``: a change of owner that
# needs room on a filesystem that copies on write, or a file of several names copied into one of its own), or putting
# the working directory back after it (``variant-restore``, on each try).
_NO_ROOM = """
import builtins, errno
copy_file, clone = launcher._copy_file, launcher._clone_tree
lends, clones = [], []

def no_room(*args, **kwargs):
    raise OSError(errno.ENOSPC, "No space left on device")

def full_workdir(source, target, mode, times=None):
    if target.startswith({workdir!r}) and os.sep + "replayed" + os.sep in source:
        no_room()
    return copy_file(source, target, mode, times)

def no_log(path, *args, **kwargs):
    if os.path.basename(str(path)).startswith("variant-") and str(path).endswith(".log"):
        no_room()
    return builtins.open(path, *args, **kwargs)

def full_after_the_variant(source, target, *args, **kwargs):
    if target == {workdir!r}:
        clones.append(target)
        if len(clones) > 1:  # each put back after the variant's replay, the replay's own being the first
            no_room()
    return clone(source, target, *args, **kwargs)

class FullOnLend(WithoutRoot):
    def lend(self, workdir):
        lends.append(workdir)
        if len(lends) > 1:  # the variant's, the replay's own being the first
            no_room()
        WithoutRoot.lend(self, workdir)

{assignment}
"""
_FULL = {
    "variant-inputs": ("launcher._lay_over = no_room", "WithoutRoot()"),
    "variant-log": ("launcher.open = no_log", "WithoutRoot()"),
    "variant-lend": ("", "FullOnLend()"),
    "variant-restore": ("launcher._clone_tree = full_after_the_variant", "WithoutRoot()"),
    "workdir": ("launcher._copy_file = full_workdir", "WithoutRoot()"),
}
_FULL_DETAIL = {
    "variant-inputs": "the replay of variant fifty finds no room in the working directory, once the replays left",
    "variant-log": "the replay of variant fifty finds no room in the working directory, once the replays left",
    "variant-lend": "the replay of variant fifty finds no room in the working directory, once the replays left",
    "variant-restore": "the working directory finds no room as the launcher puts it back, once the replays left",
    "workdir": "the replay's outputs find no room in the working directory, once the replays left",
}


@pytest.mark.parametrize("left", [True, False], ids=["the-replays-left-files", "nothing-left"])
@pytest.mark.parametrize("where", list(_FULL))
def test_a_full_filesystem_after_the_replays_left_files_is_the_solvers_zero(verifier, left, where):
    """I2 for a variant: where a replay's user left files where every user may write, a variant's inputs laid over the
    working directory, the file of its output, its lending, or the working directory put back after it, or the
    replay's outputs put back in the working directory, that finds its filesystem full is the solver's 0, never the
    grader owner's protocol fault; with nothing left, the trusted side's fault, as a full filesystem is before any
    replay ran (putting the directory back, once the filesystem wrote out what it holds and the launcher tried once
    more)."""

    shm = verifier.temporary[2]
    _checked(verifier, SOLVE + (f"\nopen({str(shm / 'left')!r}, 'w').write('x')\n" if left else ""))
    verifier.variant()
    assignment, ops = _FULL[where]
    extra = _NO_ROOM.format(workdir=str(verifier.workdir), assignment=assignment)

    _, reward, report = verifier.run(extra=extra, ops=ops)

    if not left:
        assert reward == {"invalid": 1, "failure_kind": "verifier_trusted_launcher"}, report
        assert "No space left on device" in report["detail"]
    else:
        reason = "solver_variant_failed" if where.startswith("variant") else "solver_nonreproducible"
        assert (reward, report["reason"]) == ({"reward": 0}, reason), report
        assert report["detail"].startswith(_FULL_DETAIL[where]), report["detail"]


def test_a_submission_that_follows_its_inputs_passes_every_variant_and_its_replay_is_graded(verifier):
    """Each variant replays the submission on its own inputs (a data.txt of 50, then of 7), which gives what it
    expects; the grader then grades what the replay wrote from the working inputs, whose SHA-256 the report holds."""

    _checked(verifier)
    verifier.variant("fifty", meta={"covers": [[0, 1]]})
    verifier.variant("seven", inputs={"data.txt": "7"}, expected={"answer.json#/answer": 14})

    _, reward, report = verifier.run()

    assert reward == {"reward": 1}, report
    ran = report["replay_checks"]["variants"]
    assert [(entry["name"], entry["passed"], entry["exit_code"]) for entry in ran] == [
        ("fifty", True, 0),
        ("seven", True, 0),
    ]
    assert not {"meta", "output_tail", "missed"} & set(ran[0])  # a variant that passed keeps a short record
    assert json.loads((verifier.workdir / "answer.json").read_text()) == {"answer": 42}  # the replay's, no variant's
    assert report[launcher.REPLAYED_SHA256] == SOLVED_SHA256


def test_a_submission_that_ignores_its_inputs_fails_a_variant_as_the_solvers_fault(verifier):
    _checked(verifier, IGNORING_SOLVE)
    verifier.variant("fifty", meta={"covers": [[0, 1]]})

    _, reward, report = verifier.run()

    assert reward == {"reward": 0}
    assert (report["reason"], report["detail"]) == (
        "solver_variant_failed",
        "the replay of variant fifty gave answer.json#/answer outside the tolerance of what the variant expects",
    )
    (entry,) = report["replay_checks"]["variants"]
    assert (entry["passed"], entry["missed"], entry["meta"]) == (False, ["answer.json#/answer"], {"covers": [[0, 1]]})
    assert report["replay"]["exit_code"] == 0 and "output_tail" in entry
    assert "grader" not in report and launcher.REPLAYED_SHA256 not in report


def test_each_variant_replays_from_the_clean_inputs_with_its_own_laid_over_them(verifier):
    """A variant's replay starts from the working inputs as the replay found them: without the replay's outputs or
    what an earlier replay left there, with the variant's inputs in place of the files at their paths (data.txt) and
    beside them (notes/extra.txt), which neither the replay nor another variant's replay sees."""

    probe = """
        import json, os, sys
        if os.path.exists("answer.json") or os.path.exists("left.txt"):
            sys.exit(7)
        open("left.txt", "w").write("left by a replay")
        with open("data.txt") as stream:
            value = 2 * int(stream.read())
        with open("answer.json", "w") as stream:
            json.dump({"answer": value, "extra": int(os.path.exists("notes/extra.txt"))}, stream)
    """
    _checked(verifier, probe)
    verifier.variant(
        inputs={"data.txt": "50", "notes/extra.txt": "seen"},
        expected={"answer.json#/answer": 100, "answer.json#/extra": 1},
    )
    verifier.variant("seven", inputs={"data.txt": "7"}, expected={"answer.json#/answer": 14, "answer.json#/extra": 0})
    verifier.grader(HONEST_GRADER.replace('json.load(stream) == {"answer": 42}', 'json.load(stream)["answer"] == 42'))

    _, reward, report = verifier.run()

    assert reward == {"reward": 1}, report
    assert [entry["passed"] for entry in report["replay_checks"]["variants"]] == [True, True]
    assert json.loads((verifier.workdir / "answer.json").read_text()) == {"answer": 42, "extra": 0}


@pytest.mark.parametrize(
    ("code", "detail"),
    [
        ("sys.exit(3)", "the replay of variant fifty exited 3"),
        ("os.remove('answer.json')", "after the replay of variant fifty, answer.json is missing"),
        ("open('answer.json', 'w').write('{nope')", "after the replay of variant fifty, answer.json is not JSON"),
        ("json.dump({'answer': True}, open('answer.json', 'w'))", "gave answer.json#/answer outside the tolerance"),
        ("json.dump({'answers': 100}, open('answer.json', 'w'))", "gave answer.json#/answer outside the tolerance"),
    ],
    ids=["exits", "no-output", "not-json", "a-boolean", "no-such-key"],
)
def test_a_variants_replay_that_does_not_give_what_it_expects_scores_zero_as_the_solvers_fault(verifier, code, detail):
    _checked(verifier, SOLVE + _in_variants(code))
    verifier.variant()

    _, reward, report = verifier.run()

    assert reward == {"reward": 0}
    assert report["reason"] == "solver_variant_failed" and detail in report["detail"], report["detail"]
    assert report["replay"]["exit_code"] == 0 and "grader" not in report


def test_a_variants_replay_that_outlives_its_time_scores_zero_and_takes_what_the_replay_left_of_theirs(verifier):
    """12 s leave the launcher's steps 7 s, and the replays half of it: the replay may take half of that, and the
    variant has whatever is left."""

    _checked(verifier, SOLVE + _in_variants("__import__('time').sleep(60)"))
    verifier.variant()
    verifier.config["timeout_s"] = 12.0

    report = verifier.run()[2]

    assert (
        report["reason"] == "solver_variant_failed" and "the replay of variant fifty outlived its" in report["detail"]
    )
    (entry,) = report["replay_checks"]["variants"]
    assert entry["timed_out"] and report["replay"]["timeout_s"] <= 1.8 < entry["timeout_s"] <= 3.5


def test_the_replays_share_the_replays_time_each_an_equal_part_of_what_is_left(verifier):
    """The replay and two variants share half of the time the launcher's steps have (108 s of 120): the replay a third
    of it, the first variant half of what is left then, the second the rest."""

    _checked(verifier)
    verifier.variant()
    verifier.variant("seven", inputs={"data.txt": "7"}, expected={"answer.json#/answer": 14})

    report = verifier.run()[2]

    first, second = (entry["timeout_s"] for entry in report["replay_checks"]["variants"])
    assert 17.0 < report["replay"]["timeout_s"] <= 18.0 and 26.0 < first <= 27.0 and 52.0 < second <= 54.0, report


def test_a_variant_whose_time_leaves_none_to_put_the_directory_back_scores_zero_as_the_solvers_fault(verifier):
    """Each replay ends as long before its time is over as putting the working directory back after it takes: a
    variant's replay whose time leaves none once that is kept never starts, the solver's 0."""

    _checked(verifier)
    verifier.variant()
    extra = """
        run = launcher._run_replay

        def late(config, paths, ops, place, log, deadline, replays, variant=None):
            if variant is not None:
                place.put_back = 1e6
            return run(config, paths, ops, place, log, deadline, replays, variant)

        launcher._run_replay = late
    """

    _, reward, report = verifier.run(extra=extra)

    assert (reward, report["reason"], report["detail"]) == (
        {"reward": 0},
        "solver_variant_failed",
        "putting the working directory back after the replay of variant fifty leaves it no time",
    )
    (entry,) = report["replay_checks"]["variants"]
    assert not entry["passed"] and "exit_code" not in entry and "grader" not in report


def test_putting_the_directory_back_past_a_variants_time_scores_zero_as_the_solvers_fault(verifier):
    """Putting the working directory back after a variant's replay that outlives the variant's time, and the window
    past it, is the solver's 0 (as copying it back past that time would be: :func:`launcher._in_time`), however well the
    variant's replay did."""

    _checked(verifier)
    verifier.variant()
    extra = f"""
        import time
        clone, clones = launcher._clone_tree, []

        def slow(source, target, deadline=None, limit=None, device=None):
            if target == {str(verifier.workdir)!r}:
                clones.append(target)
                if len(clones) == 2:  # the variant's, the replay's own being the first
                    launcher._in_time(time.monotonic() - 1)
            return clone(source, target, deadline, limit, device)

        launcher._clone_tree = slow
    """

    _, reward, report = verifier.run(extra=extra)

    assert (reward, report["reason"], report["detail"]) == (
        {"reward": 0},
        "solver_variant_failed",
        "putting the working directory back outlived the replay of variant fifty's time",
    )
    (entry,) = report["replay_checks"]["variants"]
    assert (entry["passed"], entry["exit_code"]) == (False, 0) and "grader" not in report


def test_a_variants_replay_runs_in_place_and_reads_its_inputs_by_their_absolute_paths(verifier):
    """A variant replays in the working directory itself, at its own path, with its inputs laid over the working
    inputs there: a submission that names its files by their absolute paths reads the variant's data.txt and writes
    its outputs where the launcher reads them, as the replay does."""

    workdir = str(verifier.workdir)
    data, answer = os.path.join(workdir, "data.txt"), os.path.join(workdir, "answer.json")
    absolute = f"""
        import json, os
        with open({data!r}) as stream:
            value = 2 * int(stream.read())
        with open({answer!r}, "w") as stream:
            json.dump({{"answer": value, "here": int(os.getcwd() == os.path.realpath({workdir!r}))}}, stream)
    """
    _checked(verifier, textwrap.dedent(absolute))
    verifier.variant(expected={"answer.json#/answer": 100, "answer.json#/here": 1})
    verifier.variant("seven", inputs={"data.txt": "7"}, expected={"answer.json#/answer": 14, "answer.json#/here": 1})
    verifier.grader(_ANSWER_GRADER)

    _, reward, report = verifier.run()

    assert reward == {"reward": 1}, report
    assert [entry["passed"] for entry in report["replay_checks"]["variants"]] == [True, True]
    assert json.loads((verifier.workdir / "answer.json").read_text()) == {"answer": 42, "here": 1}


def test_no_later_replay_nor_the_grader_finds_what_a_replay_changed_in_the_working_directory(verifier):
    """I5 across the replays of a grade: each replay changes an input, its own code and a file of the image's, removes
    one, and leaves files and a directory; each variant's replay finds the working directory as its pristine copy holds
    it with the variant's inputs laid over it, and the grader finds it as the pristine copy holds it with the replayed
    output in place, whatever the replays did there."""

    tampering = """
        import json, os, shutil
        data = open("data.txt").read().strip()
        fresh = int(
            open("solve.py").read().startswith("import json, os, shutil")
            and open("notes/keep.txt").read() == "kept"
            and os.path.exists("gone.txt")
            and not any(os.path.lexists(name) for name in ("extra.txt", "left"))
        )
        with open("answer.json", "w") as stream:
            json.dump({"answer": 2 * int(data), "fresh": fresh}, stream)
        open("data.txt", "w").write("1")
        open("solve.py", "w").write("raise SystemExit(9)")
        open("notes/keep.txt", "a").write(" and changed")
        os.remove("gone.txt")
        open("extra.txt", "w").write("left by " + data)
        os.makedirs("left/inner")
    """
    _checked(verifier, textwrap.dedent(tampering).lstrip())
    (verifier.workdir / "notes").mkdir()
    (verifier.workdir / "notes" / "keep.txt").write_text("kept")
    (verifier.workdir / "gone.txt").write_text("here")
    verifier.variant(expected={"answer.json#/answer": 100, "answer.json#/fresh": 1})
    verifier.variant("seven", inputs={"data.txt": "7"}, expected={"answer.json#/answer": 14, "answer.json#/fresh": 1})
    verifier.grader(
        """
        import json, os
        workdir = os.environ["PLATOON_WORKDIR"]
        read = lambda name: open(os.path.join(workdir, name)).read()
        passed = (
            json.loads(read("answer.json")) == {"answer": 42, "fresh": 1}
            and read("data.txt") == "21"
            and read("solve.py").startswith("import json, os, shutil")
            and read("notes/keep.txt") == "kept"
            and read("gone.txt") == "here"
            and sorted(os.listdir(workdir)) == ["answer.json", "data.txt", "gone.txt", "notes", "solve.py"]
        )
        with open(os.environ["PLATOON_VERDICT"], "w") as stream:
            json.dump({"passed": passed}, stream)
        """
    )

    _, reward, report = verifier.run()

    assert reward == {"reward": 1}, report
    assert [entry["passed"] for entry in report["replay_checks"]["variants"]] == [True, True]


def test_what_a_variants_replay_leaves_running_is_swept(verifier):
    marker = verifier.root / "daemon.pid"
    daemonizing = f"""
        import os, time
        if open("data.txt").read().strip() == "50":
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
    _checked(verifier, SOLVE + textwrap.dedent(daemonizing))
    verifier.variant()

    try:
        _, reward, report = verifier.run()

        assert reward == {"reward": 1}, report
        (entry,) = report["replay_checks"]["variants"]
        assert entry["swept"] == 1 and report["replay"]["swept"] == 0
        with pytest.raises(ProcessLookupError):
            os.kill(int(marker.read_text()), 0)
    finally:  # a daemon the launcher failed to kill must not outlive the test
        with contextlib.suppress(OSError, ValueError):
            os.kill(int(marker.read_text()), signal.SIGKILL)


def test_no_variant_replays_once_the_replay_failed(verifier):
    """The scan ran before the replay, and found nothing; no variant replayed after the replay failed."""

    _checked(verifier, "import sys\nsys.exit(4)\n")
    verifier.variant()
    verifier.forbid([(7.5, 0)])

    _, reward, report = verifier.run()

    assert (reward, report["reason"], report["detail"]) == (
        {"reward": 0},
        "solver_nonreproducible",
        "the replay exited 4",
    )
    assert list(report["replay_checks"]) == ["forbidden_literals"]
    assert {**report["replay_checks"]["forbidden_literals"], "python": None} == {
        "scan": launcher.LITERAL_SCAN,
        "python": None,
        "sources": ["solve.py"],
        "literals": 1,
        "hits": [],
    }


def test_each_grade_replays_the_variants_again_as_a_regrade_does(tmp_path):
    """Two grades of one submission, each in a verifier of its own: the second replays the variants again, and catches
    a submission that follows its inputs only the first time."""

    once = tmp_path / "followed-once"
    solve = f"""
import json, os
with open("data.txt") as stream:
    value = 2 * int(stream.read())
if value == 100:
    if os.path.exists({str(once)!r}):
        value = 42
    open({str(once)!r}, "w").close()
with open("answer.json", "w") as stream:
    json.dump({{"answer": value}}, stream)
"""
    grades = []
    for grade in ("first", "regrade"):
        verifier = Verifier(tmp_path / grade)
        _checked(verifier, solve)
        verifier.variant()
        report = verifier.run()[2]
        grades.append(
            (report["reward"], report["reason"], [entry["passed"] for entry in report["replay_checks"]["variants"]])
        )

    assert grades == [(1, None, [True]), (0, "solver_variant_failed", [False])]


def test_what_the_replay_wrote_waits_as_roots_alone_while_the_variants_replay(verifier):
    """What the replay wrote waits in the launcher's own directory, beside the replays' home, alone (what else the
    replay left in the working directory is gone), the directory 700 and its files 600, which the seal makes root's.
    Run as this user, a variant's replay can look; one of the replay's user never can."""

    code = (
        "held = os.path.join(os.path.dirname(os.environ['HOME']), 'replayed')\n    "
        "json.dump({'answer': 100, 'directory': os.stat(held).st_mode & 0o777, "
        "'file': os.stat(os.path.join(held, 'answer.json')).st_mode & 0o777, "
        "'entries': len(os.listdir(held))}, open('answer.json', 'w'))"
    )
    _checked(verifier, SOLVE + _in_variants(code))
    expected = {"answer": 100, "directory": 0o700, "file": 0o600, "entries": 1}
    verifier.variant(expected={f"answer.json#/{key}": value for key, value in expected.items()})

    _, reward, report = verifier.run()

    assert reward == {"reward": 1}, report
    assert json.loads((verifier.workdir / "answer.json").read_text()) == {"answer": 42}


def test_an_output_below_directories_waits_with_them_and_nothing_else_of_the_replays(verifier):
    """An output below directories keeps them while the variants replay, each 700, and nothing else the replay left
    there; the grader then grades it back in its place."""

    nested = """
import json, os
with open("data.txt") as stream:
    value = 2 * int(stream.read())
os.makedirs("out/deep", exist_ok=True)
open("out/notes.txt", "w").write("left beside the output")
with open("out/deep/answer.json", "w") as stream:
    json.dump({"answer": value}, stream)
"""
    held = "held = os.path.join(os.path.dirname(os.environ['HOME']), 'replayed', 'out')\n    "
    code = held + (
        "json.dump({'answer': 100, 'entries': int(os.listdir(held) == ['deep']), "
        "'mode': os.stat(held).st_mode & 0o777}, open('out/deep/answer.json', 'w'))"
    )
    verifier.replay(("out/deep/answer.json",))
    verifier.config["artifacts"] = ["out", "solve.py"]
    (verifier.workdir / "out" / "deep").mkdir(parents=True)
    (verifier.workdir / "out" / "deep" / "answer.json").write_text(json.dumps({"answer": 42}))
    verifier.submit(None, **{"solve.py": nested + _in_variants(code), "data.txt": "21"})
    verifier.grader(HONEST_GRADER.replace('"answer.json"', '"out", "deep", "answer.json"'))
    expected = {"answer": 100, "entries": 1, "mode": 0o700}
    verifier.variant(expected={f"out/deep/answer.json#/{key}": value for key, value in expected.items()})

    _, reward, report = verifier.run()

    assert reward == {"reward": 1}, report
    assert sorted(path.name for path in (verifier.workdir / "out").iterdir()) == ["deep"]


def test_a_variants_replay_that_leaves_processes_that_cannot_be_killed_scores_zero(verifier):
    _checked(verifier)
    verifier.variant()
    extra = """
        kill, sweeps = launcher.kill_left, []

        def stubborn(also=None, window_s=launcher.KILL_WINDOW_S):
            if also is None:
                return kill(also, window_s)
            sweeps.append(also)
            return (2, 1) if len(sweeps) == 2 else (0, 0)  # the variant's replay's sweep, after the replay's

        launcher.kill_left = stubborn
    """

    _, reward, report = verifier.run(extra=extra)

    assert (reward, report["reason"], report["detail"]) == (
        {"reward": 0},
        "solver_variant_failed",
        "processes the replay of variant fifty left could not be killed (1)",
    )
    (entry,) = report["replay_checks"]["variants"]
    assert (entry["passed"], entry["survivors"], entry["swept"]) == (False, 1, 2) and "grader" not in report


def test_a_laid_over_input_keeps_its_executable_bit(verifier):
    code = (
        "json.dump({'answer': 100, 'tool': int(os.access('tool.sh', os.X_OK)), "
        "'data': int(os.access('data.txt', os.X_OK))}, open('answer.json', 'w'))"
    )
    _checked(verifier, SOLVE + _in_variants(code))
    expected = {"answer.json#/answer": 100, "answer.json#/tool": 1, "answer.json#/data": 0}
    root = verifier.variant(inputs={"data.txt": "50", "tool.sh": "#!/bin/sh\n"}, expected=expected)
    (root / "inputs" / "tool.sh").chmod(0o755)

    _, reward, report = verifier.run()

    assert reward == {"reward": 1}, report


def test_a_source_whose_literals_take_past_the_deadline_breaks_the_output_contract(verifier):
    """The scan looks at the time before each source and every 1,024 literals: here the second look finds it past."""

    _checked(verifier, SOLVE + "x = [" + ", ".join(str(n) for n in range(2000)) + "]\n")
    verifier.forbid([(4.2e9, 0)])
    extra = """
        looks = []

        def past(deadline):
            looks.append(deadline)
            return len(looks) > 1

        launcher._past = past
    """

    _, reward, report = verifier.run(extra=extra)

    assert (reward, report["reason"], report["detail"]) == (
        {"reward": 0},
        "solver_output_contract",
        "the sources take longer to scan than the verifier's time limit allows",
    )
    assert "replay" not in report


def test_a_source_that_is_a_directory_breaks_the_output_contract(verifier):
    verifier.replay()
    verifier.config["artifacts"] = ["answer.json", "solve.py", "lib.py"]
    (verifier.workdir / "lib.py").mkdir()
    verifier.submit(42, **{"solve.py": SOLVE, "data.txt": "21"})
    verifier.forbid([(42, 0.5)], sources=("lib.py",))

    _, reward, report = verifier.run()

    assert (reward, report["reason"], report["detail"]) == (
        {"reward": 0},
        "solver_output_contract",
        "lib.py, a source the task scans for forbidden literals, cannot be read: lib.py is not a regular file",
    )


def test_the_windows_find_a_match_with_one_exact_test_however_they_overlap(monkeypatch):
    """A wide window over 9,999 narrow ones: a number in the wide one alone, between two narrow ones, and one in both
    each take one exact test, and one outside every window none."""

    forbidden = [(0.0, 1e300)] + [(1000.0 + n, 0.4) for n in range(launcher.MAX_FORBIDDEN - 1)]
    windows, narrow = launcher.Windows(forbidden), launcher.Windows(forbidden[1:])
    close, tests = launcher._close, []
    monkeypatch.setattr(launcher, "_close", lambda *args: tests.append(args) or close(*args))

    assert (windows.first(10998.5), windows.first(10998.3), windows.first(-1e301)) == (0, 0, None)
    assert len(tests) == 2
    assert (narrow.first(10998.5), narrow.first(10998.3), narrow.first(999.6), narrow.first(999.5)) == (
        None,
        9998,
        0,
        None,
    )


def test_an_overlay_conflicts_with_a_path_of_the_submissions_at_it_below_it_or_above_it():
    protected = ["answer.json", "src", "pkg/mod.py"]

    assert launcher.overlay_conflict("src/extra.py", False, protected) == (
        "src/extra.py would change src, which the submission holds or the replay writes"
    )
    assert launcher.overlay_conflict("pkg", False, protected).startswith("pkg would change pkg/mod.py")
    assert launcher.overlay_conflict("pkg", True, protected) is None  # the directory around it, which it holds too
    assert launcher.overlay_conflict("data/answer.json", False, protected) is None


def _inputs_a_file(verifier: Verifier) -> None:
    root = verifier.variant()
    shutil.rmtree(root / "inputs")
    (root / "inputs").write_text("not a directory")


@pytest.mark.parametrize(("owner", "prefix"), [("generated", "verifier_generated"), ("trusted", "verifier_trusted")])
@pytest.mark.parametrize(
    ("setup", "problem"),
    [
        (
            lambda v: v.variant(expected="{nope"),
            "fifty's expected outputs, tests/gen/variants/fifty/expected.json, cannot",
        ),
        (lambda v: v.variant(expected=[100]), "are no object of 1 to 10000 output keys"),
        (lambda v: v.variant(expected={"result.json": 100}), 'hold the key "result.json", which names no output'),
        (lambda v: v.variant(expected={"answer.json#answer": 100}), "which names no output of the replay"),
        (lambda v: v.variant(expected={"answer.json#/answer": "100"}), "which is no finite number or list of them"),
        (
            lambda v: v.variant(tolerance={"default": 0, "keys": {"answer.json#/other": 1}}),
            'lack "answer.json#/other", which the variant\'s tolerance names',
        ),
        (
            lambda v: shutil.rmtree(v.variant() / "inputs"),
            "variant fifty's inputs, tests/gen/variants/fifty/inputs, cannot",
        ),
        (_inputs_a_file, "are a file, not a directory"),
        (lambda v: v.forbid("{}"), "the forbidden literals, tests/gen/forbidden.json, are no list of at most"),
        (lambda v: v.forbid("[1, 2]"), "hold, as literal 0, no object of a finite value and a tolerance"),
        (lambda v: v.forbid([(42, -1)]), "hold, as literal 0, no object"),
        (lambda v: v.forbid('[{"value": NaN, "tolerance": 1}]'), "hold, as literal 0, no object"),
        (lambda v: v.forbid('[{"value": 1, "tolerance": 1, "cell": 3}]'), "hold, as literal 0, no object"),
        (lambda v: v.variant(expected={}), "are no object of 1 to 10000 output keys"),
        (
            lambda v: v.variant(expected={f"answer.json#/{n}": 1 for n in range(launcher.MAX_EXPECTED_KEYS + 1)}),
            "are no object of 1 to 10000 output keys",
        ),
        (
            lambda v: v.variant(expected=" " * launcher.MAX_CHECK_BYTES + "{}"),
            f"cannot be read as JSON: expected.json holds more than {launcher.MAX_CHECK_BYTES} bytes",
        ),
        (lambda v: v.forbid([(n, 0) for n in range(launcher.MAX_FORBIDDEN + 1)]), "are no list of at most 10000"),
        (
            lambda v: (v.replay(("answer.json", "notes.txt")), v.variant(expected={"notes.txt": 1})),
            'hold the key "notes.txt", whose output notes.txt is no .json file',
        ),
        (
            lambda v: v.variant(expected=DEEP_JSON["parsable"]),
            "expected.json, are nested too deeply to read as JSON",
        ),
        (
            lambda v: v.forbid(DEEP_JSON["beyond-some-parsers"]),
            "the forbidden literals, tests/gen/forbidden.json, are nested too deeply to read as JSON",
        ),
    ],
    ids=[
        "not-json",
        "no-object",
        "no-output",
        "no-pointer",
        "no-number",
        "tolerance-elsewhere",
        "no-inputs",
        "inputs-a-file",
        "literals-no-list",
        "literal-no-object",
        "negative-tolerance",
        "nan",
        "more-than-value-and-tolerance",
        "no-keys",
        "too-many-keys",
        "too-large",
        "too-many-literals",
        "an-output-no-json",
        "expected-nested-too-deeply",
        "literals-nested-too-deeply",
    ],
)
def test_check_data_the_launcher_cannot_use_is_the_protocol_fault_of_the_graders_owner(
    verifier, owner, prefix, setup, problem
):
    """The data of the replay checks lie in tests/gen, where the grader's owner put them: data the launcher cannot use
    make no measurement, and are found before anything replays."""

    _checked(verifier)
    verifier.config["grader_owner"] = owner
    setup(verifier)

    _, reward, report = verifier.run()

    assert reward == {"invalid": 1, "failure_kind": f"{prefix}_protocol"}
    assert problem in report["detail"], report["detail"]
    assert "replay" not in report and "grader" not in report


def _workdir_has(*artifacts: str, directories: tuple[str, ...] = ()) -> Any:
    """A setup that gives the working directory ``directories`` and the files ``artifacts``, declared as artifacts."""

    def setup(verifier: Verifier) -> None:
        for directory in directories:
            (verifier.workdir / directory).mkdir(parents=True)
        for artifact in artifacts:
            if artifact not in directories:
                (verifier.workdir / artifact).write_text("x = 1\n")
            verifier.config["artifacts"].append(artifact)

    return setup


@pytest.mark.parametrize(
    ("inputs", "setup", "problem"),
    [
        (
            {"solve.py": "print(1)"},
            None,
            "solve.py would change solve.py, which the submission holds or the replay writes",
        ),
        (
            {"answer.json": "{}"},
            None,
            "answer.json would change answer.json, which the submission holds or the replay writes",
        ),
        (
            {"src/extra.py": "x = 1\n"},
            _workdir_has("src", directories=("src",)),
            "src would change src, which the submission holds or the replay writes",
        ),
        (
            {"pkg": "a file"},
            _workdir_has("pkg/mod.py", directories=("pkg",)),
            "pkg would change pkg/mod.py, which the submission holds or the replay writes",
        ),
    ],
    ids=["an-artifact", "an-output", "a-directory-artifact", "a-file-above-an-artifact"],
)
def test_variant_inputs_that_cannot_be_laid_over_the_working_inputs_are_the_owners_protocol_fault(
    verifier, inputs, setup, problem
):
    _checked(verifier)
    if setup is not None:
        setup(verifier)
    verifier.variant(inputs=inputs)

    _, reward, report = verifier.run()

    assert reward == {"invalid": 1, "failure_kind": "verifier_generated_protocol"}
    assert report["detail"] == f"variant fifty's inputs cannot be laid over the working inputs: {problem}"
    assert report["replay"]["exit_code"] == 0 and "grader" not in report


def _workdir_link(name: str) -> Any:
    def setup(verifier: Verifier) -> None:
        (verifier.workdir / name).symlink_to("/etc")

    return setup


@pytest.mark.parametrize(
    ("inputs", "setup", "problem"),
    [
        (
            {"data.txt/inner": "x"},
            None,
            "data.txt is a directory of the variant's and a file of the working directory's",
        ),
        (
            {"tools": "a file"},
            _workdir_has(directories=("tools",)),
            "tools is a file of the variant's and a directory of the working directory's",
        ),
        (
            {"sub/data.txt": "50"},
            _workdir_link("sub"),
            "sub is a directory of the variant's and a file of the working directory's",
        ),
    ],
    ids=["a-directory-for-a-file", "a-file-for-a-directory", "a-directory-for-a-link"],
)
def test_variant_inputs_the_working_directory_the_submission_left_differs_from_are_the_solvers_zero(
    verifier, inputs, setup, problem
):
    # materialize checked the inputs against the task's environment/: what differs there, the submission left.
    _checked(verifier)
    if setup is not None:
        setup(verifier)
    verifier.variant(inputs=inputs)

    _, reward, report = verifier.run()

    assert reward == {"reward": 0} and report["reason"] == launcher.VARIANT_FAILED
    assert report["detail"] == (
        f"variant fifty's inputs cannot be laid over the working inputs: {problem}, which the submission changed"
    )
    assert report["replay"]["exit_code"] == 0 and "grader" not in report


@pytest.mark.parametrize(
    ("special", "kind"),
    [(lambda path: path.symlink_to("/etc/passwd"), "a symbolic link"), (os.mkfifo, "not a regular file")],
    ids=["a-link", "a-fifo"],
)
def test_variant_inputs_that_hold_a_link_or_a_special_file_are_the_owners_protocol_fault(verifier, special, kind):
    _checked(verifier)
    root = verifier.variant()
    special(root / "inputs" / "passwd")

    _, reward, report = verifier.run()

    assert reward == {"invalid": 1, "failure_kind": "verifier_generated_protocol"}
    assert report["detail"].endswith(f": passwd is {kind}")


HARDCODED_SOLVE = """
import json
with open("answer.json", "w") as stream:
    json.dump({"answer": 42.2}, stream)
"""


def test_a_source_that_states_a_forbidden_literal_is_hardcoded_before_anything_replays(verifier):
    """The report says where the source states it and which literal it matches, never what is forbidden."""

    _checked(verifier, HARDCODED_SOLVE)
    verifier.forbid([(41.0, 0.1), (42.0, 0.25)])

    _, reward, report = verifier.run()

    assert reward == {"reward": 0}
    assert (report["reason"], report["detail"]) == (
        "solver_hardcoded",
        "solve.py states 42.2 on line 4, within the tolerance of forbidden literal 1",
    )
    record = report["replay_checks"]["forbidden_literals"]
    assert re.fullmatch(r"3\.[0-9]+\.[0-9]+", record.pop("python"))  # the verifier's python3, which parsed the sources
    assert record == {
        "scan": launcher.LITERAL_SCAN,
        "sources": ["solve.py"],
        "literals": 1,
        "hits": [{"source": "solve.py", "line": 4, "literal": "42.2", "forbidden": 1}],
    }
    assert not {"41.0", "42.0"} & set(re.findall(r"[0-9.]+", json.dumps(report)))
    assert "replay" not in report and "grader" not in report


def test_a_source_that_states_no_forbidden_literal_replays_and_is_graded(verifier):
    """A comment, a string that is no number and a number outside every window state nothing forbidden."""

    _checked(verifier, SOLVE + '# the paper reports 42\nnote = "about 42"\nnear = 41.4\n')
    verifier.forbid([(42, 0.5)])
    verifier.variant()

    _, reward, report = verifier.run()

    assert reward == {"reward": 1}, report
    assert report["replay_checks"]["forbidden_literals"]["hits"] == []


def test_every_hit_counts_and_the_report_keeps_the_first(verifier):
    solve = "".join(f"x{n} = 42\n" for n in range(launcher.REPORTED_HITS + 5)) + SOLVE
    _checked(verifier, solve)
    verifier.forbid([(42, 0.0)])

    report = verifier.run()[2]

    assert report["detail"].endswith(f"forbidden literal 0, and {launcher.REPORTED_HITS + 4} more")
    hits = report["replay_checks"]["forbidden_literals"]["hits"]
    assert [hit["line"] for hit in hits] == list(range(1, launcher.REPORTED_HITS + 1))


@pytest.mark.parametrize(
    ("source", "detail"),
    [
        ("def broken(:\n", "is no Python the scan reads: SyntaxError"),
        ("x = 1\0\n", "is no Python the scan reads"),
        (
            "x = 1\n" + "#" * launcher.MAX_SOURCE_BYTES,
            f"cannot be read: solve.py holds more than {launcher.MAX_SOURCE_BYTES}",
        ),
        (
            "# coding: punycode\nx = 1\n",
            "is no Python the scan reads: ValueError: its coding line names punycode, and the scan reads UTF-8 alone",
        ),
    ],
    ids=["syntax", "nul", "too-large", "another-encoding"],
)
def test_a_source_the_scan_cannot_read_breaks_the_output_contract(verifier, source, detail):
    _checked(verifier)
    (verifier.workdir / "solve.py").write_text(source)
    verifier.forbid([(42, 0.5)])

    _, reward, report = verifier.run()

    assert reward == {"reward": 0}
    assert report["reason"] == "solver_output_contract", report
    assert (
        report["detail"].startswith("solve.py, a source the task scans for forbidden literals,")
        and detail in report["detail"]
    )


def test_a_source_the_submission_lacks_breaks_the_output_contract(verifier):
    verifier.replay()
    verifier.config["artifacts"] = ["answer.json", "solve.py", "src"]
    (verifier.workdir / "src").mkdir()
    verifier.submit(42, **{"solve.py": SOLVE, "data.txt": "21"})
    verifier.forbid([(42, 0.5)], sources=("solve.py", "src/main.py"))

    _, reward, report = verifier.run()

    assert (reward, report["reason"]) == ({"reward": 0}, "solver_output_contract")
    assert report["detail"] == "src/main.py, a source the task scans for forbidden literals, is missing"


@pytest.mark.parametrize(
    ("source", "numbers"),
    [
        ("x = 4.5\n", [4.5]),
        ("x = -4.5\n", [-4.5]),
        ("x = +4.5\n", [4.5]),
        ("x = --4.5\n", [4.5]),  # a chain of signs is one value: an even number of minus signs is positive
        ("x = -(-4.5)\n", [4.5]),
        ("x = +-4.5\n", [-4.5]),
        ("x = -+-+-4.5\n", [-4.5]),
        ("x = 1 - -4.5\n", [1, -4.5]),  # a binary minus is no sign
        ("x = 2 - 4.5\n", [2, 4.5]),
        ("x = -~4\n", [4]),  # nothing is evaluated: ~ ends the chain, and the constant stands alone
        ("x = -'4.5'\n", [4.5]),  # a string takes no sign
        ("x = ' 4.5 '\n", [4.5]),  # a string that float() reads, once stripped
        ("x = '4.5 apples'\n", []),
        ("x = '" + " " * 100 + "4.5'\n", [4.5]),  # whatever its length
        ("x = '" + "0" * 100 + "4.5'\n", [4.5]),
        ("x = '4.2E+1', 4.2e1, '4_2'\n", [42.0, 42.0, 42.0]),
        ("x = 'nan', 'inf', 1e400, -1e400\n", []),  # no finite number
        ("x = True, None, ...\n", []),
        ("x = b' 4.5', b'\\xff4'\n", [4.5]),  # bytes that are ASCII read as a string does
        ("x = 3j, -2.5j\n", [3.0, -2.5]),  # a complex constant's imaginary part
        ("# 4.5\nx = 1\n", [1]),
        ('"""4.5"""\n', [4.5]),
        ("x = 10**400\n", [10, 400]),
        ("x = 10**400 * -10**400\n", [10, 400, 10, 400]),  # the sign applies to a power, not to its base
        ("x = f'{y}4.5'\n", [4.5]),
        ("x = 1_000 + 0x10\n", [1000, 16]),
    ],
)
def test_source_literals_reads_the_numbers_a_source_states_by_its_rule(source, numbers):
    assert sorted(value for _, value in launcher.source_literals(source)) == sorted(numbers)


def test_constant_numbers_reads_a_complex_numbers_real_part_where_it_is_not_0():
    assert launcher.constant_numbers(complex(2, 3)) == [2.0, 3.0] and launcher.constant_numbers(-0j) == [-0.0]
    assert launcher.constant_numbers(complex("inf+2j")) == [2.0] and launcher.constant_numbers(bytearray(b"1")) == []


def test_source_literals_says_on_which_line_each_number_is():
    assert sorted(launcher.source_literals(b"a = 1\n\nb = (\n    -2.5)\nc = -(\n-3)\n")) == [(1, 1), (4, -2.5), (5, 3)]


@pytest.mark.parametrize(
    ("data", "problem"),
    [
        (b"# coding: punycode\nx = 1\n", "its coding line names punycode, and the scan reads UTF-8 alone"),
        (b"#!/bin/python3\n# -*- coding: latin-1 -*-\nx = '\\xe9'\n", "its coding line names latin-1"),
        (b"\xef\xbb\xbf# coding: latin-1\nx = 1\n", "its coding line names latin-1"),
        (b"# coding: no-such-codec\nx = 1\n", "its coding line names no-such-codec, a codec Python lacks"),
        (b"x = '\xff'\n", "can't decode byte 0xff"),
        (b"x = 1\ny = 2\nz = '\xff'\n", "can't decode byte 0xff"),
    ],
    ids=["a-codec-in-python", "latin-1", "a-mark-and-latin-1", "an-unknown-codec", "no-utf-8", "no-utf-8-later"],
)
def test_the_scan_reads_utf_8_alone_whatever_a_coding_line_names(data, problem):
    with pytest.raises(ValueError, match=re.escape(problem)):
        launcher.source_text(data)


def test_the_scan_finds_a_coding_line_without_running_its_codec():
    """A coding line that names a codec, here one that fails whenever it runs, as one written in Python could take
    seconds to decode a line of 256 KiB, is refused without running it; and neither a third line nor one after code is
    a coding line."""

    def ran(*args: Any) -> Any:
        raise AssertionError("the codec ran")

    def search(name: str) -> Any:
        return codecs.CodecInfo(ran, ran, name="platoon-test-codec") if name == "platoon_test_codec" else None

    codecs.register(search)
    try:
        data = b"# coding: platoon-test-codec " + b"a" * launcher.MAX_SOURCE_BYTES + b"\nx = 1\n"
        with pytest.raises(ValueError, match="its coding line names platoon-test-codec, and the scan reads UTF-8"):
            launcher.source_text(data)
    finally:
        codecs.unregister(search)
    for later in (b"x = 1\n# coding: latin-1\n", b"\n\n# coding: latin-1\nx = 4.5\n"):
        assert launcher.source_text(later) == later.decode()


def test_the_scan_reads_utf_8_with_or_without_a_byte_order_mark():
    sources = (
        b"x = 4.5\n",
        b"\xef\xbb\xbfx = 4.5\n",
        b"# coding: utf8\nx = 4.5\n",
        b"\n# -*- coding: UTF-8 -*-\nx = 4.5\n",
        b"# coding: utf-8-sig\nx = 4.5\n",
        "x = '\u00e9', 4.5\n".encode(),
    )
    for data in sources:
        assert [value for _, value in launcher.source_literals(launcher.source_text(data))] == [4.5]


def test_the_windows_find_the_first_forbidden_literal_a_number_lies_within():
    windows = launcher.Windows([(42, 0.5), (0.1, 0.2), (41.8, 0.5), (1e300, 1e299)])

    assert windows.first(42.5) == 0  # a window's ends are in it
    assert windows.first(42.0) == 0  # in two windows: the first listed
    assert windows.first(41.4) == 2
    assert windows.first(-0.1) == 1
    assert windows.first(10**300) == 3
    for outside in (42.6, 0.30000000000000004, 10**30, -1e308):  # by the exact test: |x - value| <= tolerance
        assert windows.first(outside) is None
    assert launcher.Windows([]).first(1.0) is None


def test_an_output_key_names_one_output_and_a_json_pointer_into_it():
    outputs = ["answer.json", "out/table.json", "a#b"]

    assert launcher.output_key("answer.json", outputs) == ("answer.json", [])
    assert launcher.output_key("out/table.json#/rows/0", outputs) == ("out/table.json", ["rows", "0"])
    assert launcher.output_key("out/table.json#", outputs) == ("out/table.json", [])
    assert launcher.output_key("a#b", outputs) == ("a#b", [])  # an output whose name holds a #
    assert launcher.output_key("a#b#/x~1y~0z", outputs) == ("a#b", ["x/y~z"])
    for key in ("other.json", "answer.json#rows", "answer.json#/a~2", "", 7, "x" * (launcher.MAX_KEY_CHARS + 1)):
        assert launcher.output_key(key, outputs) is None


@pytest.mark.parametrize(
    ("document", "key", "expected", "tolerance", "held"),
    [
        ({"a": 1.05}, "answer.json#/a", 1, 0.1, True),
        ({"a": 1.2}, "answer.json#/a", 1, 0.1, False),
        ({"a": [1, 2.04]}, "answer.json#/a", [1, 2], 0.05, True),
        ({"a": [1, 2]}, "answer.json#/a", [1, 2, 3], 0.05, False),  # not as many
        ({"a": [1, "2"]}, "answer.json#/a", [1, 2], 0.05, False),
        ({"a": True}, "answer.json#/a", 1, 0.5, False),  # a boolean is no number
        ([[0, 1], [2, 3.5]], "answer.json#/1/1", 3.5, 0, True),
        ([[0, 1]], "answer.json#/00/0", 0, 0, False),  # an index has no leading zero
        ({"a/b": {"~c": 7}}, "answer.json#/a~1b/~0c", 7, 0, True),
        (5, "answer.json", 5, 0, True),  # the whole of the output
        ({"a": 1}, "answer.json#/b", 1, 0, False),
        ({"a": 1e308}, "answer.json#/a", -1e308, 1, False),  # too far for a float to tell how far
    ],
)
def test_a_variant_matches_what_it_expects_at_each_key(tmp_path, document, key, expected, tolerance, held):
    (tmp_path / "answer.json").write_text(json.dumps(document))
    keys = launcher.expected_keys({key: expected}, ["answer.json"], {"default": tolerance, "keys": {}})

    assert (launcher.missed_keys(str(tmp_path), keys, 1 << 20) == []) is held


def test_a_variants_tolerance_gives_the_keys_it_names_their_own(tmp_path):
    (tmp_path / "answer.json").write_text(json.dumps({"a": 1.5, "b": 1.5}))
    tolerance = {"default": 0.1, "keys": {"answer.json#/b": 0.5}}
    keys = launcher.expected_keys({"answer.json#/a": 1, "answer.json#/b": 1}, ["answer.json"], tolerance)

    assert launcher.missed_keys(str(tmp_path), keys, 1 << 20) == ["answer.json#/a"]


def test_what_a_variants_replay_wrote_is_read_as_the_launcher_reads_every_file(tmp_path):
    """Never through a link, never waiting on a FIFO: either is no output, and its keys are missed."""

    os.mkfifo(tmp_path / "answer.json")
    (tmp_path / "real.json").write_text('{"a": 1}')
    (tmp_path / "link.json").symlink_to(tmp_path / "real.json")
    tolerance = {"default": 0, "keys": {}}

    for output in ("answer.json", "link.json"):
        keys = launcher.expected_keys({f"{output}#/a": 1}, [output], tolerance)
        assert launcher.missed_keys(str(tmp_path), keys, 1 << 20) == [f"{output}#/a"]


def test_a_variants_output_nested_deeper_than_the_launcher_reads_holds_no_key(tmp_path):
    """The output contract refuses such an output once the variant's replay wrote it; read anyway, it holds nothing,
    on every Python."""

    depth = launcher.MAX_JSON_DEPTH
    (tmp_path / "answer.json").write_text("[1, " + "[" * depth + "]" * depth + "]")
    keys = launcher.expected_keys({"answer.json#/0": 1}, ["answer.json"], {"default": 0, "keys": {}})

    assert launcher.missed_keys(str(tmp_path), keys, 1 << 20) == ["answer.json#/0"]


def _deep(depth: int) -> Any:
    """A list ``depth`` levels deep."""

    value: Any = 0
    for _ in range(depth):
        value = [value]
    return value


_VARIANT = {
    "name": "fifty",
    "inputs": "tests/gen/i",
    "expected": "tests/gen/e.json",
    "tolerance": {"default": 0, "keys": {}},
    "meta": None,
}
_FORBIDDEN = {"path": "tests/gen/f.json", "sources": ["solve.py"], "scan": launcher.LITERAL_SCAN}


@pytest.mark.parametrize(
    "checks",
    [
        {"variants": []},
        {"variants": [{"name": "fifty"}], "forbidden_literals": None},
        {"variants": [{**_VARIANT, "name": "fifty two"}], "forbidden_literals": None},
        {"variants": [{**_VARIANT, "inputs": "tests/platoon"}], "forbidden_literals": None},
        {"variants": [{**_VARIANT, "expected": "tests/gen"}], "forbidden_literals": None},
        {"variants": [_VARIANT, _VARIANT], "forbidden_literals": None},
        {"variants": [{**_VARIANT, "tolerance": {"default": -1, "keys": {}}}], "forbidden_literals": None},
        {"variants": [{**_VARIANT, "tolerance": {"default": 0, "keys": {"k": True}}}], "forbidden_literals": None},
        {"variants": [], "forbidden_literals": {**_FORBIDDEN, "scan": "regex/v1"}},
        {"variants": [], "forbidden_literals": {**_FORBIDDEN, "path": "/tests/gen/f.json"}},
        {"variants": [], "forbidden_literals": {**_FORBIDDEN, "sources": ["../solve.py"]}},
        {"variants": [], "forbidden_literals": {**_FORBIDDEN, "sources": []}},
        {"variants": [], "forbidden_literals": {**_FORBIDDEN, "sources": ["solve.R"]}},
        {"variants": [{**_VARIANT, "meta": _deep(launcher.MAX_META_DEPTH + 1)}], "forbidden_literals": None},
    ],
    ids=[
        "no-forbidden-literals",
        "a-variant-without-its-fields",
        "a-name-with-a-space",
        "inputs-outside-gen",
        "expected-gen-itself",
        "two-variants-one-name",
        "negative-tolerance",
        "boolean-tolerance",
        "another-scan",
        "absolute-path",
        "source-outside",
        "no-sources",
        "no-python-source",
        "meta-too-deep",
    ],
)
def test_replay_checks_the_launcher_cannot_run_are_settings_it_cannot_use(verifier, checks):
    _checked(verifier)
    verifier.config["replay_checks"] = checks

    _, reward, report = verifier.run()

    assert reward == {"invalid": 1, "failure_kind": "verifier_trusted_launcher"}
    assert "hold no usable replay_checks" in report["detail"]


def test_replay_checks_without_a_replay_are_settings_the_launcher_cannot_use(verifier):
    verifier.submit(42)
    verifier.forbid([(42, 0.5)])

    _, reward, report = verifier.run()

    assert reward == {"invalid": 1, "failure_kind": "verifier_trusted_launcher"}
    assert "hold no usable replay_checks" in report["detail"]


def test_a_report_too_large_for_its_bound_leaves_out_the_replay_checks_records_before_the_outcome():
    record = {"scan": launcher.LITERAL_SCAN, "hits": [{"source": "s" * 300}] * 10}
    report = {
        "schema": launcher.SCHEMA,
        "replay_checks": {
            "variants": [{"name": f"v{n}", "passed": True} for n in range(200)],
            "forbidden_literals": record,
        },
        "reward": 0,
        "failure_kind": None,
        "reason": "solver_variant_failed",
        "detail": "d",
        "seconds": 1.0,
    }
    cut = {**report, "replay_checks": {"forbidden_literals": record}, "truncated": ["replay_checks.variants"]}
    limit = len(json.dumps(cut, sort_keys=True))

    assert json.loads(launcher.bounded_report(report, limit=limit)) == cut
    least = json.loads(launcher.bounded_report(report, limit=limit - 1))
    assert least["replay_checks"] == {} and least["truncated"] == [
        "replay_checks.variants",
        "replay_checks.forbidden_literals",
    ]


# ------------------------------------------------------------------ the variants' replays keep the invariants (I1-I4)

# A variant's replay is the replay's own code path (launcher._run_replay): the same sweep, demotion, environment and
# clearing, so each invariant above holds for every replay of a grade, the replay's and each variant's.


def test_a_process_of_the_replays_user_that_a_variants_sweep_cannot_signal_is_a_survivor(verifier):
    """I1 for a variant's replay: a process of the replay's user that its sweep may not signal is a survivor, the
    solver's 0, never excluded as foreign.  The Ops seam names one such process once the variant's replay ran, and
    signalling it is refused; the replay's own sweep, before it, found none."""

    marker = verifier.root / "variant-ran"
    _checked(verifier, SOLVE + _in_variants(f"open({str(marker)!r}, 'w').close()"))
    verifier.variant()
    extra = f"""
        impossible = 2**31 - 1
        signal_one, kill = launcher._signal, launcher.kill_left

        def refused(pid):
            if pid == impossible:
                raise PermissionError(pid)
            return signal_one(pid)

        class Unreachable(WithoutRoot):
            def processes_of(self, uid):
                return [impossible] if os.path.exists({str(marker)!r}) else []

        launcher._signal = refused
        launcher.kill_left = lambda also=None, window_s=None: kill(also, 0.2)
    """

    _, reward, report = verifier.run(extra=extra, ops="Unreachable()")

    assert (reward, report["reason"], report["detail"]) == (
        {"reward": 0},
        "solver_variant_failed",
        "processes the replay of variant fifty left could not be killed (1)",
    )
    (entry,) = report["replay_checks"]["variants"]
    assert (entry["passed"], entry["survivors"]) == (False, 1) and "survivors" not in report["replay"]
    assert "grader" not in report


# The launcher's own demotion (Ops.demote: the OOM killer's disfavor, the replay's limits and address space, no new
# privileges), but for the switch to the replay's user, which needs root: each call is noted in ``record``.
_OWN_DEMOTION = """
import json
REPLAY = (launcher.REPLAY_UID,) * 3
os.setgroups = lambda groups: None
os.setgid = os.setuid = lambda number: None
os.getresuid = os.getresgid = lambda: REPLAY
os.getgroups = lambda: []

class OwnDemotion(WithoutRoot):
    def demote(self, max_file_bytes, max_address_space_bytes=None):
        with open({record!r}, "a") as stream:
            stream.write(json.dumps([max_file_bytes, max_address_space_bytes]) + "\\n")
        return launcher.Ops.demote(self, max_file_bytes, max_address_space_bytes)
"""


@pytest.mark.skipif(not _oom_adj_writable(), reason="/proc/self/oom_score_adj is not writable here")
def test_every_replay_of_a_grade_is_demoted_as_the_replay_is(verifier):
    """I2 for every variant's replay: each replay of the grade, the replay's and both variants', is demoted by the
    launcher's own ``Ops.demote`` with the settings' limits and address space, so each runs as the most killable to the
    OOM killer, within that address space and without new privileges, as the variant's replay finds."""

    record = verifier.root / "demoted.jsonl"
    space = 1 << 36
    hard = resource.getrlimit(resource.RLIMIT_AS)[1]
    want = space if hard == resource.RLIM_INFINITY else min(space, hard)
    code = (
        "status = open('/proc/self/status').read()\n    "
        "json.dump({'answer': 100, 'oom': int(open('/proc/self/oom_score_adj').read()), "
        "'space': __import__('resource').getrlimit(__import__('resource').RLIMIT_AS)[1], "
        "'unprivileged': int('NoNewPrivs:\\t1' in status)}, open('answer.json', 'w'))"
    )
    _checked(verifier, SOLVE + _in_variants(code))
    expected = {"answer": 100, "oom": launcher.OOM_SCORE_ADJ, "space": want, "unprivileged": 1}
    verifier.variant(expected={f"answer.json#/{key}": value for key, value in expected.items()})
    verifier.variant("seven", inputs={"data.txt": "7"}, expected={"answer.json#/answer": 14})
    verifier.config["max_address_space_bytes"] = space

    _, reward, report = verifier.run(extra=_OWN_DEMOTION.format(record=str(record)), ops="OwnDemotion()")

    assert reward == {"reward": 1}, report
    assert [json.loads(line) for line in record.read_text().splitlines()] == [[1 << 20, space]] * 3
    assert [entry["passed"] for entry in report["replay_checks"]["variants"]] == [True, True]


def test_every_variants_replay_finds_the_working_directory_hardened_and_an_absolute_path(verifier):
    """I3 and I4 for every variant's replay: the working directory, hardened before the replay and put back hardened
    after each, holds no file or directory that its group or other users may write while a variant replays (a
    submitted file any user could write among them), lent to the replay's user alone, and a variant's replay gets the
    replay's PATH, its absolute entries alone."""

    workdir = str(verifier.workdir)
    code = (
        f"exposed = [n for d, ds, fs in os.walk({workdir!r}) for n in ds + fs "
        "if not os.path.islink(os.path.join(d, n)) and os.lstat(os.path.join(d, n)).st_mode & 0o022]\n    "
        "json.dump({'answer': 100, 'exposed': len(exposed), "
        "'path': int(os.environ['PATH'] == '/usr/bin:/bin')}, open('answer.json', 'w'))"
    )
    _checked(verifier, SOLVE + _in_variants(code))
    loose = verifier.workdir / "sub" / "loose.txt"
    loose.parent.mkdir()
    loose.write_text("x")
    loose.chmod(0o666)
    verifier.variant(expected={"answer.json#/answer": 100, "answer.json#/exposed": 0, "answer.json#/path": 1})

    _, reward, report = verifier.run(env={"PATH": "/usr/bin:.:rel::/bin"})

    assert reward == {"reward": 1}, report
    assert report["replay_checks"]["variants"][0]["passed"]
    assert not stat.S_IMODE(loose.stat().st_mode) & (stat.S_IWGRP | stat.S_IWOTH)


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


@_IPC_NAMESPACES
@pytest.mark.parametrize("removed", [True, False], ids=["removed", "cannot-be-removed"])
def test_the_system_v_ipc_objects_a_variants_replay_left_are_removed_before_the_next_replay(verifier, removed):
    """I2 for a variant's replay: what it left of System V IPC objects is removed with what it left where every user
    may write, before the next replay or the grader runs; one that cannot be removed is the variant's failure, the
    solver's 0."""

    leaves_in_the_variant = """
if open("data.txt").read().strip() == "50":
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    libc.shmget(0, 1 << 20, 0o1600)
    libc.semget(0, 1, 0o1600)
    libc.msgget(0, 0o1600)
"""
    _checked(verifier, SOLVE + leaves_in_the_variant)
    verifier.variant()
    verifier.grader(_NO_IPC_GRADER)
    extra = _IPC_OPS + ("" if removed else "launcher.remove_ipc = lambda objects: 0\n")

    _, reward, report = verifier.run(extra=extra, ops="IpcOps()", prefix=("unshare", "-r", "-i"))

    (entry,) = report["replay_checks"]["variants"]
    if removed:
        assert reward == {"reward": 1}, report
        assert entry["cleared"] == 3 and "cleared" not in report["replay"]
        return
    assert (reward, report["reason"]) == ({"reward": 0}, "solver_variant_failed"), report
    assert report["detail"] == (
        "the replay of variant fifty left what the launcher cannot remove in its time: 3 System V IPC object(s) of its "
        "user cannot be removed"
    )


# ------------------------------------------------------------------------------------- hidden evaluation conditions

# A grader that judges what each condition's replay wrote against hidden targets of its own (twice the condition's
# data.txt), as a corpus grader does, and the replayed answer of the working inputs (42); it notes what it found in
# PLATOON_CONDITIONS in ``seen`` (a file of the test's, beside the verifier).
_CONDITION_GRADER = """
import json, os, stat
conditions = os.environ.get("PLATOON_CONDITIONS")
seen = {{"conditions": conditions}}
passed = False
if conditions is not None:
    manifest = json.load(open(os.path.join(conditions, "manifest.json")))
    seen["manifest"] = manifest
    seen["modes"] = {{
        os.path.relpath(os.path.join(d, n), conditions): stat.S_IMODE(os.lstat(os.path.join(d, n)).st_mode)
        for d, ds, fs in os.walk(conditions) for n in ds + fs
    }}
    seen["top"] = stat.S_IMODE(os.lstat(conditions).st_mode)
    answers = {{
        entry["id"]: json.load(open(os.path.join(conditions, "outputs", entry["id"], "answer.json")))
        for entry in manifest["conditions"]
    }}
    seen["answers"] = answers
    seen["resolved"] = [os.path.isdir(entry["inputs"]) for entry in manifest["conditions"]]  # from /tests/gen
    targets = {targets!r}
    passed = all(answers.get(key, {{}}).get("answer") == value for key, value in targets.items())
with open(os.path.join(os.environ["PLATOON_WORKDIR"], "answer.json")) as stream:
    passed = passed and json.load(stream)["answer"] == 42
json.dump(seen, open({seen!r}, "w"))
with open(os.environ["PLATOON_VERDICT"], "w") as stream:
    json.dump({{"passed": passed}}, stream)
"""


def _conditioned(verifier: Verifier, solve: str = SOLVE, targets: dict[str, int] | None = None) -> Path:
    """A task with a replay (:func:`_checked`) and two hidden conditions, fifty and sixty (data.txt of 50 and 60), whose
    grader passes where the replay wrote 42 and each condition's replay twice its data; the file where the grader notes
    what it found."""

    _checked(verifier, solve)
    verifier.condition("fifty")
    verifier.condition("sixty", {"data.txt": "60"})
    seen = verifier.root / "seen.json"
    verifier.grader(_CONDITION_GRADER.format(targets=targets or {"fifty": 100, "sixty": 120}, seen=str(seen)))
    return seen


def _in_condition(data: str, code: str) -> str:
    """Python that runs ``code`` only in the replay whose data.txt holds ``data``."""

    return f"\nimport json, os, sys\nif open('data.txt').read().strip() == {data!r}:\n    {code}\n"


def test_a_submission_that_follows_its_inputs_is_graded_on_what_each_condition_wrote(verifier):
    """After the replay, the launcher replays the submission on each condition and hands the grader what each wrote,
    in PLATOON_CONDITIONS, beside the replayed answer in the working directory."""

    seen = _conditioned(verifier)

    _, reward, report = verifier.run()

    assert reward == {"reward": 1}, report
    found = json.loads(seen.read_text())
    assert found["answers"] == {"fifty": {"answer": 100}, "sixty": {"answer": 120}}
    assert [(entry["id"], entry["passed"], entry["bytes"]) for entry in report["conditions"]["ran"]] == [
        ("fifty", True, 15),
        ("sixty", True, 15),
    ]
    assert json.loads((verifier.workdir / "answer.json").read_text()) == {"answer": 42}
    assert report[launcher.REPLAYED_SHA256] == SOLVED_SHA256


def test_a_submission_that_passes_the_replay_but_ignores_its_inputs_fails_the_graders_hidden_targets(verifier):
    """The working inputs' answer, written again on every condition, is the grader's 0 (no reason), never a launcher's:
    the launcher judges no condition's outputs."""

    _conditioned(verifier, IGNORING_SOLVE)

    _, reward, report = verifier.run()

    assert (reward, report["reason"], report["detail"]) == ({"reward": 0}, None, "the verdict failed"), report
    assert [entry["passed"] for entry in report["conditions"]["ran"]] == [True, True]


def test_the_grader_gets_each_conditions_outputs_root_s_alone_with_a_manifest_of_them(verifier):
    """The layout of PLATOON_CONDITIONS: manifest.json and outputs/<id>/<output>, directories 700 and files 600 (sealed
    root's), the manifest listing the conditions in their order with the size and SHA-256 of each output as the grader
    gets it, and the SHA-256 of them all, which the report holds too."""

    seen = _conditioned(verifier)

    _, reward, report = verifier.run()

    assert reward == {"reward": 1}, report
    found = json.loads(seen.read_text())
    digest = hashlib.sha256(b'{"answer": 100}').hexdigest(), hashlib.sha256(b'{"answer": 120}').hexdigest()
    manifest = found["manifest"]
    assert manifest["schema"] == launcher.CONDITIONS_SCHEMA == "platoon.conditions/v1"
    assert [(entry["id"], entry["inputs"], entry["outputs"], sorted(entry)) for entry in manifest["conditions"]] == [
        (
            name,
            f"conditions/{name}",  # as grader.json gives it: relative to the grader's own directory
            [{"path": "answer.json", "bytes": 15, "sha256": sha}],
            ["id", "inputs", "outputs", "seconds", "timeout_s"],
        )
        for name, sha in zip(("fifty", "sixty"), digest, strict=True)
    ]
    assert found["resolved"] == [True, True]
    assert (
        manifest["sha256"]
        == report[launcher.CONDITIONS_SHA256]
        == launcher.outputs_sha256(
            str(Path(found["conditions"]) / "outputs"),
            ["fifty/answer.json", "sixty/answer.json"],
            rule=launcher.CONDITION_OUTPUTS_RULE,
        )
    )
    assert found["top"] == 0o700 and found["modes"] == {
        "manifest.json": 0o600,
        "outputs": 0o700,
        "outputs/fifty": 0o700,
        "outputs/fifty/answer.json": 0o600,
        "outputs/sixty": 0o700,
        "outputs/sixty/answer.json": 0o600,
    }
    assert Path(found["conditions"]).parent.name == report_directory(verifier)


def test_a_task_without_conditions_gives_the_grader_no_conditions_and_its_report_none(verifier):
    """Not even the verifier's own PLATOON_CONDITIONS or PLATOON_SUBMITTED, which would name what no replay wrote: the
    grader gets them from the launcher alone."""

    seen = verifier.root / "seen.json"
    verifier.grader(_CONDITION_GRADER.format(targets={}, seen=str(seen)))
    stale = {"PLATOON_CONDITIONS": str(verifier.root), "PLATOON_SUBMITTED": str(verifier.root)}

    _checked(verifier)
    _, reward, report = verifier.run(env=stale)

    assert json.loads(seen.read_text()) == {"conditions": None} and reward == {"reward": 0}
    assert not {"conditions", launcher.CONDITIONS_SHA256} & set(report)


def test_a_grader_without_a_replay_gets_no_inherited_platoon_submitted(verifier):
    seen = verifier.root / "seen.txt"
    verifier.grader(
        f"""
        import os
        open({str(seen)!r}, "w").write(repr(os.environ.get("PLATOON_SUBMITTED")))
        open(os.environ["PLATOON_VERDICT"], "w").write('{{"passed": true}}')
        """
    )

    verifier.submit(42)

    _, reward, report = verifier.run(env={"PLATOON_SUBMITTED": str(verifier.root)})

    assert reward == {"reward": 1}, report
    assert seen.read_text() == "None"


def test_the_manifest_lists_the_conditions_in_the_order_the_task_declares_them(verifier):
    _checked(verifier)
    verifier.condition("sixty", {"data.txt": "60"})
    verifier.condition("fifty")
    seen = verifier.root / "seen.json"
    verifier.grader(_CONDITION_GRADER.format(targets={"fifty": 100, "sixty": 120}, seen=str(seen)))

    _, reward, report = verifier.run()

    assert reward == {"reward": 1}, report
    assert [entry["id"] for entry in json.loads(seen.read_text())["manifest"]["conditions"]] == ["sixty", "fifty"]
    assert [entry["id"] for entry in report["conditions"]["ran"]] == ["sixty", "fifty"]


def test_the_launcher_seals_what_the_conditions_wrote_before_the_grader_reads_it(verifier):
    sealed = verifier.root / "sealed.txt"
    seen = _conditioned(verifier)

    _, reward, report = verifier.run(
        extra=f"""
        seal = WithoutRoot.seal
        def recorded(self, path):
            open({str(sealed)!r}, "a").write(path + "\\n")
            seal(self, path)
        WithoutRoot.seal = recorded
        """
    )

    assert reward == {"reward": 1}, report
    assert json.loads(seen.read_text())["conditions"] in sealed.read_text().splitlines()


def test_the_conditions_outputs_may_fill_what_the_launcher_holds_to_the_byte(verifier):
    _conditioned(verifier)

    _, reward, report = verifier.run(extra="launcher.MAX_CONDITION_BYTES = 30")  # 15 bytes each

    assert reward == {"reward": 1}, report


@pytest.mark.parametrize(("free", "fault"), [(10, True), (3 << 20, False)], ids=["too-little", "enough"])
def test_a_launcher_directory_without_room_for_the_outputs_fails_closed_before_any_replay(verifier, free, fault):
    """The replay's outputs (1 MiB each here) and the conditions' (2 x 1 MiB) may need 3 MiB where the launcher holds
    them: a sandbox whose own directory lacks that room is the trusted side's fault, before any replay, never a full
    filesystem a submission's large outputs would make the trusted side's, censoring it."""

    _conditioned(verifier)

    _, reward, report = verifier.run(extra=f"launcher._free_bytes = lambda path: {free}")

    if fault:
        assert reward == {"invalid": 1, "failure_kind": launcher.LAUNCHER_FAULT}, report
        assert report["detail"] == (
            "its own directory has 10 bytes free, less than the 3145728 the replay's and the conditions' outputs may "
            "take there"
        )
        assert "replay" not in report
    else:
        assert reward == {"reward": 1}, report


def test_a_failed_conditions_report_holds_nothing_its_replay_printed(verifier):
    """A condition's replay may print its hidden inputs: the report, which reaches the outcome's metadata, keeps no
    tail of its output."""

    _conditioned(verifier, SOLVE + _in_condition("60", "print('HIDDEN=' + open('data.txt').read()); sys.exit(3)"))

    _, reward, report = verifier.run()

    assert (reward, report["reason"]) == ({"reward": 0}, launcher.CONDITION_FAILED), report
    assert "HIDDEN" not in (verifier.logs / launcher.REPORT).read_text()


def test_no_replay_learns_which_condition_it_runs_or_where_the_grader_finds_it(verifier):
    """A condition's replay gets the replay's thin environment, without PLATOON_CONDITIONS or any id, and its argv."""

    probe = SOLVE + textwrap.dedent(
        """
        import json, os, sys
        leaked = [name for name in os.environ if name.startswith("PLATOON_")] + [a for a in sys.argv if "fifty" in a]
        answer = json.load(open("answer.json"))
        json.dump({**answer, "leaked": leaked}, open("answer.json", "w"))
        """
    )
    seen = _conditioned(verifier, probe)

    _, _, report = verifier.run(env={"PLATOON_CONDITIONS": "/the/verifiers/own"})

    found = json.loads(seen.read_text())
    assert [answer["leaked"] for answer in found["answers"].values()] == [[], []], report


def test_conditions_replay_after_the_replay_and_its_variants_in_their_order(verifier):
    order = verifier.root / "order.txt"
    _checked(verifier, SOLVE + f"\nopen({str(order)!r}, 'a').write(open('data.txt').read().strip() + ' ')\n")
    verifier.variant("seven", inputs={"data.txt": "7"}, expected={"answer.json#/answer": 14})
    for name, data in (("c", "3"), ("a", "1"), ("b", "2")):
        verifier.condition(name, {"data.txt": data})

    _, reward, report = verifier.run()

    assert reward == {"reward": 1}, report
    assert order.read_text().split() == ["21", "7", "3", "1", "2"]
    assert [entry["id"] for entry in report["conditions"]["ran"]] == ["c", "a", "b"]


def test_each_condition_finds_the_working_directory_pristine_with_its_own_inputs_alone(verifier):
    """I5 between conditions: a condition's replay finds neither the replay's outputs, nor what an earlier replay left
    or changed there, nor an earlier condition's inputs (only-fifty.txt, data.txt) or outputs; its own inputs lie over
    the working inputs, beside them where the working directory has none.  The grader then finds the working directory
    as the pristine copy holds it, with the replayed output, and nothing of any condition's."""

    probe = """
        import json, os, sys
        data = open("data.txt").read().strip()
        others = {"50": ["only-sixty.txt"], "60": ["only-fifty.txt"]}.get(data, ["only-fifty.txt", "only-sixty.txt"])
        own = {"50": "only-fifty.txt", "60": "only-sixty.txt"}.get(data)
        dirty = int(
            any(os.path.lexists(name) for name in ["answer.json", "left.txt", *others])
            or (own is not None and not os.path.exists(own))
        )
        open("left.txt", "w").write("left by " + data)
        open("solve.py", "a").write("raise SystemExit(9)\\n")
        with open("answer.json", "w") as stream:
            json.dump({"answer": 2 * int(data), "dirty": dirty, "fresh": int(os.path.exists("notes/kept.txt"))}, stream)
    """
    seen = _conditioned(verifier, textwrap.dedent(probe).lstrip())
    (verifier.workdir / "notes").mkdir()
    (verifier.workdir / "notes" / "kept.txt").write_text("kept")
    (verifier.tests / "gen" / "conditions" / "fifty" / "only-fifty.txt").write_text("fifty's")
    (verifier.tests / "gen" / "conditions" / "sixty" / "only-sixty.txt").write_text("sixty's")
    verifier.grader(
        (verifier.tests / "gen" / "grade.py")
        .read_text()
        .replace('json.load(stream)["answer"] == 42', 'json.load(stream) == {"answer": 42, "dirty": 0, "fresh": 1}')
        .replace(
            "passed = all(",
            "passed = sorted(os.listdir(os.environ['PLATOON_WORKDIR'])) == ['answer.json', 'data.txt', 'notes', "
            "'solve.py'] and all(",
        )
    )

    _, reward, report = verifier.run()

    assert reward == {"reward": 1}, report
    answers = json.loads(seen.read_text())["answers"]
    assert answers == {
        "fifty": {"answer": 100, "dirty": 0, "fresh": 1},
        "sixty": {"answer": 120, "dirty": 0, "fresh": 1},
    }
    assert (verifier.workdir / "solve.py").read_text() == textwrap.dedent(probe).lstrip()  # as submitted


def test_what_a_conditions_replay_leaves_where_every_user_may_write_is_removed_before_the_next(verifier):
    """I2 between conditions: what each replay's user left in /tmp, /var/tmp and /dev/shm is gone before the next
    replay starts, and before the grader runs."""

    seen = verifier.root / "seen.json"
    _checked(verifier, _leaving(verifier))
    verifier.condition("fifty")
    verifier.condition("sixty", {"data.txt": "60"})
    verifier.grader(_CONDITION_GRADER.format(targets={}, seen=str(seen)).replace('["answer"] == 42', '["seen"] == 0'))

    _, reward, report = verifier.run()

    assert reward == {"reward": 1}, report
    answers = json.loads(seen.read_text())["answers"]
    assert [answer["seen"] for answer in answers.values()] == [0, 0]
    assert [entry["cleared"] for entry in report["conditions"]["ran"]] == [3, 3]
    assert [sorted(path.name for path in place.iterdir()) for place in verifier.temporary[1:]] == [[], []]


@pytest.mark.parametrize(
    ("code", "detail"),
    [
        ("sys.exit(3)", "the replay of condition sixty exited 3"),
        ("os.remove('answer.json')", "after the replay of condition sixty, answer.json is missing"),
        ("open('answer.json', 'w').write('{nope')", "after the replay of condition sixty, answer.json is not JSON"),
        (
            "os.remove('answer.json'); os.symlink('/etc/hostname', 'answer.json')",
            "after the replay of condition sixty, answer.json is a symbolic link",
        ),
        (
            "os.remove('answer.json'); os.mkfifo('answer.json')",
            "after the replay of condition sixty, answer.json is not a regular file",
        ),
        (
            "os.link('answer.json', os.path.join(os.environ['HOME'], 'other'))",
            "after the replay of condition sixty, answer.json has other links to it",
        ),
        (
            "open('answer.json', 'w').write(' ' * 2000 + '{}')",
            "after the replay of condition sixty, answer.json holds 1025 bytes, more than an output may (1024)",
        ),
    ],
    ids=["exits", "no-output", "not-json", "a-link", "a-fifo", "a-hard-link", "too-large"],
)
def test_a_conditions_replay_that_does_not_write_every_output_again_is_the_solvers_zero(verifier, code, detail):
    """The launcher decides a condition's failure, before the grader runs: the solver's 0, solver_condition_failed,
    for the first condition that fails; the grader never runs and never sees a failed condition."""

    _conditioned(verifier, SOLVE + _in_condition("60", code))
    verifier.config["max_output_bytes"] = 1024

    _, reward, report = verifier.run()

    assert (reward, report["reason"]) == ({"reward": 0}, "solver_condition_failed"), report
    assert detail in report["detail"], report["detail"]
    first, failed = report["conditions"]["ran"]
    assert first["passed"] and not failed["passed"] and failed["problem"] == report["detail"]
    assert "output_tail" not in failed and "output_tail" not in first  # it may echo the hidden inputs
    assert "grader" not in report and launcher.CONDITIONS_SHA256 not in report


def test_no_condition_replays_once_one_failed(verifier):
    _conditioned(verifier, SOLVE + _in_condition("50", "sys.exit(5)"))

    _, reward, report = verifier.run()

    assert (reward, report["reason"], report["detail"]) == (
        {"reward": 0},
        "solver_condition_failed",
        "the replay of condition fifty exited 5",
    )
    assert [entry["id"] for entry in report["conditions"]["ran"]] == ["fifty"] and "grader" not in report


def test_no_condition_replays_once_a_variant_failed(verifier):
    _conditioned(verifier, IGNORING_SOLVE)
    verifier.variant()

    _, reward, report = verifier.run()

    assert (reward, report["reason"]) == ({"reward": 0}, "solver_variant_failed")
    assert "conditions" not in report and "grader" not in report


def test_a_conditions_replay_that_outlives_its_time_is_the_solvers_zero(verifier):
    _conditioned(verifier, SOLVE + _in_condition("60", "__import__('time').sleep(60)"))
    verifier.config["replay_timeout_s"] = 2

    _, reward, report = verifier.run()

    assert (reward, report["reason"]) == ({"reward": 0}, "solver_condition_failed")
    found = re.fullmatch(r"the replay of condition sixty outlived its ([0-9.]+) s", report["detail"])
    assert found is not None and 1.0 < float(found.group(1)) <= 2.0, report["detail"]  # less putting the directory back
    assert report["conditions"]["ran"][1]["timed_out"]


def test_a_conditions_replay_that_leaves_processes_that_cannot_be_killed_is_the_solvers_zero(verifier):
    _conditioned(verifier)
    extra = """
        kill, sweeps = launcher.kill_left, []

        def stubborn(also=None, window_s=launcher.KILL_WINDOW_S):
            if also is None:
                return kill(also, window_s)
            sweeps.append(also)
            return (2, 1) if len(sweeps) == 3 else (0, 0)  # the second condition's replay's sweep

        launcher.kill_left = stubborn
    """

    _, reward, report = verifier.run(extra=extra)

    assert (reward, report["reason"], report["detail"]) == (
        {"reward": 0},
        "solver_condition_failed",
        "processes the replay of condition sixty left could not be killed (1)",
    )
    assert report["conditions"]["ran"][1]["survivors"] == 1


def test_outputs_of_the_conditions_beyond_what_the_launcher_holds_are_the_solvers_zero(verifier):
    """Each condition's outputs count toward MAX_CONDITION_BYTES, here 20: fifty's 15 bytes fit, and sixty's, with
    them, do not."""

    _conditioned(verifier)

    _, reward, report = verifier.run(extra="launcher.MAX_CONDITION_BYTES = 20")

    assert (reward, report["reason"], report["detail"]) == (
        {"reward": 0},
        "solver_condition_failed",
        "the replay of condition sixty wrote 15 bytes of outputs, which with the 15 of the conditions before it are "
        "more than the launcher holds for the grader (20)",
    )
    assert [entry.get("bytes") for entry in report["conditions"]["ran"]] == [15, 15]


def test_a_condition_whose_time_leaves_none_to_put_the_directory_back_is_the_solvers_zero(verifier):
    _conditioned(verifier)
    extra = """
        run = launcher._run_replay

        def late(config, paths, ops, place, log, deadline, replays, case=None):
            if case is not None and case.kind == "condition":
                place.put_back = 1e6
            return run(config, paths, ops, place, log, deadline, replays, case)

        launcher._run_replay = late
    """

    _, reward, report = verifier.run(extra=extra)

    assert (reward, report["reason"], report["detail"]) == (
        {"reward": 0},
        "solver_condition_failed",
        "putting the working directory back after the replay of condition fifty leaves it no time",
    )


def test_putting_the_directory_back_past_a_conditions_time_is_the_solvers_zero(verifier):
    _conditioned(verifier)
    extra = f"""
        import time
        clone, clones = launcher._clone_tree, []

        def slow(source, target, deadline=None, limit=None, device=None):
            if target == {str(verifier.workdir)!r}:
                clones.append(target)
                if len(clones) == 3:  # after the second condition's replay, the replay's and the first's before it
                    launcher._in_time(time.monotonic() - 1)
            return clone(source, target, deadline, limit, device)

        launcher._clone_tree = slow
    """

    _, reward, report = verifier.run(extra=extra)

    assert (reward, report["reason"], report["detail"]) == (
        {"reward": 0},
        "solver_condition_failed",
        "putting the working directory back outlived the replay of condition sixty's time",
    )


def test_the_replays_share_the_replays_time_with_the_conditions(verifier):
    """The replay and two conditions share half of the time the launcher's steps have (108 s of 120), each an equal
    part of what is left; with replay_timeout_s, each takes at most that."""

    _conditioned(verifier)

    report = verifier.run()[2]

    first, second = (entry["timeout_s"] for entry in report["conditions"]["ran"])
    assert 17.0 < report["replay"]["timeout_s"] <= 18.0 and 26.0 < first <= 27.0 and 52.0 < second <= 54.0, report

    timed = Verifier(verifier.root / "timed")
    _conditioned(timed)
    timed.config["replay_timeout_s"] = 3
    report = timed.run()[2]
    times = [report["replay"]["timeout_s"]] + [entry["timeout_s"] for entry in report["conditions"]["ran"]]
    assert all(2.5 < each <= 3.0 for each in times) and report["grader"]["timeout_s"] > 95.0, times


def test_replays_with_a_time_of_their_own_count_the_conditions(monkeypatch, verifier):
    monkeypatch.setattr(launcher.time, "monotonic", lambda: 1000.0)
    replays = launcher.Replays.of({"replay_timeout_s": 3}, 995.0, 1100.0, variants=1, conditions=2)
    assert replays.count == 4 and replays.end == 1007.0

    _conditioned(verifier)
    verifier.config["replay_timeout_s"] = 36

    _, reward, report = verifier.run()

    assert reward == {"invalid": 1, "failure_kind": "verifier_trusted_launcher"}
    assert "replay_timeout_s (3 replays of 36 s leave the grader none of the 108 s" in report["detail"]


@pytest.mark.parametrize(
    ("conditions", "problem"),
    [
        ({}, "they are not a list of 1 to 32"),
        ([], "they are not a list of 1 to 32"),
        ([{"id": f"c{n}", "inputs": f"tests/gen/c{n}"} for n in range(33)], "they are not a list of 1 to 32"),
        (["tests/gen/c"], "condition 0 is not an object of id, inputs"),
        ([{"id": "c", "inputs": "tests/gen/c", "outputs": []}], "condition 0 is not an object of id, inputs"),
        ([{"id": "../c", "inputs": "tests/gen/c"}], "condition 0 has no id of its own"),
        ([{"id": ".c", "inputs": "tests/gen/c"}], "condition 0 has no id of its own"),
        ([{"id": "c", "inputs": "tests/gen/c"}, {"id": "c", "inputs": "tests/gen/d"}], "condition 1 has no id of"),
        ([{"id": "c", "inputs": "tests/c"}], "condition c names its inputs outside tests/gen/"),
        ([{"id": "c", "inputs": "tests/gen/../c"}], "condition c names its inputs outside tests/gen/"),
        (
            [{"id": "c", "inputs": "tests/gen/c"}, {"id": "d", "inputs": "tests/gen/c/d"}],
            "the inputs of two conditions, tests/gen/c and tests/gen/c/d, lie one in the other",
        ),
    ],
    ids=[
        "no-list",
        "empty",
        "too-many",
        "no-object",
        "an-unknown-key",
        "a-path-for-an-id",
        "a-hidden-id",
        "an-id-twice",
        "inputs-outside-tests-gen",
        "inputs-not-normalized",
        "nested-inputs",
    ],
)
def test_conditions_the_launcher_cannot_run_are_settings_it_cannot_use(verifier, conditions, problem):
    _checked(verifier)
    verifier.config["replay"]["conditions"] = conditions

    _, reward, report = verifier.run()

    assert reward == {"invalid": 1, "failure_kind": "verifier_trusted_launcher"}
    assert f"hold no usable replay.conditions ({problem}" in report["detail"], report["detail"]


def _condition_inputs_a_file(verifier: Verifier) -> None:
    root = verifier.condition()
    shutil.rmtree(root)
    root.write_text("not a directory")


def _condition_inputs_a_link(verifier: Verifier) -> None:
    root = verifier.condition()
    shutil.rmtree(root)
    root.symlink_to(verifier.workdir)


@pytest.mark.parametrize(("owner", "prefix"), [("generated", "verifier_generated"), ("trusted", "verifier_trusted")])
@pytest.mark.parametrize(
    ("setup", "problem"),
    [
        (
            lambda v: shutil.rmtree(v.condition()),
            "condition fifty's inputs, tests/gen/conditions/fifty, cannot be read",
        ),
        (_condition_inputs_a_file, "condition fifty's inputs, tests/gen/conditions/fifty, are a file, not a directory"),
        (_condition_inputs_a_link, "are a symbolic link, not a directory"),
        (
            lambda v: (v.condition() / "notes").symlink_to("/etc"),
            "condition fifty's inputs cannot be laid over the working inputs: notes is a symbolic link",
        ),
        (
            lambda v: v.condition(inputs={"solve.py": "print(1)"}),
            "condition fifty's inputs cannot be laid over the working inputs: solve.py would change solve.py",
        ),
    ],
    ids=["missing", "a-file", "a-link", "a-link-inside", "over-the-submission"],
)
def test_condition_inputs_the_launcher_cannot_use_are_the_protocol_fault_of_the_graders_owner(
    verifier, owner, prefix, setup, problem
):
    """A condition's inputs lie in tests/gen, the grader's owner's: inputs the launcher cannot lay over the working
    directory make no measurement.  The missing and unusable directories are found before anything replays."""

    _checked(verifier)
    verifier.config["grader_owner"] = owner
    setup(verifier)

    _, reward, report = verifier.run()

    assert reward == {"invalid": 1, "failure_kind": f"{prefix}_protocol"}, report
    assert problem in report["detail"], report["detail"]
    assert "grader" not in report


@pytest.mark.parametrize(
    ("setup", "problem"),
    [
        (
            lambda v: (v.workdir / "notes.txt").mkdir(),
            "notes.txt is a file of the condition's and a directory of the working directory's",
        ),
        (
            lambda v: (v.workdir / "sub").symlink_to("/etc"),
            "sub is a directory of the condition's and a file of the working directory's",
        ),
    ],
    ids=["a-directory-for-its-file", "a-link-for-its-directory"],
)
def test_condition_inputs_the_working_directory_the_submission_left_differs_from_are_the_solvers_zero(
    verifier, setup, problem
):
    """materialize checked the condition's files against the task's environment/, which holds each: where the working
    directory differs there, the submission changed it, and the grade is the solver's 0, never the owner's fault that
    would drop it from the scores."""

    _checked(verifier)
    verifier.condition(inputs={"data.txt": "50", "notes.txt": "x", "sub/more.txt": "1"})
    setup(verifier)

    _, reward, report = verifier.run()

    assert (reward, report["reason"]) == ({"reward": 0}, launcher.CONDITION_FAILED), report
    assert report["detail"] == (
        f"condition fifty's inputs cannot be laid over the working inputs: {problem}, which the submission changed"
    )
    assert "grader" not in report


_FULL_CONDITION = {
    "inputs": """
import errno
lay = launcher._lay_over

def full(source, target, config, kind="variant"):
    if kind == "condition":
        raise OSError(errno.ENOSPC, "No space left on device")
    return lay(source, target, config, kind)

launcher._lay_over = full
""",
    "manifest": """
import errno
write = launcher._write

def full(path, text):
    if path.endswith("manifest.json"):
        raise OSError(errno.EDQUOT, "Disk quota exceeded")
    return write(path, text)

launcher._write = full
""",
}


@pytest.mark.parametrize("left", [True, False], ids=["the-replays-left-files", "nothing-left"])
@pytest.mark.parametrize("where", list(_FULL_CONDITION))
def test_a_full_filesystem_for_a_condition_is_the_solvers_zero_only_once_the_replays_left_files(verifier, left, where):
    """I2 for conditions: a condition's inputs, or the manifest of what the conditions wrote, that find their
    filesystem full once the replays' user left files where every user may write are the solver's 0; with nothing
    left, the trusted side's fault, as a full filesystem is before any replay ran."""

    shm = verifier.temporary[2]
    _conditioned(verifier, SOLVE + (f"\nopen({str(shm / 'left')!r}, 'w').write('x')\n" if left else ""))

    _, reward, report = verifier.run(extra=_FULL_CONDITION[where])

    if left:
        assert (reward, report["reason"]) == ({"reward": 0}, "solver_condition_failed"), report
        assert ("finds no room" if where == "inputs" else "finds no room where the launcher holds it") in report[
            "detail"
        ]
    else:
        assert reward == {"invalid": 1, "failure_kind": "verifier_trusted_launcher"}, report
    assert "grader" not in report


def test_each_grade_replays_the_conditions_again_and_a_regrade_compares_what_they_wrote(tmp_path):
    """Two grades of a submission deterministic on the working inputs and random on the conditions: the same
    replayed_sha256, other conditions_sha256, which a regrade tells apart from a flaky grader."""

    solve = SOLVE + _in_condition(
        "50", "json.dump({'answer': 100, 'draw': __import__('random').random()}, open('answer.json', 'w'))"
    )
    digests = []
    for grade in ("first", "regrade"):
        verifier = Verifier(tmp_path / grade)
        _conditioned(verifier, solve)
        report = verifier.run()[2]
        digests.append((report[launcher.REPLAYED_SHA256], report[launcher.CONDITIONS_SHA256]))

    assert digests[0][0] == digests[1][0] == SOLVED_SHA256 and digests[0][1] != digests[1][1]


def test_conditions_sha256_names_each_output_by_its_condition(tmp_path):
    """The digest covers each ``<id>/<output>``, sorted, by its own rule: the same bytes under other ids give another,
    and it is no replayed_sha256 of the same files."""

    for name, text in (("a", "1"), ("b", "2")):
        (tmp_path / name).mkdir()
        (tmp_path / name / "answer.json").write_text(text)
    names = ["a/answer.json", "b/answer.json"]

    digest = launcher.outputs_sha256(str(tmp_path), names, rule=launcher.CONDITION_OUTPUTS_RULE)

    expected = hashlib.sha256(launcher.CONDITION_OUTPUTS_RULE.encode())
    for name in names:
        expected.update(
            b"\0" + name.encode() + b"\0" + hashlib.sha256((tmp_path / name).read_bytes()).hexdigest().encode()
        )
    assert digest == expected.hexdigest()
    assert digest == launcher.outputs_sha256(str(tmp_path), names[::-1], rule=launcher.CONDITION_OUTPUTS_RULE)
    assert digest != launcher.outputs_sha256(str(tmp_path), names)
    (tmp_path / "a" / "answer.json").write_text("2")
    (tmp_path / "b" / "answer.json").write_text("1")
    assert digest != launcher.outputs_sha256(str(tmp_path), names, rule=launcher.CONDITION_OUTPUTS_RULE)


def test_a_report_too_large_for_its_bound_leaves_out_the_conditions_records_and_keeps_their_sha256():
    variants = [{"name": f"v{n}", "passed": True} for n in range(100)]
    report = {
        "schema": launcher.SCHEMA,
        "replay_checks": {"variants": variants, "forbidden_literals": {"scan": launcher.LITERAL_SCAN}},
        "conditions": {"ran": [{"id": f"c{n}", "passed": True, "output_tail": "t" * 50} for n in range(32)]},
        launcher.REPLAYED_SHA256: "ab" * 32,
        launcher.CONDITIONS_SHA256: "cd" * 32,
        "reward": 1,
        "failure_kind": None,
        "reason": None,
        "detail": "d" * 5000,
        "seconds": 1.0,
    }
    without = {**report, "replay_checks": {"forbidden_literals": {"scan": launcher.LITERAL_SCAN}}, "conditions": {}}
    limit = len(json.dumps({**without, "truncated": ["replay_checks.variants", "conditions.ran"]}, sort_keys=True))

    cut = json.loads(launcher.bounded_report(report, limit=limit))
    assert cut["truncated"] == ["replay_checks.variants", "conditions.ran"] and cut["conditions"] == {}
    least = json.loads(launcher.bounded_report(report, limit=200))
    assert least["truncated"] == "all but the outcome"
    assert (least[launcher.REPLAYED_SHA256], least[launcher.CONDITIONS_SHA256]) == ("ab" * 32, "cd" * 32)


# The kernel's keyrings: add_key and keyctl stand-ins for cleared_keys, each a list of results by call.
def _fake_keys(clears: list[tuple[int, int]], persistent=(5, 0), probe=(9, 0)):
    calls: list[Any] = []
    pending = list(clears)

    def keyctl(*args):
        calls.append(args)
        if args[0] == launcher._KEYCTL_GET_PERSISTENT:
            return persistent
        if args[0] == launcher._KEYCTL_INVALIDATE:
            return 0, 0
        return pending.pop(0) if pending else (0, 0)

    def add_key(*args):
        calls.append(("add_key", *args))
        return probe

    return add_key, keyctl, calls


def test_cleared_keys_clears_the_user_the_user_session_and_the_persistent_keyring():
    """Then it adds a key to a keyring new to its process, which no replay reached, and invalidates it: the user may
    make keys, and has the room for them."""

    add_key, keyctl, calls = _fake_keys([])

    assert launcher.cleared_keys(add_key, keyctl) == "ok"
    assert calls == [
        (launcher._KEYCTL_CLEAR, launcher._KEY_SPEC_USER_KEYRING),
        (launcher._KEYCTL_CLEAR, launcher._KEY_SPEC_USER_SESSION_KEYRING),
        (launcher._KEYCTL_GET_PERSISTENT, -1, launcher._KEY_SPEC_PROCESS_KEYRING),
        (launcher._KEYCTL_CLEAR, 5),
        ("add_key", b"user", b"platoon-launcher-probe", b"x", 1, launcher._KEY_SPEC_PROCESS_KEYRING),
        (launcher._KEYCTL_INVALIDATE, 9),
    ]


@pytest.mark.parametrize(
    ("clears", "probed"),
    [
        ([(-1, errno.ENOSYS)], errno.ENOSYS),
        ([(-1, errno.EPERM)], errno.EPERM),
        ([(-1, errno.EOPNOTSUPP)], errno.EOPNOTSUPP),
        ([(-1, errno.ENOSYS)], errno.EACCES),
        ([(-1, errno.EACCES)], errno.EACCES),
        ([], errno.EACCES),
    ],
    ids=["no-keys", "docker-seccomp", "unsupported", "gvisor", "older-gvisor", "clears-but-adds-none"],
)
def test_cleared_keys_finds_no_channel_where_the_user_may_make_no_keys(clears, probed):
    """A kernel without keys, or a sandbox that keeps the replay's user from them (Docker's default seccomp profile:
    EPERM; gVisor: EACCES from add_key, ENOSYS from most of keyctl, or EACCES from all of it before), leaves nothing to
    clear: a key added to a keyring new to the child, as that user, fails for any reason but a full quota, so the
    replay could make none either.  Any errors of the clearing go with it: an honest replay never scores 0 there."""

    add_key, keyctl, _ = _fake_keys(clears, probe=(-1, probed))

    assert launcher.cleared_keys(add_key, keyctl) == "absent"
    assert launcher.keys_verdict("absent") is None
    add_key, keyctl, _ = _fake_keys([], persistent=(-1, errno.EOPNOTSUPP))
    assert launcher.cleared_keys(add_key, keyctl) == "ok"  # a kernel without persistent keyrings


@pytest.mark.parametrize(
    ("clears", "probe", "found"),
    [
        ([(-1, errno.EACCES)], (9, 0), f"left user {errno.EACCES}"),
        ([(0, 0), (-1, errno.EKEYREVOKED)], (9, 0), f"left user-session {errno.EKEYREVOKED}"),
        ([(-1, errno.EDQUOT)], (9, 0), f"left user {errno.EDQUOT}"),
        ([], (-1, errno.EDQUOT), f"left quota {errno.EDQUOT}"),
        ([(-1, errno.EACCES)], (-1, errno.EDQUOT), f"left quota {errno.EDQUOT}"),
        ([(-1, errno.EIO)], (9, 0), f"fault user {errno.EIO}"),
    ],
    ids=["closed-to-its-user", "revoked", "a-full-keyring", "keys-fill-its-quota", "closed-and-full", "the-sandboxs"],
)
def test_a_keyring_that_cannot_be_cleared_is_left_over_or_a_trusted_fault(clears, probe, found):
    """A keyring the replay closed to its own user (KEYCTL_SETPERM) or revoked, where that user may make keys, is what
    the replay left: Leftover, the solver's 0; so are keys that fill the user's quota once its keyrings were cleared,
    which would keep the next replay from starting (a launcher fault there).  One that cannot be cleared otherwise is
    the sandbox's."""

    add_key, keyctl, _ = _fake_keys(clears, probe=probe)

    assert launcher.cleared_keys(add_key, keyctl) == found
    with pytest.raises(launcher.Leftover if found.startswith("left") else launcher.LauncherFault):
        launcher.keys_verdict(found)


def test_keys_verdict_fails_closed_on_what_it_cannot_read():
    for found in ("", "user 1", "nonsense here now", "ok extra"):
        with pytest.raises(launcher.LauncherFault):
            launcher.keys_verdict(found)


def test_the_launcher_knows_the_key_system_calls_of_the_machines_it_runs_on():
    add_key, keyctl = launcher._key_calls("x86_64")
    assert callable(add_key) and callable(keyctl)
    with pytest.raises(launcher.LauncherFault, match="no system call numbers of the kernel's keys on vax"):
        launcher._key_calls("vax")


# Run as root of a fresh user namespace, whose keyrings are its own (Linux 5.3 and later): a key in each keyring of
# uid 0 that outlives its processes, then the launcher's clearing as that user, then a search for each.
_KEYS_IN_A_NAMESPACE = """
import os, sys
sys.path.insert(0, {directory!r})
import launcher
os.setgroups = lambda groups: None  # an unprivileged user namespace denies setgroups; it has no groups to drop
add_key, keyctl = launcher._key_calls()
persistent, _ = keyctl(launcher._KEYCTL_GET_PERSISTENT, -1, launcher._KEY_SPEC_PROCESS_KEYRING)
rings = [launcher._KEY_SPEC_USER_KEYRING, launcher._KEY_SPEC_USER_SESSION_KEYRING, persistent]
made = [add_key(b"user", b"platoon-marker-%d" % n, b"x", 1, ring)[0] > 0 for n, ring in enumerate(rings)]
found = lambda: [keyctl(10, ring, b"user", b"platoon-marker-%d" % n, 0)[0] > 0 for n, ring in enumerate(rings)]
before = found()
{clear}
print(made, before, found())
"""


def _keys_work() -> bool:
    if not _namespaces_work("-r"):
        return False
    probe = "import ctypes, sys; libc = ctypes.CDLL(None); sys.exit(libc.syscall(248, b'user', b'p', b'x', 1, -2) < 0)"
    return (
        subprocess.run(["unshare", "-r", sys.executable, "-c", probe], capture_output=True, timeout=60).returncode == 0
    )


@pytest.mark.skipif(not _keys_work(), reason="keys in an unprivileged user namespace are unavailable")
@pytest.mark.parametrize("cleared", [True, False], ids=["cleared", "left"])
def test_the_keys_a_replays_user_leaves_in_its_keyrings_are_cleared_before_the_next_replay(cleared):
    """I2 for kernel keyrings: a key in the user keyring, the user session keyring or the persistent keyring of a uid
    outlives its processes, so another process of that uid finds it; clear_keys_as, as that uid, clears all three."""

    clear = "launcher.clear_keys_as(0)" if cleared else ""
    code = _KEYS_IN_A_NAMESPACE.format(directory=str(LAUNCHER.parent), clear=clear)

    done = subprocess.run(["unshare", "-r", sys.executable, "-c", code], capture_output=True, text=True, timeout=120)

    assert done.returncode == 0, done.stderr
    assert done.stdout.split("] [") == [
        "[True, True, True",
        "True, True, True",
        f"{not cleared}, " * 2 + f"{not cleared}]\n",
    ]


@pytest.mark.parametrize(
    ("joined", "probe", "fails"),
    [
        ((0, 0), None, False),
        ((-1, errno.ENOSYS), (-1, errno.EACCES), False),
        ((-1, errno.EPERM), (-1, errno.EPERM), False),
        ((-1, errno.EACCES), (9, 0), True),
        ((-1, errno.EDQUOT), (-1, errno.EDQUOT), True),
    ],
    ids=["joined", "gvisor", "docker-seccomp", "keys-reach-the-launchers", "a-full-quota"],
)
def test_the_replay_joins_a_session_keyring_of_its_own(monkeypatch, joined, probe, fails):
    """A replay keeps no session keyring of the launcher's, which every later replay would share: its demotion joins a
    new one, after it switched user.  Where the join fails, a key added to the session keyring it has tells: one that
    cannot be added for any reason but a full quota is a sandbox that keeps it from keys, no fault; else the replay
    does not start."""

    seen: list[Any] = []

    def keyctl(*args):
        seen.append(args)
        return joined if args[0] == launcher._KEYCTL_JOIN_SESSION_KEYRING else (0, 0)

    def add_key(*args):
        seen.append(("add_key", args[-1]))
        return probe

    monkeypatch.setattr(launcher, "_libc", lambda: None)
    monkeypatch.setattr(launcher, "_key_calls", lambda: (add_key, keyctl))
    for name in ("_oom_disfavor", "_limit_replay", "_prctl"):
        monkeypatch.setattr(launcher, name, lambda *a, **k: None)
    monkeypatch.setattr(os, "setgroups", lambda groups: None)
    monkeypatch.setattr(os, "setgid", lambda gid: None)
    monkeypatch.setattr(os, "setuid", lambda uid: seen.append("setuid"))
    monkeypatch.setattr(os, "getresuid", lambda: (launcher.REPLAY_UID,) * 3)
    monkeypatch.setattr(os, "getresgid", lambda: (launcher.REPLAY_UID,) * 3)
    monkeypatch.setattr(os, "getgroups", lambda: [])
    demote = launcher.Ops().demote(1 << 20)

    if fails:
        with pytest.raises(OSError, match="cannot join a session keyring of its own"):
            demote()
    else:
        demote()

    assert seen[:2] == ["setuid", (launcher._KEYCTL_JOIN_SESSION_KEYRING, 0)]
    assert seen[2:3] == ([] if probe is None else [("add_key", launcher._KEY_SPEC_SESSION_KEYRING)])


# Run as root of a fresh user namespace, in a session keyring of its own (the launcher's): one child adds a key to its
# session keyring, the next looks for it, each demoted as a replay's process joins its keyring, or not.
_SESSION_KEYS = """
import subprocess, sys
sys.path.insert(0, {directory!r})
import launcher
calls = launcher._key_calls()
write = "import ctypes; print(ctypes.CDLL(None).syscall({add_key}, b'user', b'platoon-marker', b'x', 1, -3) > 0)"
read = "import ctypes; print(ctypes.CDLL(None).syscall({keyctl}, 10, -3, b'user', b'platoon-marker', 0) > 0)"
calls[1](launcher._KEYCTL_JOIN_SESSION_KEYRING, 0)
pre = (lambda: launcher._join_new_session_keyring(calls)) if {join} else None
found = [subprocess.run([sys.executable, "-c", code], preexec_fn=pre, capture_output=True, text=True).stdout.strip()
         for code in (write, read)]
print(" ".join(found))
"""


@pytest.mark.skipif(not _keys_work(), reason="keys in an unprivileged user namespace are unavailable")
@pytest.mark.parametrize("join", [True, False], ids=["joined", "shared"])
def test_no_replay_finds_a_key_an_earlier_replay_left_in_its_session_keyring(join):
    """On the kernel itself: a replay's process that kept the launcher's session keyring would leave a key there that
    the next replay finds; one that joined its own leaves it in a keyring that goes with it."""

    add_key, keyctl = launcher._KEY_SYSCALLS[os.uname().machine]
    code = _SESSION_KEYS.format(directory=str(LAUNCHER.parent), add_key=add_key, keyctl=keyctl, join=join)

    done = subprocess.run(["unshare", "-r", sys.executable, "-c", code], capture_output=True, text=True, timeout=120)

    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["True", str(not join)]


def test_every_replay_clears_its_users_keyrings_after_it(verifier):
    """The keyrings are cleared after each replay of a grade, the replay's, each variant's and each condition's, as
    what it left where every user may write is; one that cannot be cleared is the solver's 0."""

    record = verifier.root / "keys.txt"
    _conditioned(verifier)
    verifier.variant()
    extra = f"""
        class Keyed(WithoutRoot):
            def clear_keys(self):
                with open({str(record)!r}, "a") as stream:
                    stream.write("cleared\\n")
                if len(open({str(record)!r}).read().split()) == {{failing}}:
                    raise launcher.Leftover("its user keyring cannot be cleared: Permission denied")
    """

    _, reward, report = verifier.run(extra=extra.format(failing=0), ops="Keyed()")

    assert reward == {"reward": 1}, report
    assert record.read_text().split() == ["cleared"] * 4
    record.unlink()
    _, reward, report = Verifier.run(verifier, extra=extra.format(failing=3), ops="Keyed()")
    assert (reward, report["reason"]) == ({"reward": 0}, "solver_condition_failed"), report
    assert report["detail"] == (
        "the replay of condition fifty left what the launcher cannot remove in its time: its user keyring cannot be "
        "cleared: Permission denied"
    )
