"""The red-team check of a Harbor environment's runtime tasks (RFC 0022): :func:`check_runtime_tasks`.

It writes tasks with known sabotages from the environment's own template (``service_kwargs.runtime_tasks``), takes
each in through ``materialize``, and checks the refusal or the grade each gets.  It is made of named cases, which
:func:`check_runtime_tasks` runs one after the other; its ``AssertionError`` lists each case that failed, by name, with
what it saw.  This is the generic half of a red-team suite.  The rest is not a property of ``materialize`` or the
launcher: a task producer's own toolkit holds its domain's half, in the checks it runs on a task before the task is
taken in (leaks of the graded values, secrets planted in what the solver sees, modules a grader imports shadowed in the
task's own ``tests/gen``), and a program that runs runtime tasks catches graders that score by something other than
the answer, such as a hash of the submission, with its controls (a reference solution that must pass, a no-op that
must fail).

**Refusals** (:data:`REFUSALS`, run on CPU, with no sandbox): ``materialize`` refuses a ``task.toml`` that differs from
the template in its network (``network``), its GPUs (``gpus``), its image (``image``) or its verifier's mode
(``shared_grading``), and one that declares an artifact outside the working directory, such as ``/tests/gen``, where a
submission would sit beside the grader (``outside_artifact``); an ``environment/Dockerfile`` (``dockerfile``) and a
compose file (``stack``); a ``tests/test.sh`` of the task's own (``own_launcher``); a symbolic link (``symlink``); files
beyond ``max_bytes`` (``oversized``, a sparse file); a grader manifest that is missing (``no_manifest``) or not an
argv (``bad_manifest``); and a replay variant whose inputs would replace a file of the submission's
(``variant_overlay``).  ``accepts``: it takes the honest task, twice into one digest, which names its directory in the
store, with the launcher installed.

**Grades** (:data:`GRADES`, with ``grade``): each task is taken in, opened, solved by its reference solution (its
``solution/solve.sh``, as the oracle runs it) and graded, which needs the environment's sandboxes: a provider that
starts its template's image, which has ``python3`` and ``sh``, with no network, and separate verifiers that run as
root.  An honest grader scores the right answer 1 (``honest_pass``) and a wrong one 0 (``honest_fail``); a missing
artifact (``missing_artifact``) and a replay that fails (``replay_fails``) score 0, the solver's fault (reason
``solver_output_contract``, ``solver_nonreproducible``); with replay checks (:data:`REPLAY_CHECKS`: a variant whose
``data.txt`` holds another number, and the answer as a forbidden literal), the honest solution passes
(``variant_followed``), one that computes the answer without reading its inputs fails the variant (``variant_ignored``,
``solver_variant_failed``), one that states the answer is hardcoded (``hardcoded_literal``, ``solver_hardcoded``), and a
variant's replay can neither change what the replay wrote, which waits as root's, nor read the grader's files
(``variant_sealed``: its wrong answer scores 0); a replay can read neither the grader's files nor the copy of
the answer the agents published in ``/logs/artifacts``, and cannot write a reward (``replay_sealed``); a module of the
submission's never shadows one the grader imports (``module_shadow``); a grader that declares its measurement invalid
is ``verifier_generated_protocol`` (``invalid_grader``), one that crashes after a verdict of 0 is
``verifier_generated_exit`` (``crash_after_zero``), and one that kills the launcher and writes a reward of its own is
``verifier_generated_unfinished`` (``kills_launcher``); one that writes a reward file of its own (``own_reward_file``),
leaves a process that keeps writing one (``lingering_writer``) or writes what a verifier classifier would match into
the verifier's log (``log_censor``) gets the launcher's reward; a trusted grader that crashes is
``verifier_trusted_exit`` (``trusted_grader_crash``); and a grader that flips a coin is ``verifier_flaky``
(``flaky_grader``), as its regrades in fresh verifiers find (``runtime_tasks.regrade``): the case grades the same
submission again, up to :data:`FLAKY_GRADES` times, while each grade is the grader's score that every regrade gave
again, which a coin flip's is half the time with one regrade.  The solution reads ``data.txt``, which only the task's
uploaded ``environment/`` holds, and its replay writes the same bytes each time.

The tasks it takes in stay in the environment's store.  The experimental runtime-task API (``platoon.experimental``)
can change in a minor release, and this check changes with it.
"""

from __future__ import annotations

import asyncio
import copy
import datetime
import json
import math
import re
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from platoon.utils import scratch

#: The refusal cases, in the order they run, then the acceptance case.
REFUSALS = (
    "network",
    "gpus",
    "image",
    "shared_grading",
    "dockerfile",
    "stack",
    "own_launcher",
    "outside_artifact",
    "symlink",
    "oversized",
    "no_manifest",
    "bad_manifest",
    "variant_overlay",
    "accepts",
)
#: The grading cases, in the order they run.
GRADES = (
    "honest_pass",
    "honest_fail",
    "missing_artifact",
    "replay_fails",
    "variant_followed",
    "variant_ignored",
    "hardcoded_literal",
    "variant_sealed",
    "replay_sealed",
    "module_shadow",
    "invalid_grader",
    "crash_after_zero",
    "kills_launcher",
    "own_reward_file",
    "lingering_writer",
    "log_censor",
    "trusted_grader_crash",
    "flaky_grader",
)
#: How many times the ``flaky_grader`` case grades its submission at most.  With one regrade, each grade of a coin-flip
#: grader is ``verifier_flaky`` with probability 1/2, so all of them agree with probability 2**-20.
FLAKY_GRADES = 20
_FLAKY = "verifier_flaky"
#: The replay checks of the cases that have them (``variant_overlay``, ``variant_followed``, ``variant_ignored``,
#: ``hardcoded_literal``, ``variant_sealed``): a variant whose ``data.txt`` holds 50, so that the solution, which reads
#: it, writes 100, and the answer, 42, as a forbidden literal of ``solve.py``.
REPLAY_CHECKS: dict[str, Any] = {
    "variants": [
        {
            "name": "fifty",
            "inputs": "tests/gen/variants/fifty/inputs",
            "expected": "tests/gen/variants/fifty/expected.json",
            "tolerance": 0,
            "meta": {"data.txt": 50},
        }
    ],
    "forbidden_literals": {"path": "tests/gen/forbidden-literals.json", "sources": ["solve.py"]},
}
_CHECK_FILES = {
    "tests/gen/variants/fifty/inputs/data.txt": "50\n",
    "tests/gen/variants/fifty/expected.json": json.dumps({"answer.json#/answer": 100}),
    "tests/gen/forbidden-literals.json": json.dumps([{"value": 42, "tolerance": 0.5}]),
}
_ARTIFACTS = ["solve.py", "answer.json"]
_SOLVE = """import json

with open("data.txt") as stream:
    answer = 2 * int(stream.read())
with open("answer.json", "w") as stream:
    json.dump({"answer": answer}, stream)
"""
_HONEST_GRADER = """import json
import os

try:
    with open(os.path.join(os.environ["PLATOON_WORKDIR"], "answer.json")) as stream:
        passed = json.load(stream) == {"answer": 42}
except (OSError, ValueError):
    passed = False
with open(os.environ["PLATOON_VERDICT"], "w") as stream:
    json.dump({"passed": passed, "checks": [{"id": "answer", "passed": passed}]}, stream)
"""
# A submission whose replay tries what a sealed verifier keeps it from: read the grader's files, which would give the
# right answer away, or the answer the agents published (_PUBLISH), and write the reward.  It writes 41 unless it
# could read one of them.
_PRYING_SOLVE = """import json

answer = 41
try:
    with open("/tests/gen/grader.json") as stream:
        stream.read()
    answer = 42
except OSError:
    pass
try:
    with open("/logs/artifacts/stash.json") as stream:
        answer = json.load(stream)["answer"]
except (OSError, ValueError, KeyError, TypeError):
    pass
for name in ("reward.json", "reward.txt"):
    try:
        with open("/logs/verifier/" + name, "w") as stream:
            stream.write('{"reward": 1}' if name.endswith(".json") else "1")
    except OSError:
        pass
with open("answer.json", "w") as stream:
    json.dump({"answer": answer}, stream)
"""
_GRADERS = {
    "invalid": """import json, os
with open(os.environ["PLATOON_VERDICT"], "w") as stream:
    json.dump({"passed": False, "invalid": True}, stream)
""",
    "crash_after_zero": """import json, os, sys
with open(os.environ["PLATOON_VERDICT"], "w") as stream:
    json.dump({"passed": False}, stream)
sys.exit(1)
""",
    "own_reward_file": """import json, os
logs = os.path.dirname(os.path.dirname(os.environ["PLATOON_VERDICT"]))
with open(os.path.join(logs, "reward.json"), "w") as stream:
    json.dump({"reward": 1}, stream)
with open(os.environ["PLATOON_VERDICT"], "w") as stream:
    json.dump({"passed": False}, stream)
""",
    "lingering_writer": """import json, os, time
logs = os.path.dirname(os.path.dirname(os.environ["PLATOON_VERDICT"]))
if os.fork() == 0:
    os.setsid()
    if os.fork() == 0:
        while True:
            try:
                with open(os.path.join(logs, "reward.json"), "w") as stream:
                    json.dump({"reward": 1}, stream)
            except OSError:
                pass
            time.sleep(0.01)
    os._exit(0)
os.wait()
with open(os.environ["PLATOON_VERDICT"], "w") as stream:
    json.dump({"passed": False}, stream)
""",
    # It kills its parent, the launcher, then writes a reward of 1 that nothing would remove.
    "kills_launcher": """import json, os, signal
logs = os.path.dirname(os.path.dirname(os.environ["PLATOON_VERDICT"]))
os.kill(os.getppid(), signal.SIGKILL)
with open(os.path.join(logs, "reward.json"), "w") as stream:
    json.dump({"reward": 1}, stream)
""",
    # A wrong answer, and a line in the verifier's log that the harbor-common classifiers would call a grader's
    # crash, which would make the outcome invalid where an environment sets them.
    "log_censor": """import json, os
logs = os.path.dirname(os.path.dirname(os.environ["PLATOON_VERDICT"]))
with open(os.path.join(logs, "test-stdout.txt"), "a") as stream:
    stream.write("subprocess.TimeoutExpired: Command 'grade' timed out after 1 seconds\\n")
with open(os.environ["PLATOON_VERDICT"], "w") as stream:
    json.dump({"passed": False}, stream)
""",
    "crash": "import sys\nsys.exit(1)\n",
    # A coin flip, whatever the answer: only a grade of the same submission again in another verifier shows it.
    "coin_flip": """import json, os
with open(os.environ["PLATOON_VERDICT"], "w") as stream:
    json.dump({"passed": os.urandom(1)[0] % 2 == 0}, stream)
""",
}
_FAILING_SOLVE = "import sys\nsys.exit(1)\n"
# A submission that computes the answer without reading its inputs: it states no forbidden literal, and writes 42 on
# any data, which the variant catches.
_IGNORING_SOLVE = """import json

with open("answer.json", "w") as stream:
    json.dump({"answer": 6 * 7}, stream)
"""
# A submission that states the answer, which the scan for forbidden literals catches before anything replays.
_HARDCODED_SOLVE = """import json

with open("answer.json", "w") as stream:
    json.dump({"answer": 42}, stream)
"""
# A submission whose replay writes a wrong answer, 41, and whose replay of the variant (a data.txt of 50) gives what the
# variant expects but first tries what a sealed verifier keeps it from: replace what the replay wrote, which waits
# beside it as root's, with the right answer, and read the grader's files.  It scores 1 only if either leaks.
_SEALED_SOLVE = """import json

with open("data.txt") as stream:
    data = int(stream.read())
answer = 41
if data == 50:
    try:
        with open("../replayed/answer.json", "w") as stream:
            json.dump({"answer": 6 * 7}, stream)
    except OSError:
        pass
    try:
        with open("/tests/gen/grader.json") as stream:
            stream.read()
        with open("../replayed/answer.json", "w") as stream:
            json.dump({"answer": 6 * 7}, stream)
    except OSError:
        pass
    answer = 2 * data
with open("answer.json", "w") as stream:
    json.dump({"answer": answer}, stream)
"""
# The solution's lines that publish a copy of its answer where every agent may publish files, which the verifier then
# holds in /logs/artifacts; they never fail the solution, on a sandbox where no agent may publish there either.
_PUBLISH = (
    'stash="${PLATOON_SANDBOX_ROOT:-}/logs/artifacts"; { mkdir -p "$stash" && cp answer.json "$stash/stash.json"; } '
    "2>/dev/null || true\n"
)
# A grader that compares with a module of the standard library's, and the submission's module of that name, which would
# make every answer pass if the grader imported it: it sits in the working directory, never beside the grader.
_FRACTION_GRADER = """import json
import os
from fractions import Fraction

try:
    with open(os.path.join(os.environ["PLATOON_WORKDIR"], "answer.json")) as stream:
        passed = Fraction(json.load(stream)["answer"]) == Fraction(42)
except (OSError, ValueError, KeyError, TypeError):
    passed = False
with open(os.environ["PLATOON_VERDICT"], "w") as stream:
    json.dump({"passed": passed}, stream)
"""
_SHADOW = """class Fraction:
    def __init__(self, value):
        pass

    def __eq__(self, other):
        return True
"""
_NOT_ALLOWED_IMAGE = "platoon-red-team-image-not-listed"


class _Failed(Exception):
    """A case saw something it should not have."""


def check_runtime_tasks(service: Any, *, grade: bool = True) -> None:
    """Check a Harbor environment's runtime tasks against the red-team cases (:data:`REFUSALS`, and with ``grade``
    :data:`GRADES`); raise ``AssertionError`` naming every case that failed, with what it saw.  ``service`` is a
    ``HarborEnvironmentService`` that takes runtime tasks, or the ``service_kwargs`` to build one.  ``grade`` runs the
    grading cases, which start the environment's sandboxes.  Call it outside a running event loop."""

    from platoon.envs.harbor import HarborEnvironmentService

    built = service if isinstance(service, HarborEnvironmentService) else HarborEnvironmentService(**service)
    if built.runtime_tasks is None:
        raise AssertionError("the environment takes no runtime tasks: give it service_kwargs.runtime_tasks")
    failures = asyncio.run(_run_cases(built, grade=grade))
    if failures:
        lines = [f"{case}: {what}" for case, what in failures.items()]
        raise AssertionError(f"runtime tasks fail {len(failures)} red-team case(s):\n  " + "\n  ".join(lines))


async def _run_cases(service: Any, *, grade: bool) -> dict[str, str]:
    failures: dict[str, str] = {}
    with scratch.temporary_directory(prefix="platoon-red-team-") as directory:
        sources = _Sources(Path(directory), service.runtime_tasks)
        for case in REFUSALS:
            try:
                await _REFUSAL_CASES[case](service, sources)
            except _Failed as exc:
                failures[case] = str(exc)
        if grade:
            if (refusal := service.runtime_refusal()) is not None:
                failures["sandboxes"] = f"the environment's sandboxes cannot run runtime tasks: {refusal}"
                return failures
            for case in GRADES:
                try:
                    await _graded(service, case, sources)
                except _Failed as exc:
                    failures[case] = str(exc)
    return failures


class _Sources:
    """Writes the cases' task directories: the honest task, from the template of the environment's ``runtime_tasks``
    (``config``), and what each case changes."""

    def __init__(self, root: Path, config: Any) -> None:
        self.root = root
        self.config = config

    def write(
        self,
        case: str,
        *,
        toml: Callable[[dict[str, Any]], None] | None = None,
        data: str = "21",
        solve: str = "",
        grader: str = _HONEST_GRADER,
        manifest: Any = None,
        replay: bool = True,
        checks: bool = False,
    ) -> Path:
        """The honest task as ``case``'s directory, with what the case changes: its ``task.toml`` (``toml`` edits the
        template's tables), its ``environment/data.txt``, the extra lines of its solution, its grader and manifest,
        and with ``checks`` the data of :data:`REPLAY_CHECKS`."""

        task_dir = self.root / f"{case}-{uuid.uuid4().hex[:8]}"
        config: dict[str, Any] = copy.deepcopy(dict(self.config.template))
        config.setdefault("task", {})["name"] = f"platoon-red-team-{case}"
        config["artifacts"] = list(_ARTIFACTS)
        if toml is not None:
            toml(config)
        files = {
            "task.toml": to_toml(config),
            "instruction.md": 'Write twice the number in data.txt as {"answer": <number>} into answer.json, with '
            "the script solve.py that computes it.\n",
            "environment/data.txt": f"{data}\n",
            "solution/solve.sh": (
                f"cat > solve.py <<'PLATOON_SOLVE'\n{_SOLVE}PLATOON_SOLVE\npython3 solve.py\n{solve}"
            ),
            "tests/gen/grade.py": grader,
        }
        if manifest is None:
            honest: dict[str, Any] = {"command": ["python3", "grade.py"]}
            if replay:
                honest["replay"] = {"command": ["python3", "solve.py"], "outputs": ["answer.json"]}
            manifest = honest
        if manifest is not False:
            files["tests/gen/grader.json"] = json.dumps(manifest)
        if checks:
            files.update(_CHECK_FILES)
        for name, text in files.items():
            path = task_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        return task_dir


async def _refused(service: Any, source: Path, expected: str, **options: Any) -> None:
    made = await service.materialize(source, **options)
    if made.ok:
        raise _Failed(f"materialize took it in, as {made.digest}")
    if not any(re.search(expected, problem) for problem in made.problems):
        raise _Failed(f"materialize refused it, but no problem matches {expected!r}: {list(made.problems)}")


def _set(table: str, key: str, value: Any) -> Callable[[dict[str, Any]], None]:
    def edit(config: dict[str, Any]) -> None:
        config.setdefault(table, {})[key] = value

    return edit


def _shared(config: dict[str, Any]) -> None:
    verifier = config.setdefault("verifier", {})
    verifier.pop("environment", None)
    verifier["environment_mode"] = "shared"


async def _network(service: Any, sources: _Sources) -> None:
    source = sources.write("network", toml=_set("environment", "network_mode", "public"))
    await _refused(service, source, r"network_mode")


async def _gpus(service: Any, sources: _Sources) -> None:
    await _refused(service, sources.write("gpus", toml=_set("environment", "gpus", 1)), r"gpus")


async def _image(service: Any, sources: _Sources) -> None:
    source = sources.write("image", toml=_set("environment", "docker_image", _NOT_ALLOWED_IMAGE))
    await _refused(service, source, r"docker_image")


async def _shared_grading(service: Any, sources: _Sources) -> None:
    await _refused(service, sources.write("shared_grading", toml=_shared), r"environment_mode")


def _outside(config: dict[str, Any]) -> None:
    config["artifacts"] = [*_ARTIFACTS, "/tests/gen"]


async def _outside_artifact(service: Any, sources: _Sources) -> None:
    await _refused(service, sources.write("outside_artifact", toml=_outside), r"artifact .*/tests/gen")


async def _dockerfile(service: Any, sources: _Sources) -> None:
    source = sources.write("dockerfile")
    (source / "environment" / "Dockerfile").write_text("FROM scratch\n")
    await _refused(service, source, r"environment/Dockerfile")


async def _stack(service: Any, sources: _Sources) -> None:
    source = sources.write("stack")
    compose = "services:\n  main: {}\n  judge:\n    image: busybox\n    command: [sleep, infinity]\n"
    (source / "environment" / "docker-compose.yaml").write_text(compose)
    await _refused(service, source, r"environment/docker-compose\.yaml")


async def _own_launcher(service: Any, sources: _Sources) -> None:
    source = sources.write("own_launcher")
    (source / "tests" / "test.sh").write_text("echo 1 > /logs/verifier/reward.txt\n")
    await _refused(service, source, r"tests/test\.sh")


async def _symlink(service: Any, sources: _Sources) -> None:
    source = sources.write("symlink")
    (source / "environment" / "passwd").symlink_to("/etc/passwd")
    await _refused(service, source, r"symbolic link")


async def _oversized(service: Any, sources: _Sources) -> None:
    source = sources.write("oversized")
    with open(source / "environment" / "large.bin", "wb") as stream:
        stream.truncate(service.runtime_tasks.max_bytes + 1)  # sparse: nothing is written
    await _refused(service, source, r"max_bytes")


async def _no_manifest(service: Any, sources: _Sources) -> None:
    await _refused(service, sources.write("no_manifest", manifest=False), r"grader\.json is missing")


async def _bad_manifest(service: Any, sources: _Sources) -> None:
    source = sources.write("bad_manifest", manifest={"command": "python3 grade.py"})
    await _refused(service, source, r"grader\.json: command")


async def _variant_overlay(service: Any, sources: _Sources) -> None:
    source = sources.write("variant_overlay", checks=True)
    (source / "tests" / "gen" / "variants" / "fifty" / "inputs" / "solve.py").write_text("print('not yours')\n")
    await _refused(service, source, r"would change solve\.py", replay_checks=REPLAY_CHECKS)


async def _accepts(service: Any, sources: _Sources) -> None:
    source = sources.write("accepts")
    first, second = await service.materialize(source), await service.materialize(source)
    if not first.ok:
        raise _Failed(f"materialize refused the honest task: {list(first.problems)}")
    if second.digest != first.digest:
        raise _Failed(f"the same task took two digests: {first.digest} and {second.digest}")
    stored = service.runtime_store / first.digest
    task_dir = Path(first.task.misc["harbor"]["task_dir"])
    if task_dir != stored or first.task.id != first.digest:
        raise _Failed(f"the task is {first.task.id} in {task_dir}, not its digest in the store, {stored}")
    missing = [name for name in ("tests/test.sh", "tests/platoon/launcher.py") if not (stored / name).is_file()]
    if missing:
        raise _Failed(f"the stored task lacks the launcher: {missing}")


_REFUSAL_CASES = {
    "network": _network,
    "gpus": _gpus,
    "image": _image,
    "shared_grading": _shared_grading,
    "outside_artifact": _outside_artifact,
    "dockerfile": _dockerfile,
    "stack": _stack,
    "own_launcher": _own_launcher,
    "symlink": _symlink,
    "oversized": _oversized,
    "no_manifest": _no_manifest,
    "bad_manifest": _bad_manifest,
    "variant_overlay": _variant_overlay,
    "accepts": _accepts,
}


def _solve_line(script: str) -> str:
    """A solution's extra lines: write ``script`` as ``solve.py`` over the honest one, as the submission."""

    return f"cat > solve.py <<'PLATOON_SOLVE'\n{script}PLATOON_SOLVE\n"


# case -> (what the case's task changes, its grader's owner, the outcome it must get: status, reward, failure kind,
# and a score's reason; an invalid outcome's reason, the launcher's detail, is not compared).
_GRADE_CASES: dict[str, tuple[dict[str, Any], str, tuple[str, float | None, str | None, str | None]]] = {
    "honest_pass": ({}, "generated", ("scored", 1.0, None, None)),
    "honest_fail": ({"data": "20"}, "generated", ("scored", 0.0, None, None)),
    "missing_artifact": (
        {"solve": "rm -f answer.json\n"},
        "generated",
        ("scored", 0.0, None, "solver_output_contract"),
    ),
    "replay_fails": (
        {"solve": _solve_line(_FAILING_SOLVE)},
        "generated",
        ("scored", 0.0, None, "solver_nonreproducible"),
    ),
    "variant_followed": ({"checks": True}, "generated", ("scored", 1.0, None, None)),
    "variant_ignored": (
        {"checks": True, "solve": _solve_line(_IGNORING_SOLVE)},
        "generated",
        ("scored", 0.0, None, "solver_variant_failed"),
    ),
    "hardcoded_literal": (
        {"checks": True, "solve": _solve_line(_HARDCODED_SOLVE)},
        "generated",
        ("scored", 0.0, None, "solver_hardcoded"),
    ),
    "variant_sealed": ({"checks": True, "solve": _solve_line(_SEALED_SOLVE)}, "generated", ("scored", 0.0, None, None)),
    "replay_sealed": ({"solve": _PUBLISH + _solve_line(_PRYING_SOLVE)}, "generated", ("scored", 0.0, None, None)),
    "module_shadow": (
        {
            "data": "20",
            "grader": _FRACTION_GRADER,
            "solve": f"cat > fractions.py <<'PLATOON_SHADOW'\n{_SHADOW}PLATOON_SHADOW\n",
            "toml": lambda config: config["artifacts"].append("fractions.py"),
        },
        "generated",
        ("scored", 0.0, None, None),
    ),
    "invalid_grader": (
        {"grader": _GRADERS["invalid"]},
        "generated",
        ("invalid", None, "verifier_generated_protocol", None),
    ),
    "crash_after_zero": (
        {"grader": _GRADERS["crash_after_zero"]},
        "generated",
        ("invalid", None, "verifier_generated_exit", None),
    ),
    "kills_launcher": (
        {"grader": _GRADERS["kills_launcher"]},
        "generated",
        ("invalid", None, "verifier_generated_unfinished", None),
    ),
    "own_reward_file": ({"grader": _GRADERS["own_reward_file"]}, "generated", ("scored", 0.0, None, None)),
    "lingering_writer": ({"grader": _GRADERS["lingering_writer"]}, "generated", ("scored", 0.0, None, None)),
    "log_censor": ({"grader": _GRADERS["log_censor"]}, "generated", ("scored", 0.0, None, None)),
    "trusted_grader_crash": (
        {"grader": _GRADERS["crash"]},
        "trusted",
        ("invalid", None, "verifier_trusted_exit", None),
    ),
    "flaky_grader": ({"grader": _GRADERS["coin_flip"]}, "generated", ("invalid", None, _FLAKY, None)),
}


async def _graded(service: Any, case: str, sources: _Sources) -> None:
    changes, owner, expected = _GRADE_CASES[case]
    checks = {"replay_checks": REPLAY_CHECKS} if changes.get("checks") else {}
    made = await service.materialize(sources.write(case, **changes), grader_owner=owner, **checks)
    if not made.ok:
        raise _Failed(f"materialize refused it: {list(made.problems)}")
    session = await service.open(made.task, rollout_id=f"platoon-red-team-{uuid.uuid4().hex[:12]}")
    try:
        solved = await session.run_solution()
        if solved.exit_code != 0:
            raise _Failed(f"its solution exited {solved.exit_code}: {(solved.stderr or solved.stdout)[-500:]}")
        outcome = await session.grade()
        grades = 1
        # A coin flip repeats as often as not: the same submission is graded again while its regrades agreed.
        while expected[2] == _FLAKY and _agreed(outcome) and grades < FLAKY_GRADES:
            outcome = await session.grade()
            grades += 1
    finally:
        await session.close()
    got = (outcome.status, outcome.reward, outcome.failure_kind, outcome.reason if outcome.trainable else None)
    if got != expected and expected[2] == _FLAKY and outcome.trainable:
        if service.runtime_tasks.regrade == 0:
            raise _Failed(f"graded {got}: runtime_tasks.regrade is 0, so no generated grader's score is graded again")
        if _agreed(outcome):
            raise _Failed(
                f"never {expected} in {grades} grades of one submission, each a score that every regrade gave again "
                f"(the last {got})"
            )
    if got != expected:
        launcher = outcome.metadata.get("launcher") or {}
        detail = launcher.get("detail") or outcome.metadata.get("test_output") or ""
        raise _Failed(f"graded {got}, not {expected}: {str(detail)[-500:]}")


def _agreed(outcome: Any) -> bool:
    """Whether a grade is the grader's score that every regrade gave again: a score with no reason (the solver's 0s name
    theirs, a regrade's among them), and regrades."""

    return outcome.trainable and outcome.reason is None and "regrades" in outcome.metadata


def to_toml(table: Mapping[str, Any]) -> str:
    """``table`` as a ``task.toml``: its tables as ``[sections]`` (an empty one too, which may mean something, as
    ``[verifier.environment]`` does), tables inside arrays inline."""

    lines: list[str] = []
    _section(table, (), lines)
    return "\n".join(lines) + "\n"


def _section(table: Mapping[str, Any], path: tuple[str, ...], lines: list[str]) -> None:
    values = {key: value for key, value in table.items() if not isinstance(value, Mapping)}
    tables = {key: value for key, value in table.items() if isinstance(value, Mapping)}
    if path and (values or not tables):
        lines.append(f"[{'.'.join(_key(part) for part in path)}]")
    lines += [f"{_key(key)} = {_value(value)}" for key, value in values.items()]
    for key, value in tables.items():
        _section(value, (*path, key), lines)


def _key(key: str) -> str:
    return key if re.fullmatch(r"[A-Za-z0-9_-]+", key) else json.dumps(key)


def _value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return repr(value) if math.isfinite(value) else ("inf" if value > 0 else "-inf")
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, datetime.date | datetime.time):
        return value.isoformat()
    if isinstance(value, Mapping):
        return "{" + ", ".join(f"{_key(key)} = {_value(item)}" for key, item in value.items()) + "}"
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_value(item) for item in value) + "]"
    raise TypeError(f"no TOML for {value!r}")
