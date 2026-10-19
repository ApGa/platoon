"""Read a Harbor verifier's output directory (``/logs/verifier``) in its sandbox and map it to rewards and an
:class:`Outcome`.

``reward.json`` (preferred) holds named numeric rewards; ``reward.txt`` holds one
number, the ``reward``. ``reward.json`` may also carry two reserved keys:
``invalid`` (a number or boolean; >= 0.5 declares the measurement invalid) and
``failure_kind`` (a string saying why). A missing, empty or malformed reward file
is an infrastructure error, never a reward of 0.  Which reward trains is the
dataset's reward contract (:mod:`platoon.envs.harbor.grading`); without one it is
``reward``.

The files are read in the sandbox by its own shell (:func:`collect`), each only if
it is a regular file reached through no symlink, in a directory with nothing
mounted on or below it: the reward files whole, up to :data:`MAX_REWARD_BYTES`,
and the verifier's logs (:data:`VERIFIER_LOGS`) as their last
:data:`LOG_TAIL_BYTES`.  A symlink would read whatever it points to (a host file,
once a download copied it out).  Grading kills the agents' processes and makes the
directory afresh before the tests run, so a reward file that is anything else (a
symlink, a directory, a FIFO, a mount, a file too large to read or one its reader
may not open) was put there by code the tests ran, the agent's: it scores 0
(``reward_file_not_regular``, ``reward_file_too_large``; :func:`suspect_outcome`),
a trainable outcome under the reward contract's first name.  The regular files read are written to a local directory
named ``verifier`` (:func:`write_local`), which the contract and the classifiers
read.
"""

from __future__ import annotations

import json
import math
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from platoon.envs.service import PLATOON_ZERO_KEY, Outcome

if TYPE_CHECKING:
    from platoon.sandbox.base import Sandbox

REWARD_JSON = "reward.json"
REWARD_TEXT = "reward.txt"
TEST_STDOUT = "test-stdout.txt"
PRIMARY_REWARD = "reward"
REWARD_FILES = (REWARD_JSON, REWARD_TEXT)
# The verifier's logs: test-stdout.txt, where platoon sends test.sh's stdout and stderr, and the ones tests may write.
VERIFIER_LOGS = (TEST_STDOUT, "test-stderr.txt", "stdout.txt", "stderr.txt")
MAX_REWARD_BYTES = 1 << 20  # a reward file longer than this is not read
LOG_TAIL_BYTES = 1 << 20  # of each log, what grading reads
TEST_OUTPUT_TAIL_BYTES = 4096  # of the tests' output kept in the outcome

# What collect reports for a file it does not read: "missing", or why it may not be trusted.
_NOT_READ = frozenset({"missing", "symlink", "special", "mount", "unreadable", "too_large"})

# sh -c _COLLECT_SCRIPT platoon-collect DIR MARK MAX_BYTES TAIL_BYTES NAME:MODE...  (MODE: whole or tail)
# Prints "MARK NAME KIND" for each name; a regular file's header is followed by its contents (MODE whole, up to
# MAX_BYTES) or its last TAIL_BYTES (MODE tail), then a line "MARK end".  A symlinked DIR reads as symlinks, one that
# is not a directory as special files, and one with anything mounted on or below it (which root in some sandboxes
# can do: a host file bound over a reward file reads as a regular file) as mounts.
_COLLECT_SCRIPT = r"""
dir=$1; mark=$2; max=$3; tail_bytes=$4; shift 4
dirkind=
if [ -L "$dir" ]; then dirkind=symlink
elif [ ! -e "$dir" ]; then dirkind=missing
elif [ ! -d "$dir" ]; then dirkind=special
elif [ -r /proc/self/mountinfo ]; then
  while read -r id parent devices root point rest; do
    case $point in "$dir"|"$dir"/*) dirkind=mount; break ;; esac
  done < /proc/self/mountinfo
fi
for entry in "$@"; do
  name=${entry%:*}; mode=${entry##*:}; path=$dir/$name
  if [ -n "$dirkind" ]; then echo "$mark $name $dirkind"; continue; fi
  if [ -L "$path" ]; then echo "$mark $name symlink"; continue; fi
  if [ ! -e "$path" ]; then echo "$mark $name missing"; continue; fi
  if [ ! -f "$path" ]; then echo "$mark $name special"; continue; fi
  if [ ! -r "$path" ]; then echo "$mark $name unreadable"; continue; fi
  if [ "$mode" = tail ]; then
    echo "$mark $name file"
    tail -c "$tail_bytes" "$path"
  else
    size=$(head -c "$((max + 1))" "$path" | wc -c)
    if [ "$((size + 0))" -gt "$max" ]; then echo "$mark $name too_large"; continue; fi
    echo "$mark $name file"
    cat "$path"
  fi
  printf '\n%s end\n' "$mark"
done
"""
_COLLECT_TIMEOUT_S = 120.0


@dataclass(frozen=True)
class VerifierFile:
    """A file of the verifier's output directory as :func:`collect` found it.

    ``kind`` is ``file`` for a regular file reached through no symlink, whose ``text`` holds its contents (or a log's
    tail); else ``missing``, ``symlink`` (the file or its directory is one), ``special`` (a directory, FIFO, socket
    or device, or the directory is not a directory), ``mount`` (something is mounted on or below the directory),
    ``unreadable`` or ``too_large``, and ``text`` is empty.
    """

    kind: str
    text: str = ""


MISSING = VerifierFile("missing")


async def collect(
    sandbox: Sandbox,
    directory: str,
    *,
    whole: Sequence[str] = REWARD_FILES,
    tail: Sequence[str] = VERIFIER_LOGS,
    tail_bytes: int = LOG_TAIL_BYTES,
) -> dict[str, VerifierFile]:
    """Read the files named in ``whole`` (up to :data:`MAX_REWARD_BYTES`) and the last ``tail_bytes`` of those in
    ``tail`` from ``directory`` in the sandbox, with the sandbox's own shell, only where each is a regular file
    reached through no symlink.

    Raises RuntimeError when the sandbox's shell cannot report them.
    """

    mark = f"platoon-{uuid.uuid4().hex}"
    names = [f"{name}:whole" for name in whole] + [f"{name}:tail" for name in tail]
    result = await sandbox.exec(
        ["/bin/sh", "-c", _COLLECT_SCRIPT, "platoon-collect", directory, mark, str(MAX_REWARD_BYTES), str(tail_bytes)]
        + names,
        cwd="/",
        timeout_s=_COLLECT_TIMEOUT_S,
    )
    if result.timed_out or result.exit_code != 0:
        detail = result.stderr.strip() or f"exit {result.exit_code}"
        raise RuntimeError(f"Could not read the verifier's output in {directory}: {detail}")
    files = _parse(result.stdout, mark)
    if missing := sorted({*whole, *tail} - set(files)):
        raise RuntimeError(f"Could not read the verifier's output in {directory}: no report for {missing}")
    return files


def _parse(stdout: str, mark: str) -> dict[str, VerifierFile]:
    header = re.compile(rf"^{re.escape(mark)} (\S+) (\S+)\n", re.MULTILINE)
    end = f"\n{mark} end\n"
    files: dict[str, VerifierFile] = {}
    position = 0
    while (match := header.search(stdout, position)) is not None:
        name, kind = match.groups()
        position = match.end()
        if kind != "file":
            files[name] = VerifierFile(kind if kind in _NOT_READ else "special")
            continue
        stop = stdout.find(end, position)
        if stop < 0:
            break  # cut off: no report for this file
        files[name] = VerifierFile("file", stdout[position:stop])
        position = stop + len(end)
    return files


def suspect_outcome(files: Mapping[str, VerifierFile], *, primary: str = PRIMARY_REWARD) -> Outcome | None:
    """A scored 0 under ``primary`` (the reward contract's first name) when a reward file is there but is not a
    regular file grading may read (see the module docstring), whatever another reward file says; None when each is
    a regular file or missing.  Platoon gives it itself, which its metadata marks (``PLATOON_ZERO_KEY``)."""

    suspect = {
        name: file.kind for name in REWARD_FILES if (file := files.get(name, MISSING)).kind not in ("file", "missing")
    }
    if not suspect:
        return None
    reason = "reward_file_too_large" if set(suspect.values()) == {"too_large"} else "reward_file_not_regular"
    return Outcome(
        status="scored",
        rewards={primary: 0.0},
        primary=primary,
        reason=reason,
        metadata={"reward_files": suspect, PLATOON_ZERO_KEY: True},
    )


def write_local(files: Mapping[str, VerifierFile], directory: Path) -> Path:
    """Write the regular files :func:`collect` read into ``directory`` (made if needed): the local copy of the
    verifier's output that the reward contract and the classifiers read.  Returns ``directory``."""

    directory.mkdir(parents=True, exist_ok=True)
    for name, file in files.items():
        if file.kind == "file" and "/" not in name and name not in (".", ".."):
            (directory / name).write_text(file.text, encoding="utf-8")
    return directory


def output_tail(files: Mapping[str, VerifierFile]) -> str | None:
    """The tail of the tests' output kept in the outcome (``test_output``); None unless it is a regular file."""

    stdout = files.get(TEST_STDOUT, MISSING)
    return stdout.text[-TEST_OUTPUT_TAIL_BYTES:] if stdout.kind == "file" else None


def read_outcome(verifier_dir: Path, *, primary: str = PRIMARY_REWARD) -> Outcome:
    """The outcome the reward files of a local copy of the verifier's output give, scoring ``primary``."""

    return _scored(read_rewards(verifier_dir), primary)


def read_rewards(verifier_dir: Path) -> dict[str, float] | Outcome:
    """The reported rewards, or the infrastructure outcome of reward files that report none."""

    if (verifier_dir / REWARD_JSON).is_file():
        return rewards_from_json((verifier_dir / REWARD_JSON).read_text())
    if (verifier_dir / REWARD_TEXT).is_file():
        return rewards_from_text((verifier_dir / REWARD_TEXT).read_text())
    return _no_reward("reward_file_missing")


def outcome_from_reward_json(raw: str, *, primary: str = PRIMARY_REWARD) -> Outcome:
    return _scored(rewards_from_json(raw), primary)


def outcome_from_reward_text(raw: str, *, primary: str = PRIMARY_REWARD) -> Outcome:
    return _scored(rewards_from_text(raw), primary)


def rewards_from_json(raw: str) -> dict[str, float] | Outcome:
    if not raw.strip():
        return _no_reward("reward_file_empty")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return _no_reward("reward_file_malformed")
    if not isinstance(data, dict):
        return _no_reward("reward_file_malformed")
    invalid = data.pop("invalid", None)
    failure_kind = data.pop("failure_kind", None)
    if (invalid is not None and not _is_finite_number(invalid)) or not isinstance(failure_kind, str | None):
        return _no_reward("reward_file_malformed")
    if invalid is not None and invalid >= 0.5:
        return Outcome.infra(failure_kind or "verifier_declared", status="invalid", metadata={"reported": data})
    if not all(_is_finite_number(value) for value in data.values()):
        return _no_reward("reward_file_malformed")
    return {key: float(value) for key, value in data.items()}


def rewards_from_text(raw: str) -> dict[str, float] | Outcome:
    if not raw.strip():
        return _no_reward("reward_file_empty")
    try:
        value = float(raw)
    except ValueError:
        return _no_reward("reward_file_malformed")
    if not math.isfinite(value):
        return _no_reward("reward_file_malformed")
    return {PRIMARY_REWARD: value}


def _scored(rewards: dict[str, float] | Outcome, primary: str) -> Outcome:
    if isinstance(rewards, Outcome):
        return rewards
    if primary not in rewards:
        return Outcome.infra(
            "verifier_no_primary",
            reason=f"the verifier reported {sorted(rewards)}, not the primary reward {primary!r}",
            metadata={"reported": rewards},
        )
    return Outcome(status="scored", rewards=rewards, primary=primary)


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, int | float) and math.isfinite(value)


def _no_reward(reason: str) -> Outcome:
    return Outcome.infra("verifier_no_reward", reason=reason)
