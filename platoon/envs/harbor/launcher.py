"""The launcher of a runtime Harbor task: it grades the task in its separate verifier, replaying the submission first.

``materialize`` (:mod:`platoon.envs.harbor.runtime`) installs this file as the task's ``tests/platoon/launcher.py``,
with its settings in ``tests/platoon/launch.json``, and a ``tests/test.sh`` that runs it with the verifier image's
``python3 -I``.  The task's own grader is in ``tests/gen/``, where ``grader.json`` names its command and the task's
replay, if any.  The launcher needs Python 3.8 or later and its standard library only, and runs as root in the
verifier's sandbox, with a umask that keeps what it writes root's alone.  In order, it:

1. seals ``/tests``, ``/logs/verifier`` and what the agents published, ``/logs/agent`` and ``/logs/artifacts``, where
   they exist: each becomes root's, mode 700, so that what it runs as another user can neither read the grader's files
   or a copy of the submission the agents left there nor write a reward.  It checks that each seal took: a directory,
   never a link, root's and open to no other user, in a directory where the replay's user cannot replace it; a seal it
   cannot make or that does not take fails closed;
2. checks the artifacts the task declares, each a path relative to the working directory, against the **output
   contract**: each is a regular file or a directory of regular files and directories, every file at most
   ``max_output_bytes``, a ``.json`` file JSON, and nothing but directories above an output of the replay.  For a task
   with a **replay**, it checks that the replay's user may write in no directory of the working directory (failing
   closed), then makes the working directory root's with the group and other write bits stripped, so nothing there is
   the replay's user's to write (:func:`Ops.harden`).  It moves the replay's outputs out of it (into
   ``$PLATOON_SUBMITTED``, where the grader may compare them), copies what is left into a directory of the replay
   user's (:data:`REPLAY_UID`, never root), and runs the replay's command there as that user alone (no supplementary
   groups, no new privileges), with a thin environment (:data:`REPLAY_VARIABLES`, and the template's own variables
   that ``replay_env`` names, its ``PATH`` cut to absolute entries alone so nothing in the replay's directory stands in
   for a program), no core files, no file larger than an output may be, at most :data:`MAX_REPLAY_PROCESSES` processes
   of its user (threads count) and :data:`MAX_REPLAY_OPEN_FILES` open files, the most killable to the OOM killer
   (:data:`OOM_SCORE_ADJ`) and, where the settings give one, that much address space a process, for at most half the
   time left.  It then kills what the replay left, below the launcher or of the replay's user (one of that user's it
   cannot signal is a survivor, not foreign), removes what that user left where every user may write
   (:data:`TEMPORARY`) and, in the replay's own copy, all but its outputs, which wait there as root's (:func:`_hold`,
   :func:`remove_tree`, within :data:`CLEAR_WINDOW_S` past its time), and puts each output the replay wrote in its
   place, with their SHA-256 in its report (:data:`REPLAYED_SHA256`), by which a regrade, which replays the submission
   again, tells a submission that computes other outputs each time it runs from a grader that judges the same outputs
   otherwise.  A violation, a submission too large to copy or too slow to check or copy, a replay that fails, outlives
   its time, leaves processes that cannot be killed, leaves more than the launcher removes in its time or fills a
   filesystem, lacks its own program, and an output it does not write again score 0, the solver's fault (reason
   ``solver_output_contract`` or ``solver_nonreproducible``);
3. runs the grader's command as root in ``/tests/gen``, with the verifier's own umask, for the rest of the verifier's
   time limit.  The grader judges the outputs in ``$PLATOON_WORKDIR``, the working directory, and writes its verdict
   to ``$PLATOON_VERDICT``: ``{"passed": true|false, "checks": [{"id": "...", "passed": true|false, "margin": 0.5}]}``,
   where ``checks`` and each ``margin`` may be left out, an ``id`` holds at most :data:`MAX_CHECK_ID_CHARS`
   characters, and nothing else may be there.  Its output goes to ``/logs/verifier/gen/grader.log``;
4. kills every process the replay and the grader left: it is their subreaper, so a daemon that detached is still one
   of its descendants;
5. writes the only reward file, ``reward.json``: the reward named ``reward_key``, 1 for a verdict that passed and 0
   for one that did not, or ``{"invalid": 1, "failure_kind": ...}``: ``<grader>_exit`` when the grader exits nonzero
   or cannot start, ``<grader>_timeout`` when it outlives its time, ``<grader>_protocol`` when it writes no verdict or
   one beyond the schema (``invalid`` among them: a grader cannot censor a measurement) or the task's replay command
   cannot start, ``<grader>_isolation`` when what it left cannot be killed, and ``<grader>_unfinished`` when the
   launcher fails once the grader ran (as root, the grader may have broken what the launcher reads), ``<grader>``
   being ``verifier_generated`` for a grader the generator wrote and ``verifier_trusted`` for a trusted one
   (``grader_owner``); and ``verifier_trusted_launcher`` when the launcher cannot do its own part before the grader
   runs (it does not run as root, cannot seal, cannot switch to the replay's user): a fault on the trusted side,
   never the task's.  It removes everything else from ``/logs/verifier`` but ``test-stdout.txt`` and ``gen/``, writes
   its report, ``launcher.json``, of at most :data:`MAX_REPORT_BYTES` (without the verdict's checks, then the
   artifacts' names, where it would be larger), and exits 0, which the Harbor service checks.

Every file whose contents it reads (its settings, the artifacts' JSON, the replay's outputs, the verdict, the logs
whose tails it keeps) is opened without following a link or waiting on a FIFO and checked on the open descriptor.
The launcher is no security boundary against the grader, which runs as root beside it: a grader may choose 0 or 1,
which it may anyway and which controls check, and anything else is a non-score that attribution charges to it.
"""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import posixpath
import re
import resource
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import traceback

SCHEMA = "platoon.launcher/v1"
CONFIG = "launch.json"  # in tests/platoon, beside this file
REPORT = "launcher.json"  # in /logs/verifier
REWARD = "reward.json"
STDOUT = "test-stdout.txt"  # where platoon sends the launcher's output
GEN = "gen"  # tests/gen: the task's grader; /logs/verifier/gen: its verdict and log
VERDICT = "verdict.json"
GRADER_LOG = "grader.log"
#: Where the agents publish what they leave besides the artifacts, which the verifier's sandbox may hold: sealed with
#: ``/tests`` and ``/logs/verifier`` where they exist.
PUBLISHED = ("/logs/agent", "/logs/artifacts")
#: Where every user of the verifier's sandbox may write: after the replay, the launcher removes what the replay's user
#: left there (:func:`clear_left`), so the grader neither finds it there nor goes without the room it took.  Besides the
#: temporary directories, ``/run/lock`` (which Debian's images give every user, ``/var/lock`` linking to it) and the
#: POSIX message queues' ``/dev/mqueue``; one the sandbox lacks is none, and one met twice through a link is cleared
#: once.
TEMPORARY = ("/tmp", "/var/tmp", "/dev/shm", "/run/lock", "/var/lock", "/dev/mqueue")
#: The user the replay runs as in the verifier's sandbox, and its group: no one else's, and never root (0).
REPLAY_UID = 65533
#: The share of the time left that the replay may take; the grader has the rest.
REPLAY_SHARE = 0.5
#: The most processes the replay's user may have, its threads counted (``RLIMIT_NPROC``), and the most files a process
#: of the replay may hold open (``RLIMIT_NOFILE``): room for threaded numerics, never enough to starve the launcher and
#: the grader.
MAX_REPLAY_PROCESSES = 1024
MAX_REPLAY_OPEN_FILES = 1024
#: What a replay keeps of the verifier's environment, besides the template's own variables that the settings'
#: ``replay_env`` names: the path, the locale and time zone, the search paths and prefixes of Python and the dynamic
#: linker, and the thread counts of numeric libraries.  The verifier's other variables, which may hold the grader's
#: credentials, stay out.
REPLAY_VARIABLES = (
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "PYTHONHASHSEED",
    "PYTHONPATH",
    "LD_LIBRARY_PATH",
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
DEFAULT_PATH = "/usr/local/bin:/usr/bin:/bin"
#: What the report keeps of each log, in characters.
TAIL_CHARS = 2000
#: The largest settings file read.
MAX_CONFIG_BYTES = 1 << 20
#: The largest verdict read, the most checks one holds and the longest id of one; the report keeps the first
#: ``REPORTED_CHECKS``.
MAX_VERDICT_BYTES = 1 << 20
MAX_CHECKS = 1000
MAX_CHECK_ID_CHARS = 256
REPORTED_CHECKS = 100
#: The largest report the launcher writes, well below what grading reads of a file in /logs/verifier (1 MiB): a report
#: that would be larger leaves out the verdict's checks, then the artifacts' names, then all but the outcome.
MAX_REPORT_BYTES = 64 << 10
#: The most a replay copies of the working directory beyond the task's own files (its ``environment/``, which
#: ``runtime_tasks.max_bytes`` bounds): ``materialize`` gives the launcher the sum, as ``max_copy_bytes``.
MAX_SUBMISSION_BYTES = 4 << 30
#: How long the launcher keeps killing what is left before it gives up.
KILL_WINDOW_S = 10.0
#: How long past the replay's time the launcher may take to remove what the replay left, in its own copy and of its
#: user's where every user may write: a replay that left more than that removes is the solver's 0.
CLEAR_WINDOW_S = 10.0
#: How deeply a tree the launcher removes may nest (:func:`remove_tree`, which holds two descriptors a level): a replay
#: that leaves one deeper scores 0, whatever the launcher's limit on open files.
MAX_TREE_DEPTH = 256
#: How many entries the launcher removes between two looks at the clock (:class:`_Clock`).
_CLEAR_BATCH = 256
#: The most killable a process may be to the OOM killer (``oom_score_adj``): the replay's processes get it, so disk or
#: memory the replay runs the sandbox out of ends the replay, not the launcher or the grader.  Raising it needs no
#: privilege; lowering it does, which is why the launcher raises it rather than trusting an inherited value.
OOM_SCORE_ADJ = 1000
_OOM_SCORE_ADJ = "/proc/self/oom_score_adj"
# The errors of a write that found its filesystem full.
_NO_ROOM = (errno.ENOSPC, errno.EDQUOT)
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
OWNERS = ("generated", "trusted")
GRADER_KINDS = {"generated": "verifier_generated", "trusted": "verifier_trusted"}
#: The faults of a grader, each a failure kind ``<grader>_<fault>``; ``unfinished`` is also the Harbor service's, for a
#: launcher that did not finish (it exited nonzero, left no report, or outlived the verifier's time).
GRADER_FAULTS = ("exit", "timeout", "protocol", "isolation", "unfinished")
UNFINISHED = "unfinished"
LAUNCHER_FAULT = "verifier_trusted_launcher"
OUTPUT_CONTRACT = "solver_output_contract"
NONREPRODUCIBLE = "solver_nonreproducible"
#: The key of the report that holds the SHA-256 of the outputs the replay wrote, which the grader judged
#: (:func:`outputs_sha256`); a task without a replay leaves it out.
REPLAYED_SHA256 = "replayed_sha256"
OUTPUTS_RULE = "platoon-replayed-outputs-v1"
_PR_SET_CHILD_SUBREAPER = 36
_PR_SET_NO_NEW_PRIVS = 38
_VERDICT_KEYS = ("passed", "checks")
_CHECK_KEYS = ("id", "passed", "margin")
_VARIABLE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC


class LauncherFault(Exception):
    """The launcher cannot do its own part: a fault on the trusted side (``verifier_trusted_launcher``), never the
    task's."""


class SolverFault(Exception):
    """The submission is more than the launcher takes (too large to copy, too slow to check or copy): a score of 0, the
    solver's fault (``reason``), never the trusted side's."""

    def __init__(self, reason, detail):
        super().__init__(detail)
        self.reason = reason


class Leftover(Exception):
    """What a replay left that the launcher cannot remove, or not in its time (:func:`remove_tree`,
    :func:`clear_left`): the solver's 0, as processes that outlive the sweep are."""


class Result:
    """What the launcher concluded: a reward (1 or 0), or the failure kind of a non-score; with why."""

    def __init__(self, reward=None, failure_kind=None, reason=None, detail=None):
        self.reward = reward
        self.failure_kind = failure_kind
        self.reason = reason
        self.detail = detail

    @classmethod
    def scored(cls, passed):
        return cls(reward=1 if passed else 0, detail="the verdict passed" if passed else "the verdict failed")

    @classmethod
    def solver(cls, reason, detail):
        return cls(reward=0, reason=reason, detail=detail)

    @classmethod
    def invalid(cls, failure_kind, detail):
        return cls(failure_kind=failure_kind, detail=detail)


class Ops:
    """What the launcher does as root; a test that runs it without root replaces these."""

    def euid(self):
        return os.geteuid()

    def seal(self, path):
        """Make ``path`` and everything below it root's, never following a link, and ``path`` itself mode 700."""

        _chown_tree(path, 0, 0)
        os.chmod(path, 0o700)

    def give(self, path, uid):
        """Make ``path`` and everything below it ``uid``'s (and the group of that number's)."""

        _chown_tree(path, uid, uid)

    def harden(self, path):
        """Make everything in the working directory ``path`` root's and strip the group and other write bits (never
        following a link), so the replay's user may write nothing there: a survivor of the replay could then neither
        change what the grader reads nor the directories an output goes back through (:func:`_harden_tree`)."""

        _harden_tree(path)

    def demote(self, max_file_bytes, max_address_space_bytes=None):
        """What runs in the replay's process before its command: no supplementary groups, the replay's group and user,
        checked to be all of its ids, the replay's limits (:func:`_limit_replay`), its disfavor for the OOM killer
        (:func:`_oom_disfavor`), and no new privileges (a setuid program gives none)."""

        libc = _libc()  # loaded before the fork, which the child then only calls

        def demote():
            _oom_disfavor()  # while still root: dropping to the replay's user makes /proc/self root's; the value lasts
            os.setgroups([])
            os.setgid(REPLAY_UID)
            os.setuid(REPLAY_UID)
            if os.getresuid() != (REPLAY_UID,) * 3 or os.getresgid() != (REPLAY_UID,) * 3 or os.getgroups():
                raise OSError(errno.EPERM, "the replay's process is not the replay's user alone")
            _limit_replay(max_file_bytes, max_address_space_bytes)
            _prctl(_PR_SET_NO_NEW_PRIVS, 1, libc)

        return demote

    def processes_of(self, uid):
        """The live processes of ``uid`` (the replay's user) that ``/proc`` shows, wherever they are in the process
        tree (:func:`user_processes`)."""

        return user_processes(uid)

    def replays_own(self, info):
        """Whether a file (its ``os.lstat`` result) is the replay's user's: what a replay left where every user may
        write (:func:`clear_left`)."""

        return info.st_uid == REPLAY_UID


class Paths:
    """Where the launcher finds the task's tests and the submission, writes the verifier's output, keeps its own files,
    finds what the agents published, and where every user may write (``temporary``, which it clears of what the
    replay's user left there)."""

    def __init__(self, tests, logs, workdir, scratch, published=(), temporary=()):
        self.tests = tests
        self.logs = logs
        self.workdir = workdir
        self.scratch = scratch
        self.published = tuple(published)
        self.temporary = tuple(temporary)

    @classmethod
    def here(cls):
        """The verifier's: ``/tests`` (above this file), ``/logs/verifier``, the working directory, ``/tmp``,
        :data:`PUBLISHED` and :data:`TEMPORARY`.  The ``process`` sandbox provider, which runs commands on the host
        below a root of its own, names that root in ``PLATOON_SANDBOX_ROOT``."""

        root = os.environ.get("PLATOON_SANDBOX_ROOT", "")
        tests = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return cls(
            tests=tests,
            logs=f"{root}/logs/verifier",
            workdir=os.getcwd(),
            scratch=f"{root}/tmp",
            published=[f"{root}{path}" for path in PUBLISHED],
            temporary=[f"{root}{path}" for path in TEMPORARY],
        )


def main():
    return run(Paths.here())


def run(paths, ops=None):
    """Grade the task (the module docstring's steps) and write the reward file and the report; the exit status, 0
    once both are written."""

    ops = ops or Ops()
    umask = os.umask(0o077)  # what the launcher writes, only root reads
    try:
        return _grade_task(paths, ops, umask)
    finally:
        os.umask(umask)


def _grade_task(paths, ops, umask):
    started = time.monotonic()
    report = {"schema": SCHEMA}
    config = None
    try:
        config = read_config(os.path.join(paths.tests, "platoon", CONFIG))
        report["grader_owner"] = config["grader_owner"]
        result = _launch(config, paths, ops, report, started, umask)
    except LauncherFault as exc:
        result = Result.invalid(LAUNCHER_FAULT, str(exc))
    except Exception as exc:  # the launcher's own failure before the grader ran (_grade answers for what follows)
        report["trace"] = traceback.format_exc()[-TAIL_CHARS:]
        result = Result.invalid(LAUNCHER_FAULT, f"the launcher failed: {type(exc).__name__}: {exc}")
    if not kill_descendants() and config is not None and result.failure_kind is None and result.reason is None:
        # A verdict's score: the grader ran, and what it left could not be killed.  (A solver's 0 stands: the replay
        # was checked for what it left before the grader ran.)
        kind = GRADER_KINDS[config["grader_owner"]]
        result = Result.invalid(f"{kind}_isolation", "processes the grader left could not be killed")
    report["seconds"] = round(time.monotonic() - started, 3)
    return _finish(paths, config, result, report)


def read_config(path):
    """The launcher's settings, which ``materialize`` wrote; a :class:`LauncherFault` for settings it cannot use."""

    try:
        config = json.loads(_read_regular(path, MAX_CONFIG_BYTES).decode("utf-8"))
    except (OSError, ValueError, RecursionError) as exc:
        raise LauncherFault(f"its settings {path} cannot be read: {exc!r}") from exc
    if not isinstance(config, dict) or config.get("schema") != SCHEMA:
        raise LauncherFault(f"its settings {path} are not {SCHEMA}")
    problems = []
    if config.get("grader_owner") not in OWNERS:
        problems.append("grader_owner")
    if not isinstance(config.get("reward_key"), str) or not config["reward_key"]:
        problems.append("reward_key")
    for key in ("timeout_s", "max_output_bytes", "max_copy_bytes"):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            problems.append(key)
    space = config.get("max_address_space_bytes")  # the replay's RLIMIT_AS; absent or null for none
    if space is not None and (isinstance(space, bool) or not isinstance(space, int) or space <= 0):
        problems.append("max_address_space_bytes")
    if not _paths(config.get("artifacts"), empty=True):
        problems.append("artifacts")
    if not isinstance(config.get("grader"), dict) or not _strings(config["grader"].get("command")):
        problems.append("grader")
    replay = config.get("replay")
    if replay is not None and not (
        isinstance(replay, dict) and _strings(replay.get("command")) and _paths(replay.get("outputs"))
    ):
        problems.append("replay")
    keep = config.get("replay_env", [])
    if not (_strings(keep, empty=True) and all(_VARIABLE.fullmatch(name) for name in keep)):
        problems.append("replay_env")
    if problems:
        raise LauncherFault(f"its settings {path} hold no usable {', '.join(problems)}")
    return config


def relative_path(path):
    """Whether ``path`` is a normalized path relative to a directory and below it: no leading ``/``, no ``..``, not
    ``.``, nothing ``posixpath.normpath`` would change."""

    return (
        isinstance(path, str)
        and bool(path)
        and "\0" not in path
        and not path.startswith("/")
        and posixpath.normpath(path) == path
        and path not in (".", "..")
        and not path.startswith("../")
    )


def _strings(value, empty=False):
    return isinstance(value, list) and (empty or bool(value)) and all(isinstance(item, str) and item for item in value)


def _paths(value, empty=False):
    return _strings(value, empty=empty) and all(relative_path(item) for item in value)


def _launch(config, paths, ops, report, started, umask):
    euid = ops.euid()
    if euid != 0:
        raise LauncherFault(
            f"it runs as uid {euid}, not root: a runtime task's verifier must run as root (a template whose [verifier] "
            "user is root, and runtime_tasks.verifier_user: honor, on a sandbox that gives root)"
        )
    replay = config.get("replay")
    if replay is not None and (isinstance(REPLAY_UID, bool) or not isinstance(REPLAY_UID, int) or REPLAY_UID <= 0):
        raise LauncherFault(f"the replay's user must be a uid above 0 (root's), not {REPLAY_UID!r}")
    workdir = os.path.realpath(paths.workdir)
    if workdir == "/":
        raise LauncherFault("the verifier's working directory is /, which a replay cannot copy")
    sealed = (paths.tests, paths.logs, *paths.published)
    for each in map(os.path.realpath, sealed):  # where the artifacts land must be apart from what is sealed
        if _within(workdir, each) or _within(each, workdir):
            raise LauncherFault(f"the verifier's working directory {workdir} and {each} are one inside the other")
    try:
        _prctl(_PR_SET_CHILD_SUBREAPER, 1)
    except (OSError, AttributeError) as exc:
        raise LauncherFault(f"it cannot be the subreaper of what it runs: {exc}") from exc
    _seal(paths.tests, ops)
    _seal(paths.logs, ops)
    for path in paths.published:
        _seal(path, ops, required=False)
    gen = os.path.join(paths.logs, GEN)
    try:
        _fresh_directory(gen)
        scratch = tempfile.mkdtemp(prefix="platoon-launcher-", dir=paths.scratch)
        os.chmod(scratch, 0o711)  # the replay's user may reach its own directory there, and list nothing
    except OSError as exc:
        raise LauncherFault(f"it cannot make its own directories: {exc}") from exc
    owner = config["grader_owner"]
    cap = int(config["max_output_bytes"])
    deadline = started + _budget(float(config["timeout_s"]))
    violation = contract_violation(config["artifacts"], paths.workdir, cap, deadline)
    if violation is None and replay is not None:
        violation = layout_violation(replay["outputs"], paths.workdir)
    report["contract"] = {"artifacts": list(config["artifacts"]), "violation": violation}
    if violation is not None:
        return Result.solver(OUTPUT_CONTRACT, violation)
    env = dict(os.environ)
    env.update(PLATOON_WORKDIR=paths.workdir, PLATOON_VERDICT=os.path.join(gen, VERDICT))
    if replay is not None:
        submitted = os.path.join(scratch, "submitted")
        replay_deadline = time.monotonic() + max(0.0, deadline - time.monotonic()) * REPLAY_SHARE
        try:
            found = _replay(config, paths, ops, report, scratch, submitted, replay_deadline)
        except SolverFault as exc:
            return Result.solver(exc.reason, str(exc))
        if found is not None:
            return found
        env["PLATOON_SUBMITTED"] = submitted
    return _grade(config["grader"]["command"], paths, env, report, deadline, owner, gen, umask)


def _seal(path, ops, required=True):
    """Seal one directory (step 1) with ``ops.seal``, and check that the seal took: ``path`` is a directory, never a
    link, of the launcher's own user and open to no other, in a directory where the replay's user cannot replace it
    (:func:`_replaceable`).  A :class:`LauncherFault` when it cannot be sealed or the seal did not take; nothing for a
    ``path`` that need not be there (``required``) and is not."""

    try:
        info = os.lstat(path)
    except FileNotFoundError:
        if required:
            raise LauncherFault(f"{path} does not exist, so it cannot be sealed") from None
        return
    except OSError as exc:
        raise LauncherFault(f"it cannot seal {path}: {exc}") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise LauncherFault(f"{path} is {_kind(info)}, which it cannot seal")
    try:
        ops.seal(path)
        info = os.lstat(path)
        holder = os.stat(os.path.dirname(os.path.abspath(path)))
    except OSError as exc:
        raise LauncherFault(f"it cannot seal {path}: {exc}") from exc
    mode = stat.S_IMODE(info.st_mode)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or mode & 0o077:
        raise LauncherFault(f"{path} is still open to other users once sealed (owner {info.st_uid}, mode {mode:o})")
    if _replaceable(holder):
        raise LauncherFault(
            f"the directory that holds {path} lets the replay's user replace it (owner {holder.st_uid}, group "
            f"{holder.st_gid}, mode {stat.S_IMODE(holder.st_mode):o})"
        )


def _replaceable(info):
    """Whether the replay's user could rename or remove what a directory (its ``os.stat`` result) holds: it owns the
    directory, or may write there and the sticky bit is not set."""

    return info.st_uid == REPLAY_UID or (_writable_by_replay(info) and not info.st_mode & stat.S_ISVTX)


def _writable_by_replay(info):
    """Whether the replay's user may add, rename or remove entries of a directory (its ``os.stat`` result): it owns
    it, or may write there as one of the other users or as its group."""

    if info.st_uid == REPLAY_UID or info.st_mode & stat.S_IWOTH:
        return True
    return info.st_gid == REPLAY_UID and bool(info.st_mode & stat.S_IWGRP)


def _exposed_directory(workdir):
    """The first directory of ``workdir``, itself or one below it (never through a link), that the replay's user may
    write in, where it could put what the launcher puts back the replay's outputs through or the grader reads; None
    when there is none."""

    for directory, _, _ in os.walk(workdir):
        if _writable_by_replay(os.lstat(directory)):
            return directory
    return None


def _budget(timeout_s):
    """What the launcher's steps may take of the verifier's time limit, which kills it: all but a margin for killing
    what is left and writing the reward."""

    return max(0.0, timeout_s - min(30.0, max(5.0, 0.1 * timeout_s)))


def contract_violation(artifacts, workdir, cap, deadline=None):
    """Why the declared artifacts, paths relative to ``workdir``, break the output contract (the first that does), or
    None; checking them past ``deadline`` (a ``time.monotonic()``) is a violation too."""

    for artifact in artifacts:
        problem = _tree_problem(os.path.join(workdir, artifact), artifact, cap, deadline)
        if problem is not None:
            return problem
    return None


def layout_violation(outputs, workdir):
    """Why the submission holds something other than a directory above one of the replay's ``outputs`` (paths relative
    to ``workdir``), where the replayed output could not go back; None when it does not."""

    for output in outputs:
        path = workdir
        for part in output.split("/")[:-1]:
            path = os.path.join(path, part)
            try:
                info = os.lstat(path)
            except FileNotFoundError:
                break
            if not stat.S_ISDIR(info.st_mode):
                return f"{os.path.relpath(path, workdir)} is {_kind(info)}, where the replay writes {output}"
    return None


def _tree_problem(path, name, cap, deadline):
    try:
        info = os.lstat(path)
    except (FileNotFoundError, NotADirectoryError):
        return f"{name} is missing"
    if stat.S_ISREG(info.st_mode):
        return _file_problem(path, name, info, cap)
    if not stat.S_ISDIR(info.st_mode):
        return f"{name} is {_kind(info)}"
    for directory, directories, files in os.walk(path):
        for entry in sorted(directories + files):
            if _past(deadline):
                return "the artifacts take longer to check than the verifier's time limit allows"
            inner = os.path.join(directory, entry)
            shown = os.path.join(name, os.path.relpath(inner, path))
            entry_info = os.lstat(inner)
            if stat.S_ISREG(entry_info.st_mode):
                problem = _file_problem(inner, shown, entry_info, cap)
                if problem is not None:
                    return problem
            elif not stat.S_ISDIR(entry_info.st_mode):
                return f"{shown} is {_kind(entry_info)}"
    return None


def _file_problem(path, name, info, cap):
    """Why a regular file ``path`` breaks the contract: larger than ``cap``, or a ``.json`` file that is not JSON."""

    if info.st_size > cap:
        return f"{name} holds {info.st_size} bytes, more than an output may ({cap})"
    if path.endswith(".json"):
        try:
            json.loads(_read_regular(path, cap).decode("utf-8"))
        except RecursionError:
            return f"{name} is nested too deeply to read as JSON"
        except (OSError, ValueError) as exc:
            return f"{name} is not JSON: {exc}"
    return None


def _kind(info):
    if stat.S_ISLNK(info.st_mode):
        return "a symbolic link"
    if stat.S_ISDIR(info.st_mode):
        return "a directory"
    if stat.S_ISREG(info.st_mode):
        return "a file"
    return "not a regular file"


def _past(deadline):
    return deadline is not None and time.monotonic() > deadline


def _within(path, directory):
    return path == directory or path.startswith(directory.rstrip("/") + "/")


def _replay(config, paths, ops, report, scratch, submitted, deadline):
    """Run the task's replay in a copy of the working directory without its outputs, and put back what it wrote: the
    solver's 0 when it fails, leaves what the launcher cannot remove in its time or fills a filesystem, a protocol
    fault when its command (the image's) cannot start, None when it wrote every output again.  Before the replay the
    working directory is made the replay user's to write nothing in (:func:`Ops.harden`); after it, what the replay's
    user left where every user may write is removed, and its copy all but the outputs (which wait as root's), before
    the grader reads them.  A :class:`SolverFault` for a working directory too large to copy, or one that takes past
    ``deadline`` to copy."""

    replay, owner, cap = config["replay"], config["grader_owner"], int(config["max_output_bytes"])
    limits = {"skip": scratch, "deadline": deadline, "limit": int(config["max_copy_bytes"])}
    outputs = replay["outputs"]
    exposed = _exposed_directory(paths.workdir)
    if exposed is not None:  # the image's, or the template's: no file of a submission makes a directory so
        raise LauncherFault(
            f"{exposed}, in the verifier's working directory, lets the replay's user write in it, where it could put "
            "what is graded: give the template's image a working directory that only root writes in"
        )
    ops.harden(paths.workdir)  # nothing left there is the replay user's to write (files and directories, any depth)
    os.mkdir(submitted, 0o700)
    for output in outputs:
        here = os.path.join(paths.workdir, output)
        if os.path.lexists(here):
            target = os.path.join(submitted, output)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            _move(here, target, **limits)
    stage = os.path.join(scratch, "replay")
    os.mkdir(stage, 0o700)
    _copy_tree(paths.workdir, stage, **limits)
    try:
        ops.give(stage, REPLAY_UID)
    except OSError as exc:
        raise LauncherFault(f"it cannot give the replay's directory to its user: {exc}") from exc
    log = os.path.join(scratch, "replay.log")
    env = _replay_env(stage, config.get("replay_env", ()))
    try:
        preexec = ops.demote(cap, config.get("max_address_space_bytes"))
        ran = _run(replay["command"], cwd=stage, env=env, deadline=deadline, log=log, preexec=preexec)
    except OSError as exc:
        report["replay"] = {"error": str(exc)}
        if _submitted_program(replay["command"]):  # a program of the submission's, which it lacks or cannot run
            return Result.solver(NONREPRODUCIBLE, f"the replay's command cannot start: {exc}")
        return Result.invalid(GRADER_KINDS[owner] + "_protocol", f"the replay's command cannot start: {exc}")
    # What the replay left, a daemon that detached among it, ends before its outputs are read, wherever it went.
    ran["swept"], left = kill_left(lambda: ops.processes_of(REPLAY_UID))
    ran["output_tail"] = _tail(log)
    report["replay"] = ran
    if left:
        ran["survivors"] = left
        return Result.solver(NONREPRODUCIBLE, f"processes the replay left could not be killed ({left})")
    if ran["timed_out"]:
        return Result.solver(NONREPRODUCIBLE, f"the replay outlived its {ran['timeout_s']} s")
    if ran["exit_code"] != 0:
        return Result.solver(NONREPRODUCIBLE, f"the replay exited {ran['exit_code']}")
    for output in outputs:
        problem = _output_problem(stage, output, cap)
        if problem is not None:
            return Result.solver(NONREPRODUCIBLE, f"after the replay, {problem}")
    # What the replay left where every user may write is removed, and its copy all but the outputs, which then wait as
    # root's: a replay that left more than the launcher removes in its time, or that filled a filesystem, is the
    # solver's 0, never a launcher fault that would censor the measurement.  A filesystem found full from here on, as
    # the outputs are held or copied back, is judged one way: the solver's once the replay left files of its user
    # where every user may write, else the trusted side's, as it is before any replay ran, since by then what the
    # replay left in its copy is gone too.
    held = os.path.join(scratch, "replayed")
    until = max(deadline, time.monotonic()) + CLEAR_WINDOW_S
    cleared, where = 0, "where the launcher holds them"
    try:
        cleared = clear_left(paths.temporary, ops.replays_own, _writable_by_replay, scratch, until)
        if cleared:
            ran["cleared"] = cleared
        _hold(stage, held, outputs, ops, until)
        where = "in the working directory"
        for output in outputs:  # moved aside above, below nothing but directories (layout_violation); nothing else runs
            target = os.path.join(paths.workdir, output)
            _make_directories(os.path.dirname(target), 0o755)
            _copy_file(os.path.join(held, output), target, 0o644)
    except Leftover as exc:
        return Result.solver(NONREPRODUCIBLE, f"the replay left what the launcher cannot remove in its time: {exc}")
    except OSError as exc:
        if exc.errno not in _NO_ROOM or not cleared:
            raise  # a full filesystem with nothing of the replay's left is the trusted side's, as before any replay ran
        return Result.solver(
            NONREPRODUCIBLE,
            f"the replay's outputs find no room {where}, once the replay left files of its user where every user may "
            f"write: {exc}",
        )
    ran["outputs"] = list(outputs)
    report[REPLAYED_SHA256] = outputs_sha256(paths.workdir, outputs)  # before the grader, root, could change them
    try:
        remove_tree(held)
    except Leftover:
        pass  # root's copies of the outputs, which the grader finds in the working directory anyway
    return None


def outputs_sha256(directory, outputs):
    """The SHA-256 of the files ``outputs`` (paths relative to ``directory``): of each one's path and the SHA-256 of its
    bytes, sorted, with :data:`OUTPUTS_RULE` first.  Replays that wrote the same bytes give the same one."""

    digest = hashlib.sha256(OUTPUTS_RULE.encode("ascii"))
    for output in sorted(outputs):  # no path holds a NUL (relative_path)
        content = _file_sha256(os.path.join(directory, output))
        digest.update(b"\0" + os.fsencode(output) + b"\0" + content.encode("ascii"))
    return digest.hexdigest()


def _submitted_program(argv):
    """Whether the program ``argv`` runs is a path into the working directory, the submission's, rather than one the
    image has (an absolute path, or a name ``PATH`` finds)."""

    return "/" in argv[0] and not argv[0].startswith("/")


def _output_problem(stage, output, cap):
    """Why the replay's ``output`` (relative to its directory) is not a regular file the grader may take: missing,
    below something other than a directory, a link or a special file, linked elsewhere, too large, or a ``.json`` file
    that is not JSON.  Nothing of the replay runs by now, so nothing changes it while it is looked at."""

    path = stage
    for part in output.split("/")[:-1]:
        path = os.path.join(path, part)
        try:
            info = os.lstat(path)
        except FileNotFoundError:
            return f"{output} is missing"
        if not stat.S_ISDIR(info.st_mode):
            return f"{output} is below {_kind(info)}"
    path = os.path.join(stage, output)
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return f"{output} is missing"
    if not stat.S_ISREG(info.st_mode):
        return f"{output} is {_kind(info)}"
    if info.st_nlink != 1:
        return f"{output} has other links to it"
    return _file_problem(path, output, info, cap)


def _replay_env(stage, keep=()):
    """The replay's environment: the launcher's variables that :data:`REPLAY_VARIABLES` and ``keep`` (the settings'
    ``replay_env``) name, where it has them, its ``PATH`` cut to absolute entries alone (:func:`_absolute_path`, so
    nothing in the replay's own directory stands in for a program), and its own directory as its home and temporary
    directory; the verifier's other variables, which may hold credentials, stay out."""

    env = {name: os.environ[name] for name in (*REPLAY_VARIABLES, *keep) if name in os.environ}
    env["PATH"] = _absolute_path(env.get("PATH", ""))
    env.update(HOME=stage, TMPDIR=stage, PWD=stage)
    return env


def _absolute_path(value):
    """``value``, a ``PATH``, with only its absolute entries kept (no empty, ``.`` or relative entry, each of which a
    program would be looked for under the working directory, the replay's own): :data:`DEFAULT_PATH` where none is."""

    entries = [entry for entry in value.split(os.pathsep) if entry and entry != "." and os.path.isabs(entry)]
    return os.pathsep.join(entries) if entries else DEFAULT_PATH


def _grade(command, paths, env, report, deadline, owner, gen, umask):
    kind = GRADER_KINDS[owner]
    log = os.path.join(gen, GRADER_LOG)
    try:
        ran = _run(
            command, cwd=os.path.join(paths.tests, GEN), env=env, deadline=deadline, log=log, preexec=_umask(umask)
        )
    except OSError as exc:
        report["grader"] = {"error": str(exc)}
        return Result.invalid(kind + "_exit", f"the grader's command cannot start: {exc}")
    report["grader"] = ran
    try:
        return _graded(ran, report, kind, gen, log)
    except Exception as exc:  # the grader ran as root beside the launcher, and may have broken what it reads
        report["trace"] = traceback.format_exc()[-TAIL_CHARS:]
        return Result.invalid(f"{kind}_{UNFINISHED}", f"the launcher failed once the grader ran: {exc!r}")


def _graded(ran, report, kind, gen, log):
    """What the grader's run (``ran``) concluded, once it is over: its fault, or its verdict's score."""

    killed = kill_descendants()
    ran["log_tail"] = _tail(log)
    if not killed:
        return Result.invalid(kind + "_isolation", "processes the grader left could not be killed")
    if ran["timed_out"]:
        return Result.invalid(kind + "_timeout", f"the grader outlived its {ran['timeout_s']} s")
    if ran["exit_code"] != 0:
        return Result.invalid(kind + "_exit", f"the grader exited {ran['exit_code']}")
    verdict, problem = read_verdict(os.path.join(gen, VERDICT))
    if problem is not None:
        return Result.invalid(kind + "_protocol", problem)
    report["verdict"] = {"passed": verdict["passed"], "checks": verdict.get("checks", [])[:REPORTED_CHECKS]}
    return Result.scored(verdict["passed"])


def read_verdict(path):
    """The grader's verdict and None, or None and why it breaks the protocol."""

    try:
        info = os.lstat(path)
    except (FileNotFoundError, NotADirectoryError):
        return None, "the grader wrote no verdict"
    if not stat.S_ISREG(info.st_mode):
        return None, f"the verdict is {_kind(info)}"
    if info.st_size > MAX_VERDICT_BYTES:
        return None, f"the verdict holds more than {MAX_VERDICT_BYTES} bytes"
    try:
        verdict = json.loads(_read_regular(path, MAX_VERDICT_BYTES).decode("utf-8"))
    except RecursionError:
        return None, "the verdict is nested too deeply to read"
    except (OSError, ValueError) as exc:
        return None, f"the verdict is not JSON: {exc}"
    problem = verdict_problem(verdict)
    return (None, problem) if problem is not None else (verdict, None)


def verdict_problem(verdict):
    """Why a verdict is outside the schema, or None."""

    if not isinstance(verdict, dict):
        return "the verdict is not an object"
    unknown = sorted(set(verdict) - set(_VERDICT_KEYS))
    if unknown:
        return f"the verdict holds {', '.join(unknown)}, beyond passed and checks"
    if not isinstance(verdict.get("passed"), bool):
        return "the verdict's passed is not true or false"
    checks = verdict.get("checks", [])
    if not isinstance(checks, list) or len(checks) > MAX_CHECKS:
        return f"the verdict's checks are not a list of at most {MAX_CHECKS}"
    for index, check in enumerate(checks):
        if not isinstance(check, dict) or set(check) - set(_CHECK_KEYS):
            return f"check {index} is not an object of id, passed and margin"
        if not isinstance(check.get("id"), str) or not isinstance(check.get("passed"), bool):
            return f"check {index} needs an id and passed true or false"
        if len(check["id"]) > MAX_CHECK_ID_CHARS:
            return f"check {index}'s id holds more than {MAX_CHECK_ID_CHARS} characters"
        margin = check.get("margin")
        if margin is not None and (
            isinstance(margin, bool) or not isinstance(margin, (int, float)) or not math.isfinite(margin)
        ):
            return f"check {index}'s margin is not a finite number"
    return None


def _run(argv, cwd, env, deadline, log, preexec=None):
    """Run ``argv`` in a session of its own until it exits or ``deadline`` passes (then its process group is killed),
    its output into ``log``.  ``OSError`` when it cannot start; a :class:`LauncherFault` when ``preexec`` failed."""

    timeout = max(0.0, deadline - time.monotonic())
    started = time.monotonic()
    with open(log, "wb") as output:
        try:
            process = subprocess.Popen(
                argv,
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                close_fds=True,
                start_new_session=True,
                preexec_fn=preexec,
            )
        except subprocess.SubprocessError as exc:  # what runs before the command: switching to the replay's user
            raise LauncherFault(f"it cannot start {argv[0]} as the replay's user: {exc}") from exc
        timed_out = False
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                pass
            code = process.wait()
    return {
        "exit_code": code,
        "timed_out": timed_out,
        "seconds": round(time.monotonic() - started, 3),
        "timeout_s": round(timeout, 3),
    }


def _umask(mask):
    """What runs in the grader's process before its command: the verifier's own umask, not the launcher's."""

    def restore():
        os.umask(mask)

    return restore


def kill_descendants(window_s=KILL_WINDOW_S):
    """Kill every process below the launcher (:func:`kill_left`); whether none is left."""

    return kill_left(window_s=window_s)[1] == 0


def kill_left(also=None, window_s=KILL_WINDOW_S):
    """Kill every process below the launcher, which a daemon that detached stays below too (the launcher is their
    subreaper), and every process ``also()`` names (the replay user's, wherever they are), until a scan finds none or
    ``window_s`` passes, and reap them: how many it signalled, and how many were still alive at the end (0 unless some
    outran it).  A process below the launcher it may not signal (another user namespace's, in a view of ``/proc`` that
    holds more than the sandbox) is foreign, counted as neither; but a process ``also()`` names it may not signal is a
    survivor all the same, since the replay's user is its own and the launcher is root over it (the ``/proc`` view would
    not show another namespace's process as that user).  Run only once the launcher has waited for what it started."""

    end = time.monotonic() + window_s
    signalled, foreign = set(), set()
    while True:
        users = set(also()) if also is not None else set()  # the replay user's: never excluded as foreign
        tree = set(descendants(os.getpid())) - users
        alive = users | (tree - foreign)
        _reap()
        if not alive:
            return len(signalled), 0
        for pid in alive:
            try:
                _signal(pid)
                signalled.add(pid)
            except ProcessLookupError:
                pass
            except PermissionError:
                if pid not in users:  # a process below the launcher in a foreign namespace, not the replay's user
                    foreign.add(pid)
        if time.monotonic() > end:
            _reap()
            users = set(also()) if also is not None else set()
            tree = set(descendants(os.getpid())) - users
            return len(signalled), len(users | (tree - foreign))
        time.sleep(0.01)


def _signal(pid):
    os.kill(pid, signal.SIGKILL)


def descendants(root, proc="/proc"):
    """The live processes below ``root``, from ``proc``'s view of every process's parent."""

    children = {}
    for name in os.listdir(proc):
        if not name.isdigit():
            continue
        try:
            with open(os.path.join(proc, name, "stat")) as stream:
                fields = stream.read().rsplit(")", 1)[1].split()
        except (OSError, IndexError):
            continue
        if len(fields) < 2 or fields[0] in ("Z", "X", "x"):  # dead: only its entry is left, for its parent to reap
            continue
        children.setdefault(int(fields[1]), []).append(int(name))
    found, frontier = [], [root]
    while frontier:
        for child in children.get(frontier.pop(), ()):
            found.append(child)
            frontier.append(child)
    return found


def user_processes(uid, proc="/proc"):
    """The live processes that ``proc`` shows with ``uid`` among their user ids (real, effective, saved or
    filesystem), but for the launcher: never a zombie, whose entry is all that is left."""

    found = []
    for name in os.listdir(proc):
        if not name.isdigit() or int(name) == os.getpid():
            continue
        uids, state = [], []
        try:
            with open(os.path.join(proc, name, "status")) as stream:
                for line in stream:
                    key, _, value = line.partition(":")
                    if key == "Uid":
                        uids = value.split()
                    elif key == "State":
                        state = value.split()[:1]
        except OSError:  # it ended
            continue
        if str(uid) in uids and state not in (["Z"], ["X"], ["x"]):
            found.append(int(name))
    return found


def _reap():
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return


def _finish(paths, config, result, report):
    """Write the only reward file and the report into ``/logs/verifier``, after removing what else is there; 0 once
    both are written."""

    key = config["reward_key"] if config is not None else "reward"
    if result.failure_kind is None:
        reward = {key: result.reward}
    else:
        reward = {"invalid": 1, "failure_kind": result.failure_kind}
    report.update(reward=result.reward, failure_kind=result.failure_kind, reason=result.reason, detail=result.detail)
    try:
        _clear_logs(paths.logs)
        _write(os.path.join(paths.logs, REWARD), _dumps(reward))
        _write(os.path.join(paths.logs, REPORT), bounded_report(report))
    except OSError as exc:
        _say(f"cannot write the reward: {exc}")
        return 1
    _say(_dumps(reward))
    if result.detail:
        _say(result.detail[:TAIL_CHARS])
    return 0


def _say(text):
    """Print a line to the launcher's output (``test-stdout.txt``) as ASCII, so that an output whose encoding takes
    fewer characters than a detail holds (a file name of the submission's) never stops the launcher."""

    try:
        print("platoon launcher: " + text.encode("ascii", "backslashreplace").decode("ascii"), flush=True)
    except (OSError, ValueError):
        pass


def bounded_report(report, limit=MAX_REPORT_BYTES):
    """``report`` as JSON (ASCII) of at most ``limit`` bytes, so that grading always reads it whole: as it is where it
    fits, else without the verdict's checks, then without the artifacts' names and the replay's outputs, then with only
    what the outcome needs (the SHA-256 of the replayed outputs among it, which a regrade compares); ``truncated`` names
    what was left out."""

    text = _dumps(report)
    if len(text) <= limit:
        return text
    report = dict(report)
    left_out = []
    for part, key in (("verdict", "checks"), ("contract", "artifacts"), ("replay", "outputs")):
        value = report.get(part)
        if isinstance(value, dict) and key in value:
            report[part] = {name: item for name, item in value.items() if name != key}
            left_out.append(f"{part}.{key}")
            report["truncated"] = list(left_out)
            text = _dumps(report)
            if len(text) <= limit:
                return text
    kept = {key: report.get(key) for key in ("schema", "grader_owner", "reward", "failure_kind", "reason", "seconds")}
    if REPLAYED_SHA256 in report:
        kept[REPLAYED_SHA256] = report[REPLAYED_SHA256]
    detail = report.get("detail")
    kept.update(detail=detail[:TAIL_CHARS] if isinstance(detail, str) else None, truncated="all but the outcome")
    return _dumps(kept)


def _dumps(data):
    return json.dumps(data, sort_keys=True)


def _clear_logs(logs):
    """Leave only the launcher's output (``test-stdout.txt``) and ``gen/`` in ``logs``, each as what it should be."""

    if os.path.islink(logs) or (os.path.lexists(logs) and not os.path.isdir(logs)):
        os.unlink(logs)
    os.makedirs(logs, mode=0o700, exist_ok=True)
    for name in os.listdir(logs):
        path = os.path.join(logs, name)
        info = os.lstat(path)
        if (name == STDOUT and stat.S_ISREG(info.st_mode)) or (name == GEN and stat.S_ISDIR(info.st_mode)):
            continue
        if stat.S_ISDIR(info.st_mode):
            shutil.rmtree(path)
        else:
            os.unlink(path)


def _write(path, text):
    """Write ``text`` (ASCII) and a newline as the file ``path``, through a new temporary file beside it, never through
    a link, in a directory that holds nothing else (``_clear_logs`` emptied it)."""

    temporary = f"{path}.platoon-tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o644)
    with os.fdopen(descriptor, "w", encoding="ascii") as stream:
        stream.write(text + "\n")
    os.replace(temporary, path)


def _make_directories(path, mode):
    """Make ``path`` and the directories above it that are missing, each of ``mode`` whatever the umask."""

    if os.path.isdir(path):
        return
    _make_directories(os.path.dirname(path), mode)
    os.mkdir(path)
    os.chmod(path, mode)


def _fresh_directory(path):
    if os.path.islink(path) or (os.path.lexists(path) and not os.path.isdir(path)):
        os.unlink(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)
    os.mkdir(path, 0o700)


def _read_regular(path, limit):
    """The bytes of the regular file ``path``, at most ``limit`` of them.  It is opened without following a link or
    waiting on a FIFO, and checked on the open descriptor, so that nothing swapped in after a check is read: an
    ``OSError`` when it cannot be opened (a link among them), a ``ValueError`` when it is no regular file, holds more
    than ``limit`` bytes or grew while it was read."""

    with os.fdopen(os.open(path, _READ_FLAGS), "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"{os.path.basename(path)} is not a regular file")
        if info.st_size > limit:
            raise ValueError(f"{os.path.basename(path)} holds more than {limit} bytes")
        data = stream.read(info.st_size + 1)
    if len(data) > info.st_size:
        raise ValueError(f"{os.path.basename(path)} grew while it was read")
    return data


def _file_sha256(path):
    """The SHA-256 of the regular file ``path``'s bytes, opened as :func:`_read_regular` opens one: never through a
    link, never waiting on a FIFO, and checked on the open descriptor."""

    digest = hashlib.sha256()
    with os.fdopen(os.open(path, _READ_FLAGS), "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError(f"{os.path.basename(path)} is not a regular file")
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tail(path):
    """The last :data:`TAIL_CHARS` characters of the log ``path``, which what wrote it may have replaced: a regular
    file only, opened as :func:`_read_regular` opens one; empty for anything else."""

    try:
        with os.fdopen(os.open(path, _READ_FLAGS), "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode):
                return ""
            stream.seek(max(0, info.st_size - 4 * TAIL_CHARS))
            return stream.read(4 * TAIL_CHARS).decode("utf-8", "replace")[-TAIL_CHARS:]
    except OSError:
        return ""


def _copy_tree(source, target, skip, deadline=None, limit=None):
    """Copy the regular files and directories below ``source`` into ``target``, never following a link, and leaving
    out ``skip`` (the launcher's own directory, should ``source`` hold it).  A :class:`SolverFault` for files of more
    than ``limit`` bytes, which only a submission brings past the task's own files (``max_copy_bytes``), and for a copy
    that takes past ``deadline``, the replay's."""

    total = 0
    skipped = os.path.realpath(skip)
    for directory, directories, files in os.walk(source):
        relative = os.path.relpath(directory, source)
        into = target if relative == "." else os.path.join(target, relative)
        for name in list(directories):
            path = os.path.join(directory, name)
            if os.path.islink(path) or os.path.realpath(path) == skipped:
                directories.remove(name)
                continue
            os.mkdir(os.path.join(into, name), 0o755)
        for name in files:
            if _past(deadline):
                raise SolverFault(NONREPRODUCIBLE, "copying the working directory for the replay outlived its time")
            path = os.path.join(directory, name)
            info = os.lstat(path)
            if not stat.S_ISREG(info.st_mode):
                continue
            total += info.st_size
            if limit is not None and total > limit:
                raise SolverFault(
                    OUTPUT_CONTRACT, f"the working directory holds more than {limit} bytes for a replay to copy"
                )
            _copy_file(path, os.path.join(into, name), 0o755 if info.st_mode & stat.S_IXUSR else 0o644)


def _move(source, target, skip, deadline=None, limit=None):
    """Move ``source``, a regular file or a directory, to ``target``, which may be on another filesystem: the working
    directory and ``/tmp`` often are (``deadline`` and ``limit`` as :func:`_copy_tree` takes them)."""

    try:
        os.replace(source, target)
        return
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
    info = os.lstat(source)
    if stat.S_ISDIR(info.st_mode):
        os.mkdir(target, 0o700)
        _copy_tree(source, target, skip=skip, deadline=deadline, limit=limit)
        shutil.rmtree(source)
        return
    if stat.S_ISREG(info.st_mode):
        _copy_file(source, target, 0o755 if info.st_mode & stat.S_IXUSR else 0o644)
    os.unlink(source)


def _copy_file(source, target, mode):
    """Copy the regular file ``source`` to a new file ``target`` of ``mode``, never following a link."""

    with os.fdopen(os.open(source, _READ_FLAGS), "rb") as source_file:
        if not stat.S_ISREG(os.fstat(source_file.fileno()).st_mode):
            raise OSError(f"{source} is not a regular file")
        writing = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, mode)
        with os.fdopen(writing, "wb") as target_file:
            shutil.copyfileobj(source_file, target_file)
            os.fchmod(target_file.fileno(), mode)


def _chown_tree(path, uid, gid):
    os.lchown(path, uid, gid)
    for directory, directories, files in os.walk(path):
        for name in directories + files:
            os.lchown(os.path.join(directory, name), uid, gid)


def _harden_tree(path):
    """Make ``path`` and everything below it root's (:func:`_chown_tree`) and strip its group and other write bits
    (:func:`_strip_foreign_write`), never following a link: the replay's user, which is neither root nor a group of
    root's files, may then write nothing in the working directory."""

    _chown_tree(path, 0, 0)
    _strip_foreign_write(path)


def _strip_foreign_write(path):
    """Strip the group and other write bits of ``path`` and everything below it, but for the links (whose own mode
    grants nothing), never following a link."""

    for current in (path, *(os.path.join(d, name) for d, ds, fs in os.walk(path) for name in ds + fs)):
        info = os.lstat(current)
        if stat.S_ISLNK(info.st_mode):
            continue
        mode = stat.S_IMODE(info.st_mode)
        stripped = mode & ~(stat.S_IWGRP | stat.S_IWOTH)
        if stripped != mode:
            os.chmod(current, stripped)


def _hold(stage, held, outputs, ops, until):
    """Remove all but ``outputs`` and the directories above them from the replay's directory ``stage``
    (:func:`_prune`), which gives back the room the replay took there before anything that may need room runs; rename
    what is left to ``held``, beside it; and make it root's alone (files 600, directories 700), so a survivor of the
    replay, run as the same user, could neither reach nor change what the grader grades.  :class:`Leftover` as
    :func:`remove_tree` raises it; an ``OSError`` that found its filesystem full (:data:`_NO_ROOM`: a filesystem that
    copies on write may need room even for a rename or a change of owner) as it is, for the caller to attribute as it
    attributes the outputs' copies back (:func:`_replay`); a :class:`LauncherFault` for any other."""

    _prune(stage, outputs, _Clock(until))  # nothing of the replay's runs by now (the sweep), so nothing changes it
    try:
        os.rename(stage, held)
        for output in outputs:  # a regular file of one link below directories (_output_problem); nothing of its runs
            path = held
            for part in output.split("/")[:-1]:
                path = os.path.join(path, part)
                os.chmod(path, 0o700)
            os.chmod(os.path.join(held, output), 0o600)
        ops.seal(held)
    except OSError as exc:
        if exc.errno in _NO_ROOM:
            raise
        raise LauncherFault(f"it cannot hold what the replay wrote as root's: {exc}") from exc


def _prune(directory, keep, clock):
    """Remove everything below ``directory`` but the files ``keep`` names (paths relative to it) and the directories
    above them, never following a link (:func:`remove_tree`); :class:`Leftover` as it raises it."""

    kept = set(keep)
    above = {"/".join(path.split("/")[:end]) for path in keep for end in range(1, path.count("/") + 1)}
    pending = [""]
    while pending:
        relative = pending.pop()
        try:
            with os.scandir(os.path.join(directory, relative) if relative else directory) as entries:
                for entry in entries:
                    name = f"{relative}/{entry.name}" if relative else entry.name
                    if name in kept:
                        continue
                    if name in above and entry.is_dir(follow_symlinks=False):
                        pending.append(name)
                        continue
                    _remove(entry.path, clock)
        except OSError as exc:
            raise Leftover(f"{relative or 'its directory'} cannot be cleared: {exc}") from exc


def clear_left(directories, owned, writable, skip, until=None):
    """Remove what the replay's user left in ``directories`` (:data:`TEMPORARY`, where every user may write): each
    entry that ``owned`` says is that user's (by its ``os.lstat`` result), with all below it, in them or below them
    through directories that ``writable`` says that user may write in, but for ``skip`` (the launcher's own
    directory), never following a link (:func:`remove_tree`).  How many entries it removed; :class:`Leftover` for one
    it cannot remove, or not before ``until`` (a ``time.monotonic()``).  A directory the sandbox lacks is none."""

    clock = _Clock(until)
    skipped = os.path.realpath(skip)
    removed, seen = 0, set()
    for top in directories:
        top = os.path.realpath(top)  # one of the sandbox's own, which no user but root may replace
        if top in seen:
            continue
        seen.add(top)
        pending = [top]
        while pending:
            directory = pending.pop()
            try:
                entries = os.scandir(directory)
            except (FileNotFoundError, NotADirectoryError):
                continue
            with entries:
                for entry in entries:
                    clock.tick()
                    if entry.path == skipped:
                        continue
                    try:
                        info = entry.stat(follow_symlinks=False)
                    except FileNotFoundError:
                        continue
                    if owned(info):
                        _remove(entry.path, clock)
                        removed += 1
                    elif stat.S_ISDIR(info.st_mode) and writable(info):
                        pending.append(entry.path)
    return removed


def remove_tree(path, until=None):
    """Remove ``path``: a file, a link, or a directory and all below it, never following a link and through
    descriptors, so that the length of its paths never stops it.  :class:`Leftover` for what cannot be removed, a tree
    that nests more than :data:`MAX_TREE_DEPTH` levels deep, and once ``until`` (a ``time.monotonic()``, or None)
    passes, which it looks at every :data:`_CLEAR_BATCH` entries."""

    _remove(path, _Clock(until))


class _Clock:
    """Counts the entries a removal reaches, and stops it once ``until`` (a ``time.monotonic()``, or None) passes."""

    def __init__(self, until):
        self.until = until
        self.count = 0

    def tick(self):
        self.count += 1
        if self.until is not None and self.count % _CLEAR_BATCH == 0 and time.monotonic() > self.until:
            raise Leftover(f"its time to remove them was over {self.count} entries in")


def _remove(path, clock):
    parent, name = os.path.split(os.path.abspath(path))
    try:
        holder = os.open(parent, _DIRECTORY_FLAGS)
    except OSError as exc:
        raise Leftover(f"{path} cannot be removed: {exc}") from exc
    try:
        _remove_at(holder, name, clock)
    except OSError as exc:
        raise Leftover(f"{path} cannot be removed: {exc}") from exc
    finally:
        os.close(holder)


def _remove_at(holder, name, clock):
    """Remove the entry ``name`` of the directory open as ``holder``, and everything below it, depth first."""

    clock.tick()
    if not stat.S_ISDIR(os.stat(name, dir_fd=holder, follow_symlinks=False).st_mode):
        os.unlink(name, dir_fd=holder)
        return
    stack = [(holder, name, *_listing(holder, name))]
    try:
        while stack:
            parent, here, descriptor, entries = stack[-1]
            for entry in entries:
                clock.tick()
                if entry.is_dir(follow_symlinks=False):
                    if len(stack) >= MAX_TREE_DEPTH:
                        raise Leftover(f"{name} nests more than {MAX_TREE_DEPTH} levels deep")
                    stack.append((descriptor, entry.name, *_listing(descriptor, entry.name)))
                    break
                os.unlink(entry.name, dir_fd=descriptor)
            else:
                stack.pop()
                entries.close()
                os.close(descriptor)
                os.rmdir(here, dir_fd=parent)
    finally:
        for _, _, descriptor, entries in stack:
            entries.close()
            os.close(descriptor)


def _listing(holder, name):
    """The directory ``name`` of the directory open as ``holder``, opened (never through a link), and its entries."""

    descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=holder)
    try:
        return descriptor, os.scandir(descriptor)
    except OSError:
        os.close(descriptor)
        raise


def _limit_files(max_file_bytes):
    """No core files, and no file larger than ``max_file_bytes``."""

    _limit(resource.RLIMIT_CORE, 0)
    _limit(resource.RLIMIT_FSIZE, max_file_bytes + 1)


def _limit_replay(max_file_bytes, max_address_space_bytes=None):
    """The replay's limits, set in its process once it is the replay's user: :func:`_limit_files`, at most
    :data:`MAX_REPLAY_PROCESSES` processes of that user, threads counted, :data:`MAX_REPLAY_OPEN_FILES` open files, and,
    where ``max_address_space_bytes`` is given (the settings' ``max_address_space_bytes``), that much address space a
    process (``RLIMIT_AS``), so a single replay process cannot reserve the sandbox's memory out from under the
    launcher."""

    _limit_files(max_file_bytes)
    _limit(resource.RLIMIT_NPROC, MAX_REPLAY_PROCESSES)
    _limit(resource.RLIMIT_NOFILE, MAX_REPLAY_OPEN_FILES)
    if max_address_space_bytes is not None:
        _limit(resource.RLIMIT_AS, max_address_space_bytes)


def _oom_disfavor(value=OOM_SCORE_ADJ):
    """Make the OOM killer pick the replay's process first (``oom_score_adj`` :data:`OOM_SCORE_ADJ`, the most
    killable), so that memory the replay runs the sandbox out of ends the replay, not the launcher or the grader.
    Raising ``oom_score_adj`` needs no privilege; a sandbox that does not let it be raised is the trusted side's to
    fix, so this fails closed (the replay's command then does not start)."""

    with open(_OOM_SCORE_ADJ, "w") as stream:
        stream.write("%d\n" % value)


def _limit(limit, value):
    """Set the soft and hard ``limit`` to ``value``, or to the hard limit there was where that is lower: a process
    without privileges may lower its limits, never raise them."""

    hard = resource.getrlimit(limit)[1]
    if hard != resource.RLIM_INFINITY:
        value = min(value, hard)
    resource.setrlimit(limit, (value, value))


def _libc():
    import ctypes

    return ctypes.CDLL(None, use_errno=True)


def _prctl(option, value, libc=None):
    import ctypes

    if (libc or _libc()).prctl(option, value, 0, 0, 0) != 0:
        number = ctypes.get_errno()
        raise OSError(number, f"prctl({option}): {os.strerror(number)}")


if __name__ == "__main__":
    sys.exit(main())
