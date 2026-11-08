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
   with a **replay**, which runs in the working directory itself, at its own path, so that a submission that names its
   files by their absolute paths finds them there, it checks, failing closed before it changes anything
   (:func:`_in_place_fault`), that nothing is mounted below the working directory, that the replay's user may write in
   no directory of it, that it holds no socket or device, that no directory above it lets that user replace what it
   holds (:func:`_replaceable`, mode bits and access ACL alike) and that the launcher's own directory is not inside it;
   then it makes the working directory root's with the group and other write bits stripped, so nothing there is the
   replay's user's to write (:meth:`Ops.harden`).  It moves the replay's outputs out of it (into ``$PLATOON_SUBMITTED``,
   where the grader may compare them), copies what is left into a **pristine copy**, root's alone, that keeps it as it
   is (:func:`_snapshot`, :func:`_clone_tree`), lends the working directory to the replay's user (:data:`REPLAY_UID`,
   never root: :meth:`Ops.lend`), and runs the replay's command there as that user alone (no supplementary groups, no
   new privileges), with a thin environment (:data:`REPLAY_VARIABLES`, and the template's own variables that
   ``replay_env`` names, its ``PATH`` cut to absolute entries alone so nothing in the working directory stands in for a
   program, and its home and temporary directory one of its own in the launcher's), no core files, no file larger than
   an output may be, at most :data:`MAX_REPLAY_PROCESSES` processes of its user (threads count) and
   :data:`MAX_REPLAY_OPEN_FILES` open files, the most killable to the OOM killer (:data:`OOM_SCORE_ADJ`) and, where the
   settings give one, that much address space a process, for at most half the time left (or the settings'
   ``replay_timeout_s``, below), less :data:`PUT_BACK_COPIES` times as long as the copy took, which putting the working
   directory back takes.  It then kills what the replay left, below the launcher or of the replay's user (one of that
   user's it cannot signal is a survivor, not foreign), checks the outputs it wrote, removes what that user left where
   every user may write (:data:`TEMPORARY`) and its System V IPC objects (:func:`remove_ipc`), its own directory and all
   it left in the working directory but its outputs, which then wait as root's (:func:`_hold`, :func:`remove_tree`,
   within :data:`CLEAR_WINDOW_S` past its time), and **puts the working directory back** from the pristine copy
   (:func:`_restore`); once the replay checks below are done, it puts each output the replay wrote in its place:
   whatever the replays did there, the grader finds the working directory as the pristine copy holds it, and the
   replayed outputs, whose SHA-256 its report holds (:data:`REPLAYED_SHA256`), by which a regrade, which replays the
   submission again, tells a submission that computes other outputs each time it runs from a grader that judges the same
   outputs otherwise.  A violation, a submission too large to copy or too slow to check, copy or put back, a replay that
   fails, outlives its time, leaves processes that cannot be killed, leaves more than the launcher removes in its time
   or fills a filesystem, lacks its own program, and an output it does not write again score 0, the solver's fault
   (reason ``solver_output_contract`` or ``solver_nonreproducible``); nothing reads the working directory after a replay
   that ends the grade so, and the launcher leaves it as that replay left it.  A task may have **replay checks** (the
   settings' ``replay_checks``), whose data, the grader owner's, lies in ``tests/gen`` (:func:`load_checks`; data it
   cannot use is ``<grader>_protocol``, below).  Before the replay, it scans the submission's sources that the checks
   name for **forbidden literals**: a number a source states by :data:`LITERAL_SCAN` (:func:`source_literals`) within
   the tolerance of one of the task's scores 0 (reason ``solver_hardcoded``), and a source it cannot read as UTF-8
   Python (by the verifier's ``python3``), or one of more than :data:`MAX_SOURCE_BYTES`, breaks the contract.  Once the
   replay wrote every output again, it replays each **variant** as the replay ran (:func:`_run_replay`: in place, as the
   same user, with the same environment, limits, sweep and clearing), in the working directory as its pristine copy
   holds it with the variant's own inputs laid over it, and puts the directory back after each, while what the replay
   wrote waits in root's hands: the outputs a variant's replay writes must hold the numbers the variant expects, each
   within its tolerance, or the submission scores 0 (reason ``solver_variant_failed``).  The replays share the replay's
   half of the time left, each taking an equal part of what is left of it; with ``replay_timeout_s``, each takes at most
   that long instead, and all of them end by ``1 + len(variants)`` times it after the launcher started, the grader
   keeping the rest;
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
   one beyond the schema (``invalid`` among them: a grader cannot censor a measurement), the task's replay command
   cannot start, or the data of its replay checks cannot be used, ``<grader>_isolation`` when what it left cannot be
   killed, and ``<grader>_unfinished`` when the launcher fails once the grader ran (as root, the grader may have broken
   what the launcher reads), ``<grader>`` being ``verifier_generated`` for a grader the generator wrote and
   ``verifier_trusted`` for a trusted one (``grader_owner``); and ``verifier_trusted_launcher`` when the launcher cannot
   do its own part before the grader runs (it does not run as root, cannot seal, cannot hand the working directory to
   the replay's user or switch to it): a fault on the trusted side, never the task's.  It removes everything else from
   ``/logs/verifier`` but ``test-stdout.txt`` and ``gen/``, writes its report, ``launcher.json``, of at most
   :data:`MAX_REPORT_BYTES` (without the verdict's checks, then the artifacts' names, the replay's outputs, the
   variants' records and the scan's record, in that order, where it would be larger: :func:`bounded_report`), and exits
   0, which the Harbor service checks.

Every file whose contents it reads (its settings, the artifacts' JSON, the sources it scans, the replay checks' data,
the replays' outputs, the verdict, the logs whose tails it keeps) is opened without following a link or waiting on a
FIFO and checked on the open descriptor.
JSON there that nests more than :data:`MAX_JSON_DEPTH` levels deep is refused before it is parsed, as text that is not
JSON is (:func:`load_json`), and so the same way on every Python.
The launcher is no security boundary against the grader, which runs as root beside it: a grader may choose 0 or 1,
which it may anyway and which controls check, and anything else is a non-score that attribution charges to it.
"""

from __future__ import annotations

import ast
import bisect
import codecs
import contextlib
import errno
import hashlib
import heapq
import itertools
import json
import math
import os
import posixpath
import re
import resource
import shutil
import signal
import stat
import struct
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
#: Where every user of the verifier's sandbox may write: after each replay (the replay's, and each variant's), the
#: launcher removes what the replay's user left there (:func:`clear_left`), so that no later replay, nor the grader,
#: finds it there or goes without the room it took.  Besides the temporary directories, ``/run/lock`` (which Debian's
#: images give every user, ``/var/lock`` linking to it) and the POSIX message queues' ``/dev/mqueue``; one the sandbox
#: lacks is none, and one met twice through a link is cleared once.  A working directory in one of them is left to the
#: launcher, which puts it back itself (:func:`_restore`).
TEMPORARY = ("/tmp", "/var/tmp", "/dev/shm", "/run/lock", "/var/lock", "/dev/mqueue")
#: The user the replay runs as in the verifier's sandbox, and its group: no one else's, and never root (0).
REPLAY_UID = 65533
#: The share of the time left that the replay may take, with its variants' replays if it has any, copying the working
#: directory aside and putting it back after each replay included; the grader has the rest.  The settings'
#: ``replay_timeout_s`` gives each replay a time of its own instead.
REPLAY_SHARE = 0.5
#: How long putting the working directory back after each replay (the replay's, and each variant's) may take, in copies
#: of it as long each as copying it into its pristine copy took (:func:`_snapshot`): one copy back, and the walks that
#: remove what was lent there and make it root's again, about as long as another.  Each replay ends that long before its
#: time is over.
PUT_BACK_COPIES = 2
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
#: How deeply the JSON the launcher reads (its settings, a ``.json`` artifact or output, the replay checks' data, the
#: verdict) may nest arrays and objects: deeper JSON is refused before it is parsed (:func:`load_json`).  Pythons give
#: up parsing at depths of their own, all deeper than this (about 500 levels without the C parser, 1,000 with it before
#: 3.12, 10,000 on 3.12 and 3.13, and on 3.14 as many as the stack holds), so what is refused is the same on every
#: Python.
MAX_JSON_DEPTH = 256
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
#: How long past the replay's time the launcher may take to remove what the replay left, in the working directory, in
#: its own directory, of its user's where every user may write and its System V IPC objects, and to put the working
#: directory back: a replay that left more than that removes, or that takes longer to put back, is the solver's 0.
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
# How much of a file the launcher copies at once.
_COPY_CHUNK = 1 << 20
# The launcher's mounts, one a line, the fifth field of each its mount point, where the kernel writes a space, a tab, a
# newline and a backslash as an octal escape.
_MOUNTINFO = "/proc/self/mountinfo"
_OCTAL_ESCAPE = re.compile(rb"\\([0-7]{3})")
# The System V IPC objects of the launcher's IPC namespace: each kind's file in /proc/sysvipc and the column of its ids;
# and the command that removes one.
_SYSVIPC = (("shm", "shmid"), ("sem", "semid"), ("msg", "msqid"))
_IPC_RMID = 0
# A POSIX access ACL as Linux keeps it, in an extended attribute: a version, then entries of a tag, permissions and an
# id (little-endian); the tags of a named user, a named group and the mask, and the permission to write.
_ACL_ACCESS = "system.posix_acl_access"
_ACL_VERSION = 2
_ACL_USER, _ACL_GROUP, _ACL_MASK = 0x02, 0x08, 0x10
_ACL_WRITE = 0x02
OWNERS = ("generated", "trusted")
GRADER_KINDS = {"generated": "verifier_generated", "trusted": "verifier_trusted"}
#: The faults of a grader, each a failure kind ``<grader>_<fault>``; ``unfinished`` is also the Harbor service's, for a
#: launcher that did not finish (it exited nonzero, left no report, or outlived the verifier's time).
GRADER_FAULTS = ("exit", "timeout", "protocol", "isolation", "unfinished")
UNFINISHED = "unfinished"
LAUNCHER_FAULT = "verifier_trusted_launcher"
OUTPUT_CONTRACT = "solver_output_contract"
NONREPRODUCIBLE = "solver_nonreproducible"
#: The reasons of the solver's 0s that the replay checks give: a variant's replay that does not give what the variant
#: expects, and a source that states a forbidden literal.
VARIANT_FAILED = "solver_variant_failed"
HARDCODED = "solver_hardcoded"
#: The key of the report that holds the SHA-256 of the outputs the replay wrote, which the grader judged
#: (:func:`outputs_sha256`); a task without a replay leaves it out.
REPLAYED_SHA256 = "replayed_sha256"
OUTPUTS_RULE = "platoon-replayed-outputs-v1"
#: Where the data of a task's replay checks lies, relative to the task's directory: in ``tests/gen``, beside its grader,
#: sealed with ``/tests`` and never in the agents' sandbox.
CHECK_DATA = "tests/gen/"
#: The rule by which the forbidden-literals check reads a source (:func:`source_literals`); a task producer that checks
#: its own reference solution by the same rule knows that the reference states none.
LITERAL_SCAN = "python-numeric-literals/v1"
#: The largest source the scan reads: Python's parser takes up to about a kilobyte of memory for each byte of a source,
#: and the launcher parses beside the grader, so it stays small; a larger source breaks the output contract.
MAX_SOURCE_BYTES = 256 << 10
#: The most sources one scan reads, variants a task has, forbidden literals one list holds, keys one variant expects
#: and characters one key holds; the largest variant's expected outputs or forbidden-literals list read; the most a
#: variant's ``meta`` holds as JSON, and how deeply it nests.
MAX_SOURCES = 64
MAX_VARIANTS = 32
MAX_FORBIDDEN = 10_000
MAX_EXPECTED_KEYS = 10_000
MAX_KEY_CHARS = 2048
MAX_CHECK_BYTES = 1 << 20
MAX_META_BYTES = 8 << 10
MAX_META_DEPTH = 16
#: What the report keeps of a failed check: the first keys a variant's replay missed, and the first hits of the scan.
REPORTED_MISSES = 10
REPORTED_HITS = 10
#: How many literals the scan matches between two looks at the clock.
_SCAN_BATCH = 1024
_PR_SET_CHILD_SUBREAPER = 36
_PR_SET_NO_NEW_PRIVS = 38
_VERDICT_KEYS = ("passed", "checks")
_CHECK_KEYS = ("id", "passed", "margin")
_VARIABLE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
#: A variant's name, and what its settings and a forbidden literal hold.
VARIANT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_VARIANT_KEYS = ("name", "inputs", "expected", "tolerance", "meta")
_FORBIDDEN_KEYS = ("path", "sources", "scan")
_LITERAL_KEYS = ("value", "tolerance")
_INDEX = re.compile(r"0|[1-9][0-9]{0,17}")  # a JSON Pointer's array index, below any length a list could have
# What reading a source as Python raises for a source that is none (or no UTF-8), or one too deep or large for the
# parser.
_UNPARSABLE = (SyntaxError, ValueError, RecursionError, MemoryError, OverflowError, LookupError)
_SIGNS = (ast.USub, ast.UAdd)
# A coding line (PEP 263), and a line a coding line may follow, as Python's own tokenizer finds them.
_CODING_LINE = re.compile(rb"^[ \t\f]*#.*?coding[:=][ \t]*([-\w.]+)", re.ASCII)
_BLANK_LINE = re.compile(rb"^[ \t\f]*(?:[#\r\n]|$)", re.ASCII)
# What json_text_depth takes out of a JSON text before it counts brackets, in order: each escape (a backslash and the
# character after it), each string (or what follows a quote that no other closes), and all but the brackets.  None of
# them can backtrack, so each pass is linear in the text's length.
_JSON_ESCAPE = re.compile(r"\\.", re.DOTALL)
_JSON_STRING = re.compile(r'"[^"]*"?')
_NOT_BRACKET = re.compile(r"[^\[\]{}]+")
_BRACKET_STEP = {"[": 1, "{": 1, "]": -1, "}": -1}


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


class NestedTooDeeply(ValueError):
    """JSON that nests arrays and objects more than :data:`MAX_JSON_DEPTH` levels deep, refused before it is parsed
    (:func:`load_json`)."""


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

    def lend(self, workdir):
        """Lend the working directory ``workdir`` to the replay's user, for a replay in place: its directories and
        files below it become that user's (and the group of that number's), readable and writable by it, never
        following a link, a file of several links first copied into a file of its own, and the directory itself stays
        root's, of that user's group, which may write there (mode 770), so that user may add, change and remove what it
        holds, but can neither rename the directory nor change its mode (:func:`_lend_tree`)."""

        _lend_tree(workdir)

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

    def ipc_of(self, uid):
        """The System V IPC objects of ``uid`` (the replay's user), its own or ones it made, that ``/proc/sysvipc``
        shows in the launcher's IPC namespace, as ``(kind, id)`` pairs (:func:`user_ipc`)."""

        return user_ipc(uid)

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
        config = load_json(_read_regular(path, MAX_CONFIG_BYTES).decode("utf-8"))
    except (OSError, ValueError) as exc:
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
    checks = config.get("replay_checks")
    if checks is not None:
        why = "the task has no replay" if replay is None else checks_problem(checks)
        if why is not None:
            problems.append(f"replay_checks ({why})")
    if "replay_timeout_s" in config and not problems:
        why = replay_time_problem(config)
        if why is not None:
            problems.append(f"replay_timeout_s ({why})")
    if problems:
        raise LauncherFault(f"its settings {path} hold no usable {', '.join(problems)}")
    return config


def replay_time_problem(config):
    """Why the settings' ``replay_timeout_s``, the time each replay of the task takes at most (the replay's and each of
    its variants'), is none the launcher can give, or None: a finite number above 0, for a task with a replay, whose
    replays together leave the grader some of the time the launcher's steps have (:func:`_budget`)."""

    each = config.get("replay_timeout_s")
    if config.get("replay") is None:
        return "the task has no replay"
    if not number(each) or each <= 0:
        return f"{each!r} is no finite number of seconds above 0"
    count = 1 + len((config.get("replay_checks") or {}).get("variants", []))
    budget = _budget(float(config["timeout_s"]))
    if count * each >= budget:
        return f"{count} replays of {each:g} s leave the grader none of the {budget:g} s the launcher's steps have"
    return None


def checks_problem(checks):
    """Why ``checks``, the replay checks of a task's settings, are none the launcher can run, or None: ``{"variants":
    [{"name", "inputs", "expected", "tolerance": {"default", "keys"}, "meta"}], "forbidden_literals": {"path",
    "sources", "scan"} or null}``, as ``materialize`` writes them, every path of their data below :data:`CHECK_DATA`
    and every name of a variant its own."""

    if not isinstance(checks, dict) or sorted(checks) != ["forbidden_literals", "variants"]:
        return "they are not an object of variants and forbidden_literals"
    variants = checks["variants"]
    if not isinstance(variants, list) or len(variants) > MAX_VARIANTS:
        return f"their variants are not a list of at most {MAX_VARIANTS}"
    names = set()
    for index, variant in enumerate(variants):
        if not isinstance(variant, dict) or sorted(variant) != sorted(_VARIANT_KEYS):
            return f"variant {index} is not an object of {', '.join(_VARIANT_KEYS)}"
        name = variant["name"]
        if not isinstance(name, str) or not VARIANT_NAME.fullmatch(name) or name in names:
            return f"variant {index} has no name of its own"
        names.add(name)
        if not (check_data_path(variant["inputs"]) and check_data_path(variant["expected"])):
            return f"variant {name} names its inputs or expected outputs outside {CHECK_DATA}"
        tolerance = variant["tolerance"]
        if (
            not isinstance(tolerance, dict)
            or sorted(tolerance) != ["default", "keys"]
            or not tolerance_value(tolerance["default"])
            or not isinstance(tolerance["keys"], dict)
            or not all(isinstance(key, str) and tolerance_value(value) for key, value in tolerance["keys"].items())
        ):
            return f"variant {name} has no tolerance of a default and keys, each a finite number of 0 or more"
        if json_depth(variant["meta"]) > MAX_META_DEPTH:
            return f"variant {name}'s meta nests more than {MAX_META_DEPTH} levels deep"
    forbidden = checks["forbidden_literals"]
    if forbidden is not None:
        if not isinstance(forbidden, dict) or sorted(forbidden) != sorted(_FORBIDDEN_KEYS):
            return f"their forbidden_literals are not an object of {', '.join(_FORBIDDEN_KEYS)}"
        sources = forbidden["sources"]
        if not _paths(sources) or len(sources) > MAX_SOURCES or len(set(sources)) != len(sources):
            return f"their forbidden literals' sources are not at most {MAX_SOURCES} distinct relative paths"
        if not all(source.endswith(".py") for source in sources):
            return f"their forbidden literals' sources are not all Python (.py), which {LITERAL_SCAN} reads"
        if not check_data_path(forbidden["path"]):
            return f"their forbidden literals lie outside {CHECK_DATA}"
        if forbidden["scan"] != LITERAL_SCAN:
            return f"their forbidden literals are scanned by {forbidden['scan']!r}, not {LITERAL_SCAN}"
    return None


def check_data_path(path):
    """Whether ``path`` names data of a replay check: a normalized path relative to the task's directory, below
    :data:`CHECK_DATA`."""

    return relative_path(path) and path.startswith(CHECK_DATA) and len(path) <= MAX_KEY_CHARS


def tolerance_value(value):
    """Whether ``value`` is a tolerance: a finite number of 0 or more, never a boolean."""

    return number(value) and value >= 0


def number(value):
    """Whether ``value`` is a finite int or float, never a boolean (nor an int too large for a float)."""

    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def json_depth(value, limit=MAX_META_DEPTH):
    """How many levels of objects and arrays (dicts, lists and tuples) ``value``, JSON as Python holds it, nests,
    counted without recursion and only up to one past ``limit``."""

    deepest, pending = 0, [(value, 1)]
    while pending:
        item, depth = pending.pop()
        if isinstance(item, dict):
            items = item.values()
        elif isinstance(item, (list, tuple)):
            items = item
        else:
            continue
        deepest = max(deepest, depth)
        if depth <= limit:
            pending.extend((inner, depth + 1) for inner in items)
    return deepest


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
    checks, problem = load_checks(config.get("replay_checks"), replay, paths.tests)
    if problem is not None:  # the data of the task's replay checks, which the grader's owner wrote
        return Result.invalid(GRADER_KINDS[owner] + "_protocol", problem)
    if checks.forbidden is not None:
        found = _scan(checks, paths.workdir, report, deadline)
        if found is not None:
            return found
    env = dict(os.environ)
    env.update(PLATOON_WORKDIR=paths.workdir, PLATOON_VERDICT=os.path.join(gen, VERDICT))
    if replay is not None:
        submitted = os.path.join(scratch, "submitted")
        replays = Replays.of(config, started, deadline, len(checks.variants))
        try:
            found = _replay(config, paths, ops, report, scratch, submitted, replays, checks.variants)
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
        above = os.path.dirname(os.path.abspath(path))
        holder = os.stat(above)
    except OSError as exc:
        raise LauncherFault(f"it cannot seal {path}: {exc}") from exc
    mode = stat.S_IMODE(info.st_mode)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or mode & 0o077:
        raise LauncherFault(f"{path} is still open to other users once sealed (owner {info.st_uid}, mode {mode:o})")
    if _replaceable(holder, above):
        raise LauncherFault(
            f"the directory that holds {path} lets the replay's user replace it (owner {holder.st_uid}, group "
            f"{holder.st_gid}, mode {stat.S_IMODE(holder.st_mode):o})"
        )


def _replaceable(info, path=None):
    """Whether the replay's user could rename or remove what a directory (its ``os.stat`` result, and its ``path``
    where given) holds: it owns the directory, or may write there (:func:`_writable_by_replay`) and the sticky bit is
    not set."""

    return info.st_uid == REPLAY_UID or (_writable_by_replay(info, path) and not info.st_mode & stat.S_ISVTX)


def _writable_by_replay(info, path=None):
    """Whether the replay's user may add, rename or remove entries of a directory (its ``os.stat`` result): it owns
    it, or may write there as one of the other users or as its group, or, given the directory's ``path``, as its access
    ACL lets it (:func:`_acl_lets_replay_write`)."""

    if info.st_uid == REPLAY_UID or info.st_mode & stat.S_IWOTH:
        return True
    if info.st_gid == REPLAY_UID and info.st_mode & stat.S_IWGRP:
        return True
    return path is not None and _acl_lets_replay_write(path)


def _acl_lets_replay_write(path):
    """Whether the POSIX access ACL of ``path`` (never read through a link) lets the replay's user write there: an
    entry of that user, or of its group, that grants writing, which the ACL's mask lets through.  Not for a path without
    an ACL beyond its mode bits, on a filesystem without ACLs, or gone; an ACL the launcher cannot read lets it, so that
    a check that asks fails closed."""

    try:
        acl = os.getxattr(path, _ACL_ACCESS, follow_symlinks=False)
    except OSError as exc:
        if exc.errno in (errno.ENODATA, errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOENT):
            return False
        raise
    if len(acl) < 4 or (len(acl) - 4) % 8 or struct.unpack_from("<I", acl)[0] != _ACL_VERSION:
        return True
    entries = [struct.unpack_from("<HHI", acl, offset) for offset in range(4, len(acl), 8)]
    mask = next((permissions for tag, permissions, _ in entries if tag == _ACL_MASK), _ACL_WRITE)
    return any(
        tag in (_ACL_USER, _ACL_GROUP) and number == REPLAY_UID and permissions & mask & _ACL_WRITE
        for tag, permissions, number in entries
    )


def _unfit_entry(workdir):
    """The first entry of ``workdir`` (never reached through a link), the directory itself first, that a replay in
    place must not find there, and why: ``"writable"`` for a directory that the replay's user may write in
    (:func:`_writable_by_replay`), where it could put what the launcher puts back the replay's outputs through or the
    grader reads, and ``"special"`` for a socket or a device, which the replay could reach and the launcher could not
    copy and put back; None when there is none.  Only an image puts either there: a submission's files and directories,
    and a task's, reach the verifier as files and directories of their own."""

    pending = [workdir]
    while pending:
        directory = pending.pop()
        if _writable_by_replay(os.lstat(directory), directory):
            return directory, "writable"
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    pending.append(entry.path)
                elif not (entry.is_file(follow_symlinks=False) or entry.is_symlink()):
                    if not stat.S_ISFIFO(entry.stat(follow_symlinks=False).st_mode):
                        return entry.path, "special"
    return None


def mount_points(text):
    """The mount points that the text of ``/proc/self/mountinfo`` (bytes) names, each the fifth field of its line with
    the kernel's octal escapes (of a space, a tab, a newline and a backslash) read back."""

    points = []
    for line in text.splitlines():
        fields = line.split(b" ")
        if len(fields) > 4:
            points.append(os.fsdecode(_OCTAL_ESCAPE.sub(lambda found: bytes([int(found.group(1), 8)]), fields[4])))
    return points


def _mounted_below(workdir, mountinfo=_MOUNTINFO):
    """The first mount point strictly below ``workdir`` (a path without links) among the launcher's mounts
    (:func:`mount_points`), a directory or a file, whatever filesystem it mounts; None when there is none.  A
    :class:`LauncherFault` when it cannot read its mounts."""

    try:
        with open(mountinfo, "rb") as stream:
            text = stream.read()
    except OSError as exc:
        raise LauncherFault(f"it cannot read its mounts in {mountinfo}: {exc}") from exc
    for point in mount_points(text):
        if point != workdir and _within(point, workdir):
            return point
    return None


def _replaceable_above(workdir):
    """The first directory above ``workdir`` (a path without links), its parent first and ``/`` last, in which the
    replay's user could rename or remove what it holds (:func:`_replaceable`), with its ``os.lstat`` result; None when
    there is none.  Since the working directory stays root's while the replay's user writes in it (:meth:`Ops.lend`), a
    directory above it with the sticky bit set holds it as safely as one that user may not write in."""

    below, directory = workdir, os.path.dirname(workdir)
    while directory != below:
        info = os.lstat(directory)
        if _replaceable(info, directory):
            return directory, info
        below, directory = directory, os.path.dirname(directory)
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
            load_json(_read_regular(path, cap).decode("utf-8"))
        except NestedTooDeeply:
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


def _replay(config, paths, ops, report, scratch, submitted, replays, variants=()):
    """Run the task's replay in place, in the working directory at its own path without its outputs, then the replay
    of each of ``variants`` (:attr:`Checks.variants`) there too, and put the directory back as it was, with what the
    replay wrote: the solver's 0 when one fails, leaves what the launcher cannot remove in its time or fills a
    filesystem, a protocol fault when its command (the image's) cannot start, None when the replay wrote every output
    again and every variant's replay gave what the variant expects.  Before the replay the working directory is checked
    (:func:`_in_place_fault`) and made the replay user's to write nothing in (:meth:`Ops.harden`), its outputs move
    aside and what is left is copied into a pristine copy, root's alone (:func:`_snapshot`).  Each replay runs as
    :func:`_run_replay` runs it, in the working directory lent to the replay's user, and takes the time ``replays``
    gives it (:class:`Replays`); what each left is removed and the working directory put back from the pristine copy
    before the next one starts or the grader runs (:func:`_clear`), the replay's own outputs held as root's while the
    variants replay and put in their places after the last: whatever the replays did there, the grader finds the
    pristine copy and the replayed outputs.  A replay that ends the grade (one that fails, or a fault) leaves the
    working directory as it left it, since nothing reads it then.  A :class:`SolverFault` for a working directory too
    large to copy, or one that takes past the replay's time to copy or to put back."""

    outputs = config["replay"]["outputs"]
    deadline = replays.deadline()
    limits = {"skip": scratch, "deadline": deadline, "limit": int(config["max_copy_bytes"])}
    workdir = os.path.realpath(paths.workdir)  # no directory above it lets the replay's user move it (below)
    fault = _in_place_fault(workdir, scratch)
    if fault is not None:  # the image's, or the template's: no file of a submission makes a directory so
        raise LauncherFault(fault)
    ops.harden(workdir)  # nothing left there is the replay user's to write (files and directories, any depth)
    os.mkdir(submitted, 0o700)
    for output in outputs:
        here = os.path.join(workdir, output)
        if os.path.lexists(here):
            target = os.path.join(submitted, output)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            _move(here, target, **limits)
    pristine = os.path.join(scratch, "pristine")
    copying = time.monotonic()
    kept = _snapshot(workdir, pristine, ops, deadline, limits["limit"])
    # Each replay ends as long before its time is over as putting the directory back takes: PUT_BACK_COPIES copies.
    put_back = PUT_BACK_COPIES * (time.monotonic() - copying)
    place = _Place(workdir, pristine, kept, put_back, os.path.join(scratch, "home"))
    ran, found = _run_replay(config, paths, ops, place, os.path.join(scratch, "replay.log"), deadline, replays)
    report["replay"] = ran
    if found is not None:
        return found
    # What the replay left where every user may write, its System V IPC objects and its own directory are removed, and
    # in the working directory all but its outputs, which then wait as root's, where no variant's replay reaches or
    # changes them, while the directory is put back: a replay that left more than the launcher removes in its time, or
    # that filled a filesystem, is the solver's 0, never a launcher fault that would censor the measurement.  A
    # filesystem found full from here on, as the outputs are held, the directory is put back or the outputs are copied
    # back, or as a variant's inputs are laid over the directory or it is lent for a variant's replay, is judged one
    # way: the solver's once the replays left files of their user where every user may write or IPC objects, else the
    # trusted side's, as it is before any replay ran, since by then what the replays left in the working directory and
    # their own is gone too.  Before it judges one found putting the directory back so, the launcher has the filesystem
    # write out what it holds and tries once more: one that frees what was removed only after a while (XFS, btrfs) may
    # lack the room the prune gave back until then.
    held = os.path.join(scratch, "replayed")
    found = _clear(paths, ops, scratch, place, replays, ran, deadline, (NONREPRODUCIBLE, "the replay"), held, outputs)
    if found is not None:
        return found
    if variants:
        found = _replay_variants(config, paths, ops, report, scratch, place, replays, variants)
        if found is not None:
            return found
    try:
        for output in outputs:  # moved aside above, below nothing but directories (layout_violation); nothing else runs
            target = os.path.join(workdir, output)
            _make_directories(os.path.dirname(target), 0o755)
            _copy_file(os.path.join(held, output), target, 0o644)
    except OSError as exc:
        if exc.errno not in _NO_ROOM or not replays.cleared:
            raise  # a full filesystem with nothing of the replays' left is the trusted side's, as before any replay ran
        return Result.solver(
            NONREPRODUCIBLE,
            "the replay's outputs find no room in the working directory, once the replays left files of their user "
            f"where every user may write, or IPC objects: {exc}",
        )
    if _past(replays.until):  # what it put back is as it should be, but the last replay's time and window are over
        raise SolverFault(NONREPRODUCIBLE, "putting the working directory back outlived the replay's time")
    ran["outputs"] = list(outputs)
    report[REPLAYED_SHA256] = outputs_sha256(workdir, outputs)  # before the grader, root, could change them
    for copy in (held, pristine):  # root's copies of what the grader finds in the working directory anyway
        try:
            remove_tree(copy)
        except Leftover:
            pass
    return None


class _Place:
    """Where the replays of one grade run, one after the other, and what puts it back after each: the working
    directory itself (``workdir``, a path without links), its pristine copy (``pristine``, :func:`_snapshot`) and what
    the launcher keeps of the directory itself (``kept``, :class:`_Directory`), which :func:`_restore` takes; how long
    before its time each replay ends, so that putting the directory back after it fits in that time (``put_back``, in
    seconds: :data:`PUT_BACK_COPIES` times as long as copying it into its pristine copy took); and the replay's home and
    temporary directory (``home``), made before each replay and removed after it."""

    def __init__(self, workdir, pristine, kept, put_back, home):
        self.workdir = workdir
        self.pristine = pristine
        self.kept = kept
        self.put_back = put_back
        self.home = home


class Replays:
    """The replays of one grade, the replay's own and then its variants', one after the other: the time each takes
    (:meth:`deadline`); how many entries their user left where every user may write, and System V IPC objects it left,
    that the launcher removed (``cleared``), by which it tells a full filesystem their doing (:func:`_replay`); and when
    the time to put the working directory back after the last of them ends (``until``, :func:`_clear`)."""

    def __init__(self, end, count, each=None):
        self.end = end
        self.count = count
        self.each = each
        self.cleared = 0
        self.until = None

    @classmethod
    def of(cls, config, started, deadline, variants):
        """The replays of a task with ``variants`` variants whose launcher started at ``started`` and must be done by
        ``deadline`` (``time.monotonic()`` values): they share :data:`REPLAY_SHARE` of the time left, each an equal
        part of what is left of it; with the settings' ``replay_timeout_s``, each takes at most that long, and all end
        by ``1 + variants`` times it after the launcher started, which :func:`replay_time_problem` checked leaves the
        grader the rest."""

        count = 1 + variants
        each = config.get("replay_timeout_s")
        if each is None:
            now = time.monotonic()
            return cls(now + max(0.0, deadline - now) * REPLAY_SHARE, count)
        return cls(min(deadline, started + count * float(each)), count, float(each))

    def deadline(self):
        """The deadline (a ``time.monotonic()``) of the next replay: an equal part of what is left until ``end``, or,
        with ``each``, ``each`` seconds; never past ``end``."""

        now = time.monotonic()
        share = self.each if self.each is not None else max(0.0, self.end - now) / max(1, self.count)
        self.count -= 1
        return min(now + share, max(now, self.end))


def _run_replay(config, paths, ops, place, log, deadline, replays, variant=None):
    """Lay a ``variant``'s inputs over the working directory (``place``, :class:`_Place`), lend it to the replay's
    user with a home of its own (:meth:`Ops.lend`), run the replay's command there as that user, its output into
    ``log``, until ``place.put_back`` before ``deadline``, and kill what it left: a record of the run, and the solver's
    0 (``solver_nonreproducible``, for a variant ``solver_variant_failed``) where it fails, outlives its time, leaves
    what cannot be killed or does not write every output again, a protocol fault where its command (the image's)
    cannot start or a variant's inputs cannot be laid over the working directory, or None.  Every replay, the replay's
    and each variant's, runs this one way, in place, in the working directory as its pristine copy holds it: with the
    thin environment and its absolute ``PATH`` (:func:`_replay_env`), demoted with the replay's limits, address space
    and disfavor for the OOM killer (:meth:`Ops.demote`), and swept of every process of its user, none excluded as
    foreign (:func:`kill_left`).  A :class:`SolverFault` where the replay's time, less putting the directory back,
    leaves it none.  A variant's inputs, the file of its output, its home and its lending (a change of owner, or a file
    of several names copied into one of its own) that find their filesystem full once the replays' user left files
    where every user may write or IPC objects (``replays.cleared``) are the solver's 0 too: the room they lack is what
    those replays took."""

    replay, owner, cap = config["replay"], config["grader_owner"], int(config["max_output_bytes"])
    if variant is None:
        who, where, reason = "the replay", "", NONREPRODUCIBLE
    else:
        who, where, reason = (
            f"the replay of variant {variant['name']}",
            f" for variant {variant['name']}",
            VARIANT_FAILED,
        )
    workdir = place.workdir
    no_room = (
        f"{who} finds no room in the working directory, once the replays left files of their user where every user "
        "may write, or IPC objects"
    )
    if variant is not None:
        try:
            problem = _lay_over(_check_file(paths.tests, variant["inputs"]), workdir, config)
            if problem is None:
                open(log, "wb").close()  # its output's file, made now: no room for it is judged as for its inputs
        except OSError as exc:
            if exc.errno not in _NO_ROOM or not replays.cleared:
                raise
            return {"error": str(exc)}, Result.solver(reason, f"{no_room}: {exc}")
        if problem is not None:
            detail = f"variant {variant['name']}'s inputs cannot be laid over the working inputs: {problem}"
            return {}, Result.invalid(GRADER_KINDS[owner] + "_protocol", detail)
    ends = deadline - place.put_back
    if ends <= time.monotonic():
        if variant is None:
            raise SolverFault(reason, "copying the working directory for the replay, and back, leaves it no time")
        raise SolverFault(reason, f"putting the working directory back after {who} leaves it no time")
    try:
        os.mkdir(place.home, 0o700)
        ops.give(place.home, REPLAY_UID)
        ops.lend(workdir)
    except OSError as exc:
        if variant is not None and exc.errno in _NO_ROOM and replays.cleared:  # a change of owner may need room too
            return {"error": str(exc)}, Result.solver(reason, f"{no_room}: {exc}")
        raise LauncherFault(f"it cannot hand the working directory to the replay's user: {exc}") from exc
    env = _replay_env(workdir, place.home, config.get("replay_env", ()))
    try:
        preexec = ops.demote(cap, config.get("max_address_space_bytes"))
        ran = _run(replay["command"], cwd=workdir, env=env, deadline=ends, log=log, preexec=preexec)
    except OSError as exc:
        detail = f"the replay's command cannot start{where}: {exc}"
        if _submitted_program(replay["command"]):  # a program of the submission's, which it lacks or cannot run
            return {"error": str(exc)}, Result.solver(reason, detail)
        return {"error": str(exc)}, Result.invalid(GRADER_KINDS[owner] + "_protocol", detail)
    # What the replay left, a daemon that detached among it, ends before its outputs are read, wherever it went.
    ran["swept"], left = kill_left(lambda: ops.processes_of(REPLAY_UID))
    ran["output_tail"] = _tail(log)
    with contextlib.suppress(OSError):  # its tail is kept: the rest takes room the next replays may need
        os.unlink(log)
    if left:
        ran["survivors"] = left
        return ran, Result.solver(reason, f"processes {who} left could not be killed ({left})")
    if ran["timed_out"]:
        return ran, Result.solver(reason, f"{who} outlived its {ran['timeout_s']} s")
    if ran["exit_code"] != 0:
        return ran, Result.solver(reason, f"{who} exited {ran['exit_code']}")
    for output in replay["outputs"]:
        problem = _output_problem(workdir, output, cap)
        if problem is not None:
            return ran, Result.solver(reason, f"after {who}, {problem}")
    return ran, None


def _replay_variants(config, paths, ops, report, scratch, place, replays, variants):
    """Replay each variant (:func:`_run_replay`), one after the other, each in the working directory as its pristine
    copy holds it with the variant's inputs laid over it, match what it wrote to what it expects, then remove what it
    left and put the directory back (:func:`_clear`): the solver's 0 (``solver_variant_failed``) for the first that
    fails, misses, leaves what cannot be removed or takes too long to put back, else None.  The report lists each
    variant that ran; the one that failed keeps its ``meta``, the tail of its output and the keys it missed."""

    ran_variants = report.setdefault("replay_checks", {}).setdefault("variants", [])
    cap = int(config["max_output_bytes"])
    for number, (variant, expected) in enumerate(variants):
        deadline = replays.deadline()
        who = f"the replay of variant {variant['name']}"
        log = os.path.join(scratch, f"variant-{number}.log")
        ran, missed = {}, []
        try:
            ran, found = _run_replay(config, paths, ops, place, log, deadline, replays, variant=variant)
            if found is None:  # nothing of its replay runs by now (the sweep), so nothing changes what it wrote
                missed = missed_keys(place.workdir, expected, cap)
                if missed:
                    shown = ", ".join(missed[:3]) + (f" and {len(missed) - 3} more" if len(missed) > 3 else "")
                    found = Result.solver(
                        VARIANT_FAILED, f"{who} gave {shown} outside the tolerance of what the variant expects"
                    )
            if found is None:
                found = _clear(paths, ops, scratch, place, replays, ran, deadline, (VARIANT_FAILED, who))
        except SolverFault as exc:  # no time for it, or to put the directory back after it: the variant's, as its own
            found = Result.solver(VARIANT_FAILED if exc.reason == NONREPRODUCIBLE else exc.reason, str(exc))
        entry = {"name": variant["name"], "passed": found is None}
        entry.update(
            (key, ran[key])
            for key in ("exit_code", "timed_out", "seconds", "timeout_s", "swept", "cleared")
            if key in ran
        )
        ran_variants.append(entry)
        if found is not None:
            entry.update(meta=variant["meta"], missed=missed[:REPORTED_MISSES], output_tail=ran.get("output_tail", ""))
            entry.update((key, ran[key]) for key in ("survivors", "error") if key in ran)
            return found
    return None


def _clear(paths, ops, scratch, place, replays, ran, deadline, fault, held=None, outputs=()):
    """Remove what a replay that ran in the working directory until ``deadline`` left, and put the directory back from
    its pristine copy, for at most :data:`CLEAR_WINDOW_S` past that (``replays.until``): what its user left where every
    user may write and its System V IPC objects (:func:`clear_left`, :func:`_clear_ipc`, counted in ``ran["cleared"]``
    and ``replays.cleared``), its home, and all it left in the working directory, with ``held`` all but ``outputs``,
    which then wait in ``held`` as root's (:func:`_hold`), before the directory is put back (:func:`_restore`).  The
    solver's 0 (``fault``: its reason and who left it) where it cannot remove them, or not in that time, and where
    holding the outputs or putting the directory back finds the filesystem full once the replays left files of their
    user where every user may write or IPC objects; else None.  A full filesystem with nothing of the replays' left is
    raised, the trusted side's, as it is before any replay ran, once the filesystem wrote out what it holds and putting
    the directory back was tried once more (:func:`_sync`).  A :class:`SolverFault` (``fault``'s reason) where putting
    the directory back takes past that time."""

    until = max(deadline, time.monotonic()) + CLEAR_WINDOW_S
    replays.until = until
    reason, who = fault
    workdir = place.workdir
    no_room = f"{who}'s outputs find no room where the launcher holds them"
    try:
        cleared = clear_left(paths.temporary, ops.replays_own, _writable_by_replay, (scratch, workdir), until)
        cleared += _clear_ipc(ops)
        if cleared:
            ran["cleared"] = cleared
            replays.cleared += cleared
        remove_tree(place.home, until)
        if held is not None:
            _hold(workdir, held, outputs, ops, until)
        no_room = "the working directory finds no room as the launcher puts it back"
        try:
            _restore(workdir, place.pristine, ops, place.kept, until)
        except OSError as exc:
            if exc.errno not in _NO_ROOM or replays.cleared:
                raise
            _sync(workdir)
            _restore(workdir, place.pristine, ops, place.kept, until)
    except Leftover as exc:
        return Result.solver(reason, f"{who} left what the launcher cannot remove in its time: {exc}")
    except SolverFault as exc:  # _restore's: a copy back that outlived the time
        raise SolverFault(reason, f"putting the working directory back outlived {who}'s time") from exc
    except OSError as exc:
        if exc.errno not in _NO_ROOM or not replays.cleared:
            raise  # a full filesystem with nothing of the replays' left is the trusted side's, as before any replay ran
        return Result.solver(
            reason,
            f"{no_room}, once the replays left files of their user where every user may write, or IPC objects: {exc}",
        )
    if _past(until):  # what it put back is as it should be, but the replay's time, and the window past it, are over
        raise SolverFault(reason, f"putting the working directory back outlived {who}'s time")
    return None


class Checks:
    """A task's replay checks with their data read (:func:`load_checks`): ``variants``, each a variant's settings and
    what it expects (:func:`expected_keys`); ``forbidden``, the forbidden literals as ``(value, tolerance)`` pairs, or
    None; and ``sources``, the submission's files the scan reads."""

    def __init__(self, variants=(), forbidden=None, sources=()):
        self.variants = list(variants)
        self.forbidden = forbidden
        self.sources = list(sources)


def load_checks(settings, replay, tests):
    """The replay checks of a task (``settings``, the ``replay_checks`` that :func:`read_config` checked; None: there
    are none) with their data read from ``tests``, the task's ``/tests``, and None; or no checks and why the data,
    which the grader's owner wrote, cannot be used: a variant's inputs that are no directory, expected outputs that
    are no object of the replay's output keys (``replay``'s outputs) and numbers, a key of a tolerance they lack, and
    forbidden literals that are no list of ``{value, tolerance}``."""

    if settings is None:
        return Checks(), None
    outputs = replay["outputs"]
    variants = []
    for variant in settings["variants"]:
        name = variant["name"]
        try:
            info = os.lstat(_check_file(tests, variant["inputs"]))
        except OSError as exc:
            return Checks(), f"variant {name}'s inputs, {variant['inputs']}, cannot be read: {exc}"
        if not stat.S_ISDIR(info.st_mode):
            return Checks(), f"variant {name}'s inputs, {variant['inputs']}, are {_kind(info)}, not a directory"
        expected, problem = _check_json(tests, variant["expected"])
        if problem is None:
            problem = expected_problem(expected, outputs)
        if problem is None and (unknown := sorted(set(variant["tolerance"]["keys"]) - set(expected))):
            problem = f"lack {_shown(unknown[0])}, which the variant's tolerance names"
        if problem is not None:
            return Checks(), f"variant {name}'s expected outputs, {variant['expected']}, {problem}"
        variants.append((variant, expected_keys(expected, outputs, variant["tolerance"])))
    forbidden = settings["forbidden_literals"]
    if forbidden is None:
        return Checks(variants), None
    entries, problem = _check_json(tests, forbidden["path"])
    if problem is None:
        problem = forbidden_problem(entries)
    if problem is not None:
        return Checks(), f"the forbidden literals, {forbidden['path']}, {problem}"
    pairs = [(entry["value"], entry["tolerance"]) for entry in entries]
    return Checks(variants, pairs, forbidden["sources"]), None


def expected_problem(expected, outputs):
    """Why ``expected``, a variant's expected outputs read as JSON, is no object that maps from 1 to
    :data:`MAX_EXPECTED_KEYS` output keys of the replay (:func:`output_key`, each naming one of ``outputs``, a ``.json``
    file, which the output contract reads as JSON) to a finite number or a list of them; None when it is one."""

    if not isinstance(expected, dict) or not expected or len(expected) > MAX_EXPECTED_KEYS:
        return f"are no object of 1 to {MAX_EXPECTED_KEYS} output keys, each with the number expected there"
    for key, value in expected.items():
        named = output_key(key, outputs)
        if named is None:
            return (
                f"hold the key {_shown(key)}, which names no output of the replay: an output's path, or the path, # "
                "and a JSON Pointer into it (answer.json#/rows/0)"
            )
        if not named[0].endswith(".json"):
            return f"hold the key {_shown(key)}, whose output {named[0]} is no .json file, which a variant reads"
        if not (number(value) or (isinstance(value, list) and all(number(item) for item in value))):
            return f"expect {_shown(key)} to be {_shown(value)}, which is no finite number or list of them"
    return None


def forbidden_problem(entries):
    """Why ``entries``, a task's forbidden literals read as JSON, is no list of at most :data:`MAX_FORBIDDEN` objects
    of a finite ``value`` and a ``tolerance`` of 0 or more; None when it is one."""

    if not isinstance(entries, list) or len(entries) > MAX_FORBIDDEN:
        return f"are no list of at most {MAX_FORBIDDEN} objects of value and tolerance"
    for index, entry in enumerate(entries):
        if (
            not isinstance(entry, dict)
            or sorted(entry) != sorted(_LITERAL_KEYS)
            or not number(entry["value"])
            or not tolerance_value(entry["tolerance"])
        ):
            return f"hold, as literal {index}, no object of a finite value and a tolerance of 0 or more"
    return None


def output_key(key, outputs):
    """The output (one of ``outputs``, the replay's) and the reference tokens of the RFC 6901 JSON Pointer into it that
    an output key names: an output's path alone, for the whole of its JSON, or its path, ``#`` and a pointer
    (``answer.json#/rows/0``).  None for a key that names no output, or more than one."""

    if not isinstance(key, str) or not key or len(key) > MAX_KEY_CHARS:
        return None
    found = []
    for output in outputs:
        if key == output:
            found.append((output, []))
        elif key.startswith(output + "#"):
            tokens = pointer_tokens(key[len(output) + 1 :])
            if tokens is not None:
                found.append((output, tokens))
    return found[0] if len(found) == 1 else None


def pointer_tokens(pointer):
    """The reference tokens of the JSON Pointer ``pointer`` (RFC 6901: empty, or ``/`` before each token, in which
    ``~1`` stands for ``/`` and ``~0`` for ``~``), or None for a string that is none."""

    if pointer == "":
        return []
    if not pointer.startswith("/"):
        return None
    tokens = pointer[1:].split("/")
    if any(re.search(r"~(?![01])", token) for token in tokens):
        return None
    return [token.replace("~1", "/").replace("~0", "~") for token in tokens]


def resolve(document, tokens):
    """Whether ``tokens`` (:func:`pointer_tokens`) point to something in ``document``, and what: an object's member by
    its name, an array's item by its index (``0``, ``12``; never ``-`` or ``012``)."""

    value = document
    for token in tokens:
        if isinstance(value, dict) and token in value:
            value = value[token]
        elif isinstance(value, list) and _INDEX.fullmatch(token) and int(token) < len(value):
            value = value[int(token)]
        else:
            return False, None
    return True, value


def expected_keys(expected, outputs, tolerance):
    """What a variant expects (``expected``, which :func:`expected_problem` found good), key by key: the key, its
    output and pointer's tokens (:func:`output_key`), the value expected and its tolerance (``tolerance``'s for the
    key, else its default)."""

    return [
        (key, *output_key(key, outputs), value, tolerance["keys"].get(key, tolerance["default"]))
        for key, value in sorted(expected.items())
    ]


def missed_keys(directory, expected, cap):
    """The keys of ``expected`` (:func:`expected_keys`) whose values the outputs in ``directory`` do not hold within
    their tolerances: an output that is no JSON of at most ``cap`` bytes (:func:`load_json`), nothing at the key, no
    number there (no list of as many numbers, for a list), or one too far from what is expected."""

    documents, missed = {}, []
    for key, output, tokens, value, tolerance in expected:
        if output not in documents:
            try:
                documents[output] = (True, load_json(_read_regular(os.path.join(directory, output), cap).decode()))
            except (OSError, ValueError):
                documents[output] = (False, None)
        readable, document = documents[output]
        found, actual = resolve(document, tokens) if readable else (False, None)
        if not (found and within(actual, value, tolerance)):
            missed.append(key)
    return missed


def within(actual, expected, tolerance):
    """Whether ``actual`` holds ``expected``, a number or a list of numbers, within ``tolerance``: a number at most
    ``tolerance`` from it, or a list of as many numbers, each at most ``tolerance`` from its own."""

    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_close(item, wanted, tolerance) for item, wanted in zip(actual, expected))
        )
    return _close(actual, expected, tolerance)


def _close(actual, expected, tolerance):
    try:
        return number(actual) and abs(actual - expected) <= tolerance
    except OverflowError:
        return False


def _check_file(tests, path):
    """Where a path of the replay checks' data (relative to the task's directory, below :data:`CHECK_DATA`) is in the
    verifier: below ``tests``, the task's ``/tests``."""

    return os.path.join(tests, path[len("tests/") :])


def _check_json(tests, path):
    """The JSON of a file of the replay checks' data (:func:`_check_file`), at most :data:`MAX_CHECK_BYTES`, and None;
    or None and why it cannot be read."""

    try:
        return load_json(_read_regular(_check_file(tests, path), MAX_CHECK_BYTES).decode("utf-8")), None
    except NestedTooDeeply:
        return None, "are nested too deeply to read as JSON"
    except (OSError, ValueError) as exc:
        return None, f"cannot be read as JSON: {exc}"


def overlay_conflict(path, directory, protected):
    """Why a variant's input at ``path`` (relative to the working directory, a ``directory`` or a file) would change
    what the submission holds or the replay writes (``protected``: the artifacts and the replay's outputs), or None: a
    file at one of their paths, below one or above one, or a directory at or below one."""

    for other in protected:
        if path == other or path.startswith(other + "/") or (not directory and other.startswith(path + "/")):
            return f"{path} would change {other}, which the submission holds or the replay writes"
    return None


def _lay_over(source, target, config):
    """Copy the regular files and directories below ``source``, a variant's inputs, over ``target``, the working
    directory as its pristine copy holds it, as root, never following a link: each file in place of the one at its
    path, executable where the input is.  Why it cannot, or None: an input that is a link or a special file, a
    directory where the working directory holds anything else (a link among it) or the reverse, or one that would change
    the submission (:func:`overlay_conflict`)."""

    protected = [*config["artifacts"], *config["replay"]["outputs"]]
    for directory, directories, files in os.walk(source):
        relative = os.path.relpath(directory, source)
        for name in sorted(directories + files):
            path = os.path.join(directory, name)
            shown = name if relative == "." else os.path.join(relative, name)
            info = os.lstat(path)
            if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
                return f"{shown} is {_kind(info)}"
            conflict = overlay_conflict(shown, stat.S_ISDIR(info.st_mode), protected)
            if conflict is not None:
                return conflict
            there = os.path.join(target, shown)
            try:
                present = os.lstat(there)
            except FileNotFoundError:
                present = None
            if stat.S_ISDIR(info.st_mode):
                if present is None:
                    os.mkdir(there, 0o755)
                elif not stat.S_ISDIR(present.st_mode):
                    return f"{shown} is a directory of the variant's and a file of the working directory's"
                continue
            if present is not None:
                if stat.S_ISDIR(present.st_mode):
                    return f"{shown} is a file of the variant's and a directory of the working directory's"
                os.unlink(there)
            _copy_file(path, there, 0o755 if info.st_mode & stat.S_IXUSR else 0o644)
    return None


def source_text(data):
    """The text of a source the scan reads, from its bytes: UTF-8, with or without a byte-order mark, as a coding line
    on one of its first two lines (PEP 263), if it has one, must say, so that the scan never runs another codec, some
    of which are slow; the coding line is found without decoding anything.  A ``ValueError`` for a coding line of
    another encoding, or bytes that are no UTF-8."""

    body = data[len(codecs.BOM_UTF8) :] if data.startswith(codecs.BOM_UTF8) else data
    for number, line in enumerate(body.split(b"\n", 2)[:2]):
        found = _CODING_LINE.match(line)
        if found is not None:
            name = found.group(1).decode("ascii")
            try:
                canonical = codecs.lookup(name).name  # finds the codec, and runs none of it
            except LookupError:
                raise ValueError(f"its coding line names {name}, a codec Python lacks") from None
            if canonical not in ("utf-8", "utf-8-sig"):  # Python reads the one as the other
                raise ValueError(f"its coding line names {name}, and the scan reads UTF-8 alone")
            break
        if number == 0 and not _BLANK_LINE.match(line):  # a coding line on the second line follows a blank or comment
            break
    return body.decode("utf-8")


def source_literals(source):
    """The numbers that ``source``, Python (text, or bytes read as :func:`source_text` reads them), states by
    :data:`LITERAL_SCAN`, each with its line: each constant's numbers (:func:`constant_numbers`), and for an int,
    float or complex constant under a chain of unary ``-`` and ``+`` the numbers of the whole chain, negative where it
    holds an odd number of ``-`` (``--5.46`` and ``-(-5.46)`` state 5.46, ``+-5.46`` states -5.46), on the line where
    the chain starts.  Comments and the rest of the text are not read, so a source may cite a published value in a
    comment, and nothing is evaluated (``-~5`` states 5).  What ``ast.parse`` raises for a source that is not
    Python."""

    tree = ast.parse(source if isinstance(source, str) else source_text(source))
    found, signed, inner = [], set(), set()
    for node in ast.walk(tree):  # breadth first: a chain's outermost sign comes before the signs below it
        if not (isinstance(node, ast.UnaryOp) and isinstance(node.op, _SIGNS)) or id(node) in inner:
            continue
        negative, operand = isinstance(node.op, ast.USub), node.operand
        while isinstance(operand, ast.UnaryOp) and isinstance(operand.op, _SIGNS):
            inner.add(id(operand))
            negative ^= isinstance(operand.op, ast.USub)
            operand = operand.operand
        if isinstance(operand, ast.Constant) and _signable(operand.value):
            signed.add(id(operand))
            found += [(node.lineno, -value if negative else value) for value in constant_numbers(operand.value)]
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and id(node) not in signed:
            found += [(node.lineno, value) for value in constant_numbers(node.value)]
    return found


def constant_numbers(value):
    """The finite numbers a constant of a source states by :data:`LITERAL_SCAN`: an int or a float itself (never a
    bool, nor an int too large for a float); a complex number's imaginary part, and its real part where it is not 0;
    and what ``float()`` reads, once stripped, of a string, or of bytes that are ASCII, whatever their length."""

    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [value] if number(value) else []
    if isinstance(value, complex):
        return [part for part in ((value.real,) if value.real else ()) + (value.imag,) if math.isfinite(part)]
    if isinstance(value, bytes):
        try:
            value = value.decode("ascii")
        except UnicodeDecodeError:
            return []
    if not isinstance(value, str):
        return []
    try:
        parsed = float(value.strip())
    except ValueError:
        return []
    return [parsed] if math.isfinite(parsed) else []


def _signable(value):
    """Whether a constant takes a sign: an int, a float or a complex number (never a bool, a string or bytes)."""

    return isinstance(value, (int, float, complex)) and not isinstance(value, bool)


class Windows:
    """The windows of forbidden literals, each ``value ± tolerance`` of a ``(value, tolerance)`` pair, widened by a
    margin for rounding.  Their ends cut the numbers into pieces, the ends themselves and the open intervals between
    them, and each piece knows the first window it lies in, so that :meth:`first` takes a binary search and one exact
    test of a match (:func:`_close`), whatever the windows and however they overlap."""

    def __init__(self, forbidden):
        self.forbidden = list(forbidden)
        windows = [_window(value, tolerance) for value, tolerance in self.forbidden]
        self.ends = sorted({end for window in windows for end in window})
        # Piece 2 * j + 1 is self.ends[j]; piece 2 * j the open interval below it, above the end before it; piece
        # 2 * len(self.ends) the numbers above every end.  A window covers the pieces from its low end to its high one.
        spans = sorted(
            (2 * bisect.bisect_left(self.ends, low) + 1, 2 * bisect.bisect_left(self.ends, high) + 1, index)
            for index, (low, high) in enumerate(windows)
        )
        self.firsts = [None] * (2 * len(self.ends) + 1)
        covering, next_span = [], 0  # a heap of (index, last piece) of the windows that began by the piece at hand
        for piece in range(len(self.firsts)):
            while next_span < len(spans) and spans[next_span][0] == piece:
                heapq.heappush(covering, (spans[next_span][2], spans[next_span][1]))
                next_span += 1
            while covering and covering[0][1] < piece:
                heapq.heappop(covering)
            if covering:
                self.firsts[piece] = covering[0][0]

    def first(self, literal):
        """The index of the first forbidden literal that ``literal`` lies within the tolerance of, or None."""

        position = bisect.bisect_left(self.ends, literal)
        on_end = position < len(self.ends) and self.ends[position] == literal
        index = self.firsts[2 * position + 1 if on_end else 2 * position]
        if index is None or _close(literal, *self.forbidden[index]):
            return index
        # The literal lies in that window's margin for rounding, outside it: each window it may lie in decides.
        return next((other for other, pair in enumerate(self.forbidden) if _close(literal, *pair)), None)


def _window(value, tolerance):
    """A forbidden literal's window, ``value ± tolerance`` widened by a margin for the rounding of its ends, which the
    exact test of a match (:func:`_close`) settles."""

    pad = (abs(value) + tolerance) * 1e-12 + 1e-300
    return value - tolerance - pad, value + tolerance + pad


def _scan(checks, workdir, report, deadline):
    """Scan the submission's sources (``checks.sources``, relative to ``workdir``) for its forbidden literals
    (:func:`source_literals`): the solver's 0 for a source that states one (``solver_hardcoded``), and for one that is
    missing, no regular file, larger than :data:`MAX_SOURCE_BYTES` or no UTF-8 Python that this ``python3`` parses, or
    sources that take past ``deadline`` to scan (``solver_output_contract``); else None.  The report counts the
    literals the sources state and says where the first hits are, never what was forbidden, and which Python parsed
    them."""

    record = {
        "scan": LITERAL_SCAN,
        "python": "%d.%d.%d" % sys.version_info[:3],
        "sources": list(checks.sources),
        "literals": 0,
        "hits": [],
    }
    report.setdefault("replay_checks", {})["forbidden_literals"] = record
    late = Result.solver(OUTPUT_CONTRACT, "the sources take longer to scan than the verifier's time limit allows")
    windows = Windows(checks.forbidden)
    hits, count = record["hits"], 0
    for source in checks.sources:
        if _past(deadline):
            return late
        named = f"{source}, a source the task scans for forbidden literals,"
        try:
            data = _read_regular(os.path.join(workdir, source), MAX_SOURCE_BYTES)
        except (FileNotFoundError, NotADirectoryError):
            return Result.solver(OUTPUT_CONTRACT, f"{named} is missing")
        except (OSError, ValueError) as exc:
            return Result.solver(OUTPUT_CONTRACT, f"{named} cannot be read: {exc}")
        try:
            literals = source_literals(source_text(data))
        except _UNPARSABLE as exc:
            return Result.solver(OUTPUT_CONTRACT, f"{named} is no Python the scan reads: {type(exc).__name__}: {exc}")
        record["literals"] += len(literals)
        for position, (line, literal) in enumerate(sorted(literals, key=lambda item: item[0]), start=1):
            if position % _SCAN_BATCH == 0 and _past(deadline):
                return late
            index = windows.first(literal)
            if index is None:
                continue
            count += 1
            if len(hits) < REPORTED_HITS:
                hits.append({"source": source, "line": line, "literal": repr(literal), "forbidden": index})
    if not count:
        return None
    first = hits[0]
    more = f", and {count - 1} more" if count > 1 else ""
    return Result.solver(
        HARDCODED,
        f"{first['source']} states {first['literal']} on line {first['line']}, within the tolerance of forbidden "
        f"literal {first['forbidden']}{more}",
    )


def _shown(value):
    """``value`` as JSON (ASCII) of at most 80 characters, for a detail."""

    text = json.dumps(value, sort_keys=True, default=str)
    return text if len(text) <= 80 else text[:77] + "..."


def outputs_sha256(directory, outputs):
    """The SHA-256 of the files ``outputs`` (paths relative to ``directory``): of each one's path and the SHA-256 of its
    bytes, sorted, with :data:`OUTPUTS_RULE` first.  Replays that wrote the same bytes give the same one."""

    digest = hashlib.sha256(OUTPUTS_RULE.encode("ascii"))
    for output in sorted(outputs):  # no path holds a NUL (relative_path)
        content = _file_sha256(os.path.join(directory, output))
        digest.update(b"\0" + os.fsencode(output) + b"\0" + content.encode("ascii"))
    return digest.hexdigest()


def _in_place_fault(workdir, scratch):
    """Why the working directory ``workdir`` (a path without links) cannot be lent to the replay's user for a replay in
    place, a fault on the trusted side (the image's or the template's: no file of a submission makes it so), or None,
    found before anything changes there: something mounted below it (:func:`_mounted_below`), whose filesystem a
    replay there would change and which the launcher could not put back; a directory of it that the replay's user may
    write in, where it could put what the launcher puts the replay's outputs back through or the grader reads, or a
    socket or a device in it (:func:`_unfit_entry`); a directory above it in which that user could replace what it
    holds, and so move the working directory away while it replays there (:func:`_replaceable_above`); or the
    launcher's own directory ``scratch`` inside it."""

    mounted = _mounted_below(workdir)
    if mounted is not None:
        return (
            f"{mounted}, in the verifier's working directory, is a mount point, which a replay there would change and "
            "the launcher could not put back: give the template's image a working directory of one filesystem, with "
            "nothing mounted below it"
        )
    unfit = _unfit_entry(workdir)
    if unfit is not None and unfit[1] == "writable":
        return (
            f"{unfit[0]}, in the verifier's working directory, lets the replay's user write in it, where it could put "
            "what is graded: give the template's image a working directory that only root writes in"
        )
    if unfit is not None:
        return (
            f"{unfit[0]}, in the verifier's working directory, is a socket or a device, which a replay there could "
            "reach and the launcher could not put back: give the template's image a working directory without them"
        )
    above = _replaceable_above(workdir)
    if above is not None:
        directory, info = above
        return (
            f"the directory {directory}, above the verifier's working directory {workdir}, lets the replay's user "
            f"replace what it holds (owner {info.st_uid}, group {info.st_gid}, mode {stat.S_IMODE(info.st_mode):o}), "
            "where it could move the working directory away as it replays there: give the template's image a working "
            "directory below directories that only root writes in, or that have the sticky bit set"
        )
    if _within(os.path.realpath(scratch), workdir):
        return (
            f"the launcher's own directory {scratch} is inside the verifier's working directory {workdir}, which the "
            "replay's user writes in: give the template a working directory apart from /tmp"
        )
    return None


class _Directory:
    """What the launcher keeps of the working directory itself, the one entry of it that it puts back without making it
    anew (:func:`_restore`): its mode, its access and modification times (in nanoseconds) and its user extended
    attributes (:func:`_user_attributes`)."""

    def __init__(self, path):
        info = os.lstat(path)
        self.mode = stat.S_IMODE(info.st_mode)
        self.times = (info.st_atime_ns, info.st_mtime_ns)
        self.attributes = _user_attributes(path)


def _snapshot(workdir, pristine, ops, deadline, limit):
    """Copy what the working directory ``workdir`` holds into ``pristine``, the pristine copy that the launcher puts it
    back from after the replay (:func:`_restore`), as it is (:func:`_clone_tree`), and seal it, root's alone
    (:func:`_seal`); what the launcher keeps of the directory itself (:class:`_Directory`).  A :class:`SolverFault` for
    files of more than ``limit`` bytes, or for a copy that takes past ``deadline``; a :class:`LauncherFault` for an
    entry on a filesystem other than the directory's, or a socket or a device, which :func:`_in_place_fault` refuses
    before."""

    kept = _Directory(workdir)
    os.mkdir(pristine, 0o700)
    _clone_tree(workdir, pristine, deadline=deadline, limit=limit, device=os.lstat(workdir).st_dev)
    _seal(pristine, ops)
    return kept


def _restore(workdir, pristine, ops, kept, until):
    """Put the working directory ``workdir`` back as ``pristine``, its pristine copy, holds it: remove all it holds,
    never following a link (:func:`_prune`), copy the pristine copy back as it is (:func:`_clone_tree`), make all of it
    root's with the group and other write bits stripped (:meth:`Ops.harden`), as before the replay, and give the
    directory itself what ``kept`` keeps of it (:class:`_Directory`): its mode, its times and exactly its user extended
    attributes, which the replay's user, who may write in it, could change.  :class:`Leftover` for what cannot be
    removed, or not before ``until``; a :class:`SolverFault` for a copy that takes past it, or for all of this ending
    past it; an ``OSError`` that found its filesystem full (:data:`_NO_ROOM`) as it is, for the caller to attribute
    (:func:`_clear`); a :class:`LauncherFault` for any other."""

    _prune(workdir, (), _Clock(until))
    try:
        _clone_tree(pristine, workdir, deadline=until)
        ops.harden(workdir)
        os.chmod(workdir, kept.mode)
        _put_user_attributes(workdir, kept.attributes)
        os.utime(workdir, ns=kept.times, follow_symlinks=False)
    except SolverFault as exc:
        raise SolverFault(NONREPRODUCIBLE, "putting the working directory back outlived the replay's time") from exc
    except OSError as exc:
        if exc.errno in _NO_ROOM:
            raise
        raise LauncherFault(f"it cannot put the working directory back: {exc}") from exc
    if _past(until):
        raise SolverFault(NONREPRODUCIBLE, "putting the working directory back outlived the replay's time")


def _sync(path):
    """Have the filesystem of ``path`` write out what it holds (``syncfs``), so that one that frees the room of what was
    removed only after a while (XFS, btrfs) has freed it; every filesystem (``sync``) with a C library without
    ``syncfs``.  Nothing where it cannot."""

    try:
        descriptor = os.open(path, _DIRECTORY_FLAGS)
    except OSError:
        return
    try:
        syncfs = getattr(_libc(), "syncfs", None)
        if syncfs is None:
            os.sync()
        else:
            syncfs(descriptor)
    finally:
        os.close(descriptor)


def _user_attributes(path):
    """The user extended attributes (``user.*``) of ``path``, never read through a link, as ``(name, value)`` pairs;
    none on a filesystem without extended attributes."""

    try:
        names = os.listxattr(path, follow_symlinks=False)
    except OSError as exc:
        if exc.errno in (errno.ENOTSUP, errno.EOPNOTSUPP):
            return []
        raise
    names = sorted(name for name in names if name.startswith("user."))
    return [(name, os.getxattr(path, name, follow_symlinks=False)) for name in names]


def _put_user_attributes(path, attributes):
    """Give ``path`` exactly the user extended attributes ``attributes`` (:func:`_user_attributes`), never following a
    link: any user that may write in a directory may give it some, and change or remove those it has, so the replay's
    user may the working directory's, the one entry of it that the launcher puts back without making it anew."""

    names = dict(attributes)
    for name, _ in _user_attributes(path):
        if name not in names:
            os.removexattr(path, name, follow_symlinks=False)
    for name, value in attributes:
        os.setxattr(path, name, value, follow_symlinks=False)


def _submitted_program(argv):
    """Whether the program ``argv`` runs is a path into the working directory, the submission's, rather than one the
    image has (an absolute path, or a name ``PATH`` finds)."""

    return "/" in argv[0] and not argv[0].startswith("/")


def _output_problem(workdir, output, cap):
    """Why the replay's ``output`` (relative to the working directory ``workdir``, where it ran) is not a regular file
    the grader may take: missing, below something other than a directory, a link or a special file, linked elsewhere,
    too large, or a ``.json`` file that is not JSON.  Nothing of the replay runs by now, so nothing changes it while it
    is looked at."""

    path = workdir
    for part in output.split("/")[:-1]:
        path = os.path.join(path, part)
        try:
            info = os.lstat(path)
        except FileNotFoundError:
            return f"{output} is missing"
        if not stat.S_ISDIR(info.st_mode):
            return f"{output} is below {_kind(info)}"
    path = os.path.join(workdir, output)
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return f"{output} is missing"
    if not stat.S_ISREG(info.st_mode):
        return f"{output} is {_kind(info)}"
    if info.st_nlink != 1:
        return f"{output} has other links to it"
    return _file_problem(path, output, info, cap)


def _replay_env(workdir, home, keep=()):
    """The replay's environment: the launcher's variables that :data:`REPLAY_VARIABLES` and ``keep`` (the settings'
    ``replay_env``) name, where it has them, its ``PATH`` cut to absolute entries alone (:func:`_absolute_path`, so
    nothing in the working directory ``workdir``, where it runs, stands in for a program), and ``home``, a directory of
    its own that the launcher removes after it, as its home and temporary directory; the verifier's other variables,
    which may hold credentials, stay out."""

    env = {name: os.environ[name] for name in (*REPLAY_VARIABLES, *keep) if name in os.environ}
    env["PATH"] = _absolute_path(env.get("PATH", ""))
    env.update(HOME=home, TMPDIR=home, PWD=workdir)
    return env


def _absolute_path(value):
    """``value``, a ``PATH``, with only its absolute entries kept (no empty, ``.`` or relative entry, each of which a
    program would be looked for under the working directory, where the replay runs): :data:`DEFAULT_PATH` where none
    is."""

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
        verdict = load_json(_read_regular(path, MAX_VERDICT_BYTES).decode("utf-8"))
    except NestedTooDeeply:
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


def user_ipc(uid, proc="/proc"):
    """The System V IPC objects that ``proc``'s ``sysvipc`` lists with ``uid`` as their owner or their creator, each
    ``(kind, id)``, ``kind`` one of ``shm``, ``sem`` and ``msg``: what of the launcher's IPC namespace a process of that
    user may have left, which outlives the process.  A kind the kernel does not list is none."""

    found = []
    for kind, column in _SYSVIPC:
        try:
            with open(os.path.join(proc, "sysvipc", kind)) as stream:
                lines = stream.read().splitlines()
        except FileNotFoundError:
            continue
        if not lines:
            continue
        header = lines[0].split()
        ident, owner, creator = header.index(column), header.index("uid"), header.index("cuid")
        for line in lines[1:]:
            fields = line.split()
            if len(fields) == len(header) and uid in (int(fields[owner]), int(fields[creator])):
                found.append((kind, int(fields[ident])))
    return found


def remove_ipc(objects):
    """Remove each System V IPC object of ``objects`` (``(kind, id)`` pairs, :func:`user_ipc`): how many it removed.
    One already gone, or one it may not remove, it leaves for its caller to find (:func:`_clear_ipc`)."""

    libc = _libc()
    removed = 0
    for kind, ident in objects:
        if kind == "shm":
            done = libc.shmctl(ident, _IPC_RMID, None)
        elif kind == "sem":
            done = libc.semctl(ident, 0, _IPC_RMID)
        else:
            done = libc.msgctl(ident, _IPC_RMID, None)
        if done == 0:
            removed += 1
    return removed


def _clear_ipc(ops):
    """Remove the System V IPC objects of the replay's user (:meth:`Ops.ipc_of`, :func:`remove_ipc`): shared memory, a
    semaphore set or a message queue outlives the process that made it, and memory it holds, the grader would go
    without.  How many it removed; :class:`Leftover` for one that is still there after it."""

    found = ops.ipc_of(REPLAY_UID)
    if not found:
        return 0
    removed = remove_ipc(found)
    left = ops.ipc_of(REPLAY_UID)
    if left:
        raise Leftover(f"{len(left)} System V IPC object(s) of its user cannot be removed")
    return removed


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
    fits, else without the verdict's checks, then without the artifacts' names, the replay's outputs, the variants'
    replays and the scan's record, then with only what the outcome needs (the SHA-256 of the replayed outputs among
    it, which a regrade compares); ``truncated`` names what was left out."""

    text = _dumps(report)
    if len(text) <= limit:
        return text
    report = dict(report)
    left_out = []
    parts = (
        ("verdict", "checks"),
        ("contract", "artifacts"),
        ("replay", "outputs"),
        ("replay_checks", "variants"),
        ("replay_checks", "forbidden_literals"),
    )
    for part, key in parts:
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

    descriptor = os.open(path, _READ_FLAGS)
    try:
        info = os.fstat(descriptor)  # before a stream, which a directory could not be
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"{os.path.basename(path)} is not a regular file")
        if info.st_size > limit:
            raise ValueError(f"{os.path.basename(path)} holds more than {limit} bytes")
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            data = stream.read(info.st_size + 1)
    finally:
        os.close(descriptor)
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


def load_json(text):
    """The JSON ``text`` as Python's values: a :class:`NestedTooDeeply` for a text that nests more than
    :data:`MAX_JSON_DEPTH` levels deep (:func:`json_text_depth`), before it is parsed, and a ``ValueError`` for one
    that is not JSON.  Python's parser then never reaches the depth at which it gives up, which differs between
    Pythons."""

    if json_text_depth(text) > MAX_JSON_DEPTH:
        raise NestedTooDeeply(f"nested too deeply: more than {MAX_JSON_DEPTH} levels of arrays and objects")
    return json.loads(text)


def json_text_depth(text):
    """How deeply the JSON ``text`` nests arrays and objects, read from its brackets outside its strings, never by
    parsing it: the most that any start of the text opens and leaves open, so that a text that only opens counts each
    one, and a parser that stops at an error was never deeper.  It takes time linear in the text's length."""

    brackets = _NOT_BRACKET.sub("", _JSON_STRING.sub("", _JSON_ESCAPE.sub("", text)))
    return max(itertools.accumulate(map(_BRACKET_STEP.__getitem__, brackets), initial=0))


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


def _copy_tree(source, target, skip=None, deadline=None, limit=None):
    """Copy the regular files and directories below ``source`` into ``target``, each file 755 or 644 as it is
    executable or not and each directory 755, never following a link, and leaving out ``skip`` (the launcher's own
    directory, should ``source`` hold it) and everything else (a link, a FIFO, a socket, a device).  A
    :class:`SolverFault` for files of more than ``limit`` bytes, which only a submission brings past the task's own
    files (``max_copy_bytes``), and for a copy that takes past ``deadline``, the replay's, which it looks at before each
    directory and each file; an ``OSError`` for a directory it cannot list, which it never skips."""

    total = 0
    skipped = os.path.realpath(skip) if skip is not None else None
    for directory, directories, files in os.walk(source, onerror=_raise):
        relative = os.path.relpath(directory, source)
        into = target if relative == "." else os.path.join(target, relative)
        for name in list(directories):
            _in_time(deadline)
            path = os.path.join(directory, name)
            if os.path.islink(path) or (skipped is not None and os.path.realpath(path) == skipped):
                directories.remove(name)
                continue
            os.mkdir(os.path.join(into, name), 0o700)
            os.chmod(os.path.join(into, name), 0o755)
        for name in files:
            _in_time(deadline)
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


def _clone_tree(source, target, deadline=None, limit=None, device=None):
    """Copy what the directory ``source`` holds into the empty directory ``target`` as it is, never following a link:
    its directories, its regular files with their holes (:func:`_copy_file`), its links (their own text) and its FIFOs,
    each with its mode, but for the group and other write bits, which nothing in the working directory has once it is
    hardened, and its access and modification times, and the names of one file there as names of one file again; not
    their owners (each copy is the launcher's), the extended attributes and ACLs of what it holds, nor its other links
    beyond it.  A :class:`SolverFault` for files of more than ``limit`` bytes (each file once, whatever its names), and
    for a copy that takes past ``deadline``, which it looks at before each entry; a :class:`LauncherFault` for an entry
    on a filesystem other than ``device``, where it is given (a mount point), and for a socket or a device, which only
    an image puts there, as :func:`_in_place_fault` finds before; an ``OSError`` for a directory it cannot list."""

    total, files, directories = 0, {}, []
    pending = [""]
    while pending:
        relative = pending.pop()
        with os.scandir(os.path.join(source, relative) if relative else source) as entries:
            for entry in entries:
                _in_time(deadline)
                info = entry.stat(follow_symlinks=False)
                name = os.path.join(relative, entry.name) if relative else entry.name
                path = os.path.join(target, name)
                if device is not None and info.st_dev != device:
                    raise LauncherFault(
                        f"{entry.path}, in the verifier's working directory, is a mount point, which a replay there "
                        "would change and the launcher could not put back: give the template's image a working "
                        "directory of one filesystem, with nothing mounted below it"
                    )
                mode = stat.S_IMODE(info.st_mode) & ~(stat.S_IWGRP | stat.S_IWOTH)
                times = (info.st_atime_ns, info.st_mtime_ns)
                if stat.S_ISDIR(info.st_mode):
                    os.mkdir(path, 0o700)
                    directories.append((path, mode, times))  # its mode and times once it holds all it does
                    pending.append(name)
                elif stat.S_ISREG(info.st_mode):
                    first = files.get((info.st_dev, info.st_ino))
                    if first is not None:  # another name of a file it copied
                        os.link(first, path, follow_symlinks=False)
                        continue
                    total += info.st_size
                    if limit is not None and total > limit:
                        raise SolverFault(
                            OUTPUT_CONTRACT, f"the working directory holds more than {limit} bytes for a replay to copy"
                        )
                    _copy_file(entry.path, path, mode, times)
                    if info.st_nlink > 1:
                        files[(info.st_dev, info.st_ino)] = path
                elif stat.S_ISLNK(info.st_mode):
                    os.symlink(os.readlink(entry.path), path)
                    os.utime(path, ns=times, follow_symlinks=False)
                elif stat.S_ISFIFO(info.st_mode):
                    os.mkfifo(path, 0o600)
                    os.chmod(path, mode)
                    os.utime(path, ns=times, follow_symlinks=False)
                else:
                    raise LauncherFault(
                        f"{entry.path}, in the verifier's working directory, is a socket or a device, which the "
                        "launcher could not put back: give the template's image a working directory without them"
                    )
    for path, mode, times in reversed(directories):  # each after the directories it holds
        os.chmod(path, mode)
        os.utime(path, ns=times, follow_symlinks=False)


def _in_time(deadline):
    """A :class:`SolverFault` once ``deadline`` (a ``time.monotonic()``, or None) has passed, for a copy of the working
    directory that outlived the replay's time."""

    if _past(deadline):
        raise SolverFault(NONREPRODUCIBLE, "copying the working directory for the replay outlived its time")


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


def _copy_file(source, target, mode, times=None):
    """Copy the regular file ``source`` to a new file ``target`` of ``mode``, never following a link, its holes kept
    (:func:`_copy_data`), and with ``times`` (its access and modification times, in nanoseconds) where given."""

    reading = os.open(source, _READ_FLAGS)
    try:
        info = os.fstat(reading)
        if not stat.S_ISREG(info.st_mode):
            raise OSError(f"{source} is not a regular file")
        writing = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        try:
            _copy_data(reading, writing, info.st_size)
            os.fchmod(writing, mode)
            if times is not None:
                os.utime(writing, ns=times)
        finally:
            os.close(writing)
    finally:
        os.close(reading)


def _copy_data(reading, writing, size):
    """Copy the first ``size`` bytes of the file open as ``reading`` into the empty file open as ``writing``, each at
    its own offset, and leave its holes holes (``SEEK_DATA``, ``SEEK_HOLE``), so that a sparse file takes no more room
    as a copy than it does; a filesystem that cannot tell where its holes are has all of it copied.  The copy has its
    size before anything is written to it, so that no write extends it, which a filesystem may meet by reserving room
    past its end (XFS) that a later extension would keep."""

    os.ftruncate(writing, size)
    offset = 0
    while offset < size:
        try:
            start = os.lseek(reading, offset, os.SEEK_DATA)
            end = min(os.lseek(reading, start, os.SEEK_HOLE), size)
        except OSError as exc:
            if exc.errno == errno.ENXIO:  # holes alone from offset to its end
                break
            if exc.errno not in (errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP):
                raise
            start, end = offset, size
        while start < end:
            data = os.pread(reading, min(_COPY_CHUNK, end - start), start)
            if not data:  # it is shorter than it was
                break
            view = memoryview(data)
            while view:
                written = os.pwrite(writing, view, start)
                view, start = view[written:], start + written
        offset = max(end, offset + 1)


def _chown_tree(path, uid, gid):
    os.lchown(path, uid, gid)
    for directory, directories, files in os.walk(path):
        for name in directories + files:
            os.lchown(os.path.join(directory, name), uid, gid)


def _lend_tree(workdir, give=True):
    """Make the directories below ``workdir`` and its regular files the replay's user's (``give``: a test without root
    leaves the owners as they are), with read and write permission for their owner, and search permission for a
    directory, never following a link, and ``workdir`` itself root's, of that user's group, mode 770
    (:meth:`Ops.lend`).  A file of several links, of which some may be names outside the working directory, is first
    copied as root into a file of its own in its place (:func:`_break_link`), so that nothing the working directory
    reaches beyond what it holds becomes that user's; a link and a FIFO stay root's, which only an image puts there (a
    submission's files and directories reach the verifier as files and directories of their own), and the replay may
    still remove them, which the launcher undoes (:func:`_restore`)."""

    for directory, directories, files in os.walk(workdir, onerror=_raise):
        for name in directories + files:
            path = os.path.join(directory, name)
            info = os.lstat(path)
            if stat.S_ISREG(info.st_mode) and info.st_nlink > 1:
                _break_link(path, info)
                info = os.lstat(path)
            if stat.S_ISDIR(info.st_mode):
                mode = stat.S_IMODE(info.st_mode) | stat.S_IRWXU
            elif stat.S_ISREG(info.st_mode):
                mode = (stat.S_IMODE(info.st_mode) & ~(stat.S_ISUID | stat.S_ISGID)) | stat.S_IRUSR | stat.S_IWUSR
            else:
                continue
            if give:
                os.lchown(path, REPLAY_UID, REPLAY_UID)
            os.chmod(path, mode)
    if give:
        os.lchown(workdir, 0, REPLAY_UID)
    os.chmod(workdir, 0o770)


def _break_link(path, info):
    """Put a file of its own in the place of ``path``, a name of a regular file of several links (``info``, its
    ``os.lstat`` result): a copy with its contents, its mode and its times, beside it, renamed over it."""

    copy = os.path.join(os.path.dirname(path), f".platoon-lend-{os.urandom(8).hex()}")
    _copy_file(path, copy, stat.S_IMODE(info.st_mode), (info.st_atime_ns, info.st_mtime_ns))
    os.replace(copy, path)


def _raise(error):
    """What ``os.walk`` calls with an error it meets: raise it, so that a directory it cannot list is never skipped."""

    raise error


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


def _hold(workdir, held, outputs, ops, until):
    """Remove all but ``outputs`` and the directories above them from the working directory ``workdir``, where the
    replay ran (:func:`_prune`), which gives back the room the replay took there before anything that may need room
    runs; then copy each output into ``held``, root's alone (files 600, directories 700, sealed), where it waits while
    the working directory is put back (:func:`_restore`).  :class:`Leftover` as :func:`remove_tree` raises it; an
    ``OSError`` that found its filesystem full (:data:`_NO_ROOM`: a filesystem that copies on write may need room even
    for a change of owner) as it is, for the caller to attribute as it attributes the outputs' copies back
    (:func:`_clear`); a :class:`LauncherFault` for any other."""

    _prune(workdir, outputs, _Clock(until))  # nothing of the replay's runs by now (the sweep), so nothing changes it
    try:
        os.mkdir(held, 0o700)
        for output in outputs:  # a regular file of one link below directories (_output_problem); nothing of its runs
            target = os.path.join(held, output)
            _make_directories(os.path.dirname(target), 0o700)
            _copy_file(os.path.join(workdir, output), target, 0o600)
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
    through directories that ``writable`` says that user may write in (by their ``os.lstat`` result and their path),
    but for ``skip`` (a path, or several: the launcher's own directory, and the working directory, which the launcher
    puts back itself), never following a link (:func:`remove_tree`).  How many entries it removed; :class:`Leftover`
    for one it cannot remove, or not before ``until`` (a ``time.monotonic()``).  A directory the sandbox lacks is
    none."""

    clock = _Clock(until)
    skipped = {os.path.realpath(path) for path in ((skip,) if isinstance(skip, str) else skip)}
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
                    if entry.path in skipped:
                        continue
                    try:
                        info = entry.stat(follow_symlinks=False)
                    except FileNotFoundError:
                        continue
                    if owned(info):
                        _remove(entry.path, clock)
                        removed += 1
                    elif stat.S_ISDIR(info.st_mode) and writable(info, entry.path):
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
