"""Kill what commands left running in a container sandbox, from inside it, and spare the sandbox's own processes.

A container sandbox (the docker, singularity and enroot providers) has a PID namespace of its own, and its
``/proc`` lists exactly the processes in it: the ones the runtime starts to keep the container alive (an init, a
keep-alive loop and its ``sleep``), which run before any command, and whatever commands started since, detached
ones included, since every orphan stays in the namespace.  The provider lists the first kind once, right after the
sandbox starts (:func:`own_processes`), each as its pid and start time, which no later process can share.
:func:`kill_other_processes` then runs :data:`KILL_SCRIPT` in the sandbox: it freezes every process but the
namespace's init and itself at once (``kill -s STOP -1``, which the kernel applies to the whole namespace, so
nothing forks past it), SIGKILLs every process that is not one of the sandbox's own, looks again until none is
left, and lets the sandbox's own processes continue.

The scripts run in the image's own ``/bin/sh`` (dash, busybox ash or bash) and use only shell builtins while they
look at ``/proc``.  They refuse to act on a ``/proc`` that does not list their own shell, as one mounted over by a
command would not.  A command run as another user that the sandbox's commands may not signal survives, and the kill
fails, which no built-in provider allows: every command runs as the same user, or as the root of the sandbox's user
namespace (``singularity`` with ``--fakeroot``), which may signal every other user's process there.

A runtime may give a sandbox no PID namespace of its own (Singularity does not where the site's configuration says
``allow pid ns = no``), and there ``kill -1`` would reach every process of the user on the host.  So
:func:`own_processes` refuses a sandbox whose ``/proc`` lists the platoon process that started it, and the kill is
never run in one.

**Kept processes.** A sandbox may also keep processes that a session started for the task before any agent ran, such as
the init a main-only compose file gives ``main`` (:mod:`platoon.envs.harbor.compose`): ``Sandbox.keep_processes`` takes
them ("<pid>:<start time>"), and the kill spares each of them with its descendants.  :func:`start_background` starts
such a command, :func:`list_processes` lists a sandbox's processes before and after (what appeared in between is the
task's own, daemons it detached included), and :func:`alive_processes` tells which still run.
"""

from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from platoon.sandbox.base import Sandbox

logger = logging.getLogger(__name__)

# How long the kill script keeps killing what is left before it gives up (passes of _PASS_PAUSE_S seconds).
_KILL_PASSES = 100
_PASS_PAUSE_S = 0.05
_COMMAND_TIMEOUT_S = 120.0
KILL_TIMEOUT_S = _COMMAND_TIMEOUT_S
_ENTRY = re.compile(r"[0-9]+:[0-9]+")

# Reads /proc/<pid>/stat into "$1" (the state), "$2" (the parent pid) and "${20}" (the start time), after the
# command name, which may hold spaces and parentheses; fails when the process is gone.
_READ_STAT = r"""
read_stat() {
  { IFS= read -r stat < "/proc/$1/stat"; } 2>/dev/null || return 1
  case $stat in "$1 ("*) ;; *) return 1 ;; esac
  stat=${stat##*") "}
}
"""

# Exits 3 unless /proc lists this shell as the process it is.
_CHECK_PROC = r"""
self=$$
if ! read_stat "$self"; then
  echo "platoon: /proc in this sandbox does not list its processes" >&2
  exit 3
fi
"""

#: Prints "<pid>:<start time>" for every live process in the sandbox but its own shell.
LIST_SCRIPT = (
    _READ_STAT
    + _CHECK_PROC
    + r"""
for dir in /proc/[0-9]*; do
  pid=${dir#/proc/}
  [ "$pid" = "$self" ] && continue
  read_stat "$pid" || continue
  set -f; set -- $stat; set +f
  case $1 in Z|X|x) continue ;; esac
  echo "$pid:${20}"
done
"""
)

#: ``sh -c KILL_SCRIPT platoon-kill PASSES PAUSE TREES KEEP...``: SIGKILL every process but the ``KEEP`` ones
#: ("<pid>:<start time>"), the ``TREES`` ones (the same, space-separated in one argument) with their descendants, the
#: namespace's init, this shell and its ancestors in the sandbox (a runtime that starts commands below a process of
#: its own); exits 4 if some are still alive after ``PASSES`` looks.
KILL_SCRIPT = (
    _READ_STAT
    + _CHECK_PROC
    + r"""
passes=$1; pause=$2; trees=" $3 "; shift 3
keep=" $* "
spare=" 1 "
in_tree() {
  case $trees in "  ") return 1 ;; esac
  node=$1
  while read_stat "$node"; do
    set -f; set -- $stat; set +f
    case $trees in *" $node:${20} "*) return 0 ;; esac
    [ "$2" -gt 1 ] 2>/dev/null || return 1
    node=$2
  done
  return 1
}
pid=$self
while read_stat "$pid"; do
  set -f; set -- $stat; set +f
  [ "$2" -gt 1 ] 2>/dev/null || break
  pid=$2; spare="$spare$pid "
done
signal_all() { kill -s "$1" -- -1 2>/dev/null || kill -s "$1" -1 2>/dev/null; }
signal_all STOP
pass=0
while :; do
  left=
  for dir in /proc/[0-9]*; do
    pid=${dir#/proc/}
    [ "$pid" = "$self" ] && continue
    case $spare in *" $pid "*) continue ;; esac
    read_stat "$pid" || continue
    set -f; set -- $stat; set +f
    case $1 in Z|X|x) continue ;; esac
    [ "$2" = "$self" ] && continue
    case $keep in *" $pid:${20} "*) continue ;; esac
    in_tree "$pid" && continue
    left="$left $pid"
  done
  [ -z "$left" ] && break
  pass=$((pass + 1))
  if [ "$pass" -gt "$passes" ]; then
    signal_all CONT
    echo "platoon: processes still alive after SIGKILL:$left" >&2
    exit 4
  fi
  kill -s KILL $left 2>/dev/null
  sleep "$pause" 2>/dev/null || :
done
signal_all CONT
exit 0
"""
)


#: ``sh -c KILL_MARKED_SCRIPT platoon-kill-marked PASSES PAUSE MARKER``: SIGKILL every process whose environment holds
#: the line ``MARKER`` (``NAME=value``), looking again until none is left (a process that forked in between carries it
#: too); exits 4 if some are still alive after ``PASSES`` looks.  GNU grep scans ``/proc`` in one process; other greps
#: fall back to one check per process.
KILL_MARKED_SCRIPT = r"""
passes=$1; pause=$2; marker=$3
marked() {
  if printf 'x\0' | grep -qz x 2>/dev/null; then
    grep -lsxzF -- "$marker" /proc/[0-9]*/environ 2>/dev/null | cut -d/ -f3
  else
    for dir in /proc/[0-9]*; do
      tr '\0' '\n' < "$dir/environ" 2>/dev/null | grep -qxF -- "$marker" && echo "${dir#/proc/}"
    done
  fi
}
pass=0
while :; do
  left=$(marked | grep -vx "$$")
  [ -z "$left" ] && exit 0
  pass=$((pass + 1))
  if [ "$pass" -gt "$passes" ]; then
    echo "platoon: processes still alive after SIGKILL:" $left >&2
    exit 4
  fi
  kill -s KILL $left 2>/dev/null
  sleep "$pause" 2>/dev/null || :
done
"""


def kill_marked_command(marker: str) -> list[str]:
    """The command that SIGKILLs, run in a sandbox (in "/", within :data:`KILL_TIMEOUT_S`), every process there whose
    environment holds ``marker`` (``NAME=value``): what one command started, when the runtime would leave it
    running.  It exits 0 once none is left."""

    return ["/bin/sh", "-c", KILL_MARKED_SCRIPT, "platoon-kill-marked", str(_KILL_PASSES), f"{_PASS_PAUSE_S:g}", marker]


#: ``sh -c START_SCRIPT platoon-start LOG ARGV...``: start ``ARGV`` in the background, its output to ``LOG``, and print
#: "<pid>:<start time>" of it.
START_SCRIPT = (
    _READ_STAT
    + r"""
log=$1; shift
mkdir -p "${log%/*}" 2>/dev/null
"$@" </dev/null >"$log" 2>&1 &
pid=$!
if ! read_stat "$pid"; then
  echo "platoon: the command is gone already" >&2
  exit 1
fi
set -f; set -- $stat; set +f
echo "$pid:${20}"
"""
)

#: ``sh -c STATUS_START_SCRIPT platoon-start LOG STATUS ARGV...``: as :data:`START_SCRIPT`, but a subshell runs ``ARGV``
#: and writes its exit status to ``STATUS`` once it ends; the process printed is that subshell, which ends with it.
STATUS_START_SCRIPT = (
    _READ_STAT
    + r"""
log=$1; status=$2; shift 2
mkdir -p "${log%/*}" "${status%/*}" 2>/dev/null
rm -f "$status" "$status.tmp"
( "$@" </dev/null >"$log" 2>&1; echo "$?" >"$status.tmp"; mv -f "$status.tmp" "$status" ) </dev/null >/dev/null 2>&1 &
pid=$!
if ! read_stat "$pid"; then
  echo "platoon: the command is gone already" >&2
  exit 1
fi
set -f; set -- $stat; set +f
echo "$pid:${20}"
"""
)

#: ``sh -c ALIVE_SCRIPT platoon-alive ENTRY...``: print the ``ENTRY`` processes ("<pid>:<start time>") still alive.
ALIVE_SCRIPT = (
    _READ_STAT
    + r"""
for entry in "$@"; do
  read_stat "${entry%%:*}" || continue
  set -f; set -- $stat; set +f
  case $1 in Z|X|x) continue ;; esac
  [ "${entry#*:}" = "${20}" ] && echo "$entry"
done
"""
)


async def start_background(
    sandbox: Sandbox,
    argv: list[str],
    *,
    log: str,
    cwd: str,
    env: dict[str, str] | None = None,
    status: str | None = None,
) -> str:
    """Start ``argv`` in ``sandbox`` in the background (in ``cwd``, its output in the sandbox file ``log``) and return
    "<pid>:<start time>" of it; RuntimeError when it could not start.  A command that exits soon after it started may
    still be returned: :func:`alive_processes` tells.  With ``status`` (a sandbox file), a subshell runs it and writes
    its exit status there once it ends, and the process returned is that subshell (:data:`STATUS_START_SCRIPT`)."""

    script = ["/bin/sh", "-c", START_SCRIPT, "platoon-start", log]
    if status is not None:
        script = ["/bin/sh", "-c", STATUS_START_SCRIPT, "platoon-start", log, status]
    result = await sandbox.exec([*script, *argv], cwd=cwd, env=env, timeout_s=60.0)
    entry = result.stdout.strip().splitlines()[-1:] if result.stdout.strip() else []
    if result.exit_code != 0 or result.timed_out or not entry or not _ENTRY.fullmatch(entry[0]):
        detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.exit_code}"
        raise RuntimeError(f"Could not start {argv[0]!r} in sandbox {sandbox.id}: {detail}")
    return entry[0]


async def list_processes(sandbox: Sandbox) -> frozenset[str]:
    """Every process ``/proc`` lists in ``sandbox`` ("<pid>:<start time>"), its listing shell's aside; RuntimeError
    when it cannot be listed.  Unlike :func:`own_processes`, it lists a sandbox that shares this host's processes
    (the ``process`` provider's) too."""

    result = await sandbox.exec(["/bin/sh", "-c", LIST_SCRIPT, "platoon-list"], cwd="/", timeout_s=60.0)
    if result.exit_code != 0 or result.timed_out:
        detail = result.stderr.strip() or f"exit {result.exit_code}"
        raise RuntimeError(f"Could not list the processes of sandbox {sandbox.id}: {detail}")
    return frozenset(line for line in result.stdout.split() if _ENTRY.fullmatch(line))


async def alive_processes(sandbox: Sandbox, entries: frozenset[str] | set[str]) -> frozenset[str]:
    """Which of ``entries`` ("<pid>:<start time>") still run in ``sandbox``; RuntimeError when it cannot tell."""

    if not entries:
        return frozenset()
    result = await sandbox.exec(
        ["/bin/sh", "-c", ALIVE_SCRIPT, "platoon-alive", *sorted(entries)], cwd="/", timeout_s=60.0
    )
    if result.exit_code != 0 or result.timed_out:
        detail = result.stderr.strip() or f"exit {result.exit_code}"
        raise RuntimeError(f"Could not look for processes in sandbox {sandbox.id}: {detail}")
    return frozenset(line for line in result.stdout.split() if line in entries)


async def own_processes(sandbox: Sandbox) -> frozenset[str] | None:
    """The sandbox's own processes ("<pid>:<start time>"), listed from inside it before any command ran.

    None when they cannot be listed (no ``/bin/sh``, or a ``/proc`` that does not show the sandbox's processes), or
    when the sandbox has no PID namespace of its own, which shows as this process in its ``/proc``:
    :func:`kill_other_processes` then refuses to run, since it could not tell them from the commands' processes, or
    would kill outside the sandbox.
    """

    try:
        result = await sandbox.exec(["/bin/sh", "-c", LIST_SCRIPT, "platoon-list"], cwd="/", timeout_s=60.0)
    except Exception:
        logger.warning("Could not list the processes of sandbox %s", sandbox.id, exc_info=True)
        return None
    if result.exit_code != 0 or result.timed_out:
        logger.warning(
            "Could not list the processes of sandbox %s (exit %s): %s",
            sandbox.id,
            result.exit_code,
            result.stderr.strip(),
        )
        return None
    processes = frozenset(line for line in result.stdout.split() if _ENTRY.fullmatch(line))
    if (this := _this_process()) is not None and this in processes:
        logger.error(
            "Sandbox %s shares the PID namespace of this process (%s): its runtime gave it none of its own, so it "
            "will not kill what its commands leave running",
            sandbox.id,
            this,
        )
        return None
    return processes


def _this_process() -> str | None:
    """This process as the sandbox scripts list one ("<pid>:<start time>"); None without a Linux ``/proc``."""

    try:
        with open("/proc/self/stat", "rb") as stat:
            fields = stat.read().rsplit(b") ", 1)[1].split()
        return f"{os.getpid()}:{int(fields[19])}"
    except (OSError, IndexError, ValueError):
        return None


async def kill_other_processes(
    sandbox: Sandbox, keep: frozenset[str] | None, trees: frozenset[str] | set[str] = frozenset()
) -> None:
    """SIGKILL, from inside ``sandbox``, every process but its own (``keep``, from :func:`own_processes`) and the kept
    ``trees`` (their roots, "<pid>:<start time>"), each with its descendants.

    Raises RuntimeError when ``keep`` is unknown, when ``/proc`` does not list the sandbox's processes, or when
    some process is still alive after the script's last look.
    """

    if keep is None:
        raise RuntimeError(
            f"Sandbox {sandbox.id} could not list its own processes when it started, or has no PID namespace of its "
            "own, so it cannot tell them from the processes its commands started"
        )
    result = await sandbox.exec(
        [
            "/bin/sh",
            "-c",
            KILL_SCRIPT,
            "platoon-kill",
            str(_KILL_PASSES),
            f"{_PASS_PAUSE_S:g}",
            " ".join(sorted(trees)),
            *sorted(keep),
        ],
        cwd="/",
        timeout_s=_COMMAND_TIMEOUT_S,
    )
    if result.timed_out or result.exit_code != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.exit_code}"
        raise RuntimeError(f"Could not kill the processes left in sandbox {sandbox.id}: {detail}")
