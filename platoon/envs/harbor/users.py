"""The users a Harbor task names, and the sandboxes that run them (``agent_user: honor``, ``verifier_user: honor``;
:mod:`platoon.envs.harbor.task`).

A user that a task names (``[agent].user``, ``[verifier].user``) is one of Docker's ``--user`` forms, which
:func:`parse_user` reads against the image's ``/etc/passwd`` and ``/etc/group`` as Docker does (:class:`User`), and
:func:`phase_users` says who runs each phase by Harbor's rule; :mod:`platoon.envs.harbor.views` runs a session's
commands as them.

A sandbox that runs them must give root and other users, as a verifier that runs as root needs too: its tests run the
agent's code as another user, nobody (65534) as a rule, so that it cannot read the grader's held-out files, and they
fail closed, scoring 0, when they cannot switch users.  So root alone is not enough: the sandbox's user namespace must
map the other users too, as Docker does and Singularity's ``--fakeroot`` does with the caller's subordinate ids;
enroot maps only the caller's uid, and a rootless sandbox only runs as the caller.  The provider's options say most of
it before any sandbox starts (:func:`platoon.sandbox.root_refusal`); every started sandbox that honors users is checked
too (:func:`root_problem`), since an image's own user decides on ``docker``, and a provider of your own may not say.
The other users must also be able to run the image's programs, which an image whose files keep them out denies them
(:func:`nobody_problem`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from platoon.sandbox.base import Sandbox

if TYPE_CHECKING:
    from platoon.envs.harbor.task import HarborTask

#: The user that graders which need root run the agent's code as; a root verifier's sandbox must map every uid and
#: gid up to it.
NOBODY = 65534
#: The umask Docker gives a container's commands, under which what root writes is readable by every user.  A root
#: verifier's commands run with it, since the users its tests switch to must read what it writes: a rootless sandbox's
#: commands get the host's, often 027 on shared machines.
DOCKER_UMASK = "022"
# Prints "uid <effective uid>", then one "uid_map <first id inside> <count>" line per range of the shell's uid map,
# and the same for its gid map, with shell builtins only.
_CHECK_SCRIPT = r"""
uid=
while read -r key real effective rest; do
  if [ "$key" = Uid: ]; then uid=$effective; break; fi
done < /proc/self/status
echo "uid ${uid:-unknown}"
for kind in uid gid; do
  [ -r "/proc/self/${kind}_map" ] || continue
  while read -r inside outside count; do echo "${kind}_map $inside $count"; done < "/proc/self/${kind}_map"
done
"""
# Runs the image's /bin/sh as nobody, as the tests switch users, with the first tool the image has for it: setpriv
# (util-linux), else su (BusyBox, shadow).  The shell prints "ran"; what the tool says when it cannot goes to stderr.
_NOBODY_SCRIPT = f"""
if command -v setpriv >/dev/null 2>&1; then
  exec setpriv --reuid={NOBODY} --regid={NOBODY} --clear-groups /bin/sh -c 'echo ran'
elif command -v su >/dev/null 2>&1; then
  exec su -s /bin/sh nobody -c 'echo ran'
fi
echo "platoon: the image has neither setpriv nor su" >&2
"""
# What a shell that nobody may not run says, through the switch: the kernel's refusal to run it, or the dynamic
# loader's failure to open its libraries (which it reports as missing when it may not enter their directory).
_DENIED = ("Permission denied", "error while loading shared libraries")
_CHECK_TIMEOUT_S = 60.0
_LINE = re.compile(r"(uid_map|gid_map) ([0-9]+) ([0-9]+)")
# Prints the image's /etc/passwd, a line of its own, then its /etc/group.
_GROUP_FILE = "platoon-group-file"
_ACCOUNTS_SCRIPT = f"cat /etc/passwd 2>/dev/null; echo; echo {_GROUP_FILE}; cat /etc/group 2>/dev/null"


@dataclass(frozen=True)
class User:
    """One of an image's users, as a task names it (``spec``, one of Docker's ``--user`` forms) and the image's
    ``/etc/passwd`` and ``/etc/group`` give it: a uid and a gid, and when ``/etc/passwd`` lists the uid, its name, home,
    the group it gives the name (``passwd_gid``) and the groups that ``/etc/group`` adds it to."""

    spec: str
    uid: int
    gid: int
    name: str | None = None
    home: str = "/"  # Docker's for a uid that /etc/passwd does not list
    groups: tuple[int, ...] = ()
    passwd_gid: int | None = None

    @property
    def login(self) -> str | None:
        """The name ``su`` may switch to, which gives it its supplementary groups: this user's, where the task names it
        no group (Docker gives those groups then alone) and its gid is the one ``/etc/passwd`` gives it."""

        named = self.name is not None and ":" not in self.spec
        return self.name if named and self.gid == self.passwd_gid else None


#: Root, as a sandbox that gives root runs platoon's own commands.
ROOT = User(spec="0", uid=0, gid=0, name="root", home="/root", passwd_gid=0)


def phase_users(harbor_task: HarborTask, *, image_user: str | None, verifier: bool = False) -> dict[str, str | None]:
    """Who runs what in a sandbox of ``harbor_task`` that honors users (its separate verifier's with ``verifier``), by
    Harbor's rule: ``agents`` and ``tests``, each the user the task names for it where its phase is honored there,
    else the image's ``USER`` (``image_user``), else None, the sandbox's own user; and ``default``, the image's own
    commands (``main``'s init and healthchecks, collect hooks that name no user), which run in the agents' sandbox
    alone, as its ``USER``.  A collect hook that names a user runs as it (:func:`hook_users`)."""

    honored = harbor_task.honored_users
    tests_here = harbor_task.verifier_mode == ("separate" if verifier else "shared")
    return {
        "agents": (honored["agent"] or image_user) if "agent" in honored and not verifier else None,
        "tests": (honored["verifier"] or image_user) if "verifier" in honored and tests_here else None,
        "default": image_user if not verifier else None,
    }


def hook_users(harbor_task: HarborTask, *, verifier: bool = False) -> list[str]:
    """The users that the task's collect hooks name, as which they run in its agents' sandbox (none in its separate
    verifier's, with ``verifier``)."""

    return [] if verifier else [hook.user for hook in harbor_task.collect if hook.user is not None]


def runs_image_user(harbor_task: HarborTask, *, verifier: bool = False) -> bool:
    """Whether a sandbox of the task that honors users runs something as its image's ``USER``
    (:func:`phase_users`): a phase honored there whose user the task does not name, or the image's own commands."""

    unnamed = "\0"  # no user's name: where it shows, the image's USER runs
    return unnamed in phase_users(harbor_task, image_user=unnamed, verifier=verifier).values()


async def read_accounts(sandbox: Sandbox) -> tuple[str, str]:
    """The image's ``/etc/passwd`` and ``/etc/group``, as ``sandbox``'s commands read them (empty where one is
    missing), for :func:`parse_user`; ``ValueError`` when they cannot be read."""

    result = await sandbox.exec(
        ["/bin/sh", "-c", _ACCOUNTS_SCRIPT, "platoon-accounts"], cwd="/", timeout_s=_CHECK_TIMEOUT_S
    )
    if result.timed_out or result.exit_code != 0:
        raise ValueError(f"cannot read the image's /etc/passwd: {(result.stderr or result.stdout).strip()}")
    passwd, _, groups = result.stdout.partition(f"\n{_GROUP_FILE}\n")
    return passwd, groups


def parse_user(spec: str, passwd: str, groups: str) -> User:
    """The user ``spec`` names, as Docker resolves ``--user`` against an image's ``passwd`` and ``groups`` (the texts
    of ``/etc/passwd`` and ``/etc/group``): a name, or a uid, which ``/etc/passwd`` need not list; then optionally a
    group's name, or a gid.  Without a group, the user's group is the one ``/etc/passwd`` gives it, else 0.
    ``ValueError`` for a name, or a group's name, that the image lacks."""

    who, _, group = spec.partition(":")
    accounts = [fields for line in passwd.splitlines() if len(fields := line.split(":")) >= 7 and fields[2].isdigit()]
    entry = next((fields for fields in accounts if fields[2 if who.isdigit() else 0] == who), None)
    if entry is None and not who.isdigit():
        raise ValueError(f"the image's /etc/passwd has no user {who!r}" if who else f"{spec!r} names no user")
    rows = [fields for line in groups.splitlines() if len(fields := line.split(":")) >= 4 and fields[2].isdigit()]
    name: str | None = None
    home = "/"
    passwd_gid: int | None = None
    if entry is not None:
        name, home = entry[0], entry[5] or "/"
        passwd_gid = int(entry[3]) if entry[3].isdigit() else None
    if not group:
        gid = passwd_gid if passwd_gid is not None else 0
    elif group.isdigit():
        gid = int(group)
    elif (row := next((row for row in rows if row[0] == group), None)) is not None:
        gid = int(row[2])
    else:
        raise ValueError(f"the image's /etc/group has no group {group!r}")
    # As Docker: the groups /etc/group adds the user to, where the task names it no group.
    member_of = sorted({int(row[2]) for row in rows if name is not None and not group and name in row[3].split(",")})
    return User(
        spec=spec,
        uid=int(who) if entry is None else int(entry[2]),
        gid=gid,
        name=name,
        home=home,
        groups=tuple(member_of),
        passwd_gid=passwd_gid,
    )


def copy_owner(spec: str, passwd: str, groups: str) -> str:
    """The owner, ``uid:gid``, that a Dockerfile's ``COPY --chown=<spec>`` gives what it copies, as BuildKit reads
    ``spec`` against the image's ``/etc/passwd`` and ``/etc/group``: unlike ``--user``, a uid without a group gets
    the gid of the same number.  ``ValueError`` for a name, or a group's name, that the image lacks."""

    user = parse_user(spec, passwd, groups)
    who, _, group = spec.partition(":")
    return f"{user.uid}:{user.uid if who.isdigit() and not group else user.gid}"


async def root_problem(sandbox: Sandbox) -> str | None:
    """What keeps ``sandbox`` from running a root verifier's tests, as they would run in Docker: its commands do not
    run as root, or its user namespace does not map every uid and gid from 0 to :data:`NOBODY`, so the tests could
    not run anything as another user; None when nothing does.  Read from inside the sandbox, from ``/``."""

    result = await sandbox.exec(
        ["/bin/sh", "-c", _CHECK_SCRIPT, "platoon-root-check"], cwd="/", timeout_s=_CHECK_TIMEOUT_S
    )
    lines = result.stdout.splitlines()
    if result.timed_out or result.exit_code != 0 or not lines or not lines[0].startswith("uid "):
        detail = result.stderr.strip() or ("timed out" if result.timed_out else f"exit {result.exit_code}")
        return f"cannot tell which user its commands run as ({detail})"
    uid = lines[0].removeprefix("uid ")
    if uid != "0":
        return f"runs its commands as uid {uid}, not as root"
    ranges: dict[str, list[tuple[int, int]]] = {"uid_map": [], "gid_map": []}
    for line in lines[1:]:
        if (match := _LINE.fullmatch(line.strip())) is not None:
            ranges[match.group(1)].append((int(match.group(2)), int(match.group(3))))
    for kind, mapped in ranges.items():
        if not _covers(mapped, NOBODY):
            ids = kind.removesuffix("_map")
            shown = ", ".join(f"{start}-{start + count - 1}" if count > 1 else str(start) for start, count in mapped)
            return (
                f"runs its commands as root, but maps only the {ids}s {shown or 'none'} into its user namespace, so "
                f"they could not run anything as another user, such as nobody ({NOBODY})"
            )
    return None


async def nobody_problem(sandbox: Sandbox) -> str | None:
    """Whether nobody (:data:`NOBODY`), as which a root verifier's tests run the agent's code, can run the image's
    ``/bin/sh`` in ``sandbox``, whose commands run as root: a problem when the switch to nobody ran and the shell
    could not start, or not load its libraries, for want of permission, as in an image whose files keep other users
    out (such as one the ``oci`` builder of platoon before 0.3.0 unpacked under a umask of 027); None when it ran, or
    when nothing tells (no ``setpriv`` or ``su`` in the image, or a switch that failed for another reason, which the
    tests then meet)."""

    result = await sandbox.exec(
        ["/bin/sh", "-c", _NOBODY_SCRIPT, "platoon-nobody-check"], cwd="/", timeout_s=_CHECK_TIMEOUT_S
    )
    if result.stdout.split() == ["ran"]:
        return None
    said = [
        line.strip()
        for line in (result.stderr + "\n" + result.stdout).splitlines()
        if any(sign in line for sign in _DENIED)
    ]
    if result.timed_out or not said:
        return None
    return (
        f"cannot run the image's /bin/sh as nobody ({NOBODY}): {said[0]}; the image's files keep the other users "
        "out, so the tests could not run anything as another user"
    )


def _covers(ranges: list[tuple[int, int]], last: int) -> bool:
    """Whether the ``(first id, count)`` ranges hold every id from 0 to ``last``."""

    following = 0  # the first id not known to be mapped yet
    for start, count in sorted(ranges):
        if start > following:
            return False
        following = max(following, start + count)
        if following > last:
            return True
    return False
