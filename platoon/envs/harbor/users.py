"""Whether a separate verifier's sandbox runs its tests as root, as a task whose ``[verifier].user`` is root asks
(``verifier_user: honor``, :mod:`platoon.envs.harbor.task`).

Such tests run the agent's code as another user, nobody (65534) as a rule, so that it cannot read the grader's
held-out files, and they fail closed, scoring 0, when they cannot switch users.  So root alone is not enough: the
sandbox's user namespace must map the other users too, as Docker does and Singularity's ``--fakeroot`` does with the
caller's subordinate ids; enroot maps only the caller's uid, and a rootless sandbox only runs as the caller.  The
provider's options say most of it before any sandbox starts (:func:`platoon.sandbox.root_refusal`); every started
sandbox of such a verifier is checked too (:func:`root_problem`), since an image's own user decides on ``docker``, and
a provider of your own may not say.  The users the tests switch to must also be able to run the image's programs,
which an image whose files keep other users out denies them (:func:`nobody_problem`).
"""

from __future__ import annotations

import re

from platoon.sandbox.base import Sandbox

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
