"""A verifier that runs as root, end to end on Singularity with ``--fakeroot``: its tests run the agent's code as
nobody, which cannot read what the tests keep for themselves, and nothing of the verifier's sandbox stays on the
node.  With an hpc-sandbox that cannot delete what the other users of a ``--fakeroot`` sandbox write, the tasks are
refused instead, when they are listed.

Skipped unless ``PLATOON_TEST_SINGULARITY_IMAGE`` names an image (a local ``.sif``, or a ``docker://`` reference the
node can pull) with ``sh``, ``find``, ``chown``, ``stat`` and ``setpriv`` or ``su``, and Singularity or Apptainer and
hpc-sandbox are installed.  Run it on a node where the user has subordinate ids (``getsubids $USER``), never on a
login node: it starts containers.  ``HPC_SANDBOX_SCRATCH`` (node-local disk) and ``HPC_SANDBOX_IMAGE_CACHE`` say
where they go.
"""

from __future__ import annotations

import inspect
import os
import shutil
from pathlib import Path

import pytest

from platoon.envs.harbor import HarborEnvironmentService
from platoon.envs.service import EnvSetupError

IMAGE = os.environ.get("PLATOON_TEST_SINGULARITY_IMAGE")
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not IMAGE, reason="set PLATOON_TEST_SINGULARITY_IMAGE to run Singularity end to end"),
    pytest.mark.skipif(
        not (shutil.which("singularity") or shutil.which("apptainer")), reason="no Singularity or Apptainer on PATH"
    ),
]
TASK = f"""artifacts = ["/app/answer.sh"]
[agent]
user = "65534"
[environment]
docker_image = "{IMAGE}"
workdir = "/app"
[verifier]
environment_mode = "separate"
user = "root"
timeout_sec = 120
"""
# As root: keep a secret for itself, run the agent's answer as nobody, and score 1 only when the answer ran as nobody,
# could not read the secret, and could read what the tests wrote for it (Docker's umask).  It leaves files of nobody's
# in the root filesystem, /tmp and /var/tmp, as such tests do.
TEST_SH = r"""
set -u
echo 0 > /logs/verifier/reward.txt
as_nobody() {
  if command -v setpriv >/dev/null 2>&1; then setpriv --reuid=65534 --regid=65534 --clear-groups "$@"
  else su -s /bin/sh nobody -c "$*"; fi
}
[ "$(id -u)" = 0 ] || { echo "not root: $(id -u)"; exit 0; }
mkdir -p /verifier_data && echo 42 > /verifier_data/answer
chmod 700 /verifier_data && chmod 600 /verifier_data/answer
echo "input" > /tmp/input.txt
mkdir -p /srv/work /tmp/work /var/tmp/work && chown 65534:65534 /srv/work /tmp/work /var/tmp/work
got=$(as_nobody sh /app/answer.sh 2>&1)
echo "the answer said: $got"
if [ "$got" = "uid=65534 secret=denied input=input" ]; then echo 1 > /logs/verifier/reward.txt; fi
"""
ANSWER = (
    'echo "uid=$(id -u) secret=$(cat /verifier_data/answer 2>/dev/null || echo denied) input=$(cat /tmp/input.txt)"; '
    "echo left > /srv/work/f; echo left > /tmp/work/f; echo left > /var/tmp/work/f\n"
)


def _hpc_sandbox_takes_fakeroot() -> bool:
    hpc_sandbox = pytest.importorskip("hpc_sandbox")
    return "fakeroot" in inspect.signature(hpc_sandbox.SingularityProvider).parameters


async def test_a_root_verifier_runs_the_agents_code_as_nobody_and_leaves_nothing(make_task, tmp_path: Path):
    takes_fakeroot = _hpc_sandbox_takes_fakeroot()
    scratch = Path(os.environ.get("HPC_SANDBOX_SCRATCH") or tmp_path / "scratch") / f"platoon-root-{os.getpid()}"
    task = make_task("root", task_toml=TASK, test_sh=TEST_SH)
    options = {"scratch_root": str(scratch), "allow_network_none": True}
    if os.environ.get("HPC_SANDBOX_IMAGE_CACHE"):
        options["image_cache"] = os.environ["HPC_SANDBOX_IMAGE_CACHE"]
    service = HarborEnvironmentService(
        task_dirs=[task],
        provider="singularity",
        provider_options=options,
        verifier_provider_options={"rootfs_layout": "copy", "fakeroot": True},
        agent_user="ignore",
        verifier_user="honor",
    )
    listed = await service.list_tasks("train")
    if not takes_fakeroot:
        with pytest.raises(EnvSetupError, match="takes no fakeroot option"):
            await service.check_tasks("train", listed)
        return

    await service.check_tasks("train", listed)
    session = await service.open(await service.get_task("root"), rollout_id="root-verifier")
    try:
        written = await session.call_tool("write", {"path": "/app/answer.sh", "content": ANSWER})
        assert not written.is_error
        outcome = await session.grade()
    finally:
        await session.close()

    assert (outcome.status, outcome.rewards) == ("scored", {"reward": 1.0}), outcome.metadata.get("test_output")
    assert outcome.metadata["verifier_user"] == "root"
    for provider in {id(p): p for p in (service.provider, service.verifier_provider)}.values():
        assert await provider.inner.wait_for_deletions(120)
    sessions = scratch / "sessions"
    left = [path.name for path in sessions.iterdir() if path.name.startswith("hpcsb-")]
    assert left == [] and list((sessions / ".trash").iterdir()) == []  # the verifier's sandbox too
