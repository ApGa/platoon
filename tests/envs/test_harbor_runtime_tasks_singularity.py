"""Runtime tasks end to end on Singularity, with verifiers that run as root under ``--fakeroot``: every red-team case of
``platoon.testing.check_runtime_tasks``, its grading ones included, which need what only such a sandbox gives the
launcher: root to seal the tests and the reward, another user to replay the submission as, and a process tree of its
own to kill what a grader leaves.

Skipped unless ``PLATOON_TEST_SINGULARITY_IMAGE`` names an image (a local ``.sif``, or a ``docker://`` reference the
node can pull) with ``python3`` and ``sh``, ``find`` and ``chown``, and Singularity or Apptainer and an hpc-sandbox
that takes ``fakeroot`` are installed.  Run it on a node where the user has subordinate ids (``getsubids $USER``),
never on a login node: it starts containers.  ``HPC_SANDBOX_SCRATCH`` (node-local disk) and ``HPC_SANDBOX_IMAGE_CACHE``
say where they go.
"""

from __future__ import annotations

import inspect
import os
import shutil
from pathlib import Path

import pytest

from platoon.envs.harbor import HarborEnvironmentService
from platoon.testing import check_runtime_tasks

IMAGE = os.environ.get("PLATOON_TEST_SINGULARITY_IMAGE")
pytestmark = [
    pytest.mark.skipif(not IMAGE, reason="set PLATOON_TEST_SINGULARITY_IMAGE to run Singularity end to end"),
    pytest.mark.skipif(
        not (shutil.which("singularity") or shutil.which("apptainer")), reason="no Singularity or Apptainer on PATH"
    ),
]
TEMPLATE = {
    "environment": {"docker_image": "red-team", "network_mode": "no-network", "workdir": "/app"},
    "agent": {"timeout_sec": 120},
    "verifier": {"environment_mode": "separate", "user": "root", "timeout_sec": 120},
}


def test_runtime_tasks_pass_every_red_team_case_with_root_verifiers(tmp_path: Path):
    hpc_sandbox = pytest.importorskip("hpc_sandbox")
    if "fakeroot" not in inspect.signature(hpc_sandbox.SingularityProvider).parameters:
        pytest.skip("this hpc-sandbox takes no fakeroot option")
    scratch = Path(os.environ.get("HPC_SANDBOX_SCRATCH") or tmp_path / "scratch") / f"platoon-runtime-{os.getpid()}"
    options = {"scratch_root": str(scratch), "allow_network_none": True}
    if os.environ.get("HPC_SANDBOX_IMAGE_CACHE"):
        options["image_cache"] = os.environ["HPC_SANDBOX_IMAGE_CACHE"]
    service = HarborEnvironmentService(
        task_dirs=[],
        provider="singularity",
        provider_options=options,
        agent_user="ignore",
        runtime_tasks={
            "images": {"red-team": IMAGE},
            "template": TEMPLATE,
            "store": str(tmp_path / "store"),
            "verifier_provider_options": {"rootfs_layout": "copy", "fakeroot": True},
        },
    )

    check_runtime_tasks(service)
