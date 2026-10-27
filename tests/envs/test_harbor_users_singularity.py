"""Users honored on Singularity with ``--fakeroot``, end to end (``agent_user: honor``): the user fixtures, their images
built with the ``oci`` builder, which records each image's owner manifest, score their reference solution 1.0 and
nothing 0.0; the agent's commands run as its user, with its own ``HOME``, and its ``edit`` tool cannot read the answer
that only root may read; nothing of the sandboxes stays on the node.  With an hpc-sandbox that applies no owner
manifest (before 0.5), the tasks whose images give files to their users are refused when they are listed instead.

Skipped unless ``PLATOON_TEST_SINGULARITY_USERS`` is set, and SingularityCE 4.1 or later (``build --oci``) and
hpc-sandbox are installed.  Run it on a node where the user has subordinate ids (``getsubids $USER``) and that reaches
Docker Hub, never on a login node: it builds images and starts containers.  ``HPC_SANDBOX_SCRATCH`` (node-local disk)
and ``HPC_SANDBOX_IMAGE_CACHE`` say where they go.
"""

from __future__ import annotations

import dataclasses
import os
import shutil
from pathlib import Path

import pytest
from user_fixtures import TASKS, fetch_and_build, fetched_service, oracle_and_noop, the_agents_view

from platoon.envs.service import EnvSetupError
from platoon.sandbox.hpc import HpcProvider

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        not os.environ.get("PLATOON_TEST_SINGULARITY_USERS"),
        reason="set PLATOON_TEST_SINGULARITY_USERS to run Singularity end to end",
    ),
    pytest.mark.skipif(not shutil.which("singularity"), reason="no singularity on PATH"),
]


def _applies_owner_manifests() -> bool:
    hpc_sandbox = pytest.importorskip("hpc_sandbox")
    return "file_owners" in {field.name for field in dataclasses.fields(hpc_sandbox.SandboxSpec)}


async def test_the_user_fixtures_score_their_oracles_and_no_ops_in_fakeroot_sandboxes(tmp_path, monkeypatch):
    scratch = Path(os.environ.get("HPC_SANDBOX_SCRATCH") or tmp_path / "scratch") / f"platoon-users-{os.getpid()}"
    options = {"scratch_root": str(scratch), "fakeroot": True, "rootfs_layout": "copy", "rootfs_templates": True}
    if os.environ.get("HPC_SANDBOX_IMAGE_CACHE"):
        options["image_cache"] = os.environ["HPC_SANDBOX_IMAGE_CACHE"]
    service = fetched_service(tmp_path, monkeypatch, provider="singularity", provider_options=options)

    assert await fetch_and_build(service, builder="oci", scratch=tmp_path / "builds") == sorted(TASKS)
    if not _applies_owner_manifests():
        with pytest.raises(EnvSetupError, match="Install hpc-sandbox 0.5"):
            await service.check_tasks("train", ["agent-user"])
        return
    await service.check_tasks("train", list(TASKS))

    for name in TASKS:
        oracle, noop = await oracle_and_noop(service, name)
        assert (oracle.status, oracle.rewards) == ("scored", {"reward": 1.0}), (name, oracle.metadata)
        assert (noop.status, noop.rewards) == ("scored", {"reward": 0.0}), (name, noop.metadata)

    view = await the_agents_view(service)
    assert view["who"] == "1000\n/srv/agent", view
    assert view["edit_error"] == "True" and "Permission denied" in view["edit"], view
    provider = service.provider
    assert isinstance(provider, HpcProvider) and await provider.inner.wait_for_deletions(120)
    sessions = scratch / "sessions"
    assert [path.name for path in sessions.iterdir() if path.name.startswith("hpcsb-")] == []
    assert list((sessions / ".trash").iterdir()) == []  # what the fixtures' users wrote was given back
