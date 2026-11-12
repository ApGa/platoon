"""The tests of the ``platoon-harness`` harness and program, which need the ``platoon-harness`` extra: without it, they
skip.  They share core's test helpers from ``tests/``, the testing kit's toys (``testing_toys``) and the training tests'
fakes (``fakes``), which this conftest puts on ``sys.path``.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip(
    "platoon_harness",
    reason="platoon-harness is not installed: install the platoon-harness extra (uv sync --dev --extra tinker --extra "
    "ors --extra platoon-harness)",
)

TESTS = Path(__file__).resolve().parents[3]
for helpers in (TESTS / "testing", TESTS / "train"):
    if str(helpers) not in sys.path:
        sys.path.insert(0, str(helpers))
pytest.register_assert_rewrite("teams_toys")

from platoon_harness.drivers.levels import Conformance  # noqa: E402
from platoon_harness.drivers.wire import WireDriver  # noqa: E402
from platoon_harness.runtime import Admission  # noqa: E402
from platoon_harness.testing import check_driver  # noqa: E402
from teams_toys import WIRED  # noqa: E402

from platoon.gateway import ModelGateway  # noqa: E402
from platoon.gateway.core import AgentSession  # noqa: E402
from platoon.harness.builtin.platoon_harness.authority import PlatoonAuthority  # noqa: E402


@pytest.fixture(autouse=True)
def _sdk_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Each test's platoon-harness root, where its sessions and kept conformance results live, a folder of its own,
    with no ``PLATOON_HARNESS_CONFIG`` read, and beside it platoon's node-local scratch, where an episode
    keeps its files and makes its working directory."""

    root = tmp_path / "platoon-harness"
    monkeypatch.setenv("PLATOON_HARNESS_HOME", str(root))
    monkeypatch.delenv("PLATOON_HARNESS_CONFIG", raising=False)
    monkeypatch.setenv("PLATOON_NODE_SCRATCH", str(tmp_path / "scratch"))
    return root


@pytest.fixture(scope="session")
def wired(tmp_path_factory: pytest.TempPathFactory) -> Conformance:
    """The conformance result that earns the wire program's sealed launch trainable, made once: each check starts the
    program afresh."""

    folder = tmp_path_factory.mktemp("wired-checks")
    return asyncio.run(check_driver(lambda: WireDriver([sys.executable, str(WIRED)]), folder=folder))


@pytest.fixture
def gateway_agents(monkeypatch: pytest.MonkeyPatch) -> list[AgentSession]:
    """Every agent platoon's gateway opened in the test's rollouts, as it closes them."""

    closed: list[AgentSession] = []
    close = ModelGateway.close_rollout

    def closing(self: ModelGateway, rollout_id: str) -> list[AgentSession]:
        agents = close(self, rollout_id)
        closed.extend(agents)
        return agents

    monkeypatch.setattr(ModelGateway, "close_rollout", closing)
    return closed


@pytest.fixture
def admitted(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, Admission]]:
    """Each agent of the test's teams and what it was admitted on, as the authority gave it: the access its driver was
    given, and the platoon agent its backend names."""

    admissions: list[tuple[str, Admission]] = []
    admit = PlatoonAuthority.admit

    async def admitting(self: PlatoonAuthority, spawn: Any) -> Admission:
        admissions.append((spawn.agent, await admit(self, spawn)))
        return admissions[-1][1]

    monkeypatch.setattr(PlatoonAuthority, "admit", admitting)
    return admissions
