"""A remote tool's meta reaches its ``ToolSpec`` however the openreward client lists it.

openreward's env server (0.1.125 to 0.1.163) sends no ``_meta``, so over ``ors+http://`` a tool's hints come from the
``read_only_tools`` and ``tool_meta`` options.  A server that sends meta anyway reaches platoon through the client's
provider-format listing (``test_ors_remote``); a client whose listed tools carry it, as a later SDK's may, must keep
it as well.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from platoon.envs.ors import sdk
from platoon.envs.service import get_service

pytestmark = pytest.mark.asyncio


class _Session:
    """A remote session whose client lists its tools as objects that carry their meta."""

    def __init__(self, tools: list[SimpleNamespace]) -> None:
        self.tools = tools

    async def list_tools(self, format=None):
        if format is not None:
            raise AssertionError(f"listed again as {format!r}: the client's own listing was enough")
        return self.tools

    async def terminal_tool(self):
        return None


@pytest.mark.parametrize("attribute", ["_meta", "meta", "metadata"])
async def test_meta_on_a_listed_tool_reaches_its_spec(attribute):
    look = SimpleNamespace(name="look", description="Look around.", input_schema=None)
    setattr(look, attribute, {"readOnlyHint": True})
    act = SimpleNamespace(name="act", description="Act.", input_schema=None)

    raw = await sdk.remote_tools(_Session([look, act]))

    assert {tool.name: tool.meta for tool in raw} == {"look": {"readOnlyHint": True}, "act": None}
    service = get_service("ors+http://127.0.0.1:9/MetaEnv")
    lease = SimpleNamespace(backend=SimpleNamespace(env_attr=lambda name: None, url="http://127.0.0.1:9"))
    specs = {spec.name: spec for spec in (service.tool_spec(tool, lease) for tool in raw)}
    assert specs["look"].read_only and specs["look"].meta == {"readOnlyHint": True}
    assert not specs["act"].read_only and specs["act"].meta is None
