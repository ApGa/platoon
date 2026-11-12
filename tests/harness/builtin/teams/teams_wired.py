"""A wire program: ScriptedDriver, named ``wired``, served on stdio as a harness in another process is, whose every turn
asks the model.  Its first argument changes that: ``hang`` asks the model in its first turn and then does nothing until
it is interrupted; ``forks`` reports a native child of its own once it is up; ``note <file>`` writes to ``<file>``, as
it starts, its command line and the gateway it was given.  It says where it is installed, its Python and this file, as
an isolation needs to confine it.  A team names it by a command of the interpreter and this file's absolute path
(``teams_toys.wire``)."""

import json
import sys
from pathlib import Path

from platoon_harness.drivers import AgentSpec, ChildSeen, DriverInfo, Gateway, Reporter
from platoon_harness.drivers.env import PYTHON
from platoon_harness.drivers.wire import serve
from platoon_harness.testing import AskModel, Hang, ScriptedDriver, every_turn_asks_the_model


class Wired(ScriptedDriver):
    def __init__(self, mode: str = "", *args: str) -> None:
        super().__init__([[AskModel(), Hang()]] if mode == "hang" else every_turn_asks_the_model(), name="wired")
        self._mode, self._args = mode, args

    def describe(self) -> DriverInfo:
        return super().describe().model_copy(update={"installed": (*PYTHON, Path(__file__))})

    async def start(self, spec: AgentSpec, report: Reporter) -> None:
        if self._mode == "note":
            access = spec.access
            assert isinstance(access, Gateway)
            seen = {"argv": sys.argv, "origin": access.origin, "tools": access.tools, "model": access.model}
            Path(self._args[0]).write_text(json.dumps(seen))
        await super().start(spec, report)
        if self._mode == "forks":
            await report(ChildSeen(native="its-own"))


if __name__ == "__main__":
    serve(Wired(*sys.argv[1:]))
