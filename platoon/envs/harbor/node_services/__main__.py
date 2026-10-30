"""``python -m platoon.envs.harbor.node_services --spec FILE --parent PID``: the supervisor of one node service on
this node (:mod:`platoon.envs.harbor.node_services.supervisor`), as a run starts it
(:mod:`platoon.envs.harbor.node_services.runner`).  ``FILE`` holds the service's declaration, the run's cleanup labels
and the CPUs the run reserved for the service (``{"spec": ..., "labels": ..., "cpus": [...]}``; without ``cpus``, the
declaration's); ``PID`` is the run's process, whose end ends the supervisor."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
from pathlib import Path

from platoon.envs.harbor.node_services.spec import NodeServiceSpec
from platoon.envs.harbor.node_services.supervisor import Supervisor, die_with_parent, pin_cpus


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m platoon.envs.harbor.node_services", description=__doc__)
    parser.add_argument("--spec", required=True, help="the file that holds the declaration and the labels")
    parser.add_argument("--parent", type=int, default=None, help="the process whose end ends this one")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    signal.signal(signal.SIGINT, lambda signum, frame: None)  # until the supervisor's own handler: the run stops it
    die_with_parent(args.parent)
    data = json.loads(Path(args.spec).read_text())
    spec = NodeServiceSpec.from_json(data["spec"])
    # Before any thread starts: every thread and process after it inherits the CPUs.
    cpus = pin_cpus(data["cpus"] if isinstance(data.get("cpus"), list) else spec.cpus)
    supervisor = Supervisor(spec, parent_pid=args.parent, labels=data.get("labels") or {}, cpus=cpus)
    return asyncio.run(supervisor.run())


if __name__ == "__main__":
    sys.exit(main())
