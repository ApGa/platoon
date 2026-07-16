"""AReaL SGLang launcher with Platoon's spawn-safe scheduler compatibility hook."""

from __future__ import annotations

import os
import sys

from platoon.sglang_scheduler_compat import install_areal_scheduler_process_target


def main(argv: list[str] | None = None) -> None:
    # Keep AReaL's launcher behavior intact; only replace the scheduler process
    # target before Engine._launch_subprocesses pickles it for ``spawn``.
    from areal.v2.inference_service.sglang.launch_server import areal_launch_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree
    from sglang.srt.utils.common import suppress_noisy_warnings

    suppress_noisy_warnings()
    server_args = prepare_server_args(list(sys.argv[1:] if argv is None else argv))
    install_areal_scheduler_process_target()

    try:
        areal_launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()

