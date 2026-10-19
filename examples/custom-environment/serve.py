"""Serve the order-packing environment on an ORS env server, for ``ors+http://`` services.

    cd examples/custom-environment
    PYTHONPATH=. uv run python serve.py --port 8080

The openreward SDK's ``Server`` serves every environment class it is given at ``/<ClassName>``.
"""

from __future__ import annotations

import argparse
import os

from openreward.environments import Server
from order_packing import OrderPackingEnv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    os.environ.setdefault("OPENREWARD_DISABLE_UPDATE_CHECK", "1")
    Server([OrderPackingEnv]).run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
