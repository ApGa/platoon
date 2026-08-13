#!/usr/bin/env python3
"""One-shot TCP reachability probe used by the ABCI PBS cluster survey."""

from __future__ import annotations

import argparse
import socket
import sys
import time


def serve(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("0.0.0.0", port))
        listener.listen(1)
        listener.settimeout(60)
        connection, address = listener.accept()
        with connection:
            payload = connection.recv(64)
            if payload != b"platoon-probe\n":
                raise RuntimeError(f"unexpected payload from {address}: {payload!r}")
            connection.sendall(b"ok\n")


def connect(host: str, port: int) -> None:
    deadline = time.monotonic() + 45
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=3) as connection:
                connection.sendall(b"platoon-probe\n")
                reply = connection.recv(64)
                if reply != b"ok\n":
                    raise RuntimeError(f"unexpected reply: {reply!r}")
                return
        except OSError as error:
            last_error = error
            time.sleep(0.25)
    raise RuntimeError(f"could not reach {host}:{port}: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    listen_parser = subparsers.add_parser("listen")
    listen_parser.add_argument("port", type=int)
    connect_parser = subparsers.add_parser("connect")
    connect_parser.add_argument("host")
    connect_parser.add_argument("port", type=int)
    args = parser.parse_args()
    if args.mode == "listen":
        serve(args.port)
    else:
        connect(args.host, args.port)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(error, file=sys.stderr)
        raise
