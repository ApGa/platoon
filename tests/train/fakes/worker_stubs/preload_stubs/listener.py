"""Keeps a listening socket open: not fork-safe."""

import socket

SOCKET = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
SOCKET.bind(("127.0.0.1", 0))
SOCKET.listen()
