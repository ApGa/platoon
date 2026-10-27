"""A standard-library PostgreSQL client for platoon's fixtures: runs the SQL it is given on $PGHOST, port 5432, as the
postgres user, whom the server trusts, and prints each row's first column."""

import os
import socket
import struct
import sys


def receive(connection: socket.socket, size: int) -> bytes:
    data = b""
    while len(data) < size:
        chunk = connection.recv(size - len(data))
        if not chunk:
            raise ConnectionError("the server closed the connection")
        data += chunk
    return data


def until_ready(connection: socket.socket) -> list[str]:
    """Read the server's messages up to ReadyForQuery: the first column of each data row; an error raises."""

    rows = []
    while True:
        head = receive(connection, 5)
        body = receive(connection, struct.unpack("!i", head[1:])[0] - 4)
        if head[:1] == b"E":
            raise RuntimeError(body.decode(errors="replace"))
        if head[:1] == b"D":
            (length,) = struct.unpack("!i", body[2:6])
            rows.append(body[6 : 6 + length].decode() if length >= 0 else "")
        if head[:1] == b"Z":
            return rows


with socket.create_connection((os.environ.get("PGHOST", "db"), 5432), timeout=30) as connection:
    startup = b"user\0postgres\0database\0postgres\0\0"
    connection.sendall(struct.pack("!ii", len(startup) + 8, 196608) + startup)
    until_ready(connection)
    query = sys.argv[1].encode() + b"\0"
    connection.sendall(b"Q" + struct.pack("!i", len(query) + 4) + query)
    for row in until_ready(connection):
        print(row)
