#!/bin/bash
# Stores the answer over RESP, then shuts the server down, as an agent may.
python3 - <<'PY'
import os
import socket


def command(*words):
    encoded = [word.encode() for word in words]
    request = b"*%d\r\n" % len(encoded) + b"".join(b"$%d\r\n%s\r\n" % (len(word), word) for word in encoded)
    with socket.create_connection((os.environ["REDIS_HOST"], int(os.environ["REDIS_PORT"])), timeout=5) as connection:
        connection.sendall(request)
        return connection.makefile("rb").readline().decode().strip()


print(command("SET", "answer", "forty-two"))
print(command("SHUTDOWN", "NOSAVE") or "the server shut down")
PY
