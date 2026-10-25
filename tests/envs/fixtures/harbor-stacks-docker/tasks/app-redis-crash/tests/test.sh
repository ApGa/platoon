#!/bin/bash
# Reads the answer back over RESP, Redis's protocol, with Python's standard library.
stored=$(python3 - <<'PY'
import os
import socket


def command(*words):
    encoded = [word.encode() for word in words]
    request = b"*%d\r\n" % len(encoded) + b"".join(b"$%d\r\n%s\r\n" % (len(word), word) for word in encoded)
    with socket.create_connection((os.environ["REDIS_HOST"], int(os.environ["REDIS_PORT"])), timeout=5) as connection:
        connection.sendall(request)
        reply = connection.makefile("rb")
        header = reply.readline().decode().strip()
        if header.startswith("$") and int(header[1:]) >= 0:
            return reply.read(int(header[1:]) + 2)[:-2].decode()
        return header


try:
    print(command("GET", "answer"))
except OSError as exc:
    print(f"nothing ({exc})")
PY
)
if [ "$stored" = "forty-two" ]; then
  echo "the cache holds the answer"
  echo 1 > /logs/verifier/reward.txt
else
  echo "the cache answered: $stored"
  echo 0 > /logs/verifier/reward.txt
fi
