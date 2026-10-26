#!/bin/bash
stored=$(python3 - <<'PYTHON'
import os, socket
try:
    with socket.create_connection((os.environ["REDIS_HOST"], 6379), timeout=10) as connection:
        connection.sendall(b"*2\r\n$3\r\nGET\r\n$6\r\nanswer\r\n")
        reply = connection.recv(64)
    print(reply.split(b"\r\n")[1].decode() if reply.startswith(b"$") and not reply.startswith(b"$-1") else "nothing")
except Exception as exc:
    print(f"nothing ({exc})")
PYTHON
)
if [ "$stored" = "forty-two" ]; then
  echo "the server holds the answer"
  echo 1 > /logs/verifier/reward.txt
else
  echo "the server answered: $stored"
  echo 0 > /logs/verifier/reward.txt
fi
