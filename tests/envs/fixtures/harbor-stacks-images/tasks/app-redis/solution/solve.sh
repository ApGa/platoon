#!/bin/bash
python3 - <<'PYTHON'
import os, socket
words = [b"SET", b"answer", b"forty-two"]
with socket.create_connection((os.environ["REDIS_HOST"], 6379), timeout=10) as connection:
    connection.sendall(b"*3\r\n" + b"".join(b"$%d\r\n%s\r\n" % (len(word), word) for word in words))
    assert connection.recv(64) == b"+OK\r\n"
PYTHON
