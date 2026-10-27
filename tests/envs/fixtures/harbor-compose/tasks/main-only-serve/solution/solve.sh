#!/bin/bash
root="${PLATOON_SANDBOX_ROOT:-}"
python3 -c '
import socket, sys
with socket.create_connection(("127.0.0.1", int(open(sys.argv[1]).read())), timeout=5) as connection:
    print(connection.recv(1000).decode().strip())
' "$root/tmp/greeting.port" > answer.txt
