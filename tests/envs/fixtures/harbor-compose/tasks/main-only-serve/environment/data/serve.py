"""Answer every connection with the greeting and the secret.  PLATOON_SANDBOX_ROOT is set by platoon's process
provider only, which maps sandbox paths in commands but not in files."""

import os
import socket

root = os.environ.get("PLATOON_SANDBOX_ROOT", "")
with open(os.environ["SECRET_FILE"]) as handle:
    secret = handle.read().strip()
server = socket.socket()
server.bind(("127.0.0.1", 0))
server.listen()
port_file = f"{root}/tmp/greeting.port"
with open(f"{port_file}.tmp", "w") as handle:
    handle.write(str(server.getsockname()[1]))
os.replace(f"{port_file}.tmp", port_file)
while True:
    connection, _ = server.accept()
    with connection:
        connection.sendall(f"{os.environ['GREETING']} {secret}\n".encode())
