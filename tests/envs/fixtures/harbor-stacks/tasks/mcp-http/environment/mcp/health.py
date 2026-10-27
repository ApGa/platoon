"""Exit 0 once the MCP server accepts connections at the URL given: the fixture server's healthcheck."""

import socket
import sys
import urllib.parse

url = urllib.parse.urlsplit(sys.argv[1])
try:
    socket.create_connection((url.hostname, url.port), timeout=2).close()
except OSError as exc:
    print(exc, file=sys.stderr)
    sys.exit(1)
