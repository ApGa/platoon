"""platoon's route forwarder: relays the connections it accepts on a loopback TCP port to a Unix socket.

Sandbox providers start it inside a sandbox without a network, so that processes there reach a service on the host,
such as the model gateway, through the Unix socket of the host directory bound into the sandbox
(:mod:`platoon.sandbox.routes`).  It is a script, not a module platoon imports: standard library only, for any
Python 3.6 or later, since it runs with whatever interpreter the sandbox has.

``python3 -I forwarder.py --socket PATH --port PORT --port-file FILE [--host 127.0.0.1]``

It listens on ``host:port`` (``--port 0``: a free port), writes the port it got to ``--port-file`` once it accepts
connections, and then serves until it is killed.  Each connection opens its own connection to the socket, so the
socket may appear after the forwarder starts.  It connects by the socket's name from inside its directory, which
keeps a long socket path within the limit of an ``AF_UNIX`` address.
"""

import argparse
import os
import signal
import socket
import sys
import threading

BUFFER_BYTES = 65536
BACKLOG = 128


def pipe(source, sink):
    """Copy ``source`` to ``sink`` until ``source`` ends, then end ``sink``'s write side."""

    try:
        while True:
            data = source.recv(BUFFER_BYTES)
            if not data:
                break
            sink.sendall(data)
    except OSError:
        pass
    finally:
        try:
            sink.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def relay(client, socket_name):
    """Serve one accepted connection: open one to the socket and copy both ways until both sides end."""

    upstream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        upstream.connect(socket_name)
    except OSError as error:
        sys.stderr.write("platoon forwarder: cannot connect to %s: %s\n" % (socket_name, error))
        upstream.close()
        client.close()
        return
    back = threading.Thread(target=pipe, args=(upstream, client))
    back.daemon = True
    back.start()
    pipe(client, upstream)
    back.join()
    upstream.close()
    client.close()


def write_port(path, port):
    """Publish the port atomically, so a reader never sees a partial file."""

    temporary = "%s.%d.tmp" % (path, os.getpid())
    with open(temporary, "w") as handle:
        handle.write("%d\n" % port)
    os.rename(temporary, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--socket", required=True, help="the Unix socket to relay to")
    parser.add_argument("--port", type=int, required=True, help="the TCP port to listen on (0: any free port)")
    parser.add_argument("--port-file", required=True, help="where to write the port once listening")
    parser.add_argument("--host", default="127.0.0.1", help="the address to listen on")
    arguments = parser.parse_args(argv)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)  # outlive the shell that started it
    port_file = os.path.abspath(arguments.port_file)
    directory, socket_name = os.path.split(os.path.abspath(arguments.socket))
    os.chdir(directory)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((arguments.host, arguments.port))
    listener.listen(BACKLOG)
    write_port(port_file, listener.getsockname()[1])
    while True:
        try:
            client, _ = listener.accept()
        except InterruptedError:
            continue
        worker = threading.Thread(target=relay, args=(client, socket_name))
        worker.daemon = True
        worker.start()


if __name__ == "__main__":
    main()
