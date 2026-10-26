"""platoon's route forwarder: relays the connections it accepts on a loopback TCP port to a Unix socket, or the other
way round.

Sandbox providers start it inside a sandbox without a network, so that processes there reach a service on the host,
such as the model gateway, through the Unix socket of the host directory bound into the sandbox
(:mod:`platoon.sandbox.routes`).  The docker provider also starts it the other way round, as the relay through which
this host reaches a port of a stack's service (:mod:`platoon.sandbox.compose`).  It is a script, not a module platoon
imports: standard library only, for any Python 3.6 or later, since it runs with whatever interpreter the sandbox or the
image has.

``python3 -I forwarder.py --socket PATH --port PORT --port-file FILE [--host 127.0.0.1]``

It listens on ``host:port`` (``--port 0``: a free port), writes the port it got to ``--port-file`` once it accepts
connections, and then serves until it is killed.  Each connection opens its own connection to the socket, so the
socket may appear after the forwarder starts.  It connects by the socket's name from inside its directory, which
keeps a long socket path within the limit of an ``AF_UNIX`` address.

``python3 -I forwarder.py --serve-socket PATH --port PORT [--host 127.0.0.1]``

It listens on the Unix socket ``PATH``, which any user who reaches its directory may connect to (mode 0666), and
relays each connection to ``host:port``.  The socket appears at ``PATH`` only once it accepts connections: it is bound
under another name, then renamed.
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


def relay(client, family, address):
    """Serve one accepted connection: open one to ``address`` (of the socket ``family``) and copy both ways until both
    sides end."""

    upstream = socket.socket(family, socket.SOCK_STREAM)
    try:
        upstream.connect(address)
    except OSError as error:
        sys.stderr.write("platoon forwarder: cannot connect to %s: %s\n" % (address, error))
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


def serve(listener, family, address):
    """Accept connections on ``listener`` until killed, relaying each to ``address`` in a thread of its own."""

    while True:
        try:
            client, _ = listener.accept()
        except InterruptedError:
            continue
        worker = threading.Thread(target=relay, args=(client, family, address))
        worker.daemon = True
        worker.start()


def listen_on_socket(path):
    """A listener on the Unix socket ``path`` that any user who reaches it may connect to, which appears at ``path``
    only once it accepts connections.  It binds by the socket's name from inside its directory, which keeps a long
    path within the limit of an ``AF_UNIX`` address."""

    directory, name = os.path.split(os.path.abspath(path))
    os.chdir(directory)
    temporary = "%s.%d.tmp" % (name, os.getpid())
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(temporary)
    listener.listen(BACKLOG)
    os.chmod(temporary, 0o666)
    os.rename(temporary, name)
    return listener


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--socket", help="the Unix socket to relay to, from a TCP port")
    mode.add_argument("--serve-socket", help="the Unix socket to listen on, relaying to a TCP port")
    parser.add_argument("--port", type=int, required=True, help="the TCP port to listen on (0: any), or relay to")
    parser.add_argument("--port-file", help="where to write the port once listening (with --socket)")
    parser.add_argument("--host", default="127.0.0.1", help="the address to listen on, or relay to")
    arguments = parser.parse_args(argv)
    if arguments.socket is not None and arguments.port_file is None:
        parser.error("--socket needs --port-file")
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)  # outlive the shell that started it
    if arguments.serve_socket is not None:
        listener = listen_on_socket(os.path.abspath(arguments.serve_socket))
        serve(listener, socket.AF_INET, (arguments.host, arguments.port))
        return
    port_file = os.path.abspath(arguments.port_file)
    directory, socket_name = os.path.split(os.path.abspath(arguments.socket))
    os.chdir(directory)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((arguments.host, arguments.port))
    listener.listen(BACKLOG)
    write_port(port_file, listener.getsockname()[1])
    serve(listener, socket.AF_UNIX, socket_name)


if __name__ == "__main__":
    main()
