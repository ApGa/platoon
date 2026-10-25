"""A fake docker CLI for the tests of stacks on the docker provider (``test_sandbox_compose.py``): it logs each call
as a JSON line to ``$FAKE_DOCKER_LOG``, and keeps its containers and networks in ``$FAKE_DOCKER_STATE`` (JSON, under a
lock: the provider runs some commands at once), changing them as Docker would.

- ``compose ... --file <file> create --no-recreate <service>`` makes the service's container from the project's file
  (its name, labels and network, and the internal network), and keeps the file as it first read it (``projects``).
  It reads the file as compose does, as YAML 1.2, where a plain ``1e3`` is a number, which an ``entrypoint`` must not
  hold; ``compose version`` answers unless ``$FAKE_DOCKER_NO_COMPOSE`` is set, and ``version`` gives the daemon's
  (``$FAKE_DOCKER_SERVER_VERSION``, 29.0.0).
- ``start`` runs a container, or has it exit at once with a status (``$FAKE_DOCKER_EXITED``: ``service:status,...``).
- ``exec`` succeeds in a running container (it lists the container's own processes, and a route forwarder prints
  its port), but a command of a ``$FAKE_DOCKER_UNHEALTHY`` service, whose healthchecks fail.
- ``inspect`` tells a container's state, or its networks; ``logs`` prints a log line on each stream.
- ``network create|connect|disconnect|rm|ls``, ``ps`` with label filters (``label=<key>=<value>``, or ``label=<key>``
  for any value), ``rm`` and ``kill`` do as Docker does, which refuses to connect a container twice, to disconnect one
  that is not connected, or to remove a network in use; ``cp`` copies nothing, into a running container.
- ``run --rm ... --volume=<dir>:/platoon-remains <image> -rf -- <paths>`` deletes those paths, as root could.

``$FAKE_DOCKER_FAIL`` lists commands that fail (``create``, ``start``, ``network connect``, ...), with
``$FAKE_DOCKER_ERROR`` on stderr, and ``$FAKE_DOCKER_FAIL_ONCE`` names one whose first call fails.  The daemon's
side: ``$FAKE_DOCKER_SLOW`` names a command that the daemon carries out a second later, in a process of its own that
outlives the CLI, as Docker completes a request whose CLI was killed (touching ``<$FAKE_DOCKER_LOG>.done`` then);
``$FAKE_DOCKER_LOST`` one that it carries out, but whose answer the CLI never gets, so that the CLI fails.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import yaml

SERVICE_LABEL = "platoon.service"
STACK_LABEL = "platoon.stack"
REMAINS = "/platoon-remains"


def main(args: list[str]) -> int:
    with open(os.environ["FAKE_DOCKER_LOG"], "a") as log:
        log.write(json.dumps(args) + "\n")
    command = command_name(args)
    if command in os.environ.get("FAKE_DOCKER_FAIL", "").split(","):
        print(os.environ.get("FAKE_DOCKER_ERROR", "simulated failure"), file=sys.stderr)
        return 1
    if args[:2] == ["compose", "version"]:
        if os.environ.get("FAKE_DOCKER_NO_COMPOSE"):
            print("docker: 'compose' is not a docker command.", file=sys.stderr)
            return 1
        print("Docker Compose version v2.30.0")
        return 0
    if args[0] == "version":
        print(os.environ.get("FAKE_DOCKER_SERVER_VERSION", "29.0.0"))
        return 0
    document = None
    if args[0] == "compose":  # compose reads its file before it asks the daemon anything
        try:
            document = yaml.load(Path(args[args.index("--file") + 1]).read_text(), Loader=_Yaml12Loader)
            invalid = _invalid(document)
        except (yaml.YAMLError, KeyError, TypeError) as exc:
            invalid = f"cannot read it: {exc}"
        if invalid is not None:
            print(f"validating compose.yaml: {invalid}", file=sys.stderr)
            return 15
    if command == os.environ.get("FAKE_DOCKER_SLOW"):
        return _later(args, command, document)
    code = _locked(args, command, document)
    if command == os.environ.get("FAKE_DOCKER_LOST"):
        print("error during connect: EOF", file=sys.stderr)  # the daemon did it; its answer never came
        return 1
    return code


def command_name(args: list[str]) -> str:
    """What ``$FAKE_DOCKER_FAIL`` names a call by: its verb (``create``, ``start``), or ``network <verb>``."""

    if args[0] == "compose":
        return next((arg for arg in args if arg in ("create", "version")), "compose")
    if args[0] == "network":
        return f"network {args[1]}"
    return args[0]


def _later(args: list[str], command: str, document: Any) -> int:
    """Carry ``args`` out a second later, as the daemon would, in a session of its own, which a kill of the CLI's
    process group does not reach; the CLI waits for it, and says what it said."""

    daemon = os.fork()
    if daemon == 0:
        os.setsid()
        time.sleep(1.0)
        code = _locked(args, command, document)
        Path(os.environ["FAKE_DOCKER_LOG"]).with_suffix(".done").touch()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    return os.waitstatus_to_exitcode(os.waitpid(daemon, 0)[1])


def _locked(args: list[str], command: str, document: Any) -> int:
    """Carry ``args`` out on the state, under its lock."""

    path = Path(os.environ["FAKE_DOCKER_STATE"])
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state: dict[str, Any] = {"containers": {}, "networks": {}, "projects": {}}
        if path.exists():
            state = json.loads(path.read_text())
        failed = state.setdefault("failed_once", [])
        if command == os.environ.get("FAKE_DOCKER_FAIL_ONCE") and command not in failed:
            failed.append(command)
            print(os.environ.get("FAKE_DOCKER_ERROR", "simulated failure"), file=sys.stderr)
            code = 1
        else:
            code = _run(args, state, document)
        path.write_text(json.dumps(state))
    return code


def _run(args: list[str], state: dict, document: Any) -> int:
    containers, networks = state["containers"], state["networks"]
    if args[0] == "compose":
        return _create(args, state, document)
    if args[0] == "start":
        container = containers.get(args[1])
        if container is None:
            return _no_such(args[1])
        exited = dict(entry.split(":") for entry in os.environ.get("FAKE_DOCKER_EXITED", "").split(",") if entry)
        service = container["labels"].get(SERVICE_LABEL)
        if service in exited:
            container.update(status="exited", exit=int(exited[service]))
        else:
            container.update(status="running", exit=0)
        return 0
    if args[0] == "inspect":
        container = containers.get(args[-1])
        if container is None:
            print(f"Error: No such object: {args[-1]}", file=sys.stderr)
            return 1
        if "NetworkSettings" in args[2]:
            addresses = {name: {"IPAddress": f"172.30.0.{2 + n}"} for n, name in enumerate(container["networks"])}
            print(json.dumps(addresses))
        else:
            print(f"{container['status']} {container['exit']}")
        return 0
    if args[0] == "exec":
        return _exec(args, containers)
    if args[0] == "logs":
        print(f"the output of {args[-1]}")
        print(f"the errors of {args[-1]}", file=sys.stderr)
        return 0
    if args[0] in ("rm", "kill"):
        names = [arg for arg in args[1:] if not arg.startswith("--")]
        missing = [name for name in names if name not in containers]
        for name in names:
            if args[0] == "rm":
                containers.pop(name, None)
            elif name in containers:
                containers[name].update(status="exited", exit=137)
        return _no_such(missing[0]) if missing else 0
    if args[0] == "network":
        return _network(args, containers, networks)
    if args[0] == "ps":
        for name, container in sorted(containers.items()):
            if _selected(container["labels"], args):
                print(name if "--quiet" in args else container["labels"].get(STACK_LABEL, ""))
        return 0
    if args[0] == "run":
        return _remove_remains(args)
    if args[0] == "cp":
        name = next(arg.split(":", 1)[0] for arg in args[1:] if ":" in arg)
        return 0 if containers.get(name, {}).get("status") == "running" else _no_such(name)
    print(f"the fake docker CLI does not know {args}", file=sys.stderr)
    return 2


def _create(args: list[str], state: dict, document: Any) -> int:
    if "create" not in args:
        print(f"the fake docker CLI does not know {args}", file=sys.stderr)
        return 2
    project, service = args[args.index("--project-name") + 1], args[-1]
    entry = document["services"][service]
    state["projects"].setdefault(project, document)
    internal = document["networks"]["internal"]
    state["networks"].setdefault(internal["name"], {"labels": internal["labels"]})
    if entry["container_name"] not in state["containers"]:
        state["containers"][entry["container_name"]] = {
            "project": project,
            "status": "created",
            "exit": 0,
            "labels": entry["labels"],
            "networks": [internal["name"]] if "networks" in entry else [],
        }
    return 0


def _invalid(document: Any) -> str | None:
    """What compose's validation says of a project file, where an ``entrypoint`` holds something but strings."""

    for service, entry in document["services"].items():
        for number, item in enumerate(entry["entrypoint"]):
            if not isinstance(item, str):
                return f"services.{service}.entrypoint.{number} must be a string"
    return None


def _exec(args: list[str], containers: dict) -> int:
    name = next((arg for arg in args[1:] if arg in containers), None)
    if name is None:
        return _no_such(args[-1])
    container = containers[name]
    if container["status"] != "running":
        print(f"Error response from daemon: container {name} is not running", file=sys.stderr)
        return 1
    if container["labels"].get(SERVICE_LABEL) in os.environ.get("FAKE_DOCKER_UNHEALTHY", "").split(","):
        print("connection refused", file=sys.stderr)
        return 1
    if "platoon-list" in args:
        print("1:1000")
    if "platoon-route" in args:
        print("18080")
    return 0


def _network(args: list[str], containers: dict, networks: dict) -> int:
    verb, names = args[1], [arg for arg in args[2:] if not arg.startswith("--")]
    if verb == "create":
        labels = dict(arg.removeprefix("--label=").split("=", 1) for arg in args if arg.startswith("--label="))
        networks[names[-1]] = {"labels": labels}
    elif verb in ("connect", "disconnect"):
        network, name = names
        if name not in containers:
            return _no_such(name)
        attached = containers[name]["networks"]
        if (network in attached) == (verb == "connect"):
            already = "already exists in" if verb == "connect" else "is not connected to"
            print(f"Error response from daemon: endpoint {name} {already} network {network}", file=sys.stderr)
            return 1
        if verb == "connect":
            attached.append(network)
        else:
            attached.remove(network)
    elif verb == "rm":
        used = [name for name in names if any(name in container["networks"] for container in containers.values())]
        for name in names:
            if name not in used:
                networks.pop(name, None)
        if used:
            error = f"error while removing network: {used[0]} has active endpoints"
            print(f"Error response from daemon: {error}", file=sys.stderr)
            return 1
    elif verb == "ls":
        for name, network in sorted(networks.items()):
            if _selected(network["labels"], args):
                print(name)
    return 0


def _remove_remains(args: list[str]) -> int:
    """``docker run --rm ... --volume=<dir>:/platoon-remains <image> -rf -- <paths>``, as root: whatever the modes."""

    (volume,) = [arg.removeprefix("--volume=") for arg in args if arg.startswith("--volume=")]
    directory = Path(volume.split(":", 1)[0])
    for path in args[args.index("--") + 1 :]:
        target = directory / os.path.relpath(path, REMAINS)
        if not target.is_dir():
            target.unlink(missing_ok=True)
            continue
        target.chmod(0o700)
        for root, dirs, _ in os.walk(target):  # top down: each directory opened up before it is entered
            for name in dirs:
                os.chmod(os.path.join(root, name), 0o700)
        shutil.rmtree(target)
    return 0


class _Yaml12Loader(yaml.SafeLoader):
    """Reads plain scalars as compose does (YAML 1.2's core schema): ``1e3``, ``.5e3`` and ``0o17`` are numbers."""


_Yaml12Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?(\.[0-9]+|[0-9]+(\.[0-9]*)?)([eE][-+]?[0-9]+)?$"),
    list("-+.0123456789"),
)
_Yaml12Loader.add_implicit_resolver("tag:yaml.org,2002:int", re.compile(r"^0o[0-7]+$"), ["0"])


def _selected(labels: dict[str, str], args: list[str]) -> bool:
    """Whether ``labels`` pass the call's label filters: ``label=<key>=<value>``, or ``label=<key>`` for any value."""

    for arg in args:
        if arg.startswith("--filter=label="):
            key, has_value, value = arg.removeprefix("--filter=label=").partition("=")
            if key not in labels or (has_value and labels[key] != value):
                return False
    return True


def _no_such(name: str) -> int:
    print(f"Error response from daemon: No such container: {name}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
