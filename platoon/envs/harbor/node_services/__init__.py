"""Node services: services that a node runs once for every rollout on it, such as a judge pool, which a task's stack
reaches in place of a sidecar of its own (RFC 0010, section 8).

A Harbor task's compose service that an overlay marks ``external: <node service>`` is not started per rollout: the
node runs the node service once, shared by every rollout on the node, and each rollout's stack reaches it by the
service's name at the node service's port, through a **scope** of its own that lets only the declared routes through
and tells the service which task the rollout is on.  The modules:

- :mod:`~platoon.envs.harbor.node_services.spec`: the declaration (an environment's ``node_services``, or the site
  profile's) and where the service lives on a node;
- :mod:`~platoon.envs.harbor.node_services.runner`: what ``platoon train`` and ``platoon eval`` start before any
  rollout, a supervisor per service, and stop with the run;
- :mod:`~platoon.envs.harbor.node_services.supervisor`: the per-node supervisor (``python -m
  platoon.envs.harbor.node_services``), which owns the service under a node lock, checks its health, starts it again
  when it fails, and dies with the run;
- :mod:`~platoon.envs.harbor.node_services.instance`: one start of the service, in a sandbox of the environment's
  provider;
- :mod:`~platoon.envs.harbor.node_services.front`: the scopes, one Unix socket per rollout;
- :mod:`~platoon.envs.harbor.node_services.client`: how a session opens its scope and probes it;
- :mod:`~platoon.envs.harbor.node_services.protocol`: the supervisor's control protocol;
- :mod:`~platoon.envs.harbor.node_services.staging`: binds copied once per node.

A rollout whose node service failed while it used it is ``invalid`` (``node_service_down``), never scored 0.
"""

from platoon.envs.harbor.node_services.client import NodeScope, NodeServiceUnavailable, open_scope, ping
from platoon.envs.harbor.node_services.runner import NodeServiceRunner, declared_node_services, node_services_of
from platoon.envs.harbor.node_services.spec import NodeServiceError, NodeServiceSpec, node_dir

__all__ = [
    "NodeScope",
    "NodeServiceError",
    "NodeServiceRunner",
    "NodeServiceSpec",
    "NodeServiceUnavailable",
    "declared_node_services",
    "node_dir",
    "node_services_of",
    "open_scope",
    "ping",
]
