# Host-network fixtures

A Harbor task written for platoon whose compose file makes it a multi-container stack, and the overlay that lets the
host-network fallback (`stacks: host_network`) run it. `tests/envs/test_harbor_host_network.py` runs them on the
`process` provider, and `tests/integration/test_harbor_stacks_host_network.py` on Singularity.

| Path | What it is |
|---|---|
| `tasks/app-kv-auth` | `main` stores a value through a key-value sidecar (`kv`), which records the write in its own verifier log directory; the tests, through `[verifier].env`, read the value back and check the record. The sidecar's healthcheck names its own port on its loopback (`http://localhost:6379/`), and the store answers only requests with its token where `KV_TOKEN` is set, on the address and port that `KV_HOST` and `KV_PORT` give where they are set |
| `overlays/authenticate` | What the fallback needs of the store: the variable that sets the port it listens on (`listen`), the one that carries the stack's token (`token`, which every service of the stack gets, so `main` and the tests send it too), and the one that sets its address (`bind`, which the fallback sets to `127.0.0.1`) |

Without the overlay the fallback refuses the task (its sidecar is not authenticated), and on every other stack provider
the overlay changes nothing: the store then listens on 6379 of every address of the stack's own network, without a
token, or on the process provider's listen variable (`PLATOON_PORT_6379`) on the host's loopback.
