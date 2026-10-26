# Stack fixtures

Harbor tasks written for platoon, whose compose files make them multi-container stacks: `main` and sidecar services
that the tests of `tests/envs/test_harbor_stacks.py` run on the `process` provider. Their services are
standard-library Python programs that platoon's process provider runs on the host, so each listens on its listen
variable (`PLATOON_PORT_<port>`) on the host's loopback when it is set, and otherwise on its declared port of every
address (a real stack's own network, where the other services reach it by name), and writes below
`$PLATOON_SANDBOX_ROOT` when that is set; real stacks leave both unset.

| Task | What it checks |
|---|---|
| `app-kv` | `main` stores a value through a key-value sidecar (`kv`), which records the write in its own verifier log directory; the tests read the value back through the sidecar and check its record |
| `app-kv-crash` | the same, with a sidecar that exits after its first write, during the agent phase |
| `app-kv-unhealthy` | a sidecar whose healthcheck never passes, so the stack never starts |
| `app-kv-seeded` | an init job (`seed`, waited for with `service_completed_successfully`) writes a value that the agents must copy |
| `sidecar-artifacts` | after Harbor's example of the same name: a separate verifier grades evidence collected from the sidecar, its own log and a snapshot that a collect hook takes |
