# Stack fixtures for Docker

Harbor tasks written for platoon, whose compose files make them multi-container stacks of real images:
`tests/integration/test_harbor_stacks_docker.py` runs them on the `docker` provider, and
`tests/sandbox/test_sandbox_compose.py` checks the compose projects they become against a fake docker CLI. Their images
are pinned by digest. In the `app-redis` tasks, `main` talks to a Redis sidecar over RESP, Redis's protocol, with a
standard-library client.

| Task | What it checks |
|---|---|
| `app-redis` | `main` stores a value in a `redis:7-alpine` sidecar, whose entrypoint takes its data directory and drops root for the `redis` user before the server starts; the tests read the value back. It runs without a network, as training does |
| `app-redis-crash` | the same, but the solution shuts the sidecar down after its write (`SHUTDOWN NOSAVE`), as an agent may: the sidecar is down when grading probes it |
| `app-redis-unhealthy` | a sidecar whose healthcheck asks a port that nothing listens on, so the stack never starts |
| `app-privileges` | what only Docker gives, to a dataset fetched with `refuse: {privileges: false}`: `router` holds `NET_ADMIN`, `judge` is privileged, and `main` holds neither. What `judge` writes as another user in the volume it shares with `main` is deleted with the stack, as root |
