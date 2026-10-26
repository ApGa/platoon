# Stack fixtures with real images

Harbor tasks written for platoon whose sidecars run official database images, which switch users when they start:
their entrypoints, as root, give the data directory to a user of their own and run the server as it. Only a provider
that runs images runs them (stacks as hpc-sandbox pods: `tests/integration/test_harbor_stacks_hpc.py`). The sidecars'
images are pinned by digest, and the databases' compose files give them no command: they run their images' `ENTRYPOINT`
and `CMD`, which `platoon prepare` pins in the image map with the images' configuration. `main` talks to the server with a
standard-library client: the tests carry their own copy, which the agents never see.

| Task | Its sidecar |
|---|---|
| `app-redis` | `redis:7-alpine`, whose entrypoint gives `/data` to `redis` and switches to it; `main` stores a value, which the tests read back |
| `app-postgres` | `postgres:16-alpine`, whose entrypoint initializes its data directory as `postgres` (initdb refuses root); `main` stores a row, which the tests read back |
| `judge-logs` | a judge on `redis:7-alpine`, whose compose file gives it a command: as root, it gives the verifier log directory it binds to `redis` and records there as that user, in a file only `redis` reads; the tests read the record, which grading copies into `main` |
