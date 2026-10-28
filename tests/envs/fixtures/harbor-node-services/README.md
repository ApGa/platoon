# Node-service fixtures

Harbor tasks written for platoon whose judge is a node service (`tests/envs/test_harbor_node_services.py`, on the
`process` provider). Each task's compose file pairs `main` with a privileged `judge`; the overlay
(`overlays/judge-pool`) marks `judge` external, served by the node service `judge-pool`, with the task's
`PROBLEM_ID` as each rollout's scope, so that no rollout starts a judge of its own.

| Path | What it is |
|---|---|
| `judge/judge.py` | the node service: a standard-library HTTP server on `$PLATOON_NODE_SERVICE_SOCKET` that refuses a request without `$PLATOON_NODE_SERVICE_TOKEN`, answers `GET /health` (503 while `/state/unhealthy` exists), grades `POST /evaluate` against the problem its `X-Platoon-Scope` names (code that starts with `slow` a minute later), and has an `/admin` route that no rollout may reach; each start appends its pid to `/state/starts.log`, each request its problem to `/state/requests.log`, and with `$CRASH_AFTER_S` it exits that long after it starts |
| `problems/<id>/answer` | each problem's answer, which the node service binds staged (copied to node scratch once) |
| `tasks/sum-a`, `tasks/sum-b` | two tasks: the agents write the answer in `/app/answer.txt`, and the tests submit it through the scope, writing `{"invalid": 1}` when the judge cannot grade it, never 0 |
| `overlays/judge-pool` | the overlay that serves `judge` from the node service |
