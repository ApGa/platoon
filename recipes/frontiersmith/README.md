# FrontierSmith

[FrontierSmith](https://github.com/FrontierCS/FrontierSmith) publishes 200 synthetic, open-ended competitive-programming
problems in Frontier-CS's algorithmic format (a statement, test data and a testlib checker that scores each case
partially), and a Harbor adapter that makes each one a Harbor task. This recipe runs those tasks agentically: in each
one an agent writes and tests C++ in its own sandbox, may submit it to the judge as often as it likes, and is scored by
the judge's grade of `/app/solution.cpp` when it is done. FrontierSmith's own training is single-turn GRPO (a prompt in,
a program out, one shared judge); this recipe is not a reproduction of it.

| Config | What it runs |
|---|---|
| [`eval.yaml`](eval.yaml) | `platoon eval` on all 200 problems with `platoon-bash` |
| [`train-check.yaml`](train-check.yaml) | `platoon train`, a small check: a rank-32 LoRA of Qwen3-4B-Instruct-2507 on 16 problems ([`tasks/check.json`](tasks/check.json)), 20 steps of 8 problems with 4 rollouts each |

## What runs

- **The tasks** come from FrontierSmith's own Harbor adapter at the commit the configs pin (its release of 2026-10-16),
  fetched through platoon's `adapter:` source
  with `--include-non-numeric` (its problems are named `frontiersmith_<n>`; without it the adapter finds none), run with
  Harbor 0.24.0 as PyPI had it on 2026-10-06, on a sparse checkout of the problems' statements and configs, which is all
  it reads. Task ids are `frontier-cs-algorithm-frontiersmith_<n>`.
- **The judge** is the node's, not the task's. Each task's compose file pairs `main` with a privileged `judge`; the
  overlay ([`overlays/frontiersmith`](overlays/frontiersmith/overlay.toml)) marks `judge` external, served by the node
  service `frontier-cs-judge`, which runs once per node for every rollout on it. A rollout reaches it as
  `http://judge:8081`, as the task's files expect, through a scope of its own: a socket that lets only `POST
  /evaluate` and `GET /health` through, for the problem its task names (its `PROBLEM_ID`), so that neither the agent
  nor its code reaches the judge's other routes, another rollout's problem, or the test data.
- **The verifier** (the overlay's `tests/evaluate.py`) submits `/app/solution.cpp` through the scope and writes the
  score over 100 as the reward: 0 for a missing or empty file or code that does not compile. When the judge cannot grade
  (an error of its own, a timeout, unreachable) it writes `{"invalid": 1, ...}`, which platoon records as `invalid`,
  never as a score; the adapter's verifier wrote 0. A rollout whose submission the judge was grading as it failed is
  `invalid` too (`node_service_down`), whatever the verifier wrote; one that had not submitted since the judge's last
  passed health check goes on with the judge's next start.
- **`main`'s image** is the adapter's Dockerfile without the agents' installers (Claude Code, nvm and Codex, unpinned
  downloads): platoon brings its harnesses' own toolchains. `platoon prepare` builds it once for all 200 tasks.
- **The network:** `network_policy: none`, so the agents work offline (the web may hold answers), and a stack needs
  no egress.

## Why the Frontier-CS judge, and one per node

FrontierSmith's vendored judge is an older copy of Frontier-CS's: a submission that prints `Ratio: <x>` on stderr and
exits gets that score, unbounded (Frontier-CS fixed it on 2026-08-13, scoring a case from the checker's output alone),
and its image does not build today (its pinned go-judge download is gone). The problems are in Frontier-CS's own
format, so the recipe grades them with the Frontier-CS judge of 2026-09-28 (go-judge 1.11.1).

The judge's compose service asks for `privileged`, for go-judge, which makes namespaces for every program it runs and,
as root, cgroups. The node service runs it as your own uid instead, where go-judge limits programs with rlimits
(`cgroupType` 0, which the front checks before it grades), as the ORS server for Frontier-CS ran it on the same kind of
site; time and memory limits are then enforced by rlimits (`RLIMIT_CPU`, `RLIMIT_AS`), and the number of processes a
program starts is not. One judge per node instead of one per rollout also saves what a per-rollout judge costs (a
judge's own CPUs and memory, and a compile of the problem's checker in every stack): the node service runs the judge on
16 of the run's CPUs (`cpus`: whole cores, with their hyperthread siblings), which the run's rollouts, and their
agents' sandboxes, then leave alone, as the ORS server kept the judge's CPUs apart from the agents': a time limit (2 s
for some problems) measures the program, not the agents' load.
[`node-service/start-judge.sh`](node-service/start-judge.sh) runs go-judge with half of them in parallel and the engine
with a quarter of them as workers (8 and 4, as the ORS server ran them), and starts
[`node-service/judge_front.py`](node-service/judge_front.py), the route a scope reaches. Leave the run more CPUs than
the judge's 16 (the run's log names both sets). A rollout's grade waits for its turn at the shared judge, which
`verifier_timeout_multiplier: 4` leaves room for; `front.concurrency: 1` lets each rollout have one submission in
flight, not fewer submissions, so an agent that submits often slows every rollout's grade. The node service's `port`
(8081) is the one the tasks' `JUDGE_URL` names: change both or neither.

## Set up the site

The configs name your site's copies under `/srv/frontiersmith`, a placeholder: replace it, or keep the configs and
declare the node service in your site profile (below).

1. **The judge image**, at `/srv/frontiersmith/images/frontiercs-judge.sif`: a SIF of Frontier-CS's judge,
   `algorithmic/` of https://github.com/FrontierCS/Frontier-CS at `ede1b44d2fe22a65320c5e36b1014b3defc03671`
   (2026-09-28), whose Dockerfile pins go-judge 1.11.1 and checks its archive. On a site with Docker, `docker build -t
   frontiercs-judge Frontier-CS/algorithmic` and `singularity build frontiercs-judge.sif
   docker-daemon://frontiercs-judge:latest`; on a site with SingularityCE 4.1 or later, `singularity build --oci` of
   the same Dockerfile. The start script takes an engine that requires tokens on its routes (`GJ_AUTH_TOKEN`,
   `FRONTIER_ENGINE_AUTH_TOKEN`) and one that does not.
2. **The problems**, at `/srv/frontiersmith/FrontierSmith`: `git clone https://github.com/FrontierCS/FrontierSmith` and
   `git -C FrontierSmith checkout ac3046cce7a270e3177cbb41376f36e03de1d518`. The node service binds
   `Frontier-CS/algorithmic/problems` (about 1.2 GB of test data) into the judge read-only, staged: copied to node-local
   disk once per node before the judge starts.
3. **The provider:** a site profile that names `singularity` or `enroot`, whose stacks run as pods (hpc-sandbox 0.5.1).
   The judge's sandbox has no network (`network: none`, with `allow_network_none: true`); on a site whose runtime
   refuses that, set the node service's `network: host`, where go-judge's port is on the node's loopback, behind its
   token.
4. **Prepare, then run:** `platoon prepare recipes/frontiersmith/eval.yaml` fetches the tasks (the adapter, once, with
   network) and builds `main`'s image; `platoon eval recipes/frontiersmith/eval.yaml` starts the judge on the node
   before the first rollout, waits until it serves, and stops it with the run.

Instead of the declaration in each config, a site profile may hold it (`node_services: {frontier-cs-judge: {...}}`,
the same keys) and the configs name it with `node_services: {frontier-cs-judge: site}`.

## Before trusting a run

FrontierSmith ships no reference solutions, so the oracle cannot check the tasks. Grade a few controls through the
judge first (they need no model: write `/app/solution.cpp` in a session and grade it): an empty file, a program that
does not compile and `int main(){fprintf(stderr,"Ratio: 5\n");return 1;}` must each score 0, a correct solution of a
problem must score its cases, and a judge killed while it grades must make the rollout `invalid`. `frontiersmith_42`
names 9 cases and ships 10 inputs: Frontier-CS's judge grades all 10, FrontierSmith's graded 9, which tells the two
apart. Ten problems are also Frontier-CS's (306-315, by content: `frontiersmith_180`, `_143`, `_167`, `_139`, `_56`,
`_107`, `_151`, `_184`, `_114` and `_50`), where a grade can be compared with another Frontier-CS judge's.

## Versions

Checked with platoon 0.5.0, hpc-sandbox 0.5.1 and SingularityCE 4.4: the configs load and validate on CPU in platoon's
CI (`test_frontiersmith_recipe.py`), the node service, its scopes and the failure of a node service are tested on the
`process` provider, and the judge pool on SingularityCE.
