# RFC 0022: Programs that run rollouts, and online environment generation

| | |
|---|---|
| Status | Draft |
| Created | 2026-11-07 |
| Authors | Apurva Gandhi (@ApGa) |
| Builds on | [RFC 0021](0021-trainers-credit-and-policies.md) (credit estimators, policies and roles, step hooks, draw provenance; its rules hold here), [RFC 0020](0020-rollout-permits.md) (rollout permits and the start limit; its section 8's `weight`, built here), [RFC 0016](0016-component-configuration.md) (program options, built here), [RFC 0015](0015-hosted-models-and-judges-in-eval.md) (hosted models, amended here), [RFC 0013](0013-extension-model.md) (components by name), [RFC 0017](0017-public-surface-and-conformance.md) (`platoon.experimental`, conformance kits), [RFC 0019](0019-training-mixtures.md) (per-entry settings, the dispatcher), [RFC 0010](0010-harbor-registry-and-multi-container-tasks.md) (Harbor tasks, root verifiers) |
| Discussion | The pull request that adds this RFC |
| Implementation | Built, in open pull requests: M0 in "fix: Refuse judge agents minted by sandboxed harnesses"; M1 in "feat: Add the running credit estimator", and M2, stacked on it, in "feat: Add the roles credit composite"; M3 in "feat: Add a root role per environment entry", and M4, stacked on it, in "feat: Add program options"; M5 in "feat: Add runtime Harbor tasks", M6, stacked on it, in "feat: Add regrades of generated graders", and M5b, stacked on M6's, in "feat: Add replay checks to runtime tasks"; M7 in "feat: Add a shared-service check for adapters". Built, each as one commit on the one before, whose pull requests open once those above merge: M8 "feat: Reserve rollout permits for episodes with several sessions", M9 "feat: Run rollouts of other entries from programs", M10 "feat: Run CLI harness children under programs", M11 "feat: Add the envgen program" and M12 "feat: Add the envgen archive". Not yet: M12b and M13 to M17. The sections of built milestones describe the code |

## Summary

An episode program runs one root agent on one fetched task and grades one session. This RFC lets a program run other
rollouts inside its episode, and adds what such episodes need. Each piece is generic; the first program built on
them, in `platoon.experimental`, trains an environment generator together with the solver on one policy.

- **Child rollouts and inspections.** `ep.rollout(task, entry=)` runs a rollout of another entry's route exactly as a
  root rollout of that entry, as a child of the episode's root whose rows train in the episode's group;
  `ep.inspect(...)` runs a sandboxed rollout that never trains: a control, a judge, a probe.
- **Runtime tasks.** `ep.materialize(source, entry=)` turns a directory into a checked, content-addressed Harbor task
  with the entry's fixed `task.toml`, an allowed prebuilt image, a separate verifier and platoon's launcher, the only
  one that grades such a task: before the grader runs, it replays the submission, also on the producer's variants of
  its inputs, and scans its sources for answers written as literals; and it grades a generated grader's scores again.
  It refuses what breaks the rules and never rewrites it; a producer ships data, never a launcher.
- **Permits for trees.** `workflow_config.rollout_permits: W` reserves permits for an episode's sandboxed sessions and
  keeps at most `W` open; from M13, children take permits from the pool as their sessions open.
- **Credit per role, in one call.** `credit: running`, a REINFORCE++-style running baseline for groups of one, and
  `credit: roles`, which gives each role of a tree its own estimator and its own share of the advantage mass.
- **Smaller pieces.** `environments[].agents.root_role` binds an entry's root to a role, and so to a policy;
  `program_options` builds RFC 0016's accepted design; a frozen policy served by a hosted model may judge in training
  (section 7); CLI harnesses run as children, then as roots; drains and a step-stall watchdog guard long episodes.
- **The envgen program** (`platoon-envgen`, experimental). A *production* writes a task from a seed, packages it,
  gates it with controls, solves it K times with the solver entry's route and audits a pre-committed sample of the
  solves with frozen judges; a pure reward scores the evidence. The K solves that price the task are the solver's
  training group on it, and they train only once the audits clear it.
- **No new trainer.** The stock `rl` trainer and its step run all of it. A config that sets none of the new keys
  writes today's bytes, and the golden trace gates every pull request; section 11 lists the few behaviour changes
  that need no key, M0's refusal first.

*A generated task is measured where it is made:* one episode tree holds the generator, the gates, the K solver
rollouts that both price the task and train the solver, and the audits, and nothing in it trains until all is known.

```text
draw (a seed)       entry envgen: group_size 1, rollout_permits W, program platoon-envgen
└─ episode: trusted program code in one rollout worker, a budget per stage         rows that train at settle
   ├─ generate   memory from the archive ─▶ root agent, role generator             the generator's
   ├─ package    collect ─▶ toolkit export ─▶ materialize ─▶ <store>/<digest>; the root's session closes
   ├─ gate       the toolkit's final check (host); ep.inspect × 6 controls (oracle × 3, no-op × 3)
   ├─ solve      ep.rollout × K of the solver entry, graded by platoon's launcher   the solvers', unless vetoed
   ├─ audit      ep.inspect × 3 frozen judges on 2 failures and 1 pass, drawn after all K land
   └─ score      attribution, canaries and one retry ─▶ ProductionEvidence ─▶ a reward ─▶ ProductionScore
settle ─▶ credit: roles, once    generator ← running, read at the version it sampled; share 0.10
                                 solver    ← keyed on [role], leave-one-out
rl step (unchanged) ─▶ platoon-envgen-archive on the draws each step committed ─▶ memory, alarms, stops
```

## Motivation

| | Scenario | What stops it today |
|---|---|---|
| S1 | **Training an environment generator together with the solver.** The policy writes a task from a seed (a paper, a repository, a document); K solver rollouts on it measure how learnable it is, which pays the generator, and the same rollouts train the solver. [AutoEnvScaling](https://arxiv.org/abs/2610.14163) has an agentic proposer write Harbor tasks in a sandbox; [SPADE](https://arxiv.org/abs/2608.19197) has a designer write executable environments, trained with a delayed, importance-weighted update; in AutoEnvScaling's self-improvement experiment and in SPADE, one policy plays both roles | a program runs one root agent, once, with a function harness (`Episode.run` in `platoon/programs.py`); a task is a row of a fetched, locked dataset; nothing counts the sandboxes an episode opens beside its own; an entry has one estimator for every role |
| S2 | **Verifiers by rollouts:** a program grades a candidate with rollouts of a checker entry | no program can run another entry's route |
| S3 | **Recursive CLI agents:** a root whose children are CLI harnesses in sandboxes of their own | `Episode.run` refuses TOML harness specs, and TOML entries refuse `agents` settings (`platoon/train/auto.py`) |
| S4 | **[PAIRED](https://arxiv.org/abs/2012.02096)-style environment design:** an adversary, a protagonist and an antagonist, each a role with its own credit | one estimator per entry; a root always samples the default policy (`ModelGateway._policy_of`) |
| S5 | **[Absolute Zero](https://arxiv.org/abs/2505.03335)-style proposing and solving** on one policy, one proposal per prompt | a group of one trains nothing under group baselines, as the config check warns (`platoon/train/entry_settings.py`) |
| S6 | today's runs | they must not change at all |

**The step is not what is missing.** The `rl` step trains asynchronous groups from many entries, each with its own
credit and staleness bound (RFC 0019, RFC 0021). What is missing sits in the episode: running another entry's rollouts,
making a task from a directory, counting a tree's sessions, and crediting its roles apart in one call. RFC 0021 defers
"a framework for producers of online auxiliary data" until a second user, and this RFC adds none: four primitives any
program can use (child rollouts, runtime tasks, permits for trees, per-role credit), their plumbing and one program.

**Why the K solves train the solver.** Prior work trains the solver on the rollouts that price a generated task: SPADE
on the no-hint plays that score its designer, self-play such as [SSP](https://arxiv.org/abs/2510.18821) on the solves
that reward its proposer; AutoEnvScaling's released code trains on fresh rollouts and leaves its calibration and review
rollouts unused. Reuse needs rollouts of the solver's own route (its harness, budgets and grading), which no program can
run. **The generator keeps training:** without proposer training, AutoEnvScaling's validity fell from 86.8% to 22.4%.

## Proposal

The sketches are Python 3.12, and paths are today's. For the built milestones (M0 to M12 and M5b) the sketches
give the names that were built and the text says what the code does, with the reasons for its choices. Every new name
but RFC 0016's program options (section 6.2), which are stable, is experimental at first (RFC 0017): new types live in
`platoon.experimental`, and new members of existing types are documented as experimental, as `EpisodeContext.draw`
is. Every behaviour sits behind a new key but the few that section 11 lists, M0's refusal (section 2.5) first. EST
marks an estimate from budgets, not a measurement.

### 1. Rules

RFC 0021's rules hold: one agent, one policy, for life; credit once, then route; generic code never imports a
trainer, a workflow or a credit estimator (`tests/train/test_layers.py`); an extension is a Protocol plus an optional
base class; components resolve by name in every process (RFC 0013); options sit next to their selector (RFC 0016);
and a config that sets none of the new keys writes today's bytes. This RFC adds five:

- **R1. A child runs exactly as a root of its entry and trains in its episode's group;** an inspection never trains.
- **R2. No agent mints a role its program reserves but its own,** and code the policy runs never mints a judge.
- **R3. An episode keeps at most `rollout_permits` sessions open,** and the horizon counts it as one group.
- **R4. Runtime tasks run only through `materialize`,** which refuses and never rewrites.
- **R5. The episode measures; a pure function of its evidence scores it,** never the best-effort per-task statistics
  (`platoon/train/task_records.py`); one estimator credits the whole tree once.

### 2. Child rollouts and inspections

```python
# platoon/programs.py: EpisodeContext gains, experimental at first but options
options: OptionsT                                  # program_options, parsed: stable, built in M4 (section 6.2)

async def rollout(self, task: Task, *, entry: str, keep_artifacts: bool = False) -> ChildRollout: ...
async def inspect(self, task: Task, *, entry: str, harness: str, role: str,  # entry: whose service opens it
                  options: Mapping[str, Any] | None = None, goal: str | None = None,
                  max_steps: int | None = None, timeout_s: float | None = None,
                  grade: bool = True, keep_artifacts: bool = False) -> ChildRollout: ...
def drop(self, child: ChildRollout, *, censor: bool = False) -> None: ...     # section 2.1
def rescore(self, child: ChildRollout, reward: float, *, reason: str) -> None: ...  # section 2.1 (M11)
def child_entry(self, entry: str) -> ChildEntry: ...  # its route, rollout config and group (M11)
async def materialize(self, source: Path, *, entry: str,                       # section 3: the entry service's
                      grader_owner: Literal["generated", "trusted"] = "generated",
                      replay_checks: ReplayChecks | None = None,
                      replay_timeout_s: float | None = None,
                      replay_address_space_bytes: int | None = None) -> MaterializedTask: ...

@dataclass(frozen=True)
class ChildRollout:
    agent_id: str
    trajectory_id: str | None                      # None for a judge, which leaves no trajectory
    entry: str
    role: str
    outcome: Outcome | None                        # its own grade, with its failure kind and metadata
    result: AgentResult | HarnessResult            # a function harness's, or a CLI harness's once it ran
    final_message: str | None                      # what a judge hands the program
    artifacts: Path | None                         # with keep_artifacts
    versions: Mapping[PolicyId, tuple[int, int]]   # per policy, the oldest and newest version it sampled
    seconds: float                                 # running: from its seat and start slot to its grade
    wait_seconds: float                            # waiting for a session seat and a start slot (from M13, a permit)
    gateway_errors: int = 0                        # its root's calls the gateway failed: a sampler's fault (M11)
```

**2.1 Rollouts.** `ep.rollout(task, entry=e)` runs what a root rollout of entry `e` on `task` runs: `e`'s root role and
that role's policy (section 6.1), its harness, the task as `e`'s service resolves it, its own rollout config (sampling
defaults and precedence, `max_steps`, `timeout`, judges), its `agents` settings, its grading and failure rules, session
needs, call cap and tally, a limit on minted agents of its own, the finalizers its root runs (a function harness's,
under the `single` program; a CLI harness's root runs none, section 2.6), and a start slot. The child takes a session
seat (section 4.1), never raises into the program (a failure is its `outcome`), and is cancelled with the call; the
program's own mistakes raise, such as an entry that is none of its child entries. With `keep_artifacts=True` it keeps
what its session collected for grading, where the session can (`ChildRollout.artifacts`, through an optional session
method): a Harbor session keeps what it handed its separate verifier, whose report stays in the outcome's metadata.
- **Its rows train in the episode's group:** the parent entry's estimator credits them with the rest of the tree, its
  staleness bound filters them (`platoon/train/step/staleness.py`), and they count under the parent draw's label,
  horizon slot and statistics. Child rows are young, so a looser bound rarely matters; a step metric of their ages is
  not built (section 12). Depth weighting and subagent datum sampling see a child's agents at their depth in the
  episode's tree, the child's root at 1, while its `max_depth` counts from its root.
- **Its subtree trains as a unit.** An agent minted or spawned under the child with no outcome of its own takes the
  child's outcome, as it would the root's were the child a root. `ep.drop(child)` excludes the child and every agent
  under it from policy training, its reward kept; `censor=True` also rewrites the child's outcome to `invalid`, so its
  reward enters no baseline. `ep.rescore(child, reward, reason=)` (M11) rewrites the outcome to that score, on which its
  rows train and which enters its siblings' baselines, and keeps what it replaced in the child's `child_rollout` record.
  Both refuse an inspection and raise once the episode's trajectories are built. Nothing of a child trains before its
  program has settled it: the children of a program that raises, or whose finalizer raises, are dropped and censored, a
  child still running when the program returns is cancelled and fails closed, and one asked for after the program
  returned is refused.
- `environments[].child_entries: [e, ...]` names the entries whose routes the program may run, which the parent's route
  then carries; only then does the worker request carry their rollout configs and conversion policies, which routes lack
  (`platoon/train/components.py`), and from M11 their groups' `group_size` and `min_successful_group_size`
  (`ep.child_entry(e)`). A rollout whose request would convert a child entry's rows otherwise refuses that entry's
  children, as their outcome (`child_conversion`).
- **Loading refuses** `child_entries` without `rollout_permits` of 2 or more (section 4), in training and in evaluation;
  a name that is no other entry of the config; a child entry with a program or children of its own (depth 1); a parent
  or child that `harness_rollout` does not run (without a harness or a service, or with a rollout function of its own),
  or a parent whose harness is a TOML spec (until M17, section 2.6); child entries in a named environment; and
  conversion settings (`filter_errors`, `filter_message_actions`, subagent datum sampling) unlike the child entry's, in
  training or in evaluation, since the parent's convert the child rows (`platoon/train/workflow/group.py`). `platoon
  eval` and `platoon rollout` refuse entries with child entries, and so does `platoon config validate` of an evaluation
  config: they reserve no permits, so their episodes would have no seats; such an entry is evaluated during training
  (`eval.every`, section 12).

**2.2 Inspections.** `ep.inspect` runs a harness and a role the program names, in a session of `entry`'s service (a
child entry's, or the episode's own entry's), and never trains: it and every agent minted under it are excluded from
training (`exclude_from_training`), as judge subtrees already are (`platoon/gateway/trace.py`; the converter honours the
exclusion for children). The role is one the program reserves (section 2.5), or platoon's `judge` role, which leaves no
trajectory: the trace folds its calls into the root's `judge_calls`, and a judge asked for while the run's judges are
off is refused, as its outcome. Its options default to its entry's for its entry's harness, its goal to its task's, and
its step budget and time limit to its entry's rollout config's; one left ungraded (`grade=False`), with every agent
under it, has no outcome. Controls (an oracle, a no-op), judges and probes are inspections, of a function harness or a
CLI harness (section 2.6).

**2.3 Start slots.** A rollout nested in another takes no start slot today (`_member_start_slot` in
`platoon/train/workflow/members.py`), so K sandboxes starting at once would bypass `train.max_starting_rollouts`.
Children take start slots: a child takes its session seat, then its permit (from M13), then its start slot, so it
never waits for a seat or a permit while it holds a slot. Two release causes join `first_call`, `end`, `timer` and
`cancelled` (`platoon/train/workflow/start_limit.py`), for children in runs that set `child_entries`, the only runs
whose metrics gain `rollout_client/start_limit/releases/{ready,child}`:
- `ready`: a child whose harness calls no model (`@pt.harness(..., calls_model=False)`, as `platoon-oracle` and
  `platoon-noop` declare) returns its slot once its session has opened, where a root holds it to its end or the
  slot's timer, as a model-free harness still does everywhere else.
- `child`: `ep.rollout` and `ep.inspect` first give back the caller's slot, if any: a slot holder waits for nothing.

**2.4 The grant channel.** Under `rollout_isolation: process` the start limit and the permit pool live in the
trainer, so a worker asks for, receives and returns start slots (and, from M13, permits) over one bidirectional
channel, in `platoon/train/workers/` with the limit passed in, served by a task on the trainer's loop; a worker's exit
releases what it held. The channel is a Unix socket that a file beside the worker's request names, with a token: only
the connection that shows the token first is served, and the trainer refuses a worker that holds or waits for more child
slots than its episode has seats. The channel and its grant-latency metric
(`rollout_client/start_limit/grant_latency_p50` and `_max`: a grant's round trip less the trainer's wait for a free
slot) exist only with `child_entries`; other runs keep the first-call marker (`platoon/train/workers/protocol.py`). If
grant latency binds, a thread that decides grants follows.

**2.5 Reserved roles (R2).** Today any key holder may mint any role (`ModelGateway.mint_subagent`), and the child
samples its role's policy for life. Two refusals answer 403, and only M0's needs no key:
- **M0, built in a fix of its own:** `mint_subagent` refuses the `judge` role from a `policy_only` parent, an agent
  whose key the policy's own code can read (a harness in a task sandbox) and whose `judge` calls are refused already,
  with `CallKindRefusedError` (403, `permission_error`), whatever `inference.judges` says, and before the mint limit
  counts the request; agents minted under such a harness inherit `policy_only`. Without it, such a harness could mint
  a judge child and sample through it, in calls that never train, the frozen judge of a run that binds `judge` to one,
  or else the trained policy. The refusal is no refused judge call (`judges_refused`), so an evaluation with its judges
  off goes on. A sandboxed harness still mints every other role, `verifier` included. A `verifier` it mints samples
  `agents.roles.verifier`'s policy (where the run binds a frozen verifier, in calls that never train), and an
  evaluation counts it with the judges: whether M0's refusal should take `verifier` too is open question 8.
- **Under a program that declares `mints=` (M9):** the program reserves its root role, the roles it opens itself
  (`@pt.program(mints=("control", "judge"))`) and its child entries' root roles, and `mint_subagent` refuses a
  reserved role to every agent of another role, before the mint limit counts the request, as an inline
  `ctx.agents.spawn` of one fails to launch; else a solver child could mint a `generator` agent, which samples the
  generator's policy and earns generator credit. A program without `mints=` reserves nothing. `mints=` refuses
  `verifier`, the role of the agents platoon opens to verify children, so no program reserves it, and a sandboxed
  harness still mints one (open question 8). Red-team cases: a solver child's `POST /v1/platoon/agents` for
  `judge` or `generator` gets 403, and a subagent it mints trains on its outcome.

**2.6 CLI harnesses under programs (M10).** A child entry's harness, and an inspection's, may be a TOML spec, which runs
as a TOML spec's root runs it, through the same code: `HarnessSetup.prepare` and `_run_and_grade` with its
`SessionNeeds` (`platoon/harness/rollout.py`), so its placement, its call kinds in a task sandbox, its call cap, its
time limit and its failure rules are a root's. Its `result` is the harness's `HarnessResult`, now experimental public
API, once the harness ran, else an error `AgentResult` that says why (a session that did not open, a toolchain that is
not built). Like a TOML spec's root, it runs no finalizer of its entry's; the program's own finalizers run over the
whole tree. A child whose session or gateway could not be set up, where a root would leave no trajectory, keeps one,
marked `child_rollout.failed_to_open` with its task.
- **A gateway server of its own.** Each CLI child reaches a server of the episode's gateway that takes only its
  subtree's keys (401 for any other), on loopback and on the socket its sandbox's route reaches, so a sandbox without a
  network reaches no other server; the episode's own server still takes every key of the rollout. Once the harness has
  run, graded, failed or cancelled, the subtree's keys end on every server (410, `ModelGateway.end_unit`) and the
  child's server stops, before its artifacts are kept and its session closes, as a root's agents close once it is
  graded: a process the harness left behind calls in vain.
- **CLI inspections.** `goal` is the spec's instruction, `options` its options, which the spec checks before anything
  opens (options it refuses raise into the program), and `max_steps`, `timeout_s` and `grade` act as for a function
  harness, `timeout_s` under the spec's own limit. A `judge` runs a CLI harness only through an entry placed `local`,
  never in the sandbox of the task it judges, where the code it judges could read its key and write its verdict
  (`ep.inspect` raises `ValueError`).

CLI roots follow (M17), lifting the refusals in `Episode.run` and `platoon/train/auto.py`: until then loading refuses a
parent entry whose harness is a TOML spec, and only children run CLI harnesses.

**2.7 Datum parity,** the acceptance test of child rollouts (M10). Under a scripted model (recorded completions and
logprobs; the fake records each request's sampling parameters and token limit, and the gateway each call's cap, since a
sampler sees no agent), with the clock the agent reads pinned (`SOURCE_DATE_EPOCH`, whose date the test finds in the
prompts), the seeds of the harness draw and of the recording, and one image, a task run as a root of the solver entry
and as an `ep.rollout` child yields equal datums (input and target tokens, masks, logprobs, the advantages before
credit, `checkpoint_version` and each datum's own version, routing handles numbered by first appearance) and equal
sampling (parameters, token limit and call cap, by role, without ids). It does so for a CLI agent of two specs drawn per
task and for function harnesses, each minting a subagent, with the solver entry's subtree finalizers off and with each
on, and with depth weighting and subagent datum sampling off, their defaults, which see a child's agents at their depth
in the episode's tree (section 2.1). Live runs cannot be byte-identical (sampling varies, and some harnesses write the
date into their prompts), so a GPU smoke, `scripts/child_parity.py`, compares 64 tasks run as roots and as children on
turns, completion tokens, reward and failure kinds, with permutation tests within each policy version, Holm-adjusted.
`platoon eval` runs no children (section 2.1), so the smoke trains at a learning rate of 0, beside the entry to check an
entry whose program runs each task as its child, with a deadline that covers its child's whole run. Children that failed
to open are left out and counted; a rollout that raises in its worker leaves no record for the script, and the run's
logs count those. The smoke needs a GPU service and has not run.

### 3. Runtime tasks

The service's half is built: `HarborEnvironmentService.materialize`, its config, the store and the launcher (M5),
regrades (M6) and replay checks (M5b). `ep.materialize(source, entry=)` reaches the child entry's service, so it ships
with child rollouts (M9); it routes the task to that entry, and a replay setting that the program leaves None takes the
service's default.

```python
@dataclass(frozen=True)
class RuntimeTasksConfig:                      # service_kwargs.runtime_tasks of a Harbor entry
    images: Mapping[str, str]                  # the docker_image names a task may give -> prebuilt images
    template: Mapping[str, Any]                # every runtime task's task.toml but its [task] name and artifacts
    verifier_user: Literal["refuse", "ignore", "honor"] = "honor"   # replaces service_kwargs.verifier_user for them
    verifier_provider_options: Mapping[str, Any] = field(default_factory=dict)   # laid over the entry's
    regrade: int = 1                           # extra grades of a generated grader's scores (section 3.3)
    max_bytes: int = 256 * 2**20
    store: str | None = None                   # None: runtime-tasks in the run's directory, else in the Harbor cache

@dataclass(frozen=True)
class MaterializedTask:                        # what materialize returns; .ok when problems is empty
    task: Task | None                          # its id is the digest; misc.harbor.task_dir is <store>/<digest>
    digest: str | None
    problems: tuple[str, ...]                  # why not, one sentence each: for the evidence and the memory

@dataclass(frozen=True)
class ReplayChecks:                            # section 3.4; their data lie in the task's tests/gen
    variants: Sequence[ReplayVariant] = ()     # each {name, inputs, expected, tolerance, meta}
    forbidden_literals: ForbiddenLiterals | None = None   # {path, sources, scan}

class HarborEnvironmentService:
    async def materialize(self, source: str | os.PathLike[str], *,
                          grader_owner: Literal["generated", "trusted"] = "generated",
                          replay_checks: ReplayChecks | Mapping[str, Any] | None = None,
                          replay_timeout_s: float | None = None,
                          replay_address_space_bytes: int | None = None) -> MaterializedTask: ...
```

**3.1 `materialize`** copies the directory's regular files and directories into a staging directory of the store,
checks the copy, never the source, which could change, and loads it with `HarborTask.load`
(`platoon/envs/harbor/task.py`). It refuses, one sentence per problem: links, devices, FIFOs and sockets; more than
`max_bytes` or 10,000 entries; a `task.toml` that differs from the template anywhere but its `[task] name` and
`artifacts`, each difference named, so that a network, GPUs, another image or shared-mode grading never gets in; an
artifact that is no normalized path below the working directory (at most 256, each of at most 1,024 characters), since
the verifier puts each artifact back at its path, where an absolute one could land beside the grader; an
`environment/Dockerfile` and an `environment/docker-compose.yaml`; anything in `tests/` but `gen/`, the task's grader,
so that `tests/test.sh` is always platoon's; and a missing or malformed `tests/gen/grader.json`. The service refuses,
when it is built, a template that gives the agents or the verifier a network, grades in the agents' sandbox, runs the
verifier on an image of its own (which would bring its own `/tests`) or names none of `images`. Then `materialize`
installs the launcher with its settings (`tests/platoon/launch.json`: the grader's owner, the reward's name, the
verifier's time limit, the artifacts, the grader's command and the replay, the variables the replay keeps, the replay
checks and time, a fingerprint of the `runtime_tasks` it checked against, and the digest of the launcher it installed),
and renames the copy to `<store>/<digest>`, a SHA-256 of every file with its executable bit, so that the same directory
twice, or from two producers at once, is one task; the store also holds the files of the tasks in flight, for a
producer's final check to read, while the producer keeps the list of them (section 9.6). Nothing a generator wrote is
patched: it must learn the rules.
- **What stays the producer's to refuse.** `materialize` refuses no dotfile and no module shadow among the task's own
  files (a `numpy.py`, a `.pth`): a repository's task needs its `.git`, and only the producer knows what its grader
  imports, so the producer's final check refuses those (section 9.7). A submission cannot shadow the grader's modules,
  since artifacts land in the working directory alone and the grader runs in `/tests/gen`, unless the image puts the
  working directory on Python's path, which `check_runtime_tasks`' `module_shadow` case catches for the entry's image.
- **Only its own entry runs it,** or one with the same settings. A runtime task's directory says it is one, by the
  launch settings that only `materialize` writes. An entry opens one only from its own store, and only if the
  fingerprint of the `runtime_tasks` it was checked against (all but `regrade` and `store`), which also records the
  reward's name and `verifier_timeout_multiplier`, is the entry's own, and only if it holds this platoon's launcher (the
  digest its settings record), so entries that share a store and those settings share their tasks, and a task stored by
  another platoon, as before an upgrade, is never graded by the launcher it was given. Another store's task, one with
  another platoon's launcher (materialize it again, under a new digest), another entry's in a shared store whose
  settings differ, one checked before a resumed run changed its template, and any runtime task of an entry without
  `runtime_tasks` are refused when they open or are looked up (`env_config`), never run as ordinary tasks.
- **How it opens.** Its sessions run the image that `images` binds its `docker_image` to, with its `environment/`
  uploaded into the working directory: the prebuilt-image plan that a session otherwise finds only in its dataset's
  lock (`platoon/envs/harbor/locked.py`), computed from the stored files. Its separate verifier starts with
  `runtime_tasks.verifier_provider_options` laid over the entry's, so that it can run as root (`verifier_user: honor`,
  RFC 0010) on a provider that needs options for that (on Singularity, `{rootfs_layout: copy, fakeroot: true}`), and
  `platoon train`, `eval` and `rollout` refuse an entry whose provider options could not run its runtime tasks (exit
  78). The entry's curated tasks keep their own options and are fetched, locked and graded as today: `runtime_tasks`
  adds a way in, not a change.

**3.2 The launcher.** A runtime task's grader is a **trusted grader**, code that the task's producer ships (envgen's
toolkit, section 9.7), or a **generated grader**, which the policy wrote. The program says which (`grader_owner=`), and
`materialize` never takes it from the directory: a generator that could write "trusted" would make its grader's faults
the trusted side's. **One launcher grades every runtime task:** platoon's, which `materialize` installs as the task's
`tests/test.sh`, Harbor's verifier entry point. A producer ships data, never a launcher or a `tests/test.sh`: the
grader's command and the replay are `tests/gen/grader.json`, a data spec of platoon's, and generic platoon code reads
only platoon's own formats (`grader.json`, the replay checks' data), never a producer's manifest, which the program
maps onto `materialize`'s arguments (section 9.7). The launcher (`platoon/envs/harbor/launcher.py`: Python 3.8 or
later and its standard library, run by the verifier image's `python3`) runs as root in the verifier, with a umask that
keeps what it writes root's. It:
1. seals `/tests`, `/logs/verifier` and what the agents published (`/logs/agent`, `/logs/artifacts`), root's and mode
   700, and checks that each seal took: a directory, never a link, open to no other user, in a directory that the
   replay's user cannot replace. A seal it cannot make or that does not take fails closed;
2. checks the declared artifacts against the output contract (regular files and directories, each file at most 64
   MiB, `.json` files that parse, nothing but directories above an output of the replay). For a task whose
   `grader.json` gives a replay (without one, the grader judges the submitted outputs as they are, and a regrade
   compares the artifacts alone), it replays the submission as `grader.json` declares **in place**: in the working
   directory itself, at its own path, without the replay's outputs, which the replay must write again. Code that names
   its files by their absolute paths (`/app/answer.json`), as code a solver writes in the working directory often does,
   then replays as it ran, where in a copy at another path it would write outside the copy and fail, the solver's 0 for
   a submission that does reproduce its outputs. So the launcher lends the working directory to the replay and puts it
   back after it:
   - **before the replay**, it checks that nothing is mounted below the working directory and that it holds no socket or
     device, none of which the launcher could put back, that no directory of it, itself included, is one that the image
     or the template leaves open to the replay's user (by mode bits or access ACL), that no directory above it lets that
     user replace what it holds (none may be that user's, or writable by it without the sticky bit), so that the
     replay cannot move the working directory away, and that the launcher's own directory is not inside it; each fails
     closed, the trusted side's fault, before anything there changes. It then makes everything in the working directory,
     files and directories at any depth, root's and closed to other users' writes, moves the submitted outputs to
     `$PLATOON_SUBMITTED`, and copies the rest, never following a link, into a **pristine copy** in its own directory,
     root's alone, which keeps its directories, files (with their holes), links and FIFOs, a file of several names as
     one file, and their modes and modification times;
   - **the replay** runs in the working directory, lent to its user: the directories and files there become that user's
     to read and write (a file with other names, which may lie outside it, is first replaced with a copy of its own, so
     that nothing beyond the working directory becomes that user's), while the directory itself stays root's, of that
     user's group, so that the replay may change what it holds but can neither move it nor change its mode. It runs for
     at most half the time left (or `replay_timeout_s`, section 3.4), less twice as long as the copy took, which putting
     the directory back takes: as a user of its own (uid 65533, never root, checked) without supplementary groups or new
     privileges, with a home and temporary directory of its own (`HOME` and `TMPDIR`), a thin environment (an explicit
     keep list: the path, cut to its absolute entries so that nothing in the working directory stands in for a program,
     the locale and time zone, Python's and the dynamic linker's search paths and prefixes, numeric libraries' thread
     counts, and the template's variables that the verifier shares) and limits (no core files, no file larger than an
     output may be, 1,024 processes of its user with threads counted, 1,024 open files, an `oom_score_adj` of 1000 so
     that the kernel's choice under memory pressure falls on the replay and never on the launcher, and, where
     `replay_address_space_bytes` sets one, a ceiling on each process's address space). A bounded sweep then ends every
     process below the launcher, which is their subreaper, and every process of the replay's user; one of that user's
     that it cannot signal or end in time is a survivor, never skipped;
   - **after the sweep**, it checks that each output is a regular file of one name that keeps the output contract,
     removes what the replay's user left where every user may write, its System V IPC objects and its home (section
     3.4), and all it left in the working directory but its outputs, never following a link, holds the outputs as root's
     in its own directory, and **puts the working directory back** from the pristine copy, root's again and closed to
     other users' writes. Once the variants, if any, are done (section 3.4), the replayed outputs go in place of the
     submitted ones. Whatever the replay changed, added or removed in the working directory, the grader finds it as the
     pristine copy holds it, with the outputs the replay wrote (the copy keeps neither the extended attributes and ACLs
     of what the directory holds nor other names of its files outside it), and finds the submitted outputs in
     `$PLATOON_SUBMITTED`;
3. runs the grader as root in `/tests/gen` for the rest of the verifier's time limit. It reads the outputs in
   `$PLATOON_WORKDIR`, the working directory (the replayed ones, for a task with a replay), and writes its verdict to
   `$PLATOON_VERDICT`, `{passed, checks: [{id, passed, margin}]}`, and nothing else;
4. kills every process the grader left, as their subreaper, never the sandbox's own; the session then kills what is
   left in the verifier, as it does for every task;
5. writes the only reward file, `{<reward name>: 0|1}` (`reward` by default) or `{"invalid": 1, "failure_kind": ...}`,
   and beside it its report, `launcher.json` (at most 64 KiB), which is the grade.

Every file whose contents the launcher reads (its settings, a JSON artifact, a replayed output, the verdict, a log's
tail) is opened without following a link or waiting on a FIFO (`O_NOFOLLOW | O_NONBLOCK`) and checked on the open
descriptor, so a grader that leaves a FIFO in place of its log cannot hang it. The entry's `verifier_classifiers`, its
reading of reward files that are not regular files and its reward contract's `check` hook apply to curated tasks only,
since a grader that runs as root could write what they read.

| What happened | Outcome |
|---|---|
| the verdict passed, or did not | `scored` 1 or 0 |
| an artifact breaks the output contract; the working directory is larger than a replay copies; the artifacts take longer to check than the time limit allows | `scored` 0, reason `solver_output_contract` |
| the replay fails, outlives its time, lacks its own program (`./run.sh`), leaves a process the sweep cannot signal or end, leaves more than the launcher removes in its time (section 3.4), takes the room the launcher's later steps need, or does not write an output again; copying the working directory aside and back leaves the replay no time, or putting it back outlives the replay's time | `scored` 0, reason `solver_nonreproducible` |
| the grader exits nonzero or cannot start, outlives its time, writes no verdict or one beyond the schema (`invalid` included: a grader cannot censor), or leaves what cannot be killed; a replay command of the image's own that cannot start | `invalid`, `<owner>_exit`, `_timeout`, `_protocol` or `_isolation` |
| the launcher fails once the grader ran, exits nonzero, leaves no report or outlives the verifier's time limit; grading cannot kill what was left or read the verifier's output | `invalid`, `<owner>_unfinished`, whatever the reward file says |
| the launcher cannot do its own part before the grader runs: no `python3`, no root, no replay user, a seal that does not take; a working directory that the image or the template leaves open to the replay's user at any depth, that holds a mount point, a socket, a device or the launcher's own directory, or that lies below a directory from which that user could replace it (by mode bits or ACL), or whose mounts it cannot read (`/proc/self/mountinfo`); outputs it cannot hold as root's, or a working directory it cannot put back, but for a full filesystem (section 3.4) | `invalid`, `verifier_trusted_launcher` |

`<owner>` is `verifier_generated` for a generated grader and `verifier_trusted` for a trusted one; what fails before
the launcher starts (the verifier's sandbox, the artifacts' upload) keeps platoon's kinds, such as `verifier_start`.
The solver's scores of 0 are measurements whose `reason` says why, never failure kinds, whoever wrote the grader. The
launcher is not a security boundary against the grader: both run in the verifier's sandbox, the grader as namespaced
root. **Attribution is the boundary:** a grader can only choose 0 or 1, which it may anyway and which controls and
audits check, and anything else is a non-score (`invalid`) that the program attributes (section 9.2). A
`verifier_generated_unfinished` may also be the sandbox's (out of memory or disk), which a known-good canary tells
apart.

**3.3 Regrades** (`runtime_tasks.regrade`, default 1). Once a generated grader scores a submission, the session grades
the same collected artifacts again `regrade` times, each in a fresh verifier that starts after the one before
stopped, and the score stands only while every regrade gives it again. Each regrade replays the submission, and the
launcher's report holds the SHA-256 of what the replay wrote (`replayed_sha256`), taken before the grader could change
it:
- another score of the same bytes (the same artifacts and replayed outputs) is `invalid`, `verifier_flaky`, the task
  author's fault, and no regrade runs after it;
- another score after a replay that wrote other bytes is the solver's 0, `solver_nonreproducible`: a submission that
  draws from an unseeded generator or reads the clock can get an honest grader's 1 one time and its 0 the next, and
  charging that to the grader would count the solver's code against the author. A flaky grader then goes unseen on
  that submission, and the task's other solves still find it; handing the regrade the first verifier's outputs instead
  would let the first grade's grader, root there, mark them for the regrade;
- a regrade that gives no score of the grader's (the solver's 0, a non-score, an infrastructure fault) makes the grade
  its own outcome, so that the author never pays for a verifier that did not start;
- only a generated grader's scores are regraded, never a trusted grader's or a curated task's, whose grading a regrade
  would change. Deadlines on whole rollouts count each regrade (`rollout_overhead_s`), and `regrade` is no part of the
  fingerprint.

A coin-flip grader survives three oracle and three no-op controls with probability 1/64, and one regrade of 8
artifacts lifts the detection of a 5% flake from 26% to 67%.

**3.4 Replay checks** (`materialize(..., replay_checks=, replay_timeout_s=)`). A replay shows that the submission's
code writes its outputs again, not that it computes them: code that states the answer, or ignores its inputs, writes
it again too. A producer often holds what catches that, perturbed copies of the inputs with a validated reference's
outputs on them, and the answers themselves, so a task with a replay may carry two optional checks. They are the
caller's word, as `grader_owner` is, recorded in the launch settings (the same directory with other checks is another
task), and their data lie in the task's sealed `tests/gen`:
- **Variants**, each `{name, inputs, expected, tolerance, meta}`: `inputs`, a directory whose files are laid over the
  working inputs; `expected`, a JSON file that maps output keys (a `.json` output, or a JSON Pointer into one,
  `answer.json#/rows/2/value`) to numbers or lists of numbers; `tolerance`, absolute, per key with a default; `meta`,
  opaque and recorded, such as the cells a variant covers. Once the replay has reproduced the submission and the
  working directory is back, each variant replays in place, as the replay ran: the launcher lays the variant's inputs
  over the working directory, lends it to the replay's user, runs the replay there, reads the outputs it wrote and
  puts the directory back from the pristine copy before the next replay or the grader. So a submission reads a
  variant's inputs whether it names them by paths relative to the working directory or by their absolute paths there,
  and no replay finds what an earlier one changed or left. Each variant's replay must give what `expected` says within
  tolerance, else the submission scores 0, reason `solver_variant_failed`, as it does where the variant's time leaves
  none for putting the directory back after it, or the put back outlives that time; no variant runs after one that
  fails. What the replay wrote waits meanwhile as root's in the launcher's own directory, where no variant's replay,
  run as the same user, can reach it, and goes back in its place after the last variant. That return is the replay's own
  step: one that ends past the last replay's time and the window after it, or finds no room once the replays left files
  where every user may write or IPC objects, is `solver_nonreproducible`, not `solver_variant_failed`.
- **Forbidden literals**: a JSON file that lists `{value, tolerance}`, and the submission's Python sources to scan (its
  `.py` artifacts, never a replay's output). Before the replay, the launcher parses each with the verifier's `python3`
  by a named rule, `python-numeric-literals/v1` (its int, float and complex constants under their chains of unary
  signs, and its numeric strings and ASCII bytes of any length; never its comments, and nothing evaluated), and a
  number within the tolerance of a forbidden value is a scored 0, reason `solver_hardcoded`. A source that is no UTF-8
  Python breaks the output contract.
- **Both are the solver's faults** (scored 0), never censoring. Check data that the launcher cannot use are the grader
  owner's protocol fault (`<owner>_protocol`), and `materialize` refuses what it can see is unusable: a task without a
  replay, inputs laid over the submission or an output, expected keys of no `.json` output, a source that is no
  artifact. Checks beyond the launcher's caps (32 variants, names of at most 128 characters, 8 KiB of `meta` each, 64
  sources, 256 KiB recorded in all) raise `ValueError`, the caller's mistake, so a producer exports within them. The
  checks run before the grader, which as root could change what they read, and after which any fault of the
  launcher's would be the grader's (`<owner>_unfinished`). So every grade, a regrade's too, pays for the variants'
  replays, and a wrong answer that also misses a variant is `solver_variant_failed`.
- **Every replay leaves nothing behind**, the replay's own too, with or without checks. After each, the launcher removes
  what its user left where every user may write (`/tmp`, `/var/tmp`, `/dev/shm`, `/run/lock`, `/var/lock`,
  `/dev/mqueue`, where the sandbox has them), its System V IPC objects, its home and all it left in the working
  directory, never through a link, and puts the working directory back (section 3.2), within 10 s past its time; a
  replay that leaves more than the launcher removes in that time, or whose put back outlives it, is the solver's 0
  (`solver_nonreproducible`, a variant's `solver_variant_failed`). So is a full filesystem in a later step of the
  launcher's, putting the directory back included, once a replay left files where every user may write or IPC objects,
  so that exhausting the disk never turns the solver's 0 into a fault of the trusted side or the grader's. Where the
  replays left neither, a full filesystem is the trusted side's, as it is before any replay ran; one found as the
  directory is put back is so only once the filesystem has written out what it holds and the launcher has tried once
  more, since some filesystems free what was removed only after a while. `replay_timeout_s` gives each replay, its own
  and each variant's, the time the producer told its solver, copying the working directory aside and putting it back
  included: the replays end within `(1 + variants) × replay_timeout_s`, and the grader keeps the rest (`materialize`
  raises `ValueError` where none is left); without it the replays share half the time left.

Two checks, not one: a perturbation catches a stated answer only where it moves the answer beyond tolerance, so the
scan for the answers themselves stands beside it. The scan reads Python only, by a rule that a producer applies to its
own reference, so that its oracle never fails it; other languages need rules of their own.

**3.5 `platoon.testing.check_runtime_tasks`** runs the generic half of a red-team suite, written from the
environment's own template. Its refusal cases need no sandbox: a network, GPUs, another image and shared grading in
`task.toml`, an artifact at `/tests/gen`, a Dockerfile, a compose file, a `tests/test.sh` of the task's own, a link,
files beyond `max_bytes`, a malformed grader manifest, a variant whose inputs overlay the submission, and the honest
task twice into one digest. Its grading cases run on the environment's sandboxes: right and wrong answers, a missing
artifact, a failing replay, a replay that reaches for the grader's files, the published answer or the reward, a replay
that reads and writes its files by their absolute paths, one that changes its inputs and its code and leaves files
beside its output, after which the grader must find the inputs and the code as they were and none of those files, a
submitted module shadow; graders that declare `invalid`, crash after a verdict of 0, kill the launcher, write a reward
file of their own or leave a process that keeps writing one, or write a verifier classifier's line into the log; a
trusted grader that crashes; a coin-flip grader, graded again up to 20 times until it comes out `verifier_flaky`; and
a variant followed, a variant ignored, a hardcoded literal, a variant's replay that reaches for the replay's outputs
and the grader's files, and the two in-place cases again with a variant. A hash-keyed grader and a planted secret are no
property of `materialize` or the launcher: the program's controls catch the first, and the toolkit's final check, which
knows the secrets, the second (section 9.7). The refusal cases run in CI; the grading cases need verifiers that run as
root, so they run end to end in an opt-in Singularity test with `--fakeroot` verifiers (one of CONTRIBUTING's [tests
that skip without local files](../../CONTRIBUTING.md#tests-that-skip-without-local-files)), which each pull request that
touches `materialize` or the launcher runs and reports in its checks; the minting cases (section 2.5) are CPU tests and
run in CI.

### 4. Permits for trees

A draw reserves its group size of permits, and a rollout that a member runs inside its own takes none (RFC 0020,
section 8), so an episode's children would run outside `train.max_concurrent_rollouts` (P below).

**4.1 Static permits (M8): `workflow_config.rollout_permits: W`.** A draw of an entry whose W is above 1 reserves
`group_size × W` permits whole and inactive (`rolling=False`, whatever `train.permit_release`), granted together and in
turn on today's whole-grant path (`platoon/train/dispatch/permits.py`), and keeps them until its group's workflow
returns: an episode that ends first keeps its W permits idle until its slowest sibling ends, and in process a member
returns only once the teardowns its rollout left running have ended (at most 60 s). Each episode of a function harness
keeps at most W sessions open on W **session seats** of its own, granted first come, first served
(`platoon/harness/seats.py`). The root's session takes one once it has opened, which comes back when the program's close
of that session returns (`HarborSession.close` is idempotent), and a child's holds one from before its start slot until
its own close has returned (M9); a close that raised or was cancelled, the root's or a child's, keeps its seat until the
episode ends, since the sandbox may still be up. So with the root's session closed, W = K runs every child at once, and
a smaller W runs them in waves. `BoundDraw.permits`, internal, carries the reservation, so `group_size` keeps meaning
the credit group (RFC 0020: "do not overload `group_size`"), and `RolloutLease.size` is the reservation of a draw that
runs its group size of rollouts (a clause of RFC 0020's lease rule 2). W reaches the episode in its rollout config's
`extra` (`platoon_rollout_permits`, only above 1), a key of platoon's that a config may not set. This is RFC 0020's
unit-plan `weight`, without the unit plan.
- **No deadlock:** a draw starts only once it holds all its permits; every child keeps one order, its seat and then its
  start slot, and waits only on its episode's seats; and loading refuses P below an entry's `group_size × W`, in
  training and in evaluation, and `child_entries` with W below 2 (M9). M8's randomized fake shows the order is
  load-bearing: a child that takes its slot first, among siblings that take their seat first, can hold the start limit
  until a slot's timer, and children that all take their slot first deadlock nowhere but hold slots while they wait for
  seats.
- **Counts:** an evaluation's group reserves its group size times its entry's W, and its worker count and the client
  metrics' ceiling are computed from permits, not group sizes. In a mixture under the default label column, whose labels
  are its entries, an entry's label cap (`train.rollout_group_limits.<label>`) defaults to ⌊P / (group_size × W)⌋, which
  the raised-bound warning counts and the skip warnings call derived; under another label column, or for a single
  environment, no label names the entry, and none is derived.
- **Evaluation:** an evaluation during training takes an entry's W from training unless its own settings set it;
  `platoon eval` reserves no permits and hands its episodes no seats (section 2.1).

**4.2 Rolling child permits (M13).** Static permits idle while the root works and children run in waves (Appendix A.2).
- **A permit stands for one open session**, not for the rollout's worker and sandboxes: for such episodes this changes
  RFC 0020's lease rule 1 (`platoon/train/leases.py`), and the label cap bounds the workers alive. When the program
  closes the root's session, the root's lease block returns its permit, as `held_slot().first_call()` returns a start
  slot. Children ask over the grant channel as their sessions open, each taking its seat, then its permit, then its
  start slot, and return their permits as they close (M13 lifts the lease's reservation beyond `size`).
- **Children open only after their root's session has closed:** `ep.rollout` and `ep.inspect` raise while it is open.
  A program that needs children beside an open session says so, `@pt.program(children_with_open_root=True)`, and
  keeps M8's static reservation until RFC 0020's lending rule (a parent blocked on a child lends it its permit) lands.
- **No deadlock, no reserve.** (1) Every held permit belongs to an open session: a working root, a child, or a member
  of an ordinary group. (2) No holder waits for a permit: a root's children open only after its session has closed
  and returned its permit, children open no children (loading refuses depth 2), and a child takes its permit before
  its start slot. (3) Child requests are served before any draw's, first come, first served. So the oldest request is
  granted when any open session ends, within one session's deadline.
- **Children never escape the horizon** (R3), which RFC 0020 forbids growing silently. Before M13 lands, RFC 0020's
  trace-driven model (its Appendix A.2) replays the measured latencies through the permit pool under both rules.

**4.3 The horizon and the label cap.** An episode tree holds one horizon slot, of `(k + 1) × B` (k:
`train.max_staleness`, B: `train.batch_size`), for its whole latency L, and its entry's label cap bounds how many run
at once. Two adapters do not escape the horizon (`platoon/train/dispatch/horizons.py`); Appendix A.2 sizes both.

### 5. Credit per role

**5.1 `credit: running` (M1).** [REINFORCE++](https://arxiv.org/abs/2501.03262) gives a sample of one
`A = clip((R − mean) / std)` over a step's batch. Groups of one settle minutes to hours apart, and platoon credits
each group alone, so `running` keeps the three operations over decayed running statistics, read at sampling time.
Each scored agent of a group is a sample, and the agents of each role of each policy have statistics of their own,
aged in that policy's versions: a root and its subagents are never centered on each other's rewards, and the estimator
works unchanged on a view of one role (section 5.2) and in a run of several policies.

```text
stratum    the task's values of options.strata, coarsest first (say [kind, mode, domain]), from CreditGroup.columns;
           a task without a column is credited at the stratum above it, warned of once per column
priors     options.priors, a measurement: a mean, a scale and per-stratum means, each weighing as 16 samples
           (weight) that decay as samples do, at the root and at the strata they name
decay      every statistic × 2^(−1/H) per version of its policy (half_life H = 4; 2H for the scale); a sample
           enters at full weight when its group settles
shrink     b_c = (S_c + 8 · b_parent(c)) / (W_c + 8), S_c and W_c being stratum c's decayed sums of rewards and
           weights, along [kind, mode, domain] → [kind, mode] → [kind] → the root, whose b is S / W (0 while empty)
read       b_i and σ_i as the statistics stood when the policy reached V_i, the oldest version among the agent's
           rows: only what settled at earlier versions
window     the statistics reach back as far as the policy's rows of the entry may be old and still train
           (CreditSetup.max_staleness), from the last settle; an agent that sampled earlier trains at 0, and settles
scale      σ = max(sqrt(decayed mean of (R_j − b_j)²), 0.1), b_j being the baseline sample j was credited against
advantage  z_i = clip((R_i − b_i) / σ_i, −3, 3) on every action token of the agent's rows;
           z_i = min(z_i, 0) when R_i < 0 (cap_penalties: a penalty never pays)
warm-up    without priors, z_i = 0 until the statistics read hold 32 settled samples
settle     the only place state moves; an unscored agent moves nothing, nor does one whose reward is not a finite
           number (warned of once), which would spoil every baseline it entered; a scored agent without rows settles
```

- **Action-independent:** read in `assign` and moved only in `settle`, as [SPO](https://arxiv.org/abs/2509.13232)'s
  pre-update baseline is, and keyed by prompt-side columns only: a key the policy chose would cancel its incentive.
- **Read at the sampling version,** since slow samples settle late and a read at `assign` time under a rising baseline
  would favour outcomes that settle fast. A snapshot never changes once the policy has reached its version, so
  `assign` and `settle` read the same one.
- **A staleness bound is required.** Without one no window covers every row that trains, so `platoon config
  validate` and `train` refuse `running` on an entry whose rows have no bound, its own `max_staleness` or
  `train.max_staleness` (exit 78).
- **The state is a base and a log:** for each role of each policy, the oldest statistics a read may still need and
  every settle since, which `load_state` replays to the bit. For 100 versions of 16 settles over 103 strata that is 19
  kB at a bound of 10, where a copy of the statistics per version would be about 440 kB.
- **Weighted at settle, not by sampling version:** weighting by sampling version over-weights fast failures, which
  settle first; at 30% invalid outputs the tracked mean would read −0.19 against a true −0.02 (EST).
- **A penalty never pays:** on a stratum whose baseline is below −0.5, a −0.5 penalty would otherwise earn positive
  credit (the error filter's sign rule, applied to outcomes). **Strata, shrunk toward their parents:** one global
  baseline caused training failures in [Dr. Zero](https://arxiv.org/abs/2601.07055). The chain ends at the root, not
  at the priors, which decay: a sparse stratum falls back to the run's own mean, not to the measurement it started
  from.
- **Metrics:** `credit/running/advantage/avg` (a mean drifting away from 0 says the baselines lag), `.../baseline/avg`,
  `.../scale/avg`, `.../age/avg`, and the shares of agents in the warm-up, past the window, clipped and capped.

M1 also adds to the credit API (`platoon/train/credit/api.py`) `TrainingSample.version: int | None`, the row's own
sampling version: the minimum of its per-call stamps whatever `train.stale_filter` is, or its rollout's when a call
recorded none (the converter's `own_version` rule, `platoon/utils/tinker_data_processing.py`), and for a row built
without one its datum's `checkpoint_version`, None when the datum carries none (`running` then takes the policy's
version when `assign` began), so credit reads versions without importing the step; `CreditGroup.columns`, the task row's
scalar columns from any service, as labels read them (one test keys on a column that is not Harbor's);
`CreditSetup.max_staleness`, the bound of each policy's rows of the entry (`main`'s always the entry's, another policy's
its own, else the entry's); and `credits_single_members: ClassVar[bool]`, set by an estimator that credits groups of one
(`running`, `roles`), so that the warning that such a group "trains nothing" (`platoon/train/entry_settings.py`) skips
it. The registry says what core's estimators declare in tables, so loading a config still imports no estimator.

**5.2 `credit: roles` (M2).** One entry, one credit call, an estimator per role:

| Piece | What it does |
|---|---|
| a view per role | runs `by_role.<role>.credit`, any estimator an entry may name but `roles`, with its `options`, on `CreditGroup.only(roles=[role])`, a view over the same rows that it also settles, and logs its metrics under `credit/roles/<role>/`; an agent of a role not named gets zeros, warned of once per role; the agents' tags come with `needs_agents` (RFC 0021) |
| a share per role | `by_role.<role>.share`, in (0, 1): the role's share of the advantage mass (Σ\|A\| over action tokens) that its policy's rows of the entry carry, held by a positive weight `w = share / (1 − Σ shares) × M̂_unshared / M̂_role`, Σ shares being the shares of the policy's roles whose mass is known; the M̂ are decayed masses of the role's rows and of the named roles of its policy without a share, kept per policy, halving every 4 of its versions, moved only in `settle`; applied once, after credit and the KL term |
| a KL term per role | `by_role.<role>.kl: {to: <frozen policy>, coef}` adds `reverse_kl`'s per-token term, `−coef · (logprob − to's logprob)`, to the rows of the role's scored agents; off by default; a scoring that fails fails the group alone |
| metrics | per group, `credit/role_mass/<role>` as sums, `credit/role_weight/<role>` and `credit/role_kl/<role>`; at the step, per role under `roles/<role>/`, over every entry's rows: `mass_share` (its share of the advantage-token mass the step sends), `log_ratio/abs_avg` (the mean \|log r\|), `ratio_outside_frac` (the share of tokens with r outside [0.5, 2]) and `entropy` (the sampler's), r being a token's importance ratio; another policy's lane logs them under `policies/<id>/` |

- **A share, not a fixed weight.** One policy shares a loss, a learning rate and a per-microbatch token mean
  (`platoon/train/step/datums.py`), so a fixed weight drifts 2-6× (EST) as validity rises; two adapters share none of
  it, so masses are per policy. Shares that add up to 1 or more, a share on every role, and a share on a policy that no
  named role without a share samples, which nothing could hold it against (as under two adapters, section 10.2), are
  refused.
- **The weight is read at the sampling version,** as `running` reads its statistics: as the masses stood when the
  role's agents began sampling, from snapshots kept back to the bound. How long a group takes to settle goes with its
  outcome (failures in minutes, successes after their audits), so a weight read at credit time is biased: in a
  simulation where successes settle six versions after failures while the solvers' mass grows, it gave the weighted
  outcomes' mean the other sign than the raw outcomes'. Positive, and once the masses are known fixed before the group
  was sampled, the weight keeps its estimator's signs and expectation. A group whose role's agents began sampling
  before its masses gave a weight reads them as they stand when its credit begins, and the role trains at weight 1
  while they are unknown even then; when a role with a share first carries mass (as `running`'s warm-up ends), its
  policy's masses start over from that group. Under a steady drift the decayed masses lag: where a generator's raw
  part of the mass rises from 0.43 to 0.64, a share of 0.10 holds at about 0.11.
- **The share is pre-Adam,** so the step also logs per-role entropy and importance ratios. KL to the starting
  checkpoint needs a scorer of the starting weights, which no run keeps, and is not built (section 12). The step's
  per-role metrics follow the estimator's class (`logs_by_role`), not the entry's `credit` text, and ρ_G of section
  9.4 is the generator's `roles/<role>/mass_share`.
- **Resume is exact.** `assign` waits on nothing unless a `kl` term scores; state (each role's estimator's, and each
  policy's masses at every version a weight may still read) moves only in `settle` and is saved with the kept groups,
  and the masses sum in the order of the roles' names, so a resume reads the weights to the last bit; a role whose
  `credit` changed starts afresh, with a warning. A censored root stops its group at `no_completed_root`
  (`platoon/train/workflow/group.py`), as today, so nothing credits a tree whose root is unscored.
- **Loading refuses**, each under its path: a role's estimator that does not resolve, is `roles`, or refuses its
  options or its check (`running`'s bound); the shares above; a `kl` whose `coef` is not above 0, or whose `to` is
  `main`, undeclared, trained or of another tokenizer. The roles themselves are not checked against the config, since
  harnesses mint them at run time. `policy_warnings` reads the policies `by_role` names at any depth, and `lock.json`
  records each role's estimator in a record of its own (`level: role`).

### 6. Root roles and program options

**6.1 `environments[].agents.root_role` (M3).** No API set a root's role: a root sampled the default policy, and
TOML entries refused every `agents` setting. With `root_role: solver` an entry's roots open with that role and sample
its bound policy for life (RFC 0021's rule 2); `@pt.program(root_role=)` gives a program's default, which an entry's
own overrides. An entry and a program name a role, never a policy: the config still binds roles to policies.
- **A root of the role `policy` samples `agents.policy`,** whatever `agents.roles.policy` says: that binding has only
  ever reached subagents minted with the role, so honouring it for roots would change what today's configs sample. A
  root of another role samples its role's binding, else the default.
- **No root takes `verifier` or `judge`,** the roles of the agents platoon opens to judge others: their calls never
  train, a judge leaves no trajectory, and an evaluation counts both apart from the policy's agents, none of which a
  root would keep. Loading and the decorator refuse them. A root that judges, such as a judge canary's (section 9.5),
  takes a role of its own bound to the judge's policy.
- **A role-bound root may sample a frozen policy** (a frozen generator to compare against, a judge canary) and makes
  no rows, while the trained agents under it train; loading refuses a run in which only roots sample and none of them
  samples the agents' default.
- **Root roles are part of the policy table.** In a run that records one (`policies:`, or roles bound to policies),
  `identity.json` records every entry's root role (`agents.root_roles`) when some root takes another role than
  `policy`, so that today's records keep their bytes; a resume refuses an entry whose root role changed (exit 78) and
  takes up entries that join, unless the run recorded no root roles: then a joining entry whose root takes another
  role than `policy` is refused too, since the record cannot tell it from an entry whose roots changed role. The record
  resolves each entry's program strictly, and one that does not resolve stops the start, where a rollout still fails
  only where it runs the program.

TOML entries, which refuse every other `agents` setting, accept `root_role`; a root role that looks like a misspelling
of a bound role is warned of; and an `ep.rollout` child takes its entry's root role (M9), so its policy matches its
entry's by construction.

**6.2 Program options and declarations (M4).** RFC 0016's accepted design, built for programs ahead of the rest of
its M2, and stable: `@pt.program(name, options=, check=)`, `environments[].program_options` with
`eval_overrides.program_options` merged over it key by key, `ep.options` (parsed, with their defaults; typed by
`pt.EpisodeContext[MyOptions]`) and `pt.CheckContext`. The schema is a dataclass, a pydantic model, whose unknown keys
are refused as a config section's are, or a function that parses the options' mapping. The check is synchronous and
runs as the config loads, once per phase that the command runs, with that phase's options and rollout config
(`pt.CheckContext(environment, phase, rollout)`, each a copy): it sees its entry and the phase's rollout config (RFC
0016, section 3). A file of environments alone, and `platoon prepare`, check the options and run no check, since only a
training or an evaluation config holds the phase's settings. A named environment that names its program may set both
keys, as their lowest layer. `lock.json` records each phase's options with their defaults, `platoon program show`
lists them, and a rollout parses the route's options again: options the program does not take end it as an `error`
(failure kind `program_options`) before the program runs.
- **Declarations.** The decorator also takes what the rest of this RFC adds, each a declaration of the program's:
  `root_role=` (M3, its roots' role unless the entry names one), `mints=` (M9, the roles it opens itself, reserved:
  section 2.5), `children_with_open_root=` (M13, section 4.2) and `needs_draws=` (M11, for `ep.draw`, which the `rl`
  trainer's `needs()` ORs in).
- **Child entries (M9).** For an entry of a training config that names `child_entries`, `pt.CheckContext` also holds
  copies of what the entry's rows depend on and such a program must check (section 9.8): the child entries (`children`,
  read-only), the run's policies (`policies`, read-only), which say whether a bound role's policy trains, the role
  bindings (`agents`) and the `train:` settings (`train`). They are None for every other entry, and every other check
  keeps RFC 0016's boundary.
- **Not built with M4:** the rest of RFC 0016's M2 (options and checks of function harnesses, named environments and
  reward processors, and a generic `pt.AgentContext`), `platoon extensions check --load` of a named environment's
  program options, and `platoon config show --defaults` filling in a program's defaults.

### 7. Frozen hosted policies may judge in training (M12b)

`policies.<id>.hosted` serves a frozen policy from a hosted model through RFC 0015's pass-through: a judge of another
model family, say. Today a hosted model cannot sit in a Tinker gateway: `ModelGateway.chat` prepares each call with
the agent's runtime, which `HostedGateway.prepare` refuses, and the HTTP server forwards a call when *the gateway* is
hosted, not when the agent's runtime is (`platoon/gateway/server.py`). M12b:
- **dispatches hosted calls per agent runtime:** `ModelGateway.chat` hands such an agent's call to its runtime's
  `chat`, which has the same signature, and the server forwards its requests, in process first, then over HTTP;
- signals `first_call` to the start limit from the hosted runtime, and refuses a hosted policy that is trainable, that
  `agents.roles.judge` does not bind, or that a role other than `judge` binds unless that role is an entry's root role
  (a judge canary set's root, which takes a role of its own since no root takes `judge`: sections 6.1 and 9.5; open
  question 6 may lift the rest);
- **amends RFC 0015:** *frozen hosted policies may judge in training; their calls never train.* A hosted call records
  text, never token ids (`platoon/gateway/hosted.py`), so nothing more could train;
- is accepted by a test: a `platoon-react` judge on a hosted frozen policy, on a fake provider, in a Tinker gateway.

### 8. Guards for long episodes

**8.1 Drains before planned stops (M14).** A stop request ends the run at the next step boundary, losing every episode
in flight (`platoon/train/trainer/signals.py`). On a drain signal from the job script, about p95(L) before a walltime,
and before the plan's end, the dispatcher stops drawing entries with `rollout_permits` above 1; their groups settle.

**8.2 A step-stall watchdog (M15): `train.step_stall_factor`.** In-flight groups feed the 600 s watchdog
(`watchdog.timeout_seconds`) with no check of progress (`platoon/train/trainer/waiting.py`), so a weight push or a
sampler that hangs is caught only after up to the longest episode. The trigger fires when work waits and nothing moves
for `step_stall_factor × p95(T_step)`, T_step being a step's duration: a step's worth of settled groups has waited
unconsumed, or groups are in flight and no model call has completed in the run. It fails the start, and supervise
(`watchdog.max_restarts`) restarts it. "No committed step" alone would loop: after a restart the first commit can come
later than 3 × p95(T_step) (EST 90-135 min for envgen's long productions). Tests: a fake refill longer than that whose
calls complete does not restart; a fake push hang and a fake sampler hang do.

**8.3 Crashes and resume.** A full save holds kept groups as whole trees with their advantages, `roles`' state (each
role's estimator's, `running`'s as a base and its settles, and the masses), the dispatcher's state and the archive's
(its journal's length and its alarms' counters, section 9.6); `recovery_every: 1` makes every step one. A resume re-runs
every outstanding episode as a new draw, since draw keys never repeat (`platoon/train/task_stats.py`); kept trees carry
their verdicts, so no judge is asked twice. What nothing names is deleted, but only what the run alone holds (M12): the
productions' directories of the archive's own starts that no kept group or journal row names, and of the runtime tasks
such a production took in (`stored.json`) those that no other production of the run took in, from the stores the run
owns alone, a solver entry's default `runtime-tasks` in the run's directory. Store entries are content-addressed and
shared (the grading canary is one task for every production of a template), so a store that `runtime_tasks.store` names,
which other runs may share, is never collected. An evaluation during training notes what its productions take in from
the run's store in directories of its own, below its rollouts (`<run>/rollouts/<scope>/<version>/envgen/`), and
collection reads their `stored.json` and evidence too, so nothing that such an evaluation took in goes with the garbage.
Appendix A.5 prices a crash.

### 9. The envgen program (experimental)

`platoon.experimental.envgen` holds the program `platoon-envgen`, its evidence and rewards, and the step hook
`platoon-envgen-archive`. A **production** is one draw of the envgen entry: one seed's generator episode, the task it
wrote, the gates, K solver rollouts on that task and their audits. A **seed** is an authoring task the toolkit built.
The **toolkit** is a task-generation pipeline's command line, such as a paper-to-task pipeline's, which supplies data
and pure functions (section 9.7); platoon runs every sandbox and every model call. The solver entry's own curated
groups train beside the productions and anchor the solver. A variant, in a package of its own (RFC 0013) that registers
its own program, as SPADE's regret does below, overrides the five stages, `archive`, `workspace` and the budgets' knobs
and names, and may add roles to `mints`; `register()` refuses one that overrides anything else, such as `run()`, which
also scores. A configuration-only project (RFC 0018) selects `platoon-envgen` and writes no Python.

```python
class EnvGenProgram:                               # run() fixes the order, budgets, attribution and scoring
    root_role = "generator"                        # register() hands these three to @pt.program
    mints = ("control", "judge")
    needs_draws = True
    async def run(self, ep) -> Outcome: ...        # the five stages, then the score: attribution and the reward
    async def generate(self, ep, ev): ...          # memory into the authoring sandbox, then ep.run()
    async def package(self, ep, ev): ...           # collect, export, materialize; then closes ep.session
    async def gate(self, ep, ev, task): ...        # the final check and the controls (inspections)
    async def solve(self, ep, ev, task): ...       # K rollouts of the child entry
    async def audit(self, ep, ev, task): ...       # the pre-committed sample (inspections)
    def archive(self, ep) -> Archive: ...          # what a production reads of the archive (section 9.6)
    def workspace(self, ep) -> Workspace: ...      # the authoring sandbox: memory in, outputs out, the seed's data
    @classmethod
    def check(cls, options, ctx) -> None: ...      # load-time refusals (section 9.8), with a variant's budgets
    @classmethod
    def register(cls, name: str, *, options: type) -> EpisodeProgram: ...   # @pt.program(name, check=cls.check, ...)

@dataclass
class ProductionEvidence:                          # envgen.evidence/v1: run() fills it, a reward only reads it
    seed: SeedRecord                               # task id, strata, visits
    generator: AgentRecord                         # status, turns, tokens, tool calls, versions, error
    stages: list[StageRecord]                      # name, ok, reason, fault, seconds, wait seconds, budget
    task: TaskRecord | None                        # digest, the grader's owner, difficulty knobs, pass_audit, and
                                                   #   from M12 its instruction and the final check's answer
    controls: list[ChildRecord]                    # per child: status, reward, reason, fault, failure kind,
    solves: list[ChildRecord]                      #   versions, seconds, replay results, regrades, margins, and
    probes: list[ChildRecord]                      #   whether it was retried, censored or rescored
    audits: list[AuditRecord]                      # kind, trials, schema-checked verdict, clean, consequential
    canaries: list[CanaryRecord]                   # attribution rule 2's canaries and their results
    verdict: Verdict                               # section 9.3
    draw: list[int] | None                         # the draw's key
    abstention: dict | None                        # the generator's claim, when it abstains
    adjudication: dict | None                      # adjudicate's answer
    reward: str | None                             # the reward that scored it,
    score: dict | None                             #   and its ProductionScore
    stop: str | None                               # why a usage error of the toolkit's stops the run (section 9.7)

class ProductionReward(Protocol):                  # pure and total
    def __call__(self, evidence: ProductionEvidence, options: Any) -> ProductionScore: ...

@dataclass(frozen=True)
class ProductionScore:
    reward: float | None                           # None: censored
    train_solvers: bool
    admitted: bool                                 # whether the task joins the exported dataset
    components: Mapping[str, float]
    reason: str

platoon_envgen = EnvGenProgram.register("platoon-envgen", options=EnvGenOptions)

class HintedEnvGen(EnvGenProgram):                 # SPADE's regret, in a package of its own
    mints = ("control", "judge", "probe")          # a variant may add the roles it opens
    async def solve(self, ep, ev, task):
        solves = await super().solve(ep, ev, task)      # the K measuring rollouts
        ev.probes = await self.hinted(ep, task, n=4)    # hinted runs: inspections, never trained
        return solves

acme_hinted = HintedEnvGen.register("acme-hinted", options=HintedOptions)
```

**9.1 Stages, sandboxes and budgets.**

| Stage | What runs | Sandbox | Typical (EST, short / long) | Budget |
|---|---|---|---|---|
| `generate` | memory files from the archive, then the trained policy through the entry's harness, with the toolkit's `check` inside the sandbox | authoring: materials read-only, no network | 2-10 min / 0.5-2 h | the seed's `[agent].timeout_sec` (20 min for short productions), at most what the deadline leaves once the later budgets are set aside |
| `package` | collect (regular files, caps), the toolkit's `export` (runs nothing generated), `materialize` with the grader's owner, replay checks and replay time that `export.json` gives (section 9.7); the root's session closes | host | under a minute | 2 min, and the export three times (its run, its retry and the toolkit's canary, 2 min each) |
| `gate` | the toolkit's final check (model-free), then 6 controls as inspections | agent and verifier sandboxes | 1-3 min / 5-15 min | the final check three times; the 6 controls, then as long again for their retries, in waves of the permits, each a sandbox start, a quarter of the solver's time limit and its grades (a verifier start and time limit, once more per regrade); the grading canary |
| `solve` | K rollouts of the solver entry, concurrent, no straggler cut | the solver's | 10-15 min / 30-60 min | the K solves, then as long again for their retries, in waves of the permits, each a sandbox start, the solver route's `timeout` (15 min) and its grades; the grading canary |
| `audit` | judge inspections of a review task; the toolkit's `adjudicate` classifies the verdicts | review: the privileged task and the sampled trials, no network | up to 10 / 20 min | as many reviews as K, in waves of the permits, each the review's export three times and two judge attempts of a start and `audit.cap_seconds` (10 / 20 min); the judge specs, the judge canary and `adjudicate`; a second round for the failures a review calls infrastructure's |
| score, in `run()` | attribution with canaries and one retry, then the reward | host | seconds | in the stages' budgets, and a grace of 1 min |

- **The trust boundary.** The host runs program code and the toolkit's pure verbs, which run nothing generated;
  everything the generator wrote runs in sandboxes; and a grader it wrote can only choose 0 or 1, which controls and
  audits check. **The generator writes content, never the envelope:** not `task.toml`, the image, the network, users,
  resources, timeouts, the verifier mode or a reward file. Its only feedback is the toolkit's validity check, never a
  solver's outcome, and its harness cannot open solvers, choose K, read verifier output or edit gates.
- **One deadline, stage budgets under it.** The worker's deadline caps the whole tree (`rollout_config.timeout`, plus
  the seed's sandbox overhead and a grace, `platoon/train/workflow/requests.py`), and `run()` holds every stage to it. A
  stage's budget is the most it can run whatever the generator wrote, since nothing the generator can cause may reach
  the deadline (attribution rule 3): it counts every retry and canary attribution may run, each toolkit verb three times
  (`toolkit_seconds`), each sandbox's start (`start_seconds`; a slower start is infrastructure's wait), each grade and
  regrade, and children in waves of the permits. The budgets are bookkeeping, not timers: each child runs under its own
  limit, each verb under its timeout, each judge attempt under `audit.cap_seconds`, and the generator under the seed's
  own limit, cut to what the deadline leaves once the later budgets are set aside; a deadline that leaves it nothing
  voids the production at once (`void(deadline)`). A child's clock starts when it starts running; its wait for a seat or
  a start slot is recorded as `wait_seconds` and charged to no one. An overrun takes its own row of the fault table; a
  production that infrastructure holds up past its deadline is `void(deadline)`, and a timeout of platoon's own within a
  stage `void(infra)`. Loading refuses a deadline at or below the sum of every budget but the generation's, which is the
  seed's and unknown at load, plus a grace, and prints the sum: 12,360 s for section 10.1's shape with a 4-minute
  verifier, so the 90-minute deadline this RFC first priced is refused (Appendix A.1, open question 9).

**9.2 Attribution.** On curated data platoon censors what it cannot measure (`platoon/envs/service.py`). On a
generated task the dataset is the policy's output, so the taxonomy flips. Four rules:
1. **Provenance.** What the generator wrote is its fault; a contract-breaking artifact or a submission that does not
   replay is the solver's; a failure of platoon, a sandbox, a sampler, a judge or a trusted command is infra.
2. **A canary and one retry before censoring.** A known-good canary on the same service: a failing canary or a
   successful retry means infra, and a healthy canary with a reproduced failure means the generator's. The program makes
   its canaries itself (M11): the grading canary, the oracle on a task with a trusted grader that its reference solution
   passes, written from the solver entry's runtime-task template and taken in with `ep.materialize`, so that it vouches
   for the sandboxes, the verifiers and the launcher at the gate as well as after it; the judge canary, a judge on a
   review whose answer its instruction gives, asked once per production; the toolkit's, its data verb (`judge-spec`)
   answering and the production's directory taking a file; and the collection's, a file the program writes into the
   authoring sandbox. No canary judges a sampler, the platform's setup and launch of the solver's harness, which the
   oracle never runs, or a verb that reads nothing generated, and the toolkit ships no known-good input per verb: a
   solve whose two attempts fail as infrastructure would, one of them in its sampler or in what the platform could not
   set up or launch (`harness_setup`, such as a toolchain the node could not stage, a gateway that cannot cap the
   harness's calls or no route to the gateway; `harness_config`, a spec that cannot launch with the run's settings), a
   review whose judge's sampler failed it or whose sandbox did not start, `judge-spec`'s failures and a verb that a
   signal killed from outside are censored (a production left with too few valid solves is `void(infra)`, as when every
   solve fails so); the generator's own sampler failure voids the production at once; and a verb that crashes on every
   input while `judge-spec` answers is charged to the generator, loudly, since every production's validity drops. An
   install of the solver's harness that fails in the task's sandbox runs among the task's own files, which can break it,
   so it has a failure kind of its own, `harness_install` (M11, proposed: open question 11), and the grading canary
   decides it as any other reproduced failure: one that infrastructure breaks on both attempts beside a healthy canary
   is the generator's, the trade of open question 12. Each finding is a `CanaryRecord` in the evidence, of its canary's
   kind (`grading`, `judge`, `toolkit`, or `workspace` for the collection's; `healthy` None where none ran) or, for a
   child's failure that no canary judges, `sampler` or `platform`. Where the table names no row, this rule decides: a
   judge out of its time twice beside a healthy judge canary is the generator's, `invalid(review_malformed)`, as a judge
   malformed twice is.
3. **No escape hatches.** Nothing the generator can cause maps to censoring, the deadline included.
4. **No role pays for another's fault.** A generator fault *drops* the solver rows, since zero-advantage rows keep
   their token denominator (`platoon/train/step/step.py`); an infra fault censors the generator, never penalizes it.

| Fault | What is observed | Verdict | Solver rows |
|---|---|---|---|
| infra, in the production | authoring infrastructure fails; a trusted command crashes and its canary fails; export refuses trusted input the host handed over (`fault: infra`, such as an unusable seed); fewer valid solves than the solver entry's `min_successful_group_size`; a judge outage with a failing judge canary, after backoff within the queueing allowance; an infra fault in the duplicate index; the worker dies; the deadline passes, which only infrastructure slower than the budgets allow can cause, or a usage error of the toolkit's | `void(infra)`; `void(deadline)` and `void(toolkit_usage)` for the last two | none train (lost with a dead worker) |
| infra, in one child | a child's sandbox start or platoon-side grading fails (`verifier_start`, `verifier_trusted_launcher`), a trusted grader gives no score (`verifier_trusted_*`) or the solver's harness install fails in the task's sandbox (`harness_install`, proposed), unless rule 2 reproduces it with a healthy canary, which makes it the generator's; a solver crashes with gateway errors or a sampler times out, or the platform cannot set up or launch the solver's harness (`harness_setup`, `harness_config`), which no canary judges; an audited failure whose review calls it infrastructure | s and n leave that member out; an audited one is replaced by the seeded rule | that member censored |
| generator | no bundle within the budget; a malformed bundle; export refuses what the generator wrote (`fault: generator`), or `materialize` or the final check refuses; a trusted command crashes on the content with a healthy canary; the oracle fails, disagrees with itself or takes over a quarter of the solver's budget; a no-op, a mutant or an author control passes; a metamorphic variant fails; a generated grader gives no score (`verifier_generated_*`; an `_unfinished` one once a healthy canary rules out the sandbox); a regrade disagrees on the same bytes (`verifier_flaky`); an audited failure whose review is not clean (a false negative, a grader the specification does not support, unresolved); a pass that exploits the grader; a judge malformed or out of its time twice with a healthy canary; a sampled all-fail audit that finds a false negative or an intractable task | `invalid(<kind>)` | dropped |
| claim | the generator abstains: no viable task in the seed; 25% of abstentions are audited | `abstained(unaudited)`, `abstained(confirmed)` or `abstained(refuted)` | none |
| duplicate | an exact or near duplicate of the archive or of a task in flight | `duplicate` | not solved |
| solver | its budget runs out; it crashes with a clean gateway; it fills its context window; the launcher scores it 0: its artifacts break the output contract (`solver_output_contract`), its replay fails or times out, or a regrade's replay writes other bytes that score otherwise (`solver_nonreproducible`), it misses a variant (`solver_variant_failed`) or states a forbidden literal (`solver_hardcoded`); a pass that its pass review finds faked, rescored as a failure | s and n count it as a failure | that member scores 0 |
| none | all n valid outcomes are equal | `valid(s, n)` with s = 0 or s = n | zero advantage |
| none | mixed outcomes, and every audit clean | `valid(s, n)` | trained, leave-one-out |

The program applies the table with `ep.drop` and `ep.rescore` (section 2.1), which only children allow: a root's rows
always train. A veto drops every solver child and needs no rewrite, since none of the production's solver rows train. A
censored member is dropped with `censor=True`: its outcome becomes `invalid`, so its reward is None and enters no
sibling's leave-one-out baseline (`platoon/train/credit/keyed.py`), which a scored 0 that was the infrastructure's would
otherwise lower. The solver's own failures that its entry leaves unscored, and a pass that its review found faked, are
rescored to 0, so that the solver learns from them, the child's record keeping what they were. A root of the solver
entry would be left out instead: under `on_harness_exit`'s defaults a time limit that ran out and a crash with a clean
gateway are, and so is a filled context window where `on_context_overflow` is `exclude` (its default, `grade`, grades
it). R1 governs how a child runs and grades; what its rows train on is its program's to settle, as `ep.drop` already is.
For a void production the program returns an infra outcome: nothing in its tree trains.

**9.3 Evidence and rewards.** `ProductionEvidence` is versioned JSON, `envgen.evidence/v1`, in the root's misc and in
a directory per draw under `DrawInfo.run_dir`, named by `DrawInfo.key` (`envgen/<start>-<seq>/evidence.json`, synced to
disk before the group settles), where the archive hook reads it; beside it lie the final check's answer of a task that
passed it (`final-check.json`) and the runtime tasks the production took in (`stored.json`), and judge transcripts go to
side files keyed by digest. **The schema is additive:** a field keeps its meaning once defined, and readers ignore
fields they do not know. Its verdict is `valid(s, n)` (n valid solves, s of them passes), `invalid(<kind>)`,
`abstained(unaudited | confirmed | refuted)`, `duplicate`, or `void(<kind>)` for what was not measured: `void(infra)`,
`void(deadline)` for a production that infrastructure held up past its deadline, `void(toolkit_usage)` for a usage error
of the toolkit's, and for audits while judges are off, `void(judges_off)`. Only a reward turns evidence into
numbers (R5), so any reward can re-score the archive offline.

Rewards are pure functions of the evidence, registered in `platoon.experimental.envgen` with
`@production_reward(name, options=)` and selected by `program_options.reward`, with `reward_options` beside it; the
evidence records which reward scored it. A name resolves as one of envgen's, one that a `package:` module registers,
or `module:attr`: RFC 0013's rule, with no component kind or entry-point group until a second user needs one.
`platoon.experimental.envgen.check_reward` checks a reward pure and total over a corpus of evidence. The default:

```text
R = ⊥     void(infra) (nothing trains; no estimator state moves)
  = −1    invalid(<kind>): a gate failure, a generator fault, a veto or an unresolved audit; abstained(refuted)
  = −0.5  abstained(unaudited); duplicate
  = 0     abstained(confirmed), and the seed retires; valid(s, n) with s = 0 or s = n
  = q · 4s(n − s) / (n(n − 1))     valid(s, n) otherwise, n being at least min_successful_group_size
q = the mean, over the audited failures f, of 1 if f's review calls it consequential, else 0.25
```

- **Learnability, unbiased.** 4s(n − s)/(n(n − 1)) is unbiased for 4p(1 − p) at every n ≥ 2, so censoring that leaves
  fewer valid solves does not bias it, where dividing by its maximum at n would. p(1 − p) is the learnability of
  [Sampling for Learnability](https://arxiv.org/abs/2408.15099). Under leave-one-out the solver's expected advantage
  mass is 2p(1 − p), so the generator is paid twice that, times q.
- **−1 for invalid output,** as AutoEnvScaling and Absolute Zero penalize it, and **−1 for `unresolved`**: censoring
  would score the sample at its baseline, which a generator could buy by breaking the judge.
- **q weighs audited failures, not their number:** a task cannot raise its reward by failing in more places, and the
  final check refuses tasks that try.

| Name | Formula | Use |
|---|---|---|
| `learnability` | above | the default |
| `tiers` | AutoEnvScaling's: −1 invalid, −0.25 uncalibrated, +1 when s/n is in [0.25, 0.75], with the vetoes above | an ablation |
| `plateau-regret` | SPADE's: 0.6 · plateau(p̂; [0.4, 0.6], ramp 0.25) + 0.4 · clip(max(0, p̂_hint − p̂) / 0.15, 0, 1), with hinted inspections in a `solve` override | an ablation; watch for hint leaks |

**9.4 Credit and staleness.** The generator's rows are credited by `running`, keyed by the seeds' strata and read at
the version the generator sampled, with a share of 0.10 of the entry's advantage mass; AutoEnvScaling updates its
proposer on about 6% of its updates. The solver's rows are credited by `keyed` on `[role]` with leave-one-out, which
over a production's K children is GRPO without the std division, `A_j = r_j − (s − r_j) / (n − 1)`
(`platoon/train/credit/keyed.py`). K and the minimum of valid solves are the solver entry's `group_size` and
`min_successful_group_size`, which the request carries with its child settings (`ep.child_entry`);
`program_options.solve: false` skips the solves, to measure generation alone in an evaluation during training
(`eval_overrides.program_options`), and loading refuses it in training, where the generator would learn validity alone.
ρ_G, the generator's share of the step's advantage-token mass over every entry's rows, is
`roles/generator/mass_share`. Generator rows are sampled at version v and train at about v + L / T_step:
- the default `cispo` loss applies truncated importance ratios (`platoon/train/config_defs.py`), SPADE's correction for
  its delayed designer;
- **the envgen entry's bound covers the deadline,** `max_staleness = ⌈deadline / p05(T_step)⌉ + 1`, so a production
  that ends within its deadline is masked only when steps run abnormally fast. A bound at p95 latency would mask, by
  construction, the generator rows of the slowest productions: mixed tasks with audits, the only class with positive
  reward. The same bound judges the production's solver rows (section 2.1). The how-to's 28 covers its 16,200 s deadline
  at a step every 10 minutes at the fastest (open question 9); no load check ties the bound to the deadline;
- `train.stale_filter: datum` is required. It is run-wide, which makes it a deliberate change for the solver entry
  too: its partly stale groups train their fresh rows instead of dropping whole. Masked generator rows raise an alarm
  (section 9.6); the structural fix is two adapters, whose generator rows age in generator versions (section 10.2);
- **routing replay.** A row carries its last call's routing handle, which the service resolves at train time
  (`platoon/utils/tinker_data_processing.py`), and generator rows train a production's latency after sampling (EST
  30-50 min for short productions, 1.5-3.5 h for long ones). Whether a service holds a handle that long is UNVERIFIED,
  and platoon records no sign that replay applied (RFC 0021's U9), so before the first study on a mixture-of-experts
  base a check trains a row whose handle is older than the envgen bound and compares its training logprobs with and
  without the handle, with any replay counters the service exposes. If replay does not apply, the study uses a dense
  base; the fallback is routing-handle coverage by row age among the staleness alarms.

**9.5 Judges and audits.**
- **A judge is an inspection** in platoon's `judge` role: `ep.inspect(review, entry=<solver entry>,
  harness="platoon-react", role="judge", options=..., grade=False)`, where `entry` names the entry whose service opens
  the session, not a route to run. `review` is a review task that the toolkit exports and the program materializes,
  which holds the privileged task, the controls' reports and the sampled trials (`trial.json`, an ATIF trajectory, the
  kept artifacts). A judge runs on the rollout host, never in the review's sandbox, where the trials it judges could
  read its key and write its verdict: a function harness, as here, or a CLI harness only through an entry placed `local`
  (section 2.6). Its tools act only on that sandbox, each of its two attempts at most takes `audit.cap_seconds` (section
  9.1), and its final message (`ChildRollout.final_message`) is checked against the judge spec's schema
  and classified by the toolkit's `adjudicate`.
- **Why not platoon's other judges.** `ep.judge_endpoint()` is a bare endpoint, and an audit needs tools on the review
  sandbox; verifier agents grade one child and gate its score, while audits are drawn across siblings after all K
  land, read the privileged task and veto the whole production.
- **What is audited**, a constant budget per informative task, drawn after all K land and seeded by the draw key
  (auditing failures as they land would favour fast ones): of a mixed task, 2 failures and 1 pass; of all-fail tasks,
  25%, with 2 failures and a tractability review each; of an all-pass task, nothing; of abstentions, 25%. One
  exception: a task whose export sets `pass_audit: every` (its variants leave a target uncovered, where only a judge
  can catch a hardcoded answer) has every pass audited, mixed or all-pass, and `adjudicate` admits it only with a
  review of each. Judges veto, rescore a faked pass as a failure, or lower q, and never pay; with the run's judges off,
  which only evaluation allows, a production whose sample holds a review is `void(judges_off)` as a whole. No stage runs
  a review of a task as a whole (`judge-spec --kind task`), so the fault table names none.
- **Calibration and deep audits, outside the productions.** A judge canary set with known verdicts runs as an entry,
  evaluated at the start and every `eval.every` steps, whose root takes a role of its own bound to the judge's policy
  (M3: no root takes `judge`); deep audits re-audit a sample of the audited trials with a full budget and a judge of a
  second family, to estimate the online audits' miss rate and the consequential flag's precision (Appendix A.4).

**9.6 The archive, alarms and stops (M12).** `platoon-envgen-archive` is a step hook (RFC 0021, section 9) and one of
core's names, with the options `into` (the archive's directory: `envgen-archive` by default, and a relative one, in the
run's directory), `strata` (the seed's columns that make the stratum whose exemplars and negatives the memory quotes,
`[kind, mode, domain]`), `keep` (the most past tasks, exemplars and negatives it keeps of a seed or a stratum, 8) and
`alarms`; its metrics and its checkpoint entry are `envgen_archive`.
- At each step's end, before the step publishes, it appends to a journal in the archive (`envgen.journal/v1`) the
  evidence of the draws the step committed (`StepEnd.consumed`), each with its fate (consumed, stale or failed), and
  builds from it the generator's memory and the duplicate index. The memory holds a seed's past tasks with their
  outcomes and, for a task that was not valid, a failure excerpt, and its stratum's exemplars (admitted tasks) and
  negatives (invalid ones), of which a production quotes 3, 2 and 1 by default (`program_options.memory`). The excerpt
  is the stage that ended the production and what `materialize` and the final check found, never what the audits found:
  those are judges' findings about the solvers' trials, and the generator's only feedback is the toolkit's validity
  check (section 9.1). The duplicate index holds the valid tasks, so that a repeat of a refused task is refused for what
  it is. Stepping independently, the hook journals at the leader's step ends, which consume every group.
- **Tasks in flight** are those that passed their final check and that the journal does not hold yet, of the current
  start or of a group its checkpoint kept, as the productions' directories record them (`final-check.json`); a
  production hands them to its final check (section 9.7), which reads their files in the store.
- **No group admission rule.** A group that a rule rejects, or that fails, is committed as it settles and reaches the
  step without its draw's key, so `StepEnd.consumed` never names it: the hook refuses a config in which any entry's rule
  (its own, else `train.group_admission`) is other than `all` (exit 78), until a step end names those draws with their
  fates (section 12).
- **Durable before referenced, and one run's.** The journal is fsynced when the hook's checkpoint participant captures
  its length, not on every append, so a saved length never points past durable bytes, and a production syncs its
  evidence and its final check's answer as it writes them. A resume cuts the journal back to the saved length and
  rebuilds the memory and the index from it, so the steps that run again journal their productions once, and deletes
  what nothing names (section 8.3). The archive's manifest names its run, and the start that keeps it holds a lock
  (`.lock`): a new run whose archive already holds a journal, a resume of another run's archive outside its own
  directory, a journal shorter than the checkpoint says and a second start are refused (exit 78).
- **Alarms and stops.** It computes its alarms from the journal and the step's metrics: exploit kinds, contamination,
  validity, judge outages, censoring, ρ_G, the generator's mean advantage, masked generator rows (from
  `roles/<role>/stale_share`, which the step logs under `credit: roles`, and records before its hooks run), the
  generator's mean |log r| against the solver's, the top domain's share, and `toolkit_usage`, a production that ended on
  a usage error of the toolkit's (section 9.7). Each has an action, `stop`, `warn` (logged as it starts to hold) or
  `off`; the thresholds are the hook's options (defaults in Appendix A.3, which also lists the alarms not built). A stop
  sets `run.shutdown` with the alarm's name (`platoon/train/trainer/run.py`), so the run stops at the step's boundary
  with a checkpoint, as on a stop request, which supervise does not restart; a hook that raised would fail the start,
  which supervise restarts. A resume restores the alarms' counters and clears no signal, so an alarm that stopped the
  run stops it again once its signal holds after the resume (validity once the refill window has closed, contamination
  and usage errors on the next such production), until its thresholds or its action change.
- **Where the generator's alarms read.** Stepping independently, with the generator's role on a trained policy other
  than the leader, on a clock of its own (section 10.2), the masked generator rows and the generator's mean |log r| are
  read at that policy's own step ends and logged in its row, the latter against the last mean |log r| that the solver's
  steps logged (the leader's, where the solver samples the leader); the generator's mean advantage, its groups' credit,
  is read at the leader's. ρ_G (`generator_share`) says something only of a step that holds the solver's rows beside the
  generator's, so it is read only where one trained policy samples both roles, at that policy's step ends (on a clock of
  its own, outside the refill window that the leader's keep). Where the two roles sample different policies (two
  adapters, in lockstep or not, or a solver on a policy of its own), each lane holds one role's rows, a share of 1 in
  the generator's by construction, and where they sample one the run does not train, no step holds their rows: there the
  alarm is off, which an INFO line of the log says, and the hook refuses a config that sets its action to anything but
  `off` (exit 78), so section 10.2's config loads as written. One shape stays blind: a solver on a clock of its own
  beside a generator on the leader leaves `generator_log_ratio` no solver ratio at the leader's step ends.
- **The refill window.** The validity and share alarms skip the step ends after each start while batches hold only fast
  groups (Appendix A.3): by default ⌈p50 / p50⌉ of the latencies of the mixed productions among the journal's last 100
  (of the measured ones when none mixed), a latency being the stages' running time, and of the last 16 step durations
  the hook timed, across starts. It is read again at each step end while it is open, so it grows while slower mixed
  productions land, and once closed it stays closed for the rest of the start (`alarms.refill_steps` sets it instead);
  whether to fix it at its first value instead is open question 10.

**9.7 The toolkit contract.** Neither side imports the other: the config names the toolkit's command
(`program_options.toolkit`), and each verb runs as a `ToolkitCommand` (a data spec: an argv, no shell, ulimits and a
timeout). Exit 0: stdout is one JSON object with `schema` (`envgen.<verb>/v1`) and `ok`, false when the verb refuses
its input, with `fault` naming whose: `generator` for what the generator wrote, `infra` for trusted input the host
handed over (attribution rule 1); an abstention names none. Exit 2, or an answer of another schema major: a usage error,
which ends the production as `void(toolkit_usage)`, with why in the evidence's `stop`, and stops the run through the
archive's `toolkit_usage` alarm (section 9.6). Other exits, and an answer that breaks the verb's contract (a refusal
that names no fault, an export without its `export.json`), are a crash (attribution rule 2). The toolkit emits data
only, never a launcher, a `tests/test.sh` or a mode that grades on its own: platoon's launcher grades every task it
exports (section 3.2).

| Verb | Where | Returns |
|---|---|---|
| `seeds build` | offline | a Harbor dataset of authoring tasks, their strata in its task index, an exclusion manifest applied; for trusted graders, secretly perturbed inputs and the reference's outputs on them, from which `export` makes replay variants |
| `check <bundle>` | in the authoring sandbox, model-free | `{ok, problems, feedback}`, with repair text |
| `check --final <dir> --archive <index> --store <dir> --in-flight <file>` | host; runs nothing generated | `{ok, problems}`: structure, a static audit, leaks and planted secrets, files that would shadow what its grader imports, contamination, duplicates; and `verdict`, `fault`, `checks` (each check's problems by its name), `digest`, `fingerprint` and `duplicate`, which the evidence records: the program reads the verdict and the fault, the archive the checks (`leak` and `contamination` for its alarms), the digest and the fingerprint (section 9.6). `--archive` is the duplicate index, `--in-flight` a JSON list of the digests of the tasks in flight |
| `export <bundle> --seed <dir> --to harbor --template <toml> --out <dir>` | host, pure | `instruction.md`, `environment/`, `solution/`, `tests/gen/` (the grader, its `grader.json`, the replay checks' data), `controls/` and `export.json` (the artifacts, the grader's owner, the replay and its time, targets, tolerances, variants, forbidden literals, `pass_audit`), and no `tests/test.sh`; `--seed` is the seed's trusted data (its inputs and their perturbations); `--review <kind>` writes a review task of the trials, with the controls' reports (`--controls`) and the solver's harness, step budget and time limit (`--learner-resources`), and `--review abstention` an abstention's |
| `judge-spec --kind failure\|pass\|tractability\|abstention` | data | prompts, rubric, JSON schema, tools, turn cap; `--kind failure` is also the toolkit's canary and, at load, its schema's check (section 9.8) |
| `adjudicate --task --trials --reviews --min-valid` | host, pure | `{admitted, clean, consequential, vetoes, tractable}` for any K, censored trials left out, `--min-valid` being the solver entry's `min_successful_group_size`; the first veto's reason, made one word, is the verdict's kind (`invalid(<reason>)`) |
| `mutants <dir> --secret <hex>` | host, pure | mutants and metamorphic variants of the oracle's artifacts, for generated graders (M16's `platoon-artifacts` runs them) |

**The program maps; generic platoon code never reads `export.json`.** `package` maps `export.json` onto
`materialize`'s arguments: the grader's owner onto `grader_owner`, the variants and forbidden literals onto
`replay_checks`, the replay's time onto `replay_timeout_s`; `audit` reads its `pass_audit` (section 9.5). The
measurement that comes before the program materializes exported tasks the same way (Appendix A.6). The toolkit's
files carry schema names too: `export.json` is `envgen.task/v1`, and a review task's manifest
`envgen.review-task/v1`. A grader writes the launcher's verdict (`$PLATOON_VERDICT`, section 3.2); the program writes
each trial for judges as `trial.json`, `agent/trajectory.json` (its ATIF projection, RFC 0007's section 4, in envgen's
module until RFC 0007's export lands), `output/` and `verifier/` (the reward file and the launcher's report).

**9.8 Load checks.** Beyond the generic refusals of sections 2.1 and 4.1, `EnvGenProgram.check` refuses, before any
service opens and through `pt.CheckContext` (section 6.2): an entry outside a training config, or with anything but one
child entry; a `train.stale_filter` other than `datum` (the message is the line to add); a `group_size` other than 1, or
`train.num_microbatches` above 1, where a production's generator rows would share a token mean with its solver rows; a
child entry without `runtime_tasks`; depth weighting or a subagent keep probability below 1; an unbound or trainable
`judge`, which would sample the trained default or train; `running` without `priors`; no deadline, or one at or below
the budgets but the generation's plus a grace (section 9.1); `solve: false` in training, where the generator would learn
validity alone; a reward that does not resolve or refuses its options; and a toolkit that does not run, exits 2 or
answers in another schema major, which loading asks of its `judge-spec` once per command: the only check that runs a
command, where RFC 0016's checks read the config alone.

### 10. Configuration and new names

**10.1 One shared policy.**

```yaml
trainer: rl
train:
  model_name: <base>                      # one policy, both roles
  renderer_name: <renderer>
  batch_size: 16
  max_staleness: 3                        # the run's k: the horizon
  stale_filter: datum                     # run-wide; platoon-envgen refuses any other (section 9.4)
  max_concurrent_rollouts: 192
  max_starting_rollouts: 32               # children take start slots too (section 2.3)
  step_stall_factor: 3                    # section 8.2
  workflow_config: {rollout_isolation: process}
checkpoint: {every: 10, recovery_every: 1, keep_recovery: 29}  # 29 × 1 > envgen's bound (checkpoints/pruning.py)
policies: {judge: {model_name: <another family>, renderer_name: <renderer>, tinker_base_url: <judge service>,
                   trainable: false}}
agents: {roles: {judge: {policy: judge}}}  # refused if unbound or trainable
environments:
  - name: curated                         # the solver entry: its curated groups, and the route of every solve
    weight: 0.2
    service: harbor
    service_kwargs:
      dataset: <curated dataset>
      agent_user: ignore
      runtime_tasks:                      # runtime tasks only; curated grading is unchanged
        images: {<image name>: <prebuilt image>}
        template: <task.toml>             # no network, a separate verifier as root, fixed limits (verifier: 4 min)
        verifier_user: honor
        verifier_provider_options: {rootfs_layout: copy, fakeroot: true}   # on Singularity
    harness: openhands
    agents: {root_role: solver}           # its groups and the productions' solves share a role
    workflow_config: {group_size: 8, leave_one_out_baseline: true, min_successful_group_size: 4,
                      rollout_config: {max_steps: 50, timeout: 900}}
  - name: envgen
    weight: 0.8
    service: harbor
    service_kwargs: {dataset: <seeds from the toolkit's seeds build>, agent_user: ignore}
    harness: platoon-react                # the generator's harness
    program: platoon-envgen               # root role generator; K and the minimum from the child entry
    program_options:
      toolkit: <the toolkit's command>
      audit: {failures: 2, passes: 1, all_fail_share: 0.25, abstain_share: 0.25, cap_seconds: 600}  # per judge attempt
      reward: learnability
      memory: {past_tasks: 3, exemplars: 2, negatives: 1}
    child_entries: [curated]
    credit: roles
    credit_options:
      by_role:
        generator: {credit: running, options: {strata: [kind, mode, domain], priors: <measured priors>},
                    share: 0.10}
        solver: {credit: keyed, options: {key: [role], leave_one_out: true}}
    max_staleness: 28                     # covers the deadline: ⌈16,200 s / p05(T_step) of 600 s⌉ + 1 (section 9.4)
    workflow_config:
      group_size: 1                       # one task per seed per draw
      rollout_permits: 8                  # sessions open at once; the label cap defaults to 192 // (1 × 8)
      rollout_config: {max_steps: 30, timeout: 16200}  # generator turns; the deadline, the how-to's (open question 9)
trainer_options:
  hooks: [{hook: platoon-envgen-archive, options: {into: <archive directory>, alarms: <thresholds>}}]
```

**10.2 Two LoRA adapters of one base** differ from 10.1 in config only: `train.stepping: independent`; a `generator`
policy with the base's `model_name` on the same URL, so an adapter of the same service with its own Adam state, loss,
bound and version (RFC 0021, section 12), say `{lora_rank: 32, batch_size: 64, max_staleness: 3}` at half of main's
learning rate; `agents.roles.generator: {policy: generator}`; and no `share` in `by_role`: the generator then trains
alone, and loading refuses a share that no role of its policy without one could hold (section 5.2). About 12.8
productions land in each main step, so the generator steps once per about 5 solver steps; the rates stay coupled,
since main's window, (3 + 1) × 16 = 64, is the smaller and gates dispatch (RFC 0021, section 11). The service must
provide RFC 0021's U1-U4, UNVERIFIED for any service until `check_shared_service` (M7) passes on it:
- **The check.** `platoon.testing.check_shared_service(url, base_model=, rank=32, learning_rate=1e-4,
  tolerance=0.02)` opens two adapters, A and B, each in a Tinker session of its own, trains each on token ids of its
  own, and reads an adapter's weights as the mean logprob they give each adapter's tokens, comparing readings of one
  kind only (a training client's `forward`, or a sampler's), so that samplers and trainers that differ by their
  numerics pass. **U1:** the two fresh adapters step together, every `forward_backward` and then every `optim_step`
  submitted before any result is awaited, as a lockstep step submits them, and each moves; then one steps beside the
  other at learning rate 0, and only the one moves, each way round. **U2:** both adapters' sampler weights, saved
  under one name, get two paths, and each kind of sampler tells A from B by at least half of what their weights do.
  **U3:** A's session ends as the Tinker SDK ends one after a fatal error, and A's model stops answering, while B
  reads, steps, samples and saves as before. **U4:** A, saved with its optimizer while B's step is in flight, reopens
  in a session of its own reading as it did at the save, and a step with no gradient, which only the optimizer's
  momentum moves, moves it about as far as the same step moved A.
- It raises one `AssertionError` that names every check it fails with what it saw; a check that raises stops the
  checks after it, listed as not run, since the SDK ends the session of a call that fails. It needs the SDK's
  telemetry, which sends the session-end event that U3 relies on, and refuses to start without it. It checks neither
  that an older sampler keeps serving older weights, which U1-U4 do not ask, nor U10. Its CPU tests run on a toy
  Tinker service of the tests' own, with a fault for each way to break U1-U4, since `FakePolicyService` steps nothing.

**10.3 New names.** All start experimental, except RFC 0016's program options, which M4 built stable:
`program_options`, `eval_overrides.program_options`, `@pt.program(options=, check=)`, `pt.CheckContext`,
`ep.options` and `platoon.testing.run_episode(program_options=)`; the members this RFC adds to `pt.CheckContext`
start experimental.

| Kind | Names (milestone) |
|---|---|
| config keys | `environments[].child_entries` (M9), `workflow_config.rollout_permits` (M8), `environments[].agents.root_role` (M3), `environments[].program_options` and `eval_overrides.program_options`, also in named environments (M4), `service_kwargs.runtime_tasks` (M5; `regrade`, M6), `policies.<id>.hosted` (M12b), `train.step_stall_factor` (M15) |
| component options | `credit: running` (M1: `strata`, `priors` with `mean`, `scale`, `weight` and `strata`, `half_life`, `shrink`, `clip`, `cap_penalties`); `credit: roles` (M2: `by_role.<role>`: `credit`, `options`, `share`, `kl: {to, coef}`); `program: platoon-envgen` (M11: `toolkit`, `solve`, `audit`, `reward`, `reward_options`, `memory`); hook `platoon-envgen-archive` (M12: `into`, `strata`, `keep`, `alarms`) |
| Python | `TrainingSample.version`, `CreditGroup.columns`, `CreditSetup.max_staleness`, `credits_single_members` (M1); `CreditGroup.only` (M2); `@pt.program(root_role=)` (M3); `ep.options`, `@pt.program(options=, check=)`, `pt.CheckContext` and `platoon.testing.run_episode(program_options=)` (M4), and the members of `pt.CheckContext` for entries with children (M9); `HarborEnvironmentService.materialize` (with `replay_address_space_bytes=`), `MaterializedTask`, `RuntimeTasksConfig` (M5); `HarborEnvironmentService.runtime_regrades` (M6); `ReplayChecks`, `ReplayVariant`, `ForbiddenLiterals`, and `materialize(..., replay_checks=, replay_timeout_s=)` (M5b); `ep.rollout`, `ep.inspect`, `ep.drop`, `ep.materialize`, `ChildRollout`, `@pt.program(mints=)`, `@pt.harness(calls_model=)` (M9); `HarnessResult` in `platoon.experimental` (M10); `@pt.program(needs_draws=)`, `StepHookBase.needs_draws`, `ep.rescore`, `ep.child_entry`, `ChildEntry`, `ChildRollout.gateway_errors`, `HarborSession.collect_outputs` and `HarborSession.workdir`, and in `platoon.experimental.envgen` `EnvGenProgram`, `EnvGenOptions` with `AuditOptions` and `MemoryOptions`, `ProductionEvidence` and its records, `Verdict`, `ProductionReward`, `ProductionScore`, `@production_reward`, `RegisteredReward`, `resolve_reward`, `registered_rewards`, the rewards' options, `ToolkitCommand`, `ToolkitResult` and its errors, `check_reward`, `evidence_corpus`, `Production`, `StageBudgets`, `stage_budgets`, `Workspace`, `HarborWorkspace`, `materialize_arguments`, `Archive`, `EmptyArchive`, `PastTask`, `memory_files`, `write_trial` and `atif_trajectory`, and in `platoon.testing` `run_production`, `ProductionScenario` and `ProductionRun` (M11); `ArchiveHook`, `ArchiveOptions`, `AlarmOptions`, `ServedArchive`, `served_archive`, `ARCHIVE_SCHEMA`, `JOURNAL_SCHEMA` and `Production.stored` (M12); `@pt.program(children_with_open_root=)` (M13). `BoundDraw.permits` (M8) is internal |
| components | the estimators `running` and `roles` (M1, M2), core names that a package can no longer register; the program `platoon-envgen` and the rewards `learnability`, `tiers` and `plateau-regret` (M11); the step hook `platoon-envgen-archive` (M12); the hosted frozen-policy runtime (M12b); the control harness `platoon-artifacts` (M16) |
| checks in `platoon.testing` | `check_runtime_tasks` (M5; its regrade case M6, its replay-check cases M5b), `check_shared_service` (M7), `check_program` (M11) |
| metrics | `credit/running/*` (M1); `credit/role_mass/<role>`, `credit/role_weight/<role>`, `credit/role_kl/<role>` and `roles/<role>/*`, ρ_G among them (M2); `rollout_client/start_limit/releases/{ready,child}` and the grant latency (`start_limit/grant_latency_p50` and `_max`), with `child_entries` only (M9); `roles/<role>/stale_share` and the archive's `envgen_archive/*` (M12) |
| protocols and formats | `agents.root_roles` in `identity.json` (M3); the failure kind `program_options`, and a program's options in `lock.json` (M4); the launcher, `tests/gen/grader.json`, the verdict, the report (`platoon.launcher/v1`), the launch settings, the store, the reasons `solver_output_contract` and `solver_nonreproducible`, and the failure kinds `verifier_generated_*` and `verifier_trusted_*` (M5); `verifier_flaky` and the report's `replayed_sha256` (M6); the reasons `solver_variant_failed` and `solver_hardcoded`, and the scan rule `python-numeric-literals/v1` (M5b); `RolloutLease.size` of a draw of several permits per rollout, and platoon's own key of a rollout config's `extra`, `platoon_rollout_permits` (M8); the grant channel, the key `platoon_child_entries`, the trace's `child_rollout` record of a child's root and the failure kind `child_conversion` (M9); the record's `failed_to_open` (M10); `envgen.evidence/v1` and its verdicts, `void(deadline)` and `void(toolkit_usage)` among them, the record's `rescored`, the toolkit's `envgen.<verb>/v1` and its arguments, `export.json` (`envgen.task/v1`), review tasks (`envgen.review-task/v1`) and the trial layout, and, proposed (open question 11), the failure kind `harness_install` (M11); the archive's manifest (`envgen.archive/v1`) and journal (`envgen.journal/v1`), a production's `final-check.json` and `stored.json`, the evidence's `task.instruction` and `task.final_check`, and the final check's `checks`, `digest` and `fingerprint` (M12); rolling child permits (M13); the drain signal (M14) |

### 11. Compatibility, the golden trace and tests

- **Routes travel as `asdict` in every rollout request** (`platoon/train/components.py`), and an entry's `agents` is a
  nested `AgentTreeConfig`, so a new field would appear in every route. The serializer omits `root_role`,
  `child_entries` and `program_options` when unset (M3 and M4 test a route without them for today's keys, order and
  pickled bytes), the worker request's child configs exist only with `child_entries`, and each pull request tests that
  today's requests are unchanged. The defaults are today's: `rollout_permits` 1, root role `policy`, and `single` (the
  default program: one root agent) on today's path.
- **The golden gains one scenario** (M11): productions, each a generator root with K = 3 children of a TOML spec, faked
  at the worker-request boundary as the trace fakes every rollout, under `roles` credit with a frozen judge,
  `rollout_permits` 3, a void production and a resume mid-run; the program itself runs in `run_production`,
  `check_program` and its tests. M12 adds the archive hook to it and records its golden again. The layer rule holds: the
  grant channel lives in `workers/`, given the start limit, and credit reads `TrainingSample.version`. Every built
  milestone keeps the golden trace (`tests/train/parity`) byte-identical.
- **Behaviour changes without a key,** none of which a documented use relies on (each pull request lists its smaller
  changes of messages and warnings):
  - M0: a sandboxed harness's mint of a `judge` agent gets 403;
  - M1, M2: `running` and `roles` are core's names, which a package may no longer register;
  - M3: a run that records its policy table resolves each entry's program when it starts, so one that does not resolve
    stops the start (exit 78) where each of its rollouts failed;
  - M4: a pydantic options schema (a task sampler's, an admission rule's, a credit estimator's or a program's) refuses
    a key it does not take, with a did-you-mean, unless the model keeps extra keys; a dataclass options schema's enum
    and path fields get the member and the path, where they kept the value as written; `train`, `eval` and `rollout`
    resolve the config's components before a `--prepare`, so that their checks refuse before it prepares anything; and
    `platoon.testing.run_episode` raises for a program that does not resolve, where the episode ended as an `error`
    outcome;
  - M5: a task directory that holds a runtime task's launch settings is refused, never run as an ordinary task;
  - M8: a decision over the pool that waits for the rest of a drawn group's permits puts the draw back, first in its
    label's line, when a stop request comes meanwhile, where it started the group; and loading refuses platoon's own
    keys of a rollout config's `extra` in a config's own, `platoon_rollout_permits` and M9's `platoon_child_entries`,
    which no earlier release read;
  - M11: `selfplay` is a reserved trainer name, as `interleaved` is; and, proposed (open question 11), a TOML spec's
    install that fails in a task's sandbox (one of its commands, or a toolchain given as a path that is missing after
    them) is an `error` of the failure kind `harness_install` on any dataset, where it was `harness_setup`, and its
    rollout trains as before: nothing.
- **What an opting-in config changes on purpose:** the solver entry's root takes the role `solver`; `stale_filter:
  datum`, which is run-wide, trains the fresh rows of the solver entry's partly stale groups; with `child_entries` the
  start limit's metrics gain `releases/{ready,child}`, and workers take their slots over the grant channel; and under
  `credit: roles` the step records its metrics by role before its hooks run, and logs each role's `stale_share` (M12).
- **What does not change:** what the `rl` step trains (it logs per-role metrics under `credit: roles`, M2; `rl.needs()`
  gains the OR of declared draw needs, M11), the checkpoint format, the credit contract (once per group, state only in
  `settle`), and Harbor grading of curated datasets.
- **Tests.** Beside `check_runtime_tasks` (section 3.5) and `check_shared_service` (RFC 0021's U1-U4 on a live service),
  `platoon.testing` gains `check_program`: a program against a fake service and model, with its stages' budgets,
  cancellation and the deadline. The red-team suite pairs `check_runtime_tasks`, whose cases hold hardcoded answers
  since M5b, with the toolkit's half (the reference as a helper, a numeric leak, a planted secret, a judge injection, a
  NaN pass) and section 2.5's minting cases, in CI and on a GPU before each study; with the parity test and a CPU
  resume of a fake production, it gates the dry run.

### 12. Not built

| Not built | Why not now | What would bring it |
|---|---|---|
| per-role KL to the starting checkpoint, as a step metric | it needs a scorer of the starting weights, which no run keeps; the step logs per-role entropy and importance ratios instead (section 5.2) | a drift that those miss; then a metric scored against a frozen policy of the base, as `kl.to` is scored |
| replay checks of code other than Python | the forbidden-literal scan is a named rule over Python's parse, which a producer applies to its own reference so that its oracle never fails it | solvers that write another language, with a rule of their own |
| a replay pool of accepted tasks | no evidence yet that replay pays here; SPADE discards its environments and still gains | a study showing it; then an append-only pool fed by the archive, its groups ordinary solver groups |
| exact per-step normalization over the generator's batch (lane normalization) | credit is per group, and batch transforms see one untagged microbatch (`platoon/train/step/step.py`); 16-32 samples a step make noisy statistics | the lag alarm on the generator's mean advantage, or two adapters, as a new objective stage |
| adaptive K in rounds | its optional stopping biases the learnability estimate and the baseline | a study that pre-registers the bias; it overrides `solve` |
| dropping zero-advantage rows from the denominator | they keep their tokens' share by design (`platoon/train/step/step.py`) | a measured case for it |
| lending a parent's permit to its children while the parent's session is open | rolling child permits need no lending when the root's session has closed | a program with children beside an open session (RFC 0020's lending rule) |
| children of children (depth 2) | depth 1 is what the no-deadlock argument of section 4.2 needs | a user and a new argument |
| resumable productions | a reissued draw re-runs its production | crash losses above the triggers of the fallback (Alternatives) |
| rollouts that platoon did not start (sessions of a product's users) | they need a journal and a resolver, not episode trees | a design of their own |
| the age of child rows, as a step metric (section 2.1) | not built with M9: child rows are young | a bound that child rows approach; the trace's `child_rollout` marks let the conversion flag child rows and the step log their ages |
| child rollouts under `platoon eval` and `platoon rollout` | they reserve no rollout permits, so their episodes would have no seats; an entry with children is evaluated during training (`eval.every`), and section 2.7's smoke trains at a learning rate of 0 | the dry run under `platoon eval` (Appendix A.6): seats and child settings per entry, its concurrency counted in permits |
| the archive under a group admission rule | a group that a rule decides as it settles reaches the step without its draw's key, so `StepEnd.consumed` never names it, and the archive refuses such a config (section 9.6) | a step end that names those draws with their fates (RFC 0021's), and an answer for a crash between their settle and the step that takes their markers |
| the alarms that need a signal from outside the run (Appendix A.3) | neither the journal nor the step's metrics hold judge canary agreement, the deep audits, the held-out generated set, red-team regressions, diversity, or censoring correlated with outcomes | evaluations that report them; for diversity, a similarity of tasks, which only the toolkit's fingerprints define |

### 13. Relation to other RFCs

| RFC | What this RFC does to it |
|---|---|
| [0021](0021-trainers-credit-and-policies.md) trainers, credit and policies | **Builds on** its credit API (adds `TrainingSample.version`, `CreditGroup.columns`, `CreditSetup.max_staleness`, `CreditGroup.only` and `credits_single_members`), its estimators (adds `running` and `roles` to core's names), step hooks and draw provenance. **Builds** `check_shared_service` for its section 12's U1-U4 (M7), not the rest of its B4c. **Keeps** its deferral of a framework for producers of online auxiliary data |
| [0020](0020-rollout-permits.md) rollout permits | **Builds** its section 8's static `weight` as `rollout_permits`, without a unit plan (M8: a draw of W permits per rollout holds a lease of `group_size × W` and runs its group size of rollouts, a clause of its lease rule 2), and a dynamic form for children whose root's session has closed, which needs no lending rule (M13); for such episodes its lease's rule 1 reads "the session"; keeps "do not overload `group_size`" and a horizon that never grows silently. Its section 8 points here |
| [0016](0016-component-configuration.md) component configuration | **Builds** its program options (`program_options` and `eval_overrides.program_options`, `@pt.program(options=, check=)`, `ep.options`, `pt.CheckContext`), stable, ahead of the rest of its M2 (M4), and widens `pt.CheckContext` for entries with children only (M9). Its section 3 points here |
| [0015](0015-hosted-models-and-judges-in-eval.md) hosted models | **Amends** it: frozen hosted policies may judge in training; their calls never train. Its status and summary point here |
| [0013](0013-extension-model.md) extension model | **Follows** its discovery rule for envgen's rewards, without a new component kind; programs gain `root_role=`, `mints=`, `children_with_open_root=` and `needs_draws=`, and harnesses `calls_model=` |
| [0017](0017-public-surface-and-conformance.md) public surface | **Follows** it: names start experimental, but RFC 0016's program options, and `platoon.testing` gains `check_runtime_tasks` and `check_program`. Its section 5 points here |
| [0001](0001-native-multi-agent.md) native multi-agent | `running` fills the role of its `rae` preset, as an estimator keyed by role and prompt-side strata instead of `[role, ordinal]`; children join its agent tree as subagents of the root |
| [0007](0007-records-and-atif.md), [0010](0010-harbor-registry-and-multi-container-tasks.md), [0018](0018-configuration-only-projects.md), [0019](0019-training-mixtures.md) | review tasks use one slice of RFC 0007's ATIF projection, without waiting for the rest; runtime tasks are Harbor tasks with root verifiers; a configuration-only project selects the program; `rollout_permits` is a per-entry setting under RFC 0019's rules |

## The hard parts

| Risk | Likelihood / impact | Mitigation |
|---|---|---|
| A child differs from a root of its entry (sampling, budgets, grading), so solver results do not transfer | medium / high | R1; the worker request carries the child entry's own rollout config; the parity test under a scripted model; a GPU smoke with permutation tests |
| Agents minted under a child escape its grade, its exclusion or its role | medium / high | subtrees graded and excluded as units; reserved roles refused with 403; the minting cases in the red-team suite |
| The generator hacks a generated grader, or aims at a frozen judge's blind spots | high / high | trusted graders before generated ones; `materialize`'s refusals; the launcher's replay and replay checks; controls; regrades; judges of another family, with canaries and deep audits by a second one; judges never pay; the red-team suite; alarms that stop the run |
| Generator rows go stale, or the bound masks exactly the positive tail | high / medium | a bound that covers the deadline; `stale_filter: datum`; `cispo`; the masked-row alarm; two adapters as the structural fix |
| Routing replay does not hold for old rows on a mixture-of-experts base | unknown / high | the retention check before the first study on such a base (section 9.4) |
| Permits idle, or trees deadlock on permits | medium / high | static permits first; rolling ones with children first, depth 1, and seat, permit, slot in that order; RFC 0020's trace-driven model before M13; `dispatch/blocked_share/*` |
| Long trees make crashes and stops costly | high / medium | a save every step, drains, the stall watchdog; the derived-draws fallback on measured triggers (Alternatives) |
| Bursts of K sandbox starts and verifier starts overload a node | medium / medium | children take start slots; `ready` releases; regrades after the first grade; verifier starts timed under bursts before the first study |
| A new field changes today's bytes | medium / high | unset fields omitted from routes and requests; the new release causes and the grant channel only with `child_entries`; the golden trace in every pull request |
| One policy's roles interfere | medium / medium | a share held in band, logged per role; a pre-registered comparison against two adapters |

## Alternatives

- **A `selfplay` trainer.** A trainer owns the step, and what is online here is the rollouts, rewards and credit. An
  `rl` subclass "never overrides `run_steps` or a private method" (RFC 0021, section 3); a sibling trainer must port
  evaluation during training, admission, independent stepping and multi-LoRA, and RFC 0021 exists because a copied
  trainer drifted. Synchronous rounds would also bake one schedule into the step, credit groups twice, and train
  generator rows many rounds old. The name stays reserved (M11) for barrier algorithms built later on these pieces.
- **Derived draws with pending parents, the fallback.** The K solves become an ordinary group of a derived-only entry,
  drawn by the parent's settle. It wins on parity by construction, exact permits and start slots, per-group isolation
  and parents that survive a crash; it costs EST 1,500 lines in the dispatcher, its codec and resume, and keeps
  "nothing trains before the audits" only if the solve group is credited at the family's resolution. Switch only on a
  measured trigger at target scale, after section 4.2: permits blocking over half the time while sandboxes idle,
  crash and stop losses above 5% of a day's session-hours that the fallback would halve, or worker deaths and deadline
  hits losing over 1% of productions.
- **A live task pool.** Generated tasks join a pool that the dispatcher samples. Its first measuring group trains
  before the audits land, a late veto by construction; its gate runs inside a verifier, with no fresh controls; K
  becomes a task-sampler setting; and its core surface is the widest (EST 2,300 lines). Replay may come this way later.
- **No change to platoon:** solvers as function harnesses on `ep.session.fork("fresh", task=...)`. A fresh fork opens
  without `SessionNeeds` (`platoon/envs/harbor/session.py`), so the solver is not the solver entry's route.
- **Doing nothing.** Each project writes its own workflow or trainer, reusing neither the solver's route nor credit.

## Rollout plan

Each milestone is one pull request that keeps the golden trace identical, in this order: M0 first; M1, then M2, and M3,
then M4, with M7 in any order; M5 → M6 → M5b, which the measurement needs; M8 → M9 → M10 → M11 → M12, and M12b if
open question 1 chooses a hosted judge; then M13-M17. M0 to M7 and M5b are built, in the pull requests the last column
names. M8 to M12 are built too, each as one commit on the one before, and their pull requests open once those of M0 to
M7 and M5b have merged. Permits land before children, so no merged pull request runs children outside the permit and
start limits. Lines are EST, code + tests; the rows of the built milestones also give the lines they added, three to
nine times their estimates, so the remaining estimates are likely low too. Appendix A.6 lists the measurement, the dry
run and the studies that set envgen's defaults; the routing-replay check of section 9.4 comes before the first study on
a mixture-of-experts base.

| Milestone | Scope | Exit check | Pull requests |
|---|---|---|---|
| M0 judge mints | section 2.5's first refusal (EST 10 + 50; built 21 + 197) | a `policy_only` agent's mint of `judge` gets 403, and so does one from an agent it minted; nothing is minted or counted; its other mints still work; the red-team case end to end through the gateway | "fix: Refuse judge agents minted by sandboxed harnesses", open |
| M1 `running` | section 5.1, with `TrainingSample.version`, `CreditGroup.columns`, `CreditSetup.max_staleness` and `credits_single_members` (EST 300 + 350; built 854 + 1,127) | `check_credit_estimator` passes; a stratum column from a service that is not Harbor; `version` under both stale filters; the group-of-one warning skipped only for such estimators; a config without a staleness bound refused; resume between `assign` and `settle` | "feat: Add the running credit estimator", open |
| M2 `roles` | section 5.2 (EST 300 + 350; built 1,103 + 1,596) | shares hold in a fake run as validity rises; a weight read at the sampling version keeps the outcomes' sign; per-role `kl`; exact resume; the envgen entry of section 10.1 loads with no group-of-one warning | "feat: Add the roles credit composite", open, on M1's |
| M3 root roles | section 6.1 (EST 150 + 200; built 386 + 859) | a role-bound root samples its policy for life; a frozen root; `verifier` and `judge` roots refused; TOML entries load with `root_role`; a resume refuses a changed root role | "feat: Add a root role per environment entry", open |
| M4 program options | section 6.2's options and checks (EST 150 + 200; built 631 + 1,119) | options checked at load and in `platoon config validate`, each problem where it is written; `ep.options` in the worker; a check sees only its entry and the phase's rollout config | "feat: Add program options", open, on M3's |
| M5 runtime tasks | section 3 but `ep.materialize`, regrades and replay checks (EST 400 + 450; built 2,886 + 2,337) | `check_runtime_tasks` refuses or grades every sabotage as listed but the flaky grader, which M6's regrade catches, end to end on Singularity with root verifiers; curated grading unchanged | "feat: Add runtime Harbor tasks", open |
| M6 regrades | section 3.3 (EST 120 + 150; built 276 + 578) | a flaky fake grader is `verifier_flaky`; an unseeded submission whose replays differ is the solver's 0, never `verifier_flaky`; curated tasks are never regraded | "feat: Add regrades of generated graders", open, on M5's |
| M5b replay checks | section 3.4 (built 1,575 + 1,830) | the honest submission scores 1, and its regrade replays the variant again; one that ignores its inputs is `solver_variant_failed`, one that states the answer `solver_hardcoded`; unusable check data is the grader owner's; a variant's replay reaches neither the replay's outputs nor the grader's files, on Singularity | "feat: Add replay checks to runtime tasks", open, on M6's |
| M7 shared-service check | `check_shared_service` for RFC 0021's U1-U4 (EST 150 + 150; built 497 + 529) | passes on a toy service that keeps two adapters apart, and each of its faults fails its own check; run on a live LoRA service before a two-adapter study | "feat: Add a shared-service check for adapters", open |
| M8 static permits | section 4.1 (EST 200 + 250; built 691 + 1,870) | a draw holds `group_size × W` permits whole until its group returns; in a randomized fake under a full pool, children that take their seat, then their start slot, never deadlock, while a child that takes its slot first among them can hold the start limit until a slot's timer; the derived label cap | "feat: Reserve rollout permits for episodes with several sessions", one commit, not open yet |
| M9 child rollouts | sections 2.1-2.4, 2.5's second refusal and `ep.materialize`, for function harnesses (EST 450 + 500; built 2,629 + 2,932) | children train in their episode's group; a subagent a solver child mints trains on that child's outcome; inspections and their subtrees never train; a solver child's mints of `judge` and `generator` get 403; a check of an entry with children sees them, and other checks see only their entry; start slots and the grant channel under process isolation; without `child_entries`, no new metric and no channel | "feat: Run rollouts of other entries from programs", one commit on M8's, not open yet |
| M10 CLI children | sections 2.6 and 2.7 (EST 200 + 300; built 463 + 2,079) | the datum-parity test of section 2.7, for a CLI harness and for function harnesses that mint a subagent; a CLI child's gateway server takes only its subtree's keys, which end at its grade; its GPU smoke, whose script has not run (it needs a GPU service) | "feat: Run CLI harness children under programs", one commit on M9's, not open yet |
| M11 the envgen program | section 9 except the archive, with `needs_draws`, the rewards, `ToolkitCommand` and `selfplay` reserved (EST 600 + 650; built 5,394 + 6,384, 3,260 of them its recorded golden trace) | the golden scenario, with a resume; `check_program` and `check_reward` pass; additive evidence; a censored member enters no baseline; a generator that uses its whole budget beside graders that never finish, under the smallest deadline loading admits, is the generator's fault, never the deadline's | "feat: Add the envgen program", one commit on M10's, not open yet |
| M12 the archive | section 9.6, with the garbage collection of section 8.3 and the alarms of Appendix A.3 (EST 350 + 400; built 2,316 + 2,105) | alarms stop a fake run through `run.shutdown`; a saved journal length never points past durable bytes after a kill; the golden scenario with the hook, recorded again | "feat: Add the envgen archive", one commit on M11's, not open yet |
| M12b frozen hosted policies | section 7 (200 + 250) | a `platoon-react` judge on a hosted frozen policy inside a Tinker gateway, and a judge canary entry's root on it; a trainable hosted policy, one that `judge` does not sample, and one bound to a role that is neither `judge` nor a root role, refused | |
| M13 rolling child permits | section 4.2 (250 + 350), after RFC 0020's trace-driven model replays the measured latencies | children first and depth 1; in a randomized fake, seat, then permit, then slot never deadlocks, and a child that takes its slot first does; `ep.rollout` raises while the root's session is open, unless the program keeps static permits | |
| M14 drains | section 8.1 (150 + 200) | on the signal no new production starts, and those in flight settle | |
| M15 step-stall watchdog | section 8.2 (110 + 180) | the three tests of section 8.2 | |
| M16 artifacts harness | `platoon-artifacts`: mutants and metamorphic variants as controls (200 + 250) | each mutant fails the oracle's grader; before the first generated-grader study | |
| M17 CLI roots | section 2.6's roots (150 + 200) | a CLI root under a program; before tasks with generated graders authored by a CLI root | |

## Open questions

1. **The judge's model and budget.** Proposed: a pinned hosted model through M12b if its egress succeeds 99% of the time
   over 24 h, else a frozen Tinker judge of another family; deep audits use a second family. Decide before the dry run.
2. **Training mode at scale.** Proposed: LoRA in both roles, since a full fine-tuning service holds one model and so
   removes the switch to two adapters. Decide before study 2.
3. **Paying tractable all-fail tasks.** Proposed: not in v1; they score 0 and join a reserve kept for the dataset
   export, so that no reward pays for tasks nobody can solve. Decide before study 2.
4. **`unresolved` audits.** Proposed: −1, and 0 if over 10% of honest tasks in the dry run come back unresolved at the
   online cap. Decide after the dry run.
5. **What an exploit alarm does.** Proposed: stop the run for a runbook (retire the lineage from the archive and the
   memory, add it to the red-team suite, resume once it passes, roll back before its first admission if solving
   regressed); negative-only generator credit is the later, automatic alternative. Decide before study 1.
6. **Hosted policies beyond judging.** A frozen hosted generator is a natural baseline for study 2, but section 7's
   amendment covers judging only. Proposed: extend it to any role only if study 2 keeps that arm. Decide before study 2.
7. **Promotion.** Proposed: `root_role` first, then `running` and `roles`, each after a release in which an outside
   project used it; until then RFC 0017's rule holds (experimental names change only in a minor release).
8. **Sandboxed mints of `verifier`.** M0 refuses only `judge`. M9 reserves roles only under a program that declares
   `mints=`, and `mints=` refuses `verifier`, the role of the agents platoon opens to verify children, so no program can
   reserve it either: a sandboxed harness still mints a `verifier`, which samples `agents.roles.verifier`'s policy.
   Proposed: refuse `verifier` from a `policy_only` parent as M0 refuses `judge`, in a fix of its own, since platoon
   opens its verifiers itself. Decide before M9's pull request opens.
9. **The production deadline.** M11's stage budgets are worst cases that count every retry and canary (section 9.1):
   with section 10.1's shape and a 4-minute verifier they come to 12,360 s after generation, where this RFC first priced
   69 minutes with generation, so its 90-minute deadline is refused (Appendix A.1). The how-to sets 16,200 s, which
   leaves a 20-minute generation and about 44 minutes of queueing, with an envgen bound of 28. Of the 12,360 s, the
   audit's second round, for the failures a review calls infrastructure's, takes 2,040. Options: keep the worst cases,
   with the how-to's deadline and bound, under which generator rows up to 28 versions old train and a production held up
   to its deadline keeps its permits for 4.5 hours; or make the second audit round cheaper, which takes up to 2,040 s
   off the sum, and the deadline and the bound with it. Proposed: run the dry run at the how-to's deadline and bound,
   which measures how far below the worst case productions land, and choose with its latencies. Decide before study 1.
10. **The refill window.** M12 reads the window at each step end while it is open, so that it grows while slower mixed
    productions land after the fast ones, and keeps it closed once it closes, for the rest of the start (section 9.6).
    The alternative fixes it at its first value, which the first, fastest mixed productions set: the validity alarm, a
    stop, and the share warning then go live on a known schedule, where M12's window stays open, and those alarms blind,
    for as long as slower productions keep raising it. Proposed: choose with the dry run's step durations and latencies,
    which show how far the window grows within a start. Decide before study 1.
11. **The failure kind `harness_install`.** M11 gives a TOML spec's install that fails in a task's sandbox, among the
    task's own files (one of its commands, or a toolchain given as a path that is missing after them), a failure kind of
    its own, `harness_install`, where it was `harness_setup`, so that envgen can censor what only the platform causes
    and leave to rule 2 what the task's files can break (section 9.2). The change reaches every dataset, not envgen
    alone: such a rollout is still an `error` that trains nothing, but a dashboard or a filter that counts
    `harness_setup` finds those failures under the new kind. Options: keep the kind, as built; or keep `harness_setup`
    for every setup failure and mark an install among the task's files in the outcome's metadata, which envgen reads, so
    that no user sees a change, while the failure kind goes on calling the platform's what the task's image or files may
    have caused. Proposed: keep the kind, which tells a user of any dataset where to look (the task's image or files,
    not the platform), and which M11's changelog announces. Decide before M11's pull request opens.
12. **An install that infrastructure breaks on one node.** The grading canary runs the oracle, which never installs the
    solver's harness, so it cannot fail for what breaks an install: a `harness_install` that reproduces beside a healthy
    canary is the generator's, `invalid(harness_install)` at a reward of −1, also where infrastructure broke both
    attempts on the worker's node (a full disk, say), where the retry and the canary run too. A break on every node is
    loud, as a toolkit verb that crashes on every input is (rule 2): every production's validity drops, and the validity
    alarm stops the run. A break on one node of many is quiet: validity drops by that node's share. Options: name the
    trade, as section 9.2 does and M11 builds; or split further, with a canary of the install's own, the solver's
    harness installed on the grading canary's task, on the same service and image and among files no generator wrote, so
    that a `harness_install` that reproduces beside a failing install canary is infrastructure's and censored. That
    canary runs only where an install reproduces, once per stage, and the solve stage's budget gains a sandbox start and
    the install's time limit, which the deadline must cover (open question 9). Proposed: split further before study 2,
    at scale; on study 1's one node a lasting break fails every production, which the validity alarm stops. Decide
    before study 2.

## Appendix A: envgen's estimates and defaults

Every number here is EST unless it says otherwise; options hold the defaults, and the phases of A.6 replace them.

**A.1 Budgets and the deadline.** The budgets are worst cases that count every retry and canary attribution may run
(section 9.1). For section 10.1's short productions, with a 4-minute verifier (the template's `[verifier] timeout_sec`)
and one regrade, they come to 12,360 s after generation: package 480 s, gate 3,015 s, solve 4,005 s, audit 4,800 s and a
grace of 60 s. A toolkit verb counts 360 s (its run, its retry and the toolkit's canary, 2 minutes each), a grade and
its regrade 600 s (a sandbox start of 1 minute and the verifier, twice), a control 885 s (a start, a quarter of the
solver's 15 minutes and its grades), a solve 1,560 s and a review 1,680 s (its export and two judge attempts of a start
and 10 minutes). The gate counts the final check, its 6 controls twice, for the retries, and the grading canary, and the
solve its K solves twice and the canary, each in one wave, since W = 8 holds them; the audit counts K reviews,
`judge-spec`, `adjudicate`, the judge canary and a second round of 2,040 s for the failures a review calls
infrastructure's. With a 20-minute generation that is 13,560 s, 226 minutes, where this appendix first priced 69 minutes
(generation 20, package 2, gate 8, solve 19, audit 10 and one judge retry 10), so loading refuses the 90-minute deadline
(`timeout: 5400`). The how-to sets 16,200 s, which leaves about 44 minutes for queueing, and an envgen bound of 28,
which covers it at a step every 10 minutes at the fastest (section 9.4); without its regrade the sum would be 10,560 s,
and with a template that gives the verifier no time limit, so that it takes 10 minutes, 16,680 s, past the how-to's
deadline. Open question 9 weighs that deadline against a cheaper second audit round. A valid mixed production still
takes 30-50 min when short, and 1.5-3.5 h with one or two hours of authoring: the worst case sets the deadline and the
bound, not the latency. The solve's 19 minutes leave its verifier about 4, which must hold the replay and each variant's
at the producer's replay time (`replay_timeout_s`), the grading and the launcher's own work: about 37 s a replay with 3
variants. The measurement sets the replay time and the verifier's time limit together; a longer verifier raises the gate
and solve budgets, and the deadline with them. A task whose variants leave a target uncovered audits every pass
(section 9.5), up to K pass reviews where a mixed task has one and an all-pass task none, which the audit's budget
prices by counting K reviews; the measurement counts such tasks and what their audits cost.

**A.2 Utilization and sizing.** Static permits keep about 55% of the permits they hold busy for half-hour productions,
and 28-38% with one or two hours of authoring, against 90-95% under rolling child permits. Under rolling permits,
size the label cap by Little's law, `cap ≈ λ × p50(L)` for a target rate λ of productions, and keep `(k + 1) × B ≥
cap + B + the other groups in flight and settled`, by raising B rather than k: a larger B also lowers the generator
rows' age in versions, about `L × λ / B`. Section 10.1's shape (B = 16, P = 192, W = 8) has a cap of 24, a step about
every 14 min, the how-to's 270-minute deadline (open question 9) and an envgen bound of 28.

**A.3 Alarms.** The share and validity alarms skip the first ⌈p50(L_mixed) / T_step⌉ steps after a start, L_mixed
being a mixed production's latency, while batches hold only the solver entry's curated groups and fast failures; M12
reads that window at each step end until it closes (section 9.6, open question 10).

| Signal | Alarm | Action |
|---|---|---|
| exploit kinds over the last 100 or more productions (flaky, lenient, brittle, false negative, hackable, leaky, contaminated) | the one-sided 90% lower bound above max(2 × the measured base rate, 5%) | stop |
| contamination; a red-team regression | any | stop |
| deep audits: the online audits' miss rate; the consequential flag's precision | the miss rate's 90% lower bound above 1.5 × its base rate; precision below 0.7 | stop |
| validity | below 50% for 3 steps after step 10, or 20 points below its maximum | stop |
| judge canaries; judge outages | agreement below 90%; 10 outage-censored productions in a row | stop |
| a held-out generated set (accepted tasks on seeds never trained on) | more than 2 standard errors below its first 3 evaluations, twice | stop |
| censoring (every void production) | above 5% of productions, or correlated with outcomes | investigate |
| ρ_G; the per-step mean of the generator's advantages | outside [0.03, 0.30]; above 0.5 in magnitude for 5 steps | review the share |
| masked generator rows; the generator's mean \|log r\| | above 1% over 10 steps; above 2 × the solver's | raise the bound; consider two adapters |
| diversity (Vendi score per task); the top domain's share | below half its value at the start; above 50% | turn on the generator's `kl` |

M12 builds what the journal and the step's metrics hold, each alarm with an action of `stop`, `warn` (the table's other
actions) or `off`, and adds `toolkit_usage`, which stops the run on a production that ended on a usage error of the
toolkit's (section 9.7). Its windows: the exploit kinds over the last 200 measured productions, at least 100 (flaky
`verifier_flaky`, lenient `noop_passed`, brittle `oracle_disagrees`, false negative `false_negative`, hackable
`task_exploit`, and the final check's `leak` and `contamination`), against a base rate measured before training
(`exploits.base_rate`); validity over the last 32 measured productions (a void one is not measured), counted at the step
ends from step 10 that journaled one, and 20 points below its most over full windows; judge outages, productions voided
by a review that could not be had or a failing judge canary; censoring (`censoring`, its signal `censored_share`), every
void production whatever voided it, over the last 100, at least 20, whose line, when it holds, counts each kind; the top
domain's share of the last 100 valid productions, at least 20; and masked generator rows from the step's
`roles/<role>/stale_share`. A production that left no evidence counts as `void(no_evidence)`. The generator's alarms
read its rows where they train (section 9.6): its masked rows and its mean |log r| at its policy's own step ends when
that policy steps on a clock of its own, the latter against the solver's last; ρ_G only where one trained policy samples
both roles, and off elsewhere; and a solver on a clock of its own beside a generator on the leader leaves the
generator's mean |log r| no solver ratio to compare. Not built, since neither the journal nor the step's metrics hold
them, each needing a signal from an evaluation or from outside the run (section 12): judge canary agreement, the deep
audits, the held-out generated set, red-team regressions, diversity, and censoring correlated with outcomes.

**A.4 Judges.** The judge canary set holds 30 or more items with known verdicts (false negatives, clean
failures, hacked and clean passes, leaky tasks), and `eval.every` runs it about every 6 h. Deep audits re-audit a
uniform 5% of the audited samples, drawn by the draw key, with a full budget and a judge of a second family.

**A.5 Crashes.** With a save every step, a crash costs half a step, the productions in flight (whose mean age,
E[L²] / (2E[L]), exceeds L / 2) and the restart: about 33 min for half-hour productions, 2% of a day, and 1.9 h, 8%,
with two hours of authoring. Study 1 measures how much larger kept trees make a save.

**A.6 Validation.**

| Phase | What runs | What it sets or decides |
|---|---|---|
| Measure (after M5, M6 and M5b; before M8) | tasks the toolkit exports, materialized with platoon's runtime tasks as the program will (a script maps each `export.json` onto `materialize` of the solver entry's own service, with an explicit `runtime_tasks.store`, after the toolkit's final check passes the task) into that store, then run with `platoon eval` as the entry's tasks (`task_dirs`): controls, 16 solves each, audits as review tasks; no exported task ships a launcher | K's reliability (split-half), latencies and flakes under bursts of K, verifier starts, the replay time and the verifier's time limit, the share of tasks that audit every pass, judge egress, the held-out generated set (300 or more accepted tasks on seeds never trained on) |
| Dry run (after M12, no training) | `platoon-envgen` with the base model as generator, under `platoon eval` once it runs children (section 12), or until then under `platoon train` at a learning rate of 0, as section 2.7's smoke runs | `running`'s priors and the roles' masses, against which the share is set, the alarms' base rates, the unresolved share, the deadline, W and the bound; a supervised cold start if validity is below 0.6 or the informative share below 0.15 |
| Study 1, one node | a small model, short productions, static permits | first the invariants (parity, ρ_G in band, masked generator rows below 1%, exact resume), then generator RL against none (the generator left out of `by_role`, whose rows then train at 0) on a paired evaluation with a fixed solver |
| Study 2, at scale, after M13-M15 | the target model and longer productions | keep generator RL only if it is non-inferior on the held-out generated set (a margin of 0.02) and clears study 1's rule on the paired evaluation; otherwise a frozen generator plus curation is the result |
| Study 3, after M16-M17 | tasks with generated graders, authored by a CLI root | the exploit alarms decide whether it continues |
