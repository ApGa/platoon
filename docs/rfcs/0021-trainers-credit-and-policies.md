# RFC 0021: Trainers, credit and policies

| | |
|---|---|
| Status | Draft |
| Created | 2026-10-26 |
| Authors | Apurva Gandhi (@ApGa) |
| Builds on | [RFC 0003](0003-multi-policy-training.md) (multi-policy training: this RFC implements its milestones M1, M2 and M4, with the changes listed in section 19), [RFC 0001](0001-native-multi-agent.md) (per-agent credit, its section 7, which this RFC replaces in part), [RFC 0013](0013-extension-model.md) (one discovery rule), [RFC 0016](0016-component-configuration.md) (a selector and its options), [RFC 0017](0017-public-surface-and-conformance.md) (`platoon.experimental`, conformance kits), [RFC 0019](0019-training-mixtures.md) (the dispatcher, per-environment settings, exact resume), [RFC 0020](0020-rollout-permits.md) (rollout permits, per-datum stale masks); the trainer (`platoon/train/tinker/rl.py`), the group workflow (`platoon/train/tinker/workflows/group_rollout_workflow.py`), the gateway (`platoon/gateway/core.py`, `platoon/train/tinker/proxy.py`), the rollout workers (`platoon/train/tinker/workers/`) and the golden trace of the training loop (`tests/train/parity/`) |
| Discussion | The pull request that adds this RFC. The design was agreed on 2026-10-26 (open questions 1 to 10); open questions 11 to 17 propose answers |
| Implementation | None yet. It starts after the reorganization of `platoon/train` that is in review (section 17) |

## Summary

Platoon trains one way today: a GRPO-style trainer, one policy on one Tinker service, and credit computed inside the
group workflow. This RFC splits `platoon/train` along the lines that users want to vary, and lets one run train
several models.

- **Trainers.** A trainer is a component chosen by `trainer:`: `rl` (today's trainer, the default), `sft`,
  `interleaved`, or a user's class, resolved like any other component (RFC 0013). Every trainer runs through one
  lifecycle, `TrainerRun`, which keeps today's order: signals, start point, service connections, the launch save, the
  step and evaluation loops, saves and teardown. A trainer writes `prepare` and `run_steps`.
- **Rollouts for any trainer.** The dispatcher, rollout permits, samplers, admission, per-task statistics and exact
  resume move off the RL trainer into a `RolloutSource`, from which any trainer takes settled groups.
- **Step hooks.** Work at the end of any trainer's step, after its own updates and before it publishes: drain side
  inference, add optimizer updates on the same training client and Adam state, and hear the publication. An outside
  package's interleaved trainer, which copied about 1,000 lines of the trainer to do this, needs nothing more.
- **Credit.** A `CreditEstimator` runs once per group, asynchronously, over every agent of every member, before the rows
  split by policy. It is chosen per environment entry (`environments[].credit`). The built-ins are `group` (today's),
  `keyed` and `reverse_kl` (on-policy distillation).
- **Objectives and learners.** An `Objective` is a policy's loss plus the counting, filtering and scaling a step does
  around it (`PolicyGradient`, `CrossEntropy`). A `Learner` updates one policy with it.
- **Policies.** One `Policy` per model, each with its own Tinker service URL and session, optimizer, loss, version,
  staleness bound, checkpoints and metrics. One trainer process drives all of them, so a group keeps one dispatcher
  and one credit computation over the whole episode. Several policies step together (`train.stepping: lockstep`, the
  default) or each on its own clock (`independent`), on separate services or as LoRA adapters of one base model behind
  one URL.
- **Compatibility.** Today's GRPO run, with no new key, stays byte-identical, and the golden trace gates every pull
  request. No config key is renamed, no resume format changes, and the package exports stay.

## Motivation

These are the cases users have asked for. The trainer serves the last one only.

| | Scenario | What stops it today |
|---|---|---|
| S1 | SFT on a dataset | the only trainer is the RL trainer: its step needs `mask` and `advantages`, its counting assumes rollouts, and its pruner keeps recovery checkpoints for samplers an SFT run never has |
| S2 | on-policy distillation: the student samples, a frozen teacher scores | a run holds one model, and credit cannot call another |
| S3 | RL interleaved with SFT or next-token prediction on one policy | auxiliary updates have no place in the step, and a weight change that is not published breaks the staleness count |
| S4 | a trainer written outside platoon: DPO, a REINFORCE variant, an interleaved trainer | rollouts, saves, resume and evaluation are private methods of the RL trainer. An outside package's interleaved trainer copied about 1,000 lines of it and pinned platoon 0.2.1 |
| S5 | PPO with a critic | credit is synchronous and stateless, has no second model, and cannot emit rows for one |
| S6 | a user's own credit for multi-agent rollouts | credit is `compute_baselines` and `assign_advantages` on `GroupRolloutWorkflow`: one number per member, reached through a subclass, with no view of agents or roles |
| S7 | several models in one run: a 35B orchestrator on one Tinker service, 4B workers on another, a frozen judge on a third, each service on its own nodes with its own URL | the gateway holds one sampler and renderer, the trainer one training client and one version counter, and every row trains one policy (RFC 0003) |
| S8 | today's GRPO runs | they must not change at all |

The code is built around S8:
- `PlatoonTinkerRLTrainer` (`platoon/train/tinker/rl.py`, about 3,000 lines) holds the lifecycle, rollout dispatch,
  the step, publishing and checkpoints in one class. The reorganization in review splits the step into stages and the
  long functions into phases, and keeps one trainer.
- `RolloutWorkflow` (`workflows/base.py`) is one method. The trainer reads nine more attributes of a workflow's output
  by duck typing, and tests `isinstance(..., GroupRolloutWorkflow)` before some features.
- `trainer_config` subclasses exist only to check options, which RFC 0016 deprecates.
- RFC 0003 designed multi-policy training, and RFC 0001 per-agent credit, as extensions of this one trainer. Building
  either inside it would grow the class further and still leave S1 to S5 out.

What the cases share is a need to separate what varies (the trainer, the workflow, the credit, the loss and the
policy) from what every run needs (launch, supervise, saves, resume, evaluation and metrics), so that a user can
change one of them without copying the others.

## Proposal

The sketches are Python 3.12. Modules that `platoon.experimental` re-exports import `tinker`, `torch` and `datasets`
only under `TYPE_CHECKING`. Paths are those of the layout in section 15. Every new API starts in
`platoon.experimental`.

### 1. Rules

1. **One model, one `Policy`.** A policy has a validated string id (`[a-z][a-z0-9_-]{0,31}`), never an index. `main`
   is the default id, the same object as the per-task statistics' `MAIN`, so their snapshots keep their format.
2. **One agent, one policy, for life.** The config binds roles to policies, and a request never picks a policy. Only
   text crosses policies, so each agent has one tokenizer (RFC 0003, section 2).
3. **Every row names its policy.** In a run with several policies, a row without a policy, or with an unknown one, is
   an error, never a silent drop. The tags never reach the Tinker service.
4. **Credit once, then route.** Credit sees every agent of every member. Only then do the rows split by policy.
5. **Per policy:** the service URL, Tinker session and clients, optimizer, loss, version, staleness bound, checkpoint
   rows, pruner and metrics. **Per run:** one dispatcher, one pool of rollout permits, and one commit record per full
   save. In lockstep the run also has one step clock and one dispatch horizon; under independent stepping each
   trainable policy has its own clock, batch and horizon.
6. **A version counts published samplers, and every weight change of a policy that rollouts sample is published at
   the end of the step that made it.** A step may hold several optimizer updates (its minibatches, and the updates its
   step hooks add), and it still publishes once, so "how many samplers ago" stays "how many steps off-policy". A
   resumed version is never below one that a kept row carries. After a crash a version may go back to its row's, as
   today: the samplers published after the row are lost, and their numbers are reused.
7. **Generic code never imports a trainer, a workflow or a credit estimator.** A test enforces it (section 15).
8. **The extension rules:** a Protocol for what platoon calls, plus an optional base class with defaults; components
   resolve by name in every process (RFC 0013); options sit next to their selector (RFC 0016); new API starts in
   `platoon.experimental` with no `tinker` import, with a check in `platoon.testing` (RFC 0017).
9. **One policy writes today's bytes:** files, `loop_state` keys and their order, rollout requests (worker protocol 2),
   metric names, message texts and the order of Tinker calls. A new file, key, request field or metric prefix appears
   only when a config sets a new key.
10. **A step reaches a checkpoint whole or not at all, and a bad update stops it.** Once a step has submitted
    training work, an interrupt saves no `<step>_interrupted` checkpoint, and the resume starts from the last full one.
    An optimizer result that is `skipped` or not finite raises before anything publishes. These are the defaults of
    `TrainerNeeds` (section 3). `rl` turns both off, which is today's behaviour, so S8 is unchanged (open question 8).

### 2. Policies (`platoon/train/policy/`)

```python
PolicyId = str
MAIN: Final[PolicyId] = "main"


@dataclass(frozen=True)
class PolicySpec:
    """One policy as the config declares it. Every field has today's train.* name, so there is one vocabulary."""

    id: PolicyId
    model_name: str
    renderer_name: str
    renderer_kwargs: Mapping[str, Any] = field(default_factory=dict)
    context_window_length: int | None = None
    sample_timeout_seconds: float | None = None
    tinker_base_url: str | None = None      # None: the run's tinker_base_url, else TINKER_BASE_URL, else hosted
    trainable: bool = True                  # False: frozen; sampled or scoring, never stepped, saved or pruned
    lora_rank: int = 32
    load_checkpoint_path: str | None = None # trainable: weights to start from; frozen: the sampler to serve
    load_optimizer_state: bool = False      # with load_checkpoint_path: Adam's state too (a new run that keeps it)
    optimizer: "AdamParams | None" = None   # None: train.optimizer
    loss_fn: str | None = None              # None: train.loss_fn
    loss_fn_config: Mapping[str, Any] | None = None
    max_staleness: int | None | Unset = UNSET   # unset: the entry's bound (RFC 0019), else train.max_staleness
    batch_size: int | None = None           # independent stepping, a non-leader policy: shares per step (section 11)
    checkpoint: "CheckpointCadence | None" = None   # the same: {every, recovery_every} in its own steps


class Policy:
    """One policy in the trainer process. Rollout workers never hold one: they follow its binding (section 13)."""

    id: PolicyId
    spec: PolicySpec
    info: "ModelInfo"                         # renderer, tokenizer, the gateway runtime's sampler
    training_client: "tinker.TrainingClient | None"    # trainable only
    sampling_client: "tinker.SamplingClient | None"
    optimizer_params: "tinker.AdamParams"
    optimizer_steps: int
    checkpoint_dir: str                       # the run directory for the leader, <run>/policies/<id> for the others

    @property
    def version(self) -> int: ...             # today's counter, one per policy
    def connect(self) -> None: ...            # its own ServiceClient, so its own Tinker session, also on a shared URL
    async def open(self, start: "PolicyStart", *, stop: asyncio.Event) -> bool: ...
        # resume from its row (weights and optimizer), else load_checkpoint_path, else new; today's retries
    async def save(self, name: str, *, loop_state: Mapping[str, Any], wait: "Waiter") -> "SavedPaths": ...
    async def save_for_sampler(self) -> "tinker.SamplingClient": ...
    def publish(self, sampling_client: "tinker.SamplingClient", *, new_version: bool = True) -> int: ...
        # today's order: the gateway runtime's sampler, then the version, then each subscriber
    def subscribe(self, write_binding: Callable[["tinker.SamplingClient", int], None]) -> None: ...
    async def score(self, datums: Sequence["tinker.Datum"]) -> list["torch.Tensor"]: ...   # a teacher's logprobs
    def snapshot(self) -> "PolicySnapshot": ...


@dataclass(frozen=True)
class PolicySnapshot:
    """The policy's published sampler at one version, for inference outside the rollouts (a trainer's own writers,
    teachers and judges): nothing it samples is recorded or trained."""

    policy: PolicyId
    version: int
    sampling_client: "tinker.SamplingClient"
    renderer: "Renderer"
    tokenizer: "Tokenizer"
    def current(self) -> bool: ...            # no publication since
    async def sample(self, prompt, params) -> "tinker.SampleResponse": ...
        # refused with PolicyMoved when the policy published before the call or by the time it returned
    def gateway(self, *, context_window_length: int | None = None) -> "ModelGateway": ...


class PolicySet(Mapping[PolicyId, Policy]):
    """The run's policies, in config order."""

    leader: PolicyId                          # the first trainable policy (main whenever main trains)
    agents: "AgentBindings"                   # which policy each agent role samples
    services: Mapping[str, tuple[PolicyId, ...]]   # normalized URL -> its tenants, in config order (section 12)
    def trainable(self) -> list[Policy]: ...
    def sampled(self) -> list[Policy]: ...    # agents map some role to it, so it publishes samplers and has a bound
    def versions(self) -> dict[PolicyId, int]: ...   # {main: v} with one policy: the dict the dispatcher reads today
```

- **Which policies.** With `train.model_name` set, the first policy is `main`, built from `train.*` and the top-level
  `tinker_base_url` exactly as today; then come `policies.<id>` in config order, whose unset optimizer, loss and loss
  options come from `train.*`. `policies.main` beside `train.model_name` is a configuration error (exit 78). Without
  `train.model_name`, `policies:` declares every policy.
- **How a policy finds its service.** The URL is data: `policies.<id>.tinker_base_url`, or for `main` the top-level
  `tinker_base_url`. A job script that starts one Tinker-compatible service per policy passes the URLs as overrides,
  for example `policies.worker.tinker_base_url=http://$WORKER_HOST:8000`. Platoon never derives a URL from a host
  name, and the config loader gains no `${VAR}` interpolation.
- **What is recorded.** `identity.json` records each policy's model and renderer, never its URL, so a new invocation
  may pass new URLs and resume (services moved to other nodes). A supervised restart reloads `config.resolved.yaml`
  and keeps its URLs.
- **One session per policy.** Each policy opens its own `ServiceClient`, also when policies share a URL. The Tinker
  SDK reports a fatal client error by ending the session it happened on, and a miles-tinker service then reclaims
  every model of that session (miles-tinker: `miles/tinker/core/service.py`, `end_session` and `_release_model`). One
  session per policy keeps one policy's failure from deleting another's model. A frozen teacher opens its own session
  too.
- All services take the one `TINKER_API_KEY`.

### 3. Trainers and the run (`platoon/train/trainer/`)

```python
class Trainer(Protocol):
    """What `platoon train` runs: resolved from `trainer:`, built with from_setup, entered, run once per start.
    Supervise, restarts, reaping, the run lock, the config snapshot and the identity check stay platoon's."""

    @classmethod
    def needs(cls, config: Any, options: Any) -> "TrainerNeeds": ...   # asked at load, before any task is listed
    @classmethod
    def from_setup(cls, setup: "TrainerSetup") -> Self: ...
    async def __aenter__(self) -> Self: ...
    async def __aexit__(self, *exc_info: object) -> bool | None: ...
    async def run(self) -> None: ...


@dataclass(frozen=True)
class TrainerNeeds:
    rollouts: bool = False           # list training tasks, preflight environments, build the train workflow
    evaluates: bool = True           # build the eval workflow and run the evaluation loop
    max_trainable: int | None = 1    # trainable policies it can train (rl: any number)
    atomic_steps: bool = True        # rule 10 (rl: False)
    fail_closed_updates: bool = True # rule 10 (rl: False)
    draw_provenance: bool = False    # each training episode sees its draw as ctx.draw (section 13)
    stepping: Literal["lockstep", "independent"] = "lockstep"   # the clocks the run builds (rl: train.stepping)


@dataclass(frozen=True)
class TrainerSetup:
    config: Any                      # loaded and checked; read-only
    options: Any                     # trainer_options, parsed by the trainer's options schema (RFC 0016)
    run_dir: str
    policy_specs: Mapping[PolicyId, PolicySpec]
    agents: "AgentBindings"
    tasks: "TaskSources | None"
    components: "RunComponents | None"   # rollout functions, task loaders, reward processor, workflow, credit


class TrainerBase:
    """Optional: every hook TrainerRun calls, with a default. A subclass writes prepare() and run_steps()."""

    def __init__(self, setup: TrainerSetup) -> None: ...   # builds the PolicySet where the trainer registers today
    def check(self, run: "TrainerRun") -> None: ...    # before any service opens; may register participants and hooks
    async def prepare(self, run: "TrainerRun", start: "StartPoint") -> "StepPlan": ...   # the policies are open
    async def started(self, run: "TrainerRun") -> None: ...   # after the launch save
    async def ready(self, run: "TrainerRun") -> None: ...     # just before the loops
    async def run_steps(self, run: "TrainerRun") -> None: ...   # required
    async def close(self, run: "TrainerRun") -> None: ...     # always, in `finally`, also after a failure
    async def run(self) -> None:
        await TrainerRun(self.setup, self).execute()


class TrainerRun:
    """One start of one trainer: the lifecycle, and the handle every hook gets."""

    policies: PolicySet
    workflows: "Workflows | None"
    clock: "StepClock"                       # the run clock: the one clock in lockstep, else the leader's
    clocks: Mapping[PolicyId, "StepClock"]   # every trainable policy's clock; lockstep maps each to `clock`
    checkpoints: "RunCheckpoints"
    stats: "RunStats"
    hooks: "StepHooks"
    shutdown: asyncio.Event

    def rollouts(self, *, batch_size: int | None = None, horizon: int | None = None) -> "RolloutSource": ...
    def learner(self, policy: PolicyId, *, objective: "Objective | None" = None) -> "Learner": ...
    async def publish(self, updated: Sequence[PolicyId]) -> dict[PolicyId, int]: ...
        # on the cadence a full save (a barrier in lockstep), else weights only; then each updated policy publishes
    async def end_step(self, updates: Mapping[PolicyId, int], *, source: "RolloutSource | None" = None,
                       policy: PolicyId | None = None) -> dict[PolicyId, int]: ...
        # the step hooks, publish(), then the clock advances; `policy` names whose step ends (independent)
    async def wait(self, awaitable, *, name: str, timeout: float | None = None, interruptible: bool = False): ...
        # a log line and a watchdog heartbeat every minute, the deadline, and stop on shutdown if interruptible


class StepClock:
    policy: PolicyId | None                  # whose clock under independent stepping; None: the run's one clock
    step: int
    end: int
    updated: asyncio.Event                   # the evaluation loop wakes on it
    step_open: bool                          # set by a step's first training submission; read by rule 10
    def advance(self, *, consumed: "RolloutSource | None" = None) -> None: ...
```

**The lifecycle.** `TrainerRun.execute` runs today's training phases in today's order: build the workflows when the
trainer needs them; `trainer.check`; signal handlers; connect the policies, read the start point, and resume the
participants registered so far, before any training client opens; open the policies; `trainer.prepare`, after which
every saved `loop_state` key must be claimed (section 8); the launch save, `trainer.started`, the pruners and each
policy's launch publication; `trainer.ready`; `run_steps` beside the evaluation loop in one task group; the exit
save; and in `finally`, `trainer.close`, then the hooks' and participants' `aclose` in reverse registration order,
with their errors raised together.

**Subclassing `rl`.** `RLTrainer` (the alias of `PlatoonTinkerRLTrainer`) is a `TrainerBase` like any other. A
subclass may override `needs`, `check`, `prepare`, `started`, `ready` and `close`, each calling `super()`, and
register participants and step hooks there. It never overrides `run_steps` or a private method: that is how a trainer
reuses the whole RL step.

### 4. Rollouts for any trainer (`dispatch/source.py`)

```python
class RolloutSource:
    """The run's training draws and their settled groups: the task order and samplers, admission, per-task
    statistics, the dispatcher (permits, labels, leases, horizon, exact resume) and the groups a resume kept. Each
    part is today's code, taken off the RL trainer. It is the checkpoint participant that writes loop_state "tasks",
    "dispatch" and "next_position", in today's order."""

    workflow: "RolloutWorkflow"
    end_step: int
    results: "SettledQueue"                  # settled groups, in the order a step reads them
    ledger: "ShareLedger | None"             # independent stepping only (section 11)
    def start(self, task_group: asyncio.TaskGroup) -> None: ...
    async def take(self, n: int) -> list["Settled"]: ...
    def committed(self, keys: Sequence["DrawKey"]) -> None: ...   # the step consumed these draws
    def step_started(self, policy: PolicyId | None = None) -> None: ...   # through StepClock.advance
```

- **A draw's identity is its key `(start, seq)`**, where `start` numbers the run's starts, so a key is never reused,
  not even by a replay after a resume from an older checkpoint.
- **Draw provenance.** When a trainer asks (`TrainerNeeds.draw_provenance`), each training episode sees
  `ctx.draw = DrawInfo(run_dir, start, seq, position, step)`; a program can then write per-draw side files where its
  trainer reads them. A committed draw has settled, and a cancelled one is never committed, so the step end's list of
  consumed draws is the "these draws finished" signal.
- The dispatcher reads `versions` (a mapping) instead of one version. With one policy it is `{main: v}`.
- The horizon stays `(k + 1) × batch_size` groups, with `k` the smallest bound among the trainable sampled policies.
  In lockstep the dispatcher keeps today's counters, unchanged.

### 5. The workflow contract (`workflow/protocol.py`, `workflow/output.py`)

```python
class RolloutWorkflow(Protocol):
    async def arun_episode(self, data: dict) -> "RolloutOutput | Sequence[tinker.Datum] | None": ...
    # Optional, read through capabilities_of(): publish_model_binding(sampling_client, version[, *, policy]),
    # rollout_isolation, warm_up(), start_limit, and classmethod from_setup(setup: WorkflowSetup) -> Self


@dataclass(frozen=True)
class WorkflowCapabilities:
    applies_entry_settings: bool = False
    gives_verdicts: bool = False
    runs_rollouts_on_lease: bool = False     # RFC 0020
    takes_start_limit: bool = False          # replaces the trainer's isinstance check
    runs_rollout_workers: bool = False
    binds_policies: bool = False             # publish_model_binding(..., policy=): needed with >1 sampled policy


def capabilities_of(workflow: object | type) -> WorkflowCapabilities: ...   # its ClassVar, else today's flags


@dataclass(frozen=True)
class OutputReport:
    """What the trainer reads of a group's output besides its rows: today's duck-typed reads in one place."""

    label: str | None = None
    entry: str | None = None
    workload: "RolloutWorkload | None" = None
    requested_rollouts: int | None = None
    observed_rollouts: int | None = None
    trainable_rollouts: int | None = None
    rollout_stats: "RolloutStats | None" = None
    members: tuple["MemberResult", ...] = ()     # each member's versions, keyed by policy
    verdict: str = "unknown"


class RolloutOutput(Protocol):
    def rows(self) -> Mapping[PolicyId, Sequence["tinker.Datum"]]: ...
    @property
    def report(self) -> OutputReport: ...


def report_of(output: object) -> OutputReport: ...     # .report, else a legacy list's attributes, read as today
def rows_by_policy(output: object, trained: PolicyId) -> Mapping[PolicyId, Sequence["tinker.Datum"]]: ...
def make_output(samples, *, members, verdict, label=None, entry=None, **report) -> "PolicyGroupOutput": ...
```

- **Two output shapes, one reader.** When every row is one policy's and no agent tags are needed, the group workflow
  returns today's output class (a list of datums), which gains only `.report` and `.rows()`. Otherwise it returns a
  `PolicyGroupOutput`, whose rows are tagged by policy and agent. The step reads both through `rows_by_policy`, so it
  has one code path for one policy and for several.
- **Kept groups** get a codec table keyed by exact type. Today's output keeps its tag and a new `policy_group` tag is
  added; the kept-group format version stays 1, because a new tag is additive and an older platoon never resumes a
  run with several policies (the identity check refuses it).
- **Custom workflows** are built with `from_setup(WorkflowSetup)`, which carries the phase's settings, each entry's
  settings and estimator, and read-only views of the policies, in place of `model_info=`, which cannot carry several
  policies. The run builds the workflows for any trainer that needs rollouts or evaluation.

### 6. Credit (`platoon/train/credit/`)

```python
@dataclass(eq=False)
class TrainingSample:
    """One datum to train, with who produced it. Its tags live in the trainer process and never reach the service."""

    datum: "tinker.Datum"
    policy: PolicyId
    member: int
    agent_id: str | None = None              # these four come from the workers' tags, when the run asks for them
    role: str | None = None
    trajectory_id: str | None = None
    depth: int | None = None
    def set_advantage(self, value: float) -> None: ...            # on the action tokens
    def set_advantages(self, values: "torch.Tensor") -> None: ...  # one value per target token


@dataclass
class AgentRun:
    agent_id: str
    parent_id: str | None
    role: str
    policy: PolicyId
    depth: int
    reward: float | None                     # None: unscored, never trained as 0
    samples: list[TrainingSample]            # empty for a policy no learner trains (a frozen judge)


@dataclass
class CreditGroup:
    task_id: str
    entry: str | None
    settings: "WorkflowConfig"               # the entry's settings
    members: list["MemberRollout"]           # each with its samples, agents, task reward and result
    policies: Mapping[PolicyId, "PolicyView"]   # read-only: version, renderer, score()
    versions: Mapping[PolicyId, int]         # every policy's version when assign() began
    def emit(self, policy: PolicyId, datums: Sequence["tinker.Datum"]) -> None: ...   # rows for another policy (S5)


class CreditEstimator(Protocol):
    """How a group's rewards become advantages. Built per workflow and entry as Class(setup, options), with options
    checked at load (RFC 0016). An estimator with state to resume is also a CheckpointParticipant."""

    needs_agents: ClassVar[bool]             # True: workers tag datums, and members carry AgentRuns
    async def assign(self, group: CreditGroup) -> None: ...
    # Optional: settle(group) -> None, synchronous. A stateful estimator changes its state here and only here; the
    # run calls it in the same section, with no await, that files the group (section 11).
```

| Name | What it does | For |
|---|---|---|
| `group` | today's credit: the completed-root group mean, or leave-one-out, on each member's datums | S8 |
| `keyed` | baselines keyed by any of `task`, `policy`, `role`, `depth`, with optional leave-one-out (the subset of RFC 0001's `BaselineSpec` that has a user) | S6, S7 |
| `reverse_kl` | per-token `-coef × (logp_student − logp_teacher)` from a frozen teacher's `score`, plus an optional reward term | S2 |
| `gae` | advantages and value targets from a critic, after a spike shows Tinker can train one | S5 |

- **Where it runs.** The group workflow has one credit call site. Without a `credit:` key it calls today's synchronous
  `assign_advantages` at today's point, so S8 is unchanged; a subclass that overrides `compute_baselines` or
  `assign_advantages` keeps having its hook called. With a key it awaits the entry's estimator. The error-token
  filter, the verdict and the output follow as today.
- **Evaluation never gets an estimator**, so it never calls a teacher or a critic.
- **What an estimator may assume**, in both stepping modes: every member has settled, and every row names its policy
  and the version of that policy it was sampled at; `group.versions` is a snapshot taken when `assign` began; its
  advantages are final, so a group is credited once and a resume that trains a kept row again reuses them; and a
  trainable scorer (S5's critic) answers at its version during the call. It may not assume that different policies'
  rows train in the same step, at the same versions, or at all, so it must hold no state that waits for another
  policy's update.

### 7. Objectives and learners (`platoon/train/objectives/`, `policy/learner.py`)

```python
class Objective(Protocol):
    loss_fn: str                             # a Tinker built-in: cispo | ppo | importance_sampling | cross_entropy
    loss_fn_config: Mapping[str, Any]
    side_channels: frozenset[str]            # inputs only the trainer reads, stripped before forward_backward
    def counts(self, datums) -> tuple[int, int, int]: ...   # datums, attention tokens, loss tokens
    def filter(self, datums) -> list["tinker.Datum"]: ...   # policy gradient: the zero-advantage filter
    def is_zero(self, datums) -> bool: ...
    def scale(self, datums, factor: float) -> None: ...     # mean reduction: advantages or weights
    def metrics(self, originals, result) -> dict[str, float]: ...
    # Optional: async forward_backward(client, datums), for forward_backward_custom (DPO, a value loss)


class Learner:
    """Updates one trainable policy on its own service, with its own optimizer and objective."""

    def update(self) -> "PolicyUpdate": ...


class PolicyUpdate:      # today's optimizer update, reading the policy instead of the trainer
    async def submit(self, datums, originals, *, minibatch: int, microbatch: int) -> None: ...
    async def apply(self, tally: "StepTally") -> None: ...   # the eval.at_start barrier, optim_step, results
```

- `PolicyGradient(loss_fn, loss_fn_config)` is today's code, moved and unchanged: counting, the zero-advantage
  filter, advantage scaling, the strip list and the training metrics. `CrossEntropy()` takes rows with `target_tokens`
  and `weights`, sent as given, so a caller may weight tokens freely; an SFT row never meets the policy-gradient
  counting that needs `mask` and `advantages`.
- A policy's objective follows its `loss_fn`. One objective per policy per step serves every scenario built first;
  several losses in one `optim_step` wait for a conformance test (section 18).
- With several learners a step submits every learner's `optim_step` and then finishes each, so the services work at
  the same time. With one learner this is today's update, call for call. The old class name stays as an alias for one
  minor release.

### 8. Checkpoints and resume (`platoon/train/checkpoints/`)

```python
class CheckpointParticipant(Protocol):
    """Part of the run's state that every full checkpoint saves and every resume takes up."""

    keys: tuple[str, ...]                    # its loop_state keys
    async def checkpoint_state(self, name: str) -> Mapping[str, Any]: ...   # may write immutable side files
    def resume(self, start: "StartPoint") -> None: ...   # start.record is None on a fresh run and a warm start
    # Optional: async aclose(); capture(name) -> Awaitable[Mapping[str, Any]], a plain method that takes what it
    # saves at once and returns the writing still to do (section 11)


@dataclass(frozen=True)
class StartPoint:
    record: "CheckpointRecord | None"        # the row the run resumes from
    step: int                                # loop_state.batch: the run clock's step
    policies: Mapping[PolicyId, "PolicyStart"]   # each policy's row, version and optimizer steps
    loop_state: Mapping[str, Any]


class RunCheckpoints:
    def register(self, participant: CheckpointParticipant) -> None: ...   # registration order = key order
    def kind_after(self, step: int) -> Literal["permanent", "recovery"] | None: ...
    async def save(self, name: str, *, step: int,
                   policy: PolicyId | None = None) -> Mapping[PolicyId, "SavedPaths"]: ...
```

**A save with one policy** is today's: the participants' entries, the policy's save, and its row in
`<run>/checkpoints.jsonl` with `{batch, wandb_run_id, policy_version, optimizer_steps, tasks, dispatch,
next_position}`, byte for byte. **In lockstep with several policies** a save is a barrier: the participants' entries
at one point, then every non-leader trainable policy saves at once into `<run>/policies/<id>/checkpoints.jsonl`, then
the leader saves into `<run>/checkpoints.jsonl` with `policies: {id: {name, state_path, policy_version,
optimizer_steps}}`. The leader's row is the commit, appended only after every save succeeded. Rows are named by
`state_path`, which is unique per training run; a name and a version repeat after a resume from an older row.
Independent stepping's commits are in section 11.

The participant rules hold for core's participants and a trainer's alike:
- **Keys are owned.** Registering a key that another participant holds, or that the run's counters use (`batch`,
  `wandb_run_id`, `policy_version`, `optimizer_steps`, `policies`, `commit_seq`), fails and names both owners.
- **Every full checkpoint, never a weights-only one.** Launch, cadenced, recovery, interrupted, final and exit saves
  call every participant; a save for a sampler calls none.
- **Snapshot, then save, then commit.** Every participant's state is taken at one point, before any policy saves; the
  row that holds it is appended only after every save succeeded. A side file is therefore committed by reference: a
  failed save leaves an orphan file, never a replaced commit.
- **Resume before the services open.** A participant registered in `check()` resumes right after the start point is
  read, before any training client opens. One registered later (the rollout source, in `prepare()`) resumes when it
  registers, which keeps today's order.
- **Nothing saved is dropped silently.** After `prepare()`, a key of the resumed row that no participant and no
  counter claims fails the start (exit 78). Core claims every key a platoon release ever wrote, so old run
  directories resume.

**Resume.** With one policy, today's rules. In lockstep, the newest leader row whose named rows exist in each other
policy's file; a newer row that does not hold is skipped with a warning and recorded in `lock.json`.

| File | One policy (S8) | Several policies |
|---|---|---|
| `<run>/checkpoints.jsonl` | today's rows and keys | the leader's rows, with `loop_state.policies`; under independent stepping every row is a commit with `commit_seq` |
| `<run>/policies/<id>/checkpoints.jsonl` | not created | each other trainable policy's rows; under independent stepping each full row is also a commit |
| `identity.json` | `model`, `renderer` | adds `policies: {id: {model, renderer}}`, and `stepping: independent` when the run steps so. A resume that changes a policy's model, adds or removes a policy, or changes the stepping exits 78 |
| `evaluations.jsonl` | unchanged | adds `versions`, and `versions_end` when a policy published during the evaluation |
| recovery pruning | one pruner | one pruner per trainable policy, on its own URL and session; `k = 0` for a policy that no rollout samples. It never deletes a row that the newest barrier names, nor a row that section 11 pins |

### 9. Step hooks (`trainer/hooks.py`)

```python
class StepHook(Protocol):
    """Work at the end of each step of any trainer, after the step's own updates and commit, before it publishes.
    Every method is optional. A hook with state to resume is also a CheckpointParticipant."""

    name: str                                              # its log lines, waits and metric prefix
    async def drain(self, end: "StepEnd") -> None: ...     # finish side work that samples the current sampler
    async def update(self, end: "StepEnd") -> None: ...    # optimizer updates, through end.learner(...) only
    async def published(self, end: "StepEnd") -> None: ...  # every step end, and once after the launch publication
    async def aclose(self) -> None: ...


@dataclass
class StepEnd:
    kind: Literal["launch", "step"]
    policy: PolicyId | None                   # independent stepping: whose step ends
    step: int
    step_updates: Mapping[PolicyId, int]      # the step's own optimizer updates
    hook_updates: Mapping[PolicyId, int]      # the hooks' updates so far, counted by platoon
    consumed: Mapping["DrawKey", str]         # the draws the step committed, each with its fate
    published: Mapping[PolicyId, int]         # filled before published(): who published, at which version
    tracker: "StatsTracker"                   # what a hook records here is logged with the step
    def learner(self, policy: PolicyId, *, objective: "Objective") -> "Learner": ...   # during update() only
```

The sequence at a step's end: the step's own updates; its commit (the draws' fates); each hook's `drain`; each hook's
`update`; then each policy whose own or hooks' updates are above zero publishes once, with one new version; each
hook's `published`; and the clock advances.
- **Nothing of the step is in flight when hooks run.** A hook's updates use the policy's training client and Adam
  state; they are not a second optimizer.
- **Platoon counts the hooks' updates** (each `apply` through `end.learner`), so the publish decision, the optimizer
  step count and what reached the service cannot disagree.
- **One version per step.** The hooks' updates spend no staleness budget, because the horizon counts steps and the
  step publishes once. A trainer that wants each extra update to be a version of its own runs it as a step of its own.
- **Failures.** A hook that raises fails the start before anything publishes or saves, as a failed `forward_backward`
  does, and supervise restarts from the last full checkpoint.
- **S8 has no hook.** The phases are skipped, not awaited empty, so S8's step is unchanged call for call.

### 10. Several policies in lockstep

This is S7 with `train.stepping: lockstep`, the default, and RFC 0003's M1 and M2.

```yaml
trainer: rl
train:
  batch_size: 16
  max_staleness: 1
  loss_fn: cispo
  workflow_config: {group_size: 8, rollout_isolation: process}
policies:
  orchestrator:                                  # the first trainable policy, so the leader
    model_name: Qwen/Qwen3.6-35B-A3B
    renderer_name: qwen3_6_preserve_thinking
    tinker_base_url: http://orchestrator-host:8000   # the job script: policies.orchestrator.tinker_base_url=...
    optimizer: {learning_rate: 1.0e-6}
  worker:
    model_name: Qwen/Qwen3-4B-Instruct-2507
    renderer_name: qwen3_instruct
    tinker_base_url: http://worker-host:8000
    lora_rank: 32
    optimizer: {learning_rate: 3.0e-5}           # its own optimizer
    loss_fn: importance_sampling                 # its own loss
    max_staleness: 2                             # its own bound
  judge:
    model_name: Qwen/Qwen3-32B
    renderer_name: qwen3
    tinker_base_url: http://judge-host:8000
    trainable: false                             # frozen: sampled, never trained, saved or pruned
agents:
  policy: orchestrator                           # the root agent's
  roles: {worker: {policy: worker}, judge: {policy: judge}}
environments:
  - name: research
    credit: keyed
    credit_options: {key: [task, policy, role]}  # the default with two or more trainable policies
```

1. **Agents.** The root agent samples the orchestrator. `POST /v1/platoon/agents {role: worker}` mints a child that
   samples the worker policy for life, with its renderer, tokenizer and sampler. Agents exchange only text.
2. **Rows.** Each datum is built from one trajectory's calls, stamped with the oldest version of its own policy that
   its trajectory sampled, and tagged `(policy, agent_id, role, trajectory_id, depth)`.
3. **Credit** runs once per group over all members and agents, here with baselines per `(task, policy, role)`.
4. **The step** takes `train.batch_size` settled groups. The stale filter runs per policy: a stale policy's rows of a
   group drop together and the group's other rows stay. `rows_by_policy` splits each microbatch; each trainable policy
   runs its transforms, objective and normalization on its own rows and submits them to its own service. At each
   minibatch's end every learner with rows submits its `optim_step`, then each one finishes, so the 35B and the 4B
   steps overlap.
5. **Publishing.** Each policy that stepped saves (weights only, or the barrier at a due step) and publishes: its
   gateway runtime is swapped, its version goes up by one, and its binding file is rewritten. A policy that did not
   step keeps its version.
6. **Staleness.** A row of policy `p` may be `policies.<p>.max_staleness` versions old, else its entry's bound (RFC
   0019), else `train.max_staleness`. The horizon is `(k + 1) × batch_size` groups with `k` the smallest bound of the
   trainable sampled policies (here 1, so 32 groups). Each policy publishes at most once per step, so nothing inside
   the horizon is stale by design.
7. **Checkpoints** are the barrier of section 8. A crash between the two saves leaves a worker row that no leader row
   names; the resume ignores it, and the worker's pruner deletes it later. A self-hosted service restarted on other
   nodes must still read the `tinker://` paths it wrote, so its checkpoint directory must be on shared storage.
8. **Metrics.** With two or more trainable policies, every per-policy metric moves under `policies/<id>/` with today's
   name after the prefix, and run-level `workload/*`, `batch/*`, `dispatch/*` and `timing/*` stay as they are. One
   trainable policy keeps today's names.
9. **Evaluation** publishes the eval workflow's binding for every sampled policy and records each policy's version
   when it starts (`versions`). It records rather than pins them: on a miles-tinker full fine-tuning service a sampler
   serves the newest weights whatever its path.
10. **Failures.** Any policy's failed training call ends the start, naming the policy; the exit save is attempted for
    every trainable policy, and the leader's row is written only if all of them saved. A member whose sampling calls
    fail ends `error`, which never trains as 0. Supervise restarts, and every policy resumes from the last complete
    barrier, so the policies never disagree.

Lockstep runs at the slower learner's pace; the next section does not change the rate at which groups start.

### 11. Independent stepping

`train.stepping: independent` lets each trainable policy step, publish and checkpoint on its own clock, with its own
batch and staleness bound, and with no global barrier. After credit, a settled group's rows split by policy, as in
lockstep; but each policy receives its own part of each group, its **share**, and trains it when its own step is
ready. With one trainable policy `independent` runs the lockstep path, and lockstep never builds anything in this
section.

```yaml
train:
  stepping: independent                      # default: lockstep
  batch_size: 16                             # the leader's shares per step, and the plan's groups per draw batch
policies:
  orchestrator: {..., max_staleness: 1}
  worker:       {..., max_staleness: 7, batch_size: 4, checkpoint: {recovery_every: 5}}
```

```python
class ShareLedger:
    """Every group that has not retired, split into shares: one per trainable policy with rows in it. Per share: its
    place in its policy's queue, and once consumed, the policy's step and the share's age then. Saved with the kept
    groups, as part of the rollout source's state."""

    def add(self, group: "Settled") -> None: ...           # at settle, after credit
    async def take(self, policy: PolicyId, n: int) -> list["Settled"]: ...   # reverted shares first
    def consumed(self, policy: PolicyId, keys: Sequence["DrawKey"], *, step: int) -> None: ...   # records ages
    def durable(self, policy: PolicyId, step: int) -> list["DrawKey"]: ...   # the groups that retire now
    def revert(self, start: StartPoint) -> None: ...       # shares consumed after each policy's row go back
    def done(self, policy: PolicyId) -> bool: ...          # its loop may end
```

**The loops.** Each policy's loop is the one-policy RL step with one learner, run as one task per policy. It takes
`policies.<p>.batch_size` shares (default `train.batch_size`), trains them, publishes the policy and advances its own
clock. `train.batch_size` is both the plan's unit and the leader's batch, so the leader takes no batch of its own. A
trainable policy that no agent samples (S5's critic) gets the shares credit emitted for it. The plan stays in draws,
`end_step × train.batch_size` groups; a policy's loop ends when the ledger says so, and the exit save follows the
last loop. `eval.every` counts the leader's steps.

**Settling is one section with no await.** Credit's asynchronous work runs first. Then, in one synchronous section
under the commit lock: a stateful estimator's `settle(group)`, the split into shares and the insert into the kept
set. A capture therefore sees a group either wholly settled or not at all, so a resume never credits a group twice or
loses one.

**Staleness.** Each policy's filter judges only its own share: its age is the policy's version now minus the oldest
version of that policy on the share's rows, against the policy's bound. A stale share drops for that policy alone.
The other policies' versions are context, not staleness: `policies/<p>/peer_lag/<q>` reports how far policy `q`
moved between the sampling of `p`'s rows and their training. A cross-policy bound waits for a run that shows it
matters.

**The horizons.** The dispatcher holds a private horizon for every trainable policy that receives shares, `(k_p + 1) ×
batch_p` shares; for a policy that no agent samples, `k_p` is its `max_staleness`, else `train.max_staleness`, and
bounds its retained groups. A dispatched group counts against every such horizon, since which policies a rollout
samples is known only when it settles; then it leaves the horizons of the policies it has no share for, and a share
leaves its policy's count when that policy's next step starts. A group starts only when every horizon it may use has
room. The rollout permits stay one pool, since one rollout samples several policies.
`dispatch/blocked_share/horizon:<id>` names the horizon that holds the dispatcher back.

**Rates stay coupled.** Every group feeds every sampled trainable policy: in S7 the root agent samples the
orchestrator, so every group holds an orchestrator share. In steady state the dispatcher therefore starts about
`batch_p` groups per step of the policy whose horizon binds, usually the slowest learner, and every other policy
receives that many shares per step of it. Independent stepping does not raise group throughput when one learner is
the bottleneck; lockstep has the same bound. What it changes:
1. No policy waits for another's training, saves or evaluation.
2. Each policy steps as often as its batch allows, about `batch_slow / batch_p` times per step of the slow policy. In
   the example the orchestrator's horizon, `(1 + 1) × 16 = 32` groups, lets about 16 groups start per orchestrator
   step, and with a batch of 4 the worker steps about four times per orchestrator step. A worker batch of 64 would
   step once every four orchestrator steps: a larger batch steps less often, never more.
3. A policy that steps more often sees its rows age faster in its own versions, so its bound grows by the same
   factor, `(k_p + 1) × batch_p ≥ (k_slow + 1) × batch_slow`; otherwise its own horizon gates dispatch. The worker's
   bound of 7 gives it `(7 + 1) × 4 = 32` shares, the orchestrator's window.
4. Keeping a slow policy from gating dispatch at all would require it to drop the shares it cannot train in time
   (open question 14).

**Versions on resume.** A policy resumes at its row's `policy_version`, unless the commit records a
`published_version` for it above that; then it resumes at `published_version + 1`, so rows sampled from samplers
that a crash lost count at least one version old and never share a number with a new sampler. Lockstep rows record no
`published_version`, so lockstep and S8 resume as today. **A reverted share keeps the age it was first judged at.** It
passed the stale filter once, at the same point of the same lineage, so it skips the filter and trains again
(`policies/<p>/retrained_shares`). Without this, a policy that resumes from a row named by another policy's commit,
with its version raised, would drop the very shares it must train again.

**Every full save is a commit.** Each trainable policy saves weights at each of its steps, to publish, and a full
save at its own cadence (`policies.<id>.checkpoint`, else `checkpoint.*`, in its own steps). A full save is a commit,
in three parts:
1. **Capture**, under the commit lock and with no await: the next `commit_seq`; each participant's `capture()`, which
   for the rollout source is its kept set and ledger by reference (kept records are immutable); and each other
   trainable policy's newest durable row and published version. The rows it names are **pinned**.
2. **Write**, after the lock is released: the participants' files, then the policy's Tinker save. The other policies
   keep stepping meanwhile.
3. **Append** the row to the policy's own file: `{batch, wandb_run_id, policy_version, optimizer_steps, commit_seq,
   <participants' keys>, policies: {id: {name, state_path, policy_version, published_version, optimizer_steps,
   step}}}`. `batch` is always the run clock's step, the leader's step at which this commit resumes it, so the start
   point, the evaluation cadence and a participant that compares its cursor with the start's step read the leader's
   step whichever policy committed. `policies` names every trainable policy's row, the committing one's own included.

- **Commit order.** `commit_seq` is assigned at capture. The resume takes the highest one whose named rows exist,
  across every policy's file; after it, numbering continues from the highest in any file plus one, so a new commit
  never ties with one an earlier start left behind.
- **Pins.** Every row named by a commit captured but not yet appended, and every row named by the newest commit in
  each policy's file, is pinned, and every pruner skips pinned rows. Otherwise a commit that names the worker's newest
  row and then spends minutes in the 35B's save could find that row pruned by the worker's later commits. A named row
  whose path is in its policy's `checkpoints_deleted.jsonl` counts as missing at resume.
- **A group retires** (it leaves the kept set) when every share has been consumed and each consuming policy has a
  durable full row at or after the step that consumed it. Its fates and per-task statistics are recorded once, when
  its last share is consumed. Until then the kept-groups file keeps it, with `shares: {policy: {step, age} | null}`
  and `recorded` on its record; a record without them reads as "every share unconsumed", which is what a lockstep
  checkpoint means.
- **After a crash.** Each policy resumes from the row the chosen commit names and trains again the shares it consumed
  since, from the kept set, with their advantages and ages, without repeating a rollout; the draws that were running
  are drawn again, as today. No share is trained twice into one policy's weights, and no group's statistics are
  recorded twice.
- **The cost.** Retention is at most the shares each policy consumed since its last full save (`run/retained_groups`),
  so frequent recovery saves keep the kept file near today's size; each commit writes it, outside the lock.
- **Exit.** Every loop stops at a step boundary, every policy saves, and one commit names them all.
- **Metrics.** Each policy's step end logs its metrics under `policies/<id>/` with `policies/<id>/step`, declared
  with `wandb.define_metric` as that prefix's axis; run-level metrics go with the leader's step ends on `run/step`.
  These rows carry no W&B step of their own.
- **Failures** stop the start in both modes, and every policy resumes from the newest commit. Training the other
  policies on while one service is down is not built (section 18).

### 12. Multi-LoRA on one service

RFC 0003's M4: only placement changes. Policies whose URLs resolve equal are **tenants** of one service, and
trainable tenants with the same `model_name` are its LoRA adapters. That needs no key:

```yaml
policies:   # the job script passes the same URL to both
  worker:   {model_name: Qwen/Qwen3-4B-Instruct-2507, renderer_name: qwen3_instruct, lora_rank: 32,
             tinker_base_url: http://worker-host:8000, optimizer: {learning_rate: 3.0e-5}, loss_fn: importance_sampling}
  verifier: {model_name: Qwen/Qwen3-4B-Instruct-2507, renderer_name: qwen3_instruct, lora_rank: 32,
             tinker_base_url: http://worker-host:8000, optimizer: {learning_rate: 1.0e-5}, loss_fn: importance_sampling}
```

- **URLs resolve equal** when they are equal after the scheme and host are lower-cased, a default port is dropped and
  a trailing slash is removed. No name is looked up, so two names of one host are two services to platoon.
- **Everything stays per adapter:** its own session and training client (`create_lora_training_client_async` with
  its rank), Adam state, loss, bound, sampler path, version, binding file, checkpoint rows and pruner. A miles-tinker
  LoRA service gives each a slot with its own optimizer; hosted Tinker gives each a training run of its own.
- **What the tenants share** is the service: its capacity, its queue and its failures. A miles-tinker service runs
  every tenant's training requests in arrival order, packs rows of adapters that share a loss into shared work units
  and merges their optimizer steps; a lockstep step therefore costs about the sum of the adapters' work, where
  separate services cost the slower one's. Under independent stepping the adapters' steps overlap on the service.
  Sampling limits are the service's own. Multi-LoRA saves service nodes; it does not save time when rollouts bind.
- **Slots.** A miles-tinker LoRA gateway holds a fixed number of adapters, a full fine-tuning gateway exactly one. A
  crashed start's sessions keep their slots until their lease expires, so a restart needs a free slot per adapter;
  today's open retry covers the lease, and the final error names the service, its tenants and the server's message.
  The docs size the slots at twice the adapters.
- **Warnings at load:** tenants with different base models, and adapters with different ranks (miles-tinker serves
  one base model and one maximum rank per service, and refuses at open).
- **What the services must support**, each checked by `platoon.testing.check_shared_service(url, base_model=...)` on
  a live service before the docs claim support:

| | Need | Status |
|---|---|---|
| U1 | several LoRA models of one base, each with its own Adam state, stepping separately | miles-tinker: in its source. Hosted: documented. UNVERIFIED from platoon |
| U2 | each adapter's sampler serves its own weights | UNVERIFIED |
| U3 | one session's end leaves the other sessions' models | miles-tinker: in its source. UNVERIFIED on hosted Tinker |
| U4 | save one adapter and reopen it with its optimizer while another trains | UNVERIFIED |
| U5 | adapters of different ranks on one service | UNVERIFIED; they warn at load until checked |
| U6 | a capacity refusal is retryable, and its status | UNVERIFIED; a fast test against the gateway's app |
| U7 | hosted Tinker trains two LoRA models of one base at once, from two sessions under one key | documented only |
| U8 | throughput: packing across adapters, and sampling split between them | measured once; a perf run before the docs recommend sharing a training-bound pair |
| U9 | a MoE base with routing replay per adapter | an open miles pull request; platoon cannot detect routing replay on a gateway, so the docs warn |
| U10 | after a session ends, the sampler paths it saved still serve | miles-tinker: no, its release unloads them. UNVERIFIED on hosted Tinker |

### 13. Workers and the gateway (only in runs with several policies or agent tags)

- **Worker protocol 2 is not touched.** A new request type, `PolicyRolloutWorkerRequest` (protocol 3), adds the
  bindings of the other sampled policies, the agents' role bindings, the trained policies and whether to tag datums.
  It is sent only when more than one policy is sampled or the entry's estimator needs agents.
- **Rollout workers** follow one binding file per policy (today's `model_binding.pkl` for the agents' default policy,
  `model_binding.<id>.pkl` for the others) and build one gateway with a runtime per policy.
- **The gateway** gains a `PolicyRuntime` per policy (sampler, renderer, tokenizer, model name, sampling defaults);
  the old keywords build the default one. `open_agent(role=...)` and child minting resolve an agent's policy through
  the role bindings, for its whole life. Each call record carries its policy and the version it sampled. A request
  that names a policy is refused.
- **What programs read stays public.** `AgentSession.interactions` and `TinkerLLMInteraction` keep their shape and
  gain defaulted `policy` and `version` fields.
- **`ctx.draw`** travels in the rollout config under a reserved key, so protocol 2's fields stay as they are.
- **The converter** builds datums per trajectory, as today, so each datum has one agent, one policy and one tokenizer.
  A datum's version stamp is the oldest version of its own policy that its trajectory sampled. Policies outside the
  trained set produce no datums (a frozen judge), but their agents and rewards still reach credit.

### 14. Configuration

Every key is new and optional, and every existing key keeps its meaning.

```yaml
trainer: rl                     # default. rl | sft | interleaved | a registered name | package.module:Class
trainer_options: {}             # checked against the trainer's options schema at load. rl takes none
policies:                       # absent: one policy, main, from train.* and tinker_base_url
  <id>: {model_name, renderer_name, renderer_kwargs, context_window_length, sample_timeout_seconds,
         tinker_base_url, trainable, lora_rank, load_checkpoint_path, load_optimizer_state, optimizer,
         loss_fn, loss_fn_config, max_staleness,
         batch_size, checkpoint: {every, recovery_every}}      # these two: independent, non-leader policies only
train:
  stepping: lockstep            # an rl field. independent: each trainable policy on its own clock
checkpoint:
  load_optimizer_state: false   # main's: with load_checkpoint_path, a new run from its weights and Adam state
agents:                         # absent: every agent samples the first policy
  policy: <id>                  # the root agent's, and that of any role not listed
  roles: {<role>: {policy: <id>}}
environments:
  - name: ...
    credit: keyed               # per entry; absent: the workflow's own credit (today's)
    credit_options: {key: [task, policy, role]}
```

- **What `train:` means.** It holds `main`'s model keys and the `rl` trainer's settings, as it does today; this is the
  one place where RFC 0016's "options next to their selector" does not hold, and it is documented as such. Under
  another trainer, an `rl` setting with a non-default value fails at load and names the trainer option to use.
- **Resolution** follows RFC 0013: core names, then names a `package:` module registered, then `module:Class`.
  Trainers also resolve from the entry-point group `platoon.trainers`, because outside packages already write them.
  The resolved component is recorded in `lock.json` only when the key is set, so S8's run directory keeps its contents.
- **Decorators** in `platoon.experimental`: `@trainer(name, options=, check=)` and `@credit_estimator(name, options=,
  check=)`, shaped like `@task_sampler`. A trainer's `check` sees the whole loaded config, at load and under
  `platoon config validate`.
- **Load checks** (before any service opens, exit 78): `load_optimizer_state` without a checkpoint path; an unknown
  policy in `agents` or a credit option; more trainable policies than the trainer trains; a frozen policy a trainer
  must train; `reverse_kl` with a teacher whose tokenizer differs from the student's; more than one sampled policy with
  a workflow that does not bind policies; a policy table or stepping that differs from `identity.json`; a
  non-default `rl` setting under another trainer; a policy's `batch_size` or `checkpoint` wherever it would be
  ignored (under lockstep, on a frozen policy, on the leader).
- **Conformance** (RFC 0017), in `platoon.testing`: `check_trainer(cls, options)` runs a trainer for two steps on a
  fake Tinker service, interrupts and resumes it, checks its saves and `loop_state`, and runs its step hooks;
  `check_credit_estimator(cls, options)` checks that advantages land on action tokens only and that state resumes;
  `check_shared_service` runs against a live service (section 12).

### 15. Layout

The reorganization in review moves `platoon/train/tinker/*` to areas under `platoon/train/` (for example
`tinker/rl.py` to `trainer/rl.py`, the group workflow to `workflow/group.py` and its credit functions to
`credit/group.py`), with warning shims at the old paths until 0.6.0. This RFC adds four packages and a few modules:

```
platoon/train/
├── launch.py            resolves trainer:, preflight by TrainerNeeds, TrainerSetup, run()
├── policy/        NEW   spec.py, policy.py, set.py, service.py, checkpoints.py, binding.py, learner.py
├── objectives/    NEW   api.py, policy_gradient.py, cross_entropy.py
├── credit/        NEW   api.py, group.py, keyed.py, reverse_kl.py
├── trainer/             api.py, run.py, registry.py, hooks.py (NEW); rl.py (today's trainer); loops.py, state.py;
│                        sft.py, interleaved.py (NEW); signals.py, waiting.py
├── step/                the rl step's stages
├── dispatch/            + source.py (RolloutSource), shares.py (ShareLedger)
├── workflow/            the group workflow; protocol.py (the contract), output.py (OutputReport, PolicyGroupOutput)
├── workers/  checkpoints/ (+ RunCheckpoints, participants, codecs)  evaluation/
└── tinker/              shims until 0.6.0
```

The trainer kinds sit side by side in `trainer/`. A test walks every import and enforces one rule: generic modules
(`policy/`, `objectives/`, the generic part of `trainer/`, `dispatch/`, `checkpoints/`, `evaluation/`, `workers/`)
import no trainer, workflow or credit estimator; workflows and estimators import no trainer.

### 16. The scenarios in brief

| | Selected by | Platoon adds | The user writes |
|---|---|---|---|
| S1 SFT | `trainer: sft` with `trainer_options` (dataset loader, batch size, epochs, `train_on`) | `trainer/sft.py`, `CrossEntropy`, a dataset cursor as a participant (EST 550 lines) | about 15 lines of YAML |
| S2 on-policy distillation | `credit: reverse_kl` and a frozen `policies.teacher` | `reverse_kl`, `Policy.score`, frozen policies | about 25 lines of YAML |
| S3 interleaved RL, SFT and NTP | `trainer: interleaved` with a schedule | `trainer/interleaved.py` (EST 250 lines): SFT and NTP as steps of their own, or as step hooks at an RL step's end | about 30 lines of YAML |
| S4 an outside trainer | `trainer: acme.dpo:DPOTrainer` | the trainer API, `run.rollouts()`, step hooks, `check_trainer` | a class of about 60 to 300 lines |
| S5 PPO with a critic | `credit: gae` and `policies.critic` | `gae` and a value objective, after a spike on Tinker | about 35 lines of YAML |
| S6 a user's own multi-agent credit | `environments[].credit: acme.credit:ByRole` | the credit API and datum tags | an estimator of about 40 to 60 lines |
| S7 several policies | `policies:` with URLs and `agents:`; optionally `train.stepping` | RFC 0003's M1, M2 (both stepping modes) and M4 | about 45 lines of YAML |
| S8 today's GRPO | no new key | nothing: byte-identical | nothing |

- **S1.** One step takes a batch of dataset rows, renders them with the policy's renderer, sends them through
  `run.learner(id, objective=CrossEntropy())` and calls `run.end_step`. It never touches the dispatcher, a workflow,
  permits, the stale filter or the horizon, so the RL-only assumptions are avoided by not running the RL step at all.
  Recovery checkpoints prune at `k = 0`. Evaluation during training works as for RL.
- **S2.** The teacher is one block under today's field names. For each student row, `reverse_kl` awaits the teacher's
  `score` and sets per-token advantages. The teacher's calls run after the members returned their permits, so they
  hold no rollout slot. No agent samples the teacher, so it publishes nothing and has no staleness. Evaluation never
  calls it. Loading refuses a teacher whose tokenizer differs from the student's.
- **S3.** Every step publishes, SFT and NTP steps included, because rollouts sample the policy (rule 6), so versions
  keep measuring off-policy distance. Only RL steps move the horizon; the trainer chooses the largest horizon for which
  no group can go stale under its schedule, and loading refuses a schedule that no horizon fits. With
  `placement: boundary` the SFT and NTP updates run as step hooks at the end of an RL step instead, which spends no
  staleness budget; this is the outside interleaved trainer's model on core's datasets. Several losses in one
  `optim_step` wait for a conformance test.
- **S4.** A DPO trainer without rollouts registers its pair cursor as a participant in `prepare` and loops
  `learner.update()`, `submit`, `apply`, `run.end_step`. A REINFORCE trainer with its own batch-mean baseline takes
  groups with `run.rollouts().take(n)`, reads rows through `rows_by_policy`, and calls `committed` and `end_step`. It
  gets resolution, supervise, saves, resume and evaluation without overriding any private method.
- **S5** is blocked on Tinker, not on structure: Tinker has no value head, and whether a value read-out through
  `forward_backward_custom` trains on hosted Tinker and miles-tinker is UNVERIFIED. The structure is there: a second
  learner with its own optimizer and checkpoints, async credit, estimator state that resumes, kept outputs, and
  `group_size: 1` with a sampler that allows it.
- **S6.** An estimator groups `AgentRun`s by role and sets each agent's samples' advantages. It needs no subclass of
  the group workflow. A workflow that is not group-shaped implements `RolloutWorkflow` with `from_setup`, runs the
  entry's estimator and returns `make_output(...)`.
- **S7** is sections 10 to 12.
- **S8** resolves to `trainer: rl`, one policy `main` from `train.*`, no `agents` and no `credit` (section 17).

### 17. Migration and compatibility

**Order.** The reorganization of `platoon/train` in review lands first, with two destinations adjusted to this
layout (`trainer/rl.py` and `credit/group.py`, both still pure moves) and a few changes of shape that keep the golden
trace identical. Then five refactors, each keeping the golden trace identical (A1-A5), then the features (B0-B6); the
rollout plan lists them.

**How S8 stays byte-identical.** The golden trace compares, per start, every Tinker call, every datum as sent, every
save and binding publication, every rollout-request field, each checkpoint row with its `loop_state`, events, the
text of INFO and higher messages, the run directory's file names and the metrics. Every new piece has a guard:

| New piece | What S8 does | Guard |
|---|---|---|
| `trainer:` resolution, `TrainerRun` | no key, so `rl`; the same constructor arguments; today's phases in order | phases lifted at phase boundaries; the golden in every pull request |
| `PolicySet` with `main` only | `register_tinker_llm` once, one `ServiceClient` at today's point | the golden patches both by identity in every module |
| `RolloutSource`, `RunCheckpoints` | the same dispatcher at the same point; keys `batch, wandb_run_id, policy_version, optimizer_steps, tasks, dispatch, next_position` in order | the source is the only participant; empty states and `policies` are not written |
| credit | no `credit:` key: `assign_advantages` runs synchronously at today's point | a coroutine that never awaits does not yield |
| objectives and learners | the same functions, strip list, calls and metric names | no metric prefix with one trainable policy |
| worker requests | protocol 2, unchanged fields, `model_binding.pkl` | protocol 3 only with several sampled policies or agent tags |
| kept groups, dispatch state | today's format versions and tags | the codec table keys on exact type |
| step hooks | none registered: today's step, call for call | the hook phases are skipped, not awaited empty |
| rule 10 | `rl` turns it off: today's interrupted save and log-only optimizer metrics | flipping `rl` is a pull request of its own that records the golden again with a reason |
| `train.stepping`, `run.clocks` | no key, so lockstep; one `StepClock`, the same object as `run.clock` | no per-policy clock in lockstep |
| independent stepping's ledger, horizons, commits, pins, `stepping` in `identity.json`, ages | none: lockstep never builds them | only with `independent` and two or more trainable policies |
| the version rule on resume | rows record no `published_version`, so the row's `policy_version`, as today | the 0.3.0 resume test and the golden's resume scenarios |
| URL tenants, one session per policy | one policy, one tenant, one session | the golden's Tinker call order |
| the W&B run config | gains the new fields' defaults, as every release that adds a config field does | W&B is off in the golden; `config.resolved.yaml` dumps the raw data and is unchanged |

**What stays compatible.**

| Surface | Kept | Until |
|---|---|---|
| config keys and values, `train.*` as `main`'s shorthand | all; no key renamed or newly required | indefinitely |
| `config.resolved.yaml` of supervised restarts | reloads; no `trainer` key means `rl` | indefinitely |
| run directories written by 0.3 to 0.5 | resume | indefinitely |
| `platoon.train` and `platoon.train.workflow` exports (`PlatoonTinkerRLTrainer`, `GroupRolloutWorkflow`, `group_baselines`, the config classes) | the same objects | indefinitely |
| `PlatoonTinkerRLTrainer(config, ...)` and `train(train_workflow, eval_workflow)` | work; `train()` runs `TrainerRun` with the given workflows | indefinitely |
| the 28 `platoon.train.tinker.*` modules | warning shims to the final homes | removed in 0.6.0, as already planned |
| private modules that moved in 0.5.0 and move again | warning aliases | one minor release |
| the old optimizer-update class name | an alias of `PolicyUpdate` | one minor release |
| the RL trainer's private seams that tests patch | delegates to the policy and run calls | until the tests use injected fakes, then one minor release |
| `trainer_config` | works | RFC 0016's own deprecation |
| worker protocol 2 | every single-policy run | indefinitely |

**What users see move:** nothing through the package exports; one move for each of the 28 `platoon.train.tinker.*`
module paths, already planned; no renamed config key, and the added keys of section 14; no change to the resume
files of single-policy runs; no change to persisted class paths.

### 18. Not built yet

| Not built | Why not now | What would bring it |
|---|---|---|
| `leader` stepping (followers parked behind one policy), token-budget batches, a timer for rarely fed policies | `lockstep` and `independent` cover the cases asked for | a run that needs them; a per-policy `batch_tokens` key |
| a slow policy that stops gating dispatch (`policies.<id>.excess: drop`) | rates stay coupled as in lockstep; dropping shares wastes the slow policy's data | open question 14 |
| training on while one policy's service is down (`on_policy_failure: continue`) | the SDK exception that would trigger it also ends the failed policy's session, and miles-tinker then unloads its samplers (U10), so its agents fail too; it would help only failures that raise no SDK exception | a backend that keeps a failed session's samplers serving; open question 15 |
| a name for a shared service (`services.<name>`) and a cap on training requests in flight on it | multi-LoRA needs no key, and nothing measured asks for a limit. A cap must release its slot when the request completes, not when the learner awaits it: a learner submits every microbatch and its `optim_step` before awaiting, so a smaller cap would wait on itself | U8 showing queueing harm |
| a cross-policy staleness bound | each policy judges its own rows; `peer_lag` reports the drift | a run that shows drift matters |
| per-policy permit pools, per-policy health and draw holds | a permit is a rollout, and a rollout samples several policies | RFC 0020's one pool per policy |
| several objectives in one `optim_step` | gradient accumulation across different built-in losses is UNVERIFIED | a conformance test (S3's `mix`) |
| the critic (`gae`, a value objective) | Tinker has no value head | a spike (open question 16) |
| cross-tokenizer distillation | needs other losses | a user |
| RFC 0001's full agent graph, edges, reward events and per-call token spans | only the datum tags have a user | the first per-step credit user |
| entry-point groups for credit estimators and objectives | `module:Class` and registered names work | an outside package that needs a short name |
| one trainer process per policy | credit needs the whole episode, and commits need one writer | — |
| multi-policy `platoon eval` and `platoon rollout` | evaluation during training covers it first | a user |
| a stable trainer, credit or workflow API | each has one or two users so far | promotion after the outside interleaved trainer runs on B0 (open question 17) |
| a framework for producers of online auxiliary data | one user, whose admission and consumption rules are its own | a second user |
| a run-keyed directory for worker IPC | one trainer process holds every policy; it changes a request field of the golden | a small pull request of its own |

### 19. Relation to other RFCs

| RFC | What this RFC does to it |
|---|---|
| [0003](0003-multi-policy-training.md) multi-policy training | **Implements** M1 (B3), M2 in both stepping modes (B4, B4b) and M4, multi-LoRA on one service (B4c). **Keeps** one agent one policy for life, text-only crossing, credit once then route, per-policy versions and atomic per-policy drops, baselines keyed on `[task, policy, role]`, per-policy pruning and frozen policies. **Changes:** (1) the policy table uses today's `train.*` names (`model_name`, `renderer_name`, `tinker_base_url`, `load_checkpoint_path`, `context_window_length`), with `optimizer`, `loss_fn` and `max_staleness` flat on each entry, and `backends` waits for RFC 0006; (2) the fixed loss family becomes a per-policy `loss_fn`; (3) frozen policies may also score; (4) `Learner` becomes a per-policy update, publishing and checkpoints belong to `Policy` and `RunCheckpoints`, and `PolicyCoordinator` becomes `train.stepping` (`lockstep` or `independent`; `leader` is not built); (5) the barrier is the leader's row naming the other policies' rows, in place of `policies.json`, and under `independent` every full save of any policy commits; (6) `open_agent` takes a role, never a policy; (7) multi-LoRA needs no key; (8) under `independent`, rates stay coupled when every group feeds every policy, so "independent for throughput" holds only for a policy that does not bind. **Answers** its open questions 1 (both, lockstep the default), 4 (same tokenizer, else refuse), 5 (separate URLs avoid it; one base model on one URL is M4) and 6 (no barrier is needed: a commit names every policy's newest durable row) |
| [0001](0001-native-multi-agent.md) native multi-agent | **Replaces** its per-agent credit (section 7): `CreditPolicy` and the `BaselineSpec` presets, which run inside a `GroupRolloutWorkflow` subclass and give one number per agent, become `CreditEstimator`: asynchronous, chosen per entry, usable by any workflow and per-token capable. Its `rao` preset is `group`, byte for byte, and `keyed` covers the `BaselineSpec` key subset. **Implements** `RoleRule.policy` as `agents.roles.<role>.policy`, plus datum tags. **Leaves** the agent graph, edges, reward events, spawning and verification as they are |
| [0013](0013-extension-model.md) extension model | **Adds** two component kinds under its one discovery rule, trainers and credit estimators, with reserved core names (`rl`, `sft`, `interleaved`; `group`, `keyed`, `reverse_kl`), `lock.json` rows and the `requires` check, and the entry-point group `platoon.trainers` |
| [0016](0016-component-configuration.md) component configuration | **Applies** it: `trainer` with `trainer_options` and `credit` with `credit_options` are selectors with sibling options, checked at load through `options=` and `check=`. **Documents** one asymmetry: the `rl` trainer's settings stay in `train:`. The `trainer_config` deprecation is unchanged |
| [0017](0017-public-surface-and-conformance.md) public surface | **Follows** it: every new API starts in `platoon.experimental` with no `tinker` import at load, and `platoon.testing` gains `check_trainer`, `check_credit_estimator` and `check_shared_service`. Nothing is promoted to stable yet (open question 17) |
| [0019](0019-training-mixtures.md) training mixtures | per-entry `credit` follows its per-entry pattern; per-entry bounds meet per-policy bounds by the precedence in section 10; versions were already keyed by policy |
| [0020](0020-rollout-permits.md) rollout permits | the permit pool stays one per run, also under independent stepping, whose horizon becomes one per policy; P5 lands before B4; its unit plan stays deferred; its PPO blockers are met except the critic itself |

## The hard parts

| Risk | Likelihood / impact | Mitigation |
|---|---|---|
| An indirection changes S8: a forwarding property, an extra `ServiceClient`, an await that yields, a new `lock.json` row, an INFO line | medium / high | a guard table in every pull request; the golden trace in every pull request; default paths stay synchronous; new lines at DEBUG |
| Lifting the lifecycle (A4) or the rollout source (A5) reorders an await or a log line | medium / high | separate pull requests, each lifting today's phases verbatim at phase boundaries, with the old seams kept as delegates |
| Rows routed to the wrong policy | medium / high | tags made in the loop that makes the datum; `TrainingSample` objects instead of aligned lists; a row without a policy raises; a CPU test with two fake services whose tokenizers differ |
| Barrier inconsistency after a crash | low / high | the leader's row commits last; resume requires every named row and falls back to the newest consistent one |
| Lockstep runs at the slowest learner's pace | high / medium | optimizer steps overlap across services, and rollouts run during steps. Independent stepping does not lift group throughput when the slow learner binds; it removes the fast policy's wait on the slow one's training and saves |
| The share ledger miscounts: a share trained twice into one policy's lineage, or lost | medium / high | one ledger owns consumption and retirement; settling and capture have no await; reverted shares keep their ages; tests kill the run at every point between consume, capture, save and append |
| A commit names a row that a pruner deletes while the commit's save runs | medium / high | pins under the commit lock; deleted paths count as missing at resume; a test prunes while a commit is in flight |
| Retained groups grow when a policy saves rarely | medium / medium | retention is bounded by the shares consumed since each policy's last full save; `run/retained_groups`; per-policy recovery cadence |
| Cross-policy drift under independent stepping | medium / medium | lockstep stays the default; `peer_lag` metrics; a cross-policy bound if a run shows it matters |
| A shared service couples its tenants: queueing, slower steps, one outage for all | high / medium | one session per policy; per-policy latency metrics; U8 measured before the docs recommend sharing a training-bound pair |
| Tinker unknowns: multi-LoRA (U1-U10), mixed losses in one step, a critic, teacher logprob throughput on miles-tinker | unknown / medium | each is a conformance test or spike that gates only its own scenario's docs |
| APIs designed before their second user | medium / medium | each ships with a second user in core (`rl` and `sft`, `group` and `keyed`, `PolicyGradient` and `CrossEntropy`) and stays experimental |
| One trainer process drives several services | medium / medium | the process holds clients and bookkeeping only; the services do the GPU work; per-policy metrics show where time goes |
| Side inference straddles a publication on a full fine-tuning service, whose samplers serve the newest weights | medium / medium | `drain()` runs before any update or publication of the step; `PolicySnapshot.sample` refuses a sample whose version moved |

## Alternatives

- **Keep one trainer and add flags.** SFT, distillation, interleaving and several policies as options of the RL
  trainer. Each would add branches to the step, and S4's outside trainers would still subclass private methods.
- **One platoon process per policy, exchanging checkpoints** (RFC 0003's alternative). Isolates failures, but credit
  needs the whole episode, and consistent checkpoints need a cross-process barrier.
- **RFC 0003's config names** (`base_model`, `renderer`, `init`, `backend`, a nested `learner:`). A second vocabulary
  for the same fields; today's `train.*` names serve every policy, and `backends` waits for RFC 0006.
- **Credit as one policy per run** (RFC 0001's `CreditPolicy` inside a workflow subclass). It gives one number per
  agent and needs a subclass; per-entry estimators serve mixtures (RFC 0019) and per-token credit.
- **Only the leader's saves commit under independent stepping, with a barrier at save points**, as miles does: every
  loop pauses at a step boundary while all policies save. It needs no retention, revert, commit numbers, version rule
  or pins, and costs the fast policy idle time during each of the slow policy's saves (minutes for a 35B). It is the
  fallback if the commit protocol proves too costly (open question 11).
- **One `ServiceClient` per URL** with a training client per policy: fewer heartbeat streams, but one adapter's fatal
  error would end the shared session and delete its co-tenants' models.
- **A run-wide W&B step** under independent stepping. Per-policy axes declared with `define_metric` make it
  unnecessary.
- **Doing nothing.** Users copy the trainer (the outside interleaved trainer did, and stayed on 0.2.1), and S2, S5 and
  S7 are not possible.

## Rollout plan

Milestones A1 to A5 are refactors, each keeping the golden trace identical; B0 to B6 are features, each keeping it
identical and adding scenarios of its own. Line counts are EST.

| Milestone | Scope | Exit check | Pull requests |
|---|---|---|---|
| A1 the workflow contract | `capabilities_of`, `OutputReport` and `report_of`, `RolloutOutput`, `rows_by_policy`, the kept-group codec table; today's duck-typed reads and `isinstance` checks replaced (+400/−150) | the golden trace identical; old workflow classes read the same through the legacy fallback | |
| A2 the policy object | `policy/` with `main` only: spec, policy, set, service, one policy's checkpoints, binding; the trainer's state fields become the leader's properties (~1,050) | the golden trace identical | |
| A3 learners and objectives | `PolicyUpdate` with the old name as an alias, `PolicyGradient`, the strip list from the objective (~300) | the golden trace identical | |
| A4 one lifecycle | `Trainer`, `TrainerBase`, `TrainerRun`, `StepClock`, `trainer:` (`rl` only), `RunCheckpoints` and participants, `check_trainer`, the import-layer test with an allow-list (+600/−400) | the golden trace identical; `check_trainer` passes on `rl` | |
| A5 the rollout source | `RolloutSource`, `run.rollouts()`, workflows built by the run, `WorkflowSetup` and `from_setup`; the allow-list emptied (+400/−300) | the golden trace identical; the import-layer test passes with no exception | |
| B0 step hooks | section 9; rule 10; the participant rules; `close` in `finally`; `PolicySnapshot`; `load_optimizer_state`; draw provenance (~600) | a CPU test whose hook runs a cross-entropy update on the fake service and publishes once; `check_trainer`'s hook and interrupt cases; the outside interleaved trainer runs on it | |
| B1 SFT | S1 (~550) | an SFT run on the fake service saves, resumes and evaluates; pruning at `k = 0` | |
| B2 credit estimators | the credit API, `keyed`, per-entry `credit`, async credit, datum tags, worker protocol 3 (tags only), `PolicyGroupOutput`, `check_credit_estimator` (~800) | `group` reproduces today's advantages; `check_credit_estimator` passes on the built-ins | |
| B3 sampled and frozen policies | S2 and RFC 0003 M1: `policies:`, `agents:`, frozen policies, a gateway runtime and binding per policy, per-policy stamps, the identity table, `Policy.score`, `reverse_kl` (~1,200) | a CPU test with two fake samplers whose tokenizers differ: no agent crosses policies, and every datum lands at its policy | |
| P5 per-datum stale masks | RFC 0020's P5 | as RFC 0020 says | |
| B4 lockstep | S7 in lockstep: learners per policy, the per-policy stale filter, the horizon's `k`, barrier saves and their fallback, per-policy pruning, metrics and evaluation versions (~900) | a CPU test with two fake services: both policies train; a kill between the two saves resumes from the last barrier | |
| B4b independent stepping | section 11 (~1,600 with its tests) | CPU tests on two fake services: stepping 1:4; a kill at every point between consume, capture, save and append trains each share once per lineage; a commit naming another policy's row four steps old retrains every reverted share, none dropped as stale; a prune while a commit is in flight keeps the named row; commits appended out of order and the next number after a resume; a resume that changes the stepping exits 78; a stateful estimator killed between `assign` and `settle` | |
| B4c multi-LoRA | section 12: URL tenants, one session per policy (also a teacher), capacity refusals retried in the open window, the load warnings, `check_shared_service` (~400); after B4b | a CPU test with two adapters on one fake service in both stepping modes; U6 as a fast test; `check_shared_service` on a miles-tinker LoRA gateway (U1-U5, U10) and on hosted Tinker (U7) before the docs claim support | |
| B5 interleaved | S3 with `placement: step` and `boundary` (~480) | the horizon it chooses never lets a group go stale under its schedule; `mix` only if the mixed-loss conformance test passes | |
| B6 the critic | S5: `gae` and a value objective (~300), after a spike | the spike shows a value read-out trains on both backends | |

A1 can run beside A2 and A3; B1 beside B2 and B3. B0 comes first among the features, because an outside trainer is
waiting on it.

## Open questions

1. **How do policies map to processes?** *Decided 2026-10-26:* one platoon trainer process drives several Tinker
   services, one URL per policy, so credit sees the whole episode and one writer commits checkpoints.
2. **How do several policies step?** *Decided 2026-10-26:* both ways, as `train.stepping`, with `lockstep` the
   default (this answers RFC 0003's open question 1).
3. **Multi-LoRA on one service?** *Decided 2026-10-26:* as an option, with no key: policies with the same base model
   on one URL are its adapters (RFC 0003's M4).
4. **The config shape.** *Decided 2026-10-26:* `policies.<id>` with today's `train.*` field names, `train.*` kept as
   `main`'s shorthand indefinitely, and credit chosen per environment entry.
5. **The default credit with two or more trainable policies.** *Decided 2026-10-26:* `keyed` on
   `[task, policy, role]` (RFC 0003; Dr. MAS found per-agent normalization better than global).
6. **A policy's service fails mid-run.** *Decided 2026-10-26:* stop the start, and resume every policy from the last
   complete barrier or commit, in both modes.
7. **A `platoon.trainers` entry-point group.** *Decided 2026-10-26:* add it now, since outside packages already write
   trainers; its names follow RFC 0013's naming rule.
8. **Rule 10 for `rl`.** *Decided 2026-10-26:* `rl` keeps today's behaviour, so S8 stays byte-identical; every other
   trainer gets the rule, and a subclass of `rl` turns it on in `needs()`. Flipping `rl` is its own pull request.
9. **The built-in trainer's name.** *Decided 2026-10-26:* `rl`, since with pluggable credit it covers GRPO,
   distillation, PPO and several policies.
10. **Recording the design.** *Decided 2026-10-26:* as this RFC, with notes on RFCs 0001 and 0003 that it supersedes
    them in part.
11. **Commits under independent stepping.** Proposed: every full save of any policy commits, naming every other
    policy's newest durable row, with consumed shares kept until durable, so each policy checkpoints on its own clock.
    The alternative is miles' barrier at save points (Alternatives). Decide before B4b.
12. **Tinker sessions on a shared service.** Proposed: one session per policy, so one adapter's fatal error cannot
    delete another's model. Decide before B4c.
13. **Rows of another policy as staleness.** Proposed: a policy judges only its own rows, and `peer_lag` reports the
    drift. A cross-policy bound waits for a run that shows it matters.
14. **Should a slow policy be able to stop gating dispatch?** With `policies.<id>.excess: drop`, its horizon would
    stop gating the dispatcher, and its step would train the freshest `batch_size` shares and drop the rest, counted,
    so a fast policy could run ahead at the cost of the slow one's data (EST 150 lines in B4b). Proposed: not yet;
    rates stay coupled, as in lockstep, and the horizon metric names the policy that binds. Decide before B4b.
15. **Training on when one policy's service fails.** Proposed: not yet (section 18). It returns if a backend keeps a
    failed session's samplers serving (U10), or for failures that raise no SDK exception.
16. **The critic.** Can hosted Tinker and miles-tinker train a value read-out through `forward_backward_custom`? A
    spike decides, before B6.
17. **Promotion to stable.** Proposed: `Trainer`, `TrainerBase`, `StepHook`, `CheckpointParticipant` and
    `PolicySnapshot` are proposed for the stable surface in the minor release after the outside interleaved trainer
    runs on B0; until then RFC 0017's rule holds (experimental names change only in a minor release).
