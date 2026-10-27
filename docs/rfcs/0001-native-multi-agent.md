# RFC 0001: Native multi-agent and swarm support

| | |
|---|---|
| Status | Draft; superseded in part by [RFC 0021](0021-trainers-credit-and-policies.md) |
| Created | 2026-10-13 |
| Builds on | [Architecture](../architecture.md): the model gateway and its agent API, subagents, credit and forks |

**Superseded in part by [RFC 0021](0021-trainers-credit-and-policies.md)**, which replaces the per-agent credit of
section 7. `CreditPolicy` and the `BaselineSpec` presets, which run inside a `GroupRolloutWorkflow` subclass and give
one number per agent, become RFC 0021's `CreditEstimator`: asynchronous, chosen per environment entry
(`environments[].credit`), usable by any workflow and able to set per-token advantages. The `rao` preset is its
`group` estimator, and its `keyed` estimator covers the `BaselineSpec` key fields. `RoleRule.policy` becomes
`agents.roles.<role>.policy`. The rest of this RFC (the agent graph, edges, reward events, spawning, messaging and
verification) stands as written.

## Summary

Make a rollout an **agent graph**: agents with a role, depth, policy, harness and attribution level; typed edges such as `spawn`, `return`, `message`, `verify` and `informed_by`; and reward events anchored to agents and model calls. Harness code mints exact child agents by extending the gateway's `POST /v1/platoon/agents`; the policy spawns through MCP tools; native subagents of Codex, Claude Code and OpenHands are observed through declarative selectors; agents talk over a rollout-scoped message bus. Credit becomes role-aware through a credit policy that `GroupRolloutWorkflow` calls, with a preset that reproduces today's RAO numbers exactly. Verification becomes a role that any harness can play on a `copy` fork. Platoon's own multi-agent harness is the reference consumer, and an external recursive harness that brings its own agent loop maps onto the same primitives (§9).

## Motivation

Platoon records multi-agent rollouts as a `TrajectoryCollection` with parent links. Identity is exact only for agents platoon itself runs: subagents started with `ctx.agents.spawn` and agents minted at the gateway. That leaves six gaps:

1. **Harness-internal subagents collapse into their parent.** Codex's `spawn_agent`, Claude Code's Task tool and OpenHands' delegate tool share the parent's key, so their calls train as the parent's.
2. **Only trees exist.** Peer swarms (DeLM, GitSwarm), messages between agents and contribution DAGs (GitSwarm's `informed_by` commits) have no representation.
3. **No asynchronous fan-out.** Orchestrators that start many children and join later (Kimi's PARL, Prime Agent's `rlm()`) need `spawn` and `await` separately.
4. **Credit mixes roles.** Every agent centers on the root group's leave-one-out baseline. Mixing roles in one baseline destabilizes training or rewards structural asymmetry (Dr. MAS, arXiv 2602.08847; SPIRAL's role-conditioned baselines), and siblings that share a workspace leak the credited agent's own actions into its baseline.
5. **GRPO by task does not scale to swarms.** One GitSwarm run spends 100-500 worker episodes, so a group of eight swarms per task is out of reach; credit has to work inside one swarm.
6. **Verification is tied to in-process agents.** A verifier is a forked copy of the running in-process agent (for example an OpenHands agent); a harness in a sandbox can neither verify nor be verified.

No existing framework gives each agent of an unmodified harness its own exact identity. That combination, with any Tinker backend, is what platoon can offer.

## Proposal

### 1. The agent graph

```python
Attribution = Literal["exact", "declared", "inferred", "forked"]
EdgeKind = Literal["spawn", "return", "message", "verify", "judge", "informed_by", "handoff", "compaction", "cancel"]
Purpose = Literal["policy", "verifier", "judge"]


@dataclass
class AgentRecord:
    agent_id: str  # "r/n0", "r/n0/c1", "r/n0/c1/verify", "r/peer2"
    parent_id: str | None
    role: str  # root, child, verifier, solver, peer, orchestrator, ...
    purpose: Purpose
    depth: int
    ordinal: int | None  # sibling or seat index; None in dynamic swarms
    policy: str  # a policy id (RFC 0003); "main" by default
    harness: str  # name@version
    attribution: Attribution
    fork_mode: ForkMode  # how its env session relates to its parent's
    outcome: Outcome | None = None


@dataclass(frozen=True)
class Edge:
    kind: EdgeKind
    src: str
    dst: str
    at_call_id: str | None  # the model call or tool event that caused it
    attribution: Attribution
    data: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RewardEvent:
    agent_id: str | None  # None: the whole rollout (a team reward)
    name: str  # outcome, verdict, behavior_gate, tests_passed, ...
    value: float | None  # None means no reward, never 0
    kind: Literal["terminal", "increment", "gate"]
    source: str  # env:<service>, verifier:<harness>, judge:<name>, stage:<name>, harness
    at_call_id: str | None = None


@dataclass
class AgentGraph:
    rollout_id: str
    agents: dict[str, AgentRecord]
    edges: list[Edge]
    rewards: list[RewardEvent]

    def children(self, agent_id: str, kinds: frozenset[EdgeKind] = frozenset({"spawn"})) -> list[AgentRecord]: ...
    def ancestry(self, agent_id: str, kinds: frozenset[EdgeKind] = frozenset({"informed_by"})) -> set[str]: ...
    def to_trajectory_collection(self, calls: Mapping[str, Sequence[str]]) -> TrajectoryCollection: ...
```

`to_trajectory_collection` replaces `gateway.trace.build_trajectory_collection`: one trajectory per agent, one step per call, parent links from `spawn` edges, and the new fields in each trajectory's `misc`. Prefix merging and the workflows therefore keep working while credit learns to read roles. Env tools may report edges in result metadata (`_meta.platoon.edges`), so a GitSwarm-style env declares `informed_by` without platoon knowing git.

### 2. Attribution levels

| Level | Established by | Trains by default |
|---|---|---|
| `exact` | the agent has its own secret: roots, program seats, verifiers, and children spawned through the agent API | yes |
| `declared` | a cooperating harness labels calls made with a shared secret (`X-Platoon-Agent: <id>`) | no |
| `inferred` | HarnessSpec `[native]` selectors over native headers or bodies (§5) | no |
| `forked` | no signal: the calls branch from the parent's context | as the parent's rows, as today |

An agent takes the weakest level of any of its calls. `credit.train_attribution` (default `{"exact"}`) lists the levels that train. A secret used on another agent's endpoint is an error, never a silent reattribution.

### 3. The cooperative agent API

The gateway's `POST /v1/platoon/agents` mints a child from `{role, goal?}`. This RFC grows it into a small API that harness code calls over HTTP and the policy calls through MCP tools; both reach one primitive.

| Route (under the agent's gateway base) or MCP tool | Purpose |
|---|---|
| `POST /v1/platoon/agents` | spawn: `launch=platoon` (platoon starts the child's harness), `external` (returns credentials; the caller runs the child), `attach` (labels a native agent) |
| `GET /v1/platoon/agents/{id}`, `POST .../wait`, `POST .../cancel` | status and final message (never the verdict); cancel drains in-flight tool calls |
| `POST /v1/platoon/agents/{id}/finish` | end an agent with a message |
| MCP `launch_subagent`, `finish`, `submit_verdict` | synchronous spawn with progress; terminal tools |
| MCP `spawn_agent`, `await_agents` | asynchronous fan-out and join |
| MCP `send_message`, `read_messages` | the agent bus (§4) |

```python
@dataclass(frozen=True)
class SpawnRequest:
    request_id: str  # idempotency: a retry returns the same handle
    goal: str | None = None
    role: str = "child"  # must be in the caller's role's may_spawn
    launch: Literal["platoon", "external", "attach"] = "platoon"
    fork: ForkMode | None = None  # None: the role's default
    limits: Mapping[str, int] | None = None  # may only narrow the role's limits
    native_id: str | None = None  # attach only


@dataclass(frozen=True)
class ChildCredentials:  # launch=external; only a fingerprint is ever logged
    api_key: str = field(repr=False)
    base_url: str  # the child's own gateway base
    env: Mapping[str, str]  # OPENAI_BASE_URL, OPENAI_API_KEY, PLATOON_AGENT_API, PLATOON_AGENT_ID


@dataclass(frozen=True)
class AgentHandle:
    agent_id: str
    role: str
    policy: str
    credentials: ChildCredentials | None = None


@dataclass(frozen=True)
class ChildReport:
    agent_id: str
    status: Literal["running", "completed", "limit", "timeout", "error", "cancelled"]
    reason: str | None = None  # no_submit, budget, ...
    message: str | None = None


class AgentAPI(Protocol):  # AgentAPI.from_env() inside a sandbox; stdlib plus httpx
    async def spawn(self, goal: str | None = None, **request: Any) -> AgentHandle: ...
    async def wait(self, agent_id: str, timeout_s: float = 240) -> ChildReport: ...  # "running" on timeout
    async def join(self, *agent_ids: str, timeout_s: float | None = None) -> list[ChildReport]: ...
    async def finish(self, agent_id: str, message: str) -> None: ...
    async def cancel(self, agent_id: str) -> None: ...


@dataclass
class RoleRule:  # config: agents.roles.<role>
    purpose: Purpose = "policy"
    policy: str = "main"  # config binds policies; a request never does
    harness: str = "inherit"
    fork: ForkMode = "shared"
    may_spawn: tuple[str, ...] = ()  # roles this one may create; grants launch_subagent
    launch_modes: frozenset[str] = frozenset({"platoon"})
    max_calls: int = 100
    verify: "VerifySpec | None" = None  # §6
    bus: tuple[str, ...] = ()  # message channels (§4)
```

Tree limits (`agents.spawn.max_depth`, `max_live_agents`, a shared call pool) are reserved at admission, so a refusal caused by the tree's own actions is a deterministic tool error, and cluster load never reaches the policy.

### 4. Messaging

A role with a non-empty `bus` gets `send_message(text, to=None, channel=None)` and `read_messages(after=0, channel=None)`, scoped to one rollout. Each delivery records an exact `message` edge from writer to reader, anchored at the sending call; when the text reappears in a reader's prompt, the gateway records the delivery anchor. Shared boards that belong to the task (DeLM's append-only context) stay env tools and report edges through `_meta.platoon.edges`.

### 5. Harness-native subagents

Harnesses that run subagents in one process cannot re-key them, so each HarnessSpec maps them with data:

```python
@dataclass(frozen=True)
class Selector:
    source: Literal["header", "body", "tool_args"]
    path: str


@dataclass(frozen=True)
class NativeAgentMap:  # HarnessSpec [native]
    agent_id: tuple[Selector, ...]  # first hit wins; a miss keeps the call on the parent
    parent_id: tuple[Selector, ...] = ()
    spawn_tools: tuple[str, ...] = ()
```

| Harness | Agent id | Parent id | Level |
|---|---|---|---|
| Codex | header `thread-id` | header `x-codex-parent-thread-id` | inferred |
| Claude Code | header `x-claude-code-agent-id` (hint headers on) | header `x-claude-code-parent-agent-id` | inferred |
| OpenHands | header `x-litellm-session-id` per sub-conversation | the delegate call that named it | inferred; exact through a delegate factory that keys each sub-agent's LLM with child credentials |
| pi subagent extension | n/a | n/a | exact: the extension calls the agent API |

Native children are recorded but train only when allow-listed in `credit.train_attribution`; their verification is a post-hoc transcript judge, because the parent does not wait for platoon.

### 6. Verification as a role

A verifier is an agent of a `verifier`-purpose role, so harness, policy, limits and training come from one place. `VerifySpec` says when and how to verify:

```python
@dataclass(frozen=True)
class VerifySpec:  # agents.roles.<role>.verify
    role: str = "verifier"
    mode: Literal["snapshot_async", "sync", "post_hoc_transcript", "off"] = "snapshot_async"
    fallback: Literal["sync", "post_hoc_transcript", "none"] = "sync"  # env cannot snapshot
    goal: str = "template:platoon/verify-v1"  # rendered over an evidence pack
    evidence: frozenset[str] = frozenset({"goal", "final_message", "action_log"})
    verdict: str = "platoon/verdict-v1"  # a registered parser
    on_unparseable: Literal["invalid", "zero"] = "invalid"
    deadline_s: float = 3600.0
```

`snapshot_async`, the default, forks the child's session with `copy` at exit, returns to the parent at once and verifies on the copy. An env that cannot snapshot falls back to `sync`, which verifies on a `shared` fork before the child returns, as today's verifiers do; configs that must reproduce today's numbers set `sync` explicitly. The verdict becomes a `verdict` reward event on the verified agent; an unparseable verdict makes the agent `invalid`, never 0, and a verifier crash is `error`. Evaluation turns verification off (`mode: off`), which generalizes `enable_subagent_verifiers: false`.

### 7. Per-agent credit

Credit runs once per group over all agents, inside a `GroupRolloutWorkflow` subclass whose `compute_baselines` and `assign_advantages` (today's estimator seam) delegate to a credit policy:

```python
BaselineField = Literal["task", "policy", "role", "depth", "ordinal", "harness"]


@dataclass(frozen=True)
class BaselineSpec:
    key: tuple[BaselineField, ...] = ("task",)
    source: Literal["roots", "same_key"] = "roots"  # roots: the completed roots' baseline (RAO)
    estimator: Literal["loo", "mean", "ema"] = "loo"
    normalize: Literal["none", "key_std"] = "none"
    exclude: Literal["none", "same_rollout"] = "none"  # keep state-sharing siblings out
    min_peers: int = 1
    fallback: tuple[tuple[BaselineField, ...] | Literal["roots", "zero"], ...] = ("roots",)


class CreditPolicy(Protocol):
    def returns(self, graph: AgentGraph) -> Mapping[str, float | None]: ...  # agent -> return
    def advantages(
        self, group: Sequence[AgentGraph], returns: Sequence[Mapping[str, float | None]]
    ) -> Mapping[tuple[str, str], float]: ...
```

| Preset | Key and source | Reproduces |
|---|---|---|
| `rao` (default for migrated configs) | `[task]`, `roots`, `loo` or `mean` per `leave_one_out_baseline` | today's advantages byte for byte; depth weighting stays a batch transform |
| `by_role` | `[task, policy, role, depth]`, `same_key`, fallback to `[task, policy, role]` then `roots`; `exclude: same_rollout` | per-role GRPO |
| `hierarchical` | solvers compared within one proposed problem | prime-rl's hierarchical GRPO |
| `rae` | `[role, ordinal]`, `ema` | SPIRAL; self-play and swarms where groups are too costly |
| `agentwise` | `[task, policy, role]`, `normalize: key_std` | Dr. MAS |

Within-swarm credit comes later on the same record: per-seat EMA, ancestry credit over `informed_by` edges, and counterfactual continuation from `copy` forks (C3, arXiv 2603.06859).

### 8. Platoon's own multi-agent harness

An inline loop that runs no model-written code itself, with spawn, join, persistent children, the bus, `finish` and `submit_verdict`, all through the gateway. Code steps spawn CodeAct children in their own sandboxes. Every edge it creates is `exact`, so ancestry, message and counterfactual credit can be developed on CPU with a fake sampler before meeting third-party harnesses. Its prompts ship with the packages that use it.

### 9. Mapping an external recursive harness

A harness that brings its own agent loop (for example one built on OpenHands) can add recursion, programmatic tool calling (PTC) and reward shaping outside platoon's core. Each such feature maps onto a primitive above:

| Harness feature | Primitive | Notes |
|---|---|---|
| `launch_subagent(goal)` tool and `atools.launch_subagent` in PTC | `POST /v1/platoon/agents` with `launch=external` from the harness's subagent factory; the child's LLM is keyed with its credentials | exact per child; children share the parent's sandbox (`fork: shared`) |
| a depth limit, live-agent and call budgets | `may_spawn`, `agents.spawn.max_depth`, `max_live_agents`, role limits | refusals stay deterministic tool errors |
| `finish(message)`; forced zero without it | `finish` tool or route; `scored/no_submit` | rewards unchanged |
| verifier: an OpenHands subagent without terminal tools on the live workspace | `verifier` role, `harness: inherit`, `mode: sync`; later `snapshot_async` | the harness package registers its verifier goal and verdict parser |
| behavior judge, parent-usage gate | reward stages emitting `gate` events | the harness package's code over the graph |
| delegation and critical-path efficiency rewards | `increment` events from its reward stages | the harness package |
| leave-one-out on root rewards, depth weighting, child datum sampling | `rao` preset, batch transform, sampler | rewards unchanged |
| PTC, cell guard, stream isolation | inside the harness, in the sandbox; PTC inner calls anchor through `parent_tool_call_id` | the harness package |
| OpenHands error and message detection | the hooks `trajectory.error_completion_ids` and `trajectory.message_completion_ids` | unchanged |

## Alternatives

- **Recursion inside each harness, as today.** Fast to ship, but every harness re-implements identity, budgets, verification and credit, and harness-internal agents stay invisible.
- **Fold subagents into one trajectory** (Polar, SkyRL's skycap). Simple records, but no per-agent credit and no role-aware baselines.
- **One trace per program seat.** Exact for seats a program launches; harness-internal children remain branches.
- **Hardcoded per-harness parsers in core.** Each new harness version needs core code; selectors make it data.
- **Adopt a multi-agent RL framework** (MARTI). It owns the trainer; platoon trains on any Tinker backend.

## Rollout plan

| Milestone | Scope | Exit check |
|---|---|---|
| M1 graph record | `AgentGraph` built from today's agent tree; role, attribution, policy and fork fields; the `rao` preset | advantages byte-identical to today's on recorded and synthetic trees |
| M2 exact children | `launch=external`, `request_id`, limits, wait, finish, cancel; an external harness spawns through the API | a recorded recursive rollout replays with the same child counts and rewards; no leaked agents after 100 cancellations |
| M3 verification role | `VerifySpec` with `sync`, then `snapshot_async` on `copy` forks; verdict parsers | recorded verdicts parse identically; `invalid` and `error` never train as 0 |
| M4 fan-out and role credit | `spawn_agent`, `await_agents`; `by_role`, `hierarchical`, `agentwise` | a CPU test plants a sibling-coupled reward and shows the bias vanish under `exclude: same_rollout` |
| M5 native observation | selector engine; Codex and Claude Code fixtures | inferred agents recorded; none train unless allow-listed |
| M6 bus and swarms | agent bus, persistent children, platoon's own harness, `rae` and ancestry credit | a peer-swarm example trains on CPU with a fake sampler |

## Open questions

1. Should `declared` attribution train by default for harness versions that passed a doctor check?
2. How should a parent and its external children share a call budget: one pool per tree, or fixed shares per agent?
3. Where do 100-child trees run: inside the parent's sandbox (no isolation between siblings) or in runner sandboxes (EST 30-60 GB per 100 children)?
4. Which message-credit method comes first: ancestry over edges or counterfactual continuation?
5. Should children default to `copy` instead of `shared` once snapshots are cheap (RFC 0004)?
6. When a verifier policy trains (RFC 0003), what does it optimize: agreement with ground truth where it exists, or with a stronger frozen judge?
