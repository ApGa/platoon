# RFC 0003: Multi-policy training across model sizes and architectures

| | |
|---|---|
| Status | Draft |
| Created | 2026-10-13 |
| Builds on | [Architecture](../architecture.md): the model gateway, credit and prefix merging; [RFC 0001](0001-native-multi-agent.md) roles; [RFC 0006](0006-backend-profiles-and-sdk-upgrade.md) backends |

## Summary

Train several policies in one run, where each policy is its own model: a base model of any size or architecture family, its own tokenizer and renderer, LoRA or full fine-tuning, and its own Tinker backend, which may be a different service on different hardware. Roles bind agents to policies, and an agent keeps one policy for its whole life, so tokens never cross tokenizers; policies exchange only text. The gateway holds a sampler and renderer per policy. Credit runs once per group over all agents, rows are routed by policy to one learner per trainable policy, and a coordinator decides when each learner steps and keeps checkpoints consistent. Admission and placement account for each policy's capacity and staleness. Multi-LoRA on one service stays as an optimization for policies that share a base model.

## Motivation

Multi-policy training must cover different model sizes and architectures, not just several LoRA adapters on one base. The use cases need it:

| Setup | Policies | Precedent |
|---|---|---|
| A large orchestrator with small workers | a 35B MoE root that delegates, 4B dense children that do the work | Kimi's PARL trains the orchestrator against frozen workers; M-GRPO trains a planner and sub-agents as separate LLMs |
| Solver and verifier co-training | a solver and a verifier of a different size or family | miles' solver-verifier example; AstraFlow's actor-and-verify workflow |
| Per-role specialists | one policy per role in a fixed pipeline | AT-GRPO: per-role policies win on coding; swapping trained role policies collapses accuracy |
| Frozen judges and opponents | a trainable policy beside frozen checkpoints or API models | prime-rl pins non-trained seats to frozen endpoints |

Cost alone can justify the first row: when children make most of a tree's calls, a small model samples them several times cheaper than the root's model.

Platoon cannot express any of this today. Its gateway holds one sampler and one renderer, the trainer holds one training client, staleness is one version counter, and every row trains one policy.

## Proposal

### 1. The policy table

```python
class PolicyEntry(BaseModel):  # YAML policies.<id>; `policy:` stays shorthand for {main: ...}
    base_model: str  # "Qwen/Qwen3.6-35B-A3B", "Qwen/Qwen3-4B-Instruct-2507", ...
    backend: str | None = None  # a backends.<name> entry (RFC 0006); None: the only backend
    renderer: str | None = None  # a cookbook or registered renderer; required unless an API model
    trainable: bool = True
    adapter: Literal["lora", "full"] = "lora"
    lora_rank: int = 32
    init: str | None = None  # tinker://<checkpoint>: warm start
    context_window: int
    sampling: "SamplingDefaults" = SamplingDefaults()
    r3: Literal["off", "prefer", "require"] = "off"
    learner: "LearnerConfig | None" = None  # overrides of trainer-wide defaults, except the loss family


class LearnerConfig(BaseModel):
    learning_rate: float | None = None
    batch_tokens: int | None = None  # batch by tokens: step time stays bounded for large models
    batch_rows: int | None = None
    max_staleness: int | None = None
    checkpoint_every: int | None = None
```

Three kinds of entry share the table: **trainable** (LoRA or full), **frozen** (`init` or a base model with `trainable: false`; sampling only) and **API** models (`base_model: "api:<provider>/<model>"`, through a passthrough sampler; never trained). The config validator checks each entry against its backend's profile: train modes, adapter limits, modalities and R3 carriers. Two policies may share a backend; a backend that serves only one base model (such as [miles-tinker](https://github.com/neulab/miles-tinker), a self-hosted server built on miles) forces separate backends for different bases.

### 2. Agents bind to one policy

`agents.roles.<role>.policy` binds roles to policies ([RFC 0001](0001-native-multi-agent.md)); harness requests never pick a policy. **An agent uses one policy for its lifetime**, because its prompt tokens, logprobs, prefix merging and routing replay are all specific to one tokenizer and one model. Information crosses policies only as text: a child's final message, tool results, bus messages and verifier evidence packs. A harness that wants two models inside one process makes the second a child agent (`launch=external`), which keeps attribution exact.

### 3. The gateway serves several policies

```python
@dataclass
class PolicyRuntime:
    policy_id: str
    sampler: "SamplerLike"  # a Tinker SamplingClient, a fake, or a passthrough client
    renderer: "Renderer"
    tokenizer: "Tokenizer"
    sampling: "SamplingDefaults"
    version: int
    max_inflight: int  # per-policy concurrency; calls queue, never 429


class ModelGateway:
    def __init__(self, policies: Mapping[str, PolicyRuntime]) -> None: ...
    def open_agent(
        self,
        *,
        rollout_id: str,
        trajectory_id: str,
        policy: str = "main",
        parent_agent_id: str | None = None,
        role: str = "policy",
        interactions: dict[str, Any],
    ) -> "AgentSession": ...
    def set_policy(self, policy_id: str, sampler: "SamplerLike", version: int) -> None: ...
```

Every call record stores the policy id and its start and end versions. `GET /v1/models` lists only the agent's own policy, under its usual alias.

### 4. Rows, credit and learners

Prefix merging runs per agent, so every datum belongs to one policy. The workflow returns rows grouped by policy:

```python
PolicyBatch = Mapping[str, Sequence[tinker.Datum]]  # policy id -> datums, advantages final


class Learner(Protocol):
    policy_id: str
    version: int

    async def submit(self, datums: Sequence[tinker.Datum]) -> None: ...
    async def maybe_step(self) -> "StepReport | None": ...  # forward_backward + optim_step when its batch is full
    async def publish(self) -> "PolicyRef": ...  # new sampler for the gateways
    async def checkpoint(self, name: str) -> "CheckpointRef": ...


class PolicyCoordinator:
    mode: Literal["independent", "lockstep", "leader"]
    learners: Mapping[str, Learner]  # one per trainable policy

    async def feed(self, batch: PolicyBatch) -> None: ...
    async def step(self) -> list["StepReport"]: ...
    async def checkpoint(self) -> "MultiPolicyCheckpoint": ...  # a barrier; writes policies.json
```

- **Credit once, then route.** Credit runs once per group over all agents, so team rewards and verdicts see the whole episode; then rows split by policy. Baselines key on policy by default (`[task, policy, role]`), because policies of different sizes earn rewards on different scales.
- **Batch transforms per learner.** Depth weighting and the zero-advantage filter run on each learner's microbatches.
- **Per-policy staleness.** Each trainable policy has its own version. A group records every policy's version at admission, and a policy's rows train only within that policy's `max_staleness`; drops are group-atomic per policy, so one policy's stale rows never drop another's.
- **Stepping.** `independent` (the default) steps each learner when its batch fills, with a timer for policies whose rows arrive rarely, such as verifiers. `lockstep` steps and publishes all learners together, which avoids evaluating a mix of versions in co-training. `leader` lets one policy set the step index while the others follow, as miles does.
- **Checkpoints.** A barrier writes `policies.json` with each policy's step and checkpoint reference, so a resume restores consistent positions. Recovery checkpoints prune per policy.

### 5. Scheduling across sizes

Policies of different sizes differ by an order of magnitude in sampling speed and step time, so scheduling becomes per policy:

| Concern | Rule |
|---|---|
| Sampler capacity | each `PolicyRuntime` has its own concurrency limit and queue; queue waits per policy feed backpressure |
| Admission | a group is admitted only when every policy it may use is inside its staleness window |
| Batch size | learners batch by tokens, so a 35B step and a 4B step each take bounded time |
| Critical path | the slowest policy's calls bound episode time; per-policy call latency and the critical path are reported per group |
| Placement | each self-hosted backend is its own pool in the deployment ([RFC 0002](0002-multi-cluster-placement.md)); a large full fine-tune and a small LoRA service get separate hosts |
| Elastic samplers | sampler pools carry a `policy` field; extra engines can serve the small policy near remote cells while the large one stays on its backend |

On one GPU cluster this could mean a 35B full fine-tune on a self-hosted Tinker-compatible service (such as miles-tinker) across 8 nodes beside a 4B LoRA service on 1 node, with cells on the remaining hosts. A hosted backend may serve several base models under one key (UNVERIFIED for concurrent training of different bases).

### 6. Example

```yaml
backends:
  big:   {profile: miles-full, url: "${endpoint:big}"}
  small: {profile: tinker-hosted, api_key: "${secret:TINKER_API_KEY}"}
policies:
  orchestrator: {base_model: Qwen/Qwen3.6-35B-A3B, backend: big, adapter: full, r3: require,
                 renderer: qwen3_6_preserve_thinking, context_window: 131072,
                 learner: {learning_rate: 1.0e-6, batch_tokens: 2000000, max_staleness: 2}}
  worker:       {base_model: Qwen/Qwen3-4B-Instruct-2507, backend: small, adapter: lora, lora_rank: 32,
                 renderer: qwen3_instruct, context_window: 32768,
                 learner: {learning_rate: 3.0e-5, max_staleness: 3}}
  judge:        {base_model: "api:<provider>/<model>", trainable: false, context_window: 200000}
agents:
  roles:
    root:  {policy: orchestrator, may_spawn: [child]}
    child: {policy: worker}
trainer: {coordinator: {mode: independent}}
algorithm: {credit: {preset: by_role}}
```

### 7. Multi-LoRA as an optimization

When two trainable policies share a base model and a backend whose profile lists `multi_tenant` with enough `max_adapters`, the policies become two LoRA training clients on one service: hosted Tinker, upstream miles' adapter slots or SkyRL's tenants. Nothing above changes except placement. `doctor backend --adapters 2` trains two adapters on different markers and checks that each samples back only its own. On MoE bases, routing replay must be keyed per adapter; until a backend proves that, multi-LoRA MoE runs turn R3 off or exclude uncovered rows.

## Alternatives

- **Multi-LoRA only.** Cheapest, but one base model and one architecture, so it cannot be the general answer. It remains §7.
- **One shared policy with role prompts.** Simple and sometimes enough (AT-GRPO finds shared policies competitive on math), but it cannot mix sizes and couples roles that want different learning rates.
- **Alternating single-policy runs.** Train one policy against frozen partners, then swap. No new machinery, but slow and non-stationary, and credit never sees both learning at once.
- **One platoon run per policy, exchanging checkpoints.** Isolates failures, but team rewards and cross-policy baselines need one episode record.
- **Switching models per call inside one agent** (model routing). Breaks prefix merging and logprob bookkeeping across tokenizers; a child agent gives the same effect with exact records.

## Rollout plan

| Milestone | Scope | Exit check |
|---|---|---|
| M1 data model | policy ids on agents, calls and datums; a gateway with several `PolicyRuntime`s; one trainable policy beside frozen and API ones | a CPU test with two fake samplers using different tokenizers: no agent crosses policies, and every datum lands at its policy |
| M2 two learners on CPU | `Learner` per trainable policy, `PolicyCoordinator` in all three modes, barrier checkpoints | a two-policy fake-Tinker run where both rewards rise; resume restores both versions |
| M3 GPU, different sizes | textcraft-synth recursive with a larger root policy and Qwen3-4B-Instruct-2507 children on separate backends | both policies learn; per-policy staleness and step time inside their bounds (EST 4-8 GPU hours) |
| M4 multi-LoRA | the shared-base optimization; adapter isolation and per-adapter R3 checks | doctor passes on a multi-tenant backend |
| M5 policies across sites | policy-specific sampler pools on other clusters ([RFC 0002](0002-multi-cluster-placement.md) S5) | a two-site, two-policy run trains both |

## Open questions

1. Which coordinator mode should co-training default to: `independent` for throughput, or `lockstep` for stability?
2. Should baselines ever compare across policies, for example a shared team return with per-policy normalization?
3. How should GPUs split between backends: a fixed split per run, or a split the autoscaler advises from measured throughput?
4. Should platoon support distillation between policies? Teacher logprobs on student tokens need a shared tokenizer, so cross-family distillation would need re-tokenized text.
5. Can hosted Tinker train two different base models concurrently under one key, with independent sampler paths?
6. Must all policies checkpoint at the same orchestrator step, or is a barrier at the coordinator's cadence enough?
