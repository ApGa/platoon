# RFC 0007: Records and ATIF export

| | |
|---|---|
| Status | Draft |
| Created | 2026-10-13 |
| Builds on | [Architecture](../architecture.md): gateway call ids and exact tokens, outcomes and prefix merging |

## Summary

Give every rollout one self-describing directory: an append-only event log, an envelope folded from it, a token sidecar with the exact tokens and logprobs of every model call, and size-capped harness-native artifacts, all joined by `call_id` and `agent_id`. Records become complete enough to rebuild a rollout's datums byte for byte offline. ATIF v1.7 becomes an **export projection** for viewing and interchange (`platoon export --to atif|harbor`), not the training record, because ATIF has no rewards, outcomes or token-efficient storage.

## Motivation

Platoon writes an event file per finished rollout, and a terminal viewer reads them. What is missing:

- **Tokens are never persisted.** The exact prompt and sampled tokens live in memory until datums are built, so nobody can replay credit, debug a token-level mismatch or reproduce a training batch.
- **Rewards are edited in place.** Post-hoc reward processing (verdicts, judges, efficiency terms) mutates trajectories, so the event log can disagree with what trained.
- **Nothing joins the artifacts.** Event logs, worker logs, metrics and gateway records share no id.
- **No interchange.** Harbor's viewer, ATIF consumers and other frameworks cannot read platoon rollouts, and platoon cannot read theirs.
- **Inode pressure.** Harness logs written one file per event (as OpenHands writes them) reach tens of millions of inodes across a few dozen runs.

The gateway already assigns a `call_id` to every model call and holds its exact tokens, so the missing piece is the record, not the capture.

## Proposal

### 1. Layout

```
runs/<run>/rollouts/<split>/<step>/<group>/<rollout>/
  events.jsonl         # L0: append-only events, written while the rollout runs
  rollout.json         # L1: the envelope, a deterministic fold of L0 and the sidecar
  calls.tokens.zst     # L3: tokens and logprobs per call_id, delta-encoded
  native.tar.zst       # harness-native artifacts (Codex --json, Claude Code stream-json, OpenHands events), scrubbed
  agent/trajectory.json   # L2: ATIF, written only by `platoon export` or the viewer
```

A crash leaves L0 readable, and L1 and L2 can always be rebuilt from it. Rollout ids read `r<step>-<group>-<member>`.

### 2. Events

```python
@dataclass(frozen=True)
class ModelCallEvent:
    type: Literal["model_call"] = "model_call"
    call_id: str
    agent_id: str
    policy: str
    kind: str  # main, compaction, side, aux, judge (RFC 0005)
    base_call_id: str | None  # the call whose prompt and completion this prompt extends
    prompt_len: int
    completion_len: int
    finish_reason: Literal["stop", "length", "tool_calls"]
    version_start: int
    version_end: int
    started_at: float
    ended_at: float


@dataclass(frozen=True)
class ToolEventRecord:
    type: Literal["tool_call", "tool_result"]
    event_id: str
    agent_id: str
    tool: str
    is_error: bool = False
    parent_tool_call_id: str | None = None  # programmatic tool calling


@dataclass(frozen=True)
class AnnotationEvent:
    type: Literal["annotation"] = "annotation"
    key: str  # every post-hoc edit is an annotation, never a mutation
    value: Any
    agent_id: str | None = None
    at_call_id: str | None = None


RecordEvent: TypeAlias = (
    "ModelCallEvent | ToolEventRecord | AnnotationEvent | AgentEvent | RewardEventRecord | OutcomeEvent"
)
```

Reward lines carry `RewardEvent`'s fields ([RFC 0001](0001-native-multi-agent.md)); outcome lines carry platoon's `Outcome`. Producers: the gateway writes model calls and the token sidecar; the tool plane ([RFC 0005](0005-gateway-dialects-and-tito.md)) writes tool events; the workflow writes rewards, annotations and outcomes.

### 3. The envelope and the token sidecar

```python
@dataclass
class Rollout:  # rollout.json mirrors this
    id: str
    task: "TaskRef"  # env label, split, index, task id
    harness: str  # name@version, or the env's own rollout function
    policies: Mapping[str, tuple[int, int]]  # policy id -> (min, max) version seen
    outcome: Outcome
    graph: "AgentGraph"  # agents, edges, reward events (RFC 0001)
    calls: Mapping[str, Sequence[str]]  # agent id -> ordered call ids
    sidecar_sha256: str


def load(rollout_dir: Path) -> Rollout: ...  # folds events.jsonl and the sidecar
def rebuild_datums(rollout: Rollout, config: "WorkflowConfig") -> list["tinker.Datum"]: ...
```

The sidecar stores, per `call_id`: the base call and the length of the prefix it extends, the new prompt tokens, the completion tokens, their logprobs, and the version span. Delta encoding keeps it linear: a 100-call trajectory at 32k-token prompts is about 0.25 MB instead of about 19 MB of per-step prompt ids (EST). With tokens, a rollout is about 1-2 MB. MoE routing data is never persisted (20-42 MB per 64k-token trajectory); it stays in the backend.

`rebuild_datums` reruns the prefix merge and credit from the record, so a CI test can assert that rebuilt datums equal the ones that trained.

### 4. The ATIF projection

Platoon emits ATIF v1.7 and reads v1.0-1.8. Its profile:

1. one completion per agent step; tool results come from the tool plane, so the last turn's results, timings and errors are present;
2. subagents are embedded `subagent_trajectories` referenced from the spawning tool call; verifiers and judges carry `extra.platoon.role`;
3. compaction is a system step with its call as an embedded trajectory;
4. programmatic tool calling nests inner calls under the outer call in `extra.platoon`;
5. everything platoon-specific lives under `extra.platoon` (`schema: platoon.atif-ext/1`);
6. tokens are never taken from a harness's own ATIF; that is a cross-check only.

`platoon export RUN --to harbor` writes Harbor's trial layout (`agent/trajectory.json`, `result.json`, `config.json`), so `harbor view` renders platoon runs. CI validates golden exports with Harbor's trajectory validator as a dev-only dependency; platoon vendors the ATIF models and keeps Harbor out of core.

| Plain ATIF lacks | Where platoon keeps it |
|---|---|
| rewards and named components; `None` distinct from 0 | envelope, reward events |
| outcome status, reason, failure kind | envelope |
| masks, advantages, training flags | envelope; masks rebuilt from the sidecar |
| token ids and logprobs at linear size | sidecar |
| policy versions, sampler, renderer | events |
| verifier and judge linkage, attribution levels, message edges | envelope graph; `extra.platoon` |
| live append and crash safety | `events.jsonl` |

### 5. Viewing

`platoon viz` reads the new records, today's event files (so existing runs stay viewable), ATIF documents and Harbor trial directories, and draws the agent graph with per-agent outcomes and advantages.

## Alternatives

- **ATIF as the training record.** No rewards, outcomes or masks; per-step prompt ids grow quadratically; a whole-document format cannot be appended safely.
- **Keep today's event files and pickle the interaction maps.** Not portable, not safe to load, and still no join.
- **A database or Parquet per run.** Good for analytics and a later export, but researchers open a single bad rollout with `cat` and `jq`.
- **OpenTelemetry spans.** Good for timing and critical paths, not for tokens; spans can be derived from the events.

## Rollout plan

| Milestone | Scope | Exit check |
|---|---|---|
| M1 call-id join | the gateway writes `model_call` events and the envelope beside today's event files | every datum's calls resolve to events of the same rollout |
| M2 token sidecar | delta-encoded sidecar; `load` and `rebuild_datums` | rebuilt datums byte-identical to trained datums on recorded rollouts |
| M3 reward events | rewards, annotations and outcomes as events; no in-place reward mutation | the folded envelope matches what trained, including judge and verdict edits |
| M4 ATIF and Harbor export | the projection; Harbor's validator in CI | golden exports validate; `harbor view` renders a recursive rollout |
| M5 viewer | `platoon viz` on new records, legacy logs and ATIF | runs recorded before the change still open |

## Open questions

1. Persist tokens for every rollout, or for a sample? A run of about 35k rollouts (for example 45 steps of 768) is 35-70 GB with tokens (EST).
2. Should harness-native artifacts be kept by default, given their size and the need to scrub secrets?
3. Should an orchestration-trace export (arXiv 2605.02801's schema) sit beside ATIF for multi-agent interchange?
4. How long do rollout directories live, and who prunes them: `platoon runs prune`, or a retention policy in the run config?
