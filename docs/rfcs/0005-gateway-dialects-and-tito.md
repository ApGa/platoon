# RFC 0005: Gateway dialects, TITO and the MCP tool plane

| | |
|---|---|
| Status | Draft |
| Created | 2026-10-13 |
| Builds on | [Architecture](../architecture.md): the model gateway, the native-tools rule and prefix merging |
| Implemented in part | the MCP tool plane (§5): env tools per agent with access levels, progress and hidden rewards; orchestrator tools and terminal semantics are not built yet ([Architecture](../architecture.md#the-mcp-tool-plane)) |

## Summary

Grow the Chat Completions gateway into the single entry point for every harness: three wire dialects (OpenAI Chat, OpenAI Responses for Codex, Anthropic Messages for Claude Code) over one canonical turn; a per-agent **token ledger** whose **TITO bridge** (token-in, token-out) reuses sampled tokens verbatim instead of re-rendering history; call kinds that keep compaction and side calls out of training; an **MCP tool plane** that serves env tools and orchestrator tools to any MCP-capable harness, which completes the native-tools rule for envs without a sandbox; and **gateway shards** per cell host instead of one server per rollout worker.

## Motivation

The gateway re-renders the whole conversation on every call and records each call's prompt and sampled tokens; prefix merging later stitches calls into datums wherever the extension property holds. Four problems remain:

- **Re-rendering is not exact.** If the renderer does not reproduce history token for token (a Qwen empty-reasoning block re-renders as a different token; a parsed tool call re-renders with different whitespace), the next prompt diverges from what was sampled. The data stays correct, because a new datum starts, but merge rates drop, train-time contexts drift from sample-time contexts, and R3 rows multiply.
- **Re-rendering is expensive.** Tokenizing a 64-131k-token prompt costs 150-300 ms of CPU per call, quadratic over an episode, and it blocks the event loop that also serves keepalives.
- **Two dialects are missing.** Codex speaks OpenAI Responses and Claude Code speaks Anthropic Messages; both are experimental today. Without them the strongest coding harnesses cannot train.
- **Env tools cannot reach unmodified harnesses.** The native-tools rule says harnesses on envs without a sandbox use the env's tools, but before the tool plane nothing served those tools over a protocol the harness already speaks, so such envs needed their own rollout functions.

## Proposal

### 1. Dialects over one canonical turn

| Dialect | Endpoints | Harnesses | Reasoning handle | Keepalive | Context overflow |
|---|---|---|---|---|---|
| OpenAI Chat | `v1/chat/completions`, `v1/models` | mini-swe-agent, OpenHands, pi, opencode | not needed | SSE comments | 400 `context_length_exceeded` |
| OpenAI Responses | `v1/responses` (SSE) | Codex | `reasoning.encrypted_content = "pg1:<handle>"` | `response.in_progress` every 10-15 s | 400 `context_length_exceeded` |
| Anthropic Messages | `v1/messages`, `v1/messages/count_tokens` | Claude Code | thinking `signature = "pg1:<HMAC>"`, minted by the gateway and never rejected | `ping` | 400 "prompt is too long" |

```python
@dataclass(frozen=True)
class CanonicalRequest:
    messages: Sequence[Mapping[str, Any]]  # OpenAI Chat shape after dialect conversion
    tools: Sequence[Mapping[str, Any]]
    kind_hints: Mapping[str, str]  # headers and turn metadata used to classify the call
    stream: bool
    retry_key: str | None  # Idempotency-Key, or a client retry counter


@dataclass(frozen=True)
class CanonicalTurn:
    content: str
    reasoning: str | None
    tool_calls: Sequence[Mapping[str, Any]]
    finish_reason: Literal["stop", "length", "tool_calls"]
    call_id: str
    handle: str  # opaque reasoning handle for Responses and Messages


class Dialect(Protocol):
    name: Literal["openai_chat", "openai_responses", "anthropic_messages"]

    def parse_request(self, body: Mapping[str, Any], headers: Mapping[str, str]) -> CanonicalRequest: ...
    def render_response(self, turn: CanonicalTurn, *, stream: bool) -> "ResponseLike": ...
    def keepalive(self) -> bytes: ...
    def overflow_error(self, prompt_tokens: int, limit: int) -> "ResponseLike": ...
```

Reasoning handles let the gateway find the exact ledger vertex when a harness echoes an opaque reasoning item back, which is how Codex and Claude Code carry thinking across turns. Codex prefers websockets; the gateway answers the upgrade with 426 and Codex falls back to HTTPS.

### 2. The token ledger and the TITO bridge

Each agent owns a message graph. Client vertices hold the tokens of messages the harness sent; model vertices hold sampled tokens, logprobs and the call record.

```python
@dataclass
class LedgerVertex:
    vertex_id: str
    parent: "LedgerVertex | None"
    tokens: Sequence[int]  # the delta this vertex adds to its path
    sampled: bool  # model vertex: tokens were sampled, with logprobs
    call_id: str | None = None


class TokenLedger(Protocol):
    def match(
        self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]]
    ) -> LedgerVertex | None: ...
    def bridge(
        self, vertex: LedgerVertex, tail: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]]
    ) -> list[int] | None: ...  # None: re-render
    def commit(
        self,
        parent: LedgerVertex | None,
        tail: Sequence[Mapping[str, Any]],
        prompt: Sequence[int],
        sampled: "SampledSequence",
        turn: CanonicalTurn,
    ) -> LedgerVertex: ...
    def path_tokens(self, vertex: LedgerVertex) -> list[int]: ...
```

**The bridge contract is token-in, token-out**, not template equality: `bridge(m, tail) = path_tokens(m) + renderer_suffix(tail, tools)`, reusing every sampled token verbatim. It falls back to a re-render only when the call needs different renderer settings (a compaction call with thinking off), when the tail holds roles the renderer cannot append as a suffix (an assistant message the gateway did not produce, a mid-conversation system message), or when the tools or system prompt changed. After every commit `path_tokens(v) == prompt + sampled` holds. A completion that stopped on length gets the end-of-turn token appended as context (mask 0), which is what a re-render would show the model.

**Prefix merging stays.** A bridged call's prompt extends the previous call's prompt and completion by construction, so `trajectory_to_data` merges it; re-rendered calls still start new datums. The ledger only raises the merge rate. Health metrics per rollout: `bridged_fraction` (alarm below 0.95 per renderer and harness), `datums_per_call`, and `template_divergence`, measured by also re-rendering 1% of bridged calls.

**Renderer support.** The suffix render needs an API the pinned tinker-cookbook does not expose publicly. The gateway computes it by rendering the full message list and checking that the matched path is a token prefix of the result, behind a golden test, and platoon asks the cookbook for a public suffix method.

### 3. Call kinds

| Kind | Evidence | Trains |
|---|---|---|
| `main` | the root's first message carries the orchestrator's instruction, and every call on its path | yes |
| `compaction` | Codex turn metadata `request_kind: compaction`; Claude Code request class `compaction`; OpenHands condenser calls tagged by its runner | only when accepted (its summary reappears in a later root) and not reactive |
| `side`, `aux` | Codex memory and prewarm requests; Claude Code auxiliary calls and its background model alias | no |
| `judge`, `verifier` | the agent's purpose | no |
| `unknown` | anything else, including model-written code calling the gateway with a leaked secret | no; alarm above 10% of an agent's calls |

Sampling settings key on the call kind (`gateway.by_call_kind.compaction.sampling`), which replaces per-harness condenser overrides.

### 4. Streaming, errors and budgets

SSE stays synthetic: the gateway samples the whole completion, then emits the dialect's events, with keepalives every 10-15 s while a call queues or samples. The gateway never returns 429; calls queue per shard and fail the agent as `error/gateway_overload` only after a timeout. Policy problems (malformed tool calls, unparseable output) return 200 with annotations, because a 5xx makes harnesses resample and biases data toward short answers. Every HarnessSpec turns harness retries off and declares a model-client timeout at least as long as the gateway's queue, push-pause and sample budgets. Per-agent budgets (`max_calls`, output tokens) end with a dialect-native terminal reply.

### 5. The MCP tool plane

One MCP endpoint per agent, `/s/<fp>/mcp` over streamable HTTP, authenticated by the agent key, serving the env session's tools (`EnvSession.tools` and `call_tool`) plus the orchestrator tools the agent's role grants (`launch_subagent`, `finish`, `submit_verdict`; [RFC 0001](0001-native-multi-agent.md)).

```python
@dataclass(frozen=True)
class ToolEvent:
    event_id: str
    agent_id: str
    tool: str
    arguments: Mapping[str, Any]
    is_error: bool
    finished: bool  # the env says the agent is done
    started_at: float
    ended_at: float
    parent_tool_call_id: str | None = None  # programmatic tool calling: the outer call


class ToolPlane(Protocol):
    async def list_tools(self, agent_id: str) -> list[ToolSpec]: ...  # filtered by access, x-platoon-* keys stripped
    async def call_tool(self, agent_id: str, name: str, arguments: Mapping[str, Any]) -> ToolResult: ...
```

- **Access levels.** `full` (roots), `no_terminal` (children and verifiers: no submit tools), `read_only`.
- **Terminal semantics.** A tool result with `finished=true` makes the agent terminal; further model calls get a dialect-native terminal reply, and the harness is stopped after a grace period. A terminal call from an agent with live children cancels the children first, by default.
- **Long-running tools.** HarnessSpecs raise the harness's MCP timeout to the env's tool timeout (Frontier-CS grading can take hours), and the plane sends MCP progress notifications every 30 s.
- **Rewards stay hidden.** Env rewards in tool results become reward events and never reach the policy.
- **Completing the native-tools rule.** With the plane, a harness on an env without a sandbox runs `local`, built-ins disabled, with the env's tools over MCP; HarnessSpec `[mcp]` says how to hand the server to the harness.

### 6. Gateway shards per cell

Today each rollout worker starts its own gateway server. With cells ([RFC 0002](0002-multi-cluster-placement.md)), each host runs gateway shards of at most about 128 rollouts each; a rollout is pinned to `hash(rollout_id) mod shards`, and children inherit their parent's shard. Rendering and tokenizing run in a per-shard process pool, so a 300 ms render never stalls keepalives or MCP progress; event-loop lag is a metric. Agents are addressed by fingerprint (`/s/<fp>/...`), never by shard address, so an agent could later move with its sandbox.

## Alternatives

- **Re-render every call** (today). Correct thanks to prefix merging, but inexact contexts, O(T²) tokenization and lower merge rates.
- **Concatenate sampled tokens heuristically** (AReaL's proxy). Covers all dialects but not exactly.
- **Match traces to steps by position** (rLLM). Breaks on retries, parallel calls and compaction.
- **Take tokens from harness-native logs.** Logs are lossy, version-specific and not always present; useful only to cross-check capture completeness.
- **Real token streaming.** No harness needs it; synthetic SSE with keepalives is enough.

## Rollout plan

| Milestone | Scope | Exit check |
|---|---|---|
| M1 ledger and bridge for Chat | ledger, bridge, reasoning matching, length closing, health metrics | on 200 synthetic conversations, prompt ids identical wherever today's re-render extended the prefix; `bridged_fraction` ≥ 0.95 with the Qwen renderers |
| M2 MCP tool plane | the plane over `EnvSession`; access levels, terminal semantics, progress, reward stripping | ORS tools served over MCP match the inline `ctx.call_tool` results on recorded fixtures |
| M3 Responses and Codex | dialect, reasoning handles, call kinds, a Codex HarnessSpec, a harness doctor | Codex fixtures pass for a pinned version; a canary trains |
| M4 Messages and Claude Code | dialect, signatures, hint headers, count_tokens | Claude Code fixtures pass for a pinned version |
| M5 shards per cell | shared shards, process pools, fingerprint addressing | 384-way load without keepalive gaps; event-loop lag under 1 s |
| M6 programmatic tool calling | a `code` MCP tool that runs model code in the agent's sandbox and calls back with the agent's key | a programmatic-tool-calling harness runs its code in the sandbox, with rewards matching in-process execution in an A/B |

## Open questions

1. Training through Claude Code raises a terms-of-use question that must be settled before M4 trains anything.
2. Will tinker-cookbook accept a public suffix-render method, or does platoon keep the private path behind its golden test?
3. Should migrated configs keep JSON-in-text tool results until an A/B shows MCP content blocks train as well?
4. How should OpenHands mark condenser calls once an OpenHands harness runs unmodified: a header from its runner, or the acceptance check alone?
