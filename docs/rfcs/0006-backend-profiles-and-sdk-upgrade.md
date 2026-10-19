# RFC 0006: Backend profiles and the Tinker SDK upgrade

| | |
|---|---|
| Status | Draft |
| Created | 2026-10-13 |
| Builds on | [Architecture](../architecture.md): the trainer, prefix merging and renderers |

## Summary

Make platoon work with any Tinker service by describing each backend with a **capability profile**: data that says what the server accepts and how it behaves, where unknown means off and code branches on capability values, never on backend names. Move the Tinker SDK from today's pin (0.16.1) to a tested window (`>=0.26.2,<0.34`) that each profile narrows, with one adapter that detects SDK features. Replace today's routing-replay plumbing for one self-hosted backend (an `extra="allow"` patch and a `routing_handle` datum input) with backend-neutral **provenance spans** that name sampled sequences, keeping that backend's exact-token lookup as a fallback. Let backends that run inside a job be **managed** by plugins, which the packages that use them can ship.

## Motivation

Platoon keeps the tinker SDK at 0.16.1 and tinker-cookbook at 0be951bd, for compatibility with [miles-tinker](https://github.com/neulab/miles-tinker), a self-hosted Tinker-compatible server built on miles that supports MoE routing replay (R3) ("the R3 server" below). That pin carries costs:

- **R3 depends on server internals.** Platoon rebinds the SDK's response models to keep the R3 server's per-sequence `routing_id` and sends it back as a `routing_handle` datum input. Neither exists on any other backend, and the monkeypatch breaks with the SDK.
- **Other backends are out of reach.** Hosted Tinker wants recent SDKs; upstream miles validates requests strictly and rejects SDK 0.29+ fields; SkyRL tests 0.25-0.27 clients. Platoon cannot target them without knowing what each accepts.
- **Cookbook renderers are blocked.** tinker-cookbook main requires `tinker>=0.30.1`, so newer cookbook renderers cannot install.
- **Behavior is implicit.** Weight-push semantics, checkpoint retention, session close and telemetry settings are the R3 server's behavior, written into code and scripts, so a different backend fails at runtime, often with a 400 that ends a training model.

## Proposal

### 1. Capability profiles

```python
class RoutingReplay(BaseModel, frozen=True):
    carrier: Literal["none", "server_token_lookup", "server_sequence_id", "client_carried"] = "none"
    tenant_scope: Literal["none", "api_key"] = "none"  # the R3 server keys routing by API key
    ttl_s: float | None = None  # how long the server keeps routing
    consume_once: bool = False  # a second pass over a datum loses its routing
    coverage_metric: str | None = None  # optim_step metric that reports coverage


class BackendProfile(BaseModel, frozen=True):  # core; never imports tinker; unknown means off
    name: str
    sdk: str  # PEP 440 range the server accepts
    url: str | None = None  # a documented service URL (hosted)
    request_validation: Literal["strict", "lenient", "unknown"] = "unknown"
    auth: Literal["account", "any_key", "none", "unknown"] = "unknown"
    train_modes: frozenset[Literal["lora", "full"]] = frozenset()
    multi_tenant: bool = False
    max_adapters: int | None = None
    losses: frozenset[str] = frozenset()
    custom_loss: bool = False  # forward_backward_custom works
    versions: Literal["pinned", "keep_last_n", "latest_only"] = "latest_only"
    stop_token_included: Literal["yes", "no", "unknown"] = "unknown"
    provenance: Literal["honored", "ignored", "rejected", "unknown"] = "unknown"
    routing_replay: RoutingReplay = RoutingReplay()
    sampler_transfer: frozenset[Literal["path", "handle", "pickle"]] = frozenset({"path"})
    ckpt_delete: bool = False
    session_close: Literal["finish", "unload_model", "lease_only"] = "lease_only"
    client_env: Mapping[str, str] = {}  # for example TINKER_TELEMETRY=0
    modalities: frozenset[str] = frozenset({"text"})


class BackendConfig(BaseModel):  # backends.<name>
    profile: str  # a shipped or plugin profile, or a TOML path
    url: str | None = None  # default: the profile's url
    api_key: str | None = None  # a secret reference, read at run time
    managed: "ManagedSpec | None" = None
    caps_overrides: Mapping[str, Any] = {}
    accept_open: bool = False  # sandboxes may reach a backend without account auth
```

Every field defaults to the pessimistic value, so a run that needs a missing feature fails at config time with a `CapabilityError`, never at runtime. Platoon ships `tinker-hosted`, `miles` (upstream), `skyrl`, `fake` and drafts written by `doctor backend --url URL --write-profile PATH`. Other packages ship profiles through an entry point, such as LoRA and full fine-tuning profiles for the R3 server.

Code branches on values. Three examples: unless `stop_token_included == "yes"`, the gateway appends the renderer's end-of-turn token after a token-id stop, because the cookbook's parser needs it; samplers move to rollout workers by `handle` where the SDK and profile allow, else by `path`; recovery checkpoints prune by TTL, then REST delete, then the managed plugin, and the learner warns when none exists.

`platoon doctor backend` confirms a profile against a live server: connect, sample with token-id stops, check sum-reduction linearity, compare forward-only and sampler logprobs, push a version and classify what happens to in-flight calls, save and resume, probe provenance and routing coverage, and close. It writes `caps.verified.json` to the state directory and stays read-only on a live single-tenant server.

### 2. The SDK window

```python
@dataclass(frozen=True)
class SdkFeatures:  # detected once from the installed SDK, never from its version string alone
    service_close: bool  # ServiceClient.close, from 0.28.1
    sampler_handles: bool  # get_sampler_handle and from_sampler_handle, from 0.32.0
    serializes_defaults: bool  # 0.29+ sends defaulted fields that strict servers reject
    blocking_publish: bool  # save_weights_and_get_sampling_client blocks the loop from 0.27


class TinkerAdapter(Protocol):  # the only module that imports tinker
    features: SdkFeatures

    async def training_client(self, policy: "PolicyEntry") -> "TrainingBackend": ...
    async def sampler(self, ref: "PolicyRef") -> "SamplerLike": ...
```

`platoon[tinker]` accepts `tinker>=0.26.2,<0.34`, the union platoon tests. A profile narrows it (`miles`: `<0.28` until a doctor run clears newer clients). The release builder resolves the run's environment with that range, so the resolver also picks a tinker-cookbook that fits: 0.5.x for strict self-hosted servers, the newest for hosted Tinker. `platoon train` compares the installed SDK with the profile and prints the install line on a mismatch. Two backends with disjoint windows in one run can use an adapter sidecar in its own environment later; core never imports tinker, which makes that possible.

### 3. R3 through provenance

Platoon never implements routing replay; it keeps the conditions a backend needs and measures coverage.

| Carrier | Backend | What platoon does |
|---|---|---|
| `none` | hosted Tinker, upstream miles, SkyRL | nothing; `r3: require` fails at config time |
| `server_token_lookup` | the R3 server | sample with the trainer's key (the tenant), send exact tokens, never run a second pass over a training datum, keep staleness inside the routing TTL |
| `server_sequence_id` | proposed to every backend | send `Datum.with_provenance(...)` spans naming the `SampledSequence.sequence_id`s each datum reproduces |
| `client_carried` | backends whose samplers return routing | carry it in adapter-private fields that core never reads |

The adapter emits provenance spans on every datum unless the profile says `rejected`. A merged datum's spans tile its tokens: prompt runs name the requests that supplied them, sampled runs the sequences that produced them, which prefix merging knows exactly. Once the R3 server resolves routing by `sequence_id`, or by its exact-token lookup when no handle is sent, the `extra="allow"` patch and the `routing_handle` input are deleted. Run config: `r3: off | prefer | require`; `prefer` logs once when the backend lacks it. Rows sampled outside the backend, by platoon-managed engines, are `routing_uncovered` ([RFC 0002](0002-multi-cluster-placement.md)).

### 4. Managed backends

| Mode | Who starts the backend | Use |
|---|---|---|
| `external` (default) | someone else | hosted Tinker; a shared service; the R3 server started by its own job script |
| `managed` | platoon, inside the run's allocation, through a plugin | one job holding trainer hosts, engine hosts and cells, on PBS or Slurm |
| `attach` | the backend's own job script, with platoon as its hook | jobs whose backend script stays the driver |

```python
class ManagedBackend(Protocol):  # entry point group platoon.backends.v1
    def plan(
        self, allocation: "Allocation", args: Mapping[str, Any]
    ) -> "BackendPlan": ...  # processes, ports, host roles
    async def wait_ready(self, plan: "BackendPlan", timeout_s: float) -> None: ...
    async def health(self, plan: "BackendPlan") -> Literal["ok", "degraded", "down"]: ...
    async def stop(self, plan: "BackendPlan", grace_s: float) -> None: ...
    def delete_checkpoint_files(self, ref: "CheckpointRef") -> None: ...  # optional
```

Platoon ships a generic `command` plugin (phases with argv, readiness checks and host roles, run through the site's container runtime) and `fake`. A backend's own plugin can ship with the package that uses it. A managed backend becomes a pool whose `train` hosts take no cells ([RFC 0002](0002-multi-cluster-placement.md)).

### 5. Security

The backend key never enters a sandbox. Some backends accept any key, so `doctor` classifies auth with a random key and with none, and a cluster check tests whether a sandbox can connect to the backend. If sandboxes can reach a backend whose auth is not `account`, training refuses unless `accept_open: true` is set and recorded.

## Alternatives

- **Stay on 0.16.1.** Zero risk now, but locks platoon to one backend and blocks cookbook renderers.
- **One global newer pin.** Strict servers reject newer clients, and hosted users would be held back by the strictest server.
- **Branch on backend names.** Quick, but every new service needs code, and every server change becomes a platoon change.
- **Discover capabilities by trial at runtime.** On upstream miles a 400 during training ends the model; failures must happen at config time.

## Rollout plan

| Milestone | Scope | Exit check |
|---|---|---|
| M1 profiles and doctor | `BackendProfile`, shipped and plugin profiles, `doctor backend` (read-only mode) | profiles load in CI; doctor confirms the R3 server's LoRA and full profiles without disturbing a live run |
| M2 adapter on 0.16.1 | all `import tinker` behind the adapter, with feature detection; no behavior change | the existing suites and one GPU canary are unchanged |
| M3 SDK step on the R3 server | move to the R3 server's tested window (0.27.1, lenient decoding); emit provenance; R3 through token lookup; delete the patch and the routing handle | a 35B MoE canary with R3 coverage 1.0 and sample-train KL no higher than on the 0.16.1 pin |
| M4 other backends | hosted Tinker, upstream miles and SkyRL profiles; per-profile resolution of SDK and cookbook; cookbook renderers adopted where they install | number-search and textcraft-synth train on hosted Tinker and on one self-hosted backend |
| M5 managed backends | `command` and `fake` plugins; a plugin for the R3 server; checkpoint retention by capability | a batch job starts the R3 server through the plugin with the host roles of a hand-written job script |

## Open questions

1. Will the R3 server accept R3 keyed by `sequence_id` with token lookup as fallback, and an API-key allow-list? Both are optional asks, not prerequisites.
2. Does hosted Tinker accept 0.27 clients, or does the hosted profile need a newer window?
3. When does platoon drop 0.16.1: after M3's canary, or after a deprecation period for configs that still need it?
4. Who maintains a self-hosted server's profiles when the server changes: the package that ships them, or the server itself, through a discovery endpoint?
5. Should custom losses through `forward_backward_custom` (GSPO, KL variants) ship with M4, given their extra compute?
