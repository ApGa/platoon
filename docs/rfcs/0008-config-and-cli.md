# RFC 0008: Config and CLI redesign

| | |
|---|---|
| Status | Draft |
| Created | 2026-10-13 |
| Builds on | [Architecture](../architecture.md#config-and-cli): `platoon train`, `platoon eval` and `environments:` mixtures |
| Implemented in part | strict loading on the existing dataclasses (M1); `extends:`, `config.resolved.yaml` and `lock.json` (part of M3); the `train`, `eval`, `rollout`, `config`, `harness` and `env` commands with exit codes 0, 1, 64, 78 and 130 (part of M4) |

## Summary

Replace permissive config loading with one strict, typed schema that is the single source of truth, split configuration into **file kinds** that never mix (what trains, where it runs, what a site is), add `extends:` presets, `arms:` and documented merge and interpolation rules, validate cross-field constraints before anything is submitted, and migrate today's YAML mechanically. Grow `platoon train` and `platoon eval` into one CLI with lazy subcommands, stable exit codes and reproducible run directories.

## Motivation

Before this RFC, platoon loaded YAML into dataclasses and applied `key=value` overrides. Concrete problems (the first two are fixed by M1):

- **Unknown keys vanish.** The loader fills only fields it knows, so a typo such as `leave_one_out_basline: true` silently trains with the default.
- **Overrides are parsed by guesswork.** `0`, `1`, `yes` and `no` become booleans before integers are tried, whatever the field's type, so `stats.trial_name=1` sets `True`.
- **Kinds mix.** One file holds the experiment, environment wiring and, through hand-written job scripts with extra override strings, the deployment. Moving a run to another cluster edits the experiment.
- **Behavior hides in environment variables** read across the code, invisible to the config and to reproducibility records.
- **Config errors loop.** The restart wrapper retries on exit code 2, so a bad config can restart many times before anyone notices.
- **No resume check.** Nothing stops a resumed run from silently picking up a changed config.

## Proposal

### 1. One schema

Strict pydantic v2 models (`extra="forbid"`, frozen) are the single source of truth. Unknown keys fail with a did-you-mean suggestion; overrides parse against the field's type; `platoon config schema` publishes the JSON schema, and CI validates every YAML block in the docs against it.

```python
class EnvEntry(PlatoonModel):  # today's EnvironmentConfig, typed
    name: str
    weight: float = 1.0
    service: str | None = None  # "harbor" or "python://module:Class"
    service_kwargs: Mapping[str, Any] = {}
    harness: str | Mapping[str, float] | None = None
    placement: Literal["auto", "task_sandbox", "local"] = "auto"
    builtins: Literal["default", "keep", "disable"] = "default"
    rollout: str | None = None  # an env's own rollout function, when no harness
    seed: int = 0  # the per-task harness draw


class RunConfig(PlatoonModel):  # an experiment: what trains
    run: "RunMeta"
    extends: tuple[str, ...] = ()
    arms: Mapping[str, Mapping[str, Any]] = {}
    envs: tuple[EnvEntry, ...]
    workflow: str = "group_rollout"  # run-level, as in today's mixtures
    policies: Mapping[str, "PolicyEntry"]  # RFC 0003; `policy:` is shorthand for {main: ...}
    agents: "AgentsConfig | None" = None  # RFC 0001
    rollout: "RolloutSection"
    algorithm: "AlgorithmSection"  # credit, baselines, RAO features
    trainer: "TrainerSection"  # loss, optimizer, batch, staleness
    eval: "EvalSection | None" = None
    checkpoint: "CheckpointSection"
    logging: "LoggingSection"


def load_config(
    files: Sequence[Path], overrides: Sequence[str] = (), *, site: str | None = None
) -> "ResolvedConfig": ...
```

### 2. File kinds

| Kind | Top-level keys | Where it lives |
|---|---|---|
| experiment, preset | `run`, `extends`, `arms`, `envs`, `policies`, `agents`, `rollout`, `algorithm`, `trainer`, `eval`, `checkpoint`, `logging`, `extensions` | `experiments/`, presets in packages |
| deployment, backend layer | `extends`, `hub`, `blobs`, `backends`, `pool_defaults`, `pools`, `exposure`, `autoscale` | `deploy/` ([RFC 0002](0002-multi-cluster-placement.md), [RFC 0006](0006-backend-profiles-and-sdk-upgrade.md)) |
| site profile | `site`, `scheduler`, `container`, `network`, `storage`, `isolation`, `sandbox`, `secrets`, `paths`, `state_dir` | `sites/`, a user's config directory, or a package entry point |

A file holds one kind, and the loader rejects a mixed file, so swapping a deployment never changes what trains. Commands take any number of files and merge them; experiments never name sites, and site profiles never hold experiment settings. A research project can ship its site profiles and deployments as a package.

### 3. Composition

- **`extends:`** takes relative paths, `pkg:<distribution>/<path>` and `preset:<namespace>/<name>`; packages ship presets as plain YAML through an entry point.
- **`arms:`** are named overlays; each becomes its own run with its own resolved config, replacing per-arm override strings in launch scripts.
- **Merge rules.** Dicts deep-merge and lists replace; changing a block's `type`, `uri` or backend `profile` replaces the block; `!replace` replaces a mapping; `~KEY` resets to the default; a nested model's schema default is the lowest layer, so a partial override keeps its siblings.
- **Interpolation.** `${...}` resolves at load (`${run.name}`, `${env:VAR,default}`, `${secret:NAME}` read at run time and never written to disk, `${endpoint:<backend>}`); `{...}` resolves in the launcher on the host (`{host.ip}`, `{port.<name>}`); `{{ ... }}` (Jinja) appears only in HarnessSpecs and goal templates. Unknown names are errors.
- **Code references** are registered names (entry points or decorators) or `python://module:attr`, recorded as `<name>@<version>`.

### 4. Validation before submission

The loader checks cross-field constraints and exits with code 78 on failure: the timing chain (tool timeout < rollout timeout < watchdog; each harness's model-client and MCP timeouts cover the gateway's budgets); `min_successful_group_size` and straggler quorum within `group_size`; every requested loss and feature inside the backend profile; R3 runs that fit the routing TTL; every env's service resolvable and every launched service used; walltime within the site's maximum.

### 5. Migration

`platoon config migrate LEGACY.yaml [--base BASE] [--from-env]` emits a delta in the new schema that writes today's semantics explicitly where defaults change (credit settings, normalizer scope, condenser sampling), and maps environment-variable knobs (`--from-env`) to config keys. `platoon train` refuses to start when a known legacy knob is set but unmapped. Acceptance: every shipped Tinker config migrates to identical effective values.

### 6. The CLI

```
platoon [-S/--site NAME] [--runs-root DIR] [--format table|json] [-v|-q] [--version]
├── train | rollout   FILES... [k=v ...] [--arm A] [--resume auto|never|must] [--dry-run]
├── eval              [FILES...] [-d DATASET|URI] [-a HARNESS] [-m POLICY] [-n N] [-k K]
├── launch            FILES... [--arm A] [--chain K] [--dry-run] [--attach RUN --join-token FILE]
├── runs              list | show | logs | resume | cancel | diff | prune
├── config            show [--provenance] | validate | diff | schema | migrate LEGACY
├── doctor            backend | sandbox | env URI | harness H@V | cluster
├── harness           list | show | prefetch H@V
├── export            RUN --to atif|harbor|sft|peft|hf
└── viz               RUN|PATH [--follow] [--rollout ID]
```

- **Cyclopts with lazy subcommands**, so `platoon --help` imports nothing heavy and returns in under 0.3 s.
- **The Python API is the CLI**: every command is a thin shell over one public function.
- **`platoon eval`** accepts Harbor's short flags (`-d -a -m -n -k`), writes a report with success@k and reward@k, and reports `invalid`, `error` and `timeout` separately, never as zeros.
- **Login-host guard.** `train`, `eval`, `rollout` and the long doctors refuse to run on login hosts; `launch` and read-only commands are allowed, and `--submit` runs a doctor in a short job.

| Exit code | Meaning |
|---|---|
| 0 | OK |
| 1 | runtime failure |
| 3 | failed precondition (doctor, login-host guard) |
| 64 | usage error |
| 75 | restartable; `attempts/<n>/exit.json` says the scope (`process`, `job`) |
| 78 | configuration error: never restarted |
| 130 | interrupted |

### 7. Run directories

```
runs/<name>/
  run.json                # name, status, config digest, schema version, attempts, release
  lock.json               # platoon and plugin versions, SDK, harness and image digests, effective knobs
  config/                 # resolved.yaml, sources, overrides, provenance
  attempts/<n>/           # launch artifacts, endpoints, exit.json, logs
  checkpoints.jsonl       # unchanged schema, for resume compatibility
  metrics.jsonl
  rollouts/...            # RFC 0007
```

`platoon runs resume NAME` re-validates the resolved config under the current code and refuses a digest or schema change unless `--allow-change` names the keys; accepted changes are recorded.

## Alternatives

- **Keep dataclasses and only add strictness.** Cheap and fixes the silent typos, but kinds still mix and there is no schema to publish or validate docs against.
- **Hydra or OmegaConf structured configs.** Mature composition, but heavy, with interpolation and merge semantics that are hard to explain, and no file kinds.
- **Typer.** No lazy subcommands, so `--help` imports the world.
- **One file with a `launch:` block.** Simpler to start, but every deployment change edits the experiment.

## Rollout plan

| Milestone | Scope | Exit check |
|---|---|---|
| M1 strict loading | unknown keys and badly typed overrides fail on today's dataclasses; `platoon config validate` | every shipped config loads; a typo test fails with a suggestion |
| M2 schema and migration | pydantic schema with today's key names as aliases; `config migrate` | all shipped Tinker configs migrate to identical effective values |
| M3 composition | `extends`, presets, `arms`, merge rules, interpolation, run directories and `lock.json` | arms replace per-arm launch scripts; resume refuses a silent change |
| M4 CLI | the command tree, exit codes, login-host guard; `train`, `eval`, `rollout`, `config`, `runs` | `--help` under 0.3 s; config errors exit 78 and never restart |
| M5 file kinds and launch | deployments, site profiles, `launch`, `doctor` ([RFC 0002](0002-multi-cluster-placement.md), [RFC 0006](0006-backend-profiles-and-sdk-upgrade.md)) | a dry run on a PBS site renders, for an existing experiment, the job a hand-written script submits |

## Open questions

1. How long do today's key names (`train.workflow_config.*`) stay as aliases once the new names exist?
2. Are pydantic and Cyclopts acceptable as core dependencies? They are the only ones this RFC adds.
3. Should a project-level `platoon.toml` set default files and the default site?
4. Which environment-variable knobs survive as documented escape hatches, and which become config keys only?
