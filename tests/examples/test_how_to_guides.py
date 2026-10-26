"""The how-to guides in docs/how-to stay true: their config snippets validate, their links and anchors resolve, and
the messages they quote are still the ones platoon prints."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from platoon import cli

REPO = Path(__file__).resolve().parents[2]
HOW_TO = REPO / "docs" / "how-to"
GUIDES = sorted(HOW_TO.glob("*.md"))

_YAML_BLOCK = re.compile(r"```yaml\n# (?P<name>[\w.-]+\.yaml)\n(?P<body>.*?)```", re.S)
_LINK = re.compile(r"\]\((?P<target>[^)\s]+)\)")
_HEADING = re.compile(r"^#{1,6} (?P<title>.+)$", re.M)


def _slug(title: str) -> str:
    """GitHub's anchor for a heading."""

    return re.sub(r"[^\w\- ]", "", title.strip().lower()).replace(" ", "-")


def test_the_guides_exist():
    assert {path.name for path in GUIDES} >= {
        "configs.md",
        "train-and-eval.md",
        "sandboxes-and-harbor.md",
        "clusters.md",
        "troubleshooting.md",
    }


def test_the_layered_config_snippets_in_the_configs_guide_validate(tmp_path, capsys):
    blocks = {match["name"]: match["body"] for match in _YAML_BLOCK.finditer((HOW_TO / "configs.md").read_text())}
    assert {"environments.yaml", "train.yaml", "smoke.yaml"} <= set(blocks)
    for name, body in blocks.items():
        (tmp_path / name).write_text(body)

    cli.main(["config", "validate", str(tmp_path / "train.yaml"), str(tmp_path / "smoke.yaml")])

    out = capsys.readouterr().out
    assert f"ok   {tmp_path / 'train.yaml'} (PlatoonTinkerRLTrainerConfig)" in out
    assert f"ok   {tmp_path / 'smoke.yaml'} (PlatoonTinkerRLTrainerConfig)" in out


@pytest.mark.parametrize("guide", GUIDES, ids=lambda path: path.name)
def test_links_and_anchors_resolve(guide):
    for match in _LINK.finditer(guide.read_text()):
        target = match["target"]
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        path_text, _, anchor = target.partition("#")
        path = (guide.parent / path_text).resolve() if path_text else guide
        assert path.exists(), f"{guide.name}: broken link {target}"
        if anchor and path.suffix == ".md":
            anchors = {_slug(title.replace("`", "")) for title in _HEADING.findall(path.read_text())}
            assert anchor in anchors, f"{guide.name}: no heading for #{anchor} in {path.name}"


@pytest.mark.parametrize(
    ("source", "fragment"),
    [
        ("platoon/envs/ors/session.py", "Not run: env busy. The call waited"),
        ("platoon/envs/ors/session.py", 'reason="no_submit"'),
        ("platoon/envs/ors/pool.py", 'DEFAULT_SESSION_URL = "http://localhost:8080"'),
        ("platoon/envs/ors/pool.py", 'POOL_ENV_VAR_PREFIX = "OPENREWARD_SESSION_URLS_"'),
        ("platoon/envs/gate.py", "DEFAULT_QUEUE_TIMEOUT_S = 240.0"),
        ("platoon/envs/harbor/task.py", "makes the environment a multi-container stack (sidecar"),
        ("platoon/envs/harbor/dataset.py", '"Skipping Harbor task %s: %s"'),
        ("platoon/envs/harbor/dataset.py", 'f"Two Harbor tasks are named {task_id!r}"'),
        ("platoon/envs/harbor/tools.py", "DEFAULT_COMMAND_TIMEOUT_S = 300.0"),
        ("platoon/gateway/dialects/openai_chat.py", "This model's maximum context length is"),
        ("platoon/gateway/core.py", 'SAMPLE_TIMEOUT_ENV = "TINKER_SAMPLE_TIMEOUT_SECONDS"'),
        ("platoon/gateway/core.py", "DEFAULT_SAMPLE_TIMEOUT_SECONDS = 600.0"),
        ("platoon/gateway/core.py", 'f"Tinker sample_async timed out after {self.sample_timeout_seconds}s"'),
        ("platoon/gateway/core.py", 'f"The sample timeout ({source}) must be finite and positive, got {seconds!r}"'),
        ("platoon/gateway/core.py", 'f"The sample timeout ({source}) must be a number of seconds, got {raw!r}"'),
        ("platoon/eval/config_defs.py", 'return f"Sampling timeout: {resolved:g} s ({source})"'),
        ("platoon/eval/config_defs.py", 'f"{key}, which wins over {SAMPLE_TIMEOUT_ENV}={variable}"'),
        ("platoon/harness/builtin/platoon_bash.py", "EXIT_API_ERROR = 2"),
        ("platoon/harness/builtin/platoon_bash.py", "EXIT_TURN_LIMIT = 3"),
        ("platoon/harness/builtin/platoon_bash.py", "EXIT_MCP_ERROR = 4"),
        ("platoon/harness/builtin/specs/platoon-bash.toml", '3 = "completed"'),
        ("platoon/sandbox/hpc.py", "provider needs the hpc-sandbox package: install the hpc extra, e.g. pip install"),
        ("platoon/sandbox/_processes.py", '"Sandbox %s shares the PID namespace of this process'),
        ("platoon/train/supervise.py", '"Training exited %s; restarting it (restart %d of at most %d)"'),
        ("platoon/train/supervise.py", '"Training exited %s, and %d starts in a row saved no new checkpoint: not'),
        ("platoon/train/supervise.py", '"Training exited %s after %d restart(s), which is watchdog.max_restarts: not'),
        ("platoon/train/supervise.py", "FRUITLESS_STARTS = 3"),
        (
            "platoon/train/policy/service.py",
            'f"The training service refused the training client (attempt {attempt}, {waited',
        ),
        ("platoon/train/policy/service.py", "TRAINING_CLIENT_OPEN_TIMEOUT_S = 20 * 60.0"),
        ("platoon/train/policy/service.py", "_FINAL_REFUSALS = frozenset({401, 403, 404, 422})"),
        ("platoon/train/reaping.py", 'f"{run_dir} is in use: {alive}; wait for it to exit, or give this platoon'),
        ("platoon/train/reaping.py", 'return f"{what} (pid {pid}) still runs it on this host"'),
        ("platoon/train/reaping.py", "ran on host {record.host}, so whether it "),
        ("platoon/train/reaping.py", '"Reaped %d sandbox(es) labelled %s %s (%s)"'),
        ("platoon/train/reaping.py", 'PROCESS_RECORD = "process.json"'),
        ("platoon/sandbox/cleanup.py", '"Cannot reap the sandboxes labelled %s (%s)"'),
        ("platoon/cli.py", 'f"platoon: not reaping the sandboxes of {run_dir}: {refusal}"'),
        (
            "platoon/train/tinker/restart_wrapper.py",
            'REMOVED = "python -m platoon.train.tinker.restart_wrapper (run_with_restart): " + removed_hint(',
        ),
        ("platoon/envs/harbor/session.py", 'reason="workdir_missing"'),
        ("platoon/envs/harbor/grading.py", '"verifier_ambiguous_primary"'),
        ("platoon/envs/harbor/grading.py", '"verifier_ambiguous_reward_files"'),
        ("platoon/envs/harbor/holdout.py", 'raise KeyError(f"Unknown split {split!r}: a hold-out lists the splits'),
        ("platoon/envs/harbor/holdout.py", 'lines = [f"hold-out: {self.rule.describe()}"]'),
        ("platoon/envs/harbor/fetch.py", '"uploads_over_mb: a session uploads'),
        ("platoon/envs/harbor/artifacts.py", "the sandbox has no tar to pack"),
        ("platoon/envs/harbor/service.py", "tasks of split {split!r} run without a network ({shown}), and"),
        ("platoon/envs/harbor/service.py", 'tasks of split {split!r} build their images from Dockerfiles, which "'),
        ("platoon/prepare.py", 'lines = ["what the run needs is not prepared on this site:", *self.problems]'),
        ("platoon/prepare.py", 'f"prepare it with: {command}"'),
        ("platoon/prepare.py", "return f\"ready: {made}{counts['current']} already current\", 0"),
        ("platoon/harness/toolchains.py", "a run of another config or platoon may be running it, so platoon prepare"),
        ("platoon/harness/toolchains.py", 'f"as it is; rebuild it in place, while no job uses it, with: {command}"'),
        ("platoon/sandbox/singularity.py", "starts a sandbox without a network only with allow_network_none: true"),
        ("platoon/sandbox/enroot.py", "starts no sandbox without a network while its provider_options set"),
        ("platoon/harness/runner.py", "which does not share this host's network (its"),
        ("platoon/cli.py", "EXIT_CONFIG = 78"),
        ("platoon/train/checkpoints/pruning.py", '"checkpoint.keep_recovery needs train.max_staleness: '),
        (
            "platoon/train/checkpoints/pruning.py",
            "{keep_recovery * recovery_every} must exceed {bound}: {max_staleness}{largest}: a recovery checkpoint ",
        ),
        ("platoon/train/checkpoints/pruning.py", '"Could not delete recovery checkpoint %s: %s'),
        (
            "platoon/train/checkpoints/pruning.py",
            '"Recovery checkpoint %s is not on the service (%s), so it is not asked for again. ',
        ),
        (
            "platoon/train/checkpoints/pruning.py",
            "stopped deleting recovery checkpoints for the rest of this start",
        ),
        ("platoon/train/checkpoints/pruning.py", "{self._failures_in_a_row} deletes failed in a row"),
        ("platoon/train/checkpoints/pruning.py", "MAX_FAILURES_IN_A_ROW = 3"),
        ("platoon/train/checkpoints/pruning.py", "the Tinker service cannot delete checkpoints: deleting"),
        ("platoon/train/checkpoints/pruning.py", "_NO_DELETION_STATUSES = (405, 501)"),
        (
            "platoon/train/checkpoints/pruning.py",
            '"Stopped deleting recovery checkpoints after waiting %g s at the end',
        ),
        ("platoon/train/checkpoints/pruning.py", "EXIT_WAIT_S = 30.0"),
        ("platoon/train/checkpoints/pruning.py", 'DELETED_CHECKPOINTS_NAME = "checkpoints_deleted.jsonl"'),
        ("platoon/cli.py", "inference.output_dir is not set, so this evaluation writes to"),
        ("platoon/eval/config_defs.py", 'EVAL_RUNS_DIR = os.path.join("runs", "eval")'),
        ("platoon/eval/run.py", "stage: report builds the report of rollouts already on disk: set inference."),
        ("platoon/eval/run.py", "{now or datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"),
        ("platoon/extensions.py", '].package: {environment.package!r} does not import "'),
        ("platoon/deprecations.py", 'f"{self.path}: deprecated; use {self.replacement} instead (platoon'),
        ("platoon/deprecations.py", 'return f"platoon {removed_in} removed it; use {replacement}"'),
        ("platoon/utils/config.py", 'return f"{_join(parent, key)}: unknown key; {hint}"'),
        ("platoon/train/config_defs.py", '"task_order": removed_hint("0.4.0", "train.task_sampler")'),
        ("platoon/site.py", "f\"the site profile's cache, whose {directory} directory it named (~/.config"),
        ("platoon/envs/harbor/service.py", '"cache": removed_hint("0.3.0", "the site profile\'s cache, whose harbor'),
        ("platoon/envs/harbor/cli.py", 'raise ConfigError(f"{path}: a fetch config file; {FETCH_CONFIG_REMOVED}")'),
        ("platoon/envs/harbor/cli.py", 'f"platoon harbor fetch {flag}, which applied to fetch config files alone: '),
        ("platoon/train/auto.py", 'f"{where}: harness {spec.ref} is a TOML harness spec, which runs no episode'),
        ("platoon/train/auto.py", '"has no grader, so every rollout would be invalid (self_grade)'),
        ("platoon/train/auto.py", 'f"{where}: program {environment.program!r} never runs: harness {names} is a TOML'),
        ("platoon/train/auto.py", 'f"{where}: agents settings never apply: harness {names} is a TOML harness spec'),
        ("platoon/train/auto.py", 'problems.append(f"{where}.agents: {problem}")'),
        ("platoon/train/auto.py", '"child_grading: env grades a child by its forked session, and a task table'),
        ("platoon/train/auto.py", '"child_grading: env on shared forks grades a child by its parent'),
        ("platoon/train/auto.py", 'f"{where}: harness {spec.ref} takes no env tools'),
        ("platoon/train/auto.py", 'takes no env tools ([mcp] is not enabled), and {reached} its own "'),
        ("platoon/train/auto.py", 'f"tools never reach the environment, so {scored} and nothing would train'),
        ("platoon/train/auto.py", 'raise ConfigError(f"{where}.service: {exc}") from exc'),
        ("platoon/envs/ors/service.py", 'serves ORS env {self.env_name!r} with other tools than env_class "'),
        ("platoon/envs/ors/service.py", 'f"env_class {self._env_class!r} does not import: '),
        ("platoon/train/exclusion.py", 'f"{self.items} match no task id or name of the environment"'),
        ("platoon/train/exclusion.py", "its tasks have ids and names like"),
        ("platoon/train/auto.py", 'f"{where}.exclude_tasks leaves none of the {listed} tasks of split {split!r}"'),
        ("platoon/train/exclusion.py", 'LOCK_KEY = "excluded_tasks"'),
        ("platoon/cli.py", 'ConfigError(f"renderer_kwargs.tool_call_parser: {exc}")'),
        ("platoon/train/tinker/renderers/tool_parsers/probe.py", "No tool-call parser reads the chat template's own"),
        ("platoon/train/tinker/renderers/tool_parsers/probe.py", "What each parser made of it:"),
        ("platoon/train/tinker/renderers/tool_parsers/probe.py", '"read no calls"'),
        ("platoon/train/tinker/renderers/tool_parsers/probe.py", '"Tool-call parser%s: %s (auto, round trip %s)"'),
        ("platoon/train/tinker/renderers/tool_parsers/probe.py", 'f"round trip ({probe.mode})"'),
        ("platoon/train/tinker/renderers/tool_parsers/probe.py", '"one turn with every call"'),
        ("platoon/train/tinker/proxy.py", '"the renderer parses tool calls itself"'),
        ("platoon/gateway/core.py", '"unparsed_tool_calls": self.unparsed_tool_calls'),
        ("platoon/utils/rollout_stats.py", '"num_tool_calls", "num_recovered_tool_calls", "num_unparsed_tool_calls"'),
        ("platoon/train/labels.py", 'f"train.label_by: no training task has a {column!r} value to label it by; their'),
        ("platoon/train/labels.py", "MAX_METRIC_LABELS = 64"),
        ("platoon/train/auto.py", "names {name!r}, a column this environment's tasks have (an index field) whose"),
        ("platoon/train/config_defs.py", '"rollout_group_key": removed_hint("0.3.0", "train.label_by")'),
        ("platoon/train/components.py", '"0.3.0", "train.label_by, whose labels straggler_policies key on"'),
        ("platoon/train/config_defs.py", '{"none": removed_hint("0.3.0", "eval.every: 0, which never'),
        ("platoon/train/config_defs.py", '"train.max_concurrent_rollouts, which counts rollouts, not groups'),
        ("platoon/train/config_defs.py", "EVAL_ROLLOUTS_AT_ONCE = 256"),
        ("platoon/train/config_defs.py", 'f"{section}.every must be a number of {unit}, or 0 for never, got'),
        ("platoon/train/config_defs.py", 'unit = "passes over the training tasks (strategy: epoch)" if'),
        ("platoon/utils/rollout_stats.py", 'REWARD_KEY_MISMATCH = "reward_key_mismatch"'),
        ("platoon/envs/service.py", 'PLATOON_ZERO_KEY = "platoon_zero"'),
        ("platoon/train/components.py", 'f"Mixture entries {by_tasks} weigh by their tasks (weight: {TASKS_WEIGHT}, '),
        ("platoon/train/components.py", 'TASKS_WEIGHT = "tasks"'),
        ("platoon/train/sampling.py", "REPEAT_RATIO_LIMIT = 4.0"),
        ("platoon/train/sampling.py", '"weight: tasks on every entry draws every task of the mixture equally often"'),
        ("platoon/train/sampling.py", 'f"training tasks: {self.tasks}, drawn in epochs of {self.epoch_size}'),
        ("platoon/train/sampling.py", 'f"{entry.share:.1%} of the slots: each task drawn about {_number(self.draws'),
        ("platoon/train/sampling.py", "f\"environments[{most.label}]'s {most.tasks} training tasks are each drawn"),
        ("platoon/train/sampling.py", "f\"slots) gives it about {reached:.0f} of the run's {slots} task slots"),
        ("platoon/train/sampling.py", 'f"{entry.tasks - reached:.0f} of them never train; {_FIX}"'),
        ("platoon/train/samplers.py", "is not one of {', '.join(map(repr, core))}, a name an environment's package"),
        ("platoon/train/samplers.py", "shares a run's slots among its entries; an entry's sampler chooses among the"),
        ("platoon/train/samplers.py", 'f"explore: must be between 0 and 1, got {explore!r}"'),
        (
            "platoon/train/samplers.py",
            '"run itself, with no entry slots, so it asks no entry\'s sampler; leave it out, or set "',
        ),
        ("platoon/train/samplers.py", "hands no slots to entries (its entry_slots is "),
        (
            "platoon/train/samplers.py",
            "every training task of the run by its statistics, with no entry's share, so it takes no weights",
        ),
        (
            "platoon/train/samplers.py",
            "chooses the tasks whose next group is likely to mix passes and failures, which a group",
        ),
        ("platoon/train/samplers.py", 'f"{where}: {name} hands slots to entries, which only the run\'s sampler does"'),
        (
            "platoon/train/samplers.py",
            "the run's per-task statistics are not available (they stopped, or never started)",
        ),
        ("platoon/options.py", "f\"{where}: {self.name} takes no options; leave {where.rsplit('.', 1)[-1]} out\""),
        ("platoon/train/sampler_api.py", 'f"unknown task sampler {ref!r}"'),
        ("platoon/train/sampler_api.py", 'f"{ref} does not import ({type(exc).__name__}: {exc})"'),
        ("platoon/train/sampler_api.py", 'f"{ref} is not a task sampler: a class with next(ctx), state_dict() and'),
        ("platoon/train/sampler_api.py", 'f"{ref} hands slots to entries and is adaptive; in this release only a'),
        (
            "platoon/train/sampler_runtime.py",
            '"Task sampler %s raised at decision %d (%s: %s): the run draws a fallback',
        ),
        ("platoon/train/sampler_runtime.py", 'f"decision {decision} answered {_brief(answer)}, which it may not"'),
        ("platoon/train/sampler_runtime.py", '"The state that task sampler %s saved does not load (%s: %s): it starts'),
        ("platoon/train/sampler_api.py", 'f"task sampler {sampler}: {message}"'),
        ("platoon/train/auto.py", '"environments[%s]: task ids repeat (%s%s): %s draws by task id, so only the first'),
        (
            "platoon/train/sampling.py",
            "f\"environments[{label}].task_sampler: {old.get('task_sampler', 'uniform')} -> \"",
        ),
        ("platoon/train/dispatch/setup.py", "entry and adaptive samplers need AutoDataset.training's sampling, which"),
        ("platoon/train/sampler_runtime.py", "its metric %r is one the dispatcher writes itself under the sampler's"),
        (
            "platoon/train/checkpoints/resume.py",
            '"Resuming from checkpoint %s: %s; each draw drawn again takes the first row',
        ),
        (
            "platoon/train/checkpoints/dispatch_state.py",
            'f"cannot read the outstanding draws\' rows: {type(exc).__name__}: {exc}"',
        ),
        ("platoon/train/checkpoints/dispatch_state.py", 'f"cannot read the tokens: {exc}"'),
        # Group admission (RFC 0019's M5): troubleshooting.md's rows.
        (
            "platoon/train/admissions.py",
            "is not one of {', '.join(map(repr, RULES))}, a name an environment's package registers, or",
        ),
        ("platoon/train/admission_api.py", 'f"unknown group admission rule {ref!r}"'),
        ("platoon/train/admissions.py", 'f"band: must be two numbers lo < hi between 0 and 1, got {shown!r}"'),
        ("platoon/train/admissions.py", "an entry's options go with the rule it sets: set {where}.group_admission"),
        (
            "platoon/train/admission_api.py",
            '"must be a finite number of at least 1 (a step under a group admission rule counts at most this many "',
        ),
        (
            "platoon/train/admissions.py",
            'f"train.num_minibatches: {minibatches} takes {minibatches} optimizer steps a batch, and a group admission',
        ),
        (
            "platoon/train/config_defs.py",
            '"when_short": "not in this release: a short step trains the groups it admitted"',
        ),
        (
            "platoon/train/config_defs.py",
            '"on_stale": "not in this release: a stale group is dropped, and under a group admission rule the step',
        ),
        (
            "platoon/train/admissions.py",
            'f"{where}: {choice.name} rejects the groups whose workflow\'s verdict says they train nothing; the "',
        ),
        ("platoon/train/admission_api.py", 'f"{ref} does not import ({type(exc).__name__}: {exc})"'),
        ("platoon/train/admission_api.py", 'f"{ref} is not a group admission rule: a class with admit(group, stats)"'),
        (
            "platoon/train/admissions.py",
            'f"train.loss_fn: {loss} with loss_fn_config keys {keys} may train terms that do not depend on the',
        ),
        (
            "platoon/train/admissions.py",
            '"Group admission rule %s failed on the group of task %s (%s: %s): the group is admitted, and "',
        ),
        (
            "platoon/train/admissions.py",
            '"Group admission rule %s admits the groups of environments[%s] whose output carries no verdict "',
        ),
        (
            "platoon/train/admissions.py",
            '"Batch %s trains %d admitted groups, fewer than train.batch_size (%d): it counted %d groups, its cap "',
        ),
        (
            "platoon/train/admissions.py",
            '"Batch %s admitted no group: of the groups it dequeued, %s, and it counted %d toward its cap, so it "',
        ),
        (
            "platoon/train/admissions.py",
            'f"{total.rejected} were rejected, {total.stale} stale and {total.failed} failed"',
        ),
        ("platoon/train/admissions.py", '"Most of them failed: see the failed groups\' errors, which no rule decides"'),
        (
            "platoon/train/admissions.py",
            '"Batch %s has counted %d groups and admitted %d of train.batch_size (%d); it waits for more, up to %d "',
        ),
        ("platoon/train/sampling.py", "f\"{where}.group_admission: {old.get('rule')} -> {new.get('rule')}\""),
        ("platoon/train/checkpoints/resume.py", 'f"position {dispatcher.next_position}, where its row says it stood"'),
    ],
)
def test_the_quoted_messages_and_defaults_are_still_in_the_code(source, fragment):
    assert fragment in (REPO / source).read_text()


def test_the_entry_in_the_harbor_guides_one_dataset_section_builds_its_service(tmp_path, monkeypatch):
    """The guide's complete Harbor entry (reward names, metadata index fields, a hold-out, a size refusal) is one the
    Harbor service takes as written; building it fetches nothing."""

    import yaml
    from site_cache import use_site_cache

    from platoon.envs.harbor import HarborEnvironmentService

    use_site_cache(monkeypatch, tmp_path / "cache")
    guide = (HOW_TO / "sandboxes-and-harbor.md").read_text()
    block = re.search(r"```yaml\n# configs/environment\.yaml\n(?P<body>.*?)```", guide, re.S)
    assert block is not None
    (entry,) = yaml.safe_load(block["body"])["environments"]
    kwargs: dict[str, Any] = dict(entry["service_kwargs"])
    service = HarborEnvironmentService(**kwargs)

    assert service.reward.names == ("reward", "correctness") and service.reward.range == (0.0, 1.0)
    assert service.holdout is not None and service.holdout.group_by == ("kaggle_dataset_name", "split_group")
    assert service.holdout.describe() == (
        "5% of each source_dataset, grouped by kaggle_dataset_name, else split_group, joined by shared input files, "
        "seed 0"
    )
    fetch = service._locked.fetch if service._locked is not None else None
    assert (
        fetch is not None and fetch.refuse.uploads_over_mb == 100 and fetch.metadata_fields == ("kaggle_dataset_name",)
    )
    assert f"hold-out: {service.holdout.describe()}" in guide  # the report the guide shows is this rule's
