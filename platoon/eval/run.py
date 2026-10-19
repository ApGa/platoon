"""``platoon eval``: evaluate a policy on the environments a config selects.

The evaluation counterpart of :mod:`platoon.train.tinker.train`: the config's
``environments`` select the tasks and their rollout (``harness_rollout`` for an
environment with a service and a harness), the ``inference`` section names the
policy (a Tinker model with its renderer and optional checkpoint, or a hosted
model behind the gateway: ``inference.hosted``) and whether judges run
(``inference.judges``), and :class:`EvalRunner` collects the rollouts and
writes a report.

Before any task is listed, every check that would otherwise fail rollouts runs
and exits 78: the config's settings, the policy's key and sampling timeout
(:func:`sample_timeout_note`, which the log gets), a hosted model's probe in
each dialect the run's harnesses speak (:func:`run_dialects`), the
environments' routes, and the entries that cannot run with judges off.
``lock.json`` gets the policy (``policy``: never a key) and the judges setting
(``judges``).  A Tinker key that the service refuses ends the evaluation at the
first rollout that opens the policy's session
(:class:`~platoon.eval.config_defs.TinkerKeyRefusedError`): no other starts.

Before any of that, and before anything is written into it, the output
directory is chosen (:func:`prepare_output_dir`): ``inference.output_dir``,
which is refused when it holds rollouts another policy made, or, when that is
unset, a new directory for this start, ``<runs>/eval/<policy>/<UTC time>``
(:func:`new_output_dir`), where ``<runs>`` is the site profile's ``runs``
(:mod:`platoon.site`), else ``runs`` under the working directory.  So an
evaluation never resumes another policy's rollouts, and one that is refused
leaves the directory as it found it.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from platoon.eval.config_defs import (
    EVAL_RUNS_DIR,
    EvalConfig,
    EvalPolicy,
    EvalWorkflowConfig,
    require_tinker_api_key,
    sample_timeout_note,
    unserved_policy_settings,
)
from platoon.eval.judges import judges_lock_record, judges_off_problems
from platoon.eval.runner import EvalRunner, check_provenance
from platoon.eval.workflow import DefaultEvalGroupWorkflow
from platoon.registry import import_from_string, resolve_component
from platoon.site import site_profile
from platoon.train.auto import (
    AutoDataset,
    AutoEnvironment,
    AutoRewardProcessor,
    AutoRollout,
    AutoTaskLoader,
    check_environment_routes,
    check_harness_toolchains,
    check_mixture_pools,
)
from platoon.train.components import HARNESS_ROLLOUT, EnvironmentConfig, normalize_environment_configs
from platoon.train.exclusion import TaskExclusion, record_exclusions
from platoon.utils.config import ConfigError, load_config, update_run_lock

logger = logging.getLogger(__name__)


@dataclass
class PlatoonEvalConfig:
    """An evaluation config (``platoon eval``): the policy, the environments and the split to evaluate."""

    # The policy, the output directory and how the rollouts run.
    inference: EvalConfig
    # The environments to evaluate, as in a training config.
    environments: list[EnvironmentConfig] = field(default_factory=list)
    # Dataset split loaded through the environment's dataset loader; any split
    # other than "train" uses its eval loader, kwargs and rollout.
    split: str = "eval"
    # full: collect rollouts, then report; rollouts: only collect rollouts;
    # report: only build the report from rollouts already on disk.
    stage: Literal["full", "rollouts", "report"] = "full"

    def __post_init__(self) -> None:
        self.environments = normalize_environment_configs(self.environments)


async def arun_eval(args: list[str] | None = None) -> dict[str, Any]:
    """Evaluate the policy of an environment-backed Platoon config and return the runner's result."""

    config, _ = load_config(args=args or sys.argv[1:], config_class=PlatoonEvalConfig)
    config: PlatoonEvalConfig = config
    inference = config.inference
    apply_judges(inference)
    unserved_settings = unserved_policy_settings(
        inference.renderer_name,
        model_path=inference.model_path,
        renderer_kwargs=inference.renderer_kwargs,
        context_window_length=inference.context_window_length,
        sample_timeout_seconds=inference.sample_timeout_seconds,
    )
    if unserved_settings and inference.hosted is None:
        raise ConfigError(
            f"inference.{', inference.'.join(unserved_settings)} configure the policy, which is served only with an "
            "inference.renderer_name; set one"
        )
    output_dir = prepare_output_dir(config, runs=site_profile().runs)  # before anything is written there
    policy = inference.policy()
    if config.stage != "report" and policy is not None:
        check_policy_environment(policy)  # else every rollout errors on it, and the report says success_rate 0.0
        if not getattr(policy, "is_hosted", False):  # exit 78 here for a sampling timeout every rollout would refuse
            logger.info(sample_timeout_note("inference.sample_timeout_seconds", inference.sample_timeout_seconds))
        for_processes = getattr(policy, "for_processes", None)
        if for_processes is not None:  # each worker process's share of a hosted model's request limit
            policy = for_processes(rollout_processes(inference.workflow))
    AutoEnvironment.load(config)
    if config.stage != "report" and policy is None:
        # Before any service is contacted: every rollout of these would fail for want of a model gateway.
        unserved = harness_rollout_environments(config, config.split)
        if unserved:
            raise ConfigError(
                f"environments {unserved} roll out with harness_rollout, whose harnesses sample the policy, but "
                "inference.renderer_name is not set, so no policy is served; set inference.renderer_name, or "
                "inference.hosted for a hosted model"
            )
    if config.stage != "report":
        check_mixture_pools(AutoEnvironment.environments(config))  # a report contacts no env server
        check_environment_routes(AutoEnvironment.environments(config))  # before any task is listed
        check_harness_toolchains(AutoEnvironment.environments(config))  # a run never builds one
        check_judges(AutoEnvironment.environments(config), enabled=inference.judges)
        update_run_lock(
            output_dir,
            {"judges": judges_lock_record(AutoEnvironment.environments(config), enabled=inference.judges)},
        )
        if policy is not None:
            hosted = getattr(policy, "is_hosted", False)
            dialects = run_dialects(AutoEnvironment.environments(config)) if hosted else []
            await lock_policy(policy, output_dir, dialects)

    dataset: list[Any] = []
    if config.stage != "report":
        exclusions: list[TaskExclusion] = []
        dataset = list(AutoDataset.from_config(config, config.split, exclusions=exclusions))
        record_exclusions(output_dir, exclusions)  # the lock.json platoon eval wrote there
    workflow = DefaultEvalGroupWorkflow(
        rollout_fn=AutoRollout.from_config(config, config.split),
        get_task_fn=AutoTaskLoader.from_config(config, config.split),
        config=config.inference.workflow,
        model_name=config.inference.model_name,
        policy=policy,
        reward_processor=AutoRewardProcessor.from_config(config),
        packages=AutoEnvironment.packages(config),
    )
    runner = EvalRunner(workflow=workflow, output_dir=output_dir)
    return await runner.arun(
        dataset=dataset,
        resume=config.inference.resume,
        retry_errored=config.inference.retry_errored,
        run_rollouts=config.stage != "report",
        generate_report=config.stage != "rollouts",
    )


def prepare_output_dir(
    config: PlatoonEvalConfig, *, runs: str | os.PathLike[str] | None = None, now: datetime | None = None
) -> str:
    """Choose the directory the evaluation writes to, before anything is written there, and set
    ``inference.output_dir`` to it.

    - ``inference.output_dir`` set: that directory, unless its rollouts were made by another policy or with judges
      switched (:func:`~platoon.eval.runner.check_provenance`, a :class:`ConfigError` that leaves it as it was).
      Its rollouts of this policy are resumed (``inference.resume``).
    - Unset: a new directory for this start, ``<runs>/eval/<policy>/<UTC time>`` (:func:`new_output_dir`), so that no
      two evaluations share one, where ``<runs>`` is ``runs`` (the site profile's ``runs``), else ``runs`` under the
      working directory. ``stage: report``, which reads rollouts already on disk, must name theirs.
    """

    inference = config.inference
    output_dir = inference.output_dir
    if output_dir is None:
        if config.stage == "report":
            raise ConfigError(
                "stage: report builds the report of rollouts already on disk: set inference.output_dir to the "
                "directory of the evaluation that made them"
            )
        root = EVAL_RUNS_DIR if runs is None else os.path.join(runs, "eval")
        output_dir = inference.output_dir = str(new_output_dir(inference, root, now=now))
    else:
        check_provenance(output_dir, inference.provenance())
    return output_dir


def new_output_dir(
    inference: EvalConfig, root: str | os.PathLike[str] = EVAL_RUNS_DIR, *, now: datetime | None = None
) -> Path:
    """Create and return a new directory for one start of an evaluation: ``<root>/<policy>/<UTC time>``, where
    ``<policy>`` is :meth:`EvalConfig.policy_dir_name` and ``<UTC time>`` is ``now`` (default: the current time) as
    ``YYYYMMDDTHHMMSSZ``.  A start in a second that already has a directory gets ``-2``, ``-3`` and so on after the
    time: the directory is created here, with ``mkdir``, which fails when another start created it first."""

    parent = Path(root) / inference.policy_dir_name()
    parent.mkdir(parents=True, exist_ok=True)
    stamp = f"{now or datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    attempt = 1
    while True:
        path = parent / (stamp if attempt == 1 else f"{stamp}-{attempt}")
        try:
            path.mkdir()
            return path
        except FileExistsError:
            attempt += 1


def check_policy_environment(policy: EvalPolicy) -> None:
    """:meth:`EvalPolicy.check_environment`: what this process lacks to serve the policy, as a ConfigError.  A policy
    object that installs a gateway of its own (a test's stand-in) and has no such check needs a Tinker key, as the
    policy it stands in for does."""

    check = getattr(policy, "check_environment", None)
    if check is None:
        require_tinker_api_key()
    else:
        check()


async def lock_policy(policy: EvalPolicy, run_dir: str | os.PathLike[str], dialects: Sequence[str] = ()) -> None:
    """Probe a hosted policy in each of ``dialects`` (:meth:`EvalPolicy.probe`, a ConfigError when it fails) and
    record the policy in the ``lock.json`` of ``run_dir``: its backend, model and base URL origin, and a hosted
    model's key variable (its name) and what the probe found (``EvalPolicy.lock_record``).  A policy object without
    them records nothing."""

    probe = await policy.probe(dialects) if hasattr(policy, "probe") else None
    lock_record = getattr(policy, "lock_record", None)
    if lock_record is not None:
        update_run_lock(run_dir, {"policy": lock_record(probe)})


def run_dialects(environments: Sequence[EnvironmentConfig]) -> list[str]:
    """The dialects the run's harnesses call the policy in, which a hosted model's base URL must speak: each harness
    spec's ``dialect`` (a verifier runs its child's harness), and Chat Completions for a function harness, whose
    ``ctx.chat`` is a Chat Completions call in process.  An entry without a harness, whose rollout is its own, adds
    none."""

    from platoon.harness.registry import get_harness
    from platoon.harness.spec import HarnessSpec

    dialects: set[str] = set()
    for environment in environments:
        for ref in environment.harness_refs():
            harness = get_harness(ref)
            dialects.add(harness.dialect if isinstance(harness, HarnessSpec) else "openai_chat")
    return sorted(dialects)


def rollout_processes(workflow: EvalWorkflowConfig) -> int:
    """How many processes call the policy at once: the worker processes of ``use_subprocesses`` that run rollouts at
    once (``subprocess_max_workers``, at most ``num_concurrent_workers``), else this one."""

    if not workflow.use_subprocesses:
        return 1
    return min(workflow.subprocess_max_workers or workflow.num_concurrent_workers, workflow.num_concurrent_workers)


def apply_judges(inference: EvalConfig) -> None:
    """Carry ``inference.judges`` into the rollouts (their config's ``judges``), which platoon eval sets; a rollout
    config that turns judges off by itself is refused, so that the report and ``lock.json`` say what ran."""

    rollout_config = inference.workflow.rollout_config
    if not rollout_config.judges and inference.judges:
        raise ConfigError(
            "inference.workflow.rollout_config.judges is set by platoon eval from inference.judges; set "
            "inference.judges: false instead"
        )
    rollout_config.judges = inference.judges


def check_judges(environments: list[EnvironmentConfig], *, enabled: bool) -> None:
    """With judges off, refuse the entries whose own code may read the verdicts no verifier gives
    (:func:`platoon.eval.judges.judges_off_problems`), before any task is listed."""

    if not enabled:
        problems = judges_off_problems(environments)
        if problems:
            raise ConfigError(problems)


def harness_rollout_environments(config: Any, split: str) -> list[str]:
    """The environments whose rollout for ``split`` is ``harness_rollout``: an environment with a harness and no
    rollout of its own (``rollout``, or for splits other than ``train`` ``eval_rollout`` first), or one that names
    ``harness_rollout``."""

    names = []
    for environment in AutoEnvironment.environments(config):
        spec = environment.rollout if split == "train" else environment.eval_rollout or environment.rollout
        if (spec is None and environment.harness is not None) or _is_harness_rollout(spec):
            names.append(environment.label)
    return names


def _is_harness_rollout(spec: str | None) -> bool:
    if spec is None:
        return False
    if spec == HARNESS_ROLLOUT:
        return True
    try:
        return resolve_component("rollout", spec) is import_from_string(HARNESS_ROLLOUT)
    except Exception:  # an unresolvable rollout fails where it is resolved, with its own error
        return False


def run_eval(args: list[str] | None = None) -> dict[str, Any]:
    return asyncio.run(arun_eval(args=args))


if __name__ == "__main__":
    run_eval()
