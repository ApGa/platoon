"""The ``platoon`` command line.

    platoon train CONFIG [key=value ...]               train with Tinker
    platoon eval CONFIG [key=value ...]                evaluate a policy on an environment split
    platoon rollout CONFIG --task ID [key=value ...]   run one task and print its trajectory tree
    platoon prepare CONFIG [key=value ...] [--dry-run] make what a config's runs need, once per site
    platoon config show CONFIG [key=value ...]         print the resolved config
    platoon config validate CONFIG ... [key=value ...] check config files
    platoon harness list | show NAME                   the harnesses: core's and the installed extensions'
    platoon program list | show NAME                   the episode programs: core's and the installed extensions'
    platoon toolchain build SPEC ... [--root DIR]      build the toolchains CLI harnesses run from in sandboxes
    platoon env list [CONFIG] | show NAME              service schemes, named environments, a config's environments
    platoon extensions list | show NAME | check        the installed extensions' components, and their problems
    platoon harbor fetch|verify|images ...             Harbor datasets: fetch and lock, verify, plan and build images
    platoon sandbox reap RUN_DIR                       reap the sandboxes of a run that no longer runs
    platoon viz ...                                    the trajectory viewer (``platoon.visualization.cli``)

Overrides use dots for nesting and indices for lists (``environments.0.harness=platoon-codeact``), ``key+=value``
appends to a list, and each value is parsed against its field's type (see :mod:`platoon.utils.config`); an option of
a harness is set the same way (``environments.0.harness_options.num_retries=5``, or under
``environments.0.eval_overrides.harness_options`` for evaluation rollouts only) and checked like the config's own.
``train``, ``eval`` and ``rollout`` resolve every component the config names (:mod:`platoon.extensions`) and check
its options (:mod:`platoon.options`), write ``config.resolved.yaml`` and ``lock.json`` (with those components, the
options each runs with and the distributions behind them) into their run directory, and run from that snapshot.
``lock.json`` also records how the policy's responses are parsed (``tool_call_parsing``): with a chat-template
renderer, the tool-call parser chosen for the model's template, and a template that no parser reads back is a
configuration error before the run starts.  A config that sets a deprecated key still loads: the commands that read
it name the key once on stderr, with what replaces it, ``lock.json`` lists it (``deprecations``), and ``config
validate --strict`` fails on it.  Listing reads the extensions' metadata only.  Commands import what they need only
when they run, so ``platoon --help`` stays fast.  Exit codes: 0 ok, 1 runtime failure, 64 usage error, 78
configuration error (never worth a restart).
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import enum
import inspect
import json
import logging
import re
import sys
import warnings
from collections.abc import Callable, Generator, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from cyclopts import App, CycloptsError

from platoon.deprecations import Deprecation, PlatoonDeprecationWarning, deprecated_keys
from platoon.prepare import NotPrepared, prepare_command
from platoon.site import SITE_ENV
from platoon.utils.config import (
    ConfigError,
    ResolvedConfig,
    distribution_version,
    load_yaml_config,
    resolve_config,
    update_run_lock,
    write_run_record,
)

EXIT_USAGE = 64
EXIT_CONFIG = 78

Kind = Literal["auto", "train", "eval", "none"]
# The kinds of component extensions provide by name (``platoon.extensions.KINDS``).
ComponentKind = Literal["harness", "program", "environment", "reward_processor"]

# Environment service references an ``environments`` entry may name, as ``platoon.envs.get_service`` reads them.
SERVICE_SCHEMES = (
    ("harbor", "Harbor task directories, one sandbox per session"),
    ("table", "task_table rows: tasks with no tools or grader, for self-grading harnesses"),
    ("python://pkg.module:EnvClass", "an ORS Environment class, run in the rollout worker"),
    (
        "ors+http://HOST:PORT[,...]/EnvName",
        "ORS env servers over HTTP (or ors+https); a pool is sharded by rollout, and ?variant=Env picks one "
        "environment of a server that serves several",
    ),
    ("ors+http:///EnvName", "ORS env servers from OPENREWARD_SESSION_URLS_<LABEL>"),
    ("pkg.module:Class", "any EnvironmentService factory, called with service_kwargs"),
)

app = App(
    name="platoon",
    help="Build and train systems of agents.",
    version=lambda: distribution_version("platoon") or "unknown",
)
config_app = App(name="config", help="Show and validate config files.")
harness_app = App(name="harness", help="List and inspect the harnesses: core's and the installed extensions'.")
program_app = App(name="program", help="List and inspect the episode programs: core's and the installed extensions'.")
toolchain_app = App(name="toolchain", help="Build the toolchains that CLI harnesses run from inside task sandboxes.")
env_app = App(name="env", help="Environment services, named environments, and the environments a config selects.")
extensions_app = App(name="extensions", help="The installed extensions: their components, versions and problems.")
sandbox_app = App(name="sandbox", help="The sandboxes a run left running.")
app.command(config_app)
app.command(harness_app)
app.command(program_app)
app.command(toolchain_app)
app.command(env_app)
app.command(extensions_app)
app.command(sandbox_app)
# Imported when it runs (`platoon --help` stays fast): platoon.envs.harbor.cli.
app.command(
    "platoon.envs.harbor.cli:harbor_app",
    name="harbor",
    help="Fetch Harbor datasets into a locked cache, verify them, and build their images.",
)

_invocation: list[str] = ["platoon"]
# The `platoon prepare` command line for the config the running command runs (train, eval, rollout), which a refusal
# for want of what prepare makes names (NotPrepared).
_prepare_for: list[str] = []


def main(argv: Sequence[str] | None = None) -> None:
    """Run the command line; ``argv`` defaults to ``sys.argv[1:]``."""

    tokens = list(sys.argv[1:] if argv is None else argv)
    if tokens[:1] == ["viz"]:
        _viz(tokens[1:])
        return
    _invocation[:] = ["platoon", *tokens]
    _prepare_for.clear()
    with warnings.catch_warnings():  # the filters _warn_of_deprecations adds hold for this command only
        try:
            code = app(tokens, exit_on_error=False, print_error=True, result_action="return_value")
        except CycloptsError:
            sys.exit(EXIT_USAGE)
        except NotPrepared as exc:  # with the prepare command for the config the command runs
            command = exc.command or (_prepare_for[0] if _prepare_for else None)
            print(f"platoon: {NotPrepared(exc.problems, command=command)}", file=sys.stderr)
            sys.exit(EXIT_CONFIG)
        except ConfigError as exc:
            print(f"platoon: {exc}", file=sys.stderr)
            sys.exit(EXIT_CONFIG)
        except PlatoonDeprecationWarning as exc:  # raised by an error warning filter
            print(f"platoon: {exc}; the warning filters make deprecations errors", file=sys.stderr)
            sys.exit(EXIT_CONFIG)
    if isinstance(code, int) and code:
        sys.exit(code)


# --- runs ----------------------------------------------------------------------------------------------------


@app.command
def train(config: Path, /, *overrides: str, prepare: bool = False) -> int:
    """Train with Tinker on the environments a config selects.

    Every component the config names is resolved first, and its options checked (exit 78 on one that does not
    resolve or an option it does not take). The resolved config and lock.json, which records those components, the
    options each runs with and the distributions behind them, go to the run directory
    (log_path/experiment_name/trial_name; without log_path, under the site profile's runs, the experiment named after
    the config file), and training loads that snapshot. The site profile it uses (~/.config/platoon/site.yaml, or the
    file $PLATOON_SITE names) and what it gave the config are named on stderr, and recorded in lock.json; a provider
    from it that this node cannot run is refused (exit 78), and so is a Harbor entry on a provider that does not
    isolate its sandboxes which the config itself does not name. A harness toolchain that is not built is refused
    before any rollout, and so is a task whose image is not built: a run never builds them, and exits 78 with the
    platoon prepare command that does, which --prepare runs first. A run directory that another platoon command still
    runs on this host is refused (exit 78), and so is one the site profile named after this config file's name that
    holds the run of another file of that name.

    With watchdog.max_restarts above 0, training runs in a child process, started again after any exit but 0, 78 and
    an interrupt, each start resuming from the newest checkpoint (--prepare prepares once, before the first). The
    sandboxes labelled with the run are reaped before training starts, before each restart and when platoon train
    exits.

    Parameters
    ----------
    config
        YAML config file.
    overrides
        key=value or key+=value config overrides, e.g. train.batch_size=64 or
        environments.0.harness_options.reasoning_replay=false.
    prepare
        First run platoon prepare on the config and overrides, with its lines on stderr, and train only if it ends
        ready; else exit with its status. Use it on a laptop or a single node. On a cluster, prepare once on a CPU
        node instead: in a GPU job, this spends GPU time building.
    """

    from platoon.eval.config_defs import require_tinker_api_key
    from platoon.site import check_run_directory
    from platoon.train.reaping import refuse_running

    _prepare_for[:] = [prepare_command(config, overrides)]
    resolved = _resolve(config, overrides, "train")
    _site(resolved)
    require_tinker_api_key()  # training runs on Tinker: checked before anything is prepared, written or reaped
    _warn_ignored_output_dirs(resolved, "train")
    settings = resolved.config
    run_dir = Path(settings.log_path) / settings.stats.experiment_name / settings.stats.trial_name
    check_run_directory(run_dir, resolved.site)  # one the site profile named after another file of this name
    if prepare and (code := _prepare_first(config, overrides)):
        return code  # not ready: training does not start
    refuse_running(run_dir, "train")  # before the record below replaces a running run's
    record = write_run_record(run_dir, resolved, argv=_invocation, components=_run_components(resolved))
    _configure_logging()
    _lock_tool_call_parsing(record.lock_path, settings.train)
    from platoon.train.supervise import run_training

    return run_training(record.config_path, run_dir, settings)


@app.command(name="eval")
def eval_(config: Path, /, *overrides: str, prepare: bool = False) -> int | None:
    """Evaluate a policy on an environment split: run every task, grade it, write trajectories and a report.

    The resolved config and lock.json go to inference.output_dir, or, when it is unset, to a new directory for each
    start, `<runs>/eval/<policy>/<UTC time>` (the checkpoint, else the model; `<runs>` is the site profile's runs,
    else runs under the working directory), which config.resolved.yaml then names. An output_dir whose rollouts
    another policy made is refused (exit 78) before anything in it changes; its rollouts of this policy are resumed.
    The evaluation loads that snapshot, and the report summary is printed as JSON. The site profile it uses is named
    on stderr, and checked as platoon train checks it. Rollouts of every split but train run with each entry's
    eval_overrides. A harness toolchain or a task image that is not built is refused before any rollout, with the
    platoon prepare command that builds it, which --prepare runs first.

    Parameters
    ----------
    config
        YAML config file.
    overrides
        key=value or key+=value config overrides, e.g. inference.workflow.num_rollouts_per_task=4 or
        environments.0.harness_options.num_retries=5.
    prepare
        First run platoon prepare on the config and overrides, with its lines on stderr, and evaluate only if it
        ends ready; else exit with its status. With stage=report, which runs no rollout, it is ignored. Use it on a
        laptop or a single node. On a cluster, prepare once on a CPU node instead: in a GPU job, this spends GPU
        time building.
    """

    from dataclasses import replace

    _prepare_for[:] = [prepare_command(config, overrides)]
    resolved = _resolve(config, overrides, "eval")
    rollouts = resolved.config.stage != "report"  # a report-only evaluation reads finished rollouts
    _site(resolved, run=rollouts, available=rollouts)  # and starts no sandbox
    if rollouts:  # nor serves a policy
        _check_policy_environment(resolved.config.inference.policy())
    _warn_ignored_output_dirs(resolved, "eval")
    if prepare and resolved.config.stage == "report":  # it reads finished rollouts: a prepare could only delay it
        print(
            "platoon: --prepare is ignored: stage=report runs no rollout, so it needs nothing that platoon prepare "
            "makes",
            file=sys.stderr,
        )
    elif prepare and (code := _prepare_first(config, overrides)):
        return code  # not ready: the evaluation does not start
    _refuse_an_output_dir_in_use(resolved.config)  # before the record below replaces a running run's
    # A report-only run reads finished rollouts and names no component.
    components = None if resolved.config.stage == "report" else _run_components(resolved)
    named = resolved.config.inference.output_dir is not None
    # Without one, each start gets a new directory under the site profile's runs, else under the working directory's.
    runs = resolved.site.profile.runs if resolved.site is not None else None
    # Refuses a directory that holds another policy's rollouts before the record below replaces its own.
    output_dir = _evaluation().prepare_output_dir(resolved.config, runs=runs)
    if not named:
        resolved.data["inference"]["output_dir"] = output_dir  # the snapshot the evaluation loads names it
        where = ", under the site profile's runs" if runs is not None else ""
        print(
            f"platoon: inference.output_dir is not set, so this evaluation writes to {output_dir}{where}",
            file=sys.stderr,
        )
        if runs is not None and resolved.site is not None:  # the record says that the profile gave it
            gave = {**resolved.site.gave, "inference.output_dir": output_dir}
            resolved = replace(resolved, site=replace(resolved.site, gave=gave))
    record = write_run_record(Path(output_dir), resolved, argv=_invocation, components=components)
    _configure_logging()
    if getattr(resolved.config, "stage", "full") != "report":
        _lock_tool_call_parsing(record.lock_path, resolved.config.inference)
    with _held_evaluation(record.run_dir, resolved.config):
        result = _evaluation().run_eval(["--config", str(record.config_path)])
    print(json.dumps(result.get("summary", result), indent=2))


def _refuse_an_output_dir_in_use(settings: Any) -> None:
    """Refuse (exit 78) an evaluation whose output directory another platoon command still runs on this host, before
    anything is written there (:func:`platoon.train.reaping.refuse_running`): its record would replace the running
    run's, which a supervised training run loads again at each restart, and two evaluations would reap each other's
    sandboxes.  A report-only evaluation is refused too: it replaces the record as well."""

    output_dir = getattr(settings.inference, "output_dir", None)
    if output_dir:
        from platoon.train.reaping import refuse_running

        refuse_running(Path(output_dir), "eval")


@contextlib.contextmanager
def _held_evaluation(run_dir: Path, settings: Any) -> Generator[None]:
    """An evaluation that runs rollouts records itself in its directory, which :func:`_refuse_an_output_dir_in_use`
    found free, and reaps the sandboxes labelled with it when it exits (:func:`platoon.train.reaping.held_run`); a
    report-only one starts none."""

    if getattr(settings, "stage", "full") == "report":
        yield
        return
    from platoon.train.reaping import held_run

    with held_run(run_dir, "eval", settings.environments, workers=False, reap_first=False):
        yield


def _check_policy_environment(policy: Any | None) -> None:
    """Refuse (exit 78) a policy this process cannot serve for want of its key: a Tinker policy without a usable
    ``TINKER_API_KEY``, or a hosted model without the variable its ``api_key_env`` names
    (``EvalPolicy.check_environment``).  ``eval`` and ``rollout`` call it before they prepare, write or reap
    anything, as ``train`` checks its key; no policy, and a stand-in that installs a gateway of its own (and has no
    such check), need none."""

    check = getattr(policy, "check_environment", None)
    if check is not None:
        check()


@app.command
def prepare(config: Path, /, *overrides: str, dry_run: bool = False) -> int:
    """Make, once per site, what a config's runs need and never make themselves: each Harbor dataset fetched into its
    lock, the images its tasks build from Dockerfiles, its harnesses' toolchains and the policy's tokenizer.

    CONFIG is a training or evaluation config, or a file of environments that they extend; only its environments are
    read, and checked as a run checks them.  The datasets go into the Harbor cache (harbor in the site profile's
    cache, else ~/.cache/platoon/harbor), the images into each entry's image map (`images/<provider>.json` in that
    cache), built with its sandbox provider's builder, and the toolchains under the toolchain root (toolchains in the
    site profile's cache).  The site profile it uses is named on stderr first, and a provider from it that this node
    cannot run is refused (exit 78), except with --dry-run.  Each step does
    nothing when what it makes is current, so prepare can run again, also on two nodes at once.  Each step prints a
    line, and the last says whether the config is ready: exit 0 when it is, 1 when a step failed (with --dry-run,
    when anything is missing), 78 on a configuration error.

    Parameters
    ----------
    config
        YAML config file: a training or evaluation config, or a file of environments.
    overrides
        key=value or key+=value config overrides, as platoon train takes them.
    dry_run
        Change nothing: say what each step would make.
    """

    return _prepare(config, overrides, dry_run=dry_run)


def _prepare(
    config: Path,
    overrides: Sequence[str],
    *,
    dry_run: bool = False,
    out: Callable[[str], None] = print,
    report_site: bool = True,
) -> int:
    """What ``platoon prepare CONFIG [overrides]`` runs: each step's line, then whether the config is ready, go to
    ``out``.  Returns its exit status, 0 when the config is ready and 1 when not; a configuration error raises
    :class:`ConfigError` (exit 78).  With ``report_site`` it names the site profile it used on stderr first, and
    refuses a provider from it that this node cannot run, unless ``dry_run``."""

    from platoon.prepare import load_environments, summary
    from platoon.prepare import prepare as run_prepare

    resolved = load_environments(config, overrides)
    if report_site:
        _site(resolved, run=False, available=not dry_run)
    _warn_of_deprecations(resolved.deprecations)
    _run_components(resolved)  # every component resolves and takes its options, as when a run starts
    steps = run_prepare(resolved, dry_run=dry_run, command=prepare_command(config, overrides), out=out)
    line, code = summary(steps, dry_run=dry_run)
    out(line)
    return code


def _prepare_first(config: Path, overrides: Sequence[str]) -> int:
    """``--prepare`` of ``train``, ``eval`` and ``rollout``: what ``platoon prepare`` runs (:func:`_prepare`), on the
    run's config and overrides, so the same steps under the same file locks.  Its lines go to stderr, and stdout keeps
    what the run prints (an evaluation's summary, a rollout's tree).  Returns its exit status; the run starts only on
    0.  The commands call it once their config and flags check out, before they write anything."""

    return _prepare(config, overrides, out=_print_to_stderr, report_site=False)  # the command named it already


def _print_to_stderr(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


# Where a Harbor entry's provider came from (platoon.site.ProviderChoice.source), in a refusal's words.
_PROVIDER_SOURCES = {
    "environment": "the named environment the entry selects (env:)",
    "site": "the site profile",
    "default": "the Harbor service's default",
}


def _site(resolved: ResolvedConfig, *, run: bool = True, available: bool = True) -> None:
    """Name on stderr the site profile the command uses and what it gave the config (:mod:`platoon.site`), then
    refuse (exit 78) what it may not run with: with ``run``, a Harbor entry on a provider that does not isolate its
    sandboxes which the run config does not name itself (a named environment's, say); with ``available``, a provider
    from the profile that this node cannot run, quoting the profile's lines."""

    use = resolved.site
    if use is None:
        return
    for line in use.lines():
        print(f"platoon: {line}", file=sys.stderr)
    problems = []
    if run:
        problems += [
            f"{choice.entry}: its sandbox provider, {choice.provider}, runs the agents' commands on this host without "
            f"isolation, and it comes from {_PROVIDER_SOURCES.get(choice.source, choice.source)}, not the run config: "
            f"name it in the run config ({choice.entry.split(' ')[0]}.service_kwargs.provider: {choice.provider}) to "
            "run it so"
            for choice in use.unisolated()
        ]
    if available:
        from platoon.sandbox import provider_unavailable

        snippet = "\n".join(f"      {line}" for line in use.profile.snippet().splitlines())
        for choice in use.providers:
            if choice.source == "site" and (why := provider_unavailable(choice.provider, choice.options)) is not None:
                problems.append(
                    f"{choice.entry}: the site profile {use.profile.path} chooses the sandbox provider "
                    f"{choice.provider}, which this node cannot run: {why}. The profile says:\n{snippet}\n    Run on a "
                    f"node that has it, or point ${SITE_ENV} at this site's profile"
                )
    if problems:
        raise ConfigError(problems)


# The rollout_config.output_dir keys a config may set.  `train`, `eval` and `rollout` replace each with a directory
# under their run directory; the field stays for library callers of RolloutConfig.
_ROLLOUT_OUTPUT_DIRS = (
    ("train", "workflow_config", "rollout_config", "output_dir"),
    ("eval", "workflow_config", "rollout_config", "output_dir"),
    ("inference", "workflow", "rollout_config", "output_dir"),
)


def _warn_ignored_output_dirs(resolved: ResolvedConfig, command: str) -> None:
    """Say on stderr which ``rollout_config.output_dir`` the config sets in vain.  ``platoon train`` leaves the
    rollout config to a workflow class the config names, which may write where it says."""

    environments = getattr(resolved.config, "environments", None) or []
    if command == "train" and any(
        getattr(entry, "workflow", "group_rollout") != "group_rollout" for entry in environments
    ):
        return
    for path in _ROLLOUT_OUTPUT_DIRS:
        node: Any = resolved.data
        for key in path:
            node = node.get(key) if isinstance(node, Mapping) else None
        if node is not None:
            key = ".".join(path)
            print(
                f"platoon: {key} is ignored: platoon {command} writes rollouts under its run directory", file=sys.stderr
            )


def _lock_tool_call_parsing(lock_path: Path, policy: Any) -> None:
    """Record in ``lock.json`` how the policy's responses are parsed (``tool_call_parsing``).

    A chat-template renderer chooses its tool-call parser here, as the run will; ``policy`` is the section or policy
    that names the model and renderer (nothing is recorded without a renderer).  No parser for a template that
    renders tools, an unknown parser name or an unknown renderer setting is a configuration error; a model that
    cannot be loaded now is recorded as such and left to the run.
    """

    renderer_name = getattr(policy, "renderer_name", None)
    if policy is None or renderer_name is None:
        return
    from platoon.train.tinker.proxy import tool_call_parsing_record
    from platoon.train.tinker.renderers import ToolCallParserError

    model_name = policy.model_name
    kwargs = dict(getattr(policy, "renderer_kwargs", None) or {})
    try:
        record = tool_call_parsing_record(model_name, renderer_name, kwargs)
    except ToolCallParserError as exc:
        raise ConfigError(f"renderer_kwargs.tool_call_parser: {exc}") from exc
    except ValueError as exc:
        if "renderer_kwargs" not in str(exc):
            raise
        raise ConfigError(str(exc)) from exc
    except Exception as exc:  # a model that cannot be loaded now: the run reports it
        logging.getLogger(__name__).warning("Could not check how %s's tool calls are parsed: %s", model_name, exc)
        record = {"model_name": model_name, "renderer_name": renderer_name, "error": f"{type(exc).__name__}: {exc}"}
    lock = json.loads(lock_path.read_text())
    lock["tool_call_parsing"] = record
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")


def _run_components(resolved: ResolvedConfig, *, split: str | None = None) -> list[dict[str, Any]]:
    """Resolve every component the config names, for lock.json; one that does not resolve is a ConfigError.  With
    ``split``, lock.json records the options of that split's rollouts only (``platoon rollout``)."""

    from platoon.extensions import run_components

    return run_components(resolved.config, split=split)


def _evaluation() -> Any:
    """The evaluation entry module, :mod:`platoon.eval.run`: ``run_eval``, which ``eval`` runs, and
    ``PlatoonEvalConfig``, the schema of configs with an ``inference`` section.  Imported only by the commands
    that need it, so ``platoon --help`` stays fast.
    """

    import importlib

    return importlib.import_module("platoon.eval.run")


def _eval_policy_class() -> Any:
    """The evaluation package's ``EvalPolicy``, which installs a Tinker policy as the process's gateway."""

    from platoon.eval import EvalPolicy

    return EvalPolicy


@app.command
def rollout(
    config: Path,
    /,
    *overrides: str,
    task: str,
    split: str | None = None,
    output_dir: Path | None = None,
    model: str | None = None,
    model_path: str | None = None,
    renderer: str | None = None,
    json_output: bool = False,
    prepare: bool = False,
) -> int | None:
    """Run one task through its configured route and print the trajectory tree, for debugging.

    Works with training and evaluation configs.  The policy is served as `platoon eval` serves it: through Tinker
    with the config's model and renderer (a training config's train section, sampled on its tinker_base_url), or
    the flags; an evaluation config's hosted model (inference.hosted) serves its model, which --model replaces,
    after the same probe.  The rollout honours an evaluation config's inference.judges, and a split other than
    train runs with the entries' eval_overrides.  The run directory (default
    runs/adhoc/TIME-TASK) gets the resolved config, lock.json, the trajectory collection, which `platoon viz
    show-dump` opens, and the rollout's event file under rollouts/events/, which `platoon viz tail` and `replay`
    read.  A task whose dataset is not fetched, or whose toolchain or image is not built, is refused with the
    platoon prepare command that makes them (the rollout never fetches), which --prepare runs first.

    Parameters
    ----------
    config
        YAML config file.
    overrides
        key=value or key+=value config overrides.
    task
        Task id; in a mixture, ENVIRONMENT::TASK_ID.
    split
        Dataset split whose rollout runs (default: train for a training config, the config's split for an
        evaluation config).
    output_dir
        Run directory for the snapshot, rollout files and trajectory collection.
    model
        Base model the policy samples (default: the config's).
    model_path
        A tinker:// sampler checkpoint to sample instead of the base model (Tinker policies only).
    renderer
        Renderer of the policy's prompts (default: the config's; Tinker policies only).
    json_output
        Print the summary as JSON instead of a tree.
    prepare
        First run platoon prepare on the config and overrides, with its lines on stderr, and run the task only if
        it ends ready; else exit with its status. It prepares the whole config, whatever --task and --split say, and
        loads the tokenizer of the model and renderer the rollout samples (--model, --renderer). Use it on a laptop
        or a single node. On a cluster, prepare once on a CPU node instead: in a GPU job, this spends GPU time
        building.
    """

    _prepare_for[:] = [prepare_command(config, overrides)]
    resolved = _resolve(config, overrides, "auto")
    if resolved.schema is None or not any(hasattr(resolved.config, name) for name in ("train", "inference")):
        raise ConfigError(f"{config}: rollout needs a training config (train:) or an evaluation config (inference:)")
    _site(resolved)
    _warn_ignored_output_dirs(resolved, "rollout")
    settings = resolved.config
    inference = hasattr(settings, "inference")
    split = split or (getattr(settings, "split", "eval") if inference else "train")
    policy = _rollout_policy(settings, model=model, model_path=model_path, renderer=renderer)
    _check_policy_environment(policy)
    if prepare:  # the tokenizer it loads is the one the rollout's policy loads: the flags replace the config's
        code = _prepare_first(config, [*overrides, *_policy_overrides(settings, model=model, renderer=renderer)])
        if code:
            return code  # not ready: the rollout does not start
    run_dir = output_dir or Path("runs/adhoc") / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{_slug(task)}"
    record = write_run_record(run_dir, resolved, argv=_invocation, components=_run_components(resolved, split=split))
    rollout_config = _rollout_config(settings, split, run_dir, task)
    if model is not None:
        rollout_config.model_name = model
    if inference:
        from platoon.eval.judges import judges_lock_record

        update_run_lock(
            run_dir, {"judges": judges_lock_record(settings.environments, enabled=settings.inference.judges)}
        )
    _configure_logging()
    _lock_tool_call_parsing(record.lock_path, policy)
    collection = asyncio.run(_run_one(settings, task, split, rollout_config, policy, run_dir=run_dir))
    (run_dir / "trajectory_collection.json").write_text(json.dumps(collection, indent=2, default=str) + "\n")
    _write_rollout_events(rollout_config.output_dir, task, collection)
    summary = summarize_collection(collection)
    print(json.dumps(summary, indent=2) if json_output else format_summary(summary))


def _entry_rollout_config(settings: Any, split: str, task_id: str | None) -> Any:
    """The rollout config of the environment entry whose task ``task_id`` names (a mixture's ``<entry>::<task id>``,
    or a single environment's any task) in ``split``'s phase; None when the entry is not known."""

    from platoon.train import entry_settings
    from platoon.train.components import TASK_ID_SEPARATOR

    labels = [environment.label for environment in getattr(settings, "environments", None) or []]
    if len(labels) == 1:
        label = labels[0]
    elif task_id is not None and TASK_ID_SEPARATOR in task_id:
        label = task_id.partition(TASK_ID_SEPARATOR)[0]
    else:
        return None
    if hasattr(settings, "inference"):
        return entry_settings.evaluation_rollout_configs(settings, split).get(label)
    own = entry_settings.run_settings(settings).get(label)
    section = None if own is None else own.train if split == "train" else own.eval
    return None if section is None else section.rollout_config


def _write_rollout_events(output_dir: str, task_id: str, collection: Mapping[str, Any]) -> None:
    """The rollout's event file under ``<output_dir>/events/``, as training and evaluation rollouts write one, for
    ``platoon viz``; never fails the command."""

    if not collection.get("trajectories"):
        return
    try:
        from platoon.visualization.event_sinks import emit_rollout_events, rollout_events_path

        emit_rollout_events(collection, rollout_events_path(output_dir, task_id, str(collection.get("id"))))
    except Exception as exc:
        print(f"platoon: could not write the rollout's event file: {exc}", file=sys.stderr)


def _rollout_config(settings: Any, split: str, run_dir: Path, task_id: str | None = None) -> Any:
    """The rollout config a run of ``split`` uses for the task ``task_id``, the one of its environment entry (an
    entry's own rollout settings, :mod:`platoon.train.entry_settings`), pointed at the run directory."""

    if hasattr(settings, "inference"):
        _evaluation().apply_judges(settings.inference)  # the rollout honours inference.judges, as platoon eval does
        own = _entry_rollout_config(settings, split, task_id)
        rollout_config = copy.deepcopy(own if own is not None else settings.inference.workflow.rollout_config)
        rollout_config.model_name = settings.inference.model_name
    else:
        phase = settings.train if split == "train" else settings.eval
        own = _entry_rollout_config(settings, split, task_id)
        rollout_config = copy.deepcopy(own if own is not None else phase.workflow_config.rollout_config)
        rollout_config.model_name = rollout_config.model_name or settings.train.model_name
    rollout_config.return_dict = True
    rollout_config.train = False
    rollout_config.output_dir = str(run_dir / "rollouts")
    return rollout_config


def _rollout_policy(settings: Any, *, model: str | None, model_path: str | None, renderer: str | None) -> Any | None:
    """The ``EvalPolicy`` the rollout's gateway serves, or None when no renderer is configured (a config error
    if policy settings are given all the same)."""

    from platoon.eval.config_defs import unserved_policy_settings

    policy_class = _eval_policy_class()
    base_url = None  # an evaluation config names no Tinker server: TINKER_BASE_URL, else the hosted service
    if hasattr(settings, "inference"):
        section = settings.inference
        hosted = getattr(section, "hosted", None)
        if hosted is not None:
            given = [flag for flag, value in (("--renderer", renderer), ("--model-path", model_path)) if value]
            if given:
                raise ConfigError(
                    f"{' and '.join(given)} configure a Tinker policy, but inference.hosted is set, which serves the "
                    "model inference.model_name names (--model names another)"
                )
            return policy_class(model_name=model or section.model_name, hosted=hosted)
        model_name, renderer_kwargs = section.model_name, getattr(section, "renderer_kwargs", {})
        renderer_name = getattr(section, "renderer_name", None)
        model_path = model_path or getattr(section, "model_path", None)
        context_window_length = getattr(section, "context_window_length", None)
    else:
        section = settings.train
        model_name, renderer_kwargs = section.model_name, section.renderer_kwargs
        renderer_name, context_window_length = section.renderer_name, section.context_window_length
        base_url = getattr(settings, "tinker_base_url", None)  # the server `platoon train` samples
    sample_timeout_seconds = getattr(section, "sample_timeout_seconds", None)
    renderer_name = renderer or renderer_name
    unserved = unserved_policy_settings(
        renderer_name,
        model_path=model_path,
        renderer_kwargs=renderer_kwargs,
        context_window_length=context_window_length,
        sample_timeout_seconds=sample_timeout_seconds,
    )
    if unserved:
        raise ConfigError(
            f"{', '.join(unserved)} configure the policy, which is served only with a renderer; set "
            "inference.renderer_name or pass --renderer"
        )
    if renderer_name is None:
        return None
    return policy_class(
        model_name=model or model_name,
        renderer_name=renderer_name,
        model_path=model_path,
        renderer_kwargs=dict(renderer_kwargs or {}),
        context_window_length=context_window_length,
        **({"base_url": base_url} if base_url is not None else {}),
        **({"sample_timeout_seconds": sample_timeout_seconds} if sample_timeout_seconds is not None else {}),
    )


def _policy_overrides(settings: Any, *, model: str | None, renderer: str | None) -> list[str]:
    """The overrides that give ``platoon prepare`` the model and renderer ``--model`` and ``--renderer`` give a
    rollout's policy (:func:`_rollout_policy`), so that ``--prepare`` loads the tokenizer the rollout loads instead of
    the config's: a model the rollout does not sample may be one this machine cannot load."""

    section = "inference" if hasattr(settings, "inference") else "train"
    return [f"{section}.{key}={value}" for key, value in (("model_name", model), ("renderer_name", renderer)) if value]


async def _run_one(
    settings: Any, task_id: str, split: str, rollout_config: Any, policy: Any | None, *, run_dir: Path | None = None
) -> dict[str, Any]:
    from platoon.prepare import check_task_prepared
    from platoon.train.auto import (
        AutoEnvironment,
        AutoRollout,
        AutoTaskLoader,
        check_harness_toolchains,
        check_mixture_pools,
    )

    AutoEnvironment.load(settings)
    environments = AutoEnvironment.environments(settings)
    check_mixture_pools(environments)
    check_harness_toolchains(environments)  # a run never builds one
    await check_task_prepared(environments, task_id, split)  # nor fetches a dataset or builds an image
    if hasattr(settings, "inference"):
        _evaluation().check_judges(environments, enabled=settings.inference.judges)
    if policy is not None and run_dir is not None:
        # A hosted model's probe in the dialects the run's harnesses speak, then lock.json's policy, as platoon eval.
        evaluation = _evaluation()
        dialects = evaluation.run_dialects(environments) if getattr(policy, "is_hosted", False) else ()
        await evaluation.lock_policy(policy, run_dir, dialects)
    task = AutoTaskLoader.from_config(settings, split)(task_id)  # the split's options: eval_overrides off train
    if rollout_config.max_steps is not None:
        task.max_steps = rollout_config.max_steps
    if policy is not None:
        policy.install()
    result = AutoRollout.from_config(settings, split)(task, rollout_config)
    if inspect.isawaitable(result):
        result = await result
    return result.to_dict() if hasattr(result, "to_dict") else result


def summarize_collection(collection: Mapping[str, Any]) -> dict[str, Any]:
    """The tree of a serialized ``TrajectoryCollection``: per agent its depth, role, steps, reward and outcome."""

    trajectories = collection.get("trajectories") or {}
    parents = {
        trajectory_id: (trajectory.get("parent_info") or {}).get("id")
        for trajectory_id, trajectory in trajectories.items()
    }

    def depth(trajectory_id: str, seen: frozenset[str] = frozenset()) -> int:
        parent = parents.get(trajectory_id)
        if parent is None or parent not in trajectories or parent in seen:
            return 0
        return depth(parent, seen | {trajectory_id}) + 1

    agents = []
    for trajectory_id, trajectory in trajectories.items():
        misc = trajectory.get("misc") or {}
        outcome = misc.get("outcome")
        task = trajectory.get("task") or {}
        agents.append(
            {
                "id": trajectory_id,
                "parent": parents[trajectory_id],
                "depth": depth(trajectory_id),
                "role": (misc.get("agent") or {}).get("role")
                or ("root" if parents[trajectory_id] is None else "child"),
                "goal": task.get("goal"),
                "steps": len(trajectory.get("steps") or []),
                "reward": trajectory.get("reward"),
                "outcome": outcome if isinstance(outcome, Mapping) else None,
                "error": trajectory.get("error_message"),
            }
        )
    root = agents[0] if agents else None
    return {
        "id": collection.get("id"),
        "task": ((next(iter(trajectories.values()), {}) or {}).get("task") or {}).get("id"),
        "agents": agents,
        "total_steps": sum(agent["steps"] for agent in agents),
        "max_depth": max((agent["depth"] for agent in agents), default=0),
        "reward": root["reward"] if root else None,
        "outcome": root["outcome"] if root else None,
    }


def format_summary(summary: Mapping[str, Any]) -> str:
    """``summarize_collection``'s summary as an indented tree."""

    outcome = summary.get("outcome")
    lines = [
        f"rollout {summary['id']}  task {summary['task']}",
        f"agents {len(summary['agents'])}  steps {summary['total_steps']}  max depth {summary['max_depth']}  "
        f"reward {summary['reward']}  outcome {_outcome_text(outcome)}",
    ]
    children: dict[str | None, list[Mapping[str, Any]]] = {}
    known = {agent["id"] for agent in summary["agents"]}
    for agent in summary["agents"]:
        children.setdefault(agent["parent"] if agent["parent"] in known else None, []).append(agent)

    def walk(agent: Mapping[str, Any], prefix: str, last: bool) -> None:
        branch = "└─ " if last else "├─ "
        text = f"{agent['role']} {str(agent['id'])[:8]}  steps {agent['steps']}  reward {agent['reward']}"
        if agent["outcome"] is not None:
            text += f"  outcome {_outcome_text(agent['outcome'])}"
        if agent["error"]:
            text += f"  error {_truncate(agent['error'])}"
        if agent["parent"] is not None and agent["goal"]:
            text += f"  goal {_truncate(agent['goal'])!r}"
        lines.append(prefix + branch + text)
        kids = children.get(agent["id"], [])
        for index, child in enumerate(kids):
            walk(child, prefix + ("   " if last else "│  "), index == len(kids) - 1)

    roots = children.get(None, [])
    for index, agent in enumerate(roots):
        walk(agent, "", index == len(roots) - 1)
    return "\n".join(lines)


def _outcome_text(outcome: Mapping[str, Any] | None) -> str:
    if not outcome:
        return "-"
    text = str(outcome.get("status"))
    if outcome.get("rewards"):
        text += " " + ", ".join(f"{name}={value}" for name, value in outcome["rewards"].items())
    if outcome.get("reason"):
        text += f" ({outcome['reason']})"
    return text


def _truncate(text: str, limit: int = 60) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")[-60:] or "task"


# --- config ----------------------------------------------------------------------------------------------------


@config_app.command(name="show")
def config_show(
    config: Path,
    /,
    *overrides: str,
    kind: Kind = "auto",
    schema: str | None = None,
    defaults: bool = False,
    entry: str | None = None,
) -> None:
    """Print the resolved config: the file merged with its extends chain and the overrides, as YAML.

    Then, as YAML comments, the effective settings of each environment entry that sets its own (workflow_config,
    eval_overrides.workflow_config, max_staleness): one block per phase the config runs (training groups, evaluation
    during training, or platoon eval's rollouts), each key with its value, the layer it came from (default,
    train.workflow_config, the named environment, the entry, eval.workflow_config or inference.workflow,
    eval_overrides) and the value that layer replaced.

    Parameters
    ----------
    config
        YAML config file.
    overrides
        key=value or key+=value config overrides.
    kind
        Schema to validate against: train, eval, none, or auto (eval when the config has an inference
        section, train when it has a train section).
    schema
        A config class to validate against instead, as module:Class.
    defaults
        Print every field, with the schema defaults filled in.
    entry
        Print only the effective settings of the environment entry of this name, whether it sets its own or not.
    """

    import yaml

    resolved = _resolve(config, overrides, kind, schema=schema)
    if entry is not None:
        print("\n".join(_entry_settings_blocks(resolved, entry)))
        return
    data: Any = resolved.data
    if defaults and resolved.schema is not None:
        data = _plain(resolved.config)
    print(yaml.safe_dump(data, sort_keys=False, default_flow_style=False, allow_unicode=True), end="")
    if blocks := _entry_settings_blocks(resolved):
        print("# The effective settings of each environment entry that sets its own: each key's value, where it came")
        print("# from, and the value that layer replaced (platoon config show CONFIG --entry NAME prints one entry's).")
        print("\n".join(f"# {line}" for line in blocks))


def _entry_settings_blocks(resolved: ResolvedConfig, entry: str | None = None) -> list[str]:
    """The blocks that say what each environment entry runs with (:func:`platoon.train.entry_settings.describe`):
    ``entry``'s, or those of every entry that sets settings of its own; none for a config without entries.  A name
    that no entry has is a :class:`ConfigError`."""

    from platoon.train import entry_settings

    config = resolved.config
    environments = getattr(config, "environments", None) if resolved.schema is not None else None
    if not environments or not any(hasattr(config, section) for section in ("train", "inference")):
        if entry is not None:
            raise ConfigError(f"--entry {entry}: the config has no environments to show")
        return []
    labels = [environment.label for environment in environments]
    if entry is not None and entry not in labels:
        raise ConfigError(
            f"--entry {entry}: the config has no environment entry of that name; its entries are " + ", ".join(labels)
        )
    settings = entry_settings.run_settings(config) if getattr(config, "train", None) is not None else None
    chosen = (
        [entry]
        if entry is not None
        else [environment.label for environment in environments if entry_settings.sets_anything(environment)]
    )
    blocks: list[str] = []
    for label in chosen:
        blocks += entry_settings.describe(config, label, settings)
    return blocks


@config_app.command(name="validate")
def config_validate(
    *configs: str | Path,
    kind: Kind = "auto",
    schema: str | None = None,
    resolve: bool = True,
    strict: bool = False,
    tasks: bool = False,
) -> int:
    """Check config files as a run loads them: extends chains, unknown keys, value types and every component they
    name.  Exits 78 when any is invalid.

    key=value overrides among the files apply to each of them, as train and eval apply theirs, so a config whose
    launcher passes some keys (log_path, a provider) is checked as it runs.  A file of environments alone, which
    training and evaluation configs extend, is checked against the entry schema, for both phases.  A named
    environment (an entry's env:) and a trainer_config class load with the config.  Then every harness, program and
    other component the config names is resolved, and the options set for them checked, as train, eval and rollout
    do when they start, which imports them; a service's options (service_kwargs) are bound to what the service takes,
    without building it.  --no-resolve skips that, for a machine without the config's extensions.  The deprecated keys
    a config sets are listed under it, and --strict fails on them.  --tasks also lists a training config's tasks, as
    platoon train does, and shows how the run draws them.

    Parameters
    ----------
    configs
        YAML config files, and key=value or key+=value overrides that apply to each.
    kind
        Schema to validate against: train, eval, none, or auto (by the config's sections; a file of environments
        alone against the entry schema).
    schema
        A config class to validate against instead, as module:Class.
    resolve
        Resolve every component the config names and check its options, as a run does when it starts (this imports
        them).
    strict
        Also fail on deprecated keys, which the release that removes them refuses, for CI.
    tasks
        Also list a training config's training tasks as platoon train lists them, which reads the datasets and may
        contact env servers, fail on what listing them finds wrong, as train does, and print how the run draws them:
        each environment's share and how often each of its tasks is drawn over the planned steps, with a warning for a
        mixture's weights that leave tasks out of the run or draw some far more often than others.
    """

    items = [str(item) for item in configs]
    paths = [Path(item) for item in items if "=" not in item or Path(item).is_file()]
    overrides = [item for item in items if "=" in item and not Path(item).is_file()]
    if not paths:
        print("platoon config validate: name at least one config file", file=sys.stderr)
        return EXIT_USAGE
    failed = 0
    for path in paths:
        resolved: ResolvedConfig | None = None
        problems: list[str] = []
        listed: list[str] = []
        try:
            resolved = _resolve(path, overrides, kind, schema=schema, report=False)
            if resolve and _names_components(resolved):
                _run_components(resolved)
            if tasks:
                listed = _training_tasks(resolved)
        except ConfigError as exc:
            problems = exc.problems
            if exc.source is not None and Path(exc.source).resolve() != path.resolve():  # the site profile, a base
                problems = [f"{exc.source}: {problem}" for problem in problems]
        deprecations = [deprecation.message for deprecation in resolved.deprecations] if resolved is not None else []
        entry_lines = _entry_settings_lines(resolved) if resolved is not None and not problems else []
        if problems or (strict and deprecations):
            failed += 1
            print(f"FAIL {path}")
        else:
            label = resolved.schema.__name__ if resolved is not None and resolved.schema is not None else "no schema"
            print(f"ok   {path} ({label})")
        for line in (*problems, *deprecations, *entry_lines, *listed):
            print(f"  {line}")
    return EXIT_CONFIG if failed else 0


def _entry_settings_lines(resolved: ResolvedConfig) -> list[str]:
    """``config validate``'s lines about the entries' own settings: one per entry of a training config with settings
    that differ from the run's, saying what differs, then each warning (``warning: ...``); for an evaluation config,
    the entry keys ``platoon eval`` does not apply, as warnings."""

    from platoon.train import entry_settings

    config = resolved.config
    if resolved.schema is None or not getattr(config, "environments", None):
        return []
    if getattr(config, "train", None) is not None:
        settings = entry_settings.run_settings(config)
        lines = entry_settings.summary(config, settings)
        return [*lines, *(f"warning: {line}" for line in entry_settings.warnings(config, settings))]
    if getattr(config, "inference", None) is not None:
        return [f"warning: {line}" for line in entry_settings.evaluation_unapplied(config)]
    return []


def _training_tasks(resolved: ResolvedConfig) -> list[str]:
    """``config validate --tasks``: list a training config's training tasks, and its evaluation tasks when it
    evaluates, as ``platoon train`` does, before it opens the training service, and describe how the run draws them
    (:class:`~platoon.train.sampling.SamplingPlan`), its warnings and those about the listed labels prefixed
    ``warning:``.  What train would refuse once the tasks are listed (an ``exclude_tasks`` item that names no task, a
    batch larger than the training tasks, a label's straggler quorum above the group size of an entry whose tasks carry
    it) raises :class:`ConfigError`; a listing that fails otherwise, such as an env server that does not answer, is a
    problem of the config too."""

    config = resolved.config
    if resolved.schema is None or getattr(config, "train", None) is None:
        return ["tasks: only a training config's tasks are listed"]
    from platoon.train.auto import AutoDataset, AutoEnvironment, check_mixture_pools
    from platoon.train.tinker.config_defs import evaluates, training_problems

    AutoEnvironment.load(config)
    check_mixture_pools(AutoEnvironment.environments(config))  # before any env server is contacted, as train does
    try:
        training = AutoDataset.training(config)
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigError(f"the training tasks cannot be listed: {type(exc).__name__}: {exc}") from exc
    count = len(training.dataset) if hasattr(training.dataset, "__len__") else None
    without = training_problems(config)
    if problems := [problem for problem in training_problems(config, num_train_tasks=count) if problem not in without]:
        raise ConfigError(problems)
    from platoon.train import entry_settings

    try:  # with evaluation off the eval split is never read, as in platoon train
        eval_tasks = AutoDataset.from_config(config, "eval") if evaluates(config) else None
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigError(f"the evaluation tasks cannot be listed: {type(exc).__name__}: {exc}") from exc
    settings = entry_settings.run_settings(config)
    # A label's straggler quorum above the group size of an entry whose tasks carry it, as platoon train checks.
    if problems := entry_settings.listed_straggler_problems(config, settings, training.tasks, eval_tasks):
        raise ConfigError(problems)
    listed = [f"warning: {line}" for line in entry_settings.listed_permit_warnings(config, settings, training.tasks)]
    if training.plan is None:
        return ["tasks: listed", *listed]
    return [*training.plan.describe(), *(f"warning: {warning}" for warning in training.plan.warnings()), *listed]


def _names_components(resolved: ResolvedConfig) -> bool:
    """Whether a run of the config resolves the components it names when it starts: one loaded with a schema and
    environments, unless it is a report-only evaluation (``stage: report``), which reads finished rollouts."""

    config = resolved.config
    return resolved.schema is not None and hasattr(config, "environments") and getattr(config, "stage", "") != "report"


def _resolve(
    config: Path, overrides: Sequence[str], kind: Kind, *, schema: str | None = None, report: bool = True
) -> ResolvedConfig:
    """Load a config as the commands do.  With ``report``, name each deprecated key it sets on stderr
    (:func:`_warn_of_deprecations`); ``config validate`` lists them itself."""

    if not config.is_file():
        raise ConfigError(f"{config}: no such config file")
    if schema is not None:
        target = _import_class(schema)
    else:
        target = _schema_for(kind, load_yaml_config(config))
    resolved = resolve_config(config, overrides, target, warn=False)
    if report:
        _warn_of_deprecations(resolved.deprecations)
    return resolved


def _warn_of_deprecations(deprecations: Sequence[Deprecation]) -> None:
    """Name each deprecated key a config sets on stderr, once per command, with what replaces it.

    Each first goes through Python's warnings as a ``PlatoonDeprecationWarning``, so the warning filters apply: an
    error filter raises it, which exits 78, and an ignore filter silences it.  Then the same warning is ignored for
    the rest of the command (``main`` restores the filters after it), so a run that loads its snapshot of the config
    again does not repeat it.
    """

    for deprecation in deprecations:
        with warnings.catch_warnings(record=True) as shown:
            warnings.warn(deprecation.message, PlatoonDeprecationWarning, stacklevel=2)
        if shown:
            print(f"platoon: {deprecation.message}", file=sys.stderr)
        warnings.filterwarnings("ignore", re.escape(deprecation.message) + r"\Z", PlatoonDeprecationWarning)


def _schema_for(kind: Kind, data: Mapping[str, Any]) -> type | None:
    if kind == "auto" and set(data) == {"environments"}:  # a file that training and evaluation configs extend
        from platoon.train.components import EnvironmentsConfig

        return EnvironmentsConfig
    if kind == "auto":
        kind = "eval" if "inference" in data else "train" if "train" in data else "none"
    if kind == "train":
        from platoon.train.tinker.config_defs import PlatoonTinkerRLTrainerConfig

        return PlatoonTinkerRLTrainerConfig
    if kind == "eval":
        return _evaluation().PlatoonEvalConfig
    return None


def _import_class(reference: str) -> type:
    import importlib

    module_name, separator, name = reference.partition(":")
    if not separator:
        raise ConfigError(f"--schema {reference!r}: expected module:Class")
    try:
        value: Any = importlib.import_module(module_name)
        for part in name.split("."):
            value = getattr(value, part)
    except (ImportError, AttributeError) as exc:
        raise ConfigError(f"--schema {reference!r} cannot be imported: {exc}") from exc
    return value


def _plain(value: Any) -> Any:
    """A built config as plain YAML-safe data, without the deprecated keys it leaves at their defaults: a config
    copied from ``config show --defaults`` should not set a key that is going away."""

    import dataclasses

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        deprecated = deprecated_keys(type(value))
        return {
            field.name: _plain(getattr(value, field.name))
            for field in dataclasses.fields(value)
            if field.name not in deprecated or getattr(value, field.name) != field.default
        }
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, enum.Enum):
        return value.value
    return value


# --- harnesses -------------------------------------------------------------------------------------------------


def _harness_registry() -> tuple[Any, Any]:
    """``(list refs, get by ref)`` from the harness registry: TOML specs and ``@harness`` functions."""

    from platoon.harness import registry

    return registry.available_harnesses, registry.get_harness


@harness_app.command(name="list")
def harness_list() -> None:
    """List the harnesses: TOML specs and function harnesses, with the distribution and version each comes from.

    The installed extensions' harnesses come from their metadata, without importing them, so their kind shows as
    "-" until `platoon harness show NAME` imports one.
    """

    from platoon.harness.registry import harness_load_failures, harness_origin

    available, get = _harness_registry()
    rows = []
    names = set()
    for ref in available():
        harness = get(ref)
        kind, source = _harness_kind(harness)
        origin = harness_origin(harness) if hasattr(harness, "name") else None
        rows.append((ref, kind, origin.label() if origin is not None else "-", _short_source(source)))
        names.add(ref.partition("@")[0])
    rows += [
        (info.name, "-", _distribution_label(info), _with_status(info.value, info.status))
        for info in _installed("harness")
        if info.name not in names
    ]
    print(_table(("harness", "kind", "distribution", "source"), rows))
    for name, why in harness_load_failures().items():
        print(f"platoon: {name} failed to load, so its harnesses are missing: {why}", file=sys.stderr)


@harness_app.command(name="show")
def harness_show(name: str, /) -> None:
    """Show one harness: a TOML spec's launch, built-ins, limits and options, or a function harness's details.

    Parameters
    ----------
    name
        Harness reference, name[@version], or a .toml spec path.
    """

    _, get = _harness_registry()
    try:
        harness = get(name)
    except (ValueError, OSError, ImportError, AttributeError) as exc:  # as a run config's harness reference
        raise ConfigError(str(exc)) from exc
    kind, source = _harness_kind(harness)
    lines = [f"{harness.ref}  ({kind})", f"source: {source}"]
    if kind == "spec":
        lines += [
            f"dialect: {harness.dialect}",
            f"timeout_s: {harness.timeout_s}",
            f"max_steps: {harness.max_steps}",
            f"install: {list(harness.install.commands) or '-'}"
            + (f"  toolchain {harness.install.toolchain}" if harness.install.toolchain else ""),
            f"launch: {' '.join(harness.launch.argv)}",
            f"launch env: {sorted(harness.launch.env) or '-'}",
            f"built-in tools: {list(harness.builtins.tools) or '-'}"
            + ("  (can be disabled)" if harness.builtins.disable is not None else ""),
            f"final message: {_final_message_text(harness.final_message)}",
            f"exit codes: {dict(harness.exit_codes) or '-'}",
        ]
        mcp = getattr(harness, "mcp", None)
        if mcp is not None:
            lines.append(f"mcp: {mcp}")
        lines += _spec_extras(harness)
        lines += _spec_options(harness)
    else:
        lines += [f"runtime: {harness.runtime}", f"executes_model_code: {harness.executes_model_code}"]
        doc = inspect.getdoc(harness.fn)
        if doc:
            lines += ["", doc]
    print("\n".join(lines))


def _spec_extras(spec: Any) -> list[str]:
    """``harness show`` lines for a spec's toolchain recipe, resources and step-budget enforcement, when set."""

    lines = []
    recipe = getattr(spec.install, "recipe", None)
    if recipe is not None:
        generated = f", generates {list(recipe.generate)}" if recipe.generate else ""
        caches = f", caches {list(recipe.cache)}" if recipe.cache else ""
        lines.append(f"toolchain recipe: Python {recipe.python}, lock {recipe.lock.name}{caches}{generated}")
    memory_mb = getattr(getattr(spec, "resources", None), "memory_mb", None)
    if memory_mb is not None:
        lines.append(f"resources: memory_mb {memory_mb}")
    enforce = getattr(getattr(spec, "budget", None), "enforce", "harness")
    if enforce != "harness":
        lines.append(f"step budget: enforced by the {enforce}")
    return lines


def _spec_options(spec: Any) -> list[str]:
    """``harness show`` lines for a spec's options (``[options]``): name, type, default, the values it takes and what
    it does; an environments entry sets them in ``harness_options``."""

    options = getattr(spec, "options", None) or {}
    if not options:
        return ["options: - (harness_options must be empty)"]
    rows = [
        (name, option.type, option.render(option.default), option.constraint(), option.description or "-")
        for name, option in options.items()
    ]
    table = _table(("option", "type", "default", "values", "description"), rows, indent="  ")
    return ["options (set in an environments entry's harness_options):", table]


def _harness_kind(harness: Any) -> tuple[str, str]:
    if hasattr(harness, "launch"):
        return "spec", str(harness.source or "-")
    return "function", str(getattr(harness, "source", "-"))


def _short_source(source: str) -> str:
    """A path inside the platoon package, from the package (``platoon/harness/builtin/specs/...``); anything else
    as it is."""

    path = Path(source)
    if not path.is_absolute():
        return source
    package = Path(__file__).resolve().parent
    return str(path.resolve().relative_to(package.parent)) if path.resolve().is_relative_to(package) else source


def _installed(kind: str) -> list[Any]:
    """The installed extensions' components of ``kind``, from their metadata (no imports)."""

    from platoon.extensions import list_components

    return [info for info in list_components(kind) if not info.builtin]


def _distribution_label(info: Any) -> str:
    if info.distribution is None:
        return "-"
    return f"{info.distribution} {info.version}" if info.version else info.distribution


def _with_status(value: str, status: str) -> str:
    return value if status == "ok" else f"{value}  ({status})"


def _final_message_text(final_message: Any) -> str:
    if final_message is None:
        return "-"
    if final_message.file:
        return final_message.file + (f" [{final_message.json_key}]" if final_message.json_key else "")
    return f"stdout /{final_message.stdout_regex}/"


# --- programs --------------------------------------------------------------------------------------------------


@program_app.command(name="list")
def program_list() -> None:
    """List the episode programs an environment's `program:` can name, with the distribution and version each
    comes from.  The installed extensions' programs come from their metadata, without importing them."""

    from platoon.extensions import list_components

    rows = [
        (info.name, _distribution_label(info), _with_status(info.value, info.status))
        for info in list_components("program")
    ]
    print(_table(("program", "distribution", "source"), rows))


@program_app.command(name="show")
def program_show(name: str, /) -> None:
    """Show one episode program: its reference, source, distribution and docstring.

    Parameters
    ----------
    name
        Program reference, name[@version], or module:function.
    """

    from platoon.extensions import origin_of
    from platoon.programs import resolve_program

    try:
        record = resolve_program(name)
    except (ValueError, ImportError, AttributeError, TypeError) as exc:  # as a run config's program reference
        raise ConfigError(str(exc)) from exc
    lines = [
        record.ref,
        f"source: {record.source}",
        f"distribution: {origin_of('program', record.ref, record.fn).label()}",
    ]
    doc = inspect.getdoc(record.fn)
    if doc:
        lines += ["", doc]
    print("\n".join(lines))


# --- toolchains ------------------------------------------------------------------------------------------------


@toolchain_app.command(name="build")
def toolchain_build(*specs: str, root: Path | None = None, force: bool = False, uv: str | None = None) -> int:
    """Build the toolchain of each harness spec from its [install.recipe]: a uv-managed CPython, the spec's locked
    packages, caches and generated files, and a manifest with the tree hash.

    A toolchain named by name is built at ROOT/NAME, the path sandboxes bind it at; one built from the same
    recipe is left as it is.  Needs uv and network access to PyPI.  Exits 1 when a build fails.

    Parameters
    ----------
    specs
        Harness references (name[@version]) or .toml spec paths.
    root
        Directory of named toolchains (default: toolchains in the site profile's cache, else
        ~/.cache/platoon/toolchains).
    force
        Rebuild toolchains that are already built.
    uv
        The uv executable (default: uv on the PATH).
    """

    from platoon.harness.registry import get_harness_spec
    from platoon.harness.spec import HarnessSpecError
    from platoon.harness.toolchains import ToolchainBuildError, build_toolchain
    from platoon.site import variable_deprecations, variable_problems

    if not specs:
        print("platoon toolchain build: name at least one harness spec", file=sys.stderr)
        return EXIT_USAGE
    if root is None:  # the root comes from the site profile, or a deprecated variable
        if problems := variable_problems():
            raise ConfigError(problems)
        _warn_of_deprecations(variable_deprecations())
    try:
        resolved = [get_harness_spec(spec) for spec in specs]
    except (ValueError, OSError) as exc:
        raise ConfigError(str(exc)) from exc
    failed = 0
    for spec in resolved:
        try:
            manifest = build_toolchain(spec, root=root, uv=uv, force=force, log=lambda line: print(line, flush=True))
        except HarnessSpecError as exc:
            raise ConfigError(str(exc)) from exc
        except ToolchainBuildError as exc:
            failed += 1
            print(f"platoon: {spec.ref}: {exc}", file=sys.stderr)
            continue
        print(f"{spec.ref}: {manifest.build_path} ({manifest.tree_hash})")
    return 1 if failed else 0


# --- environments ----------------------------------------------------------------------------------------------


@env_app.command(name="list")
def env_list(config: Path | None = None, /) -> None:
    """Show the environment service schemes, the named environments an entry's `env:` can select, and the
    environments a config selects when one is given.

    The installed extensions' named environments come from their metadata, without importing them.

    Parameters
    ----------
    config
        YAML config file whose environments to list.
    """

    print("Environment services (an environments entry's `service`):")
    print(_table(("service", "what"), SERVICE_SCHEMES, indent="  "))
    from platoon.extensions import list_components

    named = [
        (info.name, _distribution_label(info), _with_status(_short_source(info.value), info.status))
        for info in list_components("environment")
    ]
    print("\nNamed environments (an environments entry's `env`):")
    if named:
        print(_table(("env", "distribution", "source"), named, indent="  "))
    else:
        print("  (none installed)")
    if config is None:
        return
    if not config.is_file():
        raise ConfigError(f"{config}: no such config file")
    data = load_yaml_config(config)
    environments = data.get("environments") or []
    if not isinstance(environments, list):
        raise ConfigError(f"{config}: environments must be a list")
    from platoon.envs.named import expand_environments, names_environments
    from platoon.train.components import TASKS_WEIGHT

    if names_environments(data):
        environments = expand_environments(data)["environments"]
    rows = []
    for index, entry in enumerate(environments):
        entry = entry if isinstance(entry, Mapping) else {}
        harness = entry.get("harness")
        if isinstance(harness, Mapping):
            harness = ", ".join(f"{name}:{weight}" for name, weight in harness.items())
        rollout = entry.get("rollout") or ("harness_rollout" if entry.get("harness") else "-")
        rows.append(
            (
                entry.get("name") or ("default" if len(environments) == 1 else f"#{index}"),
                entry.get("weight", TASKS_WEIGHT),
                entry.get("service") or "-",
                harness or "-",
                entry.get("program") or "-",
                rollout,
            )
        )
    print(f"\nEnvironments of {config}:")
    if not rows:
        print("  (none: the config sets no environments)")
        return
    print(_table(("name", "weight", "service", "harness", "program", "rollout"), rows, indent="  "))


@env_app.command(name="show")
def env_show(name: str, /) -> None:
    """Show one named environment: its version, distribution, description, the partial environments entry it
    stands for and its options for each harness.

    Parameters
    ----------
    name
        Named environment, name[@version], or module:attr.
    """

    import yaml

    from platoon.envs.named import environment_source, get_environment
    from platoon.extensions import origin_of

    try:
        entry = get_environment(name)
    except ValueError as exc:  # as a run config's env reference
        raise ConfigError(str(exc)) from exc
    lines = [
        entry.ref,
        f"source: {environment_source(entry.ref) or '-'}",
        f"distribution: {origin_of('environment', entry.ref, None).label()}",
    ]
    if entry.description:
        lines.append(f"description: {entry.description}")
    body = {"entry": entry.entry(), "harness_defaults": dict(entry.harness_defaults)}
    lines.append(yaml.safe_dump(body, sort_keys=False, default_flow_style=False, allow_unicode=True).rstrip())
    print("\n".join(lines))


# --- extensions ------------------------------------------------------------------------------------------------


@extensions_app.command(name="list")
def extensions_list(*, kind: ComponentKind | None = None, json_output: bool = False) -> None:
    """List every component the installation provides: core's built-ins, then each installed extension's, from
    the extensions' metadata only (nothing of an extension is imported), with the distribution and version that
    provide it and whether this platoon can load it.

    Parameters
    ----------
    kind
        Only components of this kind.
    json_output
        Print JSON instead of a table.
    """

    from platoon.extensions import list_components

    components = list_components(kind)
    if json_output:
        print(json.dumps([component.to_dict() for component in components], indent=2))
        return
    rows = [
        (
            component.kind,
            component.name,
            component.distribution or "-",
            component.version or "-",
            component.status,
            _short_source(component.value),
        )
        for component in components
    ]
    print(_table(("kind", "name", "distribution", "version", "status", "source"), rows))


@extensions_app.command(name="show")
def extensions_show(
    name: str, /, *, kind: ComponentKind | None = None, load: bool = False, json_output: bool = False
) -> int:
    """Show an installed component, as `extensions list` lists it: its kind, name, version, distribution, entry
    point, requirement on platoon, status and where configs use it.  A name that several kinds use shows one of
    each.  Reads the installed metadata only, so an extension's version shows once --load imports it.  Exits 78 when
    nothing installed has the name, or when --load fails.

    Parameters
    ----------
    name
        Component name, name[@version].
    kind
        Only a component of this kind.
    load
        Also import the component's entry point, as a config naming it would, to show what it registers.
    json_output
        Print JSON: the fields of `extensions list --json-output`, with entry_point, used_in, registers and error.
    """

    from platoon.extensions import ExtensionError, show_components

    try:
        components = show_components(name, kind, load=load)
    except ExtensionError as exc:
        raise ConfigError(str(exc)) from exc
    if json_output:
        print(json.dumps([component.to_dict() for component in components], indent=2))
    else:
        print("\n\n".join(_component_text(component) for component in components))
    return EXIT_CONFIG if any(component.error for component in components) else 0


def _component_text(component: Any) -> str:
    """``platoon extensions show``'s lines for one component (``platoon.extensions.ComponentDetails``)."""

    info = component.info
    if component.registers is None:
        version = "unknown until it is loaded (--load imports it)"
    else:
        version = ", ".join(sorted({ref.partition("@")[2] or "-" for ref in component.registers})) or "-"
    if info.builtin:
        requires = "- (part of platoon)"
    else:
        requires = ", ".join(info.requires) or "- (states no requirement on platoon)"
    lines = [
        f"{info.name}  ({info.kind})",
        f"version: {version}",
        f"distribution: {_distribution_label(info)}",
        f"entry point: {component.entry_point or '- (built into platoon)'}",
        f"source: {_short_source(info.value)}",
        f"requires: {requires}",
        f"status: {info.status if component.error is None else f'failed to load: {component.error}'}",
        f"used in: {', '.join(component.used_in)}",
    ]
    if {ref.partition("@")[0] for ref in component.registers or ()} - {info.name.partition("@")[0]}:
        lines.append(f"registers: {', '.join(component.registers)}")  # an entry point named after its module
    return "\n".join(lines)


@extensions_app.command(name="check")
def extensions_check(*distributions: str, load: bool = False) -> int:
    """Check the installed extensions, or the named distributions: component names, one owner per distribution,
    the requirement on platoon (present, bounded above, admitting this platoon) and entries in retired entry-point
    groups; with --load, also import every component as a config naming it would.  Exits 78 on an error.

    Parameters
    ----------
    distributions
        Distribution names to check (default: every installed extension).
    load
        Also import each component and check what it registers.
    """

    from platoon.extensions import check_extensions

    problems = check_extensions(distributions, load=load)
    for problem in problems:
        print(problem)
    errors = sum(problem.severity == "error" for problem in problems)
    warnings = len(problems) - errors
    print(f"{errors} error(s), {warnings} warning(s)" if problems else "ok: no problems found")
    return EXIT_CONFIG if errors else 0


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]], *, indent: str = "") -> str:
    cells = [[str(cell) for cell in header], *[[str(cell) for cell in row] for row in rows]]
    widths = [max(len(row[column]) for row in cells) for column in range(len(header))]
    return "\n".join(
        indent + "  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip() for row in cells
    )


# --- sandboxes -------------------------------------------------------------------------------------------------


@sandbox_app.command(name="reap")
def sandbox_reap(run_dir: Path, /) -> int:
    """Reap the sandboxes a run left running when it was killed: those labelled with the run (platoon.run: its run
    directory), through the sandbox providers its config.resolved.yaml names (each Harbor entry's provider, else the
    site profile's).

    platoon train and platoon eval reap them themselves when they exit, and platoon train when it starts again, so
    this is for a run whose process was killed. It refuses (exit 1) while a process of the run still runs on this
    host, and when the run last ran on another host, which holds its sandboxes. Prints how many it reaped.

    Parameters
    ----------
    run_dir
        The run directory: a training run's log_path/experiment_name/trial_name, or an evaluation's
        inference.output_dir.
    """

    from platoon.train.reaping import reap_stopped_run
    from platoon.utils.config import RESOLVED_CONFIG_NAME

    snapshot = run_dir / RESOLVED_CONFIG_NAME
    if not snapshot.is_file():
        print(
            f"platoon: {run_dir} holds no {RESOLVED_CONFIG_NAME}: it is not the directory of a run of platoon train or "
            "platoon eval",
            file=sys.stderr,
        )
        return EXIT_USAGE
    environments = load_yaml_config(snapshot).get("environments")
    _configure_logging()
    reaped, refusal = reap_stopped_run(run_dir, environments if isinstance(environments, list) else None)
    if refusal is not None:
        print(f"platoon: not reaping the sandboxes of {run_dir}: {refusal}", file=sys.stderr)
        return 1
    print(f"reaped {reaped} sandbox(es) of {run_dir}")
    return 0


# --- viz -------------------------------------------------------------------------------------------------------


@app.command
def viz() -> None:
    """Open the trajectory viewer: tail, replay or show-dump event files (`platoon viz --help` lists them)."""


def _viz(args: list[str]) -> None:
    """Hand ``args`` to ``platoon.visualization.cli.main``; ``sys.argv`` names the program for its usage line."""

    from platoon.visualization import cli as visualization_cli

    saved = sys.argv
    sys.argv = ["platoon viz", *args]
    try:
        if inspect.signature(visualization_cli.main).parameters:
            visualization_cli.main(args)
        else:  # a viewer whose main reads sys.argv
            visualization_cli.main()
    finally:
        sys.argv = saved


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


if __name__ == "__main__":
    main()
