"""The Harbor environment service: Harbor task directories served as platoon env sessions."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import posixpath
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar

from platoon.deprecations import removed_hint
from platoon.envs.gate import DEFAULT_QUEUE_TIMEOUT_S, check_queue_timeout
from platoon.envs.harbor import sidecars
from platoon.envs.harbor.artifacts import DEFAULT_MAX_ARTIFACT_BYTES
from platoon.envs.harbor.compose import MAIN, MainService, ServiceHealthcheck, image_healthcheck, resolve_deferred
from platoon.envs.harbor.dataset import (
    HarborDataset,
    SplitManifest,
    _load_splits,
    read_registry,
    read_task_index,
    task_dir_of,
)
from platoon.envs.harbor.fetch import FetchError
from platoon.envs.harbor.grading import RewardContract, classifier_list, classify
from platoon.envs.harbor.holdout import EVAL, SPLITS, HoldoutError, HoldoutRule, HoldoutSplit, hold_out
from platoon.envs.harbor.images import (
    FileOp,
    ImageMapError,
    SessionImage,
    _check_contained,
    apply_image_files,
    docker_image_reference,
    has_image,
    pulled_key,
    withholds_modes,
)
from platoon.envs.harbor.lock import LockError
from platoon.envs.harbor.locked import LockedTasks, TaskImages
from platoon.envs.harbor.session import (
    DEFAULT_CLOSE_TIMEOUT_S,
    LOG_DIRS,
    PUBLISHED_DIRS,
    TASK_TOOLS_LISTING_TIMEOUT_S,
    HarborSession,
    SandboxLease,
    StackLease,
    TaskTools,
)
from platoon.envs.harbor.task import (
    _PLANNED_HERE,
    AGENT_USER_POLICIES,
    HARBOR_MISC_KEY,
    VERIFIER_USER_POLICIES,
    AgentUserPolicy,
    HarborTask,
    HarborTaskError,
    Healthcheck,
    VerifierUserPolicy,
    is_root,
    resolve_env,
)
from platoon.envs.harbor.tools import DEFAULT_COMMAND_TIMEOUT_S, TOOL_SPECS
from platoon.envs.harbor.users import hook_users, nobody_problem, phase_users, root_problem, runs_image_user
from platoon.envs.harbor.views import SessionUsers, root_view, session_users
from platoon.envs.service import EnvNotPrepared, EnvSetupError, Outcome, PathOption, SessionNeeds
from platoon.sandbox import (
    Mount,
    Sandbox,
    SandboxProvider,
    SandboxSpec,
    StackStartError,
    file_owners_refusal,
    get_provider,
    loses_file_owners,
    network_none_refusal,
    privilege_refusal,
    root_refusal,
    rootfs_layout_refusal,
    stack_refusal,
)
from platoon.sandbox._processes import alive_processes, list_processes, start_background
from platoon.sandbox.routes import RouteError
from platoon.sandbox.stacks import Healthcheck as StackHealthcheck
from platoon.sandbox.stacks import StackNetwork, StackProvider
from platoon.task import Task

logger = logging.getLogger(__name__)

DEFAULT_SANDBOX_START_BUDGET_S = 600.0
# How a service runs a task that is a multi-container stack: where its provider runs stacks (auto), or never.
STACK_POLICIES = ("auto", "refuse")
# The network of a task's sandboxes: the task's network_mode for each phase (task), or none at all (none).
NETWORK_POLICIES = ("task", "none")
# What a sidecar that is down at grading makes the outcome: invalid (sidecar_unhealthy), or graded anyway (grade).
SIDECAR_DOWN_POLICIES = ("invalid", "grade")
# What the grading commands around the tests take at most besides them: killing processes, checking the workdir,
# making the directories, uploading the tests, reading the output and, for a separate verifier, the artifacts.
_GRADING_ALLOWANCE_S = 120.0
# Where a main-only task's init (its compose command) writes its output in the sandbox.
INIT_LOG = "/tmp/platoon-init/init.log"
_REPORTED: set[tuple[str, str]] = set()  # hold-out reports logged by this process: (dataset, eval digest)


class HarborEnvironmentService:
    """Serve Harbor tasks, one sandbox per session, graded by each task's ``tests/test.sh``.

    Task ids are the dataset's (:class:`~platoon.envs.harbor.dataset.HarborDataset`): registry names, directory
    names, or relative paths of deeper task directories; ``get_task`` also accepts an absolute task directory path.
    A session's sandbox runs the task's ``environment.docker_image`` (a registry reference gets ``docker://``; a
    ``.sif``, ``.sqsh`` or absolute path is used as written), else ``default_image``.  A task whose environment is
    a compose stack (sidecar services) runs as one, ``main`` and its sidecars each in a sandbox of their own, on a
    provider that runs stacks (:mod:`platoon.envs.harbor.sidecars`), unless ``stacks`` is ``refuse``; elsewhere it is
    refused when it loads, never started as its main container alone and scored.  So is a task that needs GPUs on a
    provider without them, when its session opens.  A main-only compose file (:mod:`platoon.envs.harbor.compose`)
    gives the sandbox its environment, working directory, resources and binds, and its command (main's init), which
    the session starts once the sandbox runs, waits for main's healthcheck, and keeps alive through grading:
    ``kill_processes`` spares it and what it started.  ``network_policy`` (``task`` or ``none``) gives every sandbox
    and stack the network its task asks for, or none, and ``on_sidecar_down`` (``invalid`` or ``grade``) decides what a
    sidecar that is down at grading makes the outcome.
    The dataset is indexed lazily, so opening a session for a task that already
    carries its directory (``Task.misc["harbor"]["task_dir"]``) never scans it,
    and the sandbox provider is built when the first session opens.

    A fetched dataset (:mod:`platoon.envs.harbor.fetch`) is named by ``dataset`` (``hf://org/name@revision``, a
    Harbor Hub package ``harbor://org/name@ref``, a dataset of Harbor's legacy registry ``harbor://name@version``
    with an optional ``registry``, ``git+<url>@<commit>#<path>`` or a directory; :mod:`platoon.envs.harbor.sources`),
    by the ``adapter`` that generates it, or by its ``lock`` alone (``tasks``, ``select``, ``overlays``, ``refuse``
    and ``index`` choose and edit what is fetched), with the Harbor cache (``harbor`` in the site profile's cache)
    holding the files. Its ids
    come from the lock, and the trainer, which lists the tasks, fetches a pinned ``dataset`` that no lock lists yet;
    rollout workers only read the lock. A task of a fetched dataset that builds from a Dockerfile runs the image map's
    image for its build key, and its deferred Dockerfile steps are applied when its session opens; one the map lacks
    is refused, never run on the default image.  A separate verifier gets its image the same way: the map's image of
    ``tests/Dockerfile``, else the task's own image with its deferred steps.  The map is the dataset's default image
    map, ``images/<provider>.json`` in the Harbor cache, when that file exists (what ``platoon prepare`` builds;
    :mod:`platoon.envs.harbor.images`); a dataset directory has none, so its tasks run their ``docker_image`` or
    ``default_image``.

    With ``holdout`` (:mod:`platoon.envs.harbor.holdout`), the service lists two splits, ``eval`` (whole groups of
    the selected tasks, held out, tasks that share an input file always in one group) and ``train`` (the rest),
    computed when the tasks are listed; the trainer logs the hold-out's report once.  ``holdout`` replaces
    ``splits``.

    Grading reads the verifier's reward files under the ``reward`` contract (:mod:`platoon.envs.harbor.grading`;
    default: the ``reward`` key trains), after the opt-in ``verifier_classifiers``.  A task with a separate
    verifier is graded in a sandbox of its own (:mod:`platoon.envs.harbor.session`), from artifacts of at most
    ``max_artifact_bytes`` each (1 GiB by default; a larger one is left out).  ``agent_timeout_multiplier``
    scales each task's ``[agent].timeout_sec``, which harness rollouts cap the harness with.  ``agent_user``
    (``refuse``, ``ignore`` or ``honor``) decides what a task naming a user to run as gets, and ``verifier_user``
    decides for ``[verifier].user`` instead when it is set: ``honor`` runs each phase as Harbor does, the agents as
    ``[agent].user``, else the image's ``USER``, the tests as ``[verifier].user``, else the image's ``USER``, in
    sandboxes that give root and other users, with the image's file owners (:mod:`platoon.envs.harbor.views`).
    Separate verifiers' sandboxes start with ``verifier_provider_options`` laid over ``provider_options``, such as
    Singularity's ``fakeroot`` for verifiers that run as root.

    Agents sharing a sandbox take turns on it: its tools run one at a time, the
    tools of a stack's MCP servers too (:class:`~platoon.envs.harbor.session.TaskTools`),
    except those named in ``concurrent_tools`` (``bash`` and ``view`` may
    overlap other calls; ``write`` and ``edit`` read, change and write a file,
    so overlapping edits of one file can lose one). ``queue_timeout_s``
    (default 240 s, None: no limit) is how long a call or a grade waits for its
    turn before it is refused unrun; ``close_timeout_s`` bounds how long the
    last close waits for the calls in flight before it stops the sandbox.

    A session's sandbox also gets what the rollout's harness needs
    (``SessionNeeds``): the gateway's upstream, which a sandbox without a
    network reaches through a route; the harness's toolchain, staged from
    the toolchain root (``toolchains`` in the site profile's cache, else
    ``~/.cache/platoon/toolchains``) and mounted read-only; and the harness's
    memory, added to the task's limit.

    A rollout takes longer than its harness: its sandbox starts first, and grading starts a separate verifier's
    sandbox and runs the tests for up to the verifier's timeout.  :meth:`rollout_overhead_s` says how much longer at
    most, with ``sandbox_start_budget_s`` (default 600 s) for each sandbox start, and deadlines on whole rollouts (a
    rollout worker's alarm, an evaluation slot's) add it, so a slow verifier never kills the worker that grades.
    :meth:`agent_timeout_s` gives a task's own time limit for its agent, which those deadlines count when the rollout
    sets no ``timeout``.
    """

    # Options a release removed, with what replaces them (platoon.deprecations): the site profile's cache says where
    # the Harbor cache, the image maps and the toolchains live (platoon.site).  A config whose entry's service_kwargs
    # give one does not load (platoon.envs.service.service_option_hints, service_option_problems).
    UNKNOWN_KEY_HINTS: ClassVar[dict[str, str]] = {
        "cache": removed_hint("0.3.0", "the site profile's cache, whose harbor directory is the Harbor cache"),
        "image_map": removed_hint(
            "0.3.0", "the default image map that platoon prepare builds, images/<provider>.json in the Harbor cache"
        ),
        "toolchain_root": removed_hint(
            "0.3.0", "the site profile's cache, whose toolchains directory holds the harness toolchains"
        ),
    }
    # The options that name files platoon reads: a config file may give each relative to itself (RFC 0018, section 5;
    # :class:`~platoon.envs.service.PathOption`).  A split manifest named relative to a ``dataset_dir`` stays relative
    # to it, the lock a fetch writes need not exist yet, and an adapter's ``repo`` may be a local repository.
    path_options: ClassVar[dict[str, PathOption]] = {
        "dataset": PathOption(),
        "registry": PathOption(),
        "adapter.repo": PathOption(),
        "dataset_dir": PathOption(),
        "task_dirs": PathOption(),
        "splits": PathOption(relative_to="dataset_dir"),
        "lock": PathOption(written=True),
        "tasks": PathOption(),
        "overlays": PathOption(),
        "mounts[].source": PathOption(),
    }

    def __init__(
        self,
        dataset_dir: str | Path | None = None,
        *,
        task_dirs: Sequence[str | Path] | None = None,
        splits: SplitManifest | None = None,
        provider: str | SandboxProvider = "docker",
        provider_options: Mapping[str, Any] | None = None,
        default_image: str | None = None,
        workdir: str = "/app",
        env: Mapping[str, str] | None = None,
        mounts: Sequence[Mount | Mapping[str, Any]] = (),
        tool_timeout_s: float = DEFAULT_COMMAND_TIMEOUT_S,
        verifier_timeout_multiplier: float = 1.0,
        concurrent_tools: Sequence[str] = (),
        queue_timeout_s: float | None = DEFAULT_QUEUE_TIMEOUT_S,
        close_timeout_s: float = DEFAULT_CLOSE_TIMEOUT_S,
        reward: RewardContract | Mapping[str, Any] | Sequence[str] | str | None = None,
        verifier_classifiers: Any = None,
        agent_timeout_multiplier: float = 1.0,
        agent_user: AgentUserPolicy = "refuse",
        verifier_user: VerifierUserPolicy | None = None,
        verifier_provider_options: Mapping[str, Any] | None = None,
        name: str = "harbor",
        dataset: str | None = None,
        lock: str | Path | None = None,
        tasks: str | Path | Sequence[str | Path] = (),
        select: Mapping[str, Any] | None = None,
        overlays: str | Path | Sequence[str | Path] = (),
        refuse: Mapping[str, Any] | None = None,
        index: Sequence[str | Mapping[str, Any]] = (),
        adapter: Mapping[str, Any] | None = None,
        registry: str | Path | None = None,
        max_artifact_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
        sandbox_start_budget_s: float = DEFAULT_SANDBOX_START_BUDGET_S,
        holdout: HoldoutRule | Mapping[str, Any] | None = None,
        stacks: str = "auto",
        network_policy: str = "task",
        on_sidecar_down: str = "invalid",
    ) -> None:
        concurrent = frozenset([concurrent_tools] if isinstance(concurrent_tools, str) else concurrent_tools)
        if unknown := sorted(concurrent - {spec.name for spec in TOOL_SPECS}):
            raise ValueError(f"concurrent_tools names unknown tools {unknown}; the tools are bash, view, write, edit")
        if not close_timeout_s > 0:
            raise ValueError("close_timeout_s must be positive")
        if isinstance(max_artifact_bytes, bool) or not isinstance(max_artifact_bytes, int) or max_artifact_bytes < 1:
            raise ValueError(f"max_artifact_bytes must be a positive number of bytes, got {max_artifact_bytes!r}")
        self.max_artifact_bytes = max_artifact_bytes
        if (
            isinstance(sandbox_start_budget_s, bool)
            or not isinstance(sandbox_start_budget_s, int | float)
            or not (math.isfinite(sandbox_start_budget_s) and sandbox_start_budget_s >= 0)
        ):
            raise ValueError(f"sandbox_start_budget_s must be a number of seconds, got {sandbox_start_budget_s!r}")
        self.sandbox_start_budget_s = float(sandbox_start_budget_s)
        self._task_limits: dict[str, tuple[float | None, float]] = {}  # id -> (agent_timeout_s, rollout_overhead_s)
        # Checked here, so a bad value fails the run at its start rather than every tool call or grade.
        for option, value in (
            ("agent_timeout_multiplier", agent_timeout_multiplier),
            ("verifier_timeout_multiplier", verifier_timeout_multiplier),
            ("tool_timeout_s", tool_timeout_s),
        ):
            number = isinstance(value, int | float) and not isinstance(value, bool)
            if not (number and math.isfinite(value) and value > 0):
                raise ValueError(f"{option} must be a positive number, got {value!r}")
        for option, value, allowed in (
            ("stacks", stacks, STACK_POLICIES),
            ("network_policy", network_policy, NETWORK_POLICIES),
            ("on_sidecar_down", on_sidecar_down, SIDECAR_DOWN_POLICIES),
        ):
            if value not in allowed:
                raise ValueError(f"{option} must be one of {allowed}, got {value!r}")
        self.stacks = stacks
        self.network_policy = network_policy
        self.on_sidecar_down = on_sidecar_down
        if agent_user not in AGENT_USER_POLICIES:
            raise ValueError(f"agent_user must be one of {AGENT_USER_POLICIES}, got {agent_user!r}")
        if verifier_user is not None and verifier_user not in VERIFIER_USER_POLICIES:
            raise ValueError(f"verifier_user must be one of {VERIFIER_USER_POLICIES}, got {verifier_user!r}")
        if verifier_provider_options is not None and not isinstance(verifier_provider_options, Mapping):
            raise ValueError(f"verifier_provider_options must be a mapping, got {verifier_provider_options!r}")
        if verifier_provider_options is not None and not isinstance(provider, str):
            raise ValueError(
                "verifier_provider_options need a provider named by its name (singularity, docker, module:Class): a "
                "provider object starts every sandbox with its own options"
            )
        self.reward = RewardContract.coerce(reward)
        self.holdout = HoldoutRule.coerce(holdout) if holdout is not None else None
        if self.holdout is not None and splits is not None:
            raise ValueError("Pass holdout or splits, not both: a hold-out lists the splits train and eval itself")
        self._holdout_split: HoldoutSplit | None = None
        self.verifier_classifiers = classifier_list(verifier_classifiers)
        self.agent_timeout_multiplier = float(agent_timeout_multiplier)
        self.agent_user: AgentUserPolicy = agent_user
        self.verifier_user: VerifierUserPolicy | None = verifier_user
        self.concurrent_tools = concurrent
        self.queue_timeout_s = check_queue_timeout(queue_timeout_s)
        self.close_timeout_s = float(close_timeout_s)
        self.name = name
        self.default_image = default_image
        self.workdir = workdir
        self.env = dict(env or {})
        self.mounts = [mount if isinstance(mount, Mount) else Mount(**mount) for mount in mounts]
        self.tool_timeout_s = float(tool_timeout_s)
        self.verifier_timeout_multiplier = float(verifier_timeout_multiplier)
        self._dataset_dir = dataset_dir
        self._task_dirs = task_dirs
        self._splits = splits
        self._provider = provider
        self._provider_options = dict(provider_options or {})
        # The options of separate verifiers' sandboxes; None: the agents' provider starts them too.
        self._verifier_options = (
            None if verifier_provider_options is None else {**self._provider_options, **verifier_provider_options}
        )
        self._locked: LockedTasks | None = None
        self._warned_of_fetch = False
        self._warned_of_copies = False
        if dataset is not None or adapter is not None or lock is not None:
            if dataset_dir is not None or task_dirs is not None:
                raise ValueError("Pass dataset, adapter or lock, or dataset_dir or task_dirs, not both")
            manifest = _load_splits(splits, None) if splits is not None else None
            self._locked = LockedTasks(
                dataset=dataset,
                adapter=adapter,
                registry=registry,
                lock=lock,
                tasks=tasks,
                select=select,
                overlays=overlays,
                refuse=refuse,
                index=index,
                splits=manifest,
            )
        # A fetched dataset's default image map; a dataset directory runs its tasks' own images.
        self._images = TaskImages(self._locked, self._provider_name())

    @property
    def locked(self) -> LockedTasks | None:
        """The fetched dataset this service serves (its fetch settings, lock and cache), or None for a directory."""

        return self._locked

    @property
    def provider_name(self) -> str:
        """The sandbox provider's name, as the entry names it (``singularity``, ``docker``, ``module:Class``)."""

        return self._provider_name()

    @property
    def task_images(self) -> TaskImages:
        """The images its sessions run: a fetched dataset's default image map, and each task's build plan."""

        return self._images

    @cached_property
    def provider(self) -> SandboxProvider:
        """Built on first use, so listing tasks needs no sandbox runtime."""

        if isinstance(self._provider, str):
            return get_provider(self._provider, **self._provider_options)
        return self._provider

    @cached_property
    def verifier_provider(self) -> SandboxProvider:
        """The provider of separate verifiers' sandboxes: the agents' provider built with ``verifier_provider_options``
        laid over ``provider_options``, else the agents' provider itself.  Built on first use."""

        if self._verifier_options is None or not isinstance(self._provider, str):
            return self.provider
        return get_provider(self._provider, **self._verifier_options)

    @property
    def _phase_options(self) -> dict[str, dict[str, Any]]:
        """The provider options of the agents' sandboxes and of separate verifiers' (the agents' when no
        ``verifier_provider_options`` are given)."""

        verifier = self._provider_options if self._verifier_options is None else self._verifier_options
        return {"agents": self._provider_options, "verifier": verifier}

    @cached_property
    def dataset(self) -> HarborDataset:
        return HarborDataset(
            self._dataset_dir,
            task_dirs=self._task_dirs,
            splits=self._splits,
            agent_user=self.agent_user,
            verifier_user=self.verifier_user,
            stack_refusal=self.stacks_refusal(),
        )

    async def list_tasks(self, split: str) -> list[str]:
        """The ids of ``split``; an ``env_config`` ``EnvSetupError`` when the lock cannot be fetched or read, or the
        hold-out cannot split the tasks, has no such split, or holds out no runnable task for an evaluation."""

        if (holdout := self.split_report()) is not None:
            try:
                task_ids = list(holdout.tasks(split))
            except KeyError as exc:
                raise EnvSetupError("env_config", str(exc.args[0])) from None
            if split == EVAL and not task_ids:
                message = (
                    f"holdout: no runnable task is held out for evaluation ({holdout.rule.describe()}): no group fits "
                    "its stratum's share, or every held-out task is refused; raise fraction, or group by a finer "
                    "field (the hold-out's report shows each stratum's makeup)"
                )
                groups = holdout.groups
                if groups is not None and groups.largest_members > 1:  # the largest group was joined by input files
                    message += (
                        f". Tasks that share an input file are one group, and the largest, {groups.largest} "
                        f"({groups.largest_tasks:,} tasks), joins {groups.largest_members:,} groups by "
                        f"{groups.largest_shared_files:,} file(s) such as {groups.largest_example}, which "
                        f"{groups.largest_example_tasks:,} task(s) hold: if such files are what every task holds by "
                        "design rather than data, such as a game's code, name them in holdout.ignore_inputs"
                    )
                raise EnvSetupError("env_config", message)
            return task_ids
        if self._locked is not None:
            try:
                return self._locked.names(split)
            except (FetchError, LockError) as exc:
                raise EnvSetupError("env_config", str(exc)) from exc
        return self.dataset.names(split)

    async def task_columns(self, task_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        """Each task's index fields, which a listing gives the run as dataset columns beside ``task_id``, where
        ``train.label_by`` can name them (:mod:`platoon.train.labels`): a locked task's lock row (the dataset's index
        files, and the ``{metadata}`` fields its ``index`` entries name), a dataset directory's ``task-index.json``
        row without the ``task_path`` that locates it, which a fetch leaves out too.  An ``env_config``
        ``EnvSetupError`` when the lock cannot be read."""

        try:
            if self._locked is not None:
                dataset = self._locked.dataset(fetch_allowed=True)
                return {task_id: dict(dataset.row(task_id).index) for task_id in task_ids}
            return {
                task_id: {name: value for name, value in self.dataset.get(task_id).index.items() if name != "task_path"}
                for task_id in task_ids
            }
        except (FetchError, LockError, KeyError, ValueError) as exc:
            raise EnvSetupError("env_config", str(exc)) from exc

    async def check_tasks(self, split: str, task_ids: Sequence[str]) -> None:
        """Refuse, with an ``env_config`` ``EnvSetupError``, the tasks of ``split`` a run keeps (``task_ids``, what
        ``exclude_tasks`` left of the listing) that the sandbox provider would refuse to start, or that could not run
        as their ``[agent].user`` and ``[verifier].user`` ask (:meth:`_check_users`), or whose sandboxes honor users
        on an image that may keep its files from other users (:meth:`_check_closed_images`), so that the run stops
        when it lists its tasks rather than lose every rollout of theirs, or that are multi-container stacks no rollout
        could start (:meth:`_check_stacks`, :meth:`_check_privileges`, :meth:`_check_stack_specs`); and, with
        :class:`EnvNotPrepared`, tasks of a fetched dataset whose images the image map lacks (:meth:`_check_images`),
        whose sidecars run registry images it has not pinned (:meth:`_check_pins`), or whose sandboxes honor users and
        lack their images' file owners (:meth:`_check_owners`)."""

        if task_ids and (refusal := rootfs_layout_refusal(self._provider, self._provider_options)):
            raise EnvSetupError("env_config", f"the {self._provider_name()} provider would start no sandbox: {refusal}")
        if task_ids and self._verifier_options is not None:
            refusal = rootfs_layout_refusal(self._provider, self._verifier_options)
            if refusal and any(self._separate(name) for name in task_ids):
                raise EnvSetupError(
                    "env_config",
                    f"the {self._provider_name()} provider would start no separate verifier's sandbox with the "
                    f"verifier_provider_options: {refusal}",
                )
        self._check_stacks(task_ids, split)
        self._check_privileges(task_ids, split)
        self._check_networks(task_ids, split)
        self._check_users(task_ids, split)
        self._check_images(task_ids, split)
        self._check_pins(task_ids, split)
        self._check_owners(task_ids, split)
        self._check_stack_specs(task_ids, split)
        self._check_closed_images(task_ids, split)

    def _check_images(self, names: Sequence[str], split: str) -> None:
        """Refuse tasks of a fetched dataset that build their image, their separate verifier's or a stack sidecar's
        from a Dockerfile whose image the image map lacks (or whose image file is gone, or is not the size its build
        recorded): a run never builds an image, and every rollout of theirs would fail when its session opens.
        ``platoon prepare`` builds them.  Tasks of a dataset directory plan their images when a session opens, and are
        not checked; nor are tasks that run a prebuilt image.  Each image is looked at once, however many tasks share
        it."""

        path = self._images.path
        if self._locked is None or path is None or not names:
            return
        try:
            image_map = self._images.load()
        except ImageMapError as exc:
            raise EnvSetupError("env_config", str(exc)) from exc  # the map's file names itself
        dataset = self._locked.dataset(fetch_allowed=False)
        present: dict[str, bool] = {}  # build key -> whether the map has its image
        missing: dict[str, list[str]] = {}  # build key -> the tasks that need its image
        for name in names:
            row = dataset.lock.get(name)
            for plan in (row.image, row.verifier_image or {}, *row.service_builds()):
                key = plan.get("key")
                if not key:
                    continue
                if key not in present:
                    present[key] = has_image(image_map, key)
                if not present[key]:
                    missing.setdefault(key, []).append(name)
        if not missing:
            return
        tasks = {name for needing in missing.values() for name in needing}
        key, needing = max(missing.items(), key=lambda item: len(item[1]))
        if path.exists():
            where = f"the image map {path} lacks"
        else:
            where = f"no image map holds (the image map {path} does not exist)"
        if image_map is not None and any(image_map.get(each) is not None for each in missing):
            where += ", or names by a file that is gone or truncated"  # an entry whose file is not whole
        more = f" and {len(missing) - 1} more" if len(missing) > 1 else ""
        raise EnvNotPrepared(
            f"{len(tasks):,} of the {len(names):,} tasks of split {split!r} build their images from Dockerfiles, which "
            f"{where}: {len(missing)} image(s), such as {key} for {len(needing):,} task(s) ({needing[0]}){more}; a run "
            "never builds them"
        )

    def _check_pins(self, names: Sequence[str], split: str) -> None:
        """Refuse tasks of a fetched dataset whose sidecars run a registry image that the image map has not pinned:
        its digest and configuration (its ``ENTRYPOINT``, ``CMD``, ``USER``, ...), which ``platoon prepare`` records,
        since platoon gives every runtime a sidecar's whole command, and a pod's runtimes apply none of an image's
        configuration by themselves.  The ``process`` provider runs no image, so its stacks run what their compose
        files give, as a dataset directory's do."""

        if self._locked is None or self._images.path is None or self._provider_name() == "process" or not names:
            return
        try:
            image_map = self._images.load()
        except ImageMapError as exc:
            raise EnvSetupError("env_config", str(exc)) from exc
        dataset = self._locked.dataset(fetch_allowed=False)
        unpinned: dict[str, list[str]] = {}  # reference -> the tasks whose sidecars run it
        for name in names:
            for reference in dataset.lock.get(name).service_images():
                if image_map is None or image_map.get(pulled_key(reference)) is None:
                    unpinned.setdefault(reference, []).append(name)
        if not unpinned:
            return
        tasks = {name for needing in unpinned.values() for name in needing}
        reference, needing = max(unpinned.items(), key=lambda item: len(item[1]))
        more = f" and {len(unpinned) - 1} more" if len(unpinned) > 1 else ""
        raise EnvNotPrepared(
            f"{len(tasks):,} of the {len(names):,} tasks of split {split!r} have sidecars that run registry images "
            f"the image map {self._images.path} has not pinned by digest with their configuration: {len(unpinned)} "
            f"image(s), such as {reference} for {len(needing):,} task(s) ({needing[0]}){more}"
        )

    def _provider_name(self) -> str:
        return self._provider if isinstance(self._provider, str) else getattr(self._provider, "name", "sandbox")

    def stacks_refusal(self) -> str | None:
        """Why no task of the service may run as a multi-container stack: ``stacks: refuse``, or a provider that runs
        none (:func:`platoon.sandbox.stack_refusal`, read without building it); None otherwise."""

        if self.stacks == "refuse":
            return "the environment sets stacks: refuse"
        return stack_refusal(self._provider, self._provider_options)

    def _check_stacks(self, names: Sequence[str], split: str) -> None:
        """Refuse the tasks of a fetched dataset that are multi-container stacks (a fetch locks them, unless its
        ``refuse`` sets ``sidecars: true``) where no stack may run (:meth:`stacks_refusal`): every rollout of theirs
        would fail as it opens.  A dataset directory leaves such tasks out when it lists them, with a warning."""

        if self._locked is None or not names or (refusal := self.stacks_refusal()) is None:
            return
        lock = self._locked.dataset(fetch_allowed=False).lock
        stacks = [name for name in names if ((lock.get(name).compose or {}).get("kind")) == "sidecars"]
        if stacks:
            shown = ", ".join(stacks[:3]) + (f" and {len(stacks) - 3} more" if len(stacks) > 3 else "")
            raise EnvSetupError(
                "env_config",
                f"{len(stacks)} of the {len(names)} tasks of split {split!r} are multi-container stacks (sidecar "
                f"services: {shown}), and {refusal}; run them on a provider that runs stacks, or leave them out "
                "(exclude_tasks, or a fetch whose refuse sets sidecars: true, which locks them refused)",
            )

    def _check_privileges(self, names: Sequence[str], split: str) -> None:
        """Refuse the stacks of a fetched dataset whose services ask for privileges (a fetch with ``refuse:
        {privileges: false}`` plans them) that the provider would not give (:func:`platoon.sandbox.privilege_refusal`):
        every rollout of theirs would be refused as it opens."""

        if self._locked is None or not names or (refusal := privilege_refusal(self._provider)) is None:
            return
        lock = self._locked.dataset(fetch_allowed=False).lock
        asking = {
            name: privileges for name in names if (privileges := (lock.get(name).compose or {}).get("privileges"))
        }
        if asking:
            name, privileges = next(iter(asking.items()))
            raise EnvSetupError(
                "env_config",
                f"{len(asking)} of the {len(names)} tasks of split {split!r} are stacks whose services ask for "
                f"privileges (such as {name}: {', '.join(privileges)}), and {refusal}; run them on the docker "
                "provider, or leave them out (exclude_tasks)",
            )

    def _check_stack_specs(self, names: Sequence[str], split: str) -> None:
        """Refuse the multi-container stacks among the tasks that no rollout could start, as a session would find
        when it opens (:func:`platoon.envs.harbor.sidecars.stack_spec`, which starts nothing): a sidecar whose image's
        command platoon does not know (a dataset directory's registry image that its compose file gives none), a
        dependency on the health of a service without a healthcheck, an unset host variable.  Where no stack runs,
        :meth:`_check_stacks` refused them already, and the images a fetched dataset lacks :meth:`_check_images`."""

        if not names or self.stacks_refusal() is not None:
            return
        unrunnable: dict[str, str] = {}
        for name in names:
            try:
                harbor_task = self._peek(name)
            except (KeyError, ValueError):  # a refused or unknown task fails on its own when it is opened
                continue
            if not harbor_task.is_stack:
                continue
            try:
                self._stack_plan(harbor_task, self._sandbox_spec(harbor_task, "check"), setup=_no_setup)
            except (KeyError, ValueError) as exc:
                unrunnable[name] = str(exc)
        if unrunnable:
            shown = ", ".join(list(unrunnable)[:3]) + (
                f" and {len(unrunnable) - 3} more" if len(unrunnable) > 3 else ""
            )
            raise EnvSetupError(
                "env_config",
                f"{len(unrunnable)} of the {len(names)} tasks of split {split!r} are multi-container stacks that no "
                f"rollout can start ({shown}): {next(iter(unrunnable.values()))}; leave them out (exclude_tasks)",
            )

    def _check_networks(self, names: Sequence[str], split: str) -> None:
        """Refuse tasks that run without a network (``network_mode = "no-network"``, for their agents or their
        separate verifier) on a provider that would refuse every such sandbox, such as ``singularity`` without
        ``allow_network_none``, with the options of the sandbox concerned."""

        refusals = self._network_refusals()
        if not any(refusals.values()):
            return
        offline: dict[str, str] = {}  # task -> the refusal of a sandbox of its that runs without a network
        for name in names:
            try:
                harbor_task = self._peek(name)
            except (KeyError, ValueError):  # a refused or unknown task fails on its own when it is opened
                continue
            if (refusal := _network_refusal(harbor_task, refusals, self.network_policy)) is not None:
                offline[name] = refusal
        if offline:
            shown = ", ".join(list(offline)[:3]) + (f" and {len(offline) - 3} more" if len(offline) > 3 else "")
            if self.network_policy == "none":  # not the tasks' doing: the policy's
                raise EnvSetupError(
                    "env_config",
                    f"{len(offline)} of the {len(names)} tasks of split {split!r} run without a network, as the "
                    f"environment's network_policy: none makes every sandbox ({shown}), and "
                    f"{next(iter(offline.values()))}; set it where the site allows it, or set network_policy: task",
                )
            raise EnvSetupError(
                "env_config",
                f"{len(offline)} of the {len(names)} tasks of split {split!r} run without a network ({shown}), and "
                f"{next(iter(offline.values()))}; set it where the site allows it, or leave those tasks out "
                "(exclude_tasks)",
            )

    def _network_refusals(self) -> dict[str, str | None]:
        """Why the agents' sandboxes, and separate verifiers', would be refused without a network (by phase)."""

        return {phase: network_none_refusal(self._provider, options) for phase, options in self._phase_options.items()}

    def _check_users(self, names: Sequence[str], split: str) -> None:
        """With ``agent_user`` or ``verifier_user`` set to ``honor``, refuse the tasks that could not run as their
        ``[agent].user`` and ``[verifier].user`` ask: those :meth:`HarborTask.load` refuses (a stack whose users are
        honored, a root verifier of a task graded in its agents' sandbox whose agents are not), and those with a sandbox
        that honors users (:meth:`HarborTask.honors_users_in`) where its provider's options cannot give root and other
        users (:func:`platoon.sandbox.root_refusal`): every rollout of theirs would fail, or score 0 when their tests
        cannot run the agent's code as another user."""

        if "honor" not in (self.agent_user, self.verifier_user) or not names:
            return
        self._warn_of_fetch_refusals()
        cannot = {phase: root_refusal(self._provider, options) for phase, options in self._phase_options.items()}
        refused: dict[str, str] = {}  # task -> why
        honoring: list[str] = []
        for name in names:
            try:
                harbor_task = self._peek(name)
            except HarborTaskError as exc:  # its users, which a fetch lets through for the service to decide about
                refused[name] = str(exc)
                continue
            except (KeyError, ValueError):  # unknown, or not readable: it fails on its own when it is opened
                continue
            if (why := _users_refusal(harbor_task, cannot)) is not None:
                refused[name] = why
            if harbor_task.honored_users:
                honoring.append(name)
        self._warn_of_dropped_owners(honoring)
        if refused:
            shown = ", ".join(list(refused)[:3]) + (f" and {len(refused) - 3} more" if len(refused) > 3 else "")
            name, why = next(iter(refused.items()))
            fix = (
                "set agent_user or verifier_user to ignore, to run them as the sandbox's own user, or leave those "
                "tasks out (exclude_tasks)"
            )
            if any(reason.startswith("its ") for reason in refused.values()):  # a sandbox that cannot give root
                fix = (
                    f"Give those sandboxes root (provider_options, verifier_provider_options for the verifiers'), {fix}"
                )
            raise EnvSetupError(
                "env_config",
                f"{len(refused)} of the {len(names)} tasks of split {split!r} cannot run as their [agent].user and "
                f"[verifier].user ask ({shown}); {name}: {why}. {fix[0].upper()}{fix[1:]}",
            )

    def _check_owners(self, names: Sequence[str], split: str) -> None:
        """Refuse the tasks whose sandboxes honor users on a provider that loses the files' owners (hpc-sandbox), and
        that could not give their images' files their owners: :class:`EnvNotPrepared` for an image whose owner manifest
        the image map lacks (``platoon prepare`` records those of pulled images, and names the build of the others),
        an ``env_config`` ``EnvSetupError`` for one no image map can hold (a dataset directory's) or that the installed
        hpc-sandbox cannot apply (before 0.5)."""

        if "honor" not in (self.agent_user, self.verifier_user) or not names or not loses_file_owners(self._provider):
            return
        missing: dict[str, str] = {}
        refused: dict[str, str] = {}
        for name in names:
            try:
                harbor_task = self._peek(name)
            except (KeyError, ValueError):  # refused or unknown: it fails on its own when it is opened
                continue
            for verifier in (False, True):
                try:
                    self._file_owners(harbor_task, verifier=verifier)
                except EnvNotPrepared as exc:
                    missing.setdefault(name, str(exc))
                except ValueError as exc:
                    refused.setdefault(name, str(exc))
        if problems := refused or missing:
            shown = ", ".join(list(problems)[:3]) + (f" and {len(problems) - 3} more" if len(problems) > 3 else "")
            message = (
                f"{len(problems)} of the {len(names)} tasks of split {split!r} honor users in sandboxes that lose the "
                f"owners of their images' files ({shown}): {next(iter(problems.values()))}"
            )
            raise EnvSetupError("env_config", message) if refused else EnvNotPrepared(message)

    def _warn_of_dropped_owners(self, names: Sequence[str]) -> None:
        """Warn, once, of the tasks that honor users (``names``) whose build plans defer a ``COPY``, or a ``WORKDIR``
        and a ``USER``, in a lock that a platoon before 0.5 wrote, which drops a deferred copy's ``--chown`` and where
        a ``USER`` falls among the deferred steps: their copies, and the directories that a ``WORKDIR`` after the
        ``USER`` makes, stay root's."""

        if self._locked is None or self._warned_of_copies:
            return
        self._warned_of_copies = True
        lock = self._locked.dataset(fetch_allowed=False).lock
        if lock.compose_rule:  # written by platoon 0.5 or later, whose plans keep --chown and their USER steps
            return
        copying = []
        for name in names:
            row = lock.get(name)
            for plan in (row.image or {}, row.verifier_image or {}):
                kinds = {step[0] for step in plan.get("deferred") or ()}
                if "COPY" in kinds or (plan.get("suffix_user") and "WORKDIR" in kinds):
                    copying.append(name)
                    break
        if copying:
            shown = ", ".join(copying[:3]) + (f" and {len(copying) - 3} more" if len(copying) > 3 else "")
            logger.warning(
                "%d task(s) of %s that honor users defer a COPY, or a WORKDIR and a USER, (%s) in a lock that a "
                "platoon before 0.5 wrote, which keeps no COPY --chown and no USER among the deferred steps: their "
                "copies, and the directories that a WORKDIR after the USER makes, stay root's. Fetch the dataset "
                "again to keep their owners (platoon harbor fetch CONFIG --repair)",
                len(copying),
                lock.dataset,
                shown,
            )

    def _warn_of_fetch_refusals(self) -> None:
        """Warn, once, of the tasks whose users ``honor`` would honor that the fetch of the service's dataset left out:
        its ``refuse.agent_user``, on by default, refuses every task that names a user, so ``honor`` never sees them."""

        if self._locked is None or self._warned_of_fetch:
            return
        self._warned_of_fetch = True
        lock = self._locked.dataset(fetch_allowed=False).lock
        phases = ("agent_user", "verifier_user") if self.agent_user == "honor" else ("verifier_user",)
        left_out = [
            task.name
            for task in lock.tasks
            if (task.refused or "").startswith("agent_user:") and any(task.flags.get(phase) for phase in phases)
        ]
        if left_out:
            shown = ", ".join(left_out[:3]) + (f" and {len(left_out) - 3} more" if len(left_out) > 3 else "")
            policy = "agent_user" if self.agent_user == "honor" else "verifier_user"
            logger.warning(
                "%s: honor runs none of the %d task(s) of %s that name users to run as (%s): the fetch left them out, "
                "since its refuse.agent_user, on by default, refuses every task that names a user. Set refuse: "
                "{agent_user: false} to fetch them",
                policy,
                len(left_out),
                lock.dataset,
                shown,
            )

    def closed_images(self, names: Sequence[str] | None = None) -> dict[str, list[str]]:
        """The images of the image map that tasks of a fetched dataset (``names``, else every runnable task of its
        lock) run in sandboxes that honor users (:meth:`HarborTask.honors_users_in`), and that may keep their files from
        the users other than root, since the ``oci`` builder of platoon before 0.3.0 built them
        (:func:`~platoon.envs.harbor.images.withholds_modes`): by build key, with the tasks that run each.  Empty
        without ``honor``, and for images the map lacks, which :meth:`_check_images` reports."""

        if "honor" not in (self.agent_user, self.verifier_user) or self._locked is None:
            return {}
        try:
            image_map = self._images.load()
        except ImageMapError:
            return {}
        if image_map is None:
            return {}
        lock = self._locked.dataset(fetch_allowed=False).lock
        closed: dict[str, list[str]] = {}
        for name in names if names is not None else [task.name for task in lock.runnable()]:
            try:
                harbor_task = self._peek(name)
            except (KeyError, ValueError):  # refused or unknown: reported, or failing, on its own
                continue
            row = lock.get(name)
            plans = [row.image] if harbor_task.honors_users_in("agents") else []
            environment = harbor_task.verifier_environment
            if harbor_task.honors_users_in("verifier") and environment is not None:
                plans.append({"task": row.image, "tests": row.verifier_image}.get(environment.image_source) or {})
            for plan in plans:
                entry = image_map.get(key) if (key := plan.get("key")) else None
                if entry is not None and withholds_modes(entry):
                    closed.setdefault(key, []).append(name)
        return closed

    def owner_gaps(self, names: Sequence[str] | None = None) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        """Of the images that tasks of a fetched dataset (``names``, else every runnable task of its lock) run in
        sandboxes that honor users, on a provider that loses the files' owners, those whose owners and ``USER`` such a
        sandbox needs (:func:`_needs_record`) and the image map does not record: by build key, those built without an
        owner manifest (by platoon before 0.5, the def builder, or the oci builder with a squashfs-tools before 4.4),
        and by registry reference, the ``docker_image`` references that the map has no pulled entry for (``platoon
        prepare`` records them); each with the tasks that run it.  Empty where no sandbox needs a record."""

        if "honor" not in (self.agent_user, self.verifier_user) or self._locked is None:
            return {}, {}
        if not loses_file_owners(self._provider):
            return {}, {}
        lock = self._locked.dataset(fetch_allowed=False).lock
        built: dict[str, list[str]] = {}
        pulled: dict[str, list[str]] = {}
        for name in names if names is not None else [task.name for task in lock.runnable()]:
            try:
                harbor_task = self._peek(name)
                images = self._honoring_images(harbor_task)
            except (KeyError, ValueError):  # refused or unknown: reported, or failing, on its own
                continue
            for image, session, verifier in images:
                if session is not None and session.file_owners is not None:
                    continue
                if not _needs_record(harbor_task, session, verifier=verifier):
                    continue
                if session is not None and session.key:
                    built.setdefault(image, []).append(name)
                elif image.startswith("docker://"):  # a registry's, which prepare pulls; another is refused
                    pulled.setdefault(image, []).append(name)
        return built, pulled

    def _honoring_images(self, harbor_task: HarborTask) -> list[tuple[str, SessionImage | None, bool]]:
        """The images that the task's sandboxes which honor users run, its agents' and its separate verifier's, each
        named by its build key, else by its reference as a sandbox runs it, with its session image (None: a
        ``docker_image`` that the image map has no pulled entry for) and whether it is the verifier's."""

        images: list[tuple[str, SessionImage | None, bool]] = []
        if harbor_task.honors_users_in("agents"):
            session = self._session_image(harbor_task)
            images.append((_image_name(session) or self._image(harbor_task), session, False))
        environment = harbor_task.verifier_environment
        if harbor_task.honors_users_in("verifier") and environment is not None:
            session = self._verifier_image(harbor_task)
            name = _image_name(session)
            if name is None and environment.docker_image:
                name = docker_image_reference(environment.docker_image)
            images.append((name or self._image(harbor_task), session, True))
        return images

    def _check_closed_images(self, names: Sequence[str], split: str) -> None:
        """Refuse tasks that honor users on an image the ``oci`` builder of platoon before 0.3.0 built
        (:meth:`closed_images`): on a host whose umask is 027, the users other than root cannot run its programs, and
        every rollout would fail or score 0.  A run never builds images; the message names the build."""

        if not names or not (closed := self.closed_images(names)):
            return
        tasks = sorted({name for needing in closed.values() for name in needing})
        shown = ", ".join(tasks[:3]) + (f" and {len(tasks) - 3} more" if len(tasks) > 3 else "")
        keys = " ".join(f"--key {key}" for key in closed)
        raise EnvSetupError(
            "env_config",
            f"{len(tasks)} of the {len(names)} tasks of split {split!r} honor users ({shown}) on {len(closed)} "
            "image(s) that the oci builder of platoon before 0.3.0 built, which unpacked them under the host's umask: "
            "under one such as 027, their files keep out the users other than root. Build them again with "
            f"`platoon harbor images build CONFIG {keys}`, or set agent_user and verifier_user to ignore, or leave "
            "those tasks out (exclude_tasks)",
        )

    def _separate(self, name: str) -> bool:
        """Whether a task grades in a separate verifier's sandbox; False for one that cannot be read, which fails on
        its own when it is opened."""

        try:
            return self._peek(name).verifier_mode == "separate"
        except (KeyError, ValueError):
            return False

    def _peek(self, task_id: str) -> HarborTask:
        """The task, read without checking a locked task's cache entry against its digest, which a session's
        opening does (hashing its files): what checks on the trainer read."""

        if self._locked is not None:
            dataset = self._locked.dataset(fetch_allowed=False)
            row = dataset.row(task_id)
            return self._load(dataset.cache / row.path, index=row.index, compose=row.compose)
        return self._lookup(task_id)

    async def list_splits(self) -> list[str]:
        """The hold-out's splits (``train`` and ``eval``), or the split manifest's; none without either, where every
        split lists every task."""

        if self.holdout is not None:
            return list(SPLITS)
        if self._locked is not None:
            try:
                return self._locked.split_names()
            except (FetchError, LockError) as exc:
                raise EnvSetupError("env_config", str(exc)) from exc
        return self.dataset.split_names()

    def split_report(self) -> HoldoutSplit | None:
        """The hold-out's split of the selected tasks, with its report (:mod:`platoon.envs.harbor.holdout`); None
        without a hold-out.  Computed once, from the lock's tasks (fetching a pinned dataset that no lock answers
        yet, as listing does) or the dataset directory's; an ``env_config`` ``EnvSetupError`` when it cannot be."""

        if self.holdout is None:
            return None
        if self._holdout_split is None:
            try:
                rows = self._locked.holdout_rows() if self._locked is not None else self.dataset.holdout_rows()
                split = hold_out(rows, self.holdout)
            except (FetchError, LockError) as exc:
                raise EnvSetupError("env_config", str(exc)) from exc
            except HoldoutError as exc:
                raise EnvSetupError("env_config", f"holdout: {exc}") from exc
            source = self._dataset_identity()
            key = (str(sorted(source.items())), split.eval_digest)
            if key not in _REPORTED:
                _REPORTED.add(key)
                logger.info("Harbor environment %s: %s", self.name, "\n".join(split.report()))
            self._holdout_split = split
        return self._holdout_split

    def split_record(self) -> dict[str, Any] | None:
        """What a run records of its hold-out (``identity.json``, :mod:`platoon.train.identity`): the rule, the
        dataset, every held-out task (refused ones included) and the evaluation's tasks, sorted; None without a
        hold-out."""

        split = self.split_report()
        if split is None:
            return None
        return {
            "holdout": split.rule.to_json(),
            "dataset": self._dataset_identity(),
            "held_out": sorted(split.held_out),
            "eval": sorted(split.eval),
            "eval_digest": split.eval_digest,
        }

    def _dataset_identity(self) -> dict[str, Any]:
        """The tasks this service serves: a lock's dataset and resolved revision, or its directories."""

        if self._locked is not None:
            lock = self._locked.lock
            return {"dataset": lock.dataset, "resolved": lock.resolved, "inputs": lock.inputs}
        if self._dataset_dir is not None:
            return {"dataset_dir": str(Path(self._dataset_dir).resolve())}
        return {"task_dirs": [str(Path(task_dir).resolve()) for task_dir in self._task_dirs or ()]}

    async def get_task(self, task_id: str) -> Task:
        if self._locked is None:
            return self._lookup(task_id).to_task(task_id)
        try:  # a task the lock does not list, refuses, or whose cache entry does not match: a config error
            task = self._lookup(task_id).to_task(task_id)
            task.misc.setdefault(HARBOR_MISC_KEY, {}).update(self._locked.task_misc(task_id))
        except (KeyError, ValueError) as exc:
            raise EnvSetupError("env_config", str(exc)) from exc
        return task

    def agent_timeout_s(self, task_id: str) -> float | None:
        """The task's own time limit for its agent: ``[agent].timeout_sec`` times ``agent_timeout_multiplier``, as
        its sessions give it (:attr:`HarborSession.agent_timeout_s`); None when the task sets none.  Harness rollouts
        cap the harness with it, and with no ``rollout_config.timeout``, deadlines on whole rollouts count it where
        they would count that timeout (:func:`platoon.train.components.harness_time_limit_s`)."""

        return self._limits(task_id)[0]

    def rollout_overhead_s(self, task_id: str) -> float:
        """The seconds a rollout of the task may take beyond its harness's time limit: a ``sandbox_start_budget_s``
        for its sandbox and one for a separate verifier's, its healthcheck, its tests for up to the verifier's
        timeout (times ``verifier_timeout_multiplier``), the grading commands around them, its collect hooks'
        timeouts, and the close.  For a stack, its services' starts, health waits (their images' healthchecks too) and
        init jobs' runs (each bounded by a start budget) count along their dependencies' longest path, and so do three
        probes of its sidecars and the listing of its MCP servers' tools.  Deadlines on whole rollouts add it
        (:func:`platoon.train.components.rollout_overhead_s`)."""

        return self._limits(task_id)[1]

    def _limits(self, task_id: str) -> tuple[float | None, float]:
        """:meth:`agent_timeout_s` and :meth:`rollout_overhead_s` of a task, from one read of it: the trainer asks
        for both before each rollout."""

        if (known := self._task_limits.get(task_id)) is not None:
            return known
        harbor_task = self._peek(task_id)
        agent = harbor_task.agent_timeout_s
        check = harbor_task.healthcheck
        health = (
            0.0 if check is None else check.start_period_s + (check.retries + 1) * (check.timeout_s + check.interval_s)
        )
        budget = self.sandbox_start_budget_s
        if harbor_task.is_stack:
            # Along the longest path of the services' dependencies, with the healthchecks they run (compose's, else
            # their images', main's too), an init job's run (bounded by a start budget), and three probes of its
            # sidecars: the start's last, and the two of grading.
            checks = self._stack_healthchecks(harbor_task)
            if (own := checks.get(MAIN)) is not None:
                health += own.worst_case_s
            start = sidecars.start_bound_s(harbor_task, checks, start_s=budget, main_health_s=health, job_s=budget)
            start += 3 * sidecars.probe_bound_s(checks)
            served = [server for server in harbor_task.mcp_servers if harbor_task.mcp_service(server) is not None]
            start += len(served) * TASK_TOOLS_LISTING_TIMEOUT_S  # their tools are listed as the session opens
        else:
            main = harbor_task.main
            if main is not None and main.healthcheck is not None:  # main's compose healthcheck runs before it
                health += main.healthcheck.worst_case_s
            start = budget + health
        if harbor_task.verifier_mode == "separate":
            start += budget
        hooks = sum(hook.timeout_s for hook in harbor_task.collect)
        verifier = harbor_task.verifier_timeout_s * self.verifier_timeout_multiplier
        overhead = start + hooks + verifier + _GRADING_ALLOWANCE_S + self.close_timeout_s
        limits = (None if agent is None else agent * self.agent_timeout_multiplier, overhead)
        self._task_limits[task_id] = limits
        return limits

    def harbor_task(self, task: Task) -> HarborTask:
        """The Harbor task behind a platoon task: the directory in ``task.misc``, else the task id."""

        misc = task.misc.get(HARBOR_MISC_KEY, {})
        if misc.get("task_dir") is not None:
            row = self._locked.row_for(misc["task_dir"]) if self._locked is not None else None
            compose = row.compose if row is not None else _PLANNED_HERE
            return self._load(misc["task_dir"], index=misc.get("index"), compose=compose)
        if task.id is None:
            raise ValueError("A Harbor task needs an id or a task directory in misc")
        return self._lookup(task.id)

    def verdict(self, verifier_dir: Path, harbor_task: HarborTask) -> Outcome:
        """The outcome of a finished verifier's output directory (a local copy of ``/logs/verifier`` named
        ``verifier``): the first ``verifier_classifiers`` match, else the reward files under the ``reward``
        contract."""

        return classify(verifier_dir, self.verifier_classifiers) or self.reward.read(verifier_dir, harbor_task)

    async def open(self, task: Task, *, rollout_id: str, needs: SessionNeeds | None = None) -> HarborSession:
        try:
            harbor_task = self.harbor_task(task)
            cannot = {phase: root_refusal(self._provider, options) for phase, options in self._phase_options.items()}
            if (refusal := _users_refusal(harbor_task, cannot)) is not None:  # refused now, not after the agent ran
                raise EnvSetupError("env_config", f"Harbor task {harbor_task.name!r}: {refusal}")
            spec = self._sandbox_spec(harbor_task, rollout_id)
            if harbor_task.verifier_mode == "separate":
                self._sandbox_spec(harbor_task, rollout_id, verifier=True)  # refused now, not after the agent ran
        except (KeyError, ValueError) as exc:  # a bad task directory (HarborTaskError), unknown id or unset variable
            raise EnvSetupError("env_config", str(exc)) from exc
        if (refusal := _network_refusal(harbor_task, self._network_refusals(), self.network_policy)) is not None:
            policy = " (the environment's network_policy: none)" if self.network_policy == "none" else ""
            raise EnvSetupError(
                "env_config", f"Harbor task {harbor_task.name!r} runs without a network{policy}, and {refusal}"
            )
        if needs is not None:  # the harness's gateway route, toolchain and memory; never the verifier's
            spec = needs.apply(spec)
        if harbor_task.is_stack:  # its copies of the task's files are its own volumes
            return await self._open_stack(task, harbor_task, spec, rollout_id=rollout_id)
        views: list[Path] = []
        try:
            if (main := harbor_task.main) is not None:
                mounts, views = await sidecars.in_thread(
                    lambda: sidecars.bind_copies(harbor_task, main.binds),
                    undo=lambda made: _remove(made[1]) if made is not None else None,
                )
                spec = replace(spec, mounts=[*spec.mounts, *mounts])
            users, init = await self._start(spec, harbor_task)
        except BaseException:
            for view in views:
                shutil.rmtree(view, ignore_errors=True)
            raise
        lease = SandboxLease(
            users.root,
            spec,
            queue_timeout_s=self.queue_timeout_s,
            close_timeout_s=self.close_timeout_s,
            init=init,
            scratch=views,
            users=users,
        )
        return HarborSession(self, task, harbor_task, lease, rollout_id=rollout_id)

    async def _open_stack(
        self, task: Task, harbor_task: HarborTask, spec: SandboxSpec, *, rollout_id: str
    ) -> HarborSession:
        """Start the task's stack (:mod:`platoon.envs.harbor.sidecars`), ``spec`` being main's sandbox, and list the
        tools of its MCP servers that the tool plane serves (:class:`TaskTools`); on any failure everything it started
        is stopped and an ``EnvSetupError`` says why: ``sidecar_start`` for a sidecar that did not start or an MCP
        server that did not list its tools, ``env_config`` where no stack may run or a service cannot."""

        provider = self.provider
        refusal = self.stacks_refusal() or stack_refusal(provider)
        if refusal is not None:
            raise EnvSetupError(
                "env_config",
                f"Harbor task {harbor_task.name!r} is a multi-container stack (sidecar services), and {refusal}; the "
                "task is refused, never started without them",
            )
        init: str | None = None

        async def setup(sandbox: Sandbox) -> None:  # main's preparation, once its dependencies are up
            nonlocal init
            try:
                init, _ = await self._prepare(sandbox, harbor_task, spec)  # a stack honors no users
            except EnvSetupError:
                raise
            except Exception as exc:
                raise EnvSetupError("env_setup", f"{type(exc).__name__}: {exc}") from exc

        try:
            plan = self._stack_plan(harbor_task, spec, setup=setup)
        except (KeyError, ValueError) as exc:  # an image without a known command or not in the map, a variable, ...
            raise EnvSetupError("env_config", str(exc)) from exc
        assert isinstance(provider, StackProvider)  # stack_refusal checked that it has start_stack
        try:
            stack = await provider.start_stack(plan.spec)
        except EnvSetupError:  # main's preparation said why
            raise
        except Exception as exc:
            raise _start_failure(exc) from exc
        try:
            task_tools = await TaskTools.open(stack, harbor_task)
        except BaseException as exc:
            await asyncio.shield(stack.stop())
            if not isinstance(exc, Exception):
                raise
            raise EnvSetupError("sidecar_start", str(exc)) from exc
        lease = StackLease(
            stack,
            spec,
            queue_timeout_s=self.queue_timeout_s,
            close_timeout_s=self.close_timeout_s,
            init=init,
            logs=plan.logs,
            task_tools=task_tools,
        )
        return HarborSession(self, task, harbor_task, lease, rollout_id=rollout_id)

    def _stack_plan(
        self, harbor_task: HarborTask, spec: SandboxSpec, *, setup: Callable[[Sandbox], Any]
    ) -> sidecars.StackPlan:
        """The spec of the task's stack, ``spec`` being main's sandbox (:func:`sidecars.stack_spec`, which makes
        nothing); ``ValueError`` or ``KeyError`` says why no rollout could start it."""

        return sidecars.stack_spec(
            harbor_task,
            spec,
            main_image=self._session_image(harbor_task),
            service_image=self._service_image(harbor_task),
            env=self.env,
            mounts=self.mounts,
            network=self.stack_network(harbor_task, harbor_task.network),
            labels=spec.labels,
            setup=setup,
            job_timeout_s=self.sandbox_start_budget_s,
            pulled_image=self._images.for_pulled,
        )

    def _service_image(self, harbor_task: HarborTask) -> Callable[[str, Mapping[str, Any]], SessionImage]:
        """How a stack's service that builds from a context of the task finds its image (the image map's)."""

        return lambda name, build: self._images.for_service(harbor_task.task_dir, name, build, task=harbor_task.name)

    def _stack_healthchecks(self, harbor_task: HarborTask) -> dict[str, StackHealthcheck | None]:
        """The healthchecks the task's stack runs (:func:`sidecars.healthchecks`), as a session would find them."""

        try:
            main_image = self._session_image(harbor_task)
        except ValueError:  # an image the map lacks: no session opens, check_tasks says so
            main_image = None
        return sidecars.healthchecks(
            harbor_task,
            main_image=main_image,
            service_image=self._service_image(harbor_task),
            pulled_image=self._images.for_pulled,
        )

    def egress_stacks(self, names: Sequence[str]) -> list[str]:
        """The tasks among ``names`` that run as stacks with egress in a phase, the agents' or their verifier's in
        ``main`` (:meth:`stack_network`): what a pod gives them through slirp4netns."""

        found = []
        for name in names:
            try:
                harbor_task = self._peek(name)
            except (KeyError, ValueError):  # a refused or unknown task fails on its own when it is opened
                continue
            phases = (harbor_task.network, harbor_task.verifier_network)
            if harbor_task.is_stack and any(self.stack_network(harbor_task, phase) == "egress" for phase in phases):
                found.append(name)
        return found

    def stack_network(self, harbor_task: HarborTask, phase: str) -> StackNetwork:
        """A stack's network in a phase whose network is ``phase`` (``host`` or ``none``): none under
        ``network_policy: none``, or where every network of its compose file is internal."""

        plan = harbor_task.compose
        egress = plan is None or plan.egress
        return "egress" if egress and phase != "none" and self.network_policy != "none" else "none"

    async def open_verifier(self, harbor_task: HarborTask, *, rollout_id: str) -> tuple[SessionUsers, SandboxSpec]:
        """Start the separate verifier sandbox of a task (:class:`~platoon.envs.harbor.task.VerifierEnvironment`),
        ready for the artifacts and tests: who runs what there (its ``root`` is the sandbox, or its root's view), and
        its spec; ``EnvSetupError`` when it cannot start.  The caller stops it."""

        if harbor_task.verifier_environment is None:
            raise ValueError(f"Harbor task {harbor_task.name!r} grades in its own sandbox, not a separate one")
        try:
            spec = self._sandbox_spec(harbor_task, rollout_id, verifier=True)
        except (KeyError, ValueError) as exc:
            raise EnvSetupError("env_config", str(exc)) from exc
        users, _ = await self._start(spec, harbor_task, verifier=True)
        return users, spec

    async def _start(
        self, spec: SandboxSpec, harbor_task: HarborTask, *, verifier: bool = False
    ) -> tuple[SessionUsers, str | None]:
        """Start the task's sandbox (``verifier``: its separate verifier's) and prepare it; who runs what there, and
        the init a main-only compose file gives it ("<pid>:<start time>", or None).  On any failure the sandbox is
        stopped again and an ``EnvSetupError`` says why."""

        try:
            sandbox = await (self.verifier_provider if verifier else self.provider).start(
                self._start_spec(spec, harbor_task, verifier=verifier)
            )
        except Exception as exc:
            raise _start_failure(exc) from exc
        try:
            init, users = await self._prepare(sandbox, harbor_task, spec, verifier=verifier)
        except BaseException as exc:
            await asyncio.shield(sandbox.stop())
            if isinstance(exc, EnvSetupError) or not isinstance(exc, Exception):
                raise
            raise EnvSetupError("env_setup", f"{type(exc).__name__}: {exc}") from exc
        return users, init

    def _load(
        self, task_dir: str | Path, *, index: Mapping[str, Any] | None = None, compose: Any = _PLANNED_HERE
    ) -> HarborTask:
        """A task directory, read with the service's ``agent_user`` and ``verifier_user`` policies; ``compose`` is
        its stack plan from the lock (a fetched dataset's task), else it is planned from its files."""

        return HarborTask.load(
            task_dir,
            agent_user=self.agent_user,
            verifier_user=self.verifier_user,
            index=index,
            compose=compose,
            stack_refusal=self.stacks_refusal(),
            privilege_refusal=privilege_refusal(self._provider),
        )

    def _lookup(self, task_id: str) -> HarborTask:
        if self._locked is not None:  # workers read the lock: no fetch, no scan
            return self._load(
                self._locked.task_dir(task_id), index=self._locked.index(task_id), compose=self._locked.compose(task_id)
            )
        # Workers build a service per rollout, so an id finds its directory without indexing the dataset: through the
        # dataset's registry when it has one, else as a path below the dataset directory.
        if Path(task_id).is_absolute():
            return self._load(task_id)
        if self._dataset_dir is not None:
            task_dir = task_dir_of(self._dataset_dir, task_id, registry=self._registry)
            if task_dir is None:
                raise KeyError(f"Unknown Harbor task {task_id!r}")
            return self._load(task_dir, index=self._task_index.get(task_id))
        return self.dataset.get(task_id)

    @cached_property
    def _registry(self) -> dict[str, str] | None:
        return read_registry(self._dataset_dir) if self._dataset_dir is not None else None

    @cached_property
    def _task_index(self) -> dict[str, dict[str, Any]]:
        return read_task_index(self._dataset_dir) if self._dataset_dir is not None else {}

    def _session_image(self, harbor_task: HarborTask, *, honoring: bool | None = None) -> SessionImage | None:
        """The image of a task of a fetched dataset; None: the task's own ``docker_image`` or the default image, as a
        dataset directory's tasks run. Raises ``ValueError`` for a missing image.  Where the sandbox honors users
        (``honoring``, by default whether the agents' sandbox does) on a provider that loses the files' owners, a
        ``docker_image`` runs its pulled entry: the image its registry served, pinned, with its ``USER`` and owners."""

        if honoring is None:
            honoring = harbor_task.honors_users_in("agents")
        return self._images.for_task(
            harbor_task.task_dir,
            docker_image=harbor_task.docker_image,
            default_image=self.default_image,
            pulled=honoring and loses_file_owners(self._provider),
        )

    def _verifier_image(self, harbor_task: HarborTask) -> SessionImage | None:
        """The built image of a task's separate verifier, with its deferred steps: the image map's image of
        ``tests/Dockerfile``, or the task's own (:meth:`_session_image`); None for a ``[verifier.environment]``
        ``docker_image``, which runs as written, unless its users are honored where the files' owners are lost, and the
        map has its pulled entry.  Raises ``ValueError`` when the map has no image for it."""

        environment = harbor_task.verifier_environment
        honoring = harbor_task.honors_users_in("verifier")
        if environment is None:
            return None
        if environment.image_source == "verifier_image":
            # Run as written, unless it honors users where the files' owners are lost: then its pulled entry.
            entry = self._images.pulled(docker_image_reference(environment.docker_image or ""))
            if not (honoring and loses_file_owners(self._provider)) or entry is None:
                return None
            return SessionImage(image=entry.image, env={}, workdir=None, user=entry.user, file_owners=entry.file_owners)
        if environment.image_source == "task":
            return self._session_image(harbor_task, honoring=honoring)
        built = self._images.for_verifier(harbor_task.task_dir)
        if built is None or not built.image:
            raise ValueError(
                f"Harbor task {harbor_task.name!r} builds its verifier's image from tests/Dockerfile, and no "
                "image is known for it; build it with platoon prepare, which builds it into the image map"
            )
        return built

    def _sandbox_spec(self, harbor_task: HarborTask, rollout_id: str, *, verifier: bool = False) -> SandboxSpec:
        """The sandbox of the task's agents, or with ``verifier`` of its separate verifier (its own resources and
        network, no writable host mounts, so nothing the agents write reaches it besides the artifacts)."""

        environment = harbor_task.verifier_environment if verifier else None
        if verifier and environment is None:
            raise ValueError(f"Harbor task {harbor_task.name!r} has no separate verifier")
        gpus = environment.gpus if environment is not None else harbor_task.gpus
        if gpus > 0 and "gpu" not in (provider := self.verifier_provider if verifier else self.provider).capabilities:
            role = "its verifier needs" if verifier else "it needs"
            raise ValueError(
                f"Harbor task {harbor_task.name!r} is refused: {role} {gpus} GPU(s), and the {provider.name!r} "
                "sandbox provider cannot give a sandbox GPUs"
            )
        compose_env: dict[str, str] = {}
        if environment is None:
            built = self._session_image(harbor_task)
            image = built.image if built is not None and built.image else self._image(harbor_task)
            main = harbor_task.main
            composed = main.working_dir if main is not None else None
            workdir = harbor_task.workdir or composed or (built.workdir if built is not None else None)
            env = harbor_task.env
            cpus = main.cpus if main is not None and main.cpus is not None else harbor_task.cpus
            memory_mb = main.memory_mb if main is not None and main.memory_mb is not None else harbor_task.memory_mb
            network = harbor_task.network
            if harbor_task.is_stack:
                network = "none" if self.stack_network(harbor_task, network) == "none" else "host"
            settings = (network, cpus, memory_mb)
            mounts = list(self.mounts)
            if main is not None:
                compose_env = {
                    name: resolve_deferred(value, harbor_task.env, os.environ)
                    for name, value in main.environment.items()
                }
                mounts += sidecars.bind_mounts(harbor_task, main.binds, copied=sidecars.copied_binds(harbor_task))
        else:
            built = self._verifier_image(harbor_task)
            if built is not None and built.image:
                image = built.image
            elif environment.image_source == "verifier_image":
                assert environment.docker_image is not None
                image = docker_image_reference(environment.docker_image)
            else:
                image = self._image(harbor_task)
            workdir = environment.workdir or (built.workdir if built is not None else None) or harbor_task.workdir
            env = environment.env
            settings = (environment.network, environment.cpus, environment.memory_mb)
            mounts = [mount for mount in self.mounts if mount.read_only]
        network, cpus, memory_mb = settings
        if self.network_policy == "none":
            network = "none"
        labels = {"platoon.env": self.name, "platoon.rollout_id": rollout_id, "platoon.task": harbor_task.name}
        if verifier:
            labels["platoon.role"] = "verifier"
        return SandboxSpec(
            image=image,
            workdir=workdir or self.workdir,
            # The image's own environment and deferred ENV steps first; the service's and the task's go on top.
            env={**(built.env if built is not None else {}), **self.env, **compose_env, **resolve_env(env)},
            mounts=mounts,
            network=network,
            cpus=cpus,
            memory_mb=memory_mb,
            labels=labels,
            file_owners=self._file_owners(harbor_task, verifier=verifier),
        )

    def _file_owners(self, harbor_task: HarborTask, *, verifier: bool) -> str | None:
        """The owner manifest that a sandbox of the task (``verifier``: its separate verifier's) applies: None where it
        honors no users, where its provider keeps the files' owners, and where the image gives none to users other
        than root.  Where it needs the image map's record of its image (:func:`_needs_record`), :class:`EnvNotPrepared`
        when the map lacks it (``platoon prepare`` records it), and ``ValueError`` when no image map can hold it (a
        dataset directory's, a local image file) or the provider cannot apply it."""

        honors = harbor_task.honors_users_in("verifier" if verifier else "agents")
        if not honors or not loses_file_owners(self._provider):
            return None
        built = self._verifier_image(harbor_task) if verifier else self._session_image(harbor_task)
        manifest = built.file_owners if built is not None else None
        if not _needs_record(harbor_task, built, verifier=verifier):
            # Only root runs there, which needs no owners, as root verifiers ran before images had manifests: apply one
            # where the provider can.
            options = self._phase_options["verifier" if verifier else "agents"]
            usable = manifest is not None and file_owners_refusal(self._provider, options) is None
            return manifest if usable and os.path.exists(manifest) and os.path.getsize(manifest) > 0 else None
        where = f"Harbor task {harbor_task.name!r} honors users on the {self._provider_name()} provider, which loses "
        where += "the owners an image gives its files and cannot tell its USER"
        if manifest is None and self._locked is None:
            raise ValueError(
                f"{where}, and a dataset directory has no image map that could record them: serve the directory as a "
                "dataset to fetch (service_kwargs.dataset), whose images platoon prepare records with their owners "
                "and USER"
            )
        if manifest is None and built is not None and built.key:
            raise EnvNotPrepared(
                f"{where}, and its image (build key {built.key}) records no owners, as images that platoon before 0.5, "
                "the def builder, or the oci builder with a squashfs-tools before 4.4 built: build it again with the "
                f"oci builder, where unsquashfs is 4.4 or later (platoon harbor images build CONFIG --key {built.key})"
            )
        if manifest is None:
            environment = harbor_task.verifier_environment
            if built is not None and built.image:
                image = built.image
            elif verifier and environment is not None and environment.docker_image:
                image = docker_image_reference(environment.docker_image)
            else:
                image = self._image(harbor_task)
            if not image.startswith("docker://"):
                raise ValueError(
                    f"{where}, and its image {image} is no registry's, whose owners and USER platoon prepare could "
                    "record: give the task a docker_image or a Dockerfile, or leave it out (exclude_tasks)"
                )
            raise EnvNotPrepared(
                f"{where}, and the image map has no pulled entry for its image {image}, which records the owners its "
                "registry's layers give and its USER: platoon prepare records it"
            )
        try:
            if os.path.getsize(manifest) == 0:
                return None
        except OSError as exc:
            raise EnvNotPrepared(f"{where}, and its image's owner manifest cannot be read: {exc}") from exc
        options = self._phase_options["verifier" if verifier else "agents"]
        if (refusal := file_owners_refusal(self._provider, options)) is not None:
            raise ValueError(f"{where}, and its image gives files to users other than root: {refusal}")
        return manifest

    def _image(self, harbor_task: HarborTask) -> str:
        """The image of the task's own environment: its ``docker_image``, else the service's ``default_image``."""

        image = docker_image_reference(harbor_task.docker_image) if harbor_task.docker_image else self.default_image
        if image is None:
            raise ValueError(f"Harbor task {harbor_task.name!r} has no environment.docker_image and no default image")
        return image

    def _start_spec(self, spec: SandboxSpec, harbor_task: HarborTask, *, verifier: bool) -> SandboxSpec:
        """The spec the task's sandbox starts from: ``spec``, but in ``/`` where the sandbox honors users and its image
        defers a ``WORKDIR`` after a ``USER``, which makes its directories as that user.  hpc-sandbox and Docker make a
        working directory the image lacks as root when the sandbox starts; :meth:`_prepare` makes them instead, as the
        ``WORKDIR`` makes them, and the session's views run its commands in ``spec.workdir``."""

        if not harbor_task.honors_users_in("verifier" if verifier else "agents"):
            return spec
        built = self._verifier_image(harbor_task) if verifier else self._session_image(harbor_task)
        if built is None or not any(op.kind == "mkdir" and op.owner for op in built.files):
            return spec
        return replace(spec, workdir="/")

    async def _prepare(
        self, sandbox: Sandbox, harbor_task: HarborTask, spec: SandboxSpec, *, verifier: bool = False
    ) -> tuple[str | None, SessionUsers]:
        """Make the task's directories, apply the deferred Dockerfile steps and, in the agents' sandbox of a main-only
        task, its compose binds and init; wait for its healthchecks.  In a sandbox that honors users, platoon's
        commands run as root under Docker's umask (:mod:`platoon.envs.harbor.views`), the directories that the agents
        publish in are every user's (Harbor's 777), and a deferred copy's ``--chown`` and the ``USER`` that a deferred
        ``WORKDIR`` follows apply.  The init ("<pid>:<start time>", or None), and who runs what in the sandbox."""

        built = self._verifier_image(harbor_task) if verifier else self._session_image(harbor_task)
        honors = harbor_task.honors_users_in("verifier" if verifier else "agents")
        root: Sandbox = root_view(sandbox) if honors else sandbox
        if honors:
            await self._check_root(root, harbor_task, built, verifier=verifier)
        # The working directory after the deferred WORKDIR steps, which make it as their USER where the image lacks it.
        workdir_later = built is not None and any(op.kind == "mkdir" for op in built.files)
        # In a root view, what is missing gets Docker's modes (its WORKDIR, Harbor's directories), so that the other
        # users may enter it, and what the image has keeps its own, as in Docker, which may keep them out on purpose.
        result = await root.exec(["mkdir", "-p", *LOG_DIRS, *([] if workdir_later else [spec.workdir])], cwd="/")
        if result.exit_code == 0 and honors:
            result = await root.exec(["chmod", "777", *PUBLISHED_DIRS], cwd="/")
        if result.exit_code != 0:
            raise EnvSetupError("env_setup", f"Cannot create the task directories: {result.stderr.strip()}")
        main = harbor_task.main if not verifier else None
        if built is not None and built.files:  # the deferred Dockerfile steps: WORKDIR directories, COPY uploads
            if main is not None:  # a compose bind hides what the image has there, its deferred copies too
                built = replace(built, files=_unbound(built.files, main, harbor_task, spec.workdir))
            await apply_image_files(root, built, workdir=spec.workdir, owners=honors)
        if workdir_later:  # one the task names (task.toml, compose) where the image has none, as Docker's -w makes it
            result = await root.exec(["mkdir", "-p", spec.workdir], cwd="/")
            if result.exit_code != 0:
                raise EnvSetupError("env_setup", f"Cannot create the working directory: {result.stderr.strip()}")
        users = SessionUsers.alone(sandbox)
        if honors:
            users = await self._session_users(sandbox, harbor_task, built, spec, verifier=verifier)
        init = None
        if main is not None:
            for target in main.volumes:  # a volume of main's own: its directory, with what the image has there
                _check_contained(sandbox, target)
                made = await root.exec(["mkdir", "-p", target], cwd="/")
                if made.exit_code != 0:
                    raise RuntimeError(f"Cannot make the volume directory {target}: {made.stderr.strip()}")
            init = await self._start_init(users.default, harbor_task, main, spec, built)
        if harbor_task.healthcheck is not None and not verifier:  # as in Harbor, none runs in a verifier's sandbox
            await _wait_until_healthy(users.default, harbor_task.healthcheck)
        return init, users

    async def _check_root(
        self, root: Sandbox, harbor_task: HarborTask, built: SessionImage | None, *, verifier: bool
    ) -> None:
        """Refuse, with an ``env_config`` ``EnvSetupError``, a sandbox that honors users whose ``root`` view's commands
        do not run as root with every uid and gid up to nobody's mapped, or whose image's shell nobody may not run
        (:mod:`platoon.envs.harbor.users`)."""

        problem = await root_problem(root)
        if problem is None and (problem := await nobody_problem(root)) is not None:
            if built is not None and built.key:  # an image of the map: platoon's oci builder may have made it so
                problem += (
                    ", as the oci builder of platoon before 0.3.0 left the images it built under a umask such as "
                    f"027: build it again (platoon harbor images build CONFIG --key {built.key})"
                )
            else:
                problem += (
                    f": give {'the verifier' if verifier else 'the task'} an image whose programs other users may run"
                )
        if problem is not None:
            if verifier:
                phase = "its verifier as root" if harbor_task.verifier_root else "its verifier as the user it names"
            else:
                phase = "its commands as the users it names"
            raise EnvSetupError(
                "env_config", f"Harbor task {harbor_task.name!r} runs {phase}, and its sandbox {problem}"
            )

    async def restored_users(self, sandbox: Sandbox, harbor_task: HarborTask, spec: SandboxSpec) -> SessionUsers:
        """Who runs what in ``sandbox``, a copy fork's restore of the task's agents' sandbox: the sandbox alone where it
        honors no users, else as at that sandbox's start (:meth:`_session_users`)."""

        if not harbor_task.honors_users_in("agents"):
            return SessionUsers.alone(sandbox)
        return await self._session_users(sandbox, harbor_task, self._session_image(harbor_task), spec, verifier=False)

    async def _session_users(
        self,
        sandbox: Sandbox,
        harbor_task: HarborTask,
        built: SessionImage | None,
        spec: SandboxSpec,
        *,
        verifier: bool,
    ) -> SessionUsers:
        """Who runs what in ``sandbox``, which honors users (:func:`platoon.envs.harbor.views.session_users`); an
        ``env_config`` ``EnvSetupError`` names a user that cannot run there.  The interpreters of its read-only mounts
        (a harness toolchain's) may switch users where the image has nothing that does."""

        pythons = [posixpath.join(mount.target, "bin", "python3") for mount in spec.mounts if mount.read_only]
        try:
            return await session_users(
                sandbox,
                harbor_task,
                image_user=built.user if built is not None else None,
                verifier=verifier,
                spec_env=spec.env,
                pythons=pythons,
                workdir=spec.workdir,
            )
        except ValueError as exc:
            raise EnvSetupError("env_config", f"Harbor task {harbor_task.name!r}: {exc}") from exc

    async def _start_init(
        self,
        sandbox: Sandbox,
        harbor_task: HarborTask,
        main: MainService,
        spec: SandboxSpec,
        built: SessionImage | None,
    ) -> str | None:
        """Start main's init, the command a main-only compose file gives it (with its image's ENTRYPOINT, where the
        image map knows it), wait for main's healthcheck (compose's, else its image's), and keep the init and what it
        started alive through grading (``Sandbox.keep_processes``); None when main only idles."""

        argv = main.init(built.entrypoint if built is not None else None)
        check: ServiceHealthcheck | None = main.healthcheck
        if check is None and not main.healthcheck_disabled and built is not None:
            check = image_healthcheck(built.healthcheck)
        if argv is None:
            if check is not None:
                await _wait_until_healthy(sandbox, check)
            return None
        keep = getattr(sandbox, "keep_processes", None)
        if keep is None:
            raise EnvSetupError(
                "env_config",
                f"Harbor task {harbor_task.name!r}: its compose file gives main a command, which the "
                f"{type(sandbox).__name__} cannot keep running through grading (it has no keep_processes)",
            )
        before = await list_processes(sandbox)
        init = await start_background(sandbox, argv, log=INIT_LOG, cwd=spec.workdir)
        if check is not None:
            await _wait_until_healthy(sandbox, check)
        if not await alive_processes(sandbox, {init}):
            log = await sandbox.exec(["tail", "-c", "2000", INIT_LOG], cwd="/")
            raise EnvSetupError(
                "env_setup",
                f"Harbor task {harbor_task.name!r}: main's command ({argv[0]}) exited as the session started: "
                f"{(log.stdout or log.stderr).strip()[-1000:]}",
            )
        # What appeared since it started is its own: the processes it runs, the daemons it detached.
        keep((await list_processes(sandbox)) - before | {init})
        return init


def _needs_record(harbor_task: HarborTask, built: SessionImage | None, *, verifier: bool) -> bool:
    """Whether a sandbox of the task that honors users (``verifier``: its separate verifier's), on a provider that
    loses the files' owners, needs what the image map records of its image (``built``), the owners its files have and
    its ``USER``: where it runs a user other than root, whose files' owners matter, or, where the map records neither
    (no ``file_owners``), something as the image's ``USER``, which only that record tells
    (:func:`~platoon.envs.harbor.users.phase_users`)."""

    recorded = built is not None and built.file_owners is not None
    if not recorded and runs_image_user(harbor_task, verifier=verifier):
        return True
    users = phase_users(harbor_task, image_user=built.user if built is not None else None, verifier=verifier)
    return any(not is_root(user) for user in [*users.values(), *hook_users(harbor_task, verifier=verifier)] if user)


def _image_name(session: SessionImage | None) -> str | None:
    """A session image's name in the image map: its build key, else the reference it runs."""

    if session is None:
        return None
    return session.key or session.image


def _users_refusal(harbor_task: HarborTask, cannot: Mapping[str, str | None]) -> str | None:
    """Why a sandbox of the task that honors users could not run (``cannot``: why the agents' sandboxes, and the
    verifiers', cannot give root and other users, :func:`platoon.sandbox.root_refusal`); None when nothing says."""

    if harbor_task.honors_users_in("agents") and cannot["agents"] is not None:
        return (
            "its agents' sandbox runs commands as the users it names, and the agents' sandboxes cannot give root and "
            f"other users: {cannot['agents']}"
        )
    if harbor_task.honors_users_in("verifier") and cannot["verifier"] is not None:
        named = harbor_task.honored_users.get("verifier")
        who = "root" if harbor_task.verifier_root else repr(named) if named else "its image's user"
        return f"its verifier runs as {who}, and the verifiers' sandboxes cannot give it root: {cannot['verifier']}"
    return None


def _start_failure(exc: Exception) -> EnvSetupError:
    """What a sandbox's or a stack's start that raised ``exc`` fails a session's start with: ``harness_setup`` for a
    route that main's image cannot serve (no Python for the forwarder), ``sidecar_start`` for a sidecar of a stack that
    did not start, else ``sandbox_start``."""

    if isinstance(exc, StackStartError):
        if exc.service != MAIN:
            return EnvSetupError("sidecar_start", str(exc))
        if isinstance(exc.__cause__, RouteError):
            return EnvSetupError("harness_setup", str(exc.__cause__))
        return EnvSetupError("sandbox_start", str(exc))
    if isinstance(exc, RouteError):  # no Python in the image for the route forwarder: the harness cannot run here
        return EnvSetupError("harness_setup", str(exc))
    return EnvSetupError("sandbox_start", f"{type(exc).__name__}: {exc}")


async def _no_setup(sandbox: Sandbox) -> None:
    """The setup of a stack's main that is only checked, never started."""


def _remove(directories: Sequence[Path]) -> None:
    for directory in directories:
        shutil.rmtree(directory, ignore_errors=True)


def _network_refusal(harbor_task: HarborTask, refusals: Mapping[str, str | None], policy: str = "task") -> str | None:
    """Of the phases' ``refusals`` (:meth:`HarborEnvironmentService._network_refusals`), the one that refuses the task:
    its agents', when they run without a network (``network_mode = "no-network"``, or ``network_policy: none``), else
    its separate verifier's, when that runs without one; None when neither applies.  A stack's network is its stack
    provider's to give: only its separate verifier's sandbox counts."""

    offline = harbor_task.network == "none" or policy == "none"
    if offline and not harbor_task.is_stack and refusals["agents"] is not None:
        return refusals["agents"]
    verifier = harbor_task.verifier_environment
    if verifier is not None and (verifier.network == "none" or policy == "none"):
        return refusals["verifier"]
    return None


def _unbound(files: Sequence[FileOp], main: MainService, harbor_task: HarborTask, workdir: str) -> tuple[FileOp, ...]:
    """The deferred file operations of a main-only task but those at or below one of ``main``'s binds, which hide
    what the image has there, as in Docker; ``env_config`` for a copy of a directory that holds a bind's target,
    which would write into the bind."""

    kept = []
    for op in files:
        target = op.target or workdir
        if any(_at_or_below(target, bind.target) for bind in main.binds if bind.logs is None):
            continue
        holds = [bind.target for bind in main.binds if bind.logs is None and _at_or_below(bind.target, target)]
        if op.kind == "copy" and op.into and holds:
            raise EnvSetupError(
                "env_config",
                f"Harbor task {harbor_task.name!r}: its Dockerfile copies {op.source} into {target}, which holds the "
                f"compose bind at {holds[0]}; platoon cannot apply both",
            )
        kept.append(op)
    return tuple(kept)


def _at_or_below(path: str, directory: str) -> bool:
    return path == directory or path.startswith(directory.rstrip("/") + "/")


async def _wait_until_healthy(sandbox: Sandbox, check: Healthcheck | ServiceHealthcheck) -> None:
    """Retry the probe like Docker's HEALTHCHECK; failures in the start period do not count."""

    started = asyncio.get_running_loop().time()
    failures = 0
    while True:
        result = await sandbox.exec(check.command, timeout_s=check.timeout_s)
        if result.exit_code == 0 and not result.timed_out:
            return
        starting = asyncio.get_running_loop().time() - started < check.start_period_s
        if not starting:
            failures += 1
            if failures >= check.retries:
                detail = (result.stderr or result.stdout).strip()[-500:]
                raise EnvSetupError("env_healthcheck", f"The healthcheck failed {failures} times: {detail}")
        await asyncio.sleep(check.start_interval_s if starting else check.interval_s)
