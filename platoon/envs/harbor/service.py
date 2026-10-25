"""The Harbor environment service: Harbor task directories served as platoon env sessions."""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Mapping, Sequence
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar

from platoon.deprecations import removed_hint
from platoon.envs.gate import DEFAULT_QUEUE_TIMEOUT_S, check_queue_timeout
from platoon.envs.harbor.artifacts import DEFAULT_MAX_ARTIFACT_BYTES
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
    ImageMapError,
    SessionImage,
    apply_image_files,
    docker_image_reference,
    has_image,
    withholds_modes,
)
from platoon.envs.harbor.lock import LockError
from platoon.envs.harbor.locked import LockedTasks, TaskImages
from platoon.envs.harbor.session import DEFAULT_CLOSE_TIMEOUT_S, LOG_DIRS, HarborSession, SandboxLease
from platoon.envs.harbor.task import (
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
from platoon.envs.harbor.users import DOCKER_UMASK, nobody_problem, root_problem
from platoon.envs.service import EnvNotPrepared, EnvSetupError, Outcome, PathOption, SessionNeeds
from platoon.sandbox import (
    Mount,
    Sandbox,
    SandboxProvider,
    SandboxSpec,
    get_provider,
    network_none_refusal,
    root_refusal,
    rootfs_layout_refusal,
)
from platoon.sandbox.routes import RouteError
from platoon.task import Task

logger = logging.getLogger(__name__)

DEFAULT_SANDBOX_START_BUDGET_S = 600.0
# What the grading commands around the tests take at most besides them: killing processes, checking the workdir,
# making the directories, uploading the tests, reading the output and, for a separate verifier, the artifacts.
_GRADING_ALLOWANCE_S = 120.0
_REPORTED: set[tuple[str, str]] = set()  # hold-out reports logged by this process: (dataset, eval digest)


class HarborEnvironmentService:
    """Serve Harbor tasks, one sandbox per session, graded by each task's ``tests/test.sh``.

    Task ids are the dataset's (:class:`~platoon.envs.harbor.dataset.HarborDataset`): registry names, directory
    names, or relative paths of deeper task directories; ``get_task`` also accepts an absolute task directory path.
    A session's sandbox runs the task's ``environment.docker_image`` (a registry reference gets ``docker://``; a
    ``.sif``, ``.sqsh`` or absolute path is used as written), else ``default_image``.  A task whose environment is
    a compose stack (sidecar services) is refused when it loads, never started as its main container alone and
    scored; so is a task that needs GPUs on a provider without them, when its session opens.
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
    (``refuse`` or ``ignore``) decides what a task naming a user to run as gets, and ``verifier_user`` decides for
    ``[verifier].user`` instead when it is set: ``honor`` runs a separate verifier whose user is root as root
    (:mod:`platoon.envs.harbor.users`).  Separate verifiers' sandboxes start with ``verifier_provider_options`` laid
    over ``provider_options``, such as Singularity's ``fakeroot`` for verifiers that run as root.

    Agents sharing a sandbox take turns on it: its tools run one at a time,
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
        ``exclude_tasks`` left of the listing) that the sandbox provider would refuse to start, or whose verifier
        could not run as their ``[verifier].user`` asks (:meth:`_check_users`), or whose verifier would run as root
        on an image that may keep its files from other users (:meth:`_check_root_images`), so that the run stops when
        it lists its tasks rather than lose every rollout of theirs; and, with :class:`EnvNotPrepared`, tasks of a
        fetched dataset whose images the image map lacks (:meth:`_check_images`)."""

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
        self._check_networks(task_ids, split)
        self._check_users(task_ids, split)
        self._check_images(task_ids, split)
        self._check_root_images(task_ids, split)

    def _check_images(self, names: Sequence[str], split: str) -> None:
        """Refuse tasks of a fetched dataset that build their image, or their separate verifier's, from a Dockerfile
        whose image the image map lacks (or whose image file is gone, or is not the size its build recorded): a run
        never builds an image, and every rollout of theirs would fail when its session opens.  ``platoon prepare``
        builds them.  Tasks of a dataset directory plan their images when a session opens, and are not checked; nor
        are tasks that run a prebuilt image.  Each image is looked at once, however many tasks share it."""

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
            for plan in (row.image, row.verifier_image or {}):
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

    def _provider_name(self) -> str:
        return self._provider if isinstance(self._provider, str) else getattr(self._provider, "name", "sandbox")

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
            if (refusal := _network_refusal(harbor_task, refusals)) is not None:
                offline[name] = refusal
        if offline:
            shown = ", ".join(list(offline)[:3]) + (f" and {len(offline) - 3} more" if len(offline) > 3 else "")
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
        """With ``verifier_user: honor``, refuse the tasks whose ``[verifier].user`` it cannot honor (another user
        than root, or a root verifier of a task graded in its agents' sandbox, which :meth:`HarborTask.load` refuses),
        and the tasks whose verifier runs as root where the verifiers' sandboxes cannot give it root and other users
        (:func:`platoon.sandbox.root_refusal`): every rollout of theirs would fail, or score 0 when their tests cannot
        run the agent's code as another user."""

        if self.verifier_user != "honor" or not names:
            return
        self._warn_of_fetch_refusals()
        cannot_give_root = root_refusal(self._provider, self._phase_options["verifier"])
        refused: dict[str, str] = {}  # task -> why
        for name in names:
            try:
                harbor_task = self._peek(name)
            except HarborTaskError as exc:  # its users, which a fetch lets through for the service to decide about
                refused[name] = str(exc)
                continue
            except (KeyError, ValueError):  # unknown, or not readable: it fails on its own when it is opened
                continue
            if harbor_task.verifier_root and cannot_give_root is not None:
                refused[name] = (
                    f"its verifier runs as root, and the verifiers' sandboxes cannot give it root: {cannot_give_root}"
                )
        if refused:
            shown = ", ".join(list(refused)[:3]) + (f" and {len(refused) - 3} more" if len(refused) > 3 else "")
            name, why = next(iter(refused.items()))
            root = any(reason.startswith("its verifier runs as root") for reason in refused.values())
            raise EnvSetupError(
                "env_config",
                f"{len(refused)} of the {len(names)} tasks of split {split!r} cannot run their verifier as their "
                f"[verifier].user asks ({shown}); {name}: {why}. "
                + ("Give the verifiers' sandboxes root (verifier_provider_options), set " if root else "Set ")
                + "verifier_user: ignore to run them as the sandbox's own user, or leave those tasks out "
                "(exclude_tasks)",
            )

    def _warn_of_fetch_refusals(self) -> None:
        """Warn, once, of the tasks whose ``[verifier].user`` is root that the fetch of the service's dataset left out:
        its ``refuse.agent_user``, on by default, refuses every task that names a user, so ``verifier_user: honor``
        never sees them."""

        if self._locked is None or self._warned_of_fetch:
            return
        self._warned_of_fetch = True
        lock = self._locked.dataset(fetch_allowed=False).lock
        left_out = [
            task.name
            for task in lock.tasks
            if (task.refused or "").startswith("agent_user:")
            and task.flags.get("verifier_user") is not None
            and is_root(str(task.flags["verifier_user"]))
        ]
        if left_out:
            shown = ", ".join(left_out[:3]) + (f" and {len(left_out) - 3} more" if len(left_out) > 3 else "")
            logger.warning(
                "verifier_user: honor runs none of the %d task(s) of %s whose [verifier].user is root (%s): the fetch "
                "left them out, since its refuse.agent_user, on by default, refuses every task that names a user. Set "
                "refuse: {agent_user: false} to fetch them, with agent_user: ignore for those that also set "
                "[agent].user",
                len(left_out),
                lock.dataset,
                shown,
            )

    def root_images_closed(self, names: Sequence[str] | None = None) -> dict[str, list[str]]:
        """With ``verifier_user: honor``, the images of the image map that tasks of a fetched dataset (``names``, else
        every runnable task of its lock) run their verifier as root on and that may keep their files from the users
        the tests switch to, since the ``oci`` builder of platoon before 0.3.0 built them
        (:func:`~platoon.envs.harbor.images.withholds_modes`): by build key, with the tasks that run each.  Empty
        otherwise, and for images the map lacks, which :meth:`_check_images` reports."""

        if self.verifier_user != "honor" or self._locked is None:
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
            environment = harbor_task.verifier_environment
            if not harbor_task.verifier_root or environment is None:
                continue
            row = lock.get(name)
            plan = {"task": row.image, "tests": row.verifier_image}.get(environment.image_source) or {}
            entry = image_map.get(key) if (key := plan.get("key")) else None
            if entry is not None and withholds_modes(entry):
                closed.setdefault(key, []).append(name)
        return closed

    def _check_root_images(self, names: Sequence[str], split: str) -> None:
        """Refuse tasks whose verifier runs as root on an image the ``oci`` builder of platoon before 0.3.0 built
        (:meth:`root_images_closed`): on a host whose umask is 027, the users its tests switch to cannot run its
        programs, and every grade would fail or score 0.  A run never builds images; the message names the build."""

        if not names or not (closed := self.root_images_closed(names)):
            return
        tasks = sorted({name for needing in closed.values() for name in needing})
        shown = ", ".join(tasks[:3]) + (f" and {len(tasks) - 3} more" if len(tasks) > 3 else "")
        keys = " ".join(f"--key {key}" for key in closed)
        raise EnvSetupError(
            "env_config",
            f"{len(tasks)} of the {len(names)} tasks of split {split!r} run their verifier as root ({shown}) on "
            f"{len(closed)} image(s) that the oci builder of platoon before 0.3.0 built, which unpacked them under the "
            "host's umask: under one such as 027, their files keep out the users the tests switch to. Build them again "
            f"with `platoon harbor images build CONFIG {keys}`, or set verifier_user: ignore, or leave those tasks out "
            "(exclude_tasks)",
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
            return self._load(dataset.cache / row.path, index=row.index)
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
        timeout (times ``verifier_timeout_multiplier``), the grading commands around them, and the close.  Deadlines
        on whole rollouts add it (:func:`platoon.train.components.rollout_overhead_s`)."""

        return self._limits(task_id)[1]

    def _limits(self, task_id: str) -> tuple[float | None, float]:
        """:meth:`agent_timeout_s` and :meth:`rollout_overhead_s` of a task, from one read of it: the trainer asks
        for both before each rollout."""

        if (known := self._task_limits.get(task_id)) is not None:
            return known
        harbor_task = self._peek(task_id)
        agent = harbor_task.agent_timeout_s
        starts = 2 if harbor_task.verifier_mode == "separate" else 1
        check = harbor_task.healthcheck
        health = (
            0.0 if check is None else check.start_period_s + (check.retries + 1) * (check.timeout_s + check.interval_s)
        )
        verifier = harbor_task.verifier_timeout_s * self.verifier_timeout_multiplier
        overhead = (
            starts * self.sandbox_start_budget_s + health + verifier + _GRADING_ALLOWANCE_S + self.close_timeout_s
        )
        limits = (None if agent is None else agent * self.agent_timeout_multiplier, overhead)
        self._task_limits[task_id] = limits
        return limits

    def harbor_task(self, task: Task) -> HarborTask:
        """The Harbor task behind a platoon task: the directory in ``task.misc``, else the task id."""

        misc = task.misc.get(HARBOR_MISC_KEY, {})
        if misc.get("task_dir") is not None:
            return self._load(misc["task_dir"], index=misc.get("index"))
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
            spec = self._sandbox_spec(harbor_task, rollout_id)
            if harbor_task.verifier_mode == "separate":
                self._sandbox_spec(harbor_task, rollout_id, verifier=True)  # refused now, not after the agent ran
        except (KeyError, ValueError) as exc:  # a bad task directory (HarborTaskError), unknown id or unset variable
            raise EnvSetupError("env_config", str(exc)) from exc
        if (refusal := _network_refusal(harbor_task, self._network_refusals())) is not None:
            raise EnvSetupError("env_config", f"Harbor task {harbor_task.name!r} runs without a network, and {refusal}")
        if harbor_task.verifier_root and (refusal := root_refusal(self._provider, self._phase_options["verifier"])):
            raise EnvSetupError(  # refused now, not after the agent ran
                "env_config",
                f"Harbor task {harbor_task.name!r} runs its verifier as root, and the verifiers' sandboxes cannot give "
                f"it root: {refusal}",
            )
        if needs is not None:  # the harness's gateway route, toolchain and memory; never the verifier's
            spec = needs.apply(spec)
        sandbox = await self._start(spec, harbor_task)
        lease = SandboxLease(sandbox, spec, queue_timeout_s=self.queue_timeout_s, close_timeout_s=self.close_timeout_s)
        return HarborSession(self, task, harbor_task, lease, rollout_id=rollout_id)

    async def open_verifier(self, harbor_task: HarborTask, *, rollout_id: str) -> tuple[Sandbox, SandboxSpec]:
        """Start the separate verifier sandbox of a task (:class:`~platoon.envs.harbor.task.VerifierEnvironment`),
        ready for the artifacts and tests; ``EnvSetupError`` when it cannot start.  The caller stops it."""

        if harbor_task.verifier_environment is None:
            raise ValueError(f"Harbor task {harbor_task.name!r} grades in its own sandbox, not a separate one")
        try:
            spec = self._sandbox_spec(harbor_task, rollout_id, verifier=True)
        except (KeyError, ValueError) as exc:
            raise EnvSetupError("env_config", str(exc)) from exc
        return await self._start(spec, harbor_task, verifier=True), spec

    async def _start(self, spec: SandboxSpec, harbor_task: HarborTask, *, verifier: bool = False) -> Sandbox:
        """Start the task's sandbox (``verifier``: its separate verifier's) and prepare it; on any failure it is
        stopped again and an ``EnvSetupError`` says why."""

        try:
            sandbox = await (self.verifier_provider if verifier else self.provider).start(spec)
        except RouteError as exc:  # no Python in the image for the route forwarder: the harness cannot run here
            raise EnvSetupError("harness_setup", str(exc)) from exc
        except Exception as exc:
            raise EnvSetupError("sandbox_start", f"{type(exc).__name__}: {exc}") from exc
        try:
            await self._prepare(sandbox, harbor_task, spec, verifier=verifier)
        except BaseException as exc:
            await asyncio.shield(sandbox.stop())
            if isinstance(exc, EnvSetupError) or not isinstance(exc, Exception):
                raise
            raise EnvSetupError("env_setup", f"{type(exc).__name__}: {exc}") from exc
        return sandbox

    def _load(self, task_dir: str | Path, *, index: Mapping[str, Any] | None = None) -> HarborTask:
        """A task directory, read with the service's ``agent_user`` and ``verifier_user`` policies."""

        return HarborTask.load(task_dir, agent_user=self.agent_user, verifier_user=self.verifier_user, index=index)

    def _lookup(self, task_id: str) -> HarborTask:
        if self._locked is not None:  # workers read the lock: no fetch, no scan
            return self._load(self._locked.task_dir(task_id), index=self._locked.index(task_id))
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

    def _session_image(self, harbor_task: HarborTask) -> SessionImage | None:
        """The image of a task of a fetched dataset; None: the task's own ``docker_image`` or the default image, as a
        dataset directory's tasks run. Raises ``ValueError`` for a missing image."""

        return self._images.for_task(
            harbor_task.task_dir, docker_image=harbor_task.docker_image, default_image=self.default_image
        )

    def _verifier_image(self, harbor_task: HarborTask) -> SessionImage | None:
        """The built image of a task's separate verifier, with its deferred steps: the image map's image of
        ``tests/Dockerfile``, or the task's own (:meth:`_session_image`); None for a ``[verifier.environment]``
        ``docker_image``, which runs as written.  Raises ``ValueError`` when the map has no image for it."""

        environment = harbor_task.verifier_environment
        if environment is None or environment.image_source == "verifier_image":
            return None
        if environment.image_source == "task":
            return self._session_image(harbor_task)
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
        if environment is None:
            built = self._session_image(harbor_task)
            image = built.image if built is not None and built.image else self._image(harbor_task)
            workdir = harbor_task.workdir or (built.workdir if built is not None else None)
            env = harbor_task.env
            settings = (harbor_task.network, harbor_task.cpus, harbor_task.memory_mb)
            mounts = list(self.mounts)
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
        labels = {"platoon.env": self.name, "platoon.rollout_id": rollout_id, "platoon.task": harbor_task.name}
        if verifier:
            labels["platoon.role"] = "verifier"
        return SandboxSpec(
            image=image,
            workdir=workdir or self.workdir,
            # The image's own environment and deferred ENV steps first; the service's and the task's go on top.
            env={**(built.env if built is not None else {}), **self.env, **resolve_env(env)},
            mounts=mounts,
            network=network,
            cpus=cpus,
            memory_mb=memory_mb,
            labels=labels,
        )

    def _image(self, harbor_task: HarborTask) -> str:
        """The image of the task's own environment: its ``docker_image``, else the service's ``default_image``."""

        image = docker_image_reference(harbor_task.docker_image) if harbor_task.docker_image else self.default_image
        if image is None:
            raise ValueError(f"Harbor task {harbor_task.name!r} has no environment.docker_image and no default image")
        return image

    async def _prepare(
        self, sandbox: Sandbox, harbor_task: HarborTask, spec: SandboxSpec, *, verifier: bool = False
    ) -> None:
        root = verifier and harbor_task.verifier_root
        built = self._verifier_image(harbor_task) if verifier else self._session_image(harbor_task)
        if root:
            problem = await root_problem(sandbox)
            if problem is None and (problem := await nobody_problem(sandbox)) is not None:
                if built is not None and built.key:  # an image of the map: platoon's oci builder may have made it so
                    problem += (
                        ", as the oci builder of platoon before 0.3.0 left the images it built under a umask such as "
                        f"027: build it again (platoon harbor images build CONFIG --key {built.key})"
                    )
                else:
                    problem += ": give the verifier an image whose programs other users may run"
            if problem is not None:
                raise EnvSetupError(
                    "env_config",
                    f"Harbor task {harbor_task.name!r} runs its verifier as root, and its sandbox {problem}",
                )
        mkdir = ["mkdir", "-p", *LOG_DIRS, spec.workdir]
        if root:
            # What is missing gets Docker's modes (its WORKDIR, Harbor's directories), so that the users the tests
            # switch to may enter it; what the image has keeps its mode, as in Docker, which may keep them out on
            # purpose.  The sandbox's start made a missing workdir already, with these modes in hpc-sandbox's --fakeroot
            # sandboxes.
            mkdir = ["/bin/sh", "-c", f'umask {DOCKER_UMASK} && mkdir -p -- "$@"', "platoon-mkdir", *mkdir[2:]]
        result = await sandbox.exec(mkdir, cwd="/")
        if result.exit_code != 0:
            raise EnvSetupError("env_setup", f"Cannot create the task directories: {result.stderr.strip()}")
        if built is not None and built.files:  # the deferred Dockerfile steps: WORKDIR directories, COPY uploads
            await apply_image_files(sandbox, built, workdir=spec.workdir, umask=DOCKER_UMASK if root else None)
        if harbor_task.healthcheck is not None and not verifier:  # as in Harbor, none runs in a verifier's sandbox
            await _wait_until_healthy(sandbox, harbor_task.healthcheck)


def _network_refusal(harbor_task: HarborTask, refusals: Mapping[str, str | None]) -> str | None:
    """Of the phases' ``refusals`` (:meth:`HarborEnvironmentService._network_refusals`), the one that refuses the task:
    its agents', when they run without a network (``network_mode = "no-network"``), else its separate verifier's,
    when that runs without one; None when neither applies."""

    if harbor_task.network == "none" and refusals["agents"] is not None:
        return refusals["agents"]
    verifier = harbor_task.verifier_environment
    if verifier is not None and verifier.network == "none":
        return refusals["verifier"]
    return None


async def _wait_until_healthy(sandbox: Sandbox, check: Healthcheck) -> None:
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
