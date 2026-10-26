"""Deadlines on whole rollouts count what a Harbor rollout takes beyond its harness: starting its sandboxes, the
verifier's timeout, grading's own commands and the close.  A rollout worker's alarm and an evaluation slot's hard
timeout add it to the rollout's timeout, so a slow verifier never kills the worker that grades.  Without a timeout,
they count the harness's own time limit instead: the earliest of its spec's ``timeout_s`` and the task's own limit
for its agent (a Harbor task's ``[agent] timeout_sec`` times ``agent_timeout_multiplier``), where platoon's harness
rollout runs the task."""

from __future__ import annotations

import asyncio
import logging
import math
from types import SimpleNamespace
from typing import Any

import pytest

from platoon.config_defs import RolloutConfig
from platoon.envs.harbor import HarborEnvironmentService
from platoon.eval.config_defs import EvalWorkflowConfig
from platoon.eval.subprocess_worker import _hard_timeout_seconds
from platoon.eval.workflow import DefaultEvalGroupWorkflow, _ProcessSlots
from platoon.programs import resolve_program
from platoon.task import Task
from platoon.train import auto
from platoon.train.components import (
    HARNESS_ROLLOUT,
    MIXTURE_ROLLOUTS,
    PLATOON_PROGRAMS,
    EnvironmentRoute,
    harness_time_limit_s,
    known_rollout_overhead_s,
    rollout_overhead_s,
)
from platoon.train.config_defs import WorkflowConfig
from platoon.train.workers.protocol import resolve_ref, validated_ref
from platoon.train.workflow.group import GroupRolloutWorkflow

SEPARATE = '[verifier]\nenvironment_mode = "separate"\ntimeout_sec = 900\n'
HEALTHCHECK = '[environment.healthcheck]\ncommand = "true"\nretries = 2\ninterval_sec = 5\ntimeout_sec = 10\n'
AGENT = "[agent]\ntimeout_sec = 300\n"  # with agent_timeout_multiplier 2 below: 600 s
OVERHEAD = 600 + 600 + 120 + 30  # one sandbox start, the default 600 s verifier, grading's commands, the close
MIXTURE_ROLLOUT, MIXTURE_EVAL_ROLLOUT = "platoon.train.auto:mixture_rollout", "platoon.train.auto:mixture_eval_rollout"
OWN_ROLLOUT = f"{__name__}:own_rollout"
OWN_PROGRAM = "my_environment.programs:judged"  # an episode program of the environment's own
SHORT_SPEC = 'name = "short-agent"\nversion = "0.1.0"\ntimeout_s = 120\n\n[launch]\nargv = ["true"]\n'


def _routed(task_id: str, service_kwargs: dict, service: str = "harbor") -> Task:
    return EnvironmentRoute(name="env", service=service, service_kwargs=service_kwargs).attach(Task(id=task_id))


def _harness_routed(
    task_id: str,
    service_kwargs: dict,
    *,
    harness: str = "platoon-bash",  # a TOML spec with timeout_s = 1800
    rollout: str = HARNESS_ROLLOUT,
    eval_rollout: str | None = None,
    service: str = "harbor",
    program: str = "single",
) -> Task:
    route = EnvironmentRoute(
        name="env",
        service=service,
        service_kwargs=service_kwargs,
        harness=harness,
        rollout=rollout,
        eval_rollout=eval_rollout or rollout,
        program=program,
    )
    return route.attach(Task(id=task_id))


def _budget_workflow(rollout_fn_ref: str | None = HARNESS_ROLLOUT) -> GroupRolloutWorkflow:
    """A workflow with what ``_rollout_budget_seconds`` reads: its config, and the rollout function it runs."""

    workflow = object.__new__(GroupRolloutWorkflow)
    workflow.config = WorkflowConfig(worker_finalize_grace_seconds=120.0, worker_unbounded_rollout_budget_seconds=7.0)
    workflow.rollout_fn_ref = rollout_fn_ref
    return workflow


@pytest.fixture
def harbor(make_task) -> dict:
    """Service options of two Harbor tasks: ``limited`` lets its agent take 300 s (600 s with the multiplier),
    ``open`` sets no limit."""

    tasks = [make_task("limited", task_toml=AGENT), make_task("open")]
    return {"task_dirs": [str(task) for task in tasks], "provider": "process", "agent_timeout_multiplier": 2.0}


ran: list[str] = []


async def own_rollout(task: Task, config) -> dict:
    """A rollout function of the environment's own: platoon cannot know which limits it applies."""

    ran.append(f"own:{task.id}")
    return {}


async def other_own_rollout(task: Task, config) -> dict:
    ran.append(f"other:{task.id}")
    return {}


class BareService:
    """A service of one's own that says nothing about its tasks' limits."""

    name = "bare"

    def __init__(self, **options) -> None:
        self.options = options

    async def list_tasks(self, split: str) -> list[str]:
        return ["short", "endless", "unset"]

    async def get_task(self, task_id: str) -> Task:
        return Task(id=task_id, goal="g")

    async def open(self, task: Task, *, rollout_id: str):
        raise NotImplementedError


class LimitedService(BareService):
    """Gives each task's own agent time limit and its overhead."""

    def agent_timeout_s(self, task_id: str) -> float | None:
        return {"short": 30.0, "endless": math.inf}.get(task_id)

    def rollout_overhead_s(self, task_id: str) -> float:
        return 45.0


class FailingService(BareService):
    """Cannot read its tasks' limits (say, a task it does not know)."""

    def agent_timeout_s(self, task_id: str) -> float | None:
        raise RuntimeError(f"no task {task_id}")


def test_a_harbor_rollout_counts_its_sandbox_starts_and_verifier(make_task):
    tasks = [make_task("shared"), make_task("separate", task_toml=SEPARATE), make_task("health", task_toml=HEALTHCHECK)]
    service = HarborEnvironmentService(task_dirs=tasks, provider="process")

    # one start (600) + the verifier's default 600 s + 120 s for grading's commands + the 30 s close
    assert service.rollout_overhead_s("shared") == 600 + 600 + 120 + 30
    # a separate verifier starts a sandbox of its own, and runs for its own timeout
    assert service.rollout_overhead_s("separate") == 2 * 600 + 900 + 120 + 30
    # a healthcheck waits at most its start period and one more than its retries, each a timeout and an interval
    assert service.rollout_overhead_s("health") == 600 + 3 * (10 + 5) + 600 + 120 + 30

    tuned = HarborEnvironmentService(
        task_dirs=tasks, provider="process", sandbox_start_budget_s=60, verifier_timeout_multiplier=2.0
    )
    assert tuned.rollout_overhead_s("separate") == 2 * 60 + 2 * 900 + 120 + 30


@pytest.mark.parametrize("budget", [-1, True, float("nan"), "600"])
def test_the_start_budget_is_checked(make_task, budget):
    with pytest.raises(ValueError, match="sandbox_start_budget_s"):
        HarborEnvironmentService(task_dirs=[make_task("t")], sandbox_start_budget_s=budget)


def test_a_task_routed_to_a_service_gets_its_overhead_and_others_get_none(make_task, tmp_path, caplog):
    task_dir = make_task("separate", task_toml=SEPARATE)
    harbor = {"task_dirs": [str(task_dir)], "provider": "process"}

    assert rollout_overhead_s(_routed("separate", harbor)) == 2 * 600 + 900 + 120 + 30
    assert rollout_overhead_s(Task(id="separate")) == 0.0  # no route
    table = tmp_path / "tasks.jsonl"
    table.write_text('{"id": "t1", "goal": "Say hi."}\n')
    assert rollout_overhead_s(_routed("t1", {"path": str(table)}, service="table")) == 0.0  # says nothing
    with caplog.at_level(logging.WARNING, logger="platoon.train.components"):
        assert rollout_overhead_s(_routed("missing", harbor)) == 0.0  # cannot say: the deadline is not the problem
        assert rollout_overhead_s(_routed("missing", harbor)) == 0.0
    assert caplog.text.count("cannot say how long its rollouts take") == 1


def test_a_rollout_workers_alarm_and_an_evaluation_slots_deadline_count_it(make_task):
    task = _routed("separate", {"task_dirs": [str(make_task("separate", task_toml=SEPARATE))], "provider": "process"})
    overhead = 2 * 600 + 900 + 120 + 30
    workflow: Any = SimpleNamespace(config=SimpleNamespace(worker_finalize_grace_seconds=120.0))  # the method's self
    config = RolloutConfig(timeout=1800)

    assert GroupRolloutWorkflow._rollout_budget_seconds(workflow, config, task) == 1800 + overhead + 120
    assert GroupRolloutWorkflow._rollout_budget_seconds(workflow, config, Task(id="plain")) == 1800 + 120
    assert _hard_timeout_seconds({"timeout": 1800}, task) == 1800 + overhead + 180
    assert DefaultEvalGroupWorkflow._subprocess_hard_timeout_seconds(config, task) == 1800 + overhead + 180
    # A rollout without a timeout whose harness has no limit (here, a route without one) keeps the unbounded budget.
    workflow.config.worker_unbounded_rollout_budget_seconds = 7.0
    assert GroupRolloutWorkflow._rollout_budget_seconds(workflow, RolloutConfig(timeout=None), task) == 7.0


def test_a_harbor_service_gives_each_tasks_own_time_limit_for_its_agent(harbor):
    service = HarborEnvironmentService(**harbor)

    assert service.agent_timeout_s("limited") == 600.0  # [agent] timeout_sec 300, times the multiplier
    assert service.agent_timeout_s("open") is None
    assert service.rollout_overhead_s("limited") == service.rollout_overhead_s("open") == OVERHEAD


def test_without_a_timeout_the_harness_runs_until_its_specs_or_its_tasks_limit(harbor):
    # platoon-bash's spec stops it after 1800 s, and the task's own limit after 600 s: the earlier counts
    assert harness_time_limit_s(_harness_routed("limited", harbor), HARNESS_ROLLOUT) == 600.0
    assert harness_time_limit_s(_harness_routed("open", harbor), HARNESS_ROLLOUT) == 1800.0
    # a function harness has no limit of its own: the task's, or none
    noop = {"harness": "platoon-noop"}
    assert harness_time_limit_s(_harness_routed("limited", harbor, **noop), HARNESS_ROLLOUT) == 600.0
    assert harness_time_limit_s(_harness_routed("open", harbor, **noop), HARNESS_ROLLOUT) is None
    # a route without a harness, a task without a route, and a task-like object without misc (a test's double)
    assert harness_time_limit_s(_routed("limited", harbor), HARNESS_ROLLOUT) is None
    assert harness_time_limit_s(Task(id="limited"), HARNESS_ROLLOUT) is None
    double: Any = SimpleNamespace(id="limited", max_steps=None)
    assert harness_time_limit_s(double, HARNESS_ROLLOUT) is None


def test_only_platoons_harness_rollout_is_limited_by_the_harness_and_the_task(harbor):
    """A rollout function of the environment's own runs its harness as it likes: nothing says how long it takes."""

    assert harness_time_limit_s(_harness_routed("limited", harbor, rollout=OWN_ROLLOUT), OWN_ROLLOUT) is None
    assert harness_time_limit_s(_harness_routed("limited", harbor), OWN_ROLLOUT) is None
    assert harness_time_limit_s(_harness_routed("limited", harbor), None) is None
    # A mixture runs each task with the rollout its route names for the phase: harness_rollout to train, its own to
    # evaluate.
    mixed = _harness_routed("limited", harbor, eval_rollout=OWN_ROLLOUT)
    assert harness_time_limit_s(mixed, MIXTURE_ROLLOUT) == 600.0
    assert harness_time_limit_s(mixed, MIXTURE_EVAL_ROLLOUT) is None
    swapped = _harness_routed("limited", harbor, rollout=OWN_ROLLOUT, eval_rollout=HARNESS_ROLLOUT)
    assert harness_time_limit_s(swapped, MIXTURE_ROLLOUT) is None
    assert harness_time_limit_s(swapped, MIXTURE_EVAL_ROLLOUT) == 600.0


def test_the_mixture_rollouts_run_the_rollout_of_the_route_field_they_are_listed_with():
    """MIXTURE_ROLLOUTS names auto's mixture rollouts as rollout workers name them, each with the route field whose
    rollout it runs."""

    assert {validated_ref(auto.mixture_rollout, what="r"), validated_ref(auto.mixture_eval_rollout, what="r")} == set(
        MIXTURE_ROLLOUTS
    )
    route = EnvironmentRoute(name="env", rollout=OWN_ROLLOUT, eval_rollout=f"{__name__}:other_own_rollout")
    task = route.attach(Task(id="t"))
    ran.clear()
    for ref, field in MIXTURE_ROLLOUTS.items():
        asyncio.run(resolve_ref(ref)(task, RolloutConfig()))
        expected = "own" if getattr(route, field) == OWN_ROLLOUT else "other"
        assert ran.pop() == f"{expected}:t"
    assert MIXTURE_ROLLOUTS == {MIXTURE_ROLLOUT: "rollout", MIXTURE_EVAL_ROLLOUT: "eval_rollout"}


def test_a_service_of_ones_own_gives_the_tasks_limit_or_leaves_the_specs(tmp_path, caplog):
    unique = {"run": str(tmp_path)}  # services are cached by their options, and each warns once
    limited, bare, failing = (f"{__name__}:{name}" for name in ("LimitedService", "BareService", "FailingService"))

    def limit(task_id: str, service: str, harness: str = "platoon-bash") -> float | None:
        return harness_time_limit_s(_harness_routed(task_id, unique, service=service, harness=harness), HARNESS_ROLLOUT)

    assert limit("short", limited) == 30.0
    assert limit("endless", limited) == 1800.0  # an infinite limit limits nothing
    assert limit("unset", limited) == 1800.0
    assert limit("short", limited, harness="platoon-noop") == 30.0
    assert limit("unset", limited, harness="platoon-noop") is None
    assert limit("short", bare) == 1800.0  # no agent_timeout_s: the spec's limit alone
    assert limit("short", bare, harness="platoon-noop") is None
    with caplog.at_level(logging.WARNING, logger="platoon.train.components"):
        assert limit("short", failing) == 1800.0  # cannot say: never a reason to cut the rollout short
        assert limit("short", failing) == 1800.0
        assert limit("short", failing, harness="platoon-noop") is None
        missing = str(tmp_path / "missing.toml")  # a harness that does not resolve here: the task's limit only
        assert limit("short", limited, harness=missing) == 30.0
        assert limit("short", limited, harness=missing) == 30.0
    assert caplog.text.count("cannot say its tasks' own time limits") == 1
    assert caplog.text.count("does not resolve for its time limit") == 1
    # A worker's deadline adds what the service says a rollout takes beyond the harness, and the finalize grace.
    workflow = _budget_workflow()
    task = _harness_routed("short", unique, service=limited)
    assert known_rollout_overhead_s(task) == 45.0
    assert workflow._rollout_budget_seconds(RolloutConfig(), task) == 30 + 45 + 120
    assert _hard_timeout_seconds({}, task, HARNESS_ROLLOUT) == 30 + 45 + 180


def test_a_service_that_does_not_say_what_a_rollout_needs_beyond_the_harness_keeps_the_fallback(tmp_path):
    """Opening its session, grading and closing take what they take: a deadline that counted only the harness's limit
    could end a rollout while it is graded.  So a worker keeps its unbounded budget, and an evaluation slot its 900 s,
    or the harness's own limit where that is longer; an explicit timeout counts as before."""

    unique = {"run": str(tmp_path)}
    spec = tmp_path / "short-agent.toml"
    spec.write_text(SHORT_SPEC)
    workflow = _budget_workflow()
    for service in (f"{__name__}:BareService", f"{__name__}:FailingService"):
        long_spec = _harness_routed("short", unique, service=service)  # platoon-bash: 1800 s
        short_spec = _harness_routed("short", unique, service=service, harness=str(spec))
        assert harness_time_limit_s(long_spec, HARNESS_ROLLOUT) == 1800.0
        assert harness_time_limit_s(short_spec, HARNESS_ROLLOUT) == 120.0
        assert known_rollout_overhead_s(long_spec) is None and rollout_overhead_s(long_spec) == 0.0
        assert workflow._rollout_budget_seconds(RolloutConfig(), long_spec) == 7.0
        assert workflow._rollout_budget_seconds(RolloutConfig(), short_spec) == 7.0
        assert _hard_timeout_seconds({}, long_spec, HARNESS_ROLLOUT) == 1800 + 180
        assert _hard_timeout_seconds({}, short_spec, HARNESS_ROLLOUT) == 900 + 180
        assert workflow._rollout_budget_seconds(RolloutConfig(timeout=60), short_spec) == 60 + 120
        assert _hard_timeout_seconds({"timeout": 60}, short_spec, HARNESS_ROLLOUT) == 60 + 180


def test_a_function_harness_that_a_program_of_the_environments_own_runs_keeps_the_fallback(harbor):
    """After the root agent, whose limit bounds its subagents too, platoon's own programs only grade and finalize; a
    program of the environment's own may run more, such as judges, which no limit bounds."""

    assert {resolve_program(name).fn.__module__ for name in PLATOON_PROGRAMS} == {"platoon.programs", "platoon.gym"}
    for program in PLATOON_PROGRAMS:
        ours = _harness_routed("limited", harbor, harness="platoon-noop", program=program)
        assert harness_time_limit_s(ours, HARNESS_ROLLOUT) == 600.0
    own = _harness_routed("limited", harbor, harness="platoon-noop", program=OWN_PROGRAM)
    assert harness_time_limit_s(own, HARNESS_ROLLOUT) is None
    assert _budget_workflow()._rollout_budget_seconds(RolloutConfig(), own) == 7.0
    assert _hard_timeout_seconds({}, own, HARNESS_ROLLOUT) == 900 + OVERHEAD + 180
    assert _budget_workflow()._rollout_budget_seconds(RolloutConfig(timeout=60), own) == 60 + OVERHEAD + 120
    # A TOML spec runs no episode program, whatever its route names: its own limits count.
    assert harness_time_limit_s(_harness_routed("limited", harbor, program=OWN_PROGRAM), HARNESS_ROLLOUT) == 600.0


def test_without_a_timeout_a_workers_and_an_evaluation_slots_deadline_count_the_harness_limit(harbor):
    workflow = _budget_workflow()
    unset = RolloutConfig(timeout=None)
    limited = _harness_routed("limited", harbor)

    assert workflow._rollout_budget_seconds(unset, limited) == 600 + OVERHEAD + 120
    assert workflow._rollout_budget_seconds(unset, _harness_routed("open", harbor)) == 1800 + OVERHEAD + 120
    # An explicit timeout counts as before, also one beyond what the harness may take.
    assert workflow._rollout_budget_seconds(RolloutConfig(timeout=3600), limited) == 3600 + OVERHEAD + 120
    assert workflow._rollout_budget_seconds(RolloutConfig(timeout=60), limited) == 60 + OVERHEAD + 120
    # Nothing limits a function harness on a task without its own limit, nor a rollout function of the env's own.
    noop = _harness_routed("open", harbor, harness="platoon-noop")
    assert workflow._rollout_budget_seconds(unset, noop) == 7.0
    assert _budget_workflow(OWN_ROLLOUT)._rollout_budget_seconds(unset, limited) == 7.0
    assert _budget_workflow(None)._rollout_budget_seconds(unset, limited) == 7.0  # no rollout function named

    # An evaluation slot counts the same limits, else 900 s as before.
    assert _hard_timeout_seconds({"timeout": None}, limited, HARNESS_ROLLOUT) == 600 + OVERHEAD + 180
    assert DefaultEvalGroupWorkflow._subprocess_hard_timeout_seconds(unset, limited, HARNESS_ROLLOUT) == (
        600 + OVERHEAD + 180
    )
    assert _hard_timeout_seconds({"timeout": 3600}, limited, HARNESS_ROLLOUT) == 3600 + OVERHEAD + 180
    assert _hard_timeout_seconds({}, limited, OWN_ROLLOUT) == 900 + OVERHEAD + 180
    assert _hard_timeout_seconds({}, noop, HARNESS_ROLLOUT) == 900 + OVERHEAD + 180
    assert _hard_timeout_seconds({}, limited) == 900 + OVERHEAD + 180  # no rollout function named


def _model_info() -> Any:
    return SimpleNamespace(model_name="m", base_url=None, api_key=None, llm=SimpleNamespace(version=0))


def test_a_runs_worker_requests_carry_each_tasks_deadline(harbor):
    """From the config to the worker request: the rollout and the routes a run builds give each rollout its
    deadline, by the rollout its route names for the workflow's phase."""

    single = SimpleNamespace(
        environments=[{"name": "science", "service": "harbor", "service_kwargs": harbor, "harness": "platoon-bash"}]
    )
    mixture = SimpleNamespace(
        environments=[
            {**single.environments[0], "eval_rollout": OWN_ROLLOUT},
            {"name": "noop", "service": "harbor", "service_kwargs": harbor, "harness": "platoon-noop"},
        ]
    )

    def budget(config, split: str, task_id: str) -> float:
        workflow = GroupRolloutWorkflow(
            rollout_fn=auto.AutoRollout.from_config(config, split),
            get_task_fn=auto.AutoTaskLoader.from_config(config, split),
            config=WorkflowConfig(group_size=1, worker_unbounded_rollout_budget_seconds=7.0),
            model_info=_model_info(),
        )
        return workflow._build_worker_request({"task_id": task_id}, 0).rollout_budget_seconds

    assert budget(single, "train", "limited") == budget(single, "eval", "limited") == 600 + OVERHEAD + 120
    assert budget(single, "train", "open") == 1800 + OVERHEAD + 120
    assert budget(mixture, "train", "science::limited") == 600 + OVERHEAD + 120
    assert budget(mixture, "eval", "science::limited") == 7.0  # evaluated by a rollout function of its own
    assert budget(mixture, "train", "noop::limited") == 600 + OVERHEAD + 120
    assert budget(mixture, "train", "noop::open") == 7.0


def test_an_evaluation_slot_and_its_parent_count_the_same_deadline(harbor, monkeypatch, tmp_path):
    calls = []

    async def run(self, timeout_s, fn, *args, grace_s=0.0):
        calls.append((timeout_s, args))
        raise TimeoutError("not run")

    monkeypatch.setattr(_ProcessSlots, "run", run)
    config = SimpleNamespace(
        environments=[{"name": "science", "service": "harbor", "service_kwargs": harbor, "harness": "platoon-bash"}]
    )
    workflow = DefaultEvalGroupWorkflow(
        rollout_fn=auto.AutoRollout.from_config(config, "eval"),
        get_task_fn=auto.AutoTaskLoader.from_config(config, "eval"),
        config=EvalWorkflowConfig(use_subprocesses=True, rollout_config=RolloutConfig()),
        model_name="m",
    )

    asyncio.run(workflow.arun_rollout({"task_id": "limited"}, 0, str(tmp_path)))

    [(timeout_s, (rollout_ref, task, rollout_config, _policy))] = calls
    assert rollout_ref == HARNESS_ROLLOUT
    assert rollout_config["timeout"] is None
    assert timeout_s == _hard_timeout_seconds(rollout_config, task, rollout_ref) == 600 + OVERHEAD + 180
