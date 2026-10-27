"""The extension guides in docs/extending stay true: their code runs, their configs are valid environment entries,
the outputs they show are what the code prints, and the claims they make about platoon hold.

Snippets are found by their first line: a file's snippet starts with a comment that names it
(``# my_rewards/partial.py``), so editing a snippet in a guide changes what these tests run.  Each test writes the
snippets it needs into a temporary package directory on ``sys.path``.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import importlib
import inspect
import io
import json
import os
import re
import sys
import tomllib
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

import platoon.hooks
from platoon.train.components import normalize_environment_configs

REPO = Path(__file__).resolve().parents[2]
GUIDES = REPO / "docs" / "extending"
_FENCE = re.compile(r"^```(?P<lang>[\w-]*)\n(?P<body>.*?)^```$", re.S | re.M)


def blocks(page: str, language: str | None = None) -> list[str]:
    """The fenced code blocks of a guide, optionally only those of one language."""

    text = (GUIDES / page).read_text(encoding="utf-8")
    return [match["body"] for match in _FENCE.finditer(text) if language is None or match["lang"] == language]


def block(page: str, first_line: str) -> str:
    """The one code block of ``page`` whose first line is ``first_line``."""

    found = [code for code in blocks(page) if code.splitlines()[0] == first_line]
    assert len(found) == 1, f"docs/extending/{page} has {len(found)} code blocks starting with {first_line!r}"
    return found[0]


def shown_output(page: str, command: str) -> str:
    """The ``text`` block that follows the ``bash`` block holding ``command``: the output the guide shows."""

    text = (GUIDES / page).read_text(encoding="utf-8")
    fences = list(_FENCE.finditer(text))
    for index, match in enumerate(fences[:-1]):
        if match["lang"] == "bash" and command in match["body"]:
            following = fences[index + 1]
            assert following["lang"] == "text", f"no output block after {command!r} in {page}"
            return following["body"]
    raise AssertionError(f"docs/extending/{page} has no bash block with {command!r}")


def environment_entries(page: str) -> list[Any]:
    """Every ``environments`` entry of the guide's YAML blocks, validated as platoon validates a config's.  Entries
    that name a named environment (``env:``) are left out: it would have to be installed to resolve."""

    entries = []
    for code in blocks(page, "yaml"):
        environments = (yaml.safe_load(code) or {}).get("environments")
        if environments:
            entries += normalize_environment_configs([entry for entry in environments if "env" not in entry])
    return entries


@pytest.fixture
def snippets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[[str, str], Path]]:
    """``write(page, first_line)`` writes the guide's snippet to the path its first line names, under a temporary
    directory on ``sys.path``; the modules it imported are dropped afterwards."""

    monkeypatch.syspath_prepend(str(tmp_path))
    written: set[str] = set()

    def write(page: str, first_line: str) -> Path:
        relative = first_line.removeprefix("# ").strip()
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.parent != tmp_path and not (target.parent / "__init__.py").exists():
            (target.parent / "__init__.py").write_text("")
        target.write_text(block(page, first_line))
        written.add(Path(relative).parts[0].removesuffix(".py"))
        return target

    yield write
    _unregister(written)
    importlib.invalidate_caches()
    for name in [name for name in sys.modules if name.split(".")[0] in written]:
        del sys.modules[name]


def _unregister(packages: set[str]) -> None:
    """Drop the function harnesses, programs, task samplers and group admission rules that the snippet ``packages``
    registered, and nothing else."""

    from platoon import programs
    from platoon.harness import function
    from platoon.registry import get_registry
    from platoon.train.admission_api import REGISTRY_KIND as ADMISSION_KIND
    from platoon.train.sampler_api import REGISTRY_KIND

    def ours(fn: Any) -> bool:
        return str(getattr(fn, "__module__", "")).split(".")[0] in packages

    with function._lock:
        for ref in [ref for ref, record in function._function_harnesses.items() if ours(record.fn)]:
            del function._function_harnesses[ref]
    with programs._lock:
        for name in [name for name, fn in programs._programs.items() if ours(fn)]:
            del programs._programs[name]
    for kind in (REGISTRY_KIND, ADMISSION_KIND):
        registry = get_registry(kind)
        registry._items = {name: item for name, item in registry._items.items() if not ours(item.value)}


def run_test_functions(code: str, name: str) -> list[str]:
    """Run every ``test_*`` function a test snippet defines; returns their names."""

    namespace: dict[str, Any] = {"__name__": name}
    exec(compile(code, name, "exec"), namespace)
    tests = [value for key, value in namespace.items() if key.startswith("test_") and callable(value)]
    for test in tests:
        test()
    return [test.__name__ for test in tests]


def run_script(code: str, name: str) -> str:
    """Run a script snippet and return what it printed."""

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, name, "exec"), {"__name__": "__main__"})
    return output.getvalue()


class FakeEpisode:
    """An ``EpisodeContext`` stand-in that records what a program does, for programs that compose ``single``."""

    def __init__(self, *, extra: dict[str, Any] | None = None, propagate_root_success: bool = False) -> None:
        self.config = SimpleNamespace(
            extra=dict(extra or {}),
            propagate_root_success=propagate_root_success,
            skip_subagent_reward_computation=False,
        )
        self.behavior_judge = None
        self.judge_at_run = "not run"
        self.finalizers: list[Any] = []

    async def run(self, harness: str | None = None, *, goal: str | None = None, options: Any = None) -> Any:
        import platoon as pt

        self.judge_at_run = self.behavior_judge
        return pt.AgentResult("completed", final_message="done")

    async def grade(self) -> Any:
        import platoon as pt

        return pt.Outcome(status="scored", rewards={"reward": 1.0})

    def add_finalizer(self, fn: Any) -> None:
        if fn not in self.finalizers:
            self.finalizers.append(fn)


# --------------------------------------------------------------------------- every guide


PAGES = sorted(path.name for path in GUIDES.glob("*.md"))


@pytest.mark.parametrize("page", PAGES)
def test_every_snippet_parses(page: str) -> None:
    for language, parse in (("python", None), ("yaml", yaml.safe_load), ("toml", tomllib.loads), ("json", json.loads)):
        for code in blocks(page, language):
            if parse is None:
                compile(code, f"docs/extending/{page}", "exec")
            else:
                parse(code)


@pytest.mark.parametrize("page", PAGES)
def test_every_environment_entry_is_valid(page: str) -> None:
    environment_entries(page)  # raises on an unknown key or a bad value, as loading a config does


def test_the_index_links_every_guide_of_the_section() -> None:
    index = (GUIDES / "index.md").read_text(encoding="utf-8")
    for page in PAGES:
        if page != "index.md":
            assert f"]({page}" in index, f"docs/extending/index.md does not link {page}"


def test_the_index_names_the_built_in_services_and_providers() -> None:
    from platoon.envs.service import _BUILTIN_SERVICES
    from platoon.sandbox import _BUILTIN_PROVIDERS

    index = (GUIDES / "index.md").read_text(encoding="utf-8")
    for name in _BUILTIN_SERVICES:
        assert f"service: {name}" in index
    providers = (REPO / "docs" / "how-to" / "sandboxes-and-harbor.md").read_text(encoding="utf-8")
    sandbox_guide = (GUIDES / "sandbox-providers.md").read_text(encoding="utf-8")
    for name in _BUILTIN_PROVIDERS:
        assert f"`{name}`" in providers and f"`{name}`" in sandbox_guide


# --------------------------------------------------------------------------- rewards.md


def test_the_reward_processor_and_the_finalizer_pass_their_tests(snippets) -> None:
    snippets("rewards.md", "# my_rewards/partial.py")
    snippets("rewards.md", "# my_rewards/step_cost.py")

    ran = run_test_functions(block("rewards.md", "# tests/test_my_rewards.py"), "test_my_rewards")

    assert shown_output("rewards.md", "tests/test_my_rewards.py").startswith(f"{'.' * len(ran)} ")
    assert f"{len(ran)} passed" in shown_output("rewards.md", "tests/test_my_rewards.py")


def test_the_step_cost_program_runs_its_finalizer_after_the_root_success_one(snippets):
    from platoon.programs import get_program
    from platoon.utils.subagent_rewards import propagate_root_success

    snippets("rewards.md", "# my_rewards/step_cost.py")
    module = importlib.import_module("my_rewards.step_cost")
    episode = FakeEpisode(extra={"step_cost": 0.02}, propagate_root_success=True)

    outcome = asyncio.run(get_program("my-step-cost")(episode))

    assert outcome.reward == 1.0
    first, second = episode.finalizers
    assert first is propagate_root_success
    assert isinstance(second, module.StepCost) and second.cost == 0.02
    [config] = [code for code in blocks("rewards.md", "yaml") if "my-step-cost" in code]
    assert yaml.safe_load(config)["train"]["workflow_config"]["rollout_config"]["extra"] == {"step_cost": 0.02}


def test_the_step_cost_leaves_untrainable_and_verifier_trajectories_alone(snippets) -> None:
    from platoon.trajectory import Trajectory, TrajectoryCollection, TrajectoryStep
    from platoon.utils.trajectory_status import TRAJECTORY_TIMED_OUT_MISC_KEY

    snippets("rewards.md", "# my_rewards/step_cost.py")
    module = importlib.import_module("my_rewards.step_cost")
    steps = [TrajectoryStep(), TrajectoryStep()]
    collection = TrajectoryCollection(
        trajectories={
            "root": Trajectory(id="root", reward=1.0, steps=list(steps), misc={TRAJECTORY_TIMED_OUT_MISC_KEY: True}),
            "verifier": Trajectory(id="verifier", reward=1.0, steps=list(steps), misc={"exclude_from_training": True}),
        }
    )

    module.StepCost(cost=0.1)(collection)

    assert [trajectory.reward for trajectory in collection.trajectories.values()] == [1.0, 1.0]


def test_reward_components_are_logged_under_the_names_the_guide_gives() -> None:
    from platoon.utils.rollout_stats import summarize_rollout_group

    stats = summarize_rollout_group({"task_reward": [0.4], "reward/close": [0.8], "root_reward/close": [0.8]})
    metrics = stats.to_metrics(prefix="batch")

    assert "batch/reward/close/avg" in metrics  # train/batch/reward/close/avg once the trainer's scope prefixes it
    text = (GUIDES / "rewards.md").read_text(encoding="utf-8")
    assert "`train/batch/reward/close/avg`" in text and "`root_<name>`" in text


# --------------------------------------------------------------------------- judges-and-verifiers.md


def test_the_judges_pass_their_tests_and_their_verdicts_are_eligible(snippets) -> None:
    from platoon.harness.verify import normalize_behavior_judgment, normalize_parent_usage_judgment
    from platoon.trajectory import Trajectory

    snippets("judges-and-verifiers.md", "# my_judges/behavior.py")
    snippets("judges-and-verifiers.md", "# my_judges/usage.py")

    ran = run_test_functions(block("judges-and-verifiers.md", "# tests/test_my_judges.py"), "test_my_judges")
    assert f"{len(ran)} passed" in shown_output("judges-and-verifiers.md", "tests/test_my_judges.py")

    behavior = importlib.import_module("my_judges.behavior").UsedTools()
    used_tools = Trajectory(id="child", misc={"tool_calls": [{"name": "read"}]})
    for trajectory, status in ((used_tools, "pass"), (Trajectory(id="idle"), "fail")):
        verdict = normalize_behavior_judgment(asyncio.run(behavior.judge(goal="g", trajectory=trajectory)))
        assert (verdict["status"], verdict["training_eligible"]) == (status, True)
    usage = importlib.import_module("my_judges.usage").QuotesTheChild()
    for child_answer, status, eligible in (
        ("42", "used", True),
        ("7", "not_used", True),
        ("", "insufficient_evidence", False),
    ):
        verdict = normalize_parent_usage_judgment(
            asyncio.run(
                usage.judge_parent_usage(
                    parent_goal="p",
                    parent_trajectory=Trajectory(id="root", finish_message="The total is 42."),
                    child_goal="c",
                    child_trajectory=Trajectory(id="child", finish_message=child_answer),
                )
            )
        )
        assert (verdict["status"], verdict["training_eligible"]) == (status, eligible)


def test_the_audited_program_sets_its_judge_before_the_root_runs(snippets) -> None:
    from platoon.programs import get_program

    snippets("judges-and-verifiers.md", "# my_judges/behavior.py")
    module = importlib.import_module("my_judges.behavior")
    episode = FakeEpisode()

    asyncio.run(get_program("my-audited")(episode))

    assert isinstance(episode.judge_at_run, module.UsedTools)


def test_the_usage_gate_drops_only_the_unused_verified_subagent(snippets) -> None:
    from platoon.programs import get_program
    from platoon.trajectory import ParentInfo, Trajectory, TrajectoryCollection, TrajectoryStep

    snippets("judges-and-verifiers.md", "# my_judges/usage.py")
    importlib.import_module("my_judges.usage")
    episode = FakeEpisode()
    asyncio.run(get_program("my-usage-gated")(episode))
    [finalizer] = episode.finalizers
    assert isinstance(finalizer, functools.partial)

    verified = {"status": "verified", "score": 1.0, "training_eligible": True}

    def child(name: str, answer: str) -> Trajectory:
        return Trajectory(
            id=name,
            parent_info=ParentInfo(id="root", fork_step=0),
            steps=[TrajectoryStep()],
            reward=1.0,
            finish_message=answer,
            misc={"subagent_reward_judgment": dict(verified)},
        )

    collection = TrajectoryCollection(
        trajectories={
            "root": Trajectory(id="root", finish_message="The total is 42."),
            "used": child("used", "42"),
            "ignored": child("ignored", "7"),
        }
    )
    asyncio.run(finalizer(collection))

    assert not collection.trajectories["used"].misc.get("exclude_from_policy_training")
    assert collection.trajectories["ignored"].misc["exclude_from_policy_training"] is True
    assert collection.trajectories["ignored"].reward == 1.0  # the reward stays for statistics and baselines
    assert collection.trajectories["ignored"].steps[-1].misc["reward_misc"]["reward/subagent_parent_usage_gate"] == 0.0


def test_the_verdict_schema_the_guide_shows_is_an_eligible_verdict() -> None:
    from platoon.harness.verify import format_verifier_goal, normalize_judgment

    [schema] = blocks("judges-and-verifiers.md", "json")
    verdict = normalize_judgment(
        schema, verifier_trajectory_id="v", verifier_error_message=None, verifier_finished=True
    )
    assert verdict["training_eligible"] is True
    goal = format_verifier_goal(child_goal="g", child_trajectory_id="c", child_message="m", child_error_message=None)
    for key in json.loads(schema):
        assert f'"{key}"' in goal  # the guide shows the keys the verifier is asked for
    for status, score, eligible in (("partial", 1.0, False), ("partial", 0.5, True), ("failed", 0.0, True)):
        message = json.dumps({"status": status, "score": score})
        normalized = normalize_judgment(
            message, verifier_trajectory_id="v", verifier_error_message=None, verifier_finished=True
        )
        assert normalized["training_eligible"] is eligible


# --------------------------------------------------------------------------- environment-services.md


def test_the_word_service_prints_what_the_guide_shows(snippets) -> None:
    snippets("environment-services.md", "# my_services/words.py")

    printed = run_script(block("environment-services.md", "# check_words.py"), "check_words.py")

    assert printed == shown_output("environment-services.md", "check_words.py")


def test_the_word_service_grades_and_forks_as_the_guide_says(snippets) -> None:
    from platoon.envs import get_service

    snippets("environment-services.md", "# my_services/words.py")
    [entry] = environment_entries("environment-services.md")
    service = get_service(entry.service, **entry.service_kwargs)

    async def play() -> tuple[Any, ...]:
        session = await service.open(await service.get_task("test:0"), rollout_id="test")
        unsubmitted = await session.grade()
        with pytest.raises(ValueError, match="fork only 'fresh'"):
            await session.fork("shared")
        child = await session.fork("fresh")
        tools = await session.tools()
        await session.close()
        await session.close()  # idempotent
        return unsubmitted, child, tools

    unsubmitted, child, tools = asyncio.run(play())
    assert (unsubmitted.status, unsubmitted.reward, unsubmitted.reason) == ("scored", 0.0, "no_submit")
    assert child.instruction == "Call submit with the word 'cloud'."
    assert [tool.terminal for tool in tools] == [True]
    assert entry.agents.fork == "fresh"


# --------------------------------------------------------------------------- sandbox-providers.md


def test_the_audited_provider_grades_the_harbor_task_as_the_guide_shows(snippets, tmp_path, monkeypatch) -> None:
    snippets("sandbox-providers.md", "# my_sandboxes/audited.py")
    workdir = tmp_path / "run"
    workdir.mkdir()
    (workdir / "examples").symlink_to(REPO / "examples")  # the snippet runs from the repository root
    monkeypatch.chdir(workdir)
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))  # the process provider keeps its sandboxes there
    (tmp_path / "tmp").mkdir()
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", None)

    printed = run_script(block("sandbox-providers.md", "# check_provider.py"), "check_provider.py")

    assert printed == shown_output("sandbox-providers.md", "check_provider.py")
    records = [json.loads(line) for line in (workdir / "commands.jsonl").read_text().splitlines()]
    assert "echo 'hello from platoon' > greeting.txt" in [record["cmd"] for record in records]


def test_the_audited_provider_keeps_the_inner_provider_s_capabilities(snippets, tmp_path) -> None:
    from platoon.sandbox import SandboxProvider, get_provider

    snippets("sandbox-providers.md", "# my_sandboxes/audited.py")
    provider = get_provider("my_sandboxes.audited:AuditedProvider", inner="process", log_path=str(tmp_path / "log"))

    assert isinstance(provider, SandboxProvider)
    assert provider.name == "audited-process"
    assert provider.capabilities == frozenset({"snapshot", "routes", "host_loopback"})  # the process provider's, but
    # stacks: the wrapper has no start_stack
    [entry] = environment_entries("sandbox-providers.md")
    assert entry.service == "harbor" and entry.service_kwargs["provider"] == "my_sandboxes.audited:AuditedProvider"


def test_the_capabilities_table_lists_the_capabilities_providers_declare() -> None:
    from platoon.sandbox import base

    declared = set(re.search(r"subset of \{(.*?)\}", base.__doc__)[1].replace('"', "").split(", "))
    assert declared == base.CAPABILITIES
    text = (GUIDES / "sandbox-providers.md").read_text(encoding="utf-8")
    table = text[text.index("| Capability |") :]
    assert {re.match(r"\| `(\w+)` \|", row)[1] for row in table.splitlines()[2 : 2 + len(declared)]} == declared


# --------------------------------------------------------------------------- renderers.md


def test_the_renderer_registers_and_takes_its_settings(snippets) -> None:
    from tinker_cookbook.renderers import get_renderer, unregister_renderer

    snippets("renderers.md", "# my_renderers/__init__.py")
    try:
        module = importlib.import_module("my_renderers")
        renderer = get_renderer("my_model", SimpleNamespace(name_or_path="my-org/my-model"))
        assert isinstance(renderer, module.MyModelRenderer) and renderer.enable_thinking is True
    finally:
        unregister_renderer("my_model")
    [config] = [code for code in blocks("renderers.md", "yaml") if "renderer_name: my_model" in code]
    from platoon.train.config_defs import TrainConfig

    fields = set(TrainConfig.__dataclass_fields__)
    assert set(yaml.safe_load(config)["train"]) <= fields
    assert yaml.safe_load(config)["environments"][0]["package"] == "my_renderers"


# --------------------------------------------------------------------------- hooks-and-metrics.md


def test_the_hooks_register_and_pass_their_tests(snippets, monkeypatch) -> None:
    monkeypatch.setattr(platoon.hooks, "_HOOKS", {})
    snippets("hooks-and-metrics.md", "# my_hooks/counters.py")

    ran = run_test_functions(block("hooks-and-metrics.md", "# tests/test_my_hooks.py"), "test_my_hooks")

    assert f"{len(ran)} passed" in shown_output("hooks-and-metrics.md", "tests/test_my_hooks.py")
    from platoon.train.workers.telemetry import local_telemetry

    assert local_telemetry() == {"my_harness_retries": 0.0}


def test_the_retrying_harness_marks_and_counts_its_retry(snippets, monkeypatch) -> None:
    import platoon as pt

    monkeypatch.setattr(platoon.hooks, "_HOOKS", {})
    snippets("hooks-and-metrics.md", "# my_hooks/counters.py")
    module = importlib.import_module("my_hooks.counters")
    replies = iter(["", "42"])
    annotations: list[dict[str, Any]] = []

    class StandIn:
        instruction = "What is 6 * 7?"

        async def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
            return SimpleNamespace(text=next(replies))

        def annotate_step(self, **misc: Any) -> None:
            annotations.append(misc)

    result = asyncio.run(module.retrying(StandIn()))

    assert (result.status, result.final_message) == ("completed", "42")
    assert annotations == [{"my_failed": True}] and module.COUNTS == {"my_harness_retries": 1.0}
    assert pt.get_harness("my-retrying").ref == "my-retrying@1"


def test_the_hook_table_lists_the_hooks_core_calls() -> None:
    names = set(re.findall(r"^``([\w.]+)``$", platoon.hooks.__doc__, re.M))
    text = (GUIDES / "hooks-and-metrics.md").read_text(encoding="utf-8")
    table = text[text.index("| Hook |") :]
    assert {re.match(r"\| `([\w.]+)` \|", row)[1] for row in table.splitlines()[2:6]} == names


def test_the_tool_call_parser_interface_the_renderer_guide_names_exists() -> None:
    from platoon.train.tinker.renderers import tool_parsers

    assert {"ToolCallParser", "ParseResult"} <= set(dir(tool_parsers))
    text = (GUIDES / "renderers.md").read_text(encoding="utf-8")
    assert "`parse(text, *, tools=None, in_reasoning=False)`" in text
    parameters = inspect.signature(tool_parsers.ToolCallParser.parse).parameters
    assert {"text", "tools", "in_reasoning"} <= set(parameters)


# --------------------------------------------------------------------------- task-samplers.md


def test_the_worked_task_sampler_passes_the_conformance_check(snippets, monkeypatch, tmp_path) -> None:
    """The guide's ``SkipSolved``, written out as its package, passes ``check_task_sampler`` at the entry level, by
    ``module:Class`` (so that its decisions are replayed in processes with other hash seeds too) and by the name its
    decorator registers; and the guide's test of it runs."""

    from platoon.testing import check_task_sampler
    from platoon.train.sampler_api import resolve_task_sampler

    snippets("task-samplers.md", "# acme_curricula/samplers.py")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), *sys.path]))
    check_task_sampler("acme_curricula.samplers:SkipSolved", options={"keep": 0.5}, levels=("entry",))
    importlib.import_module("acme_curricula.samplers")
    assert resolve_task_sampler("acme-skip-solved").__name__ == "SkipSolved"
    check_task_sampler("acme-skip-solved", levels=("entry", "run"))
    test = next(code for code in blocks("task-samplers.md", "python") if code.startswith("from platoon.testing"))
    assert run_test_functions(test, "test_skip_solved") == ["test_skip_solved_keeps_the_contract"]


# --------------------------------------------------------------------------- group-admission.md


def test_the_worked_group_admission_rule_passes_the_conformance_check(snippets, monkeypatch, tmp_path) -> None:
    """The guide's ``Quota``, written out as its package, passes ``check_group_admission`` by ``module:Class`` (so that
    its answers are replayed in processes with other hash seeds too) and by the name its decorator registers; and the
    guide's test of it runs."""

    from platoon.testing import check_group_admission
    from platoon.train.admission_api import resolve_group_admission

    snippets("group-admission.md", "# acme_rules/quota.py")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), *sys.path]))
    check_group_admission("acme_rules.quota:Quota", options={"limit": 4})
    importlib.import_module("acme_rules.quota")
    assert resolve_group_admission("acme-quota").__name__ == "Quota"
    check_group_admission("acme-quota")
    test = next(code for code in blocks("group-admission.md", "python") if code.startswith("from platoon.testing"))
    assert run_test_functions(test, "test_quota") == ["test_the_quota_keeps_the_contract"]


def _guide_tasks(config: Any, split: str) -> list[dict[str, Any]]:
    """The tasks of the group admission guide's entries, as their dataset loader and rollout."""

    return [{"task_id": f"t{index}"} for index in range(6)]


def test_the_group_admission_guide_s_config_resolves_as_platoon_train_resolves_it(snippets, tmp_path) -> None:
    """The guide's config, with its quota package written out (and each entry given tasks), loads and resolves as
    ``platoon train`` resolves it: the run's ``informative`` decides ``easy``'s groups, ``hard``'s own ``acme-quota``
    resolves from the entry's package, and both rules' options and checks pass."""

    from platoon.extensions import run_components
    from platoon.train.config_defs import PlatoonTinkerRLTrainerConfig
    from platoon.utils.config import resolve_config

    snippets("group-admission.md", "# acme_rules/quota.py")
    [guide] = [code for code in blocks("group-admission.md", "yaml") if "acme-quota" in code]
    data = yaml.safe_load(guide)
    for entry in data["environments"]:
        entry.update({"dataset_loader": f"{__name__}:_guide_tasks", "rollout": f"{__name__}:_guide_tasks"})
    data["train"].update({"model_name": "m", "renderer_name": "r", "max_training_steps": 3})
    data.update({"eval": {"every": 0}, "watchdog": {"enabled": False}, "log_path": str(tmp_path / "logs")})
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(data))
    config = resolve_config(path, [], PlatoonTinkerRLTrainerConfig).config
    records = [record for record in run_components(config) if record["kind"] == "group_admission"]
    assert sorted((record["ref"], record["level"], record["environments"]) for record in records) == [
        ("acme-quota", "entry", ["hard"]),
        ("informative", "run", ["easy"]),
    ]


# --------------------------------------------------------------------------- rollout-leases.md


def test_the_worked_workflow_runs_each_attempt_on_a_permit_of_its_own(snippets) -> None:
    """The guide's ``Attempts``, written out as its package, its attempts ending one at a time: under a lease that
    grants its permits as they free, each attempt's permit comes back as the attempt ends; outside a training draw the
    same workflow runs as it is."""

    from platoon.train.dispatch.permits import PermitPool
    from platoon.train.leases import holding

    snippets("rollout-leases.md", "# acme_workflows/attempts.py")
    attempts = importlib.import_module("acme_workflows.attempts")
    assert attempts.Attempts.runs_rollouts_on_lease is True

    async def run() -> None:
        gates = [asyncio.Event() for _ in range(3)]

        class Gated(attempts.Attempts):
            async def run_attempt(self, row: dict, index: int) -> list[str]:
                await gates[index].wait()
                return [f"{row['task_id']}:{index}"]

        workflow = Gated(config=SimpleNamespace(group_size=3), rollout_fn=None, model_info=None)
        pool = PermitPool(3)
        lease = pool.reserve(3, rolling=True, active=True)
        with holding(lease):
            episode = asyncio.ensure_future(workflow.arun_episode({"task_id": "t"}))
        for index, gate in enumerate(gates):
            for _ in range(5):
                await asyncio.sleep(0)
            assert pool.free == index  # the attempts still running hold theirs
            gate.set()
        assert await episode == ["t:0", "t:1", "t:2"] and pool.free == 3
        lease()
        assert await workflow.arun_episode({"task_id": "u"}) == ["u:0", "u:1", "u:2"]  # the no-op lease

    asyncio.run(run())


def test_the_lease_guide_s_table_of_subclasses_says_which_keep_rolling_starts() -> None:
    """Each method the guide's table names: a ``GroupRolloutWorkflow`` subclass that overrides one of the first row's
    keeps ``runs_rollouts_on_lease``, one that overrides one of the second row's reads as False unless its own class
    body sets it."""

    from platoon.train.workflow.group import GroupRolloutWorkflow

    text = (GUIDES / "rollout-leases.md").read_text(encoding="utf-8")
    rows = [line for line in text.splitlines() if line.startswith("| `")]
    assert len(rows) == 2
    keep, lose = (re.findall(r"`(\w+)`", row.split("|")[1]) for row in rows)
    assert {"compute_baselines", "arun_episode_single"} <= set(keep) and "_run_group_rollouts" in lose

    def subclass(name: str, *, declared: bool = False) -> type[GroupRolloutWorkflow]:
        body: dict[str, Any] = {name: getattr(GroupRolloutWorkflow, name)}
        if declared:
            body["runs_rollouts_on_lease"] = True
        return type(f"Overrides{name}", (GroupRolloutWorkflow,), body)

    assert all(subclass(name).runs_rollouts_on_lease for name in keep)
    assert not any(subclass(name).runs_rollouts_on_lease for name in lose)
    assert all(subclass(name, declared=True).runs_rollouts_on_lease for name in lose)
