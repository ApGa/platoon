"""The sft trainer (``trainer: sft``, :mod:`platoon.train.trainer.sft`): it keeps RFC 0021's lifecycle, walks its
conversations epoch by epoch and resumes exactly where a checkpoint left them, prunes its recovery checkpoints with
none kept for a sampler, and its config (``trainer_options``, the ``train:`` keys only rl reads) is checked when it
loads.  The trainer runs on platoon.testing's fake Tinker service, rendering with the toy renderer."""

from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import platoon.experimental as px
from platoon import cli
from platoon.testing import check_trainer, fake_tinker_service
from platoon.testing import trainers as checks
from platoon.train.config_defs import PlatoonTinkerRLTrainerConfig
from platoon.train.policy.spec import policy_specs
from platoon.train.supervised.cursor import DatasetCursor, last_step, steps_per_epoch
from platoon.train.supervised.options import SupervisedOptions
from platoon.train.trainer import registry
from platoon.train.trainer.api import TrainerSetup
from platoon.train.trainer.sft import SupervisedTrainer
from platoon.utils.config import ConfigError, build_config, resolve_config

ROWS = 10


@pytest.fixture(autouse=True)
def _datasets_cache(tmp_path, monkeypatch):
    """``datasets.load_dataset`` caches what it loads: under the test's directory, not the user's cache."""

    import datasets.config

    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "datasets-cache")


@pytest.fixture
def data(tmp_path) -> Path:
    """A dataset directory with one alpaca dataset, ``rows``, of :data:`ROWS` conversations ``q0``, ``q1``, ..."""

    directory = tmp_path / "data"
    directory.mkdir()
    (directory / "dataset_info.json").write_text(json.dumps({"rows": {"file_name": "rows.json"}}))
    rows = [{"instruction": f"q{index}", "input": "", "output": f"a{index}"} for index in range(ROWS)]
    (directory / "rows.json").write_text(json.dumps(rows))
    return directory


def sft_settings(root: Path, data: Path, **options: Any) -> dict[str, Any]:
    return {
        "trainer": "sft",
        "trainer_options": {"dataset": "rows", "dataset_dir": str(data), "batch_size": 3, "num_epochs": 2, **options},
        "train": {"model_name": "fake-model", "renderer_name": "fake-renderer"},
        "watchdog": {"enabled": False},
        "stats": {"wandb": {"mode": "disabled"}},
        "log_path": str(root / "logs"),
    }


def train(config: PlatoonTinkerRLTrainerConfig, *, stop_at: int | None = None) -> checks.FakePolicyService:
    """One start of the sft trainer on ``config``, on a fake service that asks it to stop in its ``stop_at``-th
    optimizer step; the service, with what it was sent."""

    setup = TrainerSetup(config=config, policy_specs=policy_specs(config), options=registry.options_of(config))
    with fake_tinker_service(stop_at=stop_at) as service:
        asyncio.run(checks._run(SupervisedTrainer.from_setup(setup)))
    return service


def alpaca(tmp_path: Path, rows: list[dict[str, str]]) -> Path:
    """A dataset directory with one alpaca dataset, ``rows``, of ``rows``, whose ``system`` column is the system
    prompt."""

    directory = tmp_path / "rows-data"
    directory.mkdir()
    columns = {"prompt": "instruction", "query": "input", "response": "output", "system": "system"}
    (directory / "dataset_info.json").write_text(json.dumps({"rows": {"file_name": "rows.json", "columns": columns}}))
    (directory / "rows.json").write_text(json.dumps(rows))
    return directory


def asserting_renderer(fails: Any) -> Any:
    """The toy renderer, which fails an assertion of its own on the messages ``fails`` picks, as a renderer may."""

    from platoon.testing import toy_renderer

    renderer: Any = toy_renderer()
    build = renderer.build_supervised_example

    def build_supervised_example(messages: Any, train_on_what: Any) -> Any:
        if fails(messages):
            raise AssertionError("expected a system message first")
        return build(messages, train_on_what)

    renderer.build_supervised_example = build_supervised_example
    return renderer


def logged_steps(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """The metrics each step logs to the run's stats logger, as they come."""

    from platoon.utils.stats_logger import StatsLogger

    logged: list[dict[str, Any]] = []
    monkeypatch.setattr(StatsLogger, "log", lambda self, step, stats, **kwargs: logged.append(dict(stats)))
    return logged


def prompts(service: checks.FakePolicyService) -> list[str]:
    """The conversations the service trained on, by their prompt, in order."""

    from platoon.testing import toy_tokenizer

    texts = [str(toy_tokenizer().decode(datum.model_input.to_ints())) for datum in service.datums]
    return [match.group(1) for text in texts if (match := re.search(r"user\n(q\d+)<", text))]


def rows(root: Path) -> list[dict[str, Any]]:
    path = root / "logs" / "platoon_tinker" / "run" / "checkpoints.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


# --- the lifecycle and resuming --------------------------------------------------------------------------------------


def test_the_sft_trainer_keeps_the_lifecycle(data):
    check_trainer(
        SupervisedTrainer,
        options_for_steps=lambda steps: SupervisedOptions(
            dataset="rows", dataset_dir=str(data), batch_size=4, max_training_steps=steps
        ),
    )


def test_a_resume_mid_epoch_trains_every_conversation_once_an_epoch(tmp_path, data):
    config = build_config(PlatoonTinkerRLTrainerConfig, sft_settings(tmp_path, data))  # 4 steps an epoch: 3, 3, 3, 1

    stopped = train(config, stop_at=6)  # a stop request in step 6, the second of epoch 1
    resumed = train(config)

    assert [(row["name"], row["batch"], row["sft"]["epoch"], row["sft"]["position"]) for row in rows(tmp_path)] == [
        ("recovery-000000", 0, 0, 0),
        ("000006_interrupted", 6, 1, 6),
        ("recovery-000006", 6, 1, 6),
        ("final", 8, 2, 0),
    ]
    first, second = prompts(stopped), prompts(resumed)
    assert len(first) == 16 and len(second) == 4
    assert Counter(first[:ROWS]) == Counter(f"q{index}" for index in range(ROWS))  # epoch 0
    assert Counter(first[ROWS:] + second) == Counter(f"q{index}" for index in range(ROWS))  # epoch 1, across starts
    assert first[ROWS:] + second != first[:ROWS]  # each epoch has its own order
    assert [row["optimizer_steps"] for row in rows(tmp_path)] == [0, 6, 6, 8]


def test_a_resume_with_another_batch_size_still_trains_each_conversation_once_an_epoch(tmp_path, data):
    stopped = train(build_config(PlatoonTinkerRLTrainerConfig, sft_settings(tmp_path, data)), stop_at=2)
    resumed = train(build_config(PlatoonTinkerRLTrainerConfig, sft_settings(tmp_path, data, batch_size=2)))

    # 6 conversations in 2 steps of 3; then the 4 left of epoch 0 in 2 steps of 2, and epoch 1 in 5.
    trained = prompts(stopped) + prompts(resumed)
    assert len(trained) == 2 * ROWS
    assert Counter(trained[:ROWS]) == Counter(trained[ROWS:]) == Counter(f"q{index}" for index in range(ROWS))
    final = rows(tmp_path)[-1]
    assert (final["name"], final["batch"], final["sft"]["epoch"], final["sft"]["position"]) == ("final", 9, 2, 0)


def test_a_resume_with_other_conversations_or_another_seed_is_refused(tmp_path, data):
    train(build_config(PlatoonTinkerRLTrainerConfig, sft_settings(tmp_path, data)), stop_at=2)

    with pytest.raises(ConfigError, match="it walked the conversations .* their order would differ"):
        train(build_config(PlatoonTinkerRLTrainerConfig, sft_settings(tmp_path, data, seed=7)))
    rows_file = data / "rows.json"
    rows_file.write_text(rows_file.read_text().replace('"q0"', '"q0 changed"'))
    with pytest.raises(ConfigError, match="cannot resume from 000002_interrupted: it walked the conversations"):
        train(build_config(PlatoonTinkerRLTrainerConfig, sft_settings(tmp_path, data)))


def test_the_cursor_walks_each_epoch_in_its_own_seeded_order():
    cursor = DatasetCursor(5, seed=3, fingerprint="f")

    taken = [cursor.take(2) for _ in range(6)]

    assert [len(batch) for batch in taken] == [2, 2, 1, 2, 2, 1]
    assert sorted(sum(taken[:3], [])) == sorted(sum(taken[3:], [])) == list(range(5))
    assert (cursor.epoch, cursor.position) == (2, 0)
    assert DatasetCursor(5, seed=3, fingerprint="f").order(1) == cursor.order(1) != cursor.order(0)
    assert (steps_per_epoch(10, 3), steps_per_epoch(9, 3)) == (4, 3)
    fresh = DatasetCursor(10, seed=3, fingerprint="f")
    options = SupervisedOptions(dataset="d", batch_size=3, num_epochs=2, max_training_steps=5)
    assert (
        last_step(options, fresh, 0),
        last_step(SupervisedOptions(dataset="d", batch_size=3, num_epochs=2), fresh, 0),
    ) == (5, 8)
    # Resumed after two steps of 3 with steps of 2: the 4 left of epoch 0 take 2 steps, epoch 1 takes 5.
    fresh.take(3), fresh.take(3)
    assert last_step(SupervisedOptions(dataset="d", batch_size=2, num_epochs=2), fresh, 2) == 9


def test_a_renderer_that_cannot_weight_turns_is_refused_before_the_service_opens(tmp_path, data, monkeypatch):
    from platoon.testing import toy_renderer

    def whole_conversations_only(message: Any, ctx: Any) -> Any:
        raise NotImplementedError("it renders whole conversations")

    renderer: Any = toy_renderer()
    renderer.render_message = whole_conversations_only
    monkeypatch.setattr(checks._FakeLLM, "renderer", property(lambda self: renderer))

    refused = (
        "train.renderer_name: fake-renderer cannot render rows row 0 for training (it renders whole conversations); "
        "trainer: sft needs a tinker-cookbook renderer which renders message by message"
    )
    with pytest.raises(ConfigError, match=re.escape(refused)):
        train(build_config(PlatoonTinkerRLTrainerConfig, sft_settings(tmp_path, data)))
    assert not (tmp_path / "logs" / "platoon_tinker" / "run" / "checkpoints.jsonl").exists()  # nothing was saved


def test_a_renderer_that_drops_tool_calls_is_refused_before_the_service_opens_naming_the_row(tmp_path, monkeypatch):
    """Only row 7 calls a tool: the check renders it too, before the launch save, and names it."""

    from platoon.testing import toy_renderer

    chat = [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Hello"}]
    call = [
        {"from": "human", "value": "Rome?"},
        {"from": "function_call", "value": json.dumps({"name": "get_weather", "arguments": {"city": "Rome"}})},
        {"from": "observation", "value": "sunny"},
        {"from": "gpt", "value": "Sunny."},
    ]
    directory = tmp_path / "data"
    directory.mkdir()
    entry = {"file_name": "chats.jsonl", "formatting": "sharegpt"}
    (directory / "dataset_info.json").write_text(json.dumps({"chats": entry}))
    lines = [{"conversations": call if index == 7 else chat} for index in range(10)]
    (directory / "chats.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    renderer: Any = toy_renderer()
    render_message = renderer.render_message
    renderer.render_message = lambda message, ctx: render_message(
        {key: value for key, value in message.items() if key != "tool_calls"}, ctx
    )
    monkeypatch.setattr(checks._FakeLLM, "renderer", property(lambda self: renderer))

    with pytest.raises(ConfigError) as refused:
        train(build_config(PlatoonTinkerRLTrainerConfig, sft_settings(tmp_path, directory, dataset="chats")))

    assert "cannot render chats row 7 for training (it renders the tool call of get_weather as " in str(refused.value)
    assert "needs a tinker-cookbook renderer with tool calling, such as qwen3" in str(refused.value)
    assert "llama3" not in str(refused.value)
    assert not (tmp_path / "logs" / "platoon_tinker" / "run" / "checkpoints.jsonl").exists()


def test_a_renderer_s_own_assertion_is_refused_before_the_service_opens_naming_the_row(tmp_path, monkeypatch, capsys):
    """Only row 4 has no system prompt, before which renderers such as kimi_k2 and nemotron3 add one of their own: the
    check renders it too, and a renderer that fails an assertion on it exits 78, naming the row, before the launch
    save."""

    rows = [
        {"instruction": f"q{index}", "output": f"a{index}", "system": "" if index == 4 else "Be brief."}
        for index in range(10)
    ]
    renderer = asserting_renderer(lambda messages: messages[0]["role"] != "system")
    monkeypatch.setattr(checks._FakeLLM, "renderer", property(lambda self: renderer))
    monkeypatch.setenv("TINKER_API_KEY", "tml-test")
    (tmp_path / "sft.yaml").write_text(json.dumps(sft_settings(tmp_path, alpaca(tmp_path, rows))))

    with fake_tinker_service() as service, pytest.raises(SystemExit) as exited:
        cli.main(["train", str(tmp_path / "sft.yaml")])

    assert exited.value.code == 78
    err = capsys.readouterr().err
    assert (
        "train.renderer_name: fake-renderer cannot render rows row 4 for training (AssertionError: expected a system "
        "message first); trainer: sft needs a tinker-cookbook renderer which renders message by message" in err
    )
    assert "Traceback" not in err
    assert not [call for call in service.calls if call.startswith("create_")]  # before the service opened
    assert not (tmp_path / "logs" / "platoon_tinker" / "run" / "checkpoints.jsonl").exists()


def test_a_renderer_s_own_assertion_on_a_later_conversation_stops_the_run_naming_the_row(tmp_path, data, monkeypatch):
    """A conversation the check did not render fails as a step comes to it: a ConfigError (exit 78) that names it."""

    renderer = asserting_renderer(lambda messages: any(message.get("content") == "q6" for message in messages))
    monkeypatch.setattr(checks._FakeLLM, "renderer", property(lambda self: renderer))

    with pytest.raises(ConfigError, match=re.escape("cannot render rows row 6 for training (AssertionError: expected")):
        train(build_config(PlatoonTinkerRLTrainerConfig, sft_settings(tmp_path, data)))


def test_a_conversation_with_nothing_to_train_within_max_length_is_left_out_and_counted(tmp_path, monkeypatch, caplog):
    # 8 tokens: a short reply (3 tokens) is kept whole; one of 21 tokens behind a prompt of 419 keeps no token.
    short = [{"instruction": f"q{index}", "output": f"a{index}"} for index in range(5)]
    long = [{"instruction": "x" * 400, "output": "y" * 20} for _ in range(5)]
    data = alpaca(tmp_path, [row for pair in zip(short, long, strict=True) for row in pair])
    logged = logged_steps(monkeypatch)
    settings = sft_settings(tmp_path, data, max_length=8, batch_size=10, num_epochs=1)

    service = train(build_config(PlatoonTinkerRLTrainerConfig, settings))

    assert service.calls.count("optim_step") == 1 and len(service.datums) == 5
    ((step,),) = [[stats for stats in logged if "train/step" in stats]]
    assert (step["train/conversations"], step["train/untrained_conversations"], step["train/trained_tokens"]) == (
        10,
        5,
        15,
    )
    assert re.search(
        r"Dataset rows: rows row [13579] has no token to train within trainer_options.max_length: 8\b", caplog.text
    )
    assert caplog.text.count("has no token to train within") == 1  # one warning a dataset


def test_a_run_in_which_no_conversation_has_a_token_to_train_stops_once_it_walked_them_all(tmp_path, data):
    # 3 tokens leave no reply token to train.
    config = build_config(PlatoonTinkerRLTrainerConfig, sft_settings(tmp_path, data, max_length=3, num_epochs=3))
    setup = TrainerSetup(config=config, policy_specs=policy_specs(config), options=registry.options_of(config))

    refused = "trainer_options.max_length: 3 leaves none of the 10 conversations a token to train"
    with fake_tinker_service() as service, pytest.raises(ConfigError, match=re.escape(refused)):
        asyncio.run(checks._run(SupervisedTrainer.from_setup(setup)))

    assert "optim_step" not in service.calls and not service.datums
    assert [row["name"] for row in rows(tmp_path)] == ["recovery-000000"]  # stopped in epoch 0, after its 4 steps


def test_the_run_logs_train_nll_per_trained_token_over_each_step_s_microbatches(tmp_path, data, monkeypatch):
    """Through the run's train tracker, which names the learner's metrics in its scope: each step logs its NLL per
    trained token (train/nll), and not the two metrics it divides."""

    import tinker
    import torch

    async def forward_backward_async(client: Any, datums: list[Any], loss_fn: str, loss_fn_config: Any = None) -> Any:
        # Every target token at log-probability -1 in a microbatch of two conversations, -3 in a microbatch of one.
        logprob = -1.0 if len(datums) == 2 else -3.0
        outputs = [
            {"logprobs": tinker.TensorData.from_torch(torch.full((datum.model_input.length,), logprob))}
            for datum in datums
        ]
        return checks._Done(SimpleNamespace(loss_fn_outputs=outputs, metrics={}))

    monkeypatch.setattr(checks._FakeTrainingClient, "forward_backward_async", forward_backward_async)
    logged = logged_steps(monkeypatch)

    train(build_config(PlatoonTinkerRLTrainerConfig, sft_settings(tmp_path, data, num_microbatches=2, num_epochs=1)))

    # 10 conversations with as many trained tokens each, 3 a step in microbatches of 2 and 1, then 1: each step's NLL
    # per trained token, (2 * 1 + 1 * 3) / 3, not the mean of its microbatches', 2.
    steps = [stats for stats in logged if "train/step" in stats]
    assert [stats["train/nll"] for stats in steps] == pytest.approx([5 / 3, 5 / 3, 5 / 3, 3.0])
    assert not [key for stats in steps for key in stats if "nll_" in key]


def test_recovery_checkpoints_are_pruned_with_none_kept_for_a_sampler(tmp_path, data):
    settings = sft_settings(tmp_path, data, num_epochs=1)
    settings["checkpoint"] = {"recovery_every": 1, "keep_recovery": 1}

    service = train(build_config(PlatoonTinkerRLTrainerConfig, settings))

    # Every row has the policy's one version, so a staleness bound of 0 would keep each of them for a sampler.
    assert {row["policy_version"] for row in rows(tmp_path)} == {0}
    names = [f"recovery-{step:06d}" for step in range(4)]
    assert sorted(service.deleted) == sorted(
        f"tinker://fake/{kind}/{name}" for name in names for kind in ("weights", "sampler_weights")
    )


class _Listener(px.StepHookBase):
    """A step hook that updates nothing: it hears who each step's end published."""

    name = "listener"

    def __init__(self) -> None:
        self.heard: list[dict[str, int]] = []

    async def published(self, end: Any) -> None:
        if end.kind == "step":
            self.heard.append(dict(end.published))


class _Listened(SupervisedTrainer):
    """The sft trainer, with a :class:`_Listener` attached."""

    def check(self, run: Any) -> None:
        super().check(run)
        self.listener = _Listener()
        run.hooks.register(self.listener)


@pytest.mark.parametrize("trainer", [SupervisedTrainer, _Listened])
def test_a_step_that_trains_nothing_saves_the_checkpoint_the_cadence_asks_for_with_a_hook_or_without(tmp_path, trainer):
    # Step 1 takes three conversations with nothing to train within 8 tokens, so neither it nor the hook updates.
    seed = SupervisedOptions(dataset="rows", num_epochs=1).seed
    untrained = DatasetCursor(ROWS, seed=seed, fingerprint="").order(0)[3:6]
    long, short = {"instruction": "x" * 400, "output": "y" * 20}, {"instruction": "q", "output": "a"}
    data = alpaca(tmp_path, [long if index in untrained else short for index in range(ROWS)])
    settings = sft_settings(tmp_path, data, max_length=8, num_epochs=1)
    settings["checkpoint"] = {"every": 2, "recovery_every": 1}
    config = build_config(PlatoonTinkerRLTrainerConfig, settings)
    setup = TrainerSetup(config=config, policy_specs=policy_specs(config), options=registry.options_of(config))

    with fake_tinker_service() as service:
        built = trainer.from_setup(setup)
        asyncio.run(checks._run(built))

    assert service.calls.count("optim_step") == 3
    names = ["recovery-000000", "recovery-000001", "000002", "recovery-000003", "000004", "final"]
    assert [row["name"] for row in rows(tmp_path)] == names
    if trainer is _Listened:  # a sampler a step for the hook, but none for step 1, which saves at the version it found
        assert built.listener.heard == [{"main": 1}, {}, {"main": 2}, {"main": 3}]
        assert [row["policy_version"] for row in rows(tmp_path)] == [0, 1, 1, 2, 3, 3]
    else:
        assert "save_weights_and_get_sampling_client" not in service.calls
        assert {row["policy_version"] for row in rows(tmp_path)} == {0}


# --- the config ------------------------------------------------------------------------------------------------------


def test_the_train_keys_only_rl_reads_are_refused_with_the_option_to_set_instead(tmp_path, data):
    settings = sft_settings(tmp_path, data)
    settings["train"] |= {
        "batch_size": 8,
        "max_staleness": 2,
        "stale_filter": "datum",
        "stepping": "independent",  # sft trains one policy, which has no clock but the run's
        "loss_fn": "ppo",
    }
    settings["eval"] = {"every": 5}

    with pytest.raises(ConfigError) as refused:
        build_config(PlatoonTinkerRLTrainerConfig, settings)

    assert refused.value.problems == [
        "train.batch_size: only the rl trainer reads it, and this run trains with sft: set trainer_options.batch_size",
        "train.max_staleness: only the rl trainer reads it, and this run trains with sft: leave it out",
        "train.stale_filter: only the rl trainer reads it, and this run trains with sft: leave it out",
        "train.stepping: only the rl trainer reads it, and this run trains with sft: leave it out",
        "train.loss_fn: only the rl trainer reads it, and this run trains with sft: leave it out",
        "eval.every: trainer: sft does not evaluate during training yet: set eval.every: 0, and evaluate the run's "
        "checkpoints with platoon eval",
    ]


@pytest.mark.parametrize(
    ("options", "problem"),
    [
        ({"datset": "rows"}, "trainer_options.datset"),
        ({"dataset": "rows", "num_epochs": None}, "trainer_options: set num_epochs or max_training_steps"),
        ({"dataset": " , "}, "trainer_options.dataset: name at least one dataset"),
        ({"dataset": "rows", "batch_size": 2, "num_microbatches": 3}, "num_microbatches: 3 is more than batch_size"),
        ({"dataset": "rows", "max_length": 1}, "trainer_options.max_length: must be an integer from 2"),
    ],
)
def test_trainer_options_are_checked_against_sft_s_options_when_the_config_loads(tmp_path, data, options, problem):
    settings = sft_settings(tmp_path, data)
    settings["trainer_options"] = {"dataset_dir": str(data), "num_epochs": 1, **options}

    with pytest.raises(ConfigError, match=re.escape(problem)):
        build_config(PlatoonTinkerRLTrainerConfig, settings)


def test_rl_refuses_sfts_options_and_sft_keeps_recovery_without_a_staleness_bound(tmp_path, data):
    rl = {**checks._settings(tmp_path, 2), "trainer_options": {"dataset": "rows"}}
    with pytest.raises(ConfigError, match="trainer_options: the rl trainer takes hooks, not dataset: its settings are"):
        build_config(PlatoonTinkerRLTrainerConfig, rl)

    settings = sft_settings(tmp_path, data)
    settings["checkpoint"] = {"recovery_every": 1, "keep_recovery": 1}
    config = build_config(PlatoonTinkerRLTrainerConfig, settings)
    assert registry.lock_record(config) == {
        "name": "sft",
        "options": {
            "dataset": "rows",
            "dataset_dir": str(data),
            "batch_size": 3,
            "num_epochs": 2,
            "max_training_steps": None,
            "num_microbatches": 1,
            "max_length": None,
            "mask_history": False,
            "seed": 42,
        },
    }


def test_the_dataset_directory_is_relative_to_the_config_file(tmp_path, data):
    folder = tmp_path / "configs"
    folder.mkdir()
    (folder / "data").symlink_to(data)
    settings = sft_settings(tmp_path, data)
    settings["trainer_options"]["dataset_dir"] = "data"
    (folder / "sft.yaml").write_text(json.dumps(settings))

    config = resolve_config(folder / "sft.yaml", [], PlatoonTinkerRLTrainerConfig).config

    assert config.trainer_options["dataset_dir"] == str(folder / "data")
    settings["trainer_options"]["dataset_dir"] = "elsewhere"
    (folder / "sft.yaml").write_text(json.dumps(settings))
    with pytest.raises(ConfigError, match="trainer_options.dataset_dir: elsewhere names .* which does not exist"):
        resolve_config(folder / "sft.yaml", [], PlatoonTinkerRLTrainerConfig)


def test_the_dataset_directory_left_out_is_data_beside_the_config_file(tmp_path, data, monkeypatch):
    folder = tmp_path / "configs"
    folder.mkdir()
    (folder / "data").symlink_to(data)
    settings = sft_settings(tmp_path, data)
    del settings["trainer_options"]["dataset_dir"]
    (folder / "sft.yaml").write_text(json.dumps(settings))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)  # not where the data is

    config = resolve_config(folder / "sft.yaml", [], PlatoonTinkerRLTrainerConfig).config

    assert config.trainer_options["dataset_dir"] == str(folder / "data")
    assert registry.trainer_path_options("sft") == {"dataset_dir": "data"}  # SupervisedOptions.path_options
    assert registry.trainer_path_options("rl") == {}
    (folder / "data").unlink()
    with pytest.raises(ConfigError, match=r"trainer_options.dataset_dir \(its default\): data names .* does not exist"):
        resolve_config(folder / "sft.yaml", [], PlatoonTinkerRLTrainerConfig)


def test_a_dataset_that_dataset_info_json_does_not_name_is_refused_when_the_config_loads(tmp_path, data):
    with pytest.raises(ConfigError, match="trainer_options.dataset: .*dataset_info.json has no dataset 'rows_typo'"):
        build_config(PlatoonTinkerRLTrainerConfig, sft_settings(tmp_path, data, dataset="rows_typo"))


def test_the_experimental_api_names_each_core_trainer():
    import platoon.experimental

    assert all(f"``{name}``" in (platoon.experimental.__doc__ or "") for name in registry.CORE_TRAINERS)
