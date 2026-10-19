"""``platoon eval`` and ``platoon rollout`` with a hosted model (``inference.hosted``), end to end through the CLI.

Number search (``platoon.envs.number_search``) runs in process.  Its harness calls the model gateway in its own dialect,
and the gateway forwards each call to a fake provider over real HTTP (``tests/gateway/hosted_fakes.py``), which plays
the game by bisection: ``platoon-react`` (``ctx.chat`` in process, Chat Completions), ``platoon-bash`` (its own
process, Chat Completions over HTTP), and the test agent of ``tests/hosted_agent`` (its own process, Messages or
Responses over HTTP, streamed).  Nothing here calls a real provider.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("OPENREWARD_DISABLE_UPDATE_CHECK", "1")
TESTS = Path(__file__).resolve().parent
if str(TESTS / "gateway") not in sys.path:
    sys.path.insert(0, str(TESTS / "gateway"))

from hosted_fakes import (  # noqa: E402
    CHAT,
    FAKE_KEY,
    MESSAGES,
    MODEL,
    RESPONSES,
    SERVED_MODEL,
    FakeProvider,
    Fault,
    bisect_chat,
    bisect_messages,
    bisect_responses,
    openai_error,
    with_tools,
)

import platoon as pt  # noqa: E402
from platoon import cli  # noqa: E402
from platoon.gateway import hosted as hosted_module  # noqa: E402
from platoon.utils.config import LOCKED_DISTRIBUTIONS  # noqa: E402

pytest.importorskip("openreward")

KEY_ENV = "PLATOON_TEST_PROVIDER_KEY"
SERVICE = "python://platoon.envs.number_search.env:NumberSearchEnvironment"
SYSTEM_PROMPT = "Find the hidden number. Call the guess tool with one integer per turn."
AGENTS = TESTS / "hosted_agent"


@pt.program("hostedtest-judging")
async def judging(ep: pt.EpisodeContext) -> pt.Outcome:
    ep.judge_endpoint()  # refused while judges are off
    await ep.run()
    return await ep.grade()


@pytest.fixture(autouse=True)
def _environment(monkeypatch):
    monkeypatch.setenv(KEY_ENV, FAKE_KEY)
    monkeypatch.delenv("TINKER_API_KEY", raising=False)  # a hosted model needs none
    for name in ("http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.setattr(hosted_module, "RETRY_BACKOFF_S", (0.01, 0.02))


@pytest.fixture
def provider():
    with FakeProvider(chat=bisect_chat, messages=bisect_messages, responses=bisect_responses) as fake:
        yield fake


def _config(
    tmp_path: Path,
    provider: FakeProvider,
    *,
    harness: str = "platoon-react",
    tasks: int = 3,
    hosted: dict | None = None,
    inference: dict | None = None,
    environment: dict | None = None,
) -> Path:
    entry: dict[str, Any] = {
        "service": SERVICE,
        "harness": harness,
        "agents": {"max_depth": 0},
        "eval_dataset_kwargs": {"split_name": "val", "limit": tasks},
        **(environment or {}),
    }
    if harness == "platoon-react":
        entry["harness_options"] = {"system_prompt": SYSTEM_PROMPT}
    path = tmp_path / "eval.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "inference": {
                    "model_name": MODEL,
                    "output_dir": str(tmp_path / "results"),
                    "hosted": {
                        "base_url": provider.base_url,
                        "api_key_env": KEY_ENV,
                        "max_retries": 1,
                        "pricing": {"input": 1.0, "output": 4.0},
                        **(hosted or {}),
                    },
                    "workflow": {
                        "num_concurrent_workers": 3,
                        "rollout_config": {"max_steps": 12, "timeout": 120, "verbose": False},
                    },
                    **(inference or {}),
                },
                "environments": [entry],
            }
        )
    )
    return path


def _main(*args: str) -> int:
    try:
        cli.main(list(args))
    except SystemExit as exit:
        return int(exit.code or 0)
    return 0


def _results(tmp_path: Path) -> dict[str, Any]:
    results = tmp_path / "results"
    return {
        "report": json.loads((results / "reports/final_report.json").read_text()),
        "lock": json.loads((results / "lock.json").read_text()),
        "metadata": [
            json.loads(path.read_text()) for path in sorted(results.glob("rollouts/*/rollout_*/metadata.json"))
        ],
        "collections": [
            json.loads(path.read_text())
            for path in sorted(results.glob("rollouts/*/rollout_*/trajectory_collection.json"))
        ],
    }


def _assert_no_key_anywhere(tmp_path: Path, *texts: str) -> None:
    """The provider key is in no file of the run, nor in what the run printed or logged."""

    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert FAKE_KEY.encode() not in path.read_bytes(), f"the provider key leaked into {path}"
    for text in texts:
        assert FAKE_KEY not in text


# --------------------------------------------------------------------------- end to end, per dialect


@pytest.mark.parametrize(
    ("harness", "dialect", "path"),
    [
        ("platoon-react", "openai_chat", CHAT),
        ("platoon-bash", "openai_chat", CHAT),
        (str(AGENTS / "messages-agent.toml"), "anthropic_messages", MESSAGES),
        (str(AGENTS / "responses-agent.toml"), "openai_responses", RESPONSES),
    ],
    ids=["platoon-react", "platoon-bash", "messages", "responses"],
)
def test_a_hosted_model_is_evaluated_end_to_end(tmp_path, capsys, caplog, provider, harness, dialect, path):
    caplog.set_level("DEBUG")
    assert _main("eval", str(_config(tmp_path, provider, harness=harness))) == 0

    out, err = capsys.readouterr()
    summary = json.loads(out)
    assert summary["success_rate"] == 1.0 and summary["valid_rollouts"] == 3
    run = _results(tmp_path)
    report, lock = run["report"], run["lock"]
    assert lock["policy"] == {
        "backend": "hosted",
        "model": MODEL,
        "base_url_origin": provider.origin,
        "api_key_env": KEY_ENV,
        "max_concurrent_requests": None,
        "probe": {dialect: {"served_model": SERVED_MODEL}},
    }
    # The hosted path locks no SDK: lock.json names the usual distributions, and no renderer's tool-call parsing.
    assert set(lock["versions"]) == set(LOCKED_DISTRIBUTIONS) and not {"openai", "anthropic"} & set(lock["versions"])
    assert "tool_call_parsing" not in lock
    calls = len(provider.calls(path)) - 1  # one was the probe
    assert report["model"]["backend"] == "hosted" and report["model"]["base_url_origin"] == provider.origin
    assert report["model"]["served_models"] == {SERVED_MODEL: calls}
    usage = report["usage"]["policy"]["total"]
    assert usage["calls"] == calls and usage["cached_prompt_tokens"] > 0
    assert report["cost"]["total_usd"] > 0 and report["cost"]["unpriced_calls"] == 0
    assert report["backend"]["attempts"] == calls and report["backend"]["failed_calls"] == 0
    for metadata in run["metadata"]:
        assert metadata["provenance"] == {
            "judges": True,
            "backend": "hosted",
            "model": MODEL,
            "base_url_origin": provider.origin,
        }
        assert metadata["usage"]["policy"]["calls"] > 0 and metadata["cost"]["usd"] > 0
    for collection in run["collections"]:
        root = next(iter(collection["trajectories"].values()))
        action = root["steps"][0]["misc"]["action_misc"]
        assert action["dialect"] == dialect and action["served_model"] == SERVED_MODEL and action["attempts"] == 1
    for request in provider.calls(path):
        assert request["body"]["model"] == MODEL and request["headers"].get("x-platoon-call-kind") is None
    if harness == "platoon-bash":
        # The harness's own body reached the provider; the gateway swapped the keys.
        body = provider.calls(CHAT)[1]["body"]
        assert [tool["function"]["name"] for tool in body["tools"]] == ["bash", "guess"]
    if dialect != "openai_chat":
        assert all(request["body"].get("stream") for request in provider.calls(path)[1:])
    _assert_no_key_anywhere(tmp_path, out, err, caplog.text)


def test_worker_processes_share_the_request_limit(tmp_path, capsys, provider):
    provider.delay = 0.05
    config = _config(
        tmp_path,
        provider,
        tasks=4,
        hosted={"max_concurrent_requests": 2},
        inference={
            "workflow": {
                "num_concurrent_workers": 4,
                "use_subprocesses": True,
                "subprocess_max_workers": 2,
                "rollout_config": {"max_steps": 12, "timeout": 120, "verbose": False},
            }
        },
    )
    assert _main("eval", str(config)) == 0
    assert json.loads(capsys.readouterr().out)["success_rate"] == 1.0
    assert provider.max_in_flight <= 2  # one request at a time in each of the two worker processes
    assert _results(tmp_path)["lock"]["policy"]["max_concurrent_requests"] == 2
    _assert_no_key_anywhere(tmp_path)


def test_a_request_limit_below_the_worker_processes_is_refused(tmp_path, capsys, provider):
    workflow = {
        "num_concurrent_workers": 4,
        "use_subprocesses": True,
        "subprocess_max_workers": 3,
        "rollout_config": {"max_steps": 12, "timeout": 120, "verbose": False},
    }
    config = _config(tmp_path, provider, hosted={"max_concurrent_requests": 2}, inference={"workflow": workflow})

    # Each of the 3 worker processes would make a request at a time at least: 3 in flight, over the limit of 2.
    assert _main("eval", str(config)) == 78
    error = capsys.readouterr().err
    assert "inference.hosted.max_concurrent_requests is 2, below the 3 worker processes that share it" in error
    assert provider.requests == []  # refused before the probe

    from platoon.eval.config_defs import EvalConfig, HostedConfig

    def share(limit: int, workers: int) -> int | None:
        hosted = HostedConfig(base_url=provider.base_url, max_concurrent_requests=limit)
        policy = EvalConfig(model_name=MODEL, hosted=hosted).policy()
        assert policy is not None
        return policy.for_processes(workers).max_concurrent_requests

    assert (share(5, 2), share(4, 4), share(7, 1)) == (2, 1, None)  # rounded down; one process keeps the run's limit


# --------------------------------------------------------------------------- refused before any rollout


def test_a_wrong_model_key_or_url_stops_the_evaluation_before_any_rollout(tmp_path, capsys, monkeypatch, provider):
    provider.models = ("another-model",)
    assert _main("eval", str(_config(tmp_path, provider))) == 78
    err = capsys.readouterr().err
    assert (
        f"inference.hosted: {provider.origin} refused the openai_chat probe: 404 The model `{MODEL}` does not exist"
        in err
    )
    provider.models = (MODEL,)
    monkeypatch.setenv(KEY_ENV, "sk-wrong-key-00000000")
    assert _main("eval", str(_config(tmp_path, provider))) == 78
    assert "refused the openai_chat probe: 401 Incorrect API key provided." in capsys.readouterr().err
    monkeypatch.delenv(KEY_ENV)
    sent = len(provider.requests)
    assert _main("eval", str(_config(tmp_path, provider))) == 78
    assert f"inference.hosted: set {KEY_ENV} in the environment, which api_key_env names" in capsys.readouterr().err
    assert len(provider.requests) == sent  # refused before anything was sent
    monkeypatch.setenv(KEY_ENV, FAKE_KEY)
    config = _config(tmp_path, provider, hosted={"base_url": "http://127.0.0.1:9/v1", "max_retries": 0})
    assert _main("eval", str(config)) == 78
    assert "inference.hosted: http://127.0.0.1:9 did not answer the openai_chat probe: ClientConnectorError" in (
        capsys.readouterr().err
    )
    assert not (tmp_path / "results/rollouts").exists()


def test_a_messages_server_that_lacks_a_chat_route_fails_the_probe_of_a_chat_harness(tmp_path, capsys, provider):
    provider.faults = [Fault(404, {"error": {"message": "Not Found"}}, path=CHAT)]
    assert _main("eval", str(_config(tmp_path, provider))) == 78
    assert "refused the openai_chat probe: 404 Not Found" in capsys.readouterr().err
    assert provider.calls(MESSAGES) == [] and len(provider.requests) == 1


@pytest.mark.parametrize(
    ("hosted", "message"),
    [
        ({"kind": "openai"}, "inference.hosted.kind: unknown key; base_url speaks the dialect of the harness"),
        ({"require_api_key": False}, "inference.hosted.require_api_key: unknown key; leave api_key_env out"),
        ({"headers": {"X-Org": "a"}}, "inference.hosted.headers: unknown key; put headers in a proxy"),
        ({"headers_env": {"X-Org": "ORG"}}, "inference.hosted.headers_env: unknown key; put headers in a proxy"),
        (
            {"extra_body": {"a": 1}},
            "inference.hosted.extra_body: unknown key; the gateway forwards the harness's own body",
        ),
        (
            {"drop_params": ["top_p"]},
            "inference.hosted.drop_params: unknown key; the gateway forwards the harness's own body",
        ),
        ({"base_url": "localhost:8000/v1"}, "inference.hosted.base_url must be an http or https URL"),
        ({"base_url": "https://user:secret@api.example.com/v1"}, "inference.hosted.base_url holds credentials"),
    ],
)
def test_stale_or_wrong_hosted_settings_exit_78_with_what_replaced_them(tmp_path, capsys, provider, hosted, message):
    assert _main("eval", str(_config(tmp_path, provider, hosted=hosted))) == 78
    assert message in capsys.readouterr().err
    assert provider.requests == []


@pytest.mark.parametrize(
    ("hosted", "message"),
    [
        ({"api_key_env": FAKE_KEY}, "inference.hosted.api_key_env must be the name of an environment variable"),
        ({"base_url": f"https://api.example.com/v1?key={FAKE_KEY}"}, "inference.hosted.base_url must have no query"),
        ({"base_url": f"api.example.com/v1?key={FAKE_KEY}"}, "inference.hosted.base_url must be an http or https URL"),
        ({"base_url": f"https://{FAKE_KEY}@api.example.com/v1"}, "inference.hosted.base_url holds credentials"),
    ],
    ids=["key-as-variable-name", "key-in-query", "key-in-query-without-scheme", "key-as-user"],
)
def test_a_key_put_in_the_config_is_refused_and_never_printed(tmp_path, capsys, provider, hosted, message):
    assert _main("eval", str(_config(tmp_path, provider, hosted=hosted))) == 78
    out, err = capsys.readouterr()
    assert message in err
    _assert_no_key_anywhere(tmp_path / "results", out, err)  # only the config itself holds it
    assert provider.requests == []


def test_the_first_draft_s_backend_section_is_refused_with_its_replacement(tmp_path, capsys, provider):
    config = _config(tmp_path, provider)
    data = yaml.safe_load(config.read_text())
    data["inference"]["backend"] = data["inference"].pop("hosted") | {"kind": "openai"}
    config.write_text(yaml.safe_dump(data))

    assert _main("eval", str(config)) == 78
    error = capsys.readouterr().err
    assert (
        "inference.backend: unknown key; a hosted model is inference.hosted: {base_url, api_key_env}, where base_url "
        "speaks the harness's dialect (RFC 0015)" in error
    )
    # The first draft also named a Tinker server there (kind: tinker).
    assert "a Tinker server's URL is the TINKER_BASE_URL variable" in error


def test_a_hosted_model_refuses_tinker_settings_and_training_refuses_hosted_models(tmp_path, capsys, provider):
    config = _config(tmp_path, provider, inference={"renderer_name": "qwen3_instruct", "context_window_length": 8})
    assert _main("eval", str(config)) == 78
    assert (
        "inference.renderer_name, inference.context_window_length configure a Tinker policy, but inference.hosted is "
        "set" in capsys.readouterr().err
    )

    training = tmp_path / "train.yaml"
    hosted = {"base_url": provider.base_url, "api_key_env": KEY_ENV}
    training.write_text(
        yaml.safe_dump(
            {
                "train": {"model_name": "m", "renderer_name": "qwen3_instruct", "num_epochs": 1, "hosted": hosted},
                "eval": {"backend": {"kind": "openai"}},
                "log_path": str(tmp_path / "logs"),
                "inference": {"model_name": "gpt-5.1", "hosted": hosted},
                "environments": [{"service": SERVICE, "harness": "platoon-react"}],
            }
        )
    )
    assert _main("train", str(training)) == 78
    error = capsys.readouterr().err
    assert "train.hosted: unknown key; hosted models (inference.hosted) serve evaluation only" in error
    assert "eval.backend: unknown key; hosted models (inference.hosted) serve evaluation only" in error
    assert "inference: unknown key; inference is an evaluation config's section" in error
    assert not (tmp_path / "logs").exists() and provider.requests == []


# --------------------------------------------------------------------------- failures during the run


def test_a_call_that_fails_after_retries_errors_its_rollout_and_resume_runs_it_again(tmp_path, capsys, provider):
    config = _config(tmp_path, provider, tasks=1)
    overloaded = openai_error("overloaded", kind="server_error")
    # The probe passes; the rollout's first call fails at both of its attempts.
    provider.faults = [Fault(503, overloaded, {"retry-after-ms": "5"}, when=with_tools) for _ in range(2)]
    assert _main("eval", str(config)) == 0
    first = json.loads(capsys.readouterr().out)
    (metadata,) = _results(tmp_path)["metadata"]
    assert first["errored_rollouts"] == 1 and first["valid_rollouts"] == 0
    assert metadata["status"] == "errored" and metadata["error"].startswith("backend_unavailable: 1 model call")
    assert metadata["backend"]["failed_calls"] == 1

    assert _main("eval", str(config)) == 0  # resume: the errored rollout runs again
    second = json.loads(capsys.readouterr().out)
    assert second["success_rate"] == 1.0 and second["errored_rollouts"] == 0


def test_a_refused_key_mid_run_stops_the_evaluation(tmp_path, capsys, caplog, provider):
    caplog.set_level("DEBUG")
    workflow = {"num_concurrent_workers": 1, "rollout_config": {"max_steps": 12, "verbose": False}}
    config = _config(tmp_path, provider, inference={"workflow": workflow})
    # The probe passes; the first rollout's first call is refused, and the provider repeats the key it got in its
    # error, as a proxy may: no output, record or log shows it.
    provider.faults = [Fault(401, openai_error("key revoked: {key}", code="invalid_api_key"), when=with_tools)]

    assert _main("eval", str(config)) == 78

    out, err = capsys.readouterr()
    assert (
        "the evaluation stopped after 1 rollout(s), 2 not started: backend_refused: "
        f"{provider.origin} refused an openai_chat call: 401 key revoked: [redacted]" in err
    )
    assert not (tmp_path / "results/reports").exists()
    _assert_no_key_anywhere(tmp_path, out, err, caplog.text)


def test_resuming_with_another_model_is_refused(tmp_path, capsys):
    with FakeProvider(chat=bisect_chat, models=(MODEL, "fake-model-2")) as provider:
        config = _config(tmp_path, provider, tasks=1)
        assert _main("eval", str(config)) == 0
        capsys.readouterr()
        assert _main("eval", str(config), "inference.model_name=fake-model-2") == 78
    assert "holds rollouts made with model 'fake-model' (now 'fake-model-2')" in capsys.readouterr().err


# --------------------------------------------------------------------------- the judges switch


def test_judges_off_with_a_hosted_model_is_recorded_and_a_judging_program_stops_the_run(tmp_path, capsys, provider):
    config = _config(tmp_path, provider, tasks=2, inference={"judges": False})
    assert _main("eval", str(config)) == 0
    assert json.loads(capsys.readouterr().out)["success_rate"] == 1.0
    run = _results(tmp_path)
    assert run["lock"]["judges"] == {"enabled": False, "child_grading": {}}
    assert run["report"]["judges"]["enabled"] is False and run["report"]["usage"]["judge"]["total"]["calls"] == 0
    assert all(metadata["provenance"]["judges"] is False for metadata in run["metadata"])

    judging_run = tmp_path / "judging"
    judging_run.mkdir()
    config = _config(
        judging_run, provider, tasks=2, inference={"judges": False}, environment={"program": "hostedtest-judging"}
    )
    data = yaml.safe_load(config.read_text())
    data["inference"]["workflow"]["num_concurrent_workers"] = 1
    config.write_text(yaml.safe_dump(data))
    assert _main("eval", str(config)) == 78
    err = capsys.readouterr().err
    assert "judges_off: a judge agent was refused because inference.judges is false" in err
    assert "program 'hostedtest-judging'" in err


# --------------------------------------------------------------------------- platoon rollout


def test_platoon_rollout_serves_a_hosted_model(tmp_path, capsys, provider):
    config = _config(tmp_path, provider, harness=str(AGENTS / "messages-agent.toml"))
    run_dir = tmp_path / "rollout"
    assert _main("rollout", str(config), "--task", "val:0", "--output-dir", str(run_dir), "--json-output") == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["reward"] == 1.0
    lock = json.loads((run_dir / "lock.json").read_text())
    assert lock["policy"]["backend"] == "hosted" and lock["policy"]["probe"] == {
        "anthropic_messages": {"served_model": SERVED_MODEL}
    }
    assert lock["judges"] == {"enabled": True}
    assert _main("rollout", str(config), "--task", "val:0", "--renderer", "qwen3_instruct") == 78
    assert "--renderer configure a Tinker policy, but inference.hosted is set" in capsys.readouterr().err
    _assert_no_key_anywhere(tmp_path)
