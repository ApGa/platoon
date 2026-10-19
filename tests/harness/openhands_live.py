"""Helpers for the OpenHands harness tests: the Qwen3.5-4B chat-template gateway a Harbor fast path trains with,
a scripted policy that writes completions in Qwen3.5's own format, and the reasoning-replay fixture.

``replay.json.gz`` holds the requests ``openhands`` sent through the gateway in a scripted three-call episode with its
default options (its real SDK runner, from a built toolchain) and the completion each one got.
``test_openhands_replay.py`` renders them on CPU without the toolchain.  Re-record on purpose only (an SDK or runner
change), with a built toolchain:

    PLATOON_TEST_TOOLCHAINS=/path/to/toolchains HF_HUB_OFFLINE=1 python tests/harness/openhands_live.py record
"""

from __future__ import annotations

import copy
import gzip
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from huggingface_hub.constants import HF_HUB_CACHE

MODEL_NAME = "Qwen/Qwen3.5-4B"
SNAPSHOT = Path(
    os.environ.get(
        "QWEN35_4B_SNAPSHOT",
        Path(HF_HUB_CACHE) / "models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    )
)
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "openhands" / "replay.json.gz"
# A directory holding the toolchain `platoon toolchain build openhands` built (openhands-1.54.0).
TOOLCHAINS = os.environ.get("PLATOON_TEST_TOOLCHAINS")
HARNESS = "openhands@1.54.0"


def toolchain_ready(name: str) -> bool:
    return bool(TOOLCHAINS) and (Path(TOOLCHAINS or "") / name / "manifest.json").is_file()


def turn(reasoning: str, *, text: str = "", call: tuple[str, dict[str, str]] | None = None) -> str:
    """A Qwen3.5 completion as the model writes it after the generation prompt's ``<think>\\n``: its reasoning,
    optional visible text and at most one tool call, in the chat template's own layout."""

    completion = f"{reasoning}\n</think>\n\n{text}"
    if call is not None:
        name, arguments = call
        parameters = "".join(f"<parameter={key}>\n{value}\n</parameter>\n" for key, value in arguments.items())
        completion += (
            "\n\n" if text else ""
        ) + f"<tool_call>\n<function={name}>\n{parameters}</function>\n</tool_call>"
    return completion


# openhands (SDK 1.54.0): write the answer, look at it with visible text, finish.
SDK_SCRIPT = [
    turn("The task wants 42 in answer.txt. I will write it.", call=("terminal", {"command": "echo 42 > answer.txt"})),
    turn("Let me check what I wrote.", text="Checking the file.", call=("terminal", {"command": "cat answer.txt"})),
    turn("The file holds 42.", call=("finish", {"message": "Wrote 42 to answer.txt."})),
]


@dataclass
class ScriptedPolicy:
    """A sampler that answers each call with the next scripted completion (then a plain answer), as token ids."""

    tokenizer: Any
    script: list[str]
    prompts: list[list[int]] = field(default_factory=list)
    completions: list[list[int]] = field(default_factory=list)

    def tokens(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False) + [
            self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        ]

    async def sample_async(self, *, prompt, sampling_params, num_samples):
        import tinker

        text = self.script.pop(0) if self.script else turn("Nothing is left to do.", text="Done.")
        tokens = self.tokens(text)
        self.prompts.append(prompt.to_ints())
        self.completions.append(tokens)
        sequence = {"stop_reason": "stop", "tokens": tokens, "logprobs": [-0.5] * len(tokens)}
        return tinker.types.SampleResponse.model_validate({"sequences": [sequence], "type": "sample"})

    def sample(self, **kwargs):
        raise AssertionError("the gateway samples asynchronously")


def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(SNAPSHOT))


def chat_template_renderer(tokenizer):
    """The renderer the fast path trains Qwen3.5 with (``renderer_name: chat_template``)."""

    from tinker_cookbook.renderers import get_renderer

    from platoon.train.tinker.renderers import register

    register()
    return get_renderer("chat_template", tokenizer, image_processor=None, model_name=tokenizer.name_or_path)


def recording_gateway(tokenizer, policy: ScriptedPolicy, *, context_window_length: int | None = None):
    """A real ``ModelGateway`` over ``policy`` that also keeps every request it renders (``.requests``); with
    ``context_window_length``, it refuses a prompt that fills that window, as a served model's gateway does."""

    from platoon.gateway.core import ModelGateway

    class RecordingGateway(ModelGateway):
        requests: list[dict[str, Any]]

        def prepare(self, request, *, defaults=None, **kwargs):
            render = request.render
            self.requests.append(
                {
                    "messages": copy.deepcopy(request.messages),
                    "tools": render.tools,
                    "render_kwargs": render.render_kwargs,
                }
            )
            return super().prepare(request, defaults=defaults, **kwargs)

    gateway = RecordingGateway(
        sampler=policy,
        renderer=chat_template_renderer(tokenizer),
        tokenizer=tokenizer,
        model_name=MODEL_NAME,
        context_window_length=context_window_length,
    )
    gateway.requests = []
    return gateway


def extends(previous_prompt: list[int], completion: list[int], prompt: list[int]) -> bool:
    """Whether ``prompt`` starts with the previous prompt and its completion: the calls merge into one datum."""

    head = previous_prompt + completion
    return prompt[: len(head)] == head


def first_difference(tokenizer, got: list[int], want: list[int]) -> str:
    index = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
    window = slice(max(0, index - 6), index + 6)
    return f"at token {index}: got {tokenizer.decode(got[window])!r}, want {tokenizer.decode(want[window])!r}"


def load_fixture() -> dict[str, Any]:
    with gzip.open(FIXTURE, "rt", encoding="utf-8") as handle:
        return json.load(handle)


async def run_episode(
    harness: str,
    script: list[str],
    workdir: Path,
    *,
    max_steps: int | None = None,
    options: dict[str, Any] | None = None,
    context_window_length: int | None = None,
):
    """Run ``harness`` (a built-in spec, from the toolchains under ``PLATOON_TEST_TOOLCHAINS``) with ``options`` (its
    ``harness_options``) through ``run_harness_rollout`` on the harness fakes' task (write 42 to answer.txt) with a
    real gateway (whose context window is ``context_window_length``, else unlimited).

    Returns ``(collection, gateway, policy)``.
    """

    from harness_fakes import FakeService

    from platoon.config_defs import RolloutConfig
    from platoon.harness.rollout import EmbeddedGateway, run_harness_rollout
    from platoon.harness.runner import HarnessRunner
    from platoon.task import Task
    from platoon.train.components import EnvironmentRoute

    tokenizer = load_tokenizer()
    policy = ScriptedPolicy(tokenizer, list(script))
    gateway = recording_gateway(tokenizer, policy, context_window_length=context_window_length)
    FakeService.root = workdir / "sandboxes"
    FakeService.sessions.clear()
    route = EnvironmentRoute(
        name="fake",
        service="harness_fakes:FakeService",
        harness=harness,
        resolve_task=True,
        harness_options=dict(options or {}),
    )
    config = RolloutConfig(output_dir=str(workdir / "out"), return_dict=True, verbose=False, max_steps=max_steps)
    runner = HarnessRunner(
        logs_dir=workdir / "logs", sandbox_state_root=str(workdir / "sandbox-state"), kill_grace_s=1.0
    )
    saved = os.environ.get("PLATOON_TOOLCHAINS")
    os.environ["PLATOON_TOOLCHAINS"] = str(TOOLCHAINS)  # where the runner resolves the specs' named toolchains
    try:
        collection = await run_harness_rollout(
            route.attach(Task(id="train-0")), config, gateway=EmbeddedGateway(gateway), runner=runner
        )
    finally:
        if saved is None:
            os.environ.pop("PLATOON_TOOLCHAINS", None)
        else:
            os.environ["PLATOON_TOOLCHAINS"] = saved
    return collection, gateway, policy


# Harbor's openhands-sdk adapter (harbor c178c20, src/harbor/agents/installed/openhands_sdk_runner.py) builds its agent
# with the SDK's defaults: LLM(model=..., api_key=..., base_url=...), the terminal, file editor and task tracker, and
# AgentContext(skills=[]) for a task without skills.  This runner builds it the same way and accepts the openhands
# spec's flags, so that a copy of the spec runs it in place of platoon's runner, from the same toolchain.
HARBOR_ADAPTER_RUNNER = """\
import argparse
import os

from openhands.sdk import LLM, Agent, AgentContext, Conversation, Tool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task_tracker import TaskTrackerTool
from openhands.tools.terminal import TerminalTool

parser = argparse.ArgumentParser()
parser.add_argument("--instruction-file")
parser.add_argument("--max-iterations", type=int)
parser.add_argument("--final-message-file")
args, _ = parser.parse_known_args()
llm = LLM(model=os.environ["LLM_MODEL"], api_key=os.environ["LLM_API_KEY"], base_url=os.environ["LLM_BASE_URL"])
tools = [Tool(name=TerminalTool.name), Tool(name=FileEditorTool.name), Tool(name=TaskTrackerTool.name)]
agent = Agent(llm=llm, tools=tools, agent_context=AgentContext(skills=[]))
conversation = Conversation(agent=agent, workspace=os.getcwd(), max_iteration_per_run=args.max_iterations)
with open(args.instruction_file, encoding="utf-8") as handle:
    conversation.send_message(handle.read())
conversation.run()
with open(args.final_message_file, "w", encoding="utf-8") as handle:
    handle.write("")
"""


def harbor_adapter_spec(directory: Path) -> Path:
    """A copy of the openhands spec that runs :data:`HARBOR_ADAPTER_RUNNER` instead of platoon's runner, from the same
    toolchain."""

    from platoon.harness.registry import get_harness_spec

    source = Path(str(get_harness_spec("openhands").source))
    text = source.read_text(encoding="utf-8")
    replacements = {
        'name = "openhands"': 'name = "harbor-adapter-openhands"',
        '"../openhands/run_openhands.py"': '"harbor_adapter_runner.py"',
        'lock = "openhands-1.54.0.txt"': f'lock = "{source.parent / "openhands-1.54.0.txt"}"',
    }
    for old, new in replacements.items():
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "harbor_adapter_runner.py").write_text(HARBOR_ADAPTER_RUNNER, encoding="utf-8")
    spec = directory / "harbor-adapter-openhands.toml"
    spec.write_text(text, encoding="utf-8")
    return spec


def run_specific(text: str, workdir: Path) -> str:
    """A decoded prompt without what differs between two runs of one episode: the sandbox directory under
    ``workdir`` and the date in the system prompt."""

    text = re.sub(re.escape(str(workdir)) + r"/sandboxes/[^/\s\"'\\\]]+", "<sandbox>", text)
    return re.sub(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", "<date>", text.replace(str(workdir), "<workdir>"))


def settings(collection: dict[str, Any]) -> dict[str, Any]:
    """The settings the runner reported in the root's harness log (its first ``openhands: settings`` line)."""

    (root,) = collection["trajectories"].values()
    log = Path(root["misc"]["harness"]["logs_path"]).read_text(errors="replace")
    line = next(line for line in log.splitlines() if line.startswith("openhands: settings "))
    return json.loads(line.removeprefix("openhands: settings "))


def record(workdir: Path) -> None:
    """Run the openhands episode with its default options and save what it sent as ``replay.json.gz``."""

    import asyncio

    collection, gateway, policy = asyncio.run(run_episode(HARNESS, SDK_SCRIPT, workdir))
    (root,) = collection["trajectories"].values()
    assert root["reward"] == 1.0 and len(gateway.requests) == len(SDK_SCRIPT), root["misc"]
    calls = [
        {**request, "completion": completion} for request, completion in zip(gateway.requests, SDK_SCRIPT, strict=True)
    ]
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    data = {"model": MODEL_NAME, "renderer": "chat_template", "harness": HARNESS, "calls": calls}
    with gzip.open(FIXTURE, "wt", encoding="utf-8") as handle:
        json.dump(scrub(data, str(workdir)), handle, indent=1, sort_keys=True)
    print(f"recorded {len(calls)} calls in {FIXTURE}")


def scrub(data: Any, workdir: str) -> Any:
    """``data`` with the recording machine's scratch directory replaced by ``/workspace``: the agent's working
    directory is in the SDK's prompts and tool output.  The replacement is the same in every call, so the prompts
    still extend each other."""

    return json.loads(json.dumps(data).replace(workdir, "/workspace"))


if __name__ == "__main__":
    import tempfile

    if sys.argv[1:] != ["record"] or not toolchain_ready("openhands-1.54.0"):
        sys.exit("usage: PLATOON_TEST_TOOLCHAINS=<built toolchains> python tests/harness/openhands_live.py record")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    with tempfile.TemporaryDirectory() as tmp:
        record(Path(tmp))
