"""The model side of the parity tests: the tokenizers they render with, and the scripted policy that samples.

Both tokenizers go with tinker-cookbook's ``qwen3_instruct`` renderer, so every prompt is rendered, and every
completion parsed, by the renderer training uses:

* :func:`byte_tokenizer`, built in memory: token ``i`` is byte ``i``, followed by Qwen3's chat tokens, so any text
  round-trips and nothing is downloaded.  CI runs on it.
* :func:`qwen_tokenizer`, Qwen3-4B-Instruct-2507's own tokenizer from the local Hugging Face cache (override the
  snapshot directory with ``QWEN3_4B_INSTRUCT_SNAPSHOT``), for a run with real token ids where it exists.

:class:`ScriptedSampler` stands in for the Tinker sampling client.  It reads the conversation out of each prompt and
answers with one action line of the probe harness's protocol (``parity_probe``): ``GUESS <n>``, ``DELEGATE <goal>``,
``REPORT <text>`` or ``GIVE UP``.  The answer is a pure function of the conversation and of the group member's
:class:`Plan`, the logprobs and routing ids are functions of the tokens, so every mode samples the same completions
however its rollouts interleave.
"""

from __future__ import annotations

import functools
import os
import re
from collections.abc import Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import tinker
from huggingface_hub.constants import HF_HUB_CACHE

import platoon.gateway.core  # noqa: F401  (keeps the routing ids of sampled sequences: the R3 response patch)

RENDERER_NAME = "qwen3_instruct"
END_OF_TURN = "<|im_end|>"
# Qwen3's chat tokens, in Qwen's order, with whether each is special.
CHAT_TOKENS: tuple[tuple[str, bool], ...] = (
    ("<|endoftext|>", True),
    ("<|im_start|>", True),
    (END_OF_TURN, True),
    ("<tool_call>", False),
    ("</tool_call>", False),
    ("<tool_response>", False),
    ("</tool_response>", False),
    ("<think>", False),
    ("</think>", False),
)
QWEN_SNAPSHOT = Path(
    os.environ.get(
        "QWEN3_4B_INSTRUCT_SNAPSHOT",
        Path(HF_HUB_CACHE) / "models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554",
    )
)


@functools.cache
def byte_tokenizer() -> Any:
    """A byte-level tokenizer with Qwen3's chat tokens: BPE with no merges over the 256 bytes, so the token ids of
    ASCII text are its character codes."""

    from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    characters = _byte_characters()
    tokenizer = Tokenizer(models.BPE(vocab={characters[byte]: byte for byte in range(256)}, merges=[]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    for content, special in CHAT_TOKENS:
        token = AddedToken(content, single_word=False, lstrip=False, rstrip=False, normalized=False, special=special)
        (tokenizer.add_special_tokens if special else tokenizer.add_tokens)([token])
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, eos_token=END_OF_TURN, pad_token="<|endoftext|>", clean_up_tokenization_spaces=False
    )


def qwen_tokenizer() -> Any | None:
    """Qwen3-4B-Instruct-2507's tokenizer, or None when its snapshot is not on disk."""

    if not (QWEN_SNAPSHOT / "tokenizer.json").exists():
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(QWEN_SNAPSHOT))


def _byte_characters() -> dict[int, str]:
    """The printable character standing for each byte in byte-level BPE (GPT-2's ``bytes_to_unicode``)."""

    printable = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)]
    characters = {byte: chr(byte) for byte in printable}
    for shift, byte in enumerate(byte for byte in range(256) if byte not in characters):
        characters[byte] = chr(256 + shift)
    return characters


# --------------------------------------------------------------------------------------------- the policy

RANGE = re.compile(r"between (\d+) and (\d+)")
FOUND = re.compile(r"number (\d+) correctly")
# A subagent's goal names its guess budget; a root's instruction (the env's prompt) does not.
CHILD_BUDGET = re.compile(r"with at most (\d+) guesses")
_MESSAGE = re.compile(r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>", re.S)


@dataclass(frozen=True)
class Plan:
    """How one group member's root plays: ``patience`` guesses of its own before it gives up, and the goal of a
    subagent it delegates to before its first guess (``{low}`` and ``{high}`` are filled with the range it knows).
    When the subagent finds the number, the root guesses it, or with ``trust_child`` ends without a guess of its own:
    on a fork that shares the root's environment the subagent's guess then grades the root's episode too.

    A subagent plays by its goal alone: it bisects with the guess budget its goal names, then reports the range.
    """

    patience: int
    delegate: str | None = None
    trust_child: bool = False


def conversation(prompt: str) -> list[tuple[str, str]]:
    """The ``(role, content)`` messages of a rendered prompt (the open assistant turn at its end is left out)."""

    return _MESSAGE.findall(prompt)


def next_action(messages: Sequence[tuple[str, str]], plan: Plan) -> str:
    """The policy: bisect the range in the instruction, narrowed by every observation since."""

    instruction = next(content for role, content in messages if role == "user")
    found_range = RANGE.search(instruction)
    if found_range is None:
        raise AssertionError(f"the policy got an instruction without a range: {instruction!r}")
    low, high = map(int, found_range.groups())
    budget = CHILD_BUDGET.search(instruction)
    patience = int(budget.group(1)) if budget else plan.patience
    turns = [content for role, content in messages if role in ("assistant", "user")][1:]
    guesses, delegated, answer = 0, False, None
    for action, observation in zip(turns[0::2], turns[1::2]):
        verb, _, argument = action.partition(" ")
        if verb == "GUESS":
            guesses += 1
            if observation.startswith("Too low"):
                low = int(argument) + 1
            elif observation.startswith("Too high"):
                high = int(argument) - 1
        elif verb == "DELEGATE":
            delegated = True
            if found := FOUND.search(observation):
                answer = int(found.group(1))
            elif reported := RANGE.search(observation):
                low, high = map(int, reported.groups())
    if budget is None and plan.delegate is not None and not delegated:
        return "DELEGATE " + plan.delegate.format(low=low, high=high)
    if answer is not None:
        return f"REPORT The subagent found the number {answer}." if plan.trust_child else f"GUESS {answer}"
    if guesses >= patience:
        return f"REPORT The number is between {low} and {high}." if budget else "GIVE UP"
    return f"GUESS {(low + high) // 2}"


class Member(Protocol):
    """What the sampler needs of the group member whose rollout is sampling (:data:`MEMBER`)."""

    plan: Plan
    sampling: list[dict[str, Any]]


# The group member whose rollout runs in this context: the workflow sets it for each member's task, and the
# rollout's agents (subagents included) inherit it.
MEMBER: ContextVar[Member] = ContextVar("parity_member")


@dataclass
class ScriptedSampler:
    """A Tinker sampling client that answers with :func:`next_action` for the member that calls it."""

    tokenizer: Any
    calls: int = field(default=0)

    async def sample_async(self, *, prompt: tinker.ModelInput, sampling_params: Any, num_samples: int) -> Any:
        member = MEMBER.get()
        prompt_tokens = prompt.to_ints()
        action = next_action(conversation(self.tokenizer.decode(prompt_tokens)), member.plan)
        tokens = self.tokenizer.encode(action + END_OF_TURN, add_special_tokens=False)
        member.sampling.append(
            {name: getattr(sampling_params, name) for name in ("max_tokens", "temperature", "top_k", "top_p", "seed")}
            | {"stop": list(sampling_params.stop or [])}
        )
        self.calls += 1
        sequence = {
            "stop_reason": "stop",
            "tokens": tokens,
            # Functions of the tokens alone: any mode that samples the same tokens records the same values.
            "logprobs": [-((token * 7 + index) % 23 + 1) / 16 for index, token in enumerate(tokens)],
            "routing_id": 1000 + len(prompt_tokens),
        }
        return tinker.types.SampleResponse.model_validate({"sequences": [sequence], "type": "sample"})

    def sample(self, **kwargs: Any) -> Any:
        raise AssertionError("the gateway samples asynchronously")
