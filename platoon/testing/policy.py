"""Scripted policies: a stand-in for the Tinker sampler that writes what a test says the model writes.

A :class:`ScriptedPolicy` is a gateway sampler (``sample_async``, as a Tinker ``SamplingClient`` has).  The gateway
renders each model call as usual, the policy decodes the prompt, picks the completion text, and returns it encoded
the way a Tinker server would, so everything after sampling (parsing, recording, trajectories, training data) runs
the real code.  The script is either:

* a sequence of completions, used in order: ``ScriptedPolicy(["LOOK", "FINISH done"])``;
* a function of the decoded prompt: ``ScriptedPolicy(lambda prompt: "FINISH " + ...)``, for agents whose calls
  interleave (subagents, concurrent tools), where call order is not deterministic.

A completion is a string, or a :class:`Completion` that also sets the stop reason, a routing id or a delay.
:func:`rules` builds a function that answers by the last user message, and :func:`tool_call` writes a tool call in
the Qwen3 format the toy tokenizer's renderer parses.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import tinker
    from tinker_cookbook.tokenizer_utils import Tokenizer

StopReason = Literal["stop", "length"]
# Each model call's sampled logprob, unless the Completion sets one.
DEFAULT_LOGPROB = -0.25


@dataclass(frozen=True)
class Completion:
    """What the scripted model writes for one call."""

    # The completion text; a tool call is text too (see tool_call).
    text: str
    # "stop": the model ended its turn, so the policy's stop token follows the text; "length": the completion was
    # cut off at the token limit (no stop token, and the gateway emits no tool calls).
    stop_reason: StopReason = "stop"
    # The server's routing handle for the sampled sequence, as a Tinker server with rollout routing replay sends it.
    routing_id: int | None = None
    # Seconds to wait before answering, to make concurrent calls finish in a chosen order.
    delay: float = 0.0
    # The logprob of every sampled token (None: the policy's).
    logprob: float | None = None


@dataclass(frozen=True)
class PolicyCall:
    """One model call a scripted policy answered."""

    # The rendered prompt, as the gateway sent it.
    prompt: tinker.ModelInput
    # The prompt, decoded.
    text: str
    # The sampling parameters the gateway resolved for the call.
    sampling_params: Any
    # What the policy answered.
    completion: Completion
    # The sampled tokens returned to the gateway (the completion's text and, after a stop, the stop token).
    tokens: list[int]

    @property
    def last_user_message(self) -> str:
        """The content of the prompt's last user message (see :func:`last_user_message`)."""
        return last_user_message(self.text)


Reply = str | Completion
Script = Sequence[Reply] | Callable[[str], Reply]


class ScriptExhaustedError(AssertionError):
    """A scripted policy was asked for more completions than its script has."""


class ScriptedPolicy:
    """A gateway sampler whose completions come from a script (see the module docstring).

    ``calls`` records every call in the order the policy answered them.  ``tokenizer`` is the one the gateway
    renders with; :func:`platoon.testing.make_gateway` sets it when it is None.  ``stop_token`` is the text appended
    to a completion that stopped (default: the tokenizer's end-of-sequence token, ``<|im_end|>`` for the toy and
    Qwen tokenizers).
    """

    def __init__(
        self,
        script: Script = (),
        *,
        tokenizer: Tokenizer | None = None,
        stop_token: str | None = None,
        logprob: float = DEFAULT_LOGPROB,
    ) -> None:
        if isinstance(script, (str, Completion)):
            raise TypeError("A script is a sequence of completions or a function of the prompt, not one completion")
        self._script = script
        self.tokenizer = tokenizer
        self.stop_token = stop_token
        self.logprob = logprob
        self.calls: list[PolicyCall] = []
        self._function: Callable[[str], Reply] | None = None
        self._remaining: list[Reply] | None = None
        if isinstance(script, Sequence):
            self._remaining = list(script)
        elif callable(script):
            self._function = script
        else:
            raise TypeError(f"A script is a sequence of completions or a function of the prompt, got {script!r}")
        self._lock = threading.Lock()

    @property
    def script(self) -> Script:
        """The script the policy was made with."""
        return self._script

    @property
    def prompts(self) -> list[str]:
        """The decoded prompt of every call, in order."""
        return [call.text for call in self.calls]

    @property
    def remaining(self) -> list[Reply] | None:
        """The scripted completions not used yet; None for a function script."""
        return None if self._remaining is None else list(self._remaining)

    def answer(self, prompt_text: str) -> Completion:
        """The completion for one prompt: the next scripted one, or the script function's answer."""

        if self._function is not None:
            return as_completion(self._function(prompt_text))
        with self._lock:
            if not self._remaining:
                tail = prompt_text[-300:]
                raise ScriptExhaustedError(
                    f"The scripted policy has no completion left for call {len(self.calls) + 1}; the prompt ends with "
                    f"{tail!r}"
                )
            return as_completion(self._remaining.pop(0))

    def tokens(self, completion: Completion) -> list[int]:
        """The tokens a server would return for ``completion``."""

        tokenizer = self._tokenizer()
        tokens = list(tokenizer.encode(completion.text, add_special_tokens=False))
        if completion.stop_reason == "stop":
            stop = self.stop_token if self.stop_token is not None else tokenizer.eos_token
            if not stop:
                raise ValueError("The scripted policy needs a stop_token: its tokenizer has no eos_token")
            tokens += tokenizer.encode(stop, add_special_tokens=False)
        return tokens

    async def sample_async(self, *, prompt: Any, sampling_params: Any, num_samples: int = 1) -> Any:
        """Answer one model call, as ``tinker.SamplingClient.sample_async`` does."""

        text = self._tokenizer().decode(prompt.to_ints())
        completion = self.answer(text)
        if completion.delay:
            await asyncio.sleep(completion.delay)
        return self._respond(prompt, text, sampling_params, completion, num_samples)

    def sample(self, *, prompt: Any, sampling_params: Any, num_samples: int = 1) -> Any:
        """The blocking twin of :meth:`sample_async` (the gateway's ``sample_sync``); returns the response."""

        text = self._tokenizer().decode(prompt.to_ints())
        completion = self.answer(text)
        if completion.delay:
            time.sleep(completion.delay)
        return self._respond(prompt, text, sampling_params, completion, num_samples)

    def _respond(self, prompt: Any, text: str, sampling_params: Any, completion: Completion, num_samples: int) -> Any:
        if num_samples != 1:
            raise ValueError(f"The scripted policy samples one completion per call, not {num_samples}")
        tokens = self.tokens(completion)
        with self._lock:
            self.calls.append(PolicyCall(prompt, text, sampling_params, completion, tokens))
        logprob = self.logprob if completion.logprob is None else completion.logprob
        return sample_response(
            tokens, stop_reason=completion.stop_reason, routing_id=completion.routing_id, logprob=logprob
        )

    def _tokenizer(self) -> Any:
        if self.tokenizer is None:
            from platoon.testing.tokenizer import toy_tokenizer

            self.tokenizer = toy_tokenizer()
        return self.tokenizer


def as_completion(reply: Reply) -> Completion:
    """``reply`` as a :class:`Completion` (a string is a completion that stops)."""

    if isinstance(reply, Completion):
        return reply
    if isinstance(reply, str):
        return Completion(reply)
    raise TypeError(f"A scripted completion is a string or a Completion, got {reply!r}")


def as_policy(policy: ScriptedPolicy | Script) -> ScriptedPolicy:
    """``policy`` as a :class:`ScriptedPolicy` (a script is wrapped in one)."""

    return policy if isinstance(policy, ScriptedPolicy) else ScriptedPolicy(policy)


def sample_response(
    tokens: Sequence[int],
    *,
    stop_reason: StopReason = "stop",
    routing_id: int | None = None,
    logprob: float = DEFAULT_LOGPROB,
) -> tinker.types.SampleResponse:
    """A Tinker ``SampleResponse`` with one sequence of ``tokens``, each with logprob ``logprob``."""

    import tinker

    sequence: dict[str, Any] = {"stop_reason": stop_reason, "tokens": list(tokens), "logprobs": [logprob] * len(tokens)}
    if routing_id is not None:
        sequence["routing_id"] = routing_id
    return tinker.types.SampleResponse.model_validate({"sequences": [sequence], "type": "sample"})


_USER_MESSAGE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.S)


def last_user_message(prompt: str) -> str:
    """The content of the last user message of a decoded Qwen3-format prompt ("" if there is none).

    Tool results count: the Qwen3 renderers render a ``tool`` message as a user turn wrapped in
    ``<tool_response>`` tags.
    """

    found = _USER_MESSAGE.findall(prompt)
    return found[-1] if found else ""


def rules(table: Mapping[str, Reply], *, default: Reply | None = None) -> Callable[[str], Reply]:
    """A script function: the reply of the first key of ``table`` that the last user message starts with.

    Keys are tried in order.  With no match the reply is ``default``, or, without one, an error that names the
    message, so a policy that meets a turn it does not expect fails the test instead of looping::

        policy = ScriptedPolicy(rules({"Find the key": "SPAWN search the vault", "obs:": "FINISH done"}))
    """

    def reply(prompt: str) -> Reply:
        message = last_user_message(prompt)
        for prefix, answer in table.items():
            if message.startswith(prefix):
                return answer
        if default is None:
            raise AssertionError(f"No rule matches the last user message {message[:200]!r}")
        return default

    return reply


def tool_call(name: str, arguments: Mapping[str, Any] | None = None, /, **kwargs: Any) -> str:
    """A tool call as a Qwen3 model writes it, which the Qwen3 renderers parse into an OpenAI tool call.

    ``tool_call("guess", number=5)`` is ``<tool_call>`` and ``</tool_call>`` around
    ``{"name": "guess", "arguments": {"number": 5}}``, each on its own line.
    """

    payload = {"name": name, "arguments": {**(arguments or {}), **kwargs}}
    return f"<tool_call>\n{json.dumps(payload)}\n</tool_call>"
