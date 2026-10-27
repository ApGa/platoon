"""A conversation as the datum it trains on (:func:`supervised_datum`): rendered by the policy's renderer, as the policy
sees a conversation when it samples, each token weighted 1 where it trains and 0 elsewhere, and cut to ``max_length``
tokens as LLaMA-Factory cuts a conversation to its ``cutoff_len``.

The tokens that train are those of the model's turns, its replies and its tool calls, as in LLaMA-Factory, or with
``mask_history`` the last one's only; the system prompt, the tools' descriptions, the user's turns, the tools' results
and the renderer's headers never train.  The system prompt and the tools go first, as the renderer puts them
(``create_conversation_prefix_with_tools``, as the model gateway does); a tool call is an assistant message's
``tool_calls``, and a tool's result a ``tool`` message.  The renderer weights its assistant messages, which are the
model's turns, and nothing else (tinker-cookbook's ``TrainOnWhat.ALL_ASSISTANT_MESSAGES``), so a system message it adds
of its own (as ``kimi_k2``, ``gpt_oss`` and ``nemotron3`` do) does not train either.  A renderer that cannot render a
conversation message by message, as one that renders whole conversations at once (``chat_template``) cannot, raises
:class:`CannotRender`.

A rendered conversation is one :class:`Pair` per model turn: the tokens since the previous model turn (its source: the
system prompt and the tools before the first, then a user's turn or a tool's result, and the turn's header) and the
turn's own (its target), as LLaMA-Factory's templates pair them.  :func:`cut` shares ``max_length`` among the pairs as
LLaMA-Factory's supervised processor shares ``cutoff_len`` (``processor/supervised.py``, ``infer_seqlen``): pair by
pair from the first, or with ``mask_history`` from the last, the one that trains; each pair's source and target cut
from their ends to share what is left, and the pairs past it left out.  So the turns that train keep tokens to train
on, also behind a long prompt.

:func:`check_renders` is what a run asks of its renderer before it trains on a conversation: that it renders it, and
renders its tool calls.
"""

import logging
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast

import tinker
import torch
from tinker_cookbook.renderers import Message, Renderer, ToolCall, ToolSpec, TrainOnWhat
from tinker_cookbook.supervised.common import datum_from_model_input_weights

from platoon.train.supervised.formats import MODEL_ROLES, Conversation, Turn

# tinker-cookbook's renderers' module, which warns on every call that trains every assistant message with a renderer
# without the extension property (_without_the_extension_property_warning).
_RENDERERS_LOGGER = logging.getLogger("tinker_cookbook.renderers.base")


class CannotRender(Exception):
    """A renderer cannot render a conversation for training: it cannot render it message by message, it failed an
    assertion of its own on it, or it renders a model turn as no tokens, or a tool call without the tool's name."""


@dataclass(frozen=True)
class Pair:
    """One model turn of a rendered conversation: the tokens since the previous one (``source``), its own
    (``target``), and whether its own train."""

    source: list[int]
    target: list[int]
    trains: bool


def supervised_datum(
    renderer: Renderer, conversation: Conversation, *, mask_history: bool, max_length: int | None
) -> tinker.Datum:
    """The datum ``conversation`` trains on: its tokens as ``renderer`` renders them, cut to ``max_length`` (None: no
    cut), and their weights."""

    pairs = rendered_pairs(renderer, conversation, mask_history=mask_history)
    if max_length is not None:
        pairs = cut(pairs, max_length, last_first=mask_history)
    tokens: list[int] = []
    weights: list[float] = []
    for pair in pairs:
        tokens += pair.source + pair.target
        weights += [0.0] * len(pair.source) + [float(pair.trains)] * len(pair.target)
    return datum_from_model_input_weights(tinker.ModelInput.from_ints(tokens), torch.tensor(weights))


def rendered_pairs(renderer: Renderer, conversation: Conversation, *, mask_history: bool) -> list[Pair]:
    """The conversation as ``renderer`` renders it, one pair per model turn, each training but, with
    ``mask_history``, the last.  Raises :class:`CannotRender` for a renderer that cannot render it message by message,
    that fails an assertion of its own on it, or that renders a model turn as no tokens of its own."""

    try:
        with _without_the_extension_property_warning():
            model_input, weights = renderer.build_supervised_example(
                chat_messages(renderer, conversation), TrainOnWhat.ALL_ASSISTANT_MESSAGES
            )
    except NotImplementedError as exc:
        raise CannotRender(str(exc) or type(exc).__name__) from exc
    except AssertionError as exc:  # a renderer's own check of the messages, which it asserts
        raise CannotRender(f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__) from exc
    spans: list[tuple[list[int], list[int]]] = [([], [])]
    for token, weight in zip(model_input.to_ints(), weights.tolist(), strict=True):
        if weight > 0:
            spans[-1][1].append(token)
        elif spans[-1][1]:  # the first token after a model turn: the next pair's source begins
            spans.append(([token], []))
        else:
            spans[-1][0].append(token)
    model_turns = sum(turn.role in MODEL_ROLES for turn in conversation.turns)
    if len(spans) != model_turns or not spans[-1][1]:
        raise CannotRender(
            f"{type(renderer).__name__} renders {model_turns} model turns as {len(spans)} runs of tokens of their own"
        )
    last = len(spans) - 1
    return [
        Pair(source, target, trains=not mask_history or index == last) for index, (source, target) in enumerate(spans)
    ]


def cut(pairs: list[Pair], max_length: int, *, last_first: bool) -> list[Pair]:
    """``pairs`` cut to ``max_length`` tokens as LLaMA-Factory cuts a conversation to ``cutoff_len``: each pair in turn
    (from the last with ``last_first``) takes its share of the tokens left (:func:`shares`), its source and target cut
    from their ends, until none are left; the pairs after that are left out."""

    kept: list[Pair] = []
    left = max_length
    for pair in reversed(pairs) if last_first else pairs:
        if left <= 0:
            break
        source, target = shares(len(pair.source), len(pair.target), left)
        kept.append(Pair(pair.source[:source], pair.target[:target], pair.trains))
        left -= source + target
    return kept[::-1] if last_first else kept


def shares(source: int, target: int, budget: int) -> tuple[int, int]:
    """How many of a pair's ``source`` and ``target`` tokens fit in ``budget``, as LLaMA-Factory's ``infer_seqlen``
    shares it: a target shorter than half the budget is kept whole, else one behind a source shorter than half is cut
    to what the source leaves, else the two are cut in proportion to their lengths; the source takes what is left."""

    if target * 2 < budget:
        most = budget
    elif source * 2 < budget:
        most = budget - source
    else:
        most = int(budget * (target / (source + target)))  # in infer_seqlen's order, which rounds as it does
    target = min(most, target)
    return min(budget - target, source), target


def check_renders(renderer: Renderer, conversation: Conversation) -> None:
    """Raise :class:`CannotRender` when ``renderer`` cannot render ``conversation`` for training
    (:func:`rendered_pairs`), or renders a tool call without the tool's name (as ``llama3`` renders one, as an empty
    reply)."""

    pairs = rendered_pairs(renderer, conversation, mask_history=False)
    model_turns = [turn for turn in conversation.turns if turn.role in MODEL_ROLES]
    for turn, pair in zip(model_turns, pairs, strict=True):
        rendered = str(renderer.tokenizer.decode(pair.target))
        if missing := [call.name for call in turn.calls if call.name not in rendered]:
            raise CannotRender(f"it renders the tool call of {missing[0]} as {rendered!r}, without the call")


def probes(conversations: Sequence[Conversation]) -> list[Conversation]:
    """The conversations that show what ``conversations`` ask of a renderer, each once: the first, the first with
    tools, the first with a tool call, and the first with neither a system prompt nor tools, before which a renderer
    may add a system message of its own."""

    candidates = [
        conversations[0] if conversations else None,
        next((each for each in conversations if each.tools), None),
        next((each for each in conversations if any(turn.calls for turn in each.turns)), None),
        next((each for each in conversations if not each.system and not each.tools), None),
    ]
    found: list[Conversation] = []
    for candidate in candidates:
        if candidate is not None and all(candidate is not other for other in found):
            found.append(candidate)
    return found


def chat_messages(renderer: Renderer, conversation: Conversation) -> list[Message]:
    """The conversation as the renderer's messages: the system prompt and the tools as the renderer puts them, then
    the turns, each model turn an assistant message."""

    messages: list[dict[str, Any]] = list(_prefix(renderer, conversation))
    messages += [_message(turn) for turn in conversation.turns]
    return cast("list[Message]", messages)


@contextmanager
def _without_the_extension_property_warning() -> Generator[None]:
    """tinker-cookbook warns, on every call that trains every assistant message, that a renderer without the extension
    property (as ``qwen3``, which leaves the thinking of earlier turns out) shows an earlier turn another prefix than
    the policy saw when it sampled that turn.  A conversation trains every model turn in one sequence, as LLaMA-Factory
    trains it: the warning would only repeat itself for every conversation, so it is left out while one renders."""

    quiet = _ExtensionPropertyWarning()
    _RENDERERS_LOGGER.addFilter(quiet)
    try:
        yield
    finally:
        _RENDERERS_LOGGER.removeFilter(quiet)


class _ExtensionPropertyWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "extension property" not in record.getMessage()


def _prefix(renderer: Renderer, conversation: Conversation) -> list[Any]:
    """The messages before the turns: the system prompt, with the tools' descriptions as the renderer adds them."""

    if conversation.tools:
        tools = cast("list[ToolSpec]", list(conversation.tools))  # each a name, description and parameters
        return list(renderer.create_conversation_prefix_with_tools(tools, conversation.system))
    return [{"role": "system", "content": conversation.system}] if conversation.system else []


def _message(turn: Turn) -> dict[str, Any]:
    if turn.role == "function":
        calls = [
            ToolCall(function=ToolCall.FunctionBody(name=call.name, arguments=call.arguments)) for call in turn.calls
        ]
        return {"role": "assistant", "content": "", "tool_calls": calls}
    if turn.role == "observation":
        return {"role": "tool", "content": turn.text}
    return {"role": turn.role, "content": turn.text}
