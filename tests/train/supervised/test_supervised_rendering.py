"""A conversation as the datum the sft trainer trains on (:mod:`platoon.train.supervised.rendering`), rendered by the
policy's renderer (here tinker-cookbook's ``qwen3_instruct`` over the toy tokenizer, whose ids of ASCII text are its
character codes), with the loss on the model's turns; and the cross-entropy objective it trains under."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import tinker
import torch

from platoon.testing import toy_renderer, toy_tokenizer
from platoon.train.objectives.cross_entropy import CrossEntropy, mean_per_token, step_nll, trained_tokens
from platoon.train.supervised.formats import Conversation, ToolCall, Turn
from platoon.train.supervised.rendering import CannotRender, check_renders, probes, shares, supervised_datum

WEATHER = {"name": "get_weather", "description": "A city's weather", "parameters": {"type": "object"}}
CHAT = Conversation(
    turns=(Turn("user", "Hi"), Turn("assistant", "Hello"), Turn("user", "Bye"), Turn("assistant", "Goodbye")),
    system="Be brief.",
)
TOOL = Conversation(
    turns=(
        Turn("user", "Rome?"),
        Turn("function", calls=(ToolCall("get_weather", '{"city": "Rome"}'),)),
        Turn("observation", "sunny"),
        Turn("assistant", "Sunny."),
    ),
    tools=(WEATHER,),
)


def rendered(conversation: Conversation, *, mask_history: bool = False, max_length: int | None = None) -> str:
    datum = supervised_datum(toy_renderer(), conversation, mask_history=mask_history, max_length=max_length)
    return str(
        toy_tokenizer().decode(datum.model_input.to_ints() + datum.loss_fn_inputs["target_tokens"].tolist()[-1:])
    )


def trained(conversation: Conversation, *, mask_history: bool = False, max_length: int | None = None) -> str:
    """The text of the tokens that train, as the targets they are."""

    datum = supervised_datum(toy_renderer(), conversation, mask_history=mask_history, max_length=max_length)
    targets = datum.loss_fn_inputs["target_tokens"].tolist()
    weights = datum.loss_fn_inputs["weights"].tolist()
    return str(toy_tokenizer().decode([token for token, weight in zip(targets, weights, strict=True) if weight > 0]))


def test_the_model_s_turns_train_and_the_rest_renders_as_the_policy_sees_it():
    assert rendered(CHAT) == (
        "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello"
        "<|im_end|>\n<|im_start|>user\nBye<|im_end|>\n<|im_start|>assistant\nGoodbye<|im_end|>"
    )
    assert trained(CHAT) == "Hello<|im_end|>Goodbye<|im_end|>"


def test_mask_history_trains_the_last_model_turn_only():
    assert trained(CHAT, mask_history=True) == "Goodbye<|im_end|>"


def test_tool_calls_train_and_the_tools_and_their_results_do_not():
    text = rendered(TOOL)

    assert '"name": "get_weather"' in text.split("<|im_start|>user")[0]  # the tools, in the system prompt
    assert "<tool_response>\nsunny\n</tool_response>" in text
    assert trained(TOOL) == (
        '\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Rome"}}\n</tool_call><|im_end|>Sunny.<|im_end|>'
    )


def test_max_length_shares_the_tokens_among_the_turns_as_llama_factory_s_cutoff_len():
    # infer_seqlen: a short reply is kept whole; one behind a short prompt takes what the prompt leaves; else the two
    # are cut in proportion.
    assert shares(3000, 10, 2048) == (2038, 10)
    assert shares(100, 3000, 2048) == (100, 1948)
    assert shares(1500, 1500, 2048) == (1024, 1024)
    assert shares(5, 3, 100) == (5, 3)
    # Cut in proportion, the share rounds as infer_seqlen's does: int(22 * (30 / 44)) is 14, int(22 * 30 / 44) 15.
    assert shares(14, 30, 22) == (8, 14)
    # The toy tokenizer's tokens are characters: 40 tokens take the system prompt, the first user turn and the
    # assistant's header (the first prompt, 40 tokens), then "Hello<|im_end|>" (its reply, 6): the prompt is cut to 34,
    # the reply kept whole, and the second turn is past the budget.
    cut = supervised_datum(toy_renderer(), CHAT, mask_history=False, max_length=40)
    assert cut.model_input.length == 39
    assert trained(CHAT, max_length=40) == "Hello<|im_end|>"


def test_with_mask_history_the_cut_keeps_the_last_turn_which_trains():
    # The last turn takes the budget first: its reply whole, then the end of its prompt; the first turn is left out.
    assert trained(CHAT, mask_history=True, max_length=20) == "Goodbye<|im_end|>"
    text = rendered(CHAT, mask_history=True, max_length=20)
    assert text.endswith("Bye<|im_end|>\nGoodbye<|im_end|>") and "Hello" not in text
    assert supervised_datum(toy_renderer(), CHAT, mask_history=True, max_length=20).model_input.length == 19


def test_a_prompt_longer_than_max_length_still_trains_its_reply():
    long = Conversation(turns=(Turn("user", "x" * 300), Turn("assistant", "Yes.")))

    assert trained(long, max_length=100) == "Yes.<|im_end|>"
    assert supervised_datum(toy_renderer(), long, mask_history=False, max_length=100).model_input.length == 99


def test_a_renderer_that_renders_whole_conversations_cannot_weight_turns():
    def whole_conversations_only(message: Any, ctx: Any) -> Any:
        raise NotImplementedError("whole conversations")

    renderer: Any = toy_renderer()
    renderer.render_message = whole_conversations_only

    with pytest.raises(CannotRender, match="whole conversations"):
        supervised_datum(renderer, CHAT, mask_history=False, max_length=None)


def test_a_system_message_the_renderer_adds_of_its_own_renders_and_does_not_train():
    """As kimi_k2 and nemotron3 add one before a conversation without a system prompt, and gpt_oss before any."""

    renderer: Any = toy_renderer()
    build = renderer.build_supervised_example
    renderer.build_supervised_example = lambda messages, train_on_what: build(
        [{"role": "system", "content": "You are a model."}, *messages], train_on_what
    )
    untold = Conversation(turns=CHAT.turns)

    datum = supervised_datum(renderer, untold, mask_history=False, max_length=None)

    text = str(toy_tokenizer().decode(datum.model_input.to_ints()))
    assert text.startswith("<|im_start|>system\nYou are a model.<|im_end|>\n<|im_start|>user\nHi")
    targets, weights = datum.loss_fn_inputs["target_tokens"].tolist(), datum.loss_fn_inputs["weights"].tolist()
    assert toy_tokenizer().decode([t for t, w in zip(targets, weights, strict=True) if w > 0]) == trained(untold)
    assert trained(untold) == "Hello<|im_end|>Goodbye<|im_end|>"


def test_a_renderer_s_own_assertion_is_a_conversation_it_cannot_render():
    renderer: Any = toy_renderer()

    def asserting(messages: Any, train_on_what: Any) -> Any:
        raise AssertionError("expected a single token for <|return|>")

    renderer.build_supervised_example = asserting

    with pytest.raises(CannotRender, match=r"^AssertionError: expected a single token for <\|return\|>$"):
        supervised_datum(renderer, CHAT, mask_history=False, max_length=None)


def test_a_renderer_without_the_extension_property_renders_without_tinker_cookbook_s_warning(caplog):
    """tinker-cookbook warns on every call that trains every assistant message with such a renderer (``qwen3``, which
    leaves the thinking of earlier turns out): a conversation trains every model turn in one sequence, as LLaMA-Factory
    trains it, and the warning is left out."""

    renderer = toy_renderer("qwen3")
    assert not renderer.has_extension_property

    with caplog.at_level("WARNING"):
        datum = supervised_datum(renderer, CHAT, mask_history=False, max_length=None)

    targets, weights = datum.loss_fn_inputs["target_tokens"].tolist(), datum.loss_fn_inputs["weights"].tolist()
    text = str(toy_tokenizer().decode([t for t, w in zip(targets, weights, strict=True) if w > 0]))
    assert "Hello" in text and "Goodbye" in text and "Bye" not in text.replace("Goodbye", "")
    assert "extension property" not in caplog.text


def test_a_renderer_that_drops_tool_calls_is_refused_and_probes_find_the_conversations_that_need_them():
    renderer: Any = toy_renderer()
    render_message = renderer.render_message
    # As llama3 renders an assistant message with tool calls: its text, which is empty, and its end.
    renderer.render_message = lambda message, ctx: render_message(
        {key: value for key, value in message.items() if key != "tool_calls"}, ctx
    )

    check_renders(renderer, CHAT)
    with pytest.raises(CannotRender, match="it renders the tool call of get_weather as '<.im_end.>'"):
        check_renders(renderer, TOOL)
    check_renders(toy_renderer(), TOOL)

    tools_only = Conversation(turns=CHAT.turns, tools=(WEATHER,))
    untold = Conversation(turns=CHAT.turns)
    assert probes([CHAT, CHAT, tools_only, TOOL, untold]) == [CHAT, tools_only, TOOL, untold]
    assert probes([TOOL, CHAT]) == [TOOL]
    assert probes([untold, TOOL]) == [untold, TOOL]


def test_cross_entropy_takes_the_mean_over_the_trained_tokens_and_reports_their_nll():
    datums = [supervised_datum(toy_renderer(), turns, mask_history=False, max_length=None) for turns in (CHAT, TOOL)]
    total = sum(trained_tokens(datum) for datum in datums)

    scaled = mean_per_token(datums)

    assert sum(trained_tokens(datum) for datum in scaled) == pytest.approx(1.0)
    assert [trained_tokens(datum) * total for datum in scaled] == pytest.approx([trained_tokens(d) for d in datums])
    assert [datum.model_input for datum in scaled] == [datum.model_input for datum in datums]

    objective = CrossEntropy()
    assert (objective.loss_fn, objective.loss_fn_config, objective.side_channels) == ("cross_entropy", {}, frozenset())
    # Every target token at log-probability -2.
    outputs = [{"logprobs": tinker.TensorData.from_torch(torch.full((d.model_input.length,), -2.0))} for d in scaled]
    result: Any = SimpleNamespace(loss_fn_outputs=outputs)
    assert step_nll(objective.metrics(scaled, result), scope="") == pytest.approx(2.0)


def test_a_step_s_nll_is_per_trained_token_over_all_its_microbatches():
    """The learner averages each metric over a step's microbatches: the step's NLL divides the averaged weighted sum
    by the averaged weight, which is the NLL per trained token of the whole step, not the mean of each microbatch's."""

    chat, tool = mean_per_token(
        [supervised_datum(toy_renderer(), turns, mask_history=False, max_length=None) for turns in (CHAT, TOOL)]
    )
    objective = CrossEntropy()

    def microbatch(datum: tinker.Datum, logprob: float) -> dict[str, float]:
        logprobs = tinker.TensorData.from_torch(torch.full((datum.model_input.length,), logprob))
        result: Any = SimpleNamespace(loss_fn_outputs=[{"logprobs": logprobs}])
        return objective.metrics([datum], result)

    first, second = microbatch(chat, -1.0), microbatch(tool, -3.0)
    averaged = {key: (first[key] + second[key]) / 2 for key in first}  # as the learner records them

    chat_tokens = trained_tokens(supervised_datum(toy_renderer(), CHAT, mask_history=False, max_length=None))
    tool_tokens = trained_tokens(supervised_datum(toy_renderer(), TOOL, mask_history=False, max_length=None))
    nll = step_nll(averaged, scope="")
    assert nll == pytest.approx((1.0 * chat_tokens + 3.0 * tool_tokens) / (chat_tokens + tool_tokens))
    assert nll != pytest.approx(2.0)  # the mean of the two microbatches' NLL
    assert averaged == {}  # step_nll takes the two metrics it divides out of the step's
