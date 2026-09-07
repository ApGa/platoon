"""OpenReward rollout wiring for the Tinker proxy: reasoning replay, tokenizer prefix, condenser kwargs.

Imports the real ``platoon.openreward.rollout`` module (openreward venv), unlike the stubbed loader in
``test_openreward_rollout_timeout.py``.
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pytest  # noqa: E402

rollout = pytest.importorskip("platoon.openreward.rollout")

from platoon.config_defs import RolloutConfig  # noqa: E402

TINKER_MODEL = "platoon-tinker/apurvaga/Qwen3.6-35B-A3B-preserve-thinking"


def test_openhands_resends_reasoning_content_for_preserve_thinking_models():
    from openhands.sdk.llm.utils.model_features import SEND_REASONING_CONTENT_MODELS, get_features

    assert SEND_REASONING_CONTENT_MODELS.count("preserve-thinking") == 1  # appended once at import
    assert get_features(TINKER_MODEL).send_reasoning_content
    assert get_features("openai/apurvaga/Qwen3.6-35B-A3B-preserve-thinking").send_reasoning_content
    assert not get_features("openai/Qwen/Qwen3.6-35B-A3B").send_reasoning_content
    # idempotent
    assert rollout.enable_openhands_reasoning_replay() is True
    assert SEND_REASONING_CONTENT_MODELS.count("preserve-thinking") == 1


def test_openhands_message_dict_carries_reasoning_content():
    from openhands.sdk.llm import LLM, Message, TextContent

    llm = LLM(usage_id="t", model=TINKER_MODEL, base_url="None", api_key="None")
    message = Message(role="assistant", content=[TextContent(text="done")], reasoning_content="why")
    dicts = llm.format_messages_for_llm([message])
    assert dicts[0]["reasoning_content"] == "why"


def test_custom_tokenizer_strips_platoon_tinker_prefix():
    config = RolloutConfig(model_name=TINKER_MODEL, model_endpoint="None", model_api_key="None")
    llm = rollout._build_llm(config)
    assert llm.custom_tokenizer == "apurvaga/Qwen3.6-35B-A3B-preserve-thinking"
    assert llm.model == TINKER_MODEL
    openai_cfg = RolloutConfig(model_name="openai/Qwen/Qwen3.6-35B-A3B", model_endpoint="http://x:1/v1")
    assert rollout._build_llm(openai_cfg).custom_tokenizer == "Qwen/Qwen3.6-35B-A3B"
    hosted = RolloutConfig(model_name="openai/openai/gpt-4o", model_endpoint="http://x:1/v1")
    assert rollout._build_llm(hosted).custom_tokenizer is None


@pytest.mark.parametrize("endpoint", ["None", None, "http://127.0.0.1:8000/v1"])
def test_condenser_extra_body_sent_unconditionally(endpoint):
    config = RolloutConfig(model_name=TINKER_MODEL, model_endpoint=endpoint, model_api_key="None")
    llm = rollout._build_condenser_llm(config, rollout.OpenRewardConfig())
    assert llm.litellm_extra_body == {
        "reasoning_effort": "high",
        "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": False},
    }
    assert llm.custom_tokenizer is None
    assert llm.max_output_tokens == 26214
    disabled = rollout._build_condenser_llm(config, rollout.OpenRewardConfig(condenser_disable_thinking=True))
    assert disabled.litellm_extra_body == {
        "reasoning_effort": "none",
        "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": False},
    }
