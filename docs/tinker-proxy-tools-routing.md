# Tinker proxy: tools, thinking split, chat_template_kwargs and R3 routing ids

Component "platoon-proxy-renderer" of the miles-tinker Qwen3.6 Toolathlon design (sections 8.1-8.2).
Files: `platoon/train/tinker/proxy.py`, `plugins/openreward/platoon/openreward/rollout.py`,
`platoon/train/tinker/renderers/qwen3_6_preserve_thinking.py` (pre-existing), tests
`tests/train/tinker/test_tinker_proxy_tools_routing.py`, `tests/test_openreward_rollout_tinker_llm.py`.

## Renderer registration (entry point)

`platoon.train.tinker.renderers.register()` runs at `platoon.train.tinker.proxy` import time and again inside
`register_tinker_llm()` immediately before `tinker_cookbook.renderers.get_renderer(...)`; both calls are
idempotent.  `rl.py` imports the proxy, so `train.renderer_name: qwen3_6_preserve_thinking` resolves in the
trainer without any change to `rl.py`.  `train.renderer_kwargs: {preserve_thinking, enable_thinking}` are plain
instance attributes set by `register_tinker_llm`.

## Request path (`TinkerLLM.acompletion` / `completion`)

litellm hands a `CustomLLM` everything OpenHands sent, verified against the openreward venv
(`litellm/utils.py:add_provider_specific_params_to_optional_params`, custom-provider branch):

| litellm kwarg | arrives as |
| --- | --- |
| `tools=[...]` (OpenHands native tool calling) | `optional_params["tools"]` (OpenAI function dicts) |
| `tool_choice` | `optional_params["tool_choice"]` |
| `extra_body={...}` (OpenHands `litellm_extra_body`) | `optional_params["extra_body"]` verbatim |
| `chat_template_kwargs={...}` (top-level) | `optional_params["chat_template_kwargs"]` |
| top-level `reasoning_effort`, `store` | dropped by `drop_params=True` (OpenHands default) |

`TinkerLLM._extract_render_request(kwargs) -> RenderRequest(tools, render_kwargs)`:

* `tools`: passed verbatim to `renderer.build_generation_prompt(messages, tools=...)` when the renderer accepts
  a `tools` kwarg (the platoon Qwen3.6 renderer; the prompt then equals
  `tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True)`); otherwise converted to
  flat cookbook `ToolSpec`s and folded into the leading system message via
  `renderer.create_conversation_prefix_with_tools`.  Tools are also passed to `renderer.parse_response(tokens,
  tools=...)` when supported (schema-typed argument coercion).
* `tool_choice`: only `None`/`"auto"`/`"none"` accepted; `"required"` or a named function raises `ValueError`.
* `render_kwargs`: `enable_thinking` / `preserve_thinking` from `extra_body.chat_template_kwargs`, overridden
  by a top-level `chat_template_kwargs`; `reasoning_effort == "none"` (top-level or in `extra_body`) forces
  `enable_thinking=False`.  Keys the renderer does not accept are logged and dropped.
* The first request with `max_tokens == 26214` (`condenser_max_completion_tokens`) logs its
  `optional_params` keys once (`First condenser-sized request ...`).

## Response path (`TinkerLLM._parse_response`)

Each sampled sequence becomes one `Choices` with `LitellmMessage(role, content=<visible text>,
reasoning_content=<think span or absent>, tool_calls=[ChatCompletionMessageToolCall(...)])`;
`finish_reason` is `"tool_calls"` when a tool call parsed, else `"length"` for `stop_reason == "length"`,
else `"stop"`.  Note litellm's `Message` deletes `reasoning_content` when it is `None`; OpenHands reads it
with `getattr(..., None)`.

## R3 routing id capture (contracts C1/C2)

`proxy.py` rebinds `tinker.types.SampleResponse` / `SampledSequence` (and the `tinker.types.sample_response`,
`tinker.types.sampled_sequence`, `tinker` module attributes) to frozen `extra="allow"` subclasses at import,
before any `SamplingClient` exists (`SamplingClient._sample_async_impl` resolves `types.SampleResponse` at
call time).  `TINKER_SUBPROCESS_SAMPLING` must be unset (asserted).  After sampling,
`TinkerLLM._routing_id_of(result)` reads `getattr(result.sequences[0], "routing_id", None)` and stores it as
`TinkerLLMInteraction.routing_id: int | None` (default `None`); `trajectory_to_data` carries it into
`Datum.loss_fn_inputs["routing_handle"]`.

## OpenReward rollout wiring (`rollout.py`)

* `enable_openhands_reasoning_replay()` (import time, idempotent) appends `"preserve-thinking"` to
  `openhands.sdk.llm.utils.model_features.SEND_REASONING_CONTENT_MODELS` (a plain list consulted per call), so
  OpenHands re-sends `reasoning_content` on historical assistant turns for
  `platoon-tinker/apurvaga/Qwen3.6-35B-A3B-preserve-thinking` and prompts extend turn over turn.
* `custom_tokenizer` strips `openai/` and `platoon-tinker/` before the hosted-model guard.
* `_build_condenser_llm` sends `extra_body={"reasoning_effort", "chat_template_kwargs": {enable_thinking,
  preserve_thinking: False}}` for every endpoint (previously only `http://`).
