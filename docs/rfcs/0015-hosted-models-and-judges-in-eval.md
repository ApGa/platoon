# RFC 0015: Hosted models and a judges switch for evaluation

| | |
|---|---|
| Status | Draft |
| Created | 2026-10-16 |
| Authors | Apurva Gandhi (@ApGa) |
| Builds on | The model gateway (`platoon/gateway/core.py`, `platoon/gateway/server.py`, `platoon/gateway/dialects/`, [Architecture](../architecture.md#the-model-gateway)), `platoon eval` (`platoon/eval/`), verifier grading (`platoon/harness/agents.py`, `platoon/harness/verify.py`), episode programs (`platoon/programs.py`), [RFC 0005](0005-gateway-dialects-and-tito.md), [RFC 0009](0009-gateway-step-budgets.md) |
| Discussion | The pull request that adds this RFC. 0013 is the extension model and 0014 the remote sandbox service's draft, so this draft takes 0015 |
| Implementation | M1 (the judges switch) and M2 (hosted models, rebuilt to this revision in place of the provider backends the first draft built) in the pull request that adds this RFC, as "What was built" lists; the answers to its two open questions in a follow-up pull request; M3 not yet |

## Summary

`platoon eval` and `platoon rollout` can sample the policy only through Tinker. This RFC lets an evaluation config
name a **hosted model** instead: a `base_url` that speaks the harness's own API dialect (OpenAI Chat Completions,
OpenAI Responses or Anthropic Messages), the model it serves, and the environment variable that holds its key.
Harnesses keep talking to platoon's model gateway in their own dialect, exactly as in training. The gateway forwards
each request to the base URL unchanged except for a few listed fields, and returns the reply unchanged. Tools
behave as in training: the hosted model gets the tools a trained policy would get, and a request that training
refuses is refused with the same error. Platoon still records trajectories, enforces step limits, reports usage,
applies the judges switch and keeps the key out of the agent's sandbox. A hosted call records text, tool calls and
usage instead of token ids, so hosted models serve evaluation only, and training configs cannot name them. The RFC
also adds `inference.judges`. This one switch turns off the model calls that judge agents rather than solve the
task. It says what replaces each of those calls, and records the choice in `lock.json` and the report.

## Motivation

**Evaluation is Tinker-only.** `EvalPolicy.install` always calls `register_tinker_llm`, which opens a Tinker
`ServiceClient` (`platoon/eval/config_defs.py`, `platoon/train/tinker/proxy.py`). Without a `TINKER_API_KEY` that
starts with `tml-` (`require_tinker_api_key`), `platoon eval` exits 78. It also refuses harness rollouts unless
`inference.renderer_name` is set (`platoon/eval/run.py`). So a baseline such as "the same harness and tasks with a
frontier API model, or with an open model on a vLLM server" cannot be run. Before the gateway, platoon evaluated
exactly that. In the RAO snapshot, an evaluation config named `model_name`, a `model_endpoint` and a key, and the
rollouts called that endpoint. The endpoint was a local SGLang server's `.../v1`, or a LiteLLM proxy in front of
hosted models (`platoon/inference/config_defs.py` and `plugins/*/configs/inference/` there).

**Calling providers around the gateway is not an answer.** A harness that talks to a provider directly loses what
the gateway gives every call:

- the agent identity and tree;
- the `policy`, `aux` and `judge` call kinds;
- step budgets;
- the per-call records the trace builds trajectories from;
- the judges switch.

It also needs the provider key inside its sandbox.

**Provider backends are the wrong unit.** The first draft of this RFC gave the gateway two provider backends,
`openai` and `anthropic`, built on their SDKs. A request was passed through when the harness spoke the backend's API
and translated otherwise. That took a translation for each pair of dialects and a replay store for Anthropic
thinking signatures. It took per-provider defaults, a cost header for one proxy and a usage field for one router.
It also put two SDKs between platoon and the wire. Azure OpenAI, Amazon Bedrock and Google Vertex AI still needed a
proxy. Review asked for what the RAO snapshot had: one base URL that speaks the harness's dialect, with the gateway
in between, as in training. A harness already speaks its dialect, and servers that speak each dialect exist for
every model worth evaluating, directly or through a proxy such as LiteLLM. So the gateway does not need to know
providers at all.

**Judges cost latency and money that evaluation does not need.** Evaluation measures the root: a rollout succeeds
when the reward its environment scored reaches `success_threshold`, before any reward processor
(`_default_success_fn_factory` in `platoon/eval/workflow.py`). Verifier agents run up to `agents.verifier.max_steps`
steps (default 20) for every completed child, before the parent sees the child's result, so they sit on the critical
path. Turning them off today takes `inference.workflow.rollout_config.enable_subagent_verifiers: false`, or
`agents.verifier.enabled: false` on each entry. Program judges (`ep.judge_endpoint()`) and parent-usage judges have
no switch at all. When verifiers are off, children silently share the episode's outcome, and nothing in `lock.json`
or the report says so. Verifier trajectories still count in the report's step and subtrajectory statistics
(`_compute_depth_stats`), so reports with and without verifiers are not comparable.

## What was built

M1 and M2 are built as proposed below, with these details and differences:

- **The seam.** `HostedGateway` (`platoon/gateway/hosted.py`) is a `ModelGateway` whose agents, keys, call kinds,
  budgets, judges, recording and shutdown are inherited unchanged. `forward` serves the HTTP front: the server hands
  it every model call and `count_tokens` on a hosted gateway, and relays the `Relay` it returns. `chat` serves
  `ctx.chat`. Each dialect module holds its own `hosted_body` (the request's changes) and `reply_fields` (what a
  record takes from a reply); `in_process_body` builds `ctx.chat`'s body from `ChatRequest.params`.
- **One set of tool rules.** `check_tool_choice`, `chat_tools` and `tool_function` (in `platoon/gateway/core.py`,
  which `RenderRequest.from_params` calls), `tool_functions` and `check_assistant_blocks` (Messages), and
  `function_tools` and `check_input_items` (Responses) are the training dialects' own code, which `chat_request` and
  `hosted_body` both call. Chat Completions runs `RenderRequest.from_params` itself and sends the tools it made.
  `tests/gateway/test_hosted_tools.py` has a case for each row of the tool table: the tools each path yields, the
  tools the chat-template renderer and a cookbook renderer are handed over HTTP next to those the hosted model
  receives, and refusals with the same message, status and body over HTTP.
- **One normalization.** Both paths show a tool as a function of its name, its description and its parameters, with
  the empty-object schema for one that gives none; a description is kept when the tool gives one, an empty one
  included. The fields keep the order training always showed them in, since a chat template may write a tool out as
  JSON: a Chat Completions tool's own order (a function tool's `type` and `function` too, which some clients write
  function first), and name, description, parameters for Messages and Responses tools, whatever order the client wrote
  them in. A Messages entry without a name (an MCP toolset) and a Responses function entry without one are left out,
  and a Chat Completions entry without one is refused with a 400. These were the pass-through's rules first; training
  adopted them as open question 2's answer, so the chat-template renderer gets the same functions as the cookbook
  renderers and a hosted model.
- **User turns.** Training refuses every Messages user block but text and tool results; the pass-through refuses,
  with the same message, only the blocks of tools training never offers (`container_upload`, and server-tool or MCP
  blocks), and forwards images, documents and search results.
- **Truncated replies.** Over HTTP a reply cut off at its token limit is relayed unchanged and its record keeps its
  tool calls. `ctx.chat`'s reply emits none, as a Tinker one does not: its calls are counted
  (`parsed_tool_calls`), not run.
- **The HTTP client.** One aiohttp session per event loop, with no connection limit of its own (the request limit is
  `max_concurrent_requests`), no cookie jar, and `trust_env` off: aiohttp's `trust_env` would also read `~/.netrc`
  and send its credentials to the base URL's host (and refuse them beside an `Authorization` header). The gateway
  reads the proxy for a URL from `HTTPS_PROXY`, `HTTP_PROXY`, `ALL_PROXY` and `NO_PROXY` itself, http proxies only, as
  aiohttp supports.
- **Errors.** `HostedCallError`, `HostedRefusedError`, `HostedRequestError` and `ProviderContextWindowError` live in
  `platoon/gateway/core.py` beside the gateway's other errors, so the dialects map a failed call to its timeout or
  server error without importing the hosted module, which re-exports them.
- **Retries.** `retry-after-ms` is honoured before `retry-after`, and both are relayed with a failed reply. A reply
  that succeeded but cannot be read (not JSON, or a stream without its last event) is retried as a server error.
- **`count_tokens`** is never counted as a failed call, whatever it is answered: it is not a model call, and a server
  without the route answers 404, which must not stop a run.
- **The probe** makes up to three attempts of at most `min(timeout_s, 120)` seconds. A base URL that answers nothing
  is `did not answer the <dialect> probe`, where a reply is `refused the <dialect> probe`. A run whose entries have
  no harness (custom rollouts) speaks no dialect, and probes nothing.
- **Configuration.** `inference.hosted` refuses a `base_url` with credentials, a query or a fragment, and an
  `api_key_env` that is not a variable's name (letters, digits and underscores), so a key pasted into either is
  refused with a message that does not repeat it. The first draft's `inference.backend` and its six keys under
  `hosted` are unknown keys whose messages name their replacement (section 11), and training configs refuse `hosted`
  and `backend` in `train` and `eval`, and `inference`.
- **Review fixes.** An adversarial review of the rebuild found and fixed these:
  - **Redirects.** aiohttp follows redirects by default and keeps Messages' `x-api-key` header on the way to another
    origin, so a base URL that answered 307 sent the key and the request there. The gateway now follows no redirect,
    and a 3xx reply refuses the run like a 404 (section 7).
  - **Chat Completions tools.** The pass-through forwarded Chat tools as listed, so a tool of another type (a
    provider's built-in tool, or an Anthropic server tool that a LiteLLM proxy accepts in Chat Completions) could run
    at the provider, and `defer_loading` could hide a tool. Each Chat tool is now sent as a function of its name,
    description and parameters, as the cookbook renderers read every tool (section 4). One without a name was left
    out at first; it is now refused with training's 400, as open question 2's answer has it.
  - **The request limit.** Worker processes got at least one request each, so a limit below their number was
    exceeded. Such a limit is now refused (exit 78), and the processes counted are those that run rollouts at once
    (section 7).
  - **Keys.** A key is read without the whitespace around it (a header cannot carry the newline a variable read from
    a file may end in, and the probe crashed on it), and is redacted in its `\/`-escaped and percent-encoded forms
    too (section 9). DeepSeek's `prompt_cache_hit_tokens`, a provider-specific usage field, is no longer read.
- **Dependencies.** `pyproject.toml` and the three lockfiles are `main`'s again: the `anthropic` extra is gone, and
  nothing under `platoon/` imports `openai` or `anthropic`.

The `tests/hosted_agent` test agent speaks Messages and Responses over HTTP, streamed, so `platoon eval` and
`platoon rollout` run number search end to end in every dialect, besides `platoon-react` (`ctx.chat`) and
`platoon-bash` (Chat Completions over HTTP). M3 (evaluation without the `tinker` extra) is not built. Both open
questions are answered: judges sample the policy's base URL (section 10), and training adopted the pass-through's
tool normalization (section 4). The user guide is [Evaluate a hosted model](../how-to/evaluate-hosted-models.md).

## Proposal

### 1. A base URL in the harness's dialect

A hosted model sits behind the same gateway, agents, step budgets and records as a Tinker policy. Only the last hop
changes:

1. The harness calls the gateway in its own dialect with its agent key, exactly as in training.
2. The gateway authenticates the agent, checks the call kind and the judges switch, and normalizes the request's
   tools as training does (section 4).
3. It sets the model and fills the sampling parameters the request leaves unset from the rollout's defaults
   (section 3).
4. It sends the request, in the same dialect, to `base_url` with the provider key, retrying what failed
   (section 7).
5. It records the call on the agent (section 6) and returns the provider's reply to the harness unchanged.

Nothing is translated between dialects. A base URL must speak the dialect of every caller that uses it:

- each harness the run's environments route to, verifier harnesses included (a CLI harness spec's `dialect`);
- Chat Completions for function harnesses, whose `ctx.chat` is a Chat Completions call in process (section 5);
- the dialect of any judge code that calls through a `ModelEndpoint` while judges are on.

One server that speaks several dialects covers a mixed run. OpenAI's API speaks Chat Completions and Responses, and
a LiteLLM proxy speaks all three. The probe checks each dialect the run uses before any rollout (section 8), so a
base URL that lacks one is an exit 78, not a run of errored rollouts.

### 2. Configuration

The hosted model is the `hosted` section of an evaluation config's `inference:`. Without it the policy is Tinker's,
as today.

```yaml
inference:
  model_name: gpt-5.1                     # the model the base URL serves; every request names it
  hosted:
    base_url: https://api.openai.com/v1   # where the dialect paths go: {base_url}/chat/completions, ...
    api_key_env: OPENAI_API_KEY           # the variable that holds the key (leave it out for a server without auth)
  judges: true                            # section 10
  output_dir: runs/eval/gpt-5.1
```

Four optional settings earn their place. They are generic, and each has a default that works:

| Key | Default | Why it exists |
|---|---|---|
| `timeout_s` | 600 | How long one HTTP attempt may take, from sending to the reply's last byte. A slow server with long generations needs more, and a call must not hang a rollout forever |
| `max_retries` | 6 | Retries of 408, 409, 429, 5xx, connection errors, timeouts and broken streams (section 7). Throttling should cost waiting, not rollouts, and the right number depends on the account's limits |
| `max_concurrent_requests` | null (no limit) | Requests in flight for the whole run, shared out among worker processes, so it is at least their number. Rate limits are per account, and rollout concurrency does not bound the parallel calls of subagents |
| `pricing` | null | USD per million tokens (`input`, `output`, `cached_input`, `cache_write`), for the report's cost (section 6) |

Examples of base URLs:

| Serving | `base_url` | `api_key_env` | Dialects it speaks |
|---|---|---|---|
| OpenAI | `https://api.openai.com/v1` | `OPENAI_API_KEY` | Chat Completions, Responses |
| Anthropic | `https://api.anthropic.com/v1` | `ANTHROPIC_API_KEY` | Messages |
| vLLM or SGLang on this host | `http://localhost:8000/v1` | none | Chat Completions, and whatever else the server serves |
| A LiteLLM proxy | `http://<proxy>:4000/v1` | the proxy key's variable | All three, for every provider the proxy reaches: Azure OpenAI, Amazon Bedrock, Google Vertex AI and more |

Model settings that a harness does not send, such as reasoning effort or a chat template switch, belong in the
harness's own configuration or in the proxy's model configuration. The gateway sends the harness's body.

Checks when the config loads (exit 78, before any task is listed):

- `base_url` is an `http` or `https` URL. `timeout_s` is positive, `max_retries` is not negative,
  `max_concurrent_requests` is positive and prices are not negative.
- A hosted model refuses the settings that configure Tinker rendering: `renderer_name`, `model_path`,
  `renderer_kwargs` and `context_window_length`. With it, `platoon rollout` refuses `--renderer` and `--model-path`.
- The variable that `api_key_env` names must be set and non-empty, unless `stage: report`.
- Training configs have no `inference` and no `hosted` section. `platoon train` refuses both as unknown keys, and
  the message adds that hosted models serve evaluation only, because their records hold no token ids or logprobs.
- The first draft's `inference.backend` is an unknown key, and its message names `inference.hosted`. Under
  `hosted`, the first draft's `kind`, `require_api_key`, `headers`, `headers_env`, `extra_body` and `drop_params`
  are unknown keys too, each with a message that says what replaces it (section 11).

### 3. How each dialect is forwarded

| The harness calls the gateway at | The gateway calls | The key goes as | Client headers forwarded |
|---|---|---|---|
| `POST /v1/chat/completions` | `POST {base_url}/chat/completions` | `Authorization: Bearer` | `openai-beta` |
| `POST /v1/responses` | `POST {base_url}/responses` | `Authorization: Bearer` | `openai-beta` |
| `POST /v1/messages` | `POST {base_url}/messages` | `x-api-key` | `anthropic-version`, `anthropic-beta` |
| `POST /v1/messages/count_tokens` | `POST {base_url}/messages/count_tokens` | `x-api-key` | `anthropic-version`, `anthropic-beta` |
| `GET /v1/models` | nothing: the gateway lists `inference.model_name`, as in training | | |

Rules for the path, query and headers:

- A base URL ends where the gateway's `/v1` ends, so Anthropic's is `https://api.anthropic.com/v1`.
- The path form `/s/<agent key>/v1/...` is forwarded the same way.
- A query string, such as `?beta=true`, is forwarded unchanged.
- Without `api_key_env`, no key header is sent.
- No other client header is forwarded. This covers the agent key, `X-Platoon-Call-Kind`, the account selectors
  `OpenAI-Organization` and `OpenAI-Project` (the account is the key's), user agents and hop-by-hop headers.
- Routes the training gateway does not serve are not forwarded either, such as `/v1/responses/compact` or a stored
  response's URL. They get 404, as in training.
- No redirect is followed, so the key and the request go to the base URL and nowhere else. A 3xx reply refuses the
  run (section 7).

**The body.** The gateway changes only these fields:

1. `model` becomes `inference.model_name`. The training gateway also serves one model whatever a request names.
2. Tool fields are normalized as training normalizes them (section 4).
3. Sampling parameters are resolved with training's precedence: the request's own value, then the rollout's
   `inference_params`. A parameter the request leaves unset and `inference_params` sets is added under the dialect's
   name: `max_completion_tokens`, `max_tokens` (Messages) or `max_output_tokens` (Responses), plus `temperature`
   and `top_p`. Nothing else is added. The gateway's own defaults (4096 completion tokens, temperature 1, top_p 1)
   stand in for a renderer's sampler, and a hosted model's defaults are its provider's. The completion budget is
   not clamped to a context window, since a hosted call has no tokenizer.
4. Fields that keep or edit a conversation on the provider's side are removed, because training never reads them.
   They are Responses `previous_response_id`, `conversation`, `prompt` (a stored prompt), `background` and
   `truncation`, and Messages `context_management` and `compaction`. The model then sees exactly the conversation
   the harness sent and the trajectory records, answered in one exchange.
5. A Chat Completions `n` other than 1 is refused with training's error, since a call is one step with one answer.

Everything else passes unchanged:

- messages and system prompts, with images, documents and `cache_control`;
- thinking blocks with their signatures, and encrypted reasoning items;
- stop sequences, response formats, logprobs, reasoning and thinking settings;
- `stream` and `stream_options`, metadata, service tiers and prompt-cache keys.

**The reply.** The status, content type and body (JSON, or the SSE bytes) go back as the provider sent them, with a
`retry-after` header when the provider sent one. The gateway adds its `x-platoon-call-id`, as in training. The one
exception is a reply that echoes the key, which is redacted (section 9).

**Streaming.** Training answers `stream: true` with SSE once the whole completion is sampled, with `: keepalive`
comments every 10 seconds while it samples. A hosted call works the same way:

- The gateway reads the provider's whole reply, stream or JSON, before it answers. It can then record the call and
  retry an attempt that broke midway, since the harness has seen nothing of it.
- Then it writes the provider's bytes unchanged.
- The response status is sent only when the first keepalive is due. So a provider error that comes sooner, as every
  invalid request does, reaches the harness with its own status and body.
- A call that fails after the stream has started gets the dialect's error event, as in training.

**Token counts.** `count_tokens` goes to the provider with the same model, tools and removals as a Messages request.
Its reply is returned unchanged. It is not recorded and takes no step, as in training, where the renderer counts.

### 4. Tools behave as in training

The training gateway decides at its dialect layer which tools a policy gets and how it may call them. This layer is
`RenderRequest.from_params` and each dialect's `chat_request`, before any renderer. For a hosted model, that layer
is the specification:

- **Which tools.** A trained policy gets the request's own tools as a name, a description and a parameter schema,
  whatever its renderer. For Chat Completions that is every entry, as a function of those three fields read from its
  `function`, else from its top level. For Messages it is every entry with a name, mapped to those three fields. For
  Responses it is the `function` entries with a name. Nothing runs on the model's side: there are no server
  tools, no MCP connector and no tool search, and deferred tools are shown up front. The hosted model gets the same
  tools, written in the dialect's own shape.
- **How it may call them.** Training can force no call, and refuses every `tool_choice` except `auto` and `none`,
  each with a 400. It enforces neither `none`, nor single calls, nor strict schemas. So the hosted request is
  refused in the same cases, with the same message, status and error shape. It carries no setting that would
  constrain calls more than training does.

The same functions implement both paths, so the two cannot drift: the check of `tool_choice`, the tool
normalization of each dialect, and the history checks. A parity test runs each request of the table below through
the training dialect and through the pass-through, and compares the tools each yields. Over HTTP, it also compares
the tools the chat-template renderer and a cookbook renderer are handed with those the hosted model receives. The
normalization has no exception between the two paths (open question 2's answer):

- Every tool is a function of its name, its description and its parameter schema, which every renderer and the
  hosted model get. One without a schema gets the empty-object schema that cookbook renderers always used
  (`to_cookbook_tool_specs`), which the Messages and Responses APIs require.
- No model can call a nameless tool. A Messages entry without a name (a toolset) and a Responses function entry
  without one are left out. A Chat Completions entry without one is refused with the 400 the cookbook renderers
  always gave it.

| Feature | Training (the Tinker gateway) | Hosted pass-through |
|---|---|---|
| Chat Completions `tools` | Each entry, whatever its type, becomes `{type: function, function: {name, description, parameters}}` of the fields its `function`, or else its top level, gives, with the empty-object schema for one without (`RenderRequest.from_params`, `chat_tools`). The chat-template renderer gets these functions, and cookbook renderers the same three fields (`to_cookbook_tool_specs`). A value that is not a list, and an entry that is not an object or has no name, is a 400 | The same functions, sent as they are, and nothing else; the same 400s |
| Messages `tools` | Every entry with a name, whatever its `type`, becomes a function of its `name`, its `description` and its `input_schema` as parameters, with `{"type": "object", "properties": {}}` for one without; every other field is dropped, a nameless entry is left out, and a value that is not a list or an entry that is not an object is a 400 (`tool_functions`) | Each function is sent as `{name, description, input_schema}` and nothing else |
| Responses `tools` | Only `function` entries with a name are kept, as functions of their name, description and parameters, with the empty-object schema for one without (`function_tools`); every other type is dropped | Function entries are sent as `{type: function, name, description, parameters, strict: false}`; the rest are left out |
| Server and built-in tools: Messages `web_search_*`, `web_fetch_*`, `code_execution_*`, `bash_*`, `text_editor_*`, `computer_*`, `memory_*`, advisor; Responses `web_search`, `file_search`, `code_interpreter`, `computer*`, `image_generation`, `local_shell`, `shell`, `apply_patch`, `custom`, `namespace`, `programmatic_tool_calling`; Chat tools of a type other than `function` (a provider's built-in such as `browser_search`, an Anthropic server tool a proxy accepts, a `custom` tool); Chat `web_search_options`; Messages `container` | Messages: a function of the tool's name with the empty-object schema, which nothing runs on the model's side; a call of it goes to the harness like any other. Responses: dropped. Chat: a function of the name, description and parameters the entry gives, which nothing runs; a nameless one (`{"type": "browser_search"}`, or a `custom` tool, whose name is under `custom`) is a 400. `web_search_options` and `container`: never read | The same: Messages server tools go as custom tools of their name, and Chat entries of other types as functions of their name, so the provider runs none; a nameless Chat entry is refused with training's 400; Responses built-ins are left out; `web_search_options` and `container` are removed |
| MCP connector: Messages `mcp_servers` and `mcp_toolset` entries; Responses `mcp` tools | `mcp_servers` is never read, an `mcp_toolset` entry, which has no name, is left out, and `mcp` tools are dropped: no remote MCP server reaches the policy | `mcp_servers` is removed, toolsets are left out (nameless), and `mcp` tools are left out. Environment tools reach agents through platoon's MCP tool plane, as in training |
| Tool search and `defer_loading` | `defer_loading` is dropped with the other fields, so every deferred tool is rendered up front. A Messages tool-search tool (`tool_search_tool_regex_*`, `tool_search_tool_bm25_*`) becomes a function of its name with the empty-object schema. Responses `tool_search` is dropped | The same: `defer_loading` is removed and every tool is sent up front; the tool-search tool goes as a custom tool of its name; Responses `tool_search` is left out. The provider never searches |
| `tool_choice` | `auto` and `none` are accepted and then unused (the gateway's request holds no tool choice). Anything else is a 400 `invalid_request_error`: "tool_choice=... is not supported by the model gateway (only 'auto'/'none'): forced or named tool calls cannot be enforced by the renderer". This covers Chat `required` and named or allowed-tools objects, Messages `any` and `tool` (the object's `type` is checked), and Responses `required` and objects | The same refusals, with the same message, status and error shape, before anything is sent; `auto` and `none` are removed, so the provider's default (`auto`) applies, as in training, where the policy may call any tool |
| `strict` | Never enforced: sampling has no schema constraint, and every dialect drops the field, so no renderer sees it | Calls stay unconstrained: removed from Chat functions and Messages tools, and `strict: false` on Responses function tools, which carry the field. A Chat `custom` tool's grammar never reaches the provider either (the tool is refused) |
| `parallel_tool_calls` (Chat, Responses), `disable_parallel_tool_use` (Messages `tool_choice`), Responses `max_tool_calls` | Never read: an answer yields every call the parser finds, and the Responses dialect reports `parallel_tool_calls: true` | Removed, so the model may call several tools at once, as a trained policy may |
| Other fields of a tool: `cache_control`, `input_examples`, `eager_input_streaming`, `allowed_callers` (Messages), `output_schema` (Responses), any field of a Chat tool or its `function` but the three | Dropped: only name, description and schema are kept, for every renderer | Removed. `cache_control` on system prompts and messages is not a tool field and passes unchanged |
| Legacy Chat `functions` and `function_call` | Never read: the policy gets no tools from them | Removed |
| History with calls of tools training never offers: Messages `server_tool_use`, `*_tool_result`, `mcp_tool_use`, `mcp_tool_result` and `container_upload` blocks; Responses items other than messages, reasoning, `function_call` and `function_call_output` (such as `custom_tool_call`, `local_shell_call`, `web_search_call`, `mcp_call` and `item_reference`) | A 400: "Unsupported assistant content block type ..." or "Unsupported input item type ..." | The same refusal and message. The pass-through never gives a provider such tools, so only a fabricated history holds them |
| Tool calls and results in history: Chat `tool_calls` and `role: tool`, Messages `tool_use` and `tool_result`, Responses `function_call` and `function_call_output` | Rendered. A Messages `tool_result` keeps its text only: `is_error` is dropped and an image refused | Forwarded unchanged: results are conversation content, not tools (see images below) |

The request features other than tools:

| Feature | Training (the Tinker gateway) | Hosted pass-through |
|---|---|---|
| `model` | Ignored: the gateway serves one model, names it in every answer and lists it at `/v1/models` | Replaced by `inference.model_name`; `/v1/models` answered by the gateway; the reply's own `model` (a dated snapshot, or a router's choice) is returned and recorded as the served model |
| Sampling parameters | The request's own value (`max_completion_tokens` before `max_tokens`, `temperature` from 0 to 2, `top_p` from 0 to 1, `top_k`, `seed`, `stop`; out of range is a 400), then the rollout's `inference_params` (temperature, top_p, completion budget), then the gateway's defaults (4096 tokens, temperature 1, top_p 1, top_k -1) | The request's own values unchanged (the provider validates them), then `inference_params` added under the dialect's names; no gateway defaults (section 3) |
| Completion budget and context window | The budget is clamped to what the prompt leaves of `context_window_length`. A prompt that leaves nothing gets the dialect's context-length error and counts in `context_overflows` | No clamp and no default (every Messages client sends `max_tokens`, which the API requires). The provider's context-length error is returned unchanged, as the native error harnesses already handle, and counted in `context_overflows`. `context_window_length` is refused with a hosted model |
| `n` | Anything but 1 is a 400: "Only n=1 is supported" | The same |
| Stop sequences | Chat `stop` (a string or a list) replaces the renderer's stop tokens. Messages `stop_sequences` is a 400: "stop_sequences are not supported: they would replace the renderer's end-of-turn stops" | Forwarded unchanged. The refusal protects a renderer's end-of-turn tokens, and a provider adds stop sequences to its own end of turn |
| Images and documents | Refused by the text-only stack: the chat-template renderer refuses Chat image parts (a 400), and the Messages and Responses dialects refuse `image`, `document`, `input_image` and `input_file` | Forwarded unchanged. The refusal is a limit of the renderer, not a rule about what a policy may see. A hosted model renders its own input, and refusing would only make such tasks impossible to evaluate |
| Thinking and reasoning | Messages `thinking` becomes the reasoning text and `redacted_thinking` is dropped. Signatures are never checked, and answers carry the gateway's own. `thinking: {type: disabled}` turns thinking off, and other thinking settings are ignored. Responses reasoning items decode the gateway's sealed text, and `reasoning.effort: none` is honoured. Chat `reasoning_content` reaches the renderer, and `reasoning_effort: none` and `chat_template_kwargs` are honoured | Forwarded unchanged both ways. The harness replays the provider's own thinking blocks with their signatures, or its encrypted reasoning, and the thinking and effort settings reach the provider. As in training, the model sees its earlier reasoning the way its API defines. Nothing is stored or replayed by platoon |
| Conversation state and edits on the provider's side: Responses `previous_response_id`, `conversation`, `prompt`, `background`, `truncation`; Messages `context_management`, `compaction` | Never read: the policy sees exactly the request's messages, answered in one exchange | Removed (section 3) |
| Streaming | `stream: true` gets SSE: keepalive comments every 10 seconds, then the whole answer's events. Chat `stream_options.include_usage` adds the usage chunk. A failure after the stream started is the dialect's error event | `stream` and `stream_options` are forwarded unchanged. The provider's whole reply is read, then returned byte for byte, with keepalives after 10 seconds and the status sent when the first keepalive is due (section 3) |
| `count_tokens` | The training renderer counts the rendered request, with the dialect's refusals. It is not recorded | Forwarded with the Messages request's changes; the reply is returned unchanged; not recorded. A server without the route answers with its own error |
| Headers | Only the agent key and `X-Platoon-Call-Kind` are read | `anthropic-version`, `anthropic-beta` and `openai-beta` are forwarded, since they are part of the dialect's request (the Messages API refuses a request without a version). Nothing else is forwarded (section 3) |
| Every other field: response formats, `logprobs`, verbosity, `metadata`, service tiers, prompt-cache keys, `store`, `include`, `cache_control` on system prompts and messages | Mostly never read. Chat `logprobs` returns the sampled tokens' logprobs | Forwarded unchanged |
| Tool calls of an answer cut off at its token limit (the reply side) | Never emitted, so a truncated answer runs no tool | The reply is returned unchanged, as the provider sent it. The record keeps the calls it holds, so the trajectory shows what the harness received |

### 5. Calls made in process

A function harness's `ctx.chat(messages, tools=..., **params)` is a Chat Completions call in process: its keyword
parameters are the body's fields. The gateway keeps them on the request (`ChatRequest.params`, set by
`ChatRequest.from_params`) and builds the body from the messages and those parameters. It merges `extra_body` into
the body, as the OpenAI SDK sends it. It then makes the changes of sections 3 and 4 and sends the body to
`{base_url}/chat/completions`, unstreamed. The answer becomes the `ChatReply` that `ctx.chat` returns, as with
Tinker. A failure raises what a Tinker call raises:

- `ContextWindowExceededError` for a context-length refusal, so the harness condenses;
- an error that is a `ValueError` for an invalid request;
- `JudgesOffError` for a refused judge call;
- a `GatewayError` for a call that failed after its retries.

Judge code (behavior and parent-usage judges, and programs' judges) calls through a `ModelEndpoint` over HTTP, in
whatever dialect it speaks.

### 6. Records and the report

Hosted calls are recorded on their agents like Tinker calls, so the trace builds the same trajectories. Each
`policy` or `aux` call is one step, and `judge` calls go under `judge_calls`. Every viewer and reward processor keeps
working. The step's message is read from the reply in the dialect it came in:

- Chat Completions: the message's content, `reasoning_content` and `tool_calls`;
- Messages: text, thinking and `tool_use` blocks;
- Responses: output text, reasoning and `function_call` items.

| Recorded per call | Tinker | Hosted |
|---|---|---|
| Assistant message: content, `reasoning_content`, tool calls in OpenAI's shape | Yes | Yes, read from the reply; reasoning as the provider returns it, which may be empty |
| Finish reason and the raw stop reason | Yes | Yes. A refusal (`stop_reason: refusal`, `finish_reason: content_filter`, a refusal part) is recorded as one and scored like any answer |
| Usage: prompt, completion and total tokens | Counted from token ids | As the reply reports it, plus cached-prompt, cache-write and reasoning tokens when reported. A Chat Completions stream that did not ask for `include_usage` reports none: the call records `usage_reported: false`, and the report counts it |
| Prompt and sampled token ids, logprobs, routing id | Yes | No |
| Served model | The configured name | The reply's `model`, next to the configured one |
| Dialect, response id and request id | No | Yes |
| Cost in USD | No | `pricing` applied to the usage, else none |
| Attempts and rate-limited responses | No | Yes |
| Timing spans | Yes | Yes |

`action_misc` of each step gains `kind` for calls that are not `policy` calls, and these hosted fields. A Tinker
trajectory's policy steps stay as they are. The report (`final_report.json`) gains these blocks:

- `model`: the backend (`tinker` or `hosted`), the base URL's origin, and the served models with their call counts;
- `usage`: calls and tokens by kind (`policy`, `aux`, `judge`), with verifier agents' calls counted apart, as totals
  and per-rollout means;
- `cost`: the total and the per-rollout mean in USD, and the calls left unpriced;
- `backend`: attempts, rate-limited responses, calls that failed after retries, refusals, calls without usage, and
  the rollouts that had a failed call;
- `judges` (section 10).

Each rollout's `metadata.json` carries its usage, cost, failed calls and the run's provenance (section 9), so
`stage: report` rebuilds all of this from disk.

### 7. Failures, retries, deadlines and concurrency

- **Retries.** The gateway retries a failed attempt up to `max_retries` times. A failed attempt is any of these:
  - a 408, 409, 429 or 5xx reply (529 included);
  - a connection error;
  - an attempt that took more than `timeout_s`;
  - a stream that ended before its last event or with an error event.

  The backoff doubles from 0.5 seconds up to 8 seconds, with jitter. When a reply asks for a wait with
  `retry-after-ms` or `retry-after`, the gateway waits that long, up to 60 seconds. Every kind of failure counts
  against one budget, so a call makes at most `max_retries + 1` attempts and ends within about
  `(max_retries + 1) × timeout_s` plus the backoff. A Tinker sample is bounded the same way, by
  `TINKER_SAMPLE_TIMEOUT_SECONDS`.
- **A call that fails after its retries** goes back to the harness as the provider's last reply. When there was no
  reply, it is the dialect's timeout or server error. The agent counts the failure (`backend_failures`). The rollout
  is then errored with failure kind `backend_unavailable`, whatever the harness did next. An outage must not be
  scored as the model's failure, and `retry_errored` runs the rollout again on resume.
- **A refused run.** A model call answered with 401, 402, 403 or 404 means the key, the billing, the permission, the
  model or the dialect's route is wrong for every call. So does a redirect (3xx), which the gateway never follows: an
  HTTP client that follows one sends the request, and an `x-api-key` header with it, to wherever it points. The
  evaluation stops after the rollouts in flight, as `fail_fast` does, and resumes once the problem is fixed.
  `count_tokens` is not a model call and never stops a run.
- **Invalid requests** (other 4xx replies) are the harness's. They go back unchanged and are counted as rejected,
  not as outages. A context-length refusal is recognized by its status (400, 413 or 422) and its wording. OpenAI's
  `context_length_exceeded`, Anthropic's "prompt is too long" and the wordings of vLLM and SGLang are each tested
  against recorded bodies. It goes back unchanged and counts in the agent's `context_overflows`. So a harness that
  dies of it is graded, or excluded with `on_context_overflow: exclude`, exactly as under Tinker.
- **Concurrency.** `max_concurrent_requests` bounds the requests in flight for the whole run. With
  `use_subprocesses`, each worker process that runs rollouts at once (`subprocess_max_workers`, at most
  `num_concurrent_workers`) gets an equal share, rounded down. A limit below their number is refused (exit 78), since
  each process makes a request at a time at least. Rollout concurrency (`num_concurrent_workers`) is unchanged.
- **Event loops.** A subprocess worker runs each rollout under its own `asyncio.run`, and an HTTP session belongs to
  the loop that made it. So the gateway keeps one session per running loop, and closes it with the loop.

### 8. The probe

Before the first rollout, `platoon eval` and `platoon rollout` send one minimal request in each dialect the run
uses, for `inference.model_name`:

- a one-message Chat Completions request with `max_completion_tokens: 16`;
- a Messages request with `max_tokens: 16` and an `anthropic-version` header;
- a Responses request with `max_output_tokens: 16` and `store: false`.

The dialects come from the routes of the run's environments: each CLI harness spec's `dialect`, verifier harnesses
included, and Chat Completions for function harnesses. A failure is one line and exit 78. The line names the
dialect, the base URL's origin, the status and the provider's message, with the key redacted. Typical failures are a
wrong key (401), a model the server does not serve, and a base URL without the dialect's route (404). The served
model of each dialect goes to `lock.json`. Judge code is not probed, since its dialect is its own.

### 9. Keys and provenance

**Keys.** The key is read from the variable `api_key_env` names, in each process that sends requests: the
evaluation process and its worker processes, which inherit the environment. A key never appears in any of these:

- a config (the schema has no key field), `config.resolved.yaml` or `lock.json`;
- trajectories, event files or logs;
- worker arguments, which are pickled;
- anything given to a harness. Harnesses only ever hold their agent key, so the provider key never enters a
  sandbox. RFC 0006 states the same rule for Tinker keys.

A reply that echoes the key, as some proxies' errors do, has it replaced with `[redacted]` before it reaches the
harness, a log or a record, as it is, JSON-escaped (with or without `/` escaped) or percent-encoded. A failed request
is logged by its status, error type and request id, never its body. The key is read without the whitespace around
it. `api_key_env` must be a variable's name, and a config error never repeats a value that may hold a key (an
`api_key_env`, or a `base_url` with user info, a query or a fragment).
The HTTP client reads nothing else from the environment, except the standard proxy variables (`HTTPS_PROXY`,
`NO_PROXY`), which a cluster may need. The official SDKs do more: they read `OPENAI_ORG_ID`, `OPENAI_PROJECT_ID`,
`OPENAI_CUSTOM_HEADERS` and `ANTHROPIC_CUSTOM_HEADERS`, and `ANTHROPIC_AUTH_TOKEN` when no key is passed, and send
them to whatever base URL they are given (section 11). Two places are not guarded, and the guide says so:

- Model code that a harness runs in the evaluation's own process, such as `platoon-codeact`'s cells, sees that
  process's environment, the key's variable included.
- So do commands in a `process` sandbox.

**Provenance.**

```json
"policy": {
  "backend": "hosted",
  "model": "gpt-5.1",
  "base_url_origin": "https://api.openai.com",
  "api_key_env": "OPENAI_API_KEY",
  "max_concurrent_requests": null,
  "probe": {"openai_chat": {"served_model": "gpt-5.1-2025-11-13"}, "openai_responses": {"served_model": "gpt-5.1-2025-11-13"}}
},
"judges": {"enabled": false, "child_grading": {"textcraft": {"configured": "verifier", "applied": "none"}}}
```

The lock records only the origin of the base URL: the scheme, host and port, never its path, query or user info.
It records variable names, never values. A Tinker policy gets a `policy` block too. Every run rewrites `lock.json`,
so each rollout's `metadata.json` also records its provenance: the backend, the model, the base URL's origin and
`judges`. If the rollouts already in an output directory disagree with the config, two things happen:

- Resuming into that directory exits 78 and asks for a new one, instead of mixing two models or two judge settings in
  one report.
- `stage: report` refuses to label those rollouts with the config's settings.

Metadata written before this change counts as Tinker with judges on.

### 10. The judges switch

`inference.judges` says whether platoon runs model calls that judge agents rather than solve the task. Its default
is `true`, which is today's behavior. `false` turns off exactly the calls in the table, which also says what
replaces each one:

| Judge | With `judges: false` | Effect on what evaluation reports |
|---|---|---|
| Verifier agents (`agents.child_grading: verifier`) and the behavior judges that run with them | Not started. The child **falls back** to the episode's outcome, as `child_grading: none` and today's `enable_subagent_verifiers: false` do. Its trajectory records `child_grading: {configured: verifier, applied: none, reason: judges_off}` | None on success metrics, which are the root's. Some entries are **refused** (exit 78): an entry that may spawn children and names its own `reward_processor`, or a `program` other than core's. That code may read child verdicts, which platoon cannot check, so the entry must state the fallback itself (`agents.child_grading: none`) or keep judges on |
| Parent-usage judges (`apply_subagent_parent_usage_gates`) | Skipped; each candidate child records `{status: skipped, reason: judges_off}` | None: their verdict only gates training eligibility, never reward |
| Judge agents (`ep.judge_endpoint()`, or a `judge`-role agent minted over HTTP) and `judge`-kind calls, a harness's included | **Refused** with `JudgesOffError` (over HTTP, the dialect's permission error, 403), and counted in the refusing agent's misc (`judges_refused`). Programs read the new `ep.judges` to do without them. The first rollout that meets a refusal stops the evaluation with exit 78, naming the program or harness, instead of erroring every rollout | Whatever a program computed with its judge: it must choose, since platoon cannot |

`policy` and `aux` calls are the agents solving the task, and they keep running. So do environment graders,
including a Harbor task's tests, or an environment's own grader that calls a model, since they are the measurement.
`enable_subagent_verifiers` and `agents.verifier.enabled` stay as finer switches, and their fallback is now recorded
the same way.

**Judges of a hosted policy** sample the policy, as they do under Tinker: while judges are on, verifier agents and
judge calls go to the same base URL and model, and bill the hosted model. Judge code that calls through a
`ModelEndpoint` does so in its own dialect, usually Chat Completions, which the base URL must then speak too (section
1). Judges get no base URL or model of their own (open question 1's answer). `judges: false` is the switch for an
evaluation that should not pay for them, or whose base URL speaks only the harnesses' dialects.

**Recorded** in four places:

- `lock.json` (`judges`, with each entry whose grading changed);
- every rollout's `metadata.json`;
- each affected child trajectory;
- the report: `judges.enabled`, verifier agents and steps, judge calls, refusals, and the children that shared the
  episode outcome because of the switch.

Verifier and judge agents leave the report's step and subtrajectory statistics, which count policy agents only, so
reports with judges on and off compare directly. `platoon rollout` honours `inference.judges` of an evaluation
config. `platoon eval` sets the new `RolloutConfig.judges` field that carries the switch into rollouts, as it sets
`train` and `return_dict`. A rollout config that turns judges off by itself is refused. **Training keeps its
judges.** A training config whose `train` or `eval` rollout config sets `judges: false` exits 78. Nothing would
record the switch there, and a program's judge would error every rollout. Verifiers keep their own switches for
training.

### 11. What this revision drops from the first draft

The first draft was built, and this revision replaces its hosted half. The judges switch stays as built.

- **Provider kinds** (`kind: openai | anthropic`), each with a default URL, a default key variable and an SDK. The
  harness's dialect decides how a request is sent, and one base URL serves every dialect the run uses. Azure OpenAI,
  Amazon Bedrock, Google Vertex AI and others are reached through any proxy that speaks the harness's dialect, such
  as a LiteLLM proxy. This also settles the first draft's open question on more providers.
- **Translation between dialects.** The first draft added the reverse direction for each format (gateway request to
  native body, native reply to gateway response) and decided how one API's features map onto another's: tool-use
  ids, thinking blocks, citations, usage. A harness's request now reaches a provider only in its own dialect, and a
  base URL that lacks the dialect fails the probe. Codex's Responses requests go to a Responses API unchanged, which
  answers the first draft's open question on Codex.
- **Thinking replay.** The Anthropic backend kept each agent's native assistant turns, so that a Chat Completions
  harness under translation could send Anthropic's signed thinking blocks back. With pass-through, the harness
  replays the provider's own blocks itself.
- **The SDKs.** Pass-through needs a plain HTTP client. platoon already depends on `aiohttp`, which its gateway server
  uses, so the hosted path adds no requirement:
  - The `anthropic` extra goes. Nothing under `platoon/` imports `openai` either, so it can leave the core
    requirements.
  - The SDKs re-serialize a body they did not build: the first draft passed most of it as `extra_body`.
  - Their retries had to share one budget with the gateway's own sending again of broken streams.
  - They read credentials and headers from the environment and send them to any base URL (section 9).
- **Provider-specific handling:**
  - OpenRouter's `usage.cost` and a LiteLLM proxy's cost header;
  - the 16,000-token default completion budget for Messages;
  - adding `stream_options.include_usage` to Chat Completions requests;
  - the 4-characters-a-token estimate for `count_tokens` on non-Anthropic backends;
  - per-model advice in the guide, such as `drop_params` for current Claude models.

  Cost now comes from `pricing` alone.
- **Settings:**
  - `require_api_key`: leave out `api_key_env` instead;
  - `headers` and `headers_env`: a proxy adds headers;
  - `extra_body` and `drop_params`: the body is the harness's, so its settings belong to the harness's configuration
    or to the proxy's (LiteLLM has its own `drop_params`);
  - the Tinker server URL that `backend.base_url` gave evaluation configs: `platoon eval` reads `TINKER_BASE_URL`, as
    before.
- **The first draft's tool rules**: tool search kept in pass-through, provider-side tools dropped, `tool_choice`
  narrowed only under translation. Tools now behave as in training (section 4), so a hosted model sees the tools a
  trained policy would.
- **The Tinker gateway's dialect changes for hosted replies**: usage details in Chat Completions answers, a
  provider's wording for context-length errors, and `RenderRequest.tool_choice`. The training dialects answer as on
  `main`.

## The hard parts

- **Keeping tools in step with training.** The tool table is only as good as its parity with the training dialects,
  which other work keeps changing. Both paths call the same functions for the `tool_choice` check, the tool
  normalization of each dialect, and the history checks. A parity test feeds every row's requests through both
  paths, so a change to training's tool handling either changes the hosted path with it or fails a test.
- **Status before the answer.** A streaming harness must hear something within its idle timeout, but sending the
  status commits it. Committing at the first keepalive keeps every early provider error, the common case, with its
  own status and body. Only failures after 10 seconds become an error event, as in training.
- **Context-length errors are not standardized.** OpenAI and Anthropic name theirs, while vLLM, SGLang and routers
  word theirs differently. Recognition only decides whether a refusal counts as a context overflow, since the reply
  goes back unchanged and the harness reacts to its own native error. A miss leaves a policy failure counted as a
  rejected request. Recognition is tested against recorded bodies of each server, and an unrecognized 400 is logged
  with its type so a pattern can be added.
- **Usage that is not reported.** A Chat Completions stream reports usage only when the harness asks for it, and
  adding the request would change the reply's chunks. Such calls are recorded as `usage_reported: false`, left
  unpriced and counted in the report. Harnesses that stream Chat Completions are rare, since `ctx.chat` and the
  common SDK-based harnesses do not stream.
- **One base URL for several dialects.** A run that mixes a Messages harness with function harnesses, or with judges
  that speak Chat Completions, needs a server that speaks both. The probe finds out before any rollout. A LiteLLM
  proxy is the general answer, and `judges: false` removes the judges' dialect.

## Alternatives

- **Provider kinds with translation**, the first draft (section 11). It evaluated any harness on either of two
  providers. The cost was a translation per pair of dialects, a thinking replay store, per-provider defaults and two
  SDKs, and other clouds still needed a proxy.
- **The official SDKs for pass-through.** They give retries and typed errors. The cost is a dependency per dialect,
  bodies passed through as `extra_body`, a retry budget to reconcile, and credentials read from the environment and
  sent to third-party URLs (section 9).
- **Harnesses call the base URL directly**, with the gateway out of the path. There would be no agent records, step
  limits, call kinds or judges switch, and the provider key would sit inside sandboxes.
- **Forward tool features unchanged**, so that a hosted model gets server tools, tool search and forced choices. The
  hosted model would then work in a different environment from the trained policy: it could search the web or run
  code at the provider, outside the environment and its records. Scores would stop being comparable.
- **Relay streams as they arrive.** This keeps the provider's timing. But an attempt that breaks after its first byte
  could not be retried, and the record would have to be built from a partial reply. Training does not stream as it
  samples either.
- **LiteLLM as a library**, as the RAO snapshot used it. It is a large and fast-moving dependency in every
  evaluation. A LiteLLM proxy stays the recommended way to reach other providers.
- **A separate benchmarking command** would mean a second workflow, report and resume logic for the same rollouts.
- **Refuse every verifier-graded entry when judges are off.** This is always explicit, but then the switch does not
  switch anything off, since each entry would need `child_grading: none` anyway.
- **Judges off by default in evaluation.** The comments on `enable_subagent_verifiers` and in
  `platoon.harness.verify` already recommend it. But it would change reported child outcomes and judge-based shaping
  of existing evaluation configs without notice. Judges stay on by default.

## Rollout plan

Each milestone merges on its own. They change `platoon/gateway/core.py`, `server.py`, the dialect modules,
`platoon/eval/` and `platoon/harness/rollout.py`.

| Milestone | Scope | Exit check | Pull requests |
|---|---|---|---|
| M1 judges switch | `inference.judges`, the `RolloutConfig` field, recorded fallbacks, `ep.judges` and `JudgesOffError`, the load-time refusals, `lock.json`, metadata and report fields, judge-free step statistics | CPU tests: with judges off, no verifier agent opens and no judge call is made, children record the grading applied, and refused configs exit 78. With judges on, trajectories equal today's | The pull request that adds this RFC |
| M2 hosted models | `inference.hosted` with its checks, forwarding of the three dialects and of `ctx.chat`, the tool rules shared with the training dialects, retries and deadlines, the request limit, the probe, records, report blocks, provenance and the resume check, and the stale-key messages | CPU tests against a fake provider that speaks all three dialects. (1) Bodies reach it unchanged except for the fields section 3 lists. (2) For every row of section 4, the pass-through sends the tools the training dialect yields and refuses what it refuses, with the same message. (3) Replies come back byte for byte, early errors keep their status, and keepalives come after 10 seconds. (4) Retries share one budget, a failed call errors its rollout, and a refused key stops the run. (5) The fake key is in no file, log or reply. (6) Number search runs end to end with `platoon-react` and `platoon-bash`. Then three smokes: a local vLLM or SGLang server (Chat Completions), a LiteLLM proxy (all three dialects), and 5 tasks on one hosted API for each dialect | The pull request that adds this RFC |
| M3 evaluation without Tinker | tinker and tinker-cookbook imported only by the Tinker gateway and the trainer | `pip install platoon` runs `platoon eval` with `inference.hosted` without the `tinker` extra | |

## Open questions

None remain. The two that review left open were answered on 2026-10-17, and the text above follows the answers:

1. **Judges of a hosted policy.** Judge code samples the same base URL and model as the policy, so it bills the
   hosted model, and while judges are on the base URL must also speak the judge code's dialect. Should judges get an
   endpoint of their own, with a base URL and model of their own? **No.** Judges keep the policy's base URL, and
   `judges: false` is the switch for an evaluation that does without them (section 10).
2. **Training's two schema gaps.** The pass-through gave a schema-less Messages entry the empty-object schema, where
   training handed `parameters: null` to the chat-template renderer, and left out a nameless toolset, which training
   rendered as a nameless function. It sent each Chat Completions tool as a function of its name, description and
   parameters, where the chat-template renderer got the tools as they came. Should the training dialects adopt these
   rules, so that the two paths share one normalization with no exception? **Yes** (section 4). A tool that already
   is a plain function of a name, a description and parameters renders as before. The chat-template renderer's
   prompts change for the others: tools of other types, of another shape or with other fields, tools without a
   schema or with a null description, Messages tools without a description, and Messages toolsets. The cookbook
   renderers already read tools this way; for them, only a Messages toolset (shown without a name) and a Responses
   function without a name (a 400) change, and both are now left out, and a Chat Completions tool whose name is
   empty or not a string is now refused. A nameless Chat Completions tool, which the pass-through left out and the
   cookbook renderers refused, is now refused with that 400 on both paths, by every renderer.
