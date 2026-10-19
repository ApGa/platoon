# Evaluate a hosted model

`platoon eval` samples the policy through Tinker unless its config names a hosted model. With `inference.hosted`, it
evaluates a model behind a **base URL that speaks your harness's API dialect**: OpenAI's API, Anthropic's, a vLLM or
SGLang server, OpenRouter, or a LiteLLM proxy in front of any of them. Your harness keeps calling platoon's model
gateway in its own dialect, exactly as in training. The gateway forwards each call to the base URL in that same
dialect and hands the reply back as it came. So the harness, its subagents, step budgets, grading, trajectories
and the report work as they do with Tinker, and the provider's key never reaches the harness. This guide also covers
`inference.judges`, which turns off the model calls that judge agents rather than solve the task.

**Goal:** a baseline on the tasks and harness you train on: a frontier API model, or an open model on your own
server, evaluated the way your policy is.

**Prerequisites:**

- platoon installed as for evaluation ([Install](../getting-started/install.md)). The hosted path needs nothing
  more.
- A server that speaks the dialect of each harness you evaluate (step 1), and its key in an environment variable,
  unless the server needs none.
- An evaluation config ([Train and evaluate](train-and-eval.md#3-evaluate)).

Hosted models serve evaluation and `platoon rollout` only. Their calls are recorded without token ids or logprobs,
so nothing can train on them. A training config has no `inference` section, and `platoon train` exits 78 for
`train.hosted`.

## Steps

### 1. Pick a base URL that speaks your harness's dialect

A harness speaks one API dialect: a harness spec names it (`dialect`), and a function harness's `ctx.chat` is a
Chat Completions call. The base URL must speak it:

| Your harness | Its dialect | A `base_url` that speaks it |
|---|---|---|
| `platoon-react`, `platoon-codeact`, your function harnesses (`ctx.chat`); `platoon-bash`, `mini-swe-agent` and other specs with `dialect = "openai_chat"` | OpenAI Chat Completions | OpenAI (`https://api.openai.com/v1`), a vLLM or SGLang server (`http://localhost:8000/v1`), OpenRouter (`https://openrouter.ai/api/v1`), a LiteLLM proxy |
| Claude Code, and specs with `dialect = "anthropic_messages"` | Anthropic Messages | Anthropic (`https://api.anthropic.com/v1`), a LiteLLM proxy |
| Codex, and specs with `dialect = "openai_responses"` | OpenAI Responses | OpenAI (`https://api.openai.com/v1`), a LiteLLM proxy |

A base URL ends where the gateway's `/v1` ends: the gateway calls `{base_url}/chat/completions`,
`{base_url}/responses` or `{base_url}/messages`. One base URL serves the whole run, so it must speak the dialect of
every harness the run's environments use, verifier harnesses included. A LiteLLM proxy speaks all three, and
reaches Azure OpenAI, Amazon Bedrock, Google Vertex AI and many more. Platoon knows no providers of its own: any
server that speaks the dialect will do.

### 2. Name the hosted model

`inference.hosted` holds the base URL and the name of the variable that holds its key, and `inference.model_name` is
the model the base URL serves:

```yaml
# eval-hosted.yaml
inference:
  model_name: gpt-5.1                     # the model the base URL serves; every request names it
  hosted:
    base_url: https://api.openai.com/v1   # speaks the harness's dialect
    api_key_env: OPENAI_API_KEY           # the variable that holds the key, never the key itself
  workflow:
    num_rollouts_per_task: 4
    num_concurrent_workers: 16
    rollout_config:
      max_steps: 12
      timeout: 600
environments:
  - service: python://platoon.envs.number_search.env:NumberSearchEnvironment
    harness: platoon-react
    agents: {max_depth: 0}
    eval_dataset_kwargs: {split_name: val, limit: 100}
split: eval
```

The same section for other servers:

```yaml
# hosted.yaml
anthropic:                         # Claude Code, or any harness that speaks Messages
  model_name: claude-opus-5-5
  hosted:
    base_url: https://api.anthropic.com/v1
    api_key_env: ANTHROPIC_API_KEY
vllm:                              # an open model on your own vLLM or SGLang server
  model_name: Qwen/Qwen3-4B-Instruct-2507      # the server's served model name
  hosted:
    base_url: http://localhost:8000/v1         # no api_key_env: the server takes no key
openrouter:
  model_name: openai/gpt-5.1
  hosted:
    base_url: https://openrouter.ai/api/v1
    api_key_env: OPENROUTER_API_KEY
litellm_proxy:                     # all three dialects, for every provider the proxy reaches
  model_name: my-deployment
  hosted:
    base_url: http://localhost:4000/v1
    api_key_env: LITELLM_API_KEY
```

A hosted model renders its own prompts, so the Tinker rendering settings do not apply: `renderer_name`,
`model_path`, `renderer_kwargs`, `sample_timeout_seconds` and `context_window_length` are refused with it (exit 78).
`hosted.timeout_s` bounds each of its HTTP attempts instead.

The gateway sends the harness's own request, so model settings belong where the request is made. A setting the
harness does not send, such as a reasoning effort, thinking or a chat template's switch, goes in the harness's
options: `platoon-react` and `platoon-codeact` send their `harness_options.sampling` with every call
(`sampling: {reasoning_effort: high}`). A server-side default, or a parameter a model refuses, goes in the proxy's
model configuration (LiteLLM has `drop_params`).

### 3. Run it

```bash
export OPENAI_API_KEY=...         # never in a config file
uv run platoon eval eval-hosted.yaml
```

Before it prepares or writes anything, `platoon eval` checks that the key's variable is set (a hosted model needs no
`TINKER_API_KEY`); then, before any task is listed, it **probes** the base URL: one tiny request in each dialect the
run's harnesses speak, for `inference.model_name`. A wrong key, a model the server does not serve, or a base URL
without a dialect's route is one line and exit 78, not a run of errored rollouts.
`stage: report` checks and probes nothing.

The run writes to `runs/eval/gpt-5.1/<UTC time>/`, a new directory for each start, which it names on stderr; set
`inference.output_dir` to that directory to resume the run or report it again.

`platoon rollout eval-hosted.yaml --task val:0` runs one task the same way, after the same probe. `--model` replaces
the model; `--renderer` and `--model-path` configure a Tinker policy and are refused with a hosted model.

### 4. Read the report

`reports/final_report.json` has what every evaluation report has, and these blocks, which a Tinker evaluation fills
too where it can:

| Block | What it holds |
|---|---|
| `model` | The model, the backend (`hosted` or `tinker`), the base URL's origin, and `served_models`: each model the server answered with (a dated snapshot, or a router's choice) and its calls |
| `usage` | Calls and tokens of `policy` and `aux` calls, of `judge` calls, and of verifier agents' calls, each as `total` and `mean_per_rollout`: prompt, completion, cached-prompt, cache-write and reasoning tokens, as the server reported them |
| `cost` | `total_usd` and `mean_usd_per_rollout` from `inference.hosted.pricing`, and `unpriced_calls`: calls without a cost (no prices, or no usage reported) |
| `backend` | HTTP attempts, rate-limited replies, calls that failed after their retries, refusals, `calls_without_usage`, and the rollouts with a failed call |
| `judges` | Whether judges ran, verifier agents and their steps, judge calls, refused judges, and the children that shared the episode's outcome, by reason ([Turn judges off](#turn-judges-off)) |

Costs need prices: `hosted.pricing: {input: 1.25, output: 10.0, cached_input: 0.125}` (USD per million tokens;
`cached_input` and `cache_write` default to the input price).

Each rollout's `metadata.json` carries its own `usage`, `cost`, `backend` and `judges`, and the run's `provenance`
(backend, model, base URL origin, judges). Each trajectory step's `action_misc` carries its call's record: the
dialect, the served model, the response and request ids, the raw stop reason, attempts, rate limits and cost.

## What reaches the server

| The harness calls the gateway at | The gateway calls | The key goes as | Client headers forwarded |
|---|---|---|---|
| `POST /v1/chat/completions` | `POST {base_url}/chat/completions` | `Authorization: Bearer` | `openai-beta` |
| `POST /v1/responses` | `POST {base_url}/responses` | `Authorization: Bearer` | `openai-beta` |
| `POST /v1/messages` | `POST {base_url}/messages` | `x-api-key` | `anthropic-version`, `anthropic-beta` |
| `POST /v1/messages/count_tokens` | `POST {base_url}/messages/count_tokens` | `x-api-key` | `anthropic-version`, `anthropic-beta` |
| `GET /v1/models` | nothing: the gateway lists `inference.model_name` | | |

The path form `/s/<agent key>/v1/...` is forwarded the same way, and so is a query string (`?beta=true`). No other
client header is forwarded: not the agent's key, not `X-Platoon-Call-Kind`, not `OpenAI-Organization`. Without
`api_key_env`, no key is sent. A route the gateway does not serve gets 404, as in training. The gateway follows no
redirect, so the key and the request go to the base URL and nowhere else: a base URL that answers with one (3xx)
refuses the run, as a wrong key does.

**The body** reaches the server as the harness sent it, except for:

- `model`, which becomes `inference.model_name`;
- the tools, which are normalized as training normalizes them (below);
- the sampling parameters the request leaves unset and the rollout's `inference_params` set, which are added under
  the dialect's names (`max_completion_tokens`, `max_tokens` or `max_output_tokens`, `temperature`, `top_p`); nothing
  else is added, and the server's own defaults stand;
- with `rollout_config.sampling_precedence: config`, a policy call's own `temperature`, `top_p` and `top_k`, which
  are removed, and its completion budget when `inference_params` sets one, so that the rollout's values above take
  their place, as with Tinker; `aux` and `judge` calls keep their own, except the `aux` calls of a harness in a task
  sandbox, which are treated as its policy calls;
- the fields that keep or edit a conversation at the provider, which are removed (Responses `previous_response_id`,
  `conversation`, `prompt`, `background` and `truncation`; Messages `context_management` and `compaction`);
- a Chat Completions `n` other than 1, which is refused, as in training.

Messages, system prompts, images and documents, thinking blocks with their signatures, encrypted reasoning, stop
sequences, response formats, `metadata`, `stream` and `stream_options` pass unchanged. A function harness's
`ctx.chat` becomes the Chat Completions body an OpenAI client would send (its keyword parameters, `extra_body`
merged in), unstreamed.

**The reply** comes back as the server sent it: its status, content type and body, with `retry-after` when the server
sent one. Only a key the server echoes is changed, to `[redacted]`. The gateway reads a streamed reply whole before it
relays it byte for byte, so it can record the call and send a broken stream again. Meanwhile it sends `: keepalive`
comments every 10 seconds, as it does while Tinker samples. A reply that comes before the first keepalive keeps its
own status, the usual case for errors. A failure after that reaches the harness as its dialect's error event.

`count_tokens` (Claude Code's) is forwarded with the same changes as a Messages request, and its reply is returned
unchanged. It is not recorded and takes no step.

Some calls never reach the server, as with Tinker: a policy call past the agent's gateway step budget (a harness spec's
`[budget] enforce = "gateway"`, counting the calls in flight) gets the gateway's budget stop, a final message without
tool calls, and a harness in a task sandbox gets the dialect's permission error (403) for its `judge` calls, and
under a gateway step budget for its `aux` calls too.

### Tools behave as in training

A hosted model gets exactly the tools a trained policy is shown, built by the same code, so its scores compare with
your policy's:

- **The request's own tools, as a name, a description and a parameter schema.** Every other field of a tool goes
  (`strict`, `cache_control`, `defer_loading`, `input_examples`, ...), and a tool without a schema gets an
  empty-object one, for a trained policy as for a hosted model. `cache_control` on system prompts and messages is
  not a tool field and stays.
- **Nothing runs at the provider.** A Messages server tool (web search, web fetch, code execution, bash, the text
  editor, computer use, memory, tool search) reaches the model as a plain tool of its name, so a call of it goes to
  the harness like any other. So does a Chat Completions tool of any type but `function`, such as an Anthropic
  server tool a proxy would run, with the name, description and schema it gives. Responses built-in tools
  (`web_search`, `file_search`, `code_interpreter`, `mcp`, `custom`, ...), Chat `web_search_options`, Messages
  `mcp_servers` and MCP toolsets, and Messages `container` are left out. Deferred tools are offered up front.
- **A Chat Completions tool without a name is refused** with training's 400 error (`Invalid tool definition: ...`),
  since no model can call it: a built-in such as `{"type": "browser_search"}`, or a `custom` tool, whose name is
  under `custom`. A Messages MCP toolset and a Responses function without a name are left out instead.
- **`tool_choice` may be `auto` or `none`.** Anything else (`required`, `any`, a named tool) is refused with
  training's 400 error, since training can force no call. `auto` and `none` are removed, so the server's default
  applies, and nothing limits parallel calls (`parallel_tool_calls`, `disable_parallel_tool_use` and
  `max_tool_calls` are removed). Legacy Chat `functions` and `function_call` are removed too.
- **History of tools training never offers is refused,** as in training: a Messages server-tool or MCP block, a
  container upload, or a Responses item such as `web_search_call`, `custom_tool_call`, `mcp_call` or
  `item_reference`. Tool calls and tool results of the harness's own tools pass unchanged, images and `is_error`
  included.

Environment tools reach agents through platoon's MCP tool plane, as in training.

## Settings

All of `inference.hosted` ([reference](../reference/config.md#hostedconfig)):

| Setting | Default | Meaning |
|---|---|---|
| `base_url` | required | The server's base URL, where the gateway's `/v1` ends; it must speak every dialect the run uses |
| `api_key_env` | none | The variable that holds the key; none for a server without auth, which gets no key |
| `timeout_s` | 600 | Seconds one HTTP attempt may take, from sending the request to the reply's last byte |
| `max_retries` | 6 | Retries of 408, 409, 429 and 5xx replies, connection errors, timeouts, and streams cut off or failed midway: a call makes `max_retries + 1` attempts at most |
| `max_concurrent_requests` | none | Requests in flight at once for the whole run; worker processes (`use_subprocesses`) split it, so it must be at least their number (exit 78 otherwise) |
| `pricing` | none | USD per million tokens: `input`, `output`, `cached_input`, `cache_write` |

## Keys

A key is read from its variable by each process that sends requests, the evaluation's worker processes included.
It is never written to a config, `config.resolved.yaml`, `lock.json`, a trajectory, an event file or a log, and it
never reaches a harness: harnesses hold only their agent key, which the gateway swaps for the provider's. A reply
that echoes the key, as a proxy's error may, has it replaced by `[redacted]` before it reaches the harness, a record
or a log, and a failed request is logged by its status, error type and request id, never its body. A config cannot
hold a key: there is no setting for one, `api_key_env` must be a variable's name, and a `base_url` with credentials,
a query or a fragment in it is refused, with a message that does not repeat it. The whitespace around a key (the
newline a variable read from a file may end in) is not sent. `lock.json` records the policy by names only:

```json
"policy": {
  "backend": "hosted",
  "model": "gpt-5.1",
  "base_url_origin": "https://api.openai.com",
  "api_key_env": "OPENAI_API_KEY",
  "max_concurrent_requests": null,
  "probe": {"openai_chat": {"served_model": "gpt-5.1-2025-11-13"}}
}
```

The gateway's HTTP client reads nothing else from the environment but the proxy variables (`HTTPS_PROXY`,
`HTTP_PROXY`, `NO_PROXY`), which a cluster may need. It sends no organization or project header, and no credential
from `.netrc`.

The gateway does not stand between the key and **model code on the rollout host**. Code that a harness runs for the
model in the evaluation's own process (`platoon-codeact`'s cells, and any harness registered with
`executes_model_code`) can read that process's environment, the key's variable included, and so can commands in a
`process` sandbox, which has no isolation. The container sandboxes (`docker`, `singularity`, `enroot`) pass only the
variables their spec names. Evaluate such harnesses with a key you can revoke.

## Failures, retries and rate limits

- **Retries.** A 408, 409, 429 or 5xx reply (529 included), a connection error, an attempt that took more than
  `timeout_s`, a stream cut off before its last event, and a stream with an error event are tried again, with
  backoff from 0.5 to 8 seconds, or as long as `retry-after` (or `retry-after-ms`) asks, up to a minute. Every kind
  of failure counts against one budget, so a call makes at most `max_retries + 1` attempts and ends within about
  `(max_retries + 1) × timeout_s` plus the waits.
- **A call that still fails** reaches the harness as the server's last reply, or as its dialect's timeout or server
  error when there was none. The rollout is errored (`backend_unavailable: <n> model call(s) failed after retries`),
  whatever the harness did next: an outage is never scored as the model's failure. Such a call, and a refused one,
  also counts as a failure of the gateway, so `on_harness_exit`'s `grade_if_gateway_clean` does not grade a harness
  that crashed after it. Resuming the evaluation runs it again (`inference.retry_errored`).
- **A refused key, payment, permission or model** (401, 402, 403, 404), or a redirect (3xx), which the gateway does
  not follow, would fail every call: the evaluation starts no further rollout, lets those in flight finish, and exits
  78 (`backend_refused: ...`).
- **Other refusals** (400, 413, 422) are the harness's request: they reach it unchanged and count as rejected, not as
  outages. A context-length refusal (OpenAI's `context_length_exceeded`, Anthropic's `prompt is too long`, vLLM's,
  SGLang's, OpenRouter's, a LiteLLM proxy's) counts in the agent's context overflows, so a harness that dies of it is
  graded, or excluded with `on_context_overflow: exclude`, as with Tinker
  ([context_overflow](troubleshooting.md#context_overflow)).
- **Request limits.** `max_concurrent_requests` bounds the calls in flight; rollout concurrency
  (`num_concurrent_workers`) is unchanged. With `use_subprocesses`, each worker process gets an equal share, rounded
  down, so a limit below the number of worker processes that run rollouts at once is refused (exit 78).
- **Refusals by the model** (`stop_reason: refusal`, `finish_reason: content_filter`) are recorded as such and scored
  like any answer; the report counts them.

Resuming into an output directory whose rollouts were made with another backend, model, base URL or judges setting
exits 78, before anything in the directory changes: one report would mix them. Leave `inference.output_dir` unset
for a new directory, or set a new one. A report of such rollouts (`stage: report`) exits 78 too, rather than label
them with the settings it was given.

## Turn judges off

Judges are the model calls that judge agents rather than solve the task: verifier agents
(`agents.child_grading: verifier`) and the behavior judges beside them, parent-usage judges, judge agents
(`EpisodeContext.judge_endpoint`) and `judge` calls. They add latency and cost, and an evaluation measures the
root, whose success never depends on them. With a hosted model they also sample the hosted model, so its base URL
must speak their dialect too. `inference.judges: false` turns all of them off:

| Judge | With `judges: false` |
|---|---|
| Verifier agents and behavior judges | Not started. A child its environment grades by verifier shares the episode's outcome, as with `child_grading: none`, and its trajectory says so: `child_grading: {configured: verifier, applied: none, reason: judges_off}` |
| Parent-usage judges | Skipped: each candidate child records `{status: skipped, reason: judges_off}` and keeps its data |
| Judge agents and `judge` calls | Refused (`platoon.gateway.JudgesOffError`; over HTTP, a 403 permission error). The first rollout that meets a refusal stops the evaluation with exit 78, naming the program or harness. A program reads `EpisodeContext.judges` to do without them |

`policy` and `aux` calls and the environment's own grader (a Harbor task's tests, an environment's grading code)
run as before. An entry that grades children by verifier and has its own `reward_processor`, or a `program` other
than core's, is refused while judges are off: that code may read the child verdicts no verifier gives. Set
`agents.child_grading: none` on such an entry to accept the fallback for this evaluation. An environment whose
rollout is its own function rather than `harness_rollout` gets the switch as `RolloutConfig.judges`, and must honour
it itself: platoon sees no judge that such a rollout runs.

`lock.json` records the setting (`"judges": {"enabled": false, "child_grading": {...}}`), with each entry whose
children's grading falls back. The report's `judges` block counts verifier agents and judge calls, and the step and
subtrajectory statistics count the policy's agents only, so reports with judges on and off compare directly.

`enable_subagent_verifiers: false` (rollout config) and `agents.verifier.enabled: false` still turn off verifiers
alone; their children record `verifiers_off` and `verifier_disabled`. They are also the switches for training: a
training config that sets `rollout_config.judges: false` exits 78, since training runs its judges.

## Limits

- Hosted models serve evaluation and `platoon rollout`, not training.
- No logprobs or token ids are recorded: the provider's tokenizer is not the trainer's.
- Nothing is translated between dialects: a harness reaches a server only in its own dialect, and the probe refuses
  a base URL that lacks one the run needs.
- A Chat Completions stream reports usage only when the harness asks for it (`stream_options.include_usage`). The
  gateway adds nothing to a harness's request, so such calls record `usage_reported: false`, count no tokens and
  stay unpriced.
- A function harness sends back what `ctx.chat` returned, `reasoning_content` included. A server that refuses that
  field in requests needs a proxy that drops it.
- The `tinker` extra is still needed to run `platoon eval`, whichever model serves the policy.
