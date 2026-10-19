# RFC 0012: Tool-call parsing for any chat template

| | |
|---|---|
| Status | Draft |
| Created | 2026-10-16 |
| Authors | Apurva Gandhi (@ApGa) |
| Builds on | The `chat_template` renderer (`platoon/train/tinker/renderers/chat_template.py`, [Architecture](../architecture.md#renderers)), the model gateway (`platoon/gateway/core.py`), [RFC 0005](0005-gateway-dialects-and-tito.md) |
| Discussion | The pull request that adds this RFC |
| Implementation | Milestones M1 to M6 are implemented alongside this RFC; none merged yet |

## Summary

The `chat_template` renderer renders prompts with the model's own chat template but parsed every response with one
Qwen3-Coder XML parser, so the tool calls of other model families reached harnesses as plain text, and some Qwen
calls were dropped without a trace. This RFC replaces it with a registry of tool-call parsers, one small module
per format, that read complete responses into reasoning, content, tool calls and the call blocks they could not
parse. `renderer_kwargs.tool_call_parser` picks one: `auto` (the default) renders a probe turn through the model's
own template and keeps the parser that reads it back exactly, a name picks a built-in parser (SGLang's names,
vLLM's as aliases), and `package.module:attr` a parser of your own. The choice is made when the trainer, a rollout
worker or an evaluation policy starts, refused when no parser fits a template that renders tools, and recorded in
`lock.json`. Calls that do not parse stay in the content, as inference engines return them, and are counted per
model call into trajectories and rollout metrics.

## Motivation

What the parser decides never changes the training tokens: the gateway trains on exactly the tokens the model
sampled. It decides two other things. A missed call is an action the harness never takes, which usually ends the
episode with a reward of 0 for a reason nobody sees. And a call parsed wrongly is re-rendered differently in the
next prompt, which then diverges from the sampled tokens, so prefix merging cuts more datums.

Before this RFC, `parse_response` called `qwen3_coder_parser.parse_text`, a port of AReaL's Qwen3-Coder XML parser
with SGLang's `qwen3_coder` detector as its fallback:

- **Only one format.** Hermes JSON (Qwen2.5, Qwen3, Qwen3-2507), Llama JSON, DeepSeek, Kimi K2, GLM, Mistral and
  gpt-oss calls came back as text. With the default `enable_thinking: true` the parser also assumed every prompt
  ended in `<think>`, so a Qwen3-2507 reply became reasoning in full.
- **Silent drops on the format it supported**, on Qwen3.5 and later:
  - a call with no arguments next to calls with arguments was dropped, its XML left in the content;
  - an unclosed final `<tool_call>` left that call in the content when earlier calls parsed;
  - a value holding a literal `</parameter>` (code that edits a parser, say) was cut there;
  - calls after a `<think>` that never closed were read as reasoning.
- **No record of failures.** The parser returned `unparsed_tool_calls`, but the gateway never read them, so the
  text of a malformed call disappeared from what the harness saw, and no count reached a trajectory or a metric.

Engines choose a parser by an explicit name (SGLang's and vLLM's `--tool-call-parser`), and SGLang also sniffs the
template's text; neither checks the choice. transformers' `response_schema` makes the parser data, but its API is
experimental and no model in common use ships a schema.

## Proposal

### 1. One parser module per format

`platoon/train/tinker/renderers/tool_parsers/` holds one module per format. Each reads a complete response (no
streaming code: the gateway synthesizes streams after the parse) and never raises on what a model wrote:

```python
@dataclass(frozen=True)
class ParsedToolCall:
    name: str
    arguments: dict[str, Any]
    id: str | None = None  # the id the model sampled, for formats that carry one
    recovered: bool = False  # parsed only by a lenient rule


@dataclass(frozen=True)
class UnparsedToolCall:
    raw_text: str  # exactly as sampled
    error: str


@dataclass
class ParseResult:
    content: str
    reasoning: str
    tool_calls: list[ParsedToolCall]
    unparsed: list[UnparsedToolCall]


class SupportsParse(Protocol):
    def parse(
        self, text: str, *, tools: Sequence[Mapping[str, Any]] | None = None, in_reasoning: bool = False
    ) -> ParseResult: ...
```

`text` is the sampled text before the stop token, `tools` the request's tool definitions, and `in_reasoning` says
that the generation prompt ended in the format's open reasoning tag. Built-in parsers subclass `ToolCallParser`,
which splits the reasoning (between the format's tags; a reasoning block that never closes ends where a real call
starts, as SGLang's reasoning parser has it) and leaves `parse_calls` to the format.

| Name | Aliases | Format |
|---|---|---|
| `qwen3_coder` | `qwen3_xml`, `step3p5`, `nanbeige` | `<tool_call><function=NAME><parameter=KEY>VALUE</parameter>...` (Qwen3-Coder, Qwen3.5 and later) |
| `hermes` | `qwen25`, `qwen` | JSON inside `<tool_call>` (Qwen2.5, Qwen3, Qwen3-2507, Hermes) |
| `glm45` | `glm`, `glm47` | `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>` (GLM-4.5 and later) |
| `deepseekv31` | `deepseek_v31` | `<｜tool▁call▁begin｜>NAME<｜tool▁sep｜>{...}` (DeepSeek-V3.1) |
| `deepseekv3` | `deepseek_v3` | `function<｜tool▁sep｜>NAME` and a fenced JSON block (DeepSeek-V3, R1) |
| `kimi_k2` | | `<\|tool_call_begin\|>functions.NAME:N<\|tool_call_argument_begin\|>{...}` (Kimi K2, K2.5) |
| `mistral` | | `[TOOL_CALLS][{...}]` (tokenizer v3) and `[TOOL_CALLS]NAME[ARGS]{...}` (v11 and later) |
| `gpt-oss` | `openai`, `gpt_oss` | Harmony messages; the `analysis` channel is reasoning |
| `llama3` | `llama3_json`, `llama4_json` | `{"name": ..., "parameters": ...}`, optionally after `<\|python_tag\|>` |
| `pythonic` | `llama4_pythonic` | `[name(key=value), ...]` at the start of the response, read with Python's parser (Llama 4) |

Names are SGLang's `--tool-call-parser` names, and vLLM's spellings are aliases, so a model's serving docs name
its parser here too. `none` parses no tool calls (reasoning only). `package.module:attr` loads a parser of your
own: a `ToolCallParser` subclass, an object with `parse`, a class to build without arguments, or a function with
`parse`'s signature.

Shared code types argument values that XML-style formats write as text (`qwen3_coder`, `glm45`) by the request's
tool schemas: `3` is the integer only when the parameter's schema allows an integer, `True` and `None` (Python's
`str`, which Qwen3.5's template writes) read as the boolean and null, and a parameter the schema does not describe
stays text. Arguments are always JSON values: a text value that JSON cannot hold as the schema's type (a number out
of a float's range, an integer with more digits than Python converts, a literal with a set in it) stays text, and
JSON arguments with `NaN` or `Infinity` in them, or nested too deeply to decode, make the call unparsed. Lenient
rules read malformed output whose meaning the structure shows (a missing `</parameter>`, `</function>` or
`</tool_call>`; raw newlines inside a JSON string; arguments double-encoded as a JSON string) and mark the call
`recovered`. Everything else that opens a call is an `UnparsedToolCall` with its raw text, on its own: one broken
call never takes the response's other calls with it.

Formats and rules that follow SGLang are named in the modules' docstrings and in `NOTICE` in the package, with
SGLang's Apache-2.0 license next to it; the code is written for complete responses.

### 2. Configuration

Two keys of `renderer_kwargs` (`train.renderer_kwargs`, and the evaluation config's `inference.renderer_kwargs`;
`platoon rollout` reads whichever section its config has) set the parser of a `chat_template` renderer:

```yaml
train:
  renderer_name: chat_template          # or one of its presets, such as qwen3_8_preserve_thinking
  renderer_kwargs:
    tool_call_parser: auto              # auto | none | a name or alias | package.module:attr
    reasoning_tags: auto                # auto | none | [open, close], such as ["<seed:think>", "</seed:think>"]
```

`reasoning_tags` overrides the format's reasoning tags for a model that reasons between other tags; it applies to
platoon's parsers and to user parsers that subclass `ToolCallParser`. tinker-cookbook renderers parse their own
tool calls and have neither key (setting one is the usual unknown-attribute error).

### 3. `auto`: a round trip through the model's own template

At the first use of a renderer (which `build_tinker_llm` forces at start), `auto` renders a probe conversation
through the template with the renderer's own settings: a system and a user message, then an assistant turn with
reasoning, a short text and three calls: an integer, an enum and a string with a comma; a multi-line string with
leading indentation, strings that read as a number (`"123"`) and a boolean (`"true"`), a list and a boolean; and a
call with no arguments. The text a model would sample for that turn is the render after the generation prompt,
cut at the first stop token. The first registered parser that reads it back exactly wins: the same calls in order
with the same arguments by JSON value, the text and the reasoning as far as the template rendered them, and
nothing unparsed or recovered.

- A template that cannot put all three calls in one sampled turn (it raises on more than one, drops calls, or
  ends the turn at the first call's stop token, as Harmony does) is checked with one probe turn per call.
- Call ids come in two shapes, since templates demand them: Kimi K2's (`functions.NAME:N`, which carries the
  name), then Mistral's nine letters and digits when the template raises on the first.
- A template that renders tools when no parser reads the probe back stops the run with an error that shows the
  rendered turn, what each parser made of it, and what to set. A template that renders tools but not the
  assistant's calls (or raises on them) cannot be checked and stops it too. A template that renders neither
  parses no calls, with a warning.
- A parser set by name is checked the same way; a mismatch only warns, since some templates cannot be probed
  faithfully.

The choice is cached per process by tokenizer, template hash, stop tokens, template settings and parser settings,
so every renderer a trainer, worker or policy builds for the same model reuses it. A choice takes 1 to 30 ms on
the eight cached Qwen snapshots the tests use. On each of them exactly one parser reads the probe back: `hermes`
for Qwen3-4B-Instruct-2507 and `qwen3_coder` for the Qwen3.5, 3.6 and 3.8 snapshots.

### 4. Reasoning from the prompt

Whether a response starts inside reasoning is read from the prompt it was sampled for, which the gateway now
passes to `parse_response`: when the prompt's last tokens end in the parser's open reasoning tag, the response
starts in reasoning. This is right for both `enable_thinking` values, per request, and for templates that ignore
the flag (Qwen3-2507). Without a prompt (a direct call), the renderer renders a generation prompt under the
effective `enable_thinking` and reads that. A response that starts with a close tag has empty reasoning.

### 5. Failures are loud

- **In the response.** The gateway appends the raw text of each unparsed call to `content`, as vLLM and SGLang
  return text they cannot parse, so a harness's own format feedback can answer it. The first few are logged as
  warnings, then every thousandth.
- **Per model call.** `ChatResponse.generation`, which each trajectory keeps under `misc["model_generations"]`, adds
  `recovered_tool_calls` and `unparsed_tool_calls` next to `parsed_tool_calls` and `emitted_tool_calls`.
- **When the parser fails.** A parse never fails the model call: when the parser raises, or returns arguments that
  are not JSON (a user parser's bug, say), the whole response is one unparsed call, and the error is logged with its
  traceback the first few times in a process.
- **In metrics.** Each trajectory's stats sum them, and a group's rollout statistics report them per trajectory as
  `rollout_completion/num_tool_calls`, `num_recovered_tool_calls` and `num_unparsed_tool_calls`.

Penalizing unparsed calls or masking their turns out of the loss would change the RL objective, and stays an open
question.

### 6. Chosen at start, recorded in `lock.json`

`build_tinker_llm` chooses the renderer's parser as it builds the renderer, so the trainer, every rollout worker
and every evaluation policy fails at start, not at the first tool call. `platoon train`, `eval` and `rollout` choose
it before they run, exit 78 (a configuration error) when it cannot be chosen, and record it in `lock.json`:

```json
"tool_call_parsing": {
  "model_name": "Qwen/Qwen3-4B-Instruct-2507",
  "renderer_name": "chat_template",
  "setting": "auto",
  "parser": "hermes",
  "verified": true,
  "reason": "round trip (one turn with every call)"
}
```

A tinker-cookbook renderer is recorded as parsing its own calls, without loading the model.

### 7. What changes for users

Prompts do not change, and neither do the training tokens: the 864 saved renderer token fixtures and the CodeAct
golden rollouts pass unchanged, except for the two new counts in each call's generation record. What a response
parses into changes, relative to the parser before this RFC:

| Response | Before | Now |
|---|---|---|
| A call with no arguments next to calls with arguments (`qwen3_coder`) | Dropped; its XML stayed in the content | Executed |
| A final call whose `</tool_call>` is missing, after other calls | The final call stayed in the content | All execute; the final call counts as recovered |
| A final call cut off before its `</function>`, after other calls | The final call stayed in the content | The others execute; the final call is unparsed, its text in the content |
| A value holding a literal `</parameter>` | Cut at the literal | Kept whole |
| A value holding calls written out (OpenHands' `<function=` lines, Hermes blocks) | The call ran with broken arguments | Kept whole |
| A value holding a line `<parameter=KEY>`, for a parameter the call has not given yet | Split there | Split there and counted as recovered, or kept whole when `KEY` opens again later in the call |
| Calls after a `<think>` that never closed | Read as reasoning, not executed | Reasoning ends where the first call starts; the calls execute |
| A missing `</parameter>`, `</function>` or final `</tool_call>` | Parsed by the fallback, or left in the content | Parsed, counted as recovered |
| A call that does not parse | Its text disappeared from the content | Its text stays in the content, counted as unparsed |
| A response with two `</think>` tags | Content after the last one | Reasoning before the first; the rest is content |
| A `</think>` in a response whose prompt did not end in `<think>` | Split as reasoning | Content |
| A Hermes-format model under `chat_template` (Qwen3-2507) | The whole reply as reasoning, no calls | Content and calls |

For configs that use the `qwen3_6_preserve_thinking` and `qwen3_8_preserve_thinking` renderers, `auto` chooses
`qwen3_coder`, and the first seven rows are the visible change: more calls execute, and with the arguments the model
wrote, so episodes that used to end on a silently dropped or broken call go on, and rewards are not comparable call
for call with runs that used the old parser.
The module `platoon.train.tinker.renderers.qwen3_coder_parser` is removed; nothing outside the renderer imported
it.

## The hard parts

- **XML values are not escaped.** In `qwen3_coder` a value may hold any text, its own close tag included, so a
  close tag ends a value only when the structure after it confirms it: the call's next parameter, or the end of
  the function followed by nothing that belongs inside a call. Values may hold calls written out, in the format
  itself, as OpenHands' prompts write them (`<function=` lines without `<tool_call>`) or as Hermes blocks, and parse
  exactly: a call written out holds the close tags after it until it closes. A missing `</parameter>` is read from
  a parameter of the tool that the call does not have yet, opening on a new line and not again later in the call.
  A well-formed value that holds such a line is the same text, so it is split there too and the call counted as
  recovered, as the parser before this RFC split it. Other ambiguous text still misleads the parser: a value
  holding `</parameter>` followed by a `<parameter=` line of another key is split there (counted as recovered when
  that key is one the call already has, since a call never gives a parameter twice), and a value holding a
  `<tool_call>` written out without its `</tool_call>` cannot end, so its call is unparsed and the call written out
  inside it may run. The round trip's probe and the tests' literal values cover the cases models write.
- **Degenerate output.** A collapsing policy may repeat markup up to the token limit, and the gateway parses inside
  its sample call. `qwen3_coder` keeps what it learns about positions in a response instead of searching the rest of
  the response again for each tag, so a 200 kB response of repeated tags parses in tens of milliseconds.
- **Truncation and missing close tags look alike.** A call cut off at the end of the response is unparsed unless
  something shows that it ended (`</function>` for `qwen3_coder`, an argument for `glm45`). The gateway emits no
  calls from a response cut by the token limit anyway.
- **Templates that cannot be probed.** The probe can only check what the template renders. The `chat_template`
  renderer passes tool-call arguments to the template as mappings, which Qwen's templates need; a template that
  concatenates them as strings (DeepSeek-V3's own does, UNVERIFIED) cannot render a tool call from the renderer
  at all, and the probe says so instead of choosing. Apart from the cached Qwen snapshots, the formats are checked against
  templates written for the tests in each vendor's shape, not the vendors' own templates (UNVERIFIED).
- **Ids that templates write back.** Kimi K2's template writes each call's id into the next prompt, and the id
  holds the function name; Mistral's templates refuse ids that are not nine letters and digits. Those parsers
  keep the sampled id, and the Mistral parser mints one of the right shape when the model sampled none.

## Alternatives

- **transformers `response_schema`, with a platoon post-pass.** Parsing as data run by
  `tokenizer.parse_response` is attractive, and a prototype of it read every cached template. But the post-pass
  it needs (typing values from the tools, zero-argument calls, unclosed blocks, isolating a bad call, a lenient
  retry) is most of a parser, the schema language cannot express structure-confirmed value ends, the API is
  marked experimental, and its semantics changed between 5.3 and 5.12. Plain modules are simpler to read and
  test; a model that ships a schema can still be served by a user parser.
- **Engine parsers as a library** (SGLang's detectors, as miles uses them, or vLLM's). They cover more formats,
  but pull an engine's dependency tree into the CPU client, are mostly streaming code, and return malformed calls
  as content with only a log line.
- **tinker-cookbook's renderers.** They stay available as whole renderers (`renderer_name: qwen3_instruct`,
  `kimi_k2`, `gpt_oss_*`), and some parse their format exactly. Mixing their parsers into `chat_template` would
  add a second mechanism tied to cookbook internals, and the cookbook's Qwen3.5 parser is lossy (it strips
  indentation and types `"42"` as a number).
- **Template sniffing,** as SGLang selects. It is cheap but checks nothing, and finds no parser for
  Qwen3-4B-Instruct-2507's template.
- **Constrained decoding,** which prevents malformed calls. Tinker's `SamplingParams` has no grammar field, and
  training on constrained samples needs the sampler's masks; that is a separate design.
- **Doing nothing.** Only Qwen3.5 and later models work with `chat_template`, and their silent drops cost episodes
  nobody sees.

## Rollout plan

| Milestone | Scope | Exit check | Pull requests |
|---|---|---|---|
| M1 parsers | The ten parsers, argument typing, the registry, `NOTICE` | `tests/renderers/test_tool_parsers.py`: every format's well-formed output parses exactly, a cut-off or broken call is reported on its own, lenient rules mark calls recovered | |
| M2 round trip | `tool_parsers/probe.py`: `auto`, the refusal, names checked | `tests/renderers/test_tool_parser_probe.py`: templates in each format's shape choose their parser, and only it reads them back | |
| M3 renderer | `chat_template` parses with the chosen parser; reasoning from the prompt | `tests/renderers`: the 864 saved token fixtures, and `test_tool_parsing_templates.py` on every cached template, including the four Qwen drops | |
| M4 loud failures | Unparsed calls in the content, counts in the generation record | `tests/gateway`, `tests/train/tinker/test_gateway_tools_routing.py`, the CodeAct goldens | |
| M5 start and lock | Chosen in `build_tinker_llm` and by the CLI, recorded in `lock.json`, exit 78 | `tests/test_cli_tool_call_parsing.py` | |
| M6 metrics | Per-trajectory sums and rollout statistics | `tests/train/tinker/test_tool_call_metrics.py` | |
| M7 validation | A Qwen3.6 or Qwen3.8 training run through a self-hosted server | Tool-call counts per step against the same samples parsed by the old parser; no unparsed spikes | |

## Open questions

1. **Recovered calls.** Should calls read by a lenient rule execute (now, as engines and the old fallback did) or
   be recorded as unparsed under a strict setting? Decide before M7 compares runs.
2. **Penalties and masking.** Should an unparsed call cost reward, or keep its turn out of the loss, as
   tinker-cookbook's parse-error policy can? Both change what trains, so they need their own decision.
3. **Unknown tool names.** A call to a tool the request did not offer passes through, and the harness or the
   environment answers it, as engines that forward unknown tools do. Should it be unparsed instead?
4. **Vendor templates.** Before a format beyond Qwen's is called supported for a model family, its own chat
   template should pass `test_tool_parsing_templates.py`; for DeepSeek's that first needs the renderer to pass
   arguments as strings to templates that want them.
5. **transformers `response_template`.** When transformers ships it, a parser could wrap it, which would let models
   bring their own parsing.
