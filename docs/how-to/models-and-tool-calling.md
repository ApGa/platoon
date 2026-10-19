# Models and tool calling

Every model call goes through the policy's **renderer**: it turns messages and tools into the prompt's tokens, and
the tokens the model samples back into reasoning, text and tool calls. This guide is about the second half for
the `chat_template` renderer, which renders with the model's own chat template: how it reads the model's tool
calls, which models it reads, how to choose or write a parser, and what happens when a call does not parse.
Training always uses the tokens the model sampled, so parsing never changes what trains; it changes which tool
calls the harness runs. The design is [RFC 0012](../rfcs/0012-tool-call-parsing.md).

## Choose a renderer

| Your model | Set |
|---|---|
| One tinker-cookbook renders, and whose cookbook parser you trust (`qwen3_instruct`, `kimi_k2`, `gpt_oss_*`, ...) | `train.renderer_name` to that renderer; it parses its own tool calls |
| One whose own chat template should render the prompts | `train.renderer_name: chat_template` (or a [preset](train-and-eval.md#renderers)); its tool calls are read by one of the parsers below |

Evaluation configs set the same keys under `inference:`, and `platoon rollout` takes them from whichever section its
config has. Train and evaluate a checkpoint with the same renderer and settings.

## Supported formats

With `chat_template`, `renderer_kwargs.tool_call_parser` names the parser. Names are SGLang's
`--tool-call-parser` names, and vLLM's spellings work as aliases, so the name a model's serving docs give works
here too.

| `tool_call_parser` | Aliases | Models | Reasoning | Checked against |
|---|---|---|---|---|
| `qwen3_coder` | `qwen3_xml`, `step3p5`, `nanbeige` | Qwen3-Coder, Qwen3.5, Qwen3.6, Qwen3.8 | `<think>` | the own templates of Qwen3.5, 3.6 and 3.8 |
| `hermes` | `qwen25`, `qwen` | Qwen2.5, Qwen3, Qwen3-2507, Hermes | `<think>` | Qwen3-4B-Instruct-2507's own template |
| `glm45` | `glm`, `glm47` | GLM-4.5 and later | `<think>` | the format |
| `deepseekv31` | `deepseek_v31` | DeepSeek-V3.1 | `<think>` | the format |
| `deepseekv3` | `deepseek_v3` | DeepSeek-V3, DeepSeek-R1 | `<think>` | the format |
| `kimi_k2` | | Kimi K2, K2.5 | `<think>` | the format |
| `mistral` | | Mistral (tokenizer v3, and v11 and later), Magistral | `[THINK]` | the format |
| `gpt-oss` | `openai`, `gpt_oss` | gpt-oss (Harmony) | the `analysis` channel | the format |
| `llama3` | `llama3_json`, `llama4_json` | Llama 3.1, 3.2, 3.3 | `<think>` | the format |
| `pythonic` | `llama4_pythonic` | Llama 4 | `<think>` | the format |

"Own templates" means the tests render tool calls through those models' chat templates and parse them back
exactly. "The format" means the tests use output written in the format and templates written in its shape;
the vendors' own templates are not in the tests. Either way, `auto` (below) checks the model's own template before
a run starts, so a template that does not match its format stops the run instead of losing calls.

Each parser reads a whole response: the reasoning, the text, the calls in order, and every call block it could
not parse. Formats that write argument values as text (`qwen3_coder`, `glm45`) get them typed by the request's
tool schemas: `3` is an integer only when the parameter allows one, and a parameter the schema does not describe
stays text. Arguments are always JSON: a text value JSON cannot hold as its type (a number out of range, say) stays
text, and a JSON call with `NaN` or `Infinity` in its arguments, or nested more than 1,000 levels deep, is not
parsed, on every Python. `pythonic` reads a call list only at the start of the response, so prose that mentions
`[name(key=value)]` is text.

## Set the parser

```yaml
train:
  model_name: Qwen/Qwen3.6-27B
  renderer_name: chat_template
  renderer_kwargs:
    tool_call_parser: auto        # the default
```

| `tool_call_parser` | Effect |
|---|---|
| `auto` | Chooses the parser that reads the template's own rendering of a probe turn back exactly |
| a name or alias from the table | That parser; it is still checked against the template, and a mismatch is a warning |
| `package.module:attr` | A parser of your own ([below](#write-your-own-parser)) |
| `none` | No tool calls: everything outside the reasoning is text |

`reasoning_tags` sets where the model's reasoning is, for a model whose tags are not its format's:

```yaml
  renderer_kwargs:
    tool_call_parser: hermes
    reasoning_tags: ["<seed:think>", "</seed:think>"]   # or none: the model does not reason in tags
```

Whether a response starts inside reasoning comes from its prompt: when the prompt the model sampled after ends in
the open reasoning tag (Qwen's templates end it in `<think>` when thinking is on), the response starts in
reasoning. A request's `chat_template_kwargs: {enable_thinking: false}` therefore needs no parser setting.

## What `auto` checks

When the trainer, a rollout worker or an evaluation policy builds its renderer, and before `platoon train`,
`platoon eval` or `platoon rollout` start, `auto` renders a short conversation through the model's chat template:
an assistant turn with reasoning, a sentence and three tool calls (an integer, an indented multi-line string,
strings that read as a number and a boolean, a list, a boolean, and a call with no arguments). It takes what a
model would sample for that turn, the render after the generation prompt, cut at the stop token, and keeps the
first parser that reads it back exactly. A template that puts one call in each sampled turn is checked one call at
a time. The choice is made once per model and settings in each process, takes milliseconds, and is logged:

```text
Tool-call parser for Qwen/Qwen3.6-27B: qwen3_coder (auto, round trip one turn with every call)
```

`lock.json` in the run directory records it under `tool_call_parsing`:

```json
"tool_call_parsing": {
  "model_name": "Qwen/Qwen3.6-27B",
  "renderer_name": "chat_template",
  "setting": "auto",
  "parser": "qwen3_coder",
  "verified": true,
  "reason": "round trip (one turn with every call)"
}
```

`verified` is `false` for a parser set by name that did not read the probe back, and `null` when there was nothing
to check (`none`, or a template that renders no tools). With a tinker-cookbook renderer, `lock.json` records that
the renderer parses its own calls.

## When no parser fits

If the template renders tools and no parser reads its probe turn back, the command stops before anything runs, with
exit code 78 (a configuration error):

```text
platoon: invalid config:
  renderer_kwargs.tool_call_parser: No tool-call parser reads the chat template's own tool calls back for my-org/my-model.
The template renders the probe (one turn with every call) as:
    Sure.<call name='get_weather'>{"location": "Tokyo, JP", "days": 3, "unit": "celsius"}</call><call name='write_fi...
What each parser made of it:
    qwen3_coder: read no calls; read the text 'Sure.<call name=\'get_weather\'>{"location": "Tokyo, JP", "day...', not 'Sure.'
    hermes: read no calls; read the text 'Sure.<call name=\'get_weather\'>{"location": "Tokyo, JP", "day...', not 'Sure.'
    ...
Set renderer_kwargs.tool_call_parser to the parser that matches the model (qwen3_coder, hermes, ...), to
package.module:attr for a parser of your own, or to none to parse no tool calls; ...
```

What to do:

- If the model's serving docs name a parser (`--tool-call-parser` in SGLang or vLLM), set that name. If its
  reasoning tags differ from its format's, set `reasoning_tags` too. A parser set by name runs even when the check
  fails, with a warning, so you can train while a template quirk is investigated.
- If the format is new, [write a parser](#write-your-own-parser).
- If the model should not call tools, set `tool_call_parser: none`.

The same error, from a trainer started without the command line, is a `ToolCallParserError`. A template that
renders tools but leaves the assistant's tool calls out of the conversation (or raises on them) cannot be checked,
and stops the run the same way: set the parser by name.

## Malformed and unparsed calls

Models sometimes write a call that does not quite follow their format. What happens:

- **Read by a lenient rule.** A missing close tag (`</parameter>`, `</function>`, `</tool_call>`, `</arg_value>`),
  a raw newline inside a JSON string, and arguments written as a JSON string are read where the structure after
  them shows what was meant. The call runs, and counts as recovered. The XML formats do not escape markup, so a
  well-formed value can look like a missing close tag: a value that holds a line such as `<parameter=mode>`, for a
  parameter the call has not given yet (and does not give later), is split there, and the call runs with the value
  cut and counts as recovered. A rise in recovered calls is worth a look for that reason.
- **Not parsed.** A call cut off before its end, JSON that does not decode, or a `<tool_call>` that holds another
  format's call is not run. Its raw text follows the response's content, as inference engines return text they
  cannot parse, so the harness's own format feedback can answer it. Only that call is affected: the response's
  other calls still run.
- **Cut by the token limit.** A response that stopped at `max_tokens` runs no calls at all.

Every model call's generation record (`misc["model_generations"]` of each trajectory) counts `parsed_tool_calls`, `emitted_tool_calls`, `recovered_tool_calls` and `unparsed_tool_calls`. Training
reports them per trajectory as `rollout_completion/num_tool_calls`, `num_recovered_tool_calls` and
`num_unparsed_tool_calls`. A rise in unparsed calls means the policy is drifting from its format; the gateway also
warns about the first few in its log.

## Write your own parser

A parser is any object with a `parse` method that returns a `ParseResult`. The simplest way is to subclass
`ToolCallParser`, which already splits the reasoning, and read the calls out of the rest:

```python
# my_parsers.py
import json

from platoon.train.tinker.renderers.tool_parsers import ParsedToolCall, ParseResult, ToolCallParser


class AngleCallParser(ToolCallParser):
    """Calls written as <call name='NAME'>{...json arguments...}</call>."""

    name = "angle"
    default_reasoning_tags = ("<think>", "</think>")

    def parse_calls(self, text, tools):
        content, *blocks = text.split("<call name='")
        result = ParseResult(content=content.strip())
        for block in blocks:
            name, rest = block.split("'>", 1)
            arguments = json.loads(rest.split("</call>", 1)[0])
            result.tool_calls.append(ParsedToolCall(name=name, arguments=arguments))
        return result
```

```yaml
  renderer_kwargs:
    tool_call_parser: my_parsers:AngleCallParser
```

The module must be importable wherever the renderer is built: in the trainer, in the rollout workers and in
evaluation. A parser set this way is checked against the template like a built-in one. What platoon needs of it:

- `parse(text, *, tools=None, in_reasoning=False)`: `text` is the sampled text before the stop token; `tools` are
  the request's OpenAI tool definitions (`ToolSchemas(tools).coerce(name, values)` types text values by them);
  `in_reasoning` is true when the prompt ended in the parser's `reasoning_open` tag (`<think>` for a parser that has
  no such attribute).
- `ParseResult(content, reasoning, tool_calls, unparsed)`: strip `content` and `reasoning`; keep the raw text of
  anything that looks like a call but does not parse in `unparsed` (`UnparsedToolCall(raw_text, error)`), not in
  the content; set `ParsedToolCall.recovered` for calls read by a lenient rule, and `id` when the format carries
  call ids that the template writes back. Arguments must be JSON values (no `NaN`, sets or bytes).
- Never raise on what a model wrote. If the parser raises anyway, or returns arguments that are not JSON, the whole
  response becomes one unparsed call and the error is logged; the model call itself does not fail.

A function with `parse`'s signature works too (`tool_call_parser: my_parsers:parse_angle_calls`), and so does a
class that builds without arguments. To add a format to platoon itself, put a module next to the others in
`platoon/train/tinker/renderers/tool_parsers/`, register its class in `__init__.py`, and add its malformed-output
cases to `tests/renderers/test_tool_parsers.py`.
