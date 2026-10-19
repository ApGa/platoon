# Renderers and tool-call parsers

**Goal.** Train a model whose chat format platoon does not render out of the box: choose an existing renderer
when one fits, or write one, register it in every process that builds a renderer, select it in a config, and
check that it reproduces the tokens the model sampled. For a model whose chat template platoon can render but whose
tool calls it cannot read, write a tool-call parser instead.

**Prerequisites.**

- Platoon installed with the `tinker` extra (`pip install 'platoon[tinker,ors]'`, or
  `uv sync --extra tinker --extra ors` in a checkout).
- [Train and evaluate](../how-to/train-and-eval.md#renderers): the renderers platoon ships.

## What a renderer does

A renderer turns chat messages and tool definitions into the prompt's tokens, and the tokens the model samples
back into an assistant message with its reasoning, text and tool calls. The model gateway renders every model call
with the renderer that `train.renderer_name` (or `inference.renderer_name` for evaluation) names, in the trainer,
in evaluation and in every rollout worker.

Training uses the exact tokens the model sampled, and merges consecutive calls of one agent into one training
sequence when a call's prompt extends the previous prompt and reply. So a renderer must prompt the model the way the
model was trained to be prompted, and must render an earlier assistant turn back to the tokens that were sampled
for it; where it does not, training data splits into more sequences instead of training on tokens that were never
sampled.

## Choose a renderer

| Your model | Set |
|---|---|
| One that tinker-cookbook renders (`tinker_cookbook.renderers.get_renderer`: `qwen3`, `qwen3_instruct`, `llama3`, `deepseekv3`, `kimi_k2`, `gpt_oss_medium_reasoning` and more) | `train.renderer_name` to that name |
| One whose own Hugging Face chat template should render the prompts, and whose tool calls are in a format platoon reads ([supported formats](../how-to/models-and-tool-calling.md#supported-formats)) | `train.renderer_name: chat_template`, with its settings in `train.renderer_kwargs` ([Train and evaluate](../how-to/train-and-eval.md#renderers)) |
| One whose chat template suits `chat_template`, but whose tool calls are in another format | `chat_template` with a [tool-call parser](#tool-call-parsers) of your own |
| Anything else | A renderer of your own (below) |

Train and evaluate a checkpoint with the same renderer and settings.

## Steps

### 1. Write the renderer

Subclass tinker-cookbook's `Renderer`, implement its three abstract methods, and register a factory under a name.
The factory receives the tokenizer and an image processor:

```python
# my_renderers/__init__.py
from tinker_cookbook.renderers import register_renderer
from tinker_cookbook.renderers.base import Renderer


class MyModelRenderer(Renderer):
    """Prompts and response parsing for MyModel's chat format."""

    def __init__(self, tokenizer, image_processor=None, *, enable_thinking: bool = True):
        super().__init__(tokenizer)
        self.enable_thinking = enable_thinking  # settable with train.renderer_kwargs

    def render_message(self, message, ctx):
        """The tokens of one message: its header, its content and the end-of-turn marker."""
        raise NotImplementedError

    def get_stop_sequences(self):
        """The token ids that end a sampled turn."""
        raise NotImplementedError

    def parse_response(self, response):
        """The sampled tokens as an assistant message (with any tool calls), and whether they parsed."""
        raise NotImplementedError


register_renderer("my_model", MyModelRenderer)
```

Each key of `renderer_kwargs` sets the renderer's attribute of that name after the factory built it, so expose
settings as attributes; a key the renderer has no attribute for is an error. The base class has the helpers the
cookbook's own renderers use; [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)'s
renderers are complete examples to start from.

How the gateway calls it: when the renderer's `build_generation_prompt` takes a `tools` parameter, it receives the
OpenAI-format messages unchanged, and the tools as OpenAI function tools; otherwise the gateway converts the messages
to tinker-cookbook's format and puts the tools into the system message with `create_conversation_prefix_with_tools`.
Either way each tool is a plain function of the name, description and parameters the request gave, whatever the
dialect: every other field is left out, and a tool without a schema gets `{"type": "object", "properties": {}}`.
Other keyword parameters of `build_generation_prompt` receive the matching per-request `chat_template_kwargs`.

### 2. Register it in every process

`register_renderer` runs when its module is imported, and every process that builds a renderer needs it: the
trainer, evaluation and each rollout worker. Name the module as an environment entry's `package:`, which those
processes import before they build the renderer, and select the renderer by its name:

```yaml
environments:
  - service: python://guess:GuessTheNumber
    harness: platoon-react
    package: my_renderers         # imported in every process, so "my_model" is registered
train:
  model_name: my-org/my-model
  renderer_name: my_model
  renderer_kwargs: {enable_thinking: false}
```

### 3. Check it on CPU

Check the two properties training relies on with the model's tokenizer, and keep the results as fixtures so a
change to the renderer shows up in review:

- **Prompts.** Render a few conversations (with and without tools, with a tool result, with reasoning) and compare
  the prompt tokens with the ones the model's own chat template produces, or with ids recorded from a working
  setup.
- **Round trips.** Render an assistant turn, cut the prompt at the start of that turn, and check that
  `parse_response` reads the turn's tokens back into the same message, tool calls included; then check that
  rendering the conversation with that message appended starts with the earlier prompt plus the turn's tokens,
  which is what lets consecutive calls merge into one training sequence.

Platoon's own renderers are checked this way in `tests/renderers`: prompt token ids recorded as fixtures for a
set of conversations and settings, and prefix merging that cuts training data only where re-rendering differs
from what was sampled.

## Tool-call parsers

With `chat_template`, the renderer reads the model's tool calls with the parser that `renderer_kwargs.tool_call_parser`
names: `auto`, the default, picks the parser that reads the model's own chat template back exactly; a name picks one
of the formats platoon reads; `package.module:attr` names a parser of your own. A parser is an object with a
`parse(text, *, tools=None, in_reasoning=False)` method that returns a `ParseResult`, usually a subclass of
`ToolCallParser` from `platoon.train.tinker.renderers.tool_parsers`:

```yaml
train:
  model_name: my-org/my-model
  renderer_name: chat_template
  renderer_kwargs:
    tool_call_parser: my_parsers:AngleCallParser
```

The parser's module must be importable wherever the renderer is built: the trainer, the rollout workers and
evaluation. Platoon checks a parser of your own against the model's chat template before the run starts, as it
checks its own, and records the parser it chose in the run's `lock.json`. [Models and tool
calling](../how-to/models-and-tool-calling.md#write-your-own-parser) lists the formats platoon reads, explains the
check, and walks through writing a parser.

## Next steps

- [Models and tool calling](../how-to/models-and-tool-calling.md): the tool-call formats platoon reads, and what
  happens to a call that does not parse.
- [Train and evaluate](../how-to/train-and-eval.md#renderers): the built-in renderers and their settings.
- [Extending platoon](index.md): every extension point and how platoon finds it.
