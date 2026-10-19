"""A toy tokenizer that speaks the Qwen3 chat format, so tests render, sample and parse offline.

The tokenizer is byte-level: token ``i`` is byte ``i`` for ``i`` in 0..255, and there are no merges, so any text
round-trips exactly and the token ids of ASCII text are its character codes.  Above the bytes come Qwen's chat
special tokens, with Qwen's ``special`` flags::

    256 <|endoftext|>    257 <|im_start|>    258 <|im_end|>
    259 <tool_call>      260 </tool_call>    261 <tool_response>
    262 </tool_response> 263 <think>         264 </think>

That is all tinker-cookbook's Qwen3 renderers need (``qwen3_instruct``, ``qwen3``, ``qwen3_disable_thinking``), so
a gateway over :func:`toy_tokenizer` and :func:`toy_renderer` renders the same prompt text as one over a Qwen3
tokenizer, and parses tool calls and thinking the same way.  Only the token ids and counts differ, so tests that
check what platoon does with a model call need no model download; tests of a model's own chat template still need
the model's tokenizer.

The files live in ``platoon/testing/toy_tokenizer/`` (``tokenizer.json`` and ``tokenizer_config.json``, about
7 KB), so ``transformers.AutoTokenizer.from_pretrained(TOY_TOKENIZER_DIR)`` loads the same tokenizer.
:func:`build_toy_tokenizer` writes them, and documents how they were made.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from tinker_cookbook.renderers import Renderer
    from tinker_cookbook.tokenizer_utils import Tokenizer

# The model name a toy gateway reports, and the name tinker-cookbook's ``get_tokenizer`` knows the toy tokenizer by
# once :func:`toy_tokenizer` has run.
TOY_MODEL_NAME = "platoon/toy"
# The tinker-cookbook renderer :func:`toy_renderer` returns by default.
TOY_RENDERER_NAME = "qwen3_instruct"
# The directory holding the toy tokenizer's ``tokenizer.json`` and ``tokenizer_config.json``.
TOY_TOKENIZER_DIR = Path(__file__).resolve().parent / "toy_tokenizer"
# The added tokens after the 256 bytes, in id order, with whether each is special (as in Qwen3's tokenizer).
SPECIAL_TOKENS: tuple[tuple[str, bool], ...] = (
    ("<|endoftext|>", True),
    ("<|im_start|>", True),
    ("<|im_end|>", True),
    ("<tool_call>", False),
    ("</tool_call>", False),
    ("<tool_response>", False),
    ("</tool_response>", False),
    ("<think>", False),
    ("</think>", False),
)
EOS_TOKEN = "<|im_end|>"
PAD_TOKEN = "<|endoftext|>"


@functools.cache
def toy_tokenizer() -> Tokenizer:
    """The toy tokenizer, loaded once per process (a ``transformers`` fast tokenizer).

    It is also registered with tinker-cookbook as :data:`TOY_MODEL_NAME`, so ``get_tokenizer(TOY_MODEL_NAME)``, and
    a renderer that tinker-cookbook rebuilds by model name, find it without a download.
    """

    from tinker_cookbook.tokenizer_utils import register_tokenizer
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(TOY_TOKENIZER_DIR / "tokenizer.json"),
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN,
        clean_up_tokenization_spaces=False,
        name_or_path=TOY_MODEL_NAME,
    )
    register_tokenizer(TOY_MODEL_NAME, toy_tokenizer)
    return cast("Tokenizer", tokenizer)


def toy_renderer(name: str = TOY_RENDERER_NAME) -> Renderer:
    """The tinker-cookbook renderer ``name`` over :func:`toy_tokenizer`: ``qwen3_instruct`` by default.

    The other Qwen3 renderers (``qwen3``, ``qwen3_disable_thinking``) work as well; renderers of other model families
    need their own special tokens, which the toy tokenizer does not have.
    """

    from tinker_cookbook.renderers import get_renderer

    return get_renderer(name, toy_tokenizer(), model_name=TOY_MODEL_NAME)


def build_toy_tokenizer(directory: str | Path) -> Path:
    """Write the toy tokenizer's files to ``directory`` and return it; how the shipped files were made.

    ``tokenizer.json`` is a byte-level BPE model with no merges whose vocabulary maps each byte's byte-level
    character to the byte's value, followed by :data:`SPECIAL_TOKENS`.  Needs the ``tokenizers`` package.
    """

    from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers, processors

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    characters = byte_characters()
    tokenizer = Tokenizer(models.BPE(vocab={characters[byte]: byte for byte in range(256)}, merges=[]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    for content, special in SPECIAL_TOKENS:
        token = AddedToken(content, single_word=False, lstrip=False, rstrip=False, normalized=False, special=special)
        if special:
            tokenizer.add_special_tokens([token])
        else:
            tokenizer.add_tokens([token])
    tokenizer.save(str(directory / "tokenizer.json"))
    config: dict[str, Any] = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": None,
        "eos_token": EOS_TOKEN,
        "pad_token": PAD_TOKEN,
        "unk_token": None,
        "clean_up_tokenization_spaces": False,
        "model_max_length": 1048576,
    }
    (directory / "tokenizer_config.json").write_text(json.dumps(config, indent=2) + "\n")
    return directory


def byte_characters() -> dict[int, str]:
    """The byte-level alphabet: the printable character that stands for each byte (GPT-2's ``bytes_to_unicode``)."""

    printable = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)]
    characters = {byte: chr(byte) for byte in printable}
    shifted = 0
    for byte in range(256):
        if byte not in characters:
            characters[byte] = chr(256 + shifted)
            shifted += 1
    return characters


if __name__ == "__main__":
    import sys

    print(build_toy_tokenizer(sys.argv[1] if len(sys.argv) > 1 else TOY_TOKENIZER_DIR))
