"""The sft trainer's datasets (:mod:`platoon.train.supervised`), in LLaMA-Factory's conventions: a ``dataset_info.json``
registry, the ``alpaca`` and ``sharegpt`` formats as LLaMA-Factory's converters align them, and loading, as its loader
loads files of a dataset directory and samples ``num_samples`` of their rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from platoon.train.supervised.formats import Conversation, SkippedRow, ToolCall, Turn, convert
from platoon.train.supervised.loading import load_conversations
from platoon.train.supervised.options import SupervisedOptions
from platoon.train.supervised.registry import Columns, DatasetEntry, Tags, read_entries
from platoon.utils.config import ConfigError

ALPACA = DatasetEntry(name="a", file_name="a.json", columns=Columns(system="system", history="history"))
SHAREGPT = DatasetEntry(
    name="s", file_name="s.json", formatting="sharegpt", columns=Columns(system="system", tools="tools")
)
WEATHER = {"name": "get_weather", "description": "A city's weather", "parameters": {"type": "object"}}


@pytest.fixture(autouse=True)
def _datasets_cache(tmp_path, monkeypatch):
    """``datasets.load_dataset`` caches what it loads: under the test's directory, not the user's cache."""

    import datasets.config

    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "datasets-cache")


def write_registry(directory: Path, registry: dict[str, Any], files: dict[str, Any]) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "dataset_info.json").write_text(json.dumps(registry))
    for name, rows in files.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith(".jsonl"):
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        else:
            path.write_text(json.dumps(rows))
    return str(directory)


# --- the registry ----------------------------------------------------------------------------------------------------


def test_an_entry_takes_llama_factory_s_defaults_and_its_columns_and_tags(tmp_path):
    directory = write_registry(
        tmp_path,
        {
            "plain": {"file_name": "plain.json"},
            "chat": {
                "file_name": "chat.json",
                "formatting": "sharegpt",
                "num_samples": 3,
                "columns": {"messages": "messages"},
                "tags": {
                    "role_tag": "role",
                    "content_tag": "content",
                    "user_tag": "user",
                    "assistant_tag": "assistant",
                },
            },
        },
        {},
    )

    plain, chat = read_entries(directory, ["plain", "chat"])

    assert plain == DatasetEntry(name="plain", file_name="plain.json")
    assert (plain.formatting, plain.split, plain.columns.prompt, plain.columns.query, plain.columns.response) == (
        "alpaca",
        "train",
        "instruction",
        "input",
        "output",
    )
    assert chat.num_samples == 3
    # The columns and tags an entry gives replace LLaMA-Factory's defaults whole, as its DatasetAttr.join does.
    assert chat.columns == Columns(prompt=None, query=None, response=None, messages="messages")
    assert chat.tags == Tags(
        role_tag="role",
        content_tag="content",
        user_tag="user",
        assistant_tag="assistant",
        observation_tag=None,
        function_tag=None,
        system_tag=None,
    )


def test_columns_and_tags_that_an_entry_gives_leave_the_rest_unread_as_llama_factory_does(tmp_path):
    """LLaMA-Factory's own firefly entry names the prompt and the response only: the query is no column (not
    ``input``, which is the prompt here); and its ultrachat_200k entry names no system tag: a first system message is
    then out of order, and its row is skipped."""

    directory = write_registry(
        tmp_path,
        {
            "firefly": {"file_name": "f.json", "columns": {"prompt": "input", "response": "target"}},
            "ultrachat_200k": {
                "file_name": "u.json",
                "formatting": "sharegpt",
                "columns": {"messages": "messages"},
                "tags": {
                    "role_tag": "role",
                    "content_tag": "content",
                    "user_tag": "user",
                    "assistant_tag": "assistant",
                },
            },
        },
        {},
    )
    firefly, ultrachat = read_entries(directory, ["firefly", "ultrachat_200k"])

    question = {"input": "Where is the Eiffel Tower?", "target": "In Paris."}
    assert convert(firefly, question).turns == (
        Turn("user", "Where is the Eiffel Tower?"),
        Turn("assistant", "In Paris."),
    )
    system_first = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]
    with pytest.raises(SkippedRow, match="message 1 is 'system', where user goes"):
        convert(ultrachat, {"messages": system_first})
    assert convert(ultrachat, {"messages": system_first[1:]}).turns == (Turn("user", "Hi"), Turn("assistant", "Hello"))


def test_hf_hub_url_wins_over_the_other_sources_and_another_source_alone_is_refused(tmp_path):
    directory = write_registry(
        tmp_path,
        {
            "hub": {"hf_hub_url": "org/data", "ms_hub_url": "org/data", "file_name": "local.json", "subset": "en"},
            "modelscope": {"ms_hub_url": "org/data"},
        },
        {},
    )

    (hub,) = read_entries(directory, ["hub"])
    assert (hub.hf_hub_url, hub.subset) == ("org/data", "en")
    with pytest.raises(
        ConfigError,
        match="its rows come from ms_hub_url, which LLaMA-Factory reads over file_name and this release does not read",
    ):
        read_entries(directory, ["modelscope"])


@pytest.mark.parametrize(
    ("entry", "problem"),
    [
        ({"file_name": "p.json", "ranking": True}, "ranking: preference datasets are not supported yet"),
        ({"file_name": "p.json", "columns": {"images": "images"}}, "columns: images: preference and multimodal"),
        ({"file_name": "p.json", "columns": {"chosen": "c", "rejected": "r"}}, "columns: chosen, rejected"),
        ({"file_name": "p.json", "formatting": "openai"}, "formatting: 'openai' .* is not supported yet"),
        ({"file_name": "p.json", "tags": {"role": "from"}}, r"tags: unknown keys \['role'\]"),
        ({"file_name": "p.json", "num_samples": 0}, "num_samples: must be an integer from 1"),
        ({"formatting": "alpaca"}, "give it hf_hub_url or file_name"),
        # LLaMA-Factory reads these sources over file_name.
        ({"file_name": "p.json", "script_url": "s"}, "its rows come from script_url, which LLaMA-Factory reads over"),
        ({"file_name": "p.json", "cloud_file_name": "c"}, "its rows come from cloud_file_name"),
        ({"file_name": "p.json", "ms_hub_url": "m", "om_hub_url": "o"}, "its rows come from ms_hub_url"),
        # The columns and tags given replace the defaults, so what every row is read by must be among them.
        ({"file_name": "p.json", "columns": {"prompt": "q"}}, "columns.response: name it"),
        (
            {"file_name": "p.json", "formatting": "sharegpt", "tags": {"user_tag": "user"}},
            "tags.role_tag, tags.content_tag: name it",
        ),
        ({"file_name": "p.json", "formatting": "sharegpt", "columns": {"system": "s"}}, "columns.messages: name it"),
    ],
)
def test_what_this_release_does_not_read_is_refused_by_name(tmp_path, entry, problem):
    directory = write_registry(tmp_path, {"d": entry}, {})

    with pytest.raises(ConfigError, match=problem):
        read_entries(directory, ["d"])


def test_a_key_llama_factory_does_not_read_is_ignored_with_a_warning(tmp_path, caplog):
    directory = write_registry(
        tmp_path,
        {"rows": {"file_name": "rows.json", "file_sha1": "0123"}},
        {"rows.json": [{"instruction": "q", "output": "a"}]},
    )

    (entry,) = read_entries(directory, ["rows"])
    assert entry.ignored == ("file_sha1",)
    assert (
        len(load_conversations(SupervisedOptions(dataset="rows", dataset_dir=directory, num_epochs=1)).conversations)
        == 1
    )
    assert "Dataset rows: its entry's file_sha1, which LLaMA-Factory does not read, is ignored" in caplog.text


def test_a_name_the_registry_does_not_hold_and_a_missing_registry_are_refused(tmp_path):
    directory = write_registry(tmp_path / "data", {"known": {"file_name": "k.json"}}, {})

    with pytest.raises(ConfigError, match="has no dataset 'unknown'"):
        read_entries(directory, ["known", "unknown"])
    with pytest.raises(ConfigError, match="trainer_options.dataset_dir: cannot read .*dataset_info.json"):
        read_entries(str(tmp_path / "elsewhere"), ["known"])


# --- the alpaca format -----------------------------------------------------------------------------------------------


def test_alpaca_joins_instruction_and_input_and_puts_the_history_first():
    row = {
        "instruction": "Translate.",
        "input": "Bonjour",
        "output": "Hello",
        "system": "Be brief.",
        "history": [["Hi", "Hello!"], ["Who are you?", "A translator."]],
    }

    assert convert(ALPACA, row) == Conversation(
        turns=(
            Turn("user", "Hi"),
            Turn("assistant", "Hello!"),
            Turn("user", "Who are you?"),
            Turn("assistant", "A translator."),
            Turn("user", "Translate.\nBonjour"),
            Turn("assistant", "Hello"),
        ),
        system="Be brief.",
    )
    # An empty input is left out, and so is a column the entry does not name.
    assert convert(DatasetEntry(name="a"), {"instruction": "Hi", "input": "", "output": "Hello", "system": "x"}) == (
        Conversation(turns=(Turn("user", "Hi"), Turn("assistant", "Hello")))
    )


@pytest.mark.parametrize(
    ("row", "why"),
    [
        ({"instruction": "Hi", "output": None}, "its response .output. is not text"),
        ({"instruction": "Hi", "output": "x", "history": [["only a prompt"]]}, "its history holds"),
    ],
)
def test_an_alpaca_row_that_trains_nothing_is_skipped(row, why):
    with pytest.raises(SkippedRow, match=why):
        convert(ALPACA, row)


# --- the sharegpt format ---------------------------------------------------------------------------------------------


def test_sharegpt_reads_tool_calls_results_tools_and_the_system_prompt():
    row = {
        "conversations": [
            {"from": "human", "value": "Weather in Rome and Oslo?"},
            {
                "from": "function_call",
                "value": json.dumps(
                    [{"name": "get_weather", "arguments": {"city": "Rome"}}, {"name": "get_weather", "arguments": {}}]
                ),
            },
            {"from": "observation", "value": '["sunny", "rain"]'},
            {"from": "gpt", "value": "Sunny in Rome, rain in Oslo."},
        ],
        "system": "Be brief.",
        "tools": json.dumps([{"type": "function", "function": WEATHER}]),
    }

    assert convert(SHAREGPT, row) == Conversation(
        turns=(
            Turn("user", "Weather in Rome and Oslo?"),
            Turn("function", calls=(ToolCall("get_weather", '{"city": "Rome"}'), ToolCall("get_weather", "{}"))),
            Turn("observation", '["sunny", "rain"]'),
            Turn("assistant", "Sunny in Rome, rain in Oslo."),
        ),
        system="Be brief.",
        tools=(WEATHER,),
    )


def test_sharegpt_takes_a_first_system_message_over_the_column_and_role_content_tags_as_the_openai_format():
    tags = Tags(role_tag="role", content_tag="content", user_tag="user", assistant_tag="assistant")
    entry = DatasetEntry(
        name="o", formatting="sharegpt", columns=Columns(messages="messages", system="system"), tags=tags
    )
    row = {
        "messages": [
            {"role": "system", "content": "From the message."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ],
        "system": "From the column.",
    }

    assert convert(entry, row) == Conversation(
        turns=(Turn("user", "Hi"), Turn("assistant", "Hello")), system="From the message."
    )


@pytest.mark.parametrize(
    ("messages", "why"),
    [
        ([{"from": "gpt", "value": "Hello"}, {"from": "human", "value": "Hi"}], "message 1 is 'gpt', where human"),
        ([{"from": "human", "value": "Hi"}, {"from": "human", "value": "Hi"}], "message 2 is 'human', where gpt"),
        ([{"from": "human", "value": "Hi"}], "it has 1 messages, not an even number"),
        ([], "it has 0 messages"),
        (
            [{"from": "human", "value": "Hi"}, {"from": "function_call", "value": "get_weather"}],
            "a tool call is not JSON",
        ),
        ([{"from": "human", "value": "Hi"}, {"from": "gpt", "value": 3}], "message 2's value is not text"),
    ],
)
def test_a_sharegpt_row_out_of_llama_factory_s_order_is_skipped(messages, why):
    with pytest.raises(SkippedRow, match=why):
        convert(SHAREGPT, {"conversations": messages})


# --- loading ---------------------------------------------------------------------------------------------------------


def test_the_datasets_load_from_a_file_and_a_folder_in_order_and_skip_what_llama_factory_would(tmp_path, caplog):
    alpaca = [{"instruction": f"q{index}", "input": "", "output": f"a{index}"} for index in range(3)]
    sharegpt = [
        {"conversations": [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "hello"}]},
        {"conversations": [{"from": "human", "value": "dangling"}]},
    ]
    directory = write_registry(
        tmp_path,
        {
            "alpaca": {"file_name": "alpaca.json"},
            "chat": {"file_name": "chat", "formatting": "sharegpt"},
        },
        {"alpaca.json": alpaca, "chat/part-0.jsonl": sharegpt[:1], "chat/part-1.jsonl": sharegpt[1:]},
    )

    dataset = load_conversations(SupervisedOptions(dataset="alpaca, chat", dataset_dir=directory, num_epochs=1))

    assert [conversation.turns[0].text for conversation in dataset.conversations] == ["q0", "q1", "q2", "hi"]
    origins = [conversation.origin for conversation in dataset.conversations]
    assert origins == ["alpaca row 0", "alpaca row 1", "alpaca row 2", "chat row 0"]
    assert (
        "Dataset chat: skipped 1 of its 2 rows" in caplog.text and "the first, row 1: it has 1 messages" in caplog.text
    )
    again = load_conversations(SupervisedOptions(dataset=["alpaca", "chat"], dataset_dir=directory, num_epochs=1))
    assert again.fingerprint == dataset.fingerprint


def test_num_samples_takes_every_row_once_before_any_twice_as_llama_factory_does(tmp_path):
    rows = [{"instruction": f"q{index}", "output": "a"} for index in range(4)]
    directory = write_registry(
        tmp_path,
        {"few": {"file_name": "rows.json", "num_samples": 2}, "more": {"file_name": "rows.json", "num_samples": 10}},
        {"rows.json": rows},
    )

    def prompts(name: str, seed: int = 42) -> list[str]:
        options = SupervisedOptions(dataset=name, dataset_dir=directory, num_epochs=1, seed=seed)
        return [conversation.turns[0].text for conversation in load_conversations(options).conversations]

    few, more = prompts("few"), prompts("more")
    assert len(few) == 2 and len(set(few)) == 2
    assert len(more) == 10 and set(more[:4]) == {f"q{index}" for index in range(4)}
    assert prompts("few") == few  # seeded


def test_a_dataset_that_does_not_load_or_trains_nothing_stops_the_run(tmp_path):
    directory = write_registry(
        tmp_path,
        {
            "missing": {"file_name": "missing.json"},
            "text": {"file_name": "notes.txt"},
            "empty": {"file_name": "empty.json"},
        },
        {"notes.txt": [], "empty.json": [{"instruction": "q", "output": None}]},
    )

    for name, problem in [
        ("missing", "missing.json does not exist"),
        ("text", "must be files of one type, one of arrow, csv, json, jsonl, parquet"),
        ("empty", "empty has no conversation to train on"),
    ]:
        with pytest.raises(ConfigError, match=problem):
            load_conversations(SupervisedOptions(dataset=name, dataset_dir=directory, num_epochs=1))
