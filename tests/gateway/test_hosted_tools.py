"""A trained policy and a hosted model are shown the same tools (RFC 0015, section 4).

Each case is one row of the tool table: a request in one dialect goes through the training dialect's
``chat_request`` and through the hosted pass-through (the dialect's ``hosted_body``).  Both apply one normalization,
with no exception: every tool is a plain function of a name, a description and parameters (the empty-object schema
for one without), which the hosted model gets written in the dialect's own shape, and a request training refuses is
refused with the same message.  Over HTTP, the tools the chat-template renderer and a cookbook renderer are handed
are those the hosted model receives, and a refusal has the same status and error body on both paths.
"""

from __future__ import annotations

import asyncio
import gzip
import json
from pathlib import Path
from typing import Any, cast

import pytest
from aiohttp import ClientSession
from gateway_fakes import MODEL_NAME, TOOLS, Completion
from hosted_fakes import FAKE_KEY, MODEL, FakeProvider

from platoon.envs.service import ToolSpec
from platoon.eval.config_defs import HostedConfig
from platoon.gateway import GatewayServer
from platoon.gateway.core import ChatRequest, ModelGateway, chat_tools, to_cookbook_tool_specs
from platoon.gateway.dialects import anthropic, openai_chat, responses
from platoon.gateway.dialects.anthropic import AnthropicMessages
from platoon.gateway.dialects.openai_chat import OpenAIChat
from platoon.gateway.dialects.responses import OpenAIResponses
from platoon.gateway.hosted import HostedGateway
from platoon.harness.builtin import platoon_bash, react
from platoon.testing import TOY_TOKENIZER_DIR, toy_renderer

KEY_ENV = "PLATOON_TEST_PROVIDER_KEY"
EMPTY = {"type": "object", "properties": {}}
SCHEMA = {"type": "object", "properties": {"number": {"type": "integer"}}, "required": ["number"]}
CHAT_GUESS = {"type": "function", "function": {"name": "guess", "description": "Guess.", "parameters": SCHEMA}}
MESSAGES_GUESS = {"name": "guess", "description": "Guess.", "input_schema": SCHEMA}
RESPONSES_GUESS = {"type": "function", "name": "guess", "description": "Guess.", "parameters": SCHEMA}
USER = [{"role": "user", "content": "go"}]
# The tools a recorded platoon-openhands episode sent (tests/harness/openhands_live.py saved them).
OPENHANDS_REPLAY = Path(__file__).parents[1] / "harness" / "fixtures" / "openhands" / "replay.json.gz"
# A chat template that writes each tool it is handed as JSON on a line of its own (``tool | tojson``, as Qwen's
# templates do), so a prompt shows exactly the tools the chat-template renderer got.
TOOLS_TEMPLATE = (
    "{% if tools %}<|im_start|>system\n<tools>\n{% for tool in tools %}{{ tool | tojson }}\n{% endfor %}"
    "</tools><|im_end|>\n{% endif %}"
    "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

CHAT_DIALECT = OpenAIChat(cast(Any, None))  # its requests need no gateway: only logprobs read one
TRAINING = {
    "openai_chat": CHAT_DIALECT.chat_request,
    "anthropic_messages": AnthropicMessages().chat_request,
    "openai_responses": OpenAIResponses(CHAT_DIALECT).chat_request,
}
HOSTED = {
    "openai_chat": openai_chat.hosted_body,
    "anthropic_messages": anthropic.hosted_body,
    "openai_responses": responses.hosted_body,
}
PATHS = {"openai_chat": "/chat/completions", "anthropic_messages": "/messages", "openai_responses": "/responses"}


def chat(**fields: Any) -> dict[str, Any]:
    return {"model": "m", "messages": USER, **fields}


def messages(**fields: Any) -> dict[str, Any]:
    return {"model": "m", "max_tokens": 64, "messages": USER, **fields}


def responses_body(**fields: Any) -> dict[str, Any]:
    return {"model": "m", "input": "go", **fields}


def shown(dialect: str, body: dict[str, Any]) -> list[dict[str, Any]]:
    """The functions training shows a policy for ``body``: the tools its renderer is handed."""

    return [tool["function"] for tool in TRAINING[dialect](body).render.tools or []]


def as_functions(dialect: str, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The tools of a hosted request, written in ``dialect``'s shape, as the functions they are."""

    if dialect == "openai_chat":
        assert all(set(tool) == {"type", "function"} and tool["type"] == "function" for tool in tools)
        return [tool["function"] for tool in tools]
    if dialect == "anthropic_messages":
        return [{("parameters" if key == "input_schema" else key): value for key, value in t.items()} for t in tools]
    assert all(tool["type"] == "function" and tool["strict"] is False for tool in tools)
    return [{key: value for key, value in tool.items() if key not in ("type", "strict")} for tool in tools]


def cookbook_specs(fns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Functions as the cookbook renderers read them: a description of ``""`` for one without."""

    return [{"name": f["name"], "description": f.get("description") or "", "parameters": f["parameters"]} for f in fns]


# --------------------------------------------------------------------------- the tools each path yields

# (row of the tool table, dialect, request body)
PARITY = [
    ("chat tools", "openai_chat", chat(tools=[CHAT_GUESS, {**CHAT_GUESS, "function": {"name": "stop"}}])),
    (
        "chat tools without a schema or a description, flat or in another order",
        "openai_chat",
        chat(
            tools=[
                {"type": "function", "function": {"name": "a", "description": None, "parameters": None}},
                {"type": "function", "function": {"name": "b", "description": "", "parameters": {}}},
                {"type": "function", "function": {"parameters": SCHEMA, "name": "c"}},
                {"name": "d", "description": "A flat tool.", "parameters": SCHEMA},
                {
                    "function": {"name": "e", "description": "Written function first.", "parameters": SCHEMA},
                    "type": "function",
                },
            ]
        ),
    ),
    (
        "chat strict",
        "openai_chat",
        chat(tools=[{"type": "function", "function": {**CHAT_GUESS["function"], "strict": True}}]),
    ),
    (
        "chat tools of other types, and other fields",
        "openai_chat",
        chat(
            tools=[
                {
                    **CHAT_GUESS,
                    "cache_control": {"type": "ephemeral"},
                    "function": {**CHAT_GUESS["function"], "defer_loading": True},
                },
                {"type": "web_search_20250305", "name": "web_search", "max_uses": 3},
                {"type": "computer_20250124", "function": {"name": "computer", "parameters": SCHEMA}},
            ]
        ),
    ),
    (
        "messages tools and their other fields",
        "anthropic_messages",
        messages(
            tools=[
                {
                    **MESSAGES_GUESS,
                    "cache_control": {"type": "ephemeral"},
                    "input_examples": [{"number": 3}],
                    "eager_input_streaming": True,
                    "allowed_callers": ["direct"],
                    "strict": True,
                    "defer_loading": True,
                    "type": "custom",
                },
                {"name": "stop", "description": ""},
            ]
        ),
    ),
    (
        "messages server tools",
        "anthropic_messages",
        messages(
            tools=[
                MESSAGES_GUESS,
                {"type": "web_search_20250305", "name": "web_search", "max_uses": 3},
                {"type": "web_fetch_20250910", "name": "web_fetch"},
                {"type": "code_execution_20250825", "name": "code_execution"},
                {"type": "bash_20250124", "name": "bash"},
                {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"},
                {"type": "computer_20250124", "name": "computer", "display_width_px": 1024},
                {"type": "memory_20250818", "name": "memory"},
                {"type": "advisor_20260101", "name": "advisor"},
            ]
        ),
    ),
    (
        "messages MCP connector and tool search",
        "anthropic_messages",
        messages(
            tools=[
                {**MESSAGES_GUESS, "defer_loading": True},
                {"type": "mcp_toolset", "mcp_server_name": "remote", "default_config": {"defer_loading": True}},
                {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"},
                {"type": "tool_search_tool_bm25_20251119", "name": "tool_search_tool_bm25"},
            ],
            mcp_servers=[{"type": "url", "url": "https://mcp.example.com", "name": "remote"}],
        ),
    ),
    (
        "responses tools",
        "openai_responses",
        responses_body(
            tools=[
                {**RESPONSES_GUESS, "strict": True, "defer_loading": True, "output_schema": {"type": "string"}},
                {"type": "function", "name": "stop"},
                {"type": "function", "description": "A function without a name.", "parameters": SCHEMA},
                {"type": "web_search"},
                {"type": "file_search", "vector_store_ids": ["vs_1"]},
                {"type": "code_interpreter", "container": {"type": "auto"}},
                {"type": "computer_use_preview", "display_width": 1024, "display_height": 768, "environment": "linux"},
                {"type": "image_generation"},
                {"type": "local_shell"},
                {"type": "shell"},
                {"type": "apply_patch"},
                {"type": "custom", "name": "freeform"},
                {"type": "mcp", "server_label": "remote", "server_url": "https://mcp.example.com"},
                {"type": "tool_search"},
                {"type": "namespace", "name": "space", "tools": []},
            ]
        ),
    ),
]


BODIES = {row: body for row, _, body in PARITY}


@pytest.mark.parametrize(("row", "dialect", "body"), PARITY, ids=[case[0] for case in PARITY])
def test_the_hosted_model_gets_exactly_the_tools_training_shows(row, dialect, body):
    expected = shown(dialect, body)
    assert expected, "every case offers at least one tool"
    sent = HOSTED[dialect](body, model=MODEL).get("tools") or []
    assert as_functions(dialect, sent) == expected
    if dialect == "openai_chat":  # the very same tools, down to the order of their fields
        assert json.dumps(sent) == json.dumps(TRAINING[dialect](body).render.tools)


def test_training_shows_each_tool_as_a_plain_function_of_a_name_a_description_and_parameters():
    assert shown("openai_chat", BODIES["chat tools without a schema or a description, flat or in another order"]) == [
        {"name": "a", "parameters": EMPTY},
        {"name": "b", "description": "", "parameters": EMPTY},
        {"parameters": SCHEMA, "name": "c"},
        {"name": "d", "description": "A flat tool.", "parameters": SCHEMA},
        {"name": "e", "description": "Written function first.", "parameters": SCHEMA},
    ]
    # Tools of other types are functions of their name, which nothing runs; strict and every other field go.
    assert shown("openai_chat", BODIES["chat tools of other types, and other fields"]) == [
        CHAT_GUESS["function"],
        {"name": "web_search", "parameters": EMPTY},
        {"name": "computer", "parameters": SCHEMA},
    ]
    assert shown("openai_chat", BODIES["chat strict"]) == [CHAT_GUESS["function"]]
    # Messages: server tools are schema-less functions of their name, and a toolset, which has no name, is left out.
    assert shown("anthropic_messages", BODIES["messages tools and their other fields"]) == [
        {"name": "guess", "description": "Guess.", "parameters": SCHEMA},
        {"name": "stop", "description": "", "parameters": EMPTY},
    ]
    assert shown("anthropic_messages", BODIES["messages MCP connector and tool search"]) == [
        {"name": "guess", "description": "Guess.", "parameters": SCHEMA},
        {"name": "tool_search_tool_regex", "parameters": EMPTY},
        {"name": "tool_search_tool_bm25", "parameters": EMPTY},
    ]
    # Responses: the function entries that have a name.
    assert shown("openai_responses", BODIES["responses tools"]) == [
        {"name": "guess", "description": "Guess.", "parameters": SCHEMA},
        {"name": "stop", "parameters": EMPTY},
    ]


def test_plain_function_tools_reach_every_renderer_unchanged():
    """Tools sent as functions of a name, a description and parameters, as the harnesses build them, are shown as
    they came: the same JSON, so the same prompt."""

    # Saved with sorted keys, so each tool's function comes before its type, as a client whose encoder sorts keys
    # sends it.
    recorded = json.loads(gzip.open(OPENHANDS_REPLAY).read())["calls"][0]["tools"]
    assert [list(tool) for tool in recorded] == [["function", "type"]] * len(recorded)
    mcp_tool = {"name": "lookup", "inputSchema": SCHEMA}  # no description
    env_tool = ToolSpec(name="submit", description="", input_schema={})
    # The OpenAI Go SDK writes a tool's function before its type, and the description after the name.
    go_tool = {"function": {"name": "guess", "description": "Guess.", "parameters": SCHEMA}, "type": "function"}
    for tools in (
        TOOLS,
        [CHAT_GUESS],
        recorded,
        [go_tool],
        [platoon_bash.BASH_TOOL, platoon_bash.openai_tool(mcp_tool)],
        [react.openai_tool(env_tool)],
    ):
        assert json.dumps(chat_tools(tools)) == json.dumps(tools)
    assert to_cookbook_tool_specs(TOOLS) == cookbook_specs([TOOLS[0]["function"]])


def test_a_responses_function_is_shown_in_the_order_training_always_showed_it():
    """Name, description and parameters, whatever order the client writes them in: the OpenAI Agents SDK writes
    ``name, parameters, strict, type, description``, Codex ``type, name, description, strict, parameters``."""

    agents_sdk = {"name": "guess", "parameters": SCHEMA, "strict": True, "type": "function", "description": "Guess."}
    codex = {"type": "function", "name": "guess", "description": "Guess.", "strict": False, "parameters": SCHEMA}
    for tool in (agents_sdk, codex):
        assert json.dumps(shown("openai_responses", responses_body(tools=[tool]))) == json.dumps(
            [{"name": "guess", "description": "Guess.", "parameters": SCHEMA}]
        )
    no_description = {"parameters": SCHEMA, "type": "function", "name": "guess"}
    assert list(shown("openai_responses", responses_body(tools=[no_description]))[0]) == ["name", "parameters"]


def test_ctx_chat_tools_reach_the_hosted_model_as_training_shows_them():
    tools = [
        *BODIES["chat tools of other types, and other fields"]["tools"],
        *BODIES["chat tools without a schema or a description, flat or in another order"]["tools"],
    ]
    request = ChatRequest.from_params(USER, {"tools": tools, "tool_choice": "auto"})
    sent = openai_chat.hosted_body(openai_chat.in_process_body(request.messages, request.params), model=MODEL)
    assert sent["tools"] == request.render.tools and json.dumps(sent["tools"]) == json.dumps(request.render.tools)


def test_provider_side_fields_never_reach_the_hosted_model():
    chat_body = openai_chat.hosted_body(
        chat(
            tools=[CHAT_GUESS],
            tool_choice="auto",
            parallel_tool_calls=False,
            web_search_options={},
            functions=[{"name": "legacy"}],
            function_call="auto",
        ),
        model=MODEL,
    )
    assert set(chat_body) == {"model", "messages", "tools"}
    messages_body = anthropic.hosted_body(
        messages(
            tools=[MESSAGES_GUESS],
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
            mcp_servers=[{"type": "url", "url": "https://mcp.example.com", "name": "remote"}],
            container="container_1",
            context_management={"edits": [{"type": "clear_tool_uses_20250919"}]},
            compaction={"enabled": True},
        ),
        model=MODEL,
    )
    assert set(messages_body) == {"model", "max_tokens", "messages", "tools"}
    responses_hosted = responses.hosted_body(
        responses_body(
            tools=[RESPONSES_GUESS],
            tool_choice="none",
            parallel_tool_calls=False,
            max_tool_calls=3,
            previous_response_id="resp_1",
            conversation="conv_1",
            prompt={"id": "pmpt_1"},
            background=True,
            truncation="auto",
        ),
        model=MODEL,
    )
    assert set(responses_hosted) == {"model", "input", "tools"}


@pytest.mark.parametrize(
    ("dialect", "body"),
    [
        ("anthropic_messages", messages(tools=[{"type": "mcp_toolset", "mcp_server_name": "remote"}])),
        ("openai_responses", responses_body(tools=[{"type": "web_search"}], tool_choice="auto")),
        ("openai_chat", chat(tools=[])),
    ],
)
def test_a_request_whose_tools_all_leave_shows_and_sends_no_tools(dialect, body):
    assert TRAINING[dialect](body).render.tools is None
    assert "tools" not in HOSTED[dialect](body, model=MODEL)


# --------------------------------------------------------------------------- what training refuses

# (row of the tool table, dialect, request body)
REFUSED = [
    ("chat tools that are not a list", "openai_chat", chat(tools={"guess": CHAT_GUESS})),
    *[
        (f"chat tool without a name: {label}", "openai_chat", chat(tools=[CHAT_GUESS, tool]))
        for label, tool in (
            ("a provider's built-in", {"type": "browser_search"}),
            ("a custom tool", {"type": "custom", "custom": {"name": "patch", "format": {"type": "grammar"}}}),
            ("a function", {"type": "function", "function": {"description": "Nameless.", "parameters": SCHEMA}}),
            ("an empty name", {"type": "function", "function": {"name": "", "parameters": SCHEMA}}),
            ("not an object", "guess"),
        )
    ],
    ("chat tool_choice required", "openai_chat", chat(tools=[CHAT_GUESS], tool_choice="required")),
    (
        "chat named tool_choice",
        "openai_chat",
        chat(tools=[CHAT_GUESS], tool_choice={"type": "function", "function": {"name": "guess"}}),
    ),
    (
        "chat allowed tools",
        "openai_chat",
        chat(tools=[CHAT_GUESS], tool_choice={"type": "allowed_tools", "allowed_tools": {"mode": "auto", "tools": []}}),
    ),
    ("chat n", "openai_chat", chat(n=2)),
    ("messages tools that are not a list", "anthropic_messages", messages(tools={"guess": MESSAGES_GUESS})),
    ("messages tool that is not an object", "anthropic_messages", messages(tools=[MESSAGES_GUESS, "guess"])),
    ("messages tool_choice any", "anthropic_messages", messages(tools=[MESSAGES_GUESS], tool_choice={"type": "any"})),
    (
        "messages named tool_choice",
        "anthropic_messages",
        messages(tools=[MESSAGES_GUESS], tool_choice={"type": "tool", "name": "guess"}),
    ),
    (
        "responses tool_choice required",
        "openai_responses",
        responses_body(tools=[RESPONSES_GUESS], tool_choice="required"),
    ),
    (
        "responses named tool_choice",
        "openai_responses",
        responses_body(tools=[RESPONSES_GUESS], tool_choice={"type": "function", "name": "guess"}),
    ),
    *[
        (
            f"messages history with {block['type']}",
            "anthropic_messages",
            messages(messages=[*USER, {"role": "assistant", "content": [block]}, {"role": "user", "content": "more"}]),
        )
        for block in (
            {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "x"}},
            {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": []},
            {"type": "code_execution_tool_result", "tool_use_id": "srvtoolu_1", "content": {}},
            {"type": "mcp_tool_use", "id": "mcptoolu_1", "name": "x", "server_name": "remote", "input": {}},
            {"type": "mcp_tool_result", "tool_use_id": "mcptoolu_1", "content": []},
        )
    ],
    (
        "messages history with container_upload",
        "anthropic_messages",
        messages(messages=[{"role": "user", "content": [{"type": "container_upload", "file_id": "file_1"}]}]),
    ),
    *[
        (
            f"responses history with {item['type']}",
            "openai_responses",
            responses_body(input=[{"role": "user", "content": "go"}, item]),
        )
        for item in (
            {"type": "custom_tool_call", "call_id": "c1", "name": "freeform", "input": "x"},
            {"type": "local_shell_call", "call_id": "c1", "action": {"type": "exec", "command": ["ls"]}},
            {"type": "web_search_call", "id": "ws_1", "status": "completed"},
            {"type": "mcp_call", "id": "mcp_1", "name": "x", "server_label": "remote", "arguments": "{}"},
            {"type": "item_reference", "id": "msg_1"},
        )
    ],
]


@pytest.mark.parametrize(("row", "dialect", "body"), REFUSED, ids=[case[0] for case in REFUSED])
def test_what_training_refuses_the_hosted_path_refuses_with_the_same_message(row, dialect, body):
    with pytest.raises(ValueError) as training:
        TRAINING[dialect](body)
    with pytest.raises(ValueError) as hosted:
        HOSTED[dialect](body, model=MODEL)
    assert str(hosted.value) == str(training.value)


def test_a_chat_tool_without_a_name_is_refused_as_the_cookbook_renderers_always_refused_it():
    # The cookbook renderers' own message: no renderer shows such a tool now, and no hosted model gets it.
    for tool in ({"type": "browser_search"}, {"type": "custom", "custom": {"name": "patch"}}):
        with pytest.raises(ValueError, match=r"^Invalid tool definition: "):
            CHAT_DIALECT.chat_request(chat(tools=[tool]))
        with pytest.raises(ValueError, match=r"^Invalid tool definition: "):
            to_cookbook_tool_specs([tool])


@pytest.mark.parametrize(
    ("dialect", "body"),
    [
        ("openai_chat", chat(tools=[CHAT_GUESS], tool_choice="auto")),
        ("openai_chat", chat(tools=[CHAT_GUESS], tool_choice="none")),
        ("anthropic_messages", messages(tools=[MESSAGES_GUESS], tool_choice={"type": "auto"})),
        ("anthropic_messages", messages(tools=[MESSAGES_GUESS], tool_choice={"type": "none"})),
        ("openai_responses", responses_body(tools=[RESPONSES_GUESS], tool_choice="auto")),
        ("openai_responses", responses_body(tools=[RESPONSES_GUESS], tool_choice="none")),
    ],
)
def test_auto_and_none_pass_and_leave_the_provider_s_default(dialect, body):
    TRAINING[dialect](body)  # training accepts them, and forces nothing
    hosted = HOSTED[dialect](body, model=MODEL)
    assert "tool_choice" not in hosted and hosted["tools"]


def test_tool_results_in_history_pass_unchanged():
    result = {
        "type": "tool_result",
        "tool_use_id": "toolu_1",
        "is_error": True,
        "content": [
            {"type": "text", "text": "the chart"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0K"}},
        ],
    }
    history = [
        *USER,
        {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "plot", "input": {}}]},
    ]
    body = messages(messages=[*history, {"role": "user", "content": [result]}])
    with pytest.raises(ValueError, match="Unsupported content block type 'image'"):
        TRAINING["anthropic_messages"](body)  # a text-only renderer's limit, not a rule about tools
    assert anthropic.hosted_body(body, model=MODEL)["messages"][-1]["content"] == [result]


# --------------------------------------------------------------------------- both paths over HTTP


@pytest.fixture(autouse=True)
def _environment(monkeypatch):
    monkeypatch.setenv(KEY_ENV, FAKE_KEY)
    for name in ("http_proxy", "https_proxy", "all_proxy", "no_proxy"):  # the fake provider is on loopback
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.upper(), raising=False)


@pytest.fixture(scope="module")
def template_renderer():
    """platoon's chat-template renderer over a toy tokenizer of its own, with :data:`TOOLS_TEMPLATE`."""

    from transformers import PreTrainedTokenizerFast

    from platoon.train.tinker.renderers import ChatTemplateRenderer

    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(TOY_TOKENIZER_DIR))
    tokenizer.chat_template = TOOLS_TEMPLATE
    return ChatTemplateRenderer(tokenizer, tool_call_parser="none")


@pytest.fixture(params=["chat template", "cookbook"])
def training_gateway(request, make_gateway, sampler, template_renderer) -> ModelGateway:
    """A Tinker gateway over the chat-template renderer, or over a cookbook renderer (``qwen3_instruct``)."""

    if request.param == "cookbook":
        return make_gateway()
    renderer = template_renderer
    return ModelGateway(sampler=sampler, renderer=renderer, tokenizer=renderer.tokenizer, model_name=MODEL_NAME)


async def _post(base_url: str, path: str, key: str, body: dict[str, Any]) -> tuple[int, Any]:
    headers = {"Authorization": f"Bearer {key}", "x-api-key": key}
    async with ClientSession() as session:
        async with session.post(f"{base_url}/v1{path}", json=body, headers=headers) as response:
            return response.status, json.loads(await response.read())


async def _call(gateway: Any, path: str, body: dict[str, Any]) -> tuple[int, Any]:
    server = GatewayServer(gateway)
    base_url = await server.start(host="127.0.0.1")
    agent = gateway.open_agent(rollout_id="r", trajectory_id="t", interactions={})
    try:
        return await _post(base_url, path, agent.api_key, body)
    finally:
        await server.stop()
        if isinstance(gateway, HostedGateway):
            await gateway.aclose()


def _hosted_request(dialect: str, body: dict[str, Any]) -> tuple[int, Any, list[dict[str, Any]]]:
    """``body`` posted to a hosted gateway: the status and body it answered, and the requests the provider got."""

    with FakeProvider() as provider:
        settings = HostedConfig(base_url=provider.base_url, api_key_env=KEY_ENV, max_retries=0)
        status, answer = asyncio.run(_call(HostedGateway(settings, MODEL), PATHS[dialect], body))
        return status, answer, provider.requests


def _template_tools(renderer: Any, prompt: Any) -> list[dict[str, Any]]:
    """The tools a prompt rendered with :data:`TOOLS_TEMPLATE` shows, as the chat-template renderer was handed them."""

    text = renderer.tokenizer.decode(prompt.to_ints())
    block = text.split("<tools>\n", 1)[1].split("</tools>", 1)[0]
    return [json.loads(line) for line in block.splitlines() if line]


@pytest.mark.parametrize(("row", "dialect", "body"), PARITY, ids=[case[0] for case in PARITY])
def test_over_http_every_renderer_is_handed_the_tools_the_hosted_model_receives(
    row, dialect, body, tokenizer, sampler, template_renderer
):
    status, _, requests = _hosted_request(dialect, body)
    assert status == 200 and len(requests) == 1
    received = requests[0]["body"].get("tools") or []
    functions = as_functions(dialect, received)
    assert functions, "every case offers at least one tool"

    # The chat-template renderer: its prompt shows the very tools a hosted Chat Completions model receives.
    sampler.script = [Completion("OK")]
    template_gateway = ModelGateway(
        sampler=sampler, renderer=template_renderer, tokenizer=template_renderer.tokenizer, model_name=MODEL_NAME
    )
    status, _ = asyncio.run(_call(template_gateway, PATHS[dialect], {**body, "model": MODEL_NAME}))
    assert status == 200
    handed = _template_tools(template_renderer, sampler.calls[-1]["prompt"])
    assert [tool["function"] for tool in handed] == functions
    if dialect == "openai_chat":
        assert json.dumps(handed) == json.dumps(received)

    # A cookbook renderer (qwen3_instruct): the specs it folds into the system message are the same functions.
    cookbook = toy_renderer()
    specs: list[list[dict[str, Any]]] = []
    fold = cookbook.create_conversation_prefix_with_tools

    def recording_fold(tools: Any, *args: Any, **kwargs: Any) -> Any:
        specs.append(list(tools))
        return fold(tools, *args, **kwargs)

    cookbook.create_conversation_prefix_with_tools = recording_fold  # type: ignore[method-assign]
    sampler.script = [Completion("OK")]
    cookbook_gateway = ModelGateway(sampler=sampler, renderer=cookbook, tokenizer=tokenizer, model_name=MODEL_NAME)
    status, _ = asyncio.run(_call(cookbook_gateway, PATHS[dialect], {**body, "model": MODEL_NAME}))
    assert status == 200
    assert specs == [cookbook_specs(functions)]


@pytest.mark.parametrize(("row", "dialect", "body"), REFUSED, ids=[case[0] for case in REFUSED])
def test_a_refusal_over_http_has_training_s_status_and_body_and_sends_nothing(row, dialect, body, training_gateway):
    trained = asyncio.run(_call(training_gateway, PATHS[dialect], {**body, "model": MODEL_NAME}))
    status, answer, requests = _hosted_request(dialect, body)
    assert requests == []
    assert (status, answer) == trained and status == 400
