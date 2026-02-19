"""Monkey patches for areal library.

Import this module before using areal to apply patches.
"""

# Args to silently drop without warning (e.g., model is always passed but not used by areal)
SILENTLY_DROPPED_ARGS = {"model"}


def patch_proxy_chat_template_kwargs():
    """Patch areal proxy to properly handle chat_template_kwargs.

    sglang's ChatCompletionRequest extracts chat_template_kwargs from extra_body
    to a top-level field, but areal's client expects it inside extra_body.
    This patch modifies build_app to move chat_template_kwargs into extra_body
    before the proxy filters out unknown kwargs.
    """
    import areal.experimental.openai.proxy as proxy_module

    original_build_app = proxy_module.build_app

    def patched_build_app(client=None, session_cache=None):
        # Import here to avoid circular imports
        from areal.utils import logging
        from fastapi import Depends, HTTPException
        from openai.types.chat.chat_completion import ChatCompletion
        from sglang.srt.entrypoints.http_server import validate_json_request
        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

        logger = logging.getLogger("ArealOpenAI Proxy")

        # Build the original app
        app = original_build_app(client, session_cache)

        def get_shared_data():
            return app.state.shared_data

        # Remove the existing chat completions route from the router
        app.router.routes = [
            r for r in app.router.routes if not (hasattr(r, "path") and r.path == "/v1/{session_id}/chat/completions")
        ]

        # Add our patched version
        @app.post(
            "/v1/{session_id}/chat/completions",
            dependencies=[Depends(validate_json_request)],
        )
        async def chat_completions(request: ChatCompletionRequest, session_id: str) -> ChatCompletion:
            state = get_shared_data()
            session_cache = state.session_cache

            if state.client is None:
                raise HTTPException(
                    status_code=500,
                    detail=f"Client not found. request: {request.model_dump()}",
                )

            if session_id not in session_cache:
                raise HTTPException(status_code=400, detail=f"Session {session_id} not found")

            kwargs = request.model_dump(exclude={"session_id"})

            # PATCH: Move chat_template_kwargs into extra_body before filtering
            if "chat_template_kwargs" in kwargs and kwargs["chat_template_kwargs"]:
                extra_body = kwargs.get("extra_body") or {}
                extra_body["chat_template_kwargs"] = kwargs.pop("chat_template_kwargs")
                kwargs["extra_body"] = extra_body
            elif "chat_template_kwargs" in kwargs:
                del kwargs["chat_template_kwargs"]

            areal_client_allowed_args = [
                "messages",
                "frequency_penalty",
                "max_completion_tokens",
                "max_tokens",
                "metadata",
                "stop",
                "store",
                "temperature",
                "tool_choice",
                "tools",
                "top_p",
                "extra_body",
            ]

            dropped_args = []
            for k, v in kwargs.items():
                if k not in areal_client_allowed_args:
                    dropped_args.append((k, v))

            for k, _ in dropped_args:
                del kwargs[k]

            # Only warn about non-default args that aren't in the silent list
            dropped_non_default_args = [
                (k, v)
                for k, v in dropped_args
                if v != ChatCompletionRequest.model_fields[k].default and k not in SILENTLY_DROPPED_ARGS
            ]
            if len(dropped_non_default_args):
                dropped_args_str = "\n".join([f"  {k}: {v}" for k, v in dropped_non_default_args])
                logger.warning(f"dropped unsupported non-default arguments for areal client:\n{dropped_args_str}")

            if "temperature" not in kwargs:
                kwargs["temperature"] = 1.0
                logger.warning("temperature not set in request, defaulting to 1.0")
            elif kwargs["temperature"] != 1.0:
                logger.warning(
                    f"temperature is set to {kwargs['temperature']} in request, we suggest using 1.0 for RL tasks"
                )

            if "top_p" not in kwargs:
                kwargs["top_p"] = 1.0
                logger.warning("top_p not set in request, defaulting to 1.0")
            elif kwargs["top_p"] != 1.0:
                logger.warning(f"top_p is set to {kwargs['top_p']} in request, we suggest using 1.0 for RL tasks")

            try:
                completion: ChatCompletion = await state.client.chat.completions.create(
                    areal_completion_cache=session_cache[session_id].completions, **kwargs
                )
                return completion
            except ValueError as e:
                # Keep request-validation/type errors as explicit client-facing failures.
                raise HTTPException(status_code=400, detail=str(e))
            except RuntimeError as e:
                # Convert engine/runtime failures (e.g. context overflow after retries)
                # into a structured HTTP error instead of an uncaught ASGI exception.
                detail = str(e)
                status_code = 400 if "maximum context length" in detail.lower() else 502
                raise HTTPException(status_code=status_code, detail=detail)
            except Exception as e:
                logger.exception("Unexpected proxy chat completion failure for session %s", session_id)
                raise HTTPException(
                    status_code=500,
                    detail=f"Unexpected proxy chat completion failure: {e}",
                )

        return app

    proxy_module.build_app = patched_build_app


def patch_max_completion_tokens():
    """Patch areal client to properly respect max_completion_tokens.

    The original logic defaults max_new_tokens to 512 and only uses
    max_completion_tokens as an upper bound via min(). This means if
    max_tokens is not provided, max_completion_tokens > 512 is ignored.

    The fix: use max_completion_tokens directly when provided, and only
    use max_tokens as an upper bound if both are provided.
    """
    import datetime
    import uuid
    from collections.abc import Iterable
    from typing import Any

    from areal.api.cli_args import GenerationHyperparameters
    from areal.api.io_struct import ModelRequest
    from areal.experimental.openai import client as client_module
    from areal.experimental.openai.cache import CompletionCache
    from areal.experimental.openai.client import is_omitted
    from areal.experimental.openai.tool_call_parser import process_tool_calls
    from areal.experimental.openai.types import InteractionWithTokenLogpReward
    from openai._types import NOT_GIVEN, Body, NotGiven
    from openai.types.chat import (
        ChatCompletion,
        ChatCompletionMessage,
        ChatCompletionToolParam,
    )
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
    from openai.types.chat.chat_completion_tool_choice_option_param import (
        ChatCompletionToolChoiceOptionParam,
    )
    from openai.types.completion_usage import CompletionUsage
    from openai.types.shared_params.metadata import Metadata

    async def patched_create(
        self,
        *,
        messages: Iterable[ChatCompletionMessageParam],
        frequency_penalty: float | None | NotGiven = NOT_GIVEN,
        max_completion_tokens: int | None | NotGiven = NOT_GIVEN,
        max_tokens: int | None | NotGiven = NOT_GIVEN,
        metadata: Metadata | None | NotGiven = NOT_GIVEN,
        stop: str | None | list[str] | None | NotGiven = NOT_GIVEN,
        store: bool | None | NotGiven = NOT_GIVEN,
        temperature: float | None | NotGiven = NOT_GIVEN,
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN,
        tools: Iterable[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
        top_p: float | None | NotGiven = NOT_GIVEN,
        extra_body: Body | None = None,
        areal_completion_cache: CompletionCache | None = None,
        **kwargs: Any,
    ) -> ChatCompletion:
        """Override create method to use AReaL engine and cache responses."""
        # Extract and validate supported parameters
        messages_list = list(messages)
        if not messages_list:
            raise ValueError("messages cannot be empty")
        if extra_body is None:
            extra_body = {}
        # Convert messages to prompt format
        tools_val = tools if not is_omitted(tools) else None
        if self.chat_template_type == "hf":
            prompt_token_ids = self.tokenizer.apply_chat_template(
                messages_list,
                tools=tools_val,
                add_generation_prompt=True,
                tokenize=True,
                **extra_body.get("chat_template_kwargs", {}),
            )
        elif self.chat_template_type == "concat":
            # By default, follows Qwen3 chat template.
            start, end = self.messages_delimiter_start, self.messages_delimiter_end
            message_strs = []
            for msg in messages_list:
                message_strs.append(f"{start}{msg['role']}\n{msg['content']}{end}\n")
            message_strs.append(f"{start}assistant\n")
            prompt_token_ids = self.tokenizer.encode("".join(message_strs))
        else:
            raise ValueError(f"Unsupported chat_template_type {self.chat_template_type}")

        temp = 1.0 if is_omitted(temperature) else (temperature or 0.0)

        # PATCHED: Fixed max_completion_tokens logic
        # Priority: max_completion_tokens > max_tokens > default (512)
        max_new_tokens = 512  # default
        if not is_omitted(max_completion_tokens) and max_completion_tokens is not None:
            max_new_tokens = max_completion_tokens
        if not is_omitted(max_tokens) and max_tokens is not None:
            max_tokens_val: int = max_tokens  # type narrowing for ty
            calculated = max_tokens_val - len(prompt_token_ids)
            if calculated <= 0:
                raise RuntimeError("max_tokens must be greater than the number of prompt tokens")
            # If both are provided, use the smaller one
            max_new_tokens = min(max_new_tokens, calculated)

        top_p_val = 1.0 if is_omitted(top_p) else (top_p or 1.0)
        stop_tokens = None if is_omitted(stop) else stop
        if stop_tokens is not None and not isinstance(stop_tokens, list):
            stop_tokens = [stop_tokens]

        if is_omitted(frequency_penalty):
            frequency_penalty = 0.0

        # Create generation config
        gconfig = GenerationHyperparameters(
            n_samples=1,
            temperature=temp,
            max_new_tokens=max_new_tokens,
            top_p=top_p_val,
            stop=stop_tokens,
            greedy=temp == 0,
            frequency_penalty=frequency_penalty,
            stop_token_ids=list(set([self.tokenizer.eos_token_id, self.tokenizer.pad_token_id])),
        )

        model_request = ModelRequest(
            input_ids=prompt_token_ids,
            gconfig=gconfig,
            rid=str(uuid.uuid4()),
            metadata=metadata if not is_omitted(metadata) else {},
            tokenizer=self.tokenizer,
        )

        # Call inference engine
        response = await self.engine.agenerate(model_request)

        # Convert response to OpenAI format
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        current_time = int(datetime.datetime.now().timestamp())

        output_text = self.tokenizer.decode(response.output_tokens)

        # Parse tool calls.
        tool_calls = None
        if tool_choice != "none" and tools_val:
            tool_calls, output_text, response.stop_reason = process_tool_calls(
                output_text,
                tools_val,
                self.tool_call_parser,
                response.stop_reason,
            )

        # Create proper ChatCompletion object with all required fields
        chat_completion = ChatCompletion(
            id=completion_id,
            choices=[
                Choice(
                    finish_reason=response.stop_reason,
                    index=0,
                    logprobs=None,
                    message=ChatCompletionMessage(
                        content=output_text,
                        role="assistant",
                        tool_calls=tool_calls,
                    ),
                )
            ],
            created=current_time,
            model="None",
            object="chat.completion",
            usage=CompletionUsage(
                completion_tokens=len(response.output_tokens),
                prompt_tokens=len(prompt_token_ids),
                total_tokens=len(prompt_token_ids) + len(response.output_tokens),
            ),
        )

        # Cache the completion with token-level data
        cache = areal_completion_cache if areal_completion_cache is not None else self._cache
        from copy import deepcopy

        if completion_id in cache:
            raise ValueError(f"Completion {completion_id} already exists in cache")
        cache[completion_id] = InteractionWithTokenLogpReward(
            completion=deepcopy(chat_completion),
            model_response=response,  # Should not deepcopy response because of tokenizer
            messages=deepcopy(messages_list),  # Store a copy of the input messages
            chat_template_type=self.chat_template_type,
        )

        return chat_completion

    client_module.AsyncCompletionsWithReward.create = patched_create


def apply_all_patches():
    """Apply all areal patches."""
    patch_proxy_chat_template_kwargs()
    patch_max_completion_tokens()
