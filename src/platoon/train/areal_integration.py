from __future__ import annotations

from platoon.utils.llm_client import LLMClient
from areal.experimental.openai import ArealOpenAI
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.data import concat_padded_tensors
from openai.types.chat import ChatCompletion
from platoon.episode.trajectory import Trajectory, TrajectoryStep
from platoon.visualization.event_sinks import TrajectoryEventHandler
from contextvars import ContextVar
import torch
from typing import Any
from platoon.episode.context import current_trajectory

# --- Lightweight RPC server and proxy for engine.agenerate to avoid pickling non-serializable engine ---
import asyncio
import json
import socket
import threading
from typing import Dict

import aiohttp
from aiohttp import web

from areal.api.cli_args import GenerationHyperparameters
from areal.api.io_struct import ModelRequest, ModelResponse

areal_llm_clients: ContextVar[dict[str, ArealLLMClient]] = ContextVar("areal_llm_clients", default={})


# Global registry to keep RPC servers alive in the parent process,
# without storing the non-picklable engine object on the client.
_ENGINE_RPC_SERVERS_BY_ID: Dict[int, "_EngineRPCServer"] = {}


def _find_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


class _EngineRPCServer:
    def __init__(self, engine, host: str = "127.0.0.1", port: int | None = None):
        self._engine = engine
        self._host = host
        self._port = port or _find_free_port()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._runner: web.AppRunner | None = None
        self._started = threading.Event()

    @property
    def base_url(self) -> str:
        return f"http://{self._host}:{self._port}"

    async def _create_app(self) -> web.Application:
        app = web.Application()

        async def health_handler(_request: web.Request) -> web.Response:
            return web.json_response({"ok": True})

        async def agenerate_handler(request: web.Request) -> web.Response:
            payload = await request.json()
            # Reconstruct GenerationHyperparameters
            gcfg_dict = payload["gconfig"]
            gconfig = GenerationHyperparameters(
                n_samples=gcfg_dict.get("n_samples", 1),
                temperature=gcfg_dict.get("temperature", 0.0),
                max_new_tokens=gcfg_dict.get("max_new_tokens", 1),
                top_p=gcfg_dict.get("top_p", 1.0),
                top_k=gcfg_dict.get("top_k", 0),
                stop=gcfg_dict.get("stop"),
                greedy=gcfg_dict.get("greedy", False),
                frequency_penalty=gcfg_dict.get("frequency_penalty", 0.0),
                stop_token_ids=gcfg_dict.get("stop_token_ids", []),
                max_tokens=gcfg_dict.get("max_tokens", 16384),
            )

            # Build ModelRequest. We do not forward tokenizer/processor to avoid non-serializables.
            mreq = ModelRequest(
                input_ids=payload["input_ids"],
                gconfig=gconfig,
                rid=payload.get("rid"),
                metadata=None,
                tokenizer=None,
            )

            resp: ModelResponse = await self._engine.agenerate(mreq)
            # Serialize response to JSON-serializable dict
            response_payload = {
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
                "output_logprobs": resp.output_logprobs,
                "output_versions": resp.output_versions,
                "stop_reason": resp.stop_reason,
                "latency": resp.latency,
                "ttft": getattr(resp, "ttft", resp.latency),
            }
            return web.json_response(response_payload)

        app.router.add_get("/health", health_handler)
        app.router.add_post("/agenerate", agenerate_handler)
        return app

    def _run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        app = self._loop.run_until_complete(self._create_app())
        self._runner = web.AppRunner(app)
        self._loop.run_until_complete(self._runner.setup())
        site = web.TCPSite(self._runner, self._host, self._port)
        self._loop.run_until_complete(site.start())
        self._started.set()
        try:
            self._loop.run_forever()
        finally:
            self._loop.run_until_complete(self._runner.cleanup())
            self._loop.close()

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name=f"EngineRPCServer:{self._port}", daemon=True)
        self._thread.start()
        # Wait until server is ready
        self._started.wait(timeout=5.0)

    def stop(self):
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._thread = None


def _get_or_start_engine_server(engine) -> str:
    key = id(engine)
    if key in _ENGINE_RPC_SERVERS_BY_ID:
        return _ENGINE_RPC_SERVERS_BY_ID[key].base_url
    server = _EngineRPCServer(engine)
    server.start()
    _ENGINE_RPC_SERVERS_BY_ID[key] = server
    return server.base_url


class _RPCProxyEngine:
    def __init__(self, base_url: str, request_timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.request_timeout = request_timeout

    async def agenerate(self, req: ModelRequest) -> ModelResponse:
        # Convert GenerationHyperparameters to dict
        gcfg = req.gconfig
        gcfg_dict = {
            "n_samples": gcfg.n_samples,
            "temperature": gcfg.temperature,
            "max_new_tokens": gcfg.max_new_tokens,
            "top_p": gcfg.top_p,
            "top_k": getattr(gcfg, "top_k", 0),
            "stop": gcfg.stop,
            "greedy": gcfg.greedy,
            "frequency_penalty": getattr(gcfg, "frequency_penalty", 0.0),
            "stop_token_ids": gcfg.stop_token_ids,
            "max_tokens": getattr(gcfg, "max_tokens", len(req.input_ids) + gcfg.max_new_tokens),
        }
        payload = {
            "input_ids": list(req.input_ids),
            "rid": req.rid,
            "gconfig": gcfg_dict,
        }

        timeout = aiohttp.ClientTimeout(total=self.request_timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{self.base_url}/agenerate", json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()

        # Rebuild ModelResponse. We do not set tokenizer/processor here.
        return ModelResponse(
            input_tokens=data.get("input_tokens", req.input_ids),
            input_images=None,
            output_tokens=data["output_tokens"],
            output_logprobs=data["output_logprobs"],
            output_versions=data.get("output_versions", []),
            stop_reason=data.get("stop_reason"),
            latency=data.get("latency", 0.0),
            ttft=data.get("ttft", 0.0),
            tokenizer=getattr(req, "tokenizer", None),
            processor=getattr(req, "processor", None),
        )

class ArealLLMClient(LLMClient): #TODO: Decide if we want to add this to create_llm_client or not.
    def __init__(self, model: str, engine: Any):
        # Store minimal, picklable state only
        self.model = model
        self.tokenizer = load_hf_tokenizer(model)

        # If an engine instance is provided (e.g., RemoteSGLangEngine), start RPC server
        # and use a lightweight proxy for cross-process calls. If a string/URL is provided,
        # assume it points to an existing RPC server.
        if hasattr(engine, "agenerate"):
            base_url = _get_or_start_engine_server(engine)
        elif isinstance(engine, str):
            base_url = engine
        else:
            raise ValueError("engine must be an inference engine with agenerate or a base_url string.")

        # Keep only the base_url (picklable) and a proxy engine used by ArealOpenAI
        self.engine_base_url = base_url
        proxy_engine = _RPCProxyEngine(base_url=base_url)
        self.async_client = ArealOpenAI(engine=proxy_engine, tokenizer=self.tokenizer)

        # Register for event sink lookup
        curr_traj=current_trajectory.get(None)
        if curr_traj is not None:
            areal_llm_clients.get()[curr_traj.id] = self
        else:
            areal_llm_clients.get()[0] = self # TODO: Hack of using 0 for the first trajectory id. We need to fix this.

    def __getstate__(self):
        # Keep only picklable fields
        return {
            "model": self.model,
            "engine_base_url": getattr(self, "engine_base_url", None),
        }

    def __setstate__(self, state):
        self.model = state["model"]
        self.engine_base_url = state["engine_base_url"]
        self.tokenizer = load_hf_tokenizer(self.model)
        proxy_engine = _RPCProxyEngine(base_url=self.engine_base_url)
        self.async_client = ArealOpenAI(engine=proxy_engine, tokenizer=self.tokenizer)
        # Re-register for event sink lookup in the new process
        curr_traj=current_trajectory.get(None)
        if curr_traj is not None:
            areal_llm_clients.get()[curr_traj.id] = self
        else:
            areal_llm_clients.get()[0] = self
        
        curr_traj=current_trajectory.get(None)
        if curr_traj is not None:
            areal_llm_clients.get()[curr_traj.id] = self
        else:
            areal_llm_clients.get()[0] = self # TODO: Hack of using 0 for the first trajectory id. We need to fix this.
        
    async def async_chat_completion(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        auto_add_cache_control: bool = False,
        **kwargs: Any,
    ) -> ChatCompletion:
        # Keep LLMClient's cache control behavior
        if auto_add_cache_control:
            for message in messages:
                if isinstance(message.get("content"), str):
                    message["content"] = [{"type": "text", "text": message["content"]}]
            messages[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}

        # Do NOT pass 'model' to Areal's AsyncCompletions; drop 'stop' to retain tokens for training
        kwargs.pop('stop', None)
        return await self.async_client.chat.completions.create(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
            
        
    def chat_completion(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("ArealLLMClient does not support synchronous chat completion. Please use async_chat_completion instead.")


    def fork(self) -> ArealLLMClient:
        # Pass base_url so the forked client stays picklable
        return ArealLLMClient(model=self.model, engine=self.engine_base_url)


class ArealEventSink(TrajectoryEventHandler):
    
    def on_trajectory_step_added(self, trajectory: Trajectory, step: TrajectoryStep) -> None:
        # Certain steps may be system generated rather than LLM generated.
        # We skip these as we don't need to train on these.
        if not 'completion_id' in step.misc['action_misc']: 
            return
        
        completion_id = step.misc['action_misc']['completion_id']
        client = areal_llm_clients.get()[trajectory.id] if trajectory.id in areal_llm_clients.get() else areal_llm_clients.get()[0] # TODO: Hack of using 0 for the first trajectory id. We need to fix this.
        if client is None:
            raise ValueError(f"ArealLLMClient not found for trajectory {trajectory.id}")
        
        areal_completion = client.async_client.get_completions(completion_id=completion_id).response
        seq = areal_completion.input_tokens + areal_completion.output_tokens
        logprobs = [0.0] * areal_completion.input_len + areal_completion.output_logprobs
        loss_mask = [0] * areal_completion.input_len + [1] * areal_completion.output_len
        versions = [-1] * areal_completion.input_len + areal_completion.output_versions
        
        step.misc['action_misc']['areal_completion_data'] = dict(
            # unsqueeze to add an additional batch dimension
            input_ids=torch.tensor(seq).unsqueeze(0),
            loss_mask=torch.tensor(loss_mask).unsqueeze(0),
            logprobs=torch.tensor(logprobs).unsqueeze(0),
            versions=torch.tensor(versions).unsqueeze(0),
            attention_mask=torch.ones(len(seq), dtype=torch.bool).unsqueeze(0),
        )
        
        
    def on_trajectory_finished(self, trajectory: Trajectory) -> None:
        areal_completion_data_list = []
        count_found_areal_completion_data = 0
        for step in trajectory.steps:
            if 'action_misc' in step.misc and 'areal_completion_data' in step.misc['action_misc']:
                count_found_areal_completion_data += 1
                step_completion_data = step.misc['action_misc']['areal_completion_data']
                step_completion_data['rewards'] = torch.tensor([trajectory.reward])
                # Make rewards 2D [1, seq_len] so it is split/packed per-token with the batch
                seq_len = step_completion_data['attention_mask'].shape[1]
                step_completion_data['token_rewards'] = torch.full(
                    (1, seq_len), float(trajectory.reward), dtype=torch.float32
                )
                areal_completion_data_list.append(step_completion_data)
        print(f"Found {count_found_areal_completion_data} / {len(trajectory.steps)} areal completion data for trajectory {trajectory.id}")
        trajectory.misc['areal_completion_data'] = concat_padded_tensors(areal_completion_data_list)