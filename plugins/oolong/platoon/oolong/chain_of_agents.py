"""Chain-of-Agents baseline for Oolong inference."""

from __future__ import annotations

import json
from typing import Any

from platoon.config_defs import InferenceParams
from platoon.envs.codeact import CodeActAction, CodeActObservation
from platoon.episode.context import current_env, episode_step_timeout


def _response_text(response: Any) -> str:
    return response.choices[0].message.content or ""


def _chain_role_from_goal(goal: str | None) -> str:
    goal = goal or ""
    if "final manager agent" in goal:
        return "manager"
    if "You are worker" in goal:
        return "worker"
    return "subagent"


class OolongChainOfAgentsSubAgent:
    """Single-call worker/manager agent used by the CoA manager."""

    def __init__(
        self,
        llm_client,
        inference_params: InferenceParams | None = None,
    ):
        self.llm_client = llm_client
        self.inference_params = inference_params or InferenceParams()

    def _request_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "max_completion_tokens": self.inference_params.max_completion_tokens,
        }
        request_timeout = episode_step_timeout.get()
        if request_timeout is not None:
            kwargs["timeout"] = request_timeout
        if self.inference_params.temperature is not None:
            kwargs["temperature"] = self.inference_params.temperature
        if self.inference_params.top_p is not None:
            kwargs["top_p"] = self.inference_params.top_p
        return kwargs

    def _parse_result(self, response_text: str) -> tuple[str, str]:
        try:
            payload = json.loads(response_text)
        except json.JSONDecodeError:
            return "Returning the model response directly because it was not valid JSON.", response_text.strip()
        thought = str(payload.get("thought", "")).strip()
        result = str(payload.get("result", "")).strip()
        return thought or "Processed the assigned Chain-of-Agents role.", result

    async def act(self, obs: CodeActObservation) -> CodeActAction:
        env = current_env.get()
        context = getattr(env, "context", "")
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an agent in a Chain-of-Agents long-context QA system. "
                    "Follow the assigned role exactly. Return JSON with keys `thought` and `result`. "
                    "`thought` should be a short 1-3 sentence rationale. `result` should contain only the "
                    "communication unit requested by the task."
                    "You should return a communication unit that both propagates relevant "
                    "information from the previous communication unit and also adds any new relevant information from the current provided context chunk."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"# Assigned task\n{obs.task.goal}\n\n"
                    f"# Context chunk\n{context if context else 'No additional context chunk was provided.'}\n\n"
                    "Return only valid JSON: {\"thought\": \"...\", \"result\": \"...\"}"
                ),
            },
        ]
        response = await self.llm_client.async_chat_completion(messages, **self._request_kwargs())
        thought, result = self._parse_result(_response_text(response))
        code = f"finish({json.dumps(result)})"
        return CodeActAction(
            action=f"<thought>{thought}</thought>\n<python>{code}</python>",
            parsed_code=code,
            parsed_thought=thought,
            misc={
                "model": getattr(self.llm_client, "model", None),
                "completion_id": getattr(response, "id", "chain-of-agents-subagent"),
                "usage": response.usage.to_dict() if getattr(response, "usage", None) is not None else {},
                "chain_of_agents": {"role": _chain_role_from_goal(obs.task.goal)},
            },
        )

    async def reset(self) -> None:
        pass

    async def close(self) -> None:
        await self.llm_client.aclose()


class OolongChainOfAgentsAgent:
    """Root CoA manager that sequentially launches worker agents."""

    def __init__(
        self,
        llm_client,
        inference_params: InferenceParams | None = None,
        chunk_chars: int = 32000 * 4,
        max_communication_chars: int = 6000 * 4,
    ):
        self.llm_client = llm_client
        self.inference_params = inference_params or InferenceParams()
        self.chunk_chars = chunk_chars
        self.max_communication_chars = max_communication_chars

    async def act(self, obs: CodeActObservation) -> CodeActAction:
        question = obs.task.goal or ""
        code = f"""
question = {json.dumps(question)}
chunk_chars = {self.chunk_chars}
max_communication_chars = {self.max_communication_chars}
chunks = [context[i:i + chunk_chars] for i in range(0, len(context), chunk_chars)] or [""]
communication = ""

for chunk_index, chunk in enumerate(chunks):
    worker_goal = "\\n".join([
        f"You are worker {{chunk_index + 1}} of {{len(chunks)}} in a Chain-of-Agents long-context QA system.",
        f"Question: {{question}}",
        "",
        "Previous communication unit:",
        communication or "No prior information.",
        "",
        "Read only your provided context chunk. Return an updated communication unit for the next worker. "
        "Keep it concise but complete. Include only information useful for answering the question, and "
        "explicitly say if this chunk adds no useful evidence. Do not submit the final answer unless the "
        "chunk provides decisive evidence; preserve exact names, numbers, dates, and intermediate counts.",
    ])
    communication = await launch_subagent(goal=worker_goal, context=chunk)
    if len(communication) > max_communication_chars:
        communication = communication[-max_communication_chars:]

manager_goal = "\\n".join([
    "You are the final manager agent in a Chain-of-Agents long-context QA system.",
    f"Question: {{question}}",
    "",
    "Accumulated communication unit from all workers:",
    communication or "No useful evidence was found.",
    "",
    "Use the accumulated worker communication to produce the final answer. Return only the final answer "
    "to submit to the evaluator. If the task expects a boxed answer, return the boxed answer.",
])
answer = await launch_subagent(goal=manager_goal, context="")
finish(answer)
""".strip()
        thought = (
            "I will run the Chain-of-Agents baseline by launching workers sequentially over context chunks, "
            "then launching a final manager to synthesize the accumulated communication."
        )
        return CodeActAction(
            action=f"<thought>{thought}</thought>\n<python>{code}</python>",
            parsed_code=code,
            parsed_thought=thought,
            misc={
                "model": None,
                "completion_id": "chain-of-agents-root-manager",
                "usage": {},
                "chain_of_agents": {
                    "role": "root_manager",
                    "chunk_chars": self.chunk_chars,
                    "max_communication_chars": self.max_communication_chars,
                },
            },
        )

    async def fork(self, task) -> OolongChainOfAgentsSubAgent:
        return OolongChainOfAgentsSubAgent(
            llm_client=self.llm_client.fork(),
            inference_params=self.inference_params,
        )

    async def reset(self) -> None:
        pass

    async def close(self) -> None:
        await self.llm_client.aclose()
