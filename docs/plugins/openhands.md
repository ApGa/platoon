# OpenHands

Most Platoon plugins train Platoon's own CodeAct agent. This one trains a different agent entirely:
the OpenHands software-engineering agent, with its own prompts, tool set, condenser and runtime.
Platoon supplies the RL machinery around it; OpenHands supplies the agent inside it.

It is a [capability plugin](index.md): it adds an agent harness and nothing else — no dataset, no
config, no training script. Other plugins bring the tasks; [OpenReward](openreward.md) and CodeGrep
are the two in the [catalog](catalog.md) that do.

The plugin lives at <span class="pl-src">plugins/openhands/</span>, and like any Platoon plugin it
is an ordinary Python package — you can build your own OpenHands-based plugin in your own
repository without forking Platoon.

## The division of labor

| OpenHands supplies | Platoon supplies |
| --- | --- |
| The agent: system prompt, tool schemas, tool executors, LLM client | The task, the reward, and the episode that scores it |
| The `Conversation` object and its event log | The trajectory those events become |
| Context condensation | A condenser subclass that keeps reasoning out of retained context |
| Workspace and runtime for tool execution | Budget accounting, delegation, and training-data conversion |

Only two of Platoon's component Protocols are reimplemented: `OpenHandsEnv` in
<span class="pl-src">plugins/openhands/platoon/openhands/env.py</span> and `OpenHandsAgent` in
<span class="pl-src">plugins/openhands/platoon/openhands/agent.py</span>. Nothing else changes —
the same [episode loop](../architecture/execution.md) runs, the same trajectories come out, and the
same AReaL and Tinker converters read them.

## The agent does not call the model

`OpenHandsAgent.act` never talks to an LLM. OpenHands owns its own model loop; Platoon's agent only
reads what that loop has already produced.

```python title="plugins/openhands/platoon/openhands/agent.py"
async def act(self, obs: OpenHandsObservation) -> OpenHandsAction:
    step_actions = get_actions_for_last_obs(obs, require_same_llm_call_id=True)
    while not step_actions and not is_finished(obs):
        await asyncio.sleep(0.2)
        step_actions = get_actions_for_last_obs(obs, require_same_llm_call_id=True)

    action = OpenHandsAction(action_events=step_actions)

    if step_actions:
        action.misc["completion_id"] = step_actions[-1].llm_response_id
```

`OpenHandsEnv.reset` builds a `Conversation`, sends the task goal as its first user message, and
starts `conversation.arun()` as a task on the episode loop. From then on the episode is a *consumer*
of that conversation's event stream: `act` waits for the next batch of actions, `step` waits for the
observations that answer them. One model response — including a parallel tool call that emits
several actions — becomes one trajectory step.

The `completion_id` line is the load-bearing one. A step is trainable only when it names a
completion the inference proxy cached; everything else on the step is diagnostics.

### The environment wraps the runtime, not the task

`OpenHandsEnv.evaluate` returns `0.0, {}`. The base class scores nothing — it exists to run the
conversation and record it. A concrete plugin subclasses it and supplies the reward:

```python title="plugins/codegrep/platoon/codegrep/env.py"
class CodeGreEnv(OpenHandsEnv):
    async def evaluate(self) -> tuple[float, dict]:
        if is_finished(await self.observe()):
            finish_message = get_agent_final_response(self._conversation.state.events)
            reward, predicted_files, true_files = reward_function(finish_message, self.task.misc)
            return reward, {"predicted_files": predicted_files, "true_files": true_files}
        return 0.0, {}
```

`workspace` is whatever the SDK accepts — a local directory or a `BaseWorkspace`. CodeGrep clones a
repository and hands over the path. OpenReward hands over a scratch directory and keeps the real
task environment in a separate container reached over MCP, so there the workspace mostly holds the
agent's own files.

!!! note "Decide how the episode ends"
    OpenReward builds its agent with `include_default_tools=[]` and no `finish` tool, ending the
    episode from an environment callback when the task server reports completion. Sub-agents get
    `finish` added back, because a child has to return a message to its parent. Pick a termination
    path deliberately when you write a new OpenHands plugin.

## Delegation

Set `enable_recursive_subagents` and the OpenHands agent gains a `launch_subagent` tool alongside
its own. A call forks the agent and the environment, runs a full child episode, and returns the
child's finish message as the tool observation — so an OpenHands agent can hand work to another
OpenHands agent, to arbitrary depth. Budget reservation, depth scoping and reward propagation are
the same for every Platoon agent; see [multi-agent workflows](../guides/multi-agent.md).

Two things are specific to OpenHands:

- **The model cannot choose the child's budget.** `LaunchSubagentAction` carries only a `goal`. The
  step budget comes from `subagent_default_max_steps`.
- **Forking strips the launcher.** The `launch_subagent` tool is bound to one episode's loop, so a
  fork removes it and the child installs its own during `reset`. A child working on a subtask also
  gets a prompt suffix warning that siblings may be editing the same files.

OpenHands runs tool executors synchronously, off the episode loop, while `launch_subagent` must run
on it. `LaunchSubagentRuntime` in
<span class="pl-src">plugins/openhands/platoon/openhands/recursive.py</span> bridges the two, with
bounded waits on both sides so a stuck child cannot outlive the rollout's own timeouts.

## Context condensation

Long software-engineering episodes overflow the context window, and OpenHands answers with a
condenser: old events are replaced by a model-written summary. Platoon subclasses it as
`SafeLLMSummarizingCondenser` because that summary is a sampled completion from the policy you are
training. The subclass fits the prompt to the token budget, validates the response, and falls back
to a deterministic summary rather than aborting a rollout.

The consequence worth knowing: every safe, model-generated condensation becomes an extra trainable
step carrying the real completion id, while deterministic fallbacks are excluded. A rollout that
condenses ten times contributes ten summary completions to the batch. If your runs condense often,
condensation quality is a training-data variable, not only a context-management one.

## What you need to run it

Three things beyond a normal Platoon install.

**The OpenHands SDK fork.** The plugin targets a fork of the SDK, pinned by git revision in
<span class="pl-src">plugins/openreward/pyproject.toml</span> for `openhands-sdk`,
`openhands-tools`, `openhands-workspace` and `openhands-agent-server`. That fork is where
programmatic tool calling and declared-resource concurrency come from. Sync from
`plugins/openreward`, and copy its four `[tool.uv.sources.openhands-*]` blocks into any new plugin
that needs the fork — uv only honors sources declared at the root of a resolution.

**A training backend.** `platoon-openhands` depends on plain `platoon`, with no backend extra. The
backend arrives through the consuming plugin's extra.

**A task source.** OpenHands is an agent, not a benchmark. For OpenReward that means a running
environment server:

```bash
docker run --rm \
  -e OPENREWARD_PORT=8080 \
  -p 8080:8080 \
  ghcr.io/apga/openreward-toolathlon-gym:latest
```

Runs go through the consuming plugin's own training script:

=== "AReaL"

    ```bash
    cd plugins/openreward
    uv sync --extra areal
    uv run python -m platoon.openreward.train_scripts.areal.train_areal \
      --config platoon/openreward/configs/areal/toolathlon_openhands_areal.yaml \
      openreward.session_url=http://localhost:8080
    ```

=== "Tinker"

    ```bash
    cd plugins/openreward
    uv sync --extra tinker
    uv run python -m platoon.openreward.train_scripts.tinker.train_tinker \
      --config platoon/openreward/configs/tinker/toolathlon_openhands_tinker.yaml \
      --openreward.session_url http://localhost:8080
    ```

The two override syntaxes differ — bare `key=value` for AReaL, `--dotted.key value` for Tinker. See
[configuration](../reference/configuration.md).

## Agent knobs

These live under the config's `openreward:` section, in `OpenRewardConfig`
(<span class="pl-src">plugins/openreward/platoon/openreward/config_defs.py</span>).

| Key | Default | What it does |
| --- | --- | --- |
| `enable_programmatic_tool_calling` | `false` | Adds the SDK's PTC tool, letting the agent orchestrate tools from Python |
| `programmatic_tool_calling_mode` | `orchestration_only` | `orchestration_only` keeps PTC's Python out of the task environment; `unrestricted` allows it |
| `enable_task_tracker` | `false` | Adds a plan tool. Implied by recursive subagents |
| `enable_recursive_subagents` | `false` | Adds `launch_subagent` and its per-episode runtime |
| `subagent_default_max_steps` | `50` | The child step budget the model cannot override |
| `subagent_environment_access` | `shared` | `shared` or `read_only`; narrows child tool schemas |
| `openhands_system_prompt_suffix` | `null` | Appended to the agent's system message |
| `condenser_disable_thinking` | `false` | Turns off reasoning for condensation completions |
| `condenser_max_completion_tokens` | `26214` | Reasoning plus public summary, one shared budget |

To watch a run, the TUI has a dedicated OpenHands mode that renders tool calls, observation errors
and condensation summaries in their own panels:

```bash
uv run python -m platoon.visualization.cli tail --rdir <output_dir> --mode openhands
```

See [inspecting rollouts](../guides/inspect-rollouts.md).

## See also

- [OpenReward](openreward.md) — the tasks and rewards these agents are usually trained against
- [Multi-agent workflows](../guides/multi-agent.md) — what a delegation call does
- [Components](../architecture/components.md) — the Protocols implemented here
- [Write a plugin](../guides/first-plugin.md) — building your own, in your own repository
