# Inspect rollouts

A reward came out wrong and you want to know why. Platoon ships a terminal viewer that replays a
rollout's event log turn by turn, follows a run while it is still going, and lays out a multi-agent
trajectory as a tree.

Everything here reads JSONL files on disk. No GPU, no cluster, no training job — only the step that
*produced* the log needed those.

## Where event logs come from

A rollout program writes trajectory events by registering a `JsonlFileSink` on its
`TrajectoryCollection`. Every plugin's `run_rollout` does this:

```python title="plugins/number-search/platoon/number_search/rollout.py"
traj_collection = TrajectoryCollection()
current_trajectory_collection.set(traj_collection)

events_path = os.path.join(config.output_dir, "events", f"events_{task.id}_{traj_collection.id}.jsonl")

traj_collection.register_event_handlers(
    JsonlFileSink(events_path, collection_id=traj_collection.id, process_id=os.getpid())
)
```

The sink appends one JSON object per line as the episode runs: `trajectory_created`,
`trajectory_task_set`, one `trajectory_step_added` per step, and `trajectory_finished`. That file is
the interchange format — replay, tail and comparison all read it.

Where it lands depends on who set `rollout_config.output_dir`:

| Producer | Directory |
| --- | --- |
| A plain rollout script | `{output_dir}/events/` — default `rollout_results` |
| Tinker training | `{log_path}/rollouts/{train\|eval}/{checkpoint_version}/events/` |
| AReaL training | `{output_dir}/{train_rollout\|eval_rollout}/{version}/events/` |
| Inference benchmark | `{inference.output_dir}/rollouts/{task_id}/rollout_{i}/` as a dump |

Both trainers append the model version, so you get one directory per training step.

There is no console script; invoke the module.

```bash
uv run python -m platoon.visualization.cli --help
```

## Replay a finished run

Point `replay` at a directory of event logs:

```bash
uv run python -m platoon.visualization.cli replay --dir ./rollout_results/events --delay 0.25
```

Autoplay advances one event every `--delay` seconds, and **replay starts paused** — press ++space++
to begin. When you are investigating rather than demoing, load the whole file at once:

```bash
uv run python -m platoon.visualization.cli replay --dir ./rollout_results/events --delay 0
```

`rollout_results` is the default `output_dir`; the [quickstart](../get-started/quickstart.md)
sets its own and prints where the events landed.

## Follow a live run

`tail` is the live variant, and it takes `--rdir` to walk a directory tree — which is what you want
against a training job writing a new subdirectory per model version:

```bash
uv run python -m platoon.visualization.cli tail --rdir /path/to/run/rollouts
```

Each file gets its own poll loop that waits for the file to appear, so you can start tailing before
the job starts writing. Half-written lines are skipped.

As new steps arrive the viewer collapses everything except the active trajectory, scrolls the step
into view and flashes it. That is ideal for watching one rollout and unreadable when twenty workers
append at once — so tail a single file when you actually want to follow something:

```bash
uv run python -m platoon.visualization.cli tail /path/to/run/rollouts/train/12/events/events_task-7_a1b2c3.jsonl
```

Inference benchmarks write a serialized collection instead of an event stream. `show-dump` converts
it and opens the same viewer:

```bash
uv run python -m platoon.visualization.cli show-dump /path/to/rollouts/task_x/rollout_0/trajectory_collection.json
```

## What the viewer shows

The left pane is a tree — collection, then trajectory, then task and steps. The right pane renders
the payload of whatever node has focus.

A collection node names the environment, the task, a count of trajectories and the head of the
collection id:

```
collection: textcraft · task:train#442 · trajs:solver=2,verifier=1 · id:068cf326
```

A trajectory node carries its cumulative reward, colored **only once the trajectory has finished**:
red at `reward <= 0`, green at `>= 1`, yellow in between. An uncolored node that already shows a
reward is still running, or its log stopped mid-episode. Check that before anything else.

Step labels are summaries, not payloads:

```
step 0: thought: bracket the range first; code: guess(50)
step 2: tools: emails.send_email x3 -> Traceback: RuntimeError
```

Press ++ctrl+f++ and type to search. It is a case-insensitive substring match over both node labels
and serialized payloads, so a traceback fragment, a tool name or a snippet of the model's reasoning
all find the step. Results are bucketed by node kind; click one to focus it.

### Keys worth knowing

| Key | Action |
| --- | --- |
| ++space++ | Play/pause replay |
| ++right++ / ++n++ | Advance one event and pause |
| ++r++ | Restart from the first record |
| ++ctrl+f++ / ++f3++ / ++escape++ | Search, next hit, close |
| ++d++ | Details-only view |
| ++m++ | Release mouse capture so the terminal can drag-select |
| ++q++ | Quit |

### Rendering modes

`--mode auto` is the default and is usually right: CodeAct-style steps render as
`thought` / `code` / `output` / `error`, and a payload carrying `action_events` switches to
OpenHands rendering, which pairs each action with its observations. Pass `--mode codeact` when you
want the raw payload instead of the OpenHands summary.

!!! tip "Strings render as Markdown"
    The details pane treats string values as Markdown, so log output containing `#`, `*` or
    backticks is reformatted. Press ++d++ then ++m++ to select raw text in your terminal, or launch
    with `--selectable-text`.

## Navigate a multi-agent trajectory tree

When an agent delegates, one task produces many trajectories. The tree view is what makes them
legible.

Nesting follows two rules. A trajectory that judges another — its `misc` carries
`subagent_reward_verifies_trajectory_id` — is re-parented **under the trajectory it judges** and
drawn dim with a `verifier:` prefix. Everything else hangs under its `parent_info.id`, with a
`fork from <parent> @ step <n>` child recording where the parent stood when it delegated.

```
collection: textcraft · task:train#442 · trajs:solver=2,verifier=1 · id:068cf326
└── traj:a1b2… · subtree:solver=2,verifier=1 · reward:0.000
    ├── task:442 · Craft a stone pickaxe from raw materials
    ├── steps
    │   ├── step 0: thought: delegate the wood subtask; code: subagent(...)
    │   └── step 1: code: craft("stone pickaxe"); error
    └── traj:c3d4… · subtree:solver=1,verifier=1 · reward:1.000
        ├── fork from a1b2… @ step 0
        ├── steps
        └── verifier:e5f6… · subtree:solver=0,verifier=1 · reward:1.000
```

Read it top down: the root scored `0.000`, the subagent it launched scored `1.000`, and a verifier
signed off on the subagent. So the failure is in the root's step 1, after delegation returned — not
in the subagent. Without the tree that is a diff of two flat logs.

Events do not always arrive parent-first when subagents run concurrently. The tree repairs itself,
moving a node once its real parent shows up. See [multi-agent workflows](multi-agent.md) for the
delegation machinery behind these trees.

## Compare two runs

Once you know why one rollout failed, the next question is whether a change helped across the eval
set. `analyze-compare` pairs collections by task id and buckets them:

```bash
uv run python -m platoon.visualization.cli analyze-compare baseline recursive \
  --a-dir /runs/baseline/events \
  --b-dir /runs/recursive/events
```

Inputs can be event logs or dumps in any mix. The table has columns `Task | Winner | <a> | <b> |
Cluster`, sectioned into A better, B better, both succeeded, both failed and unmatched. Press ++o++
on a row to open both attempts at that task in one replay viewer, ++g++ to regroup by cluster, ++c++
to copy the details, ++q++ to quit.

Add a model for LLM explanations in the right pane, cached on disk:

```bash
uv run python -m platoon.visualization.cli analyze-compare baseline recursive \
  --a-dir /runs/baseline/events --b-dir /runs/recursive/events \
  --analysis-model openai/gpt-4o-mini --analysis-cache /runs/compare-cache
```

`--no-ui` prints the counts and analyses as JSON instead, which is the CI-friendly form.

!!! note "Success is `reward == 1.0` exactly"
    `is_success_for_collection` uses a strict test, so a partial-credit environment reads as all
    failures. The verdict and task id come from the first trajectory in the collection — for a
    multi-agent run, the root.

To group the failures in one run rather than compare two, `analyze-errors LABEL --dir …` extracts an
issue per failing trajectory and clusters them. Both analysis commands cache LLM output under
`$XDG_CACHE_HOME`, overridable with `--analysis-cache`.

## Next

- [Evaluate a model](evaluate.md) — headline accuracy numbers, computed outside the TUI.
- [CLI reference](../reference/cli.md) — every subcommand and flag.
