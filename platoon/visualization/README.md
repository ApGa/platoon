# Trajectory Visualization

The visualization CLI tails and replays Platoon rollout event files. Every finished rollout writes one,
`<rollout output_dir>/events/events_<task id>_<collection id>.jsonl`, from its trajectory collection
(`platoon.visualization.event_sinks.emit_rollout_events`): in training under
`<run directory>/rollouts/<scope>/<policy version>/`, where the run directory is
`<log_path>/<stats.experiment_name>/<stats.trial_name>`, in evaluation next to each rollout's
`trajectory_collection.json`, and from `platoon rollout` under `rollouts/` in its own run directory
(`--output-dir`, default `runs/adhoc/<time>-<task>`).

```bash
uv run -m platoon.visualization.cli --help
```

## Common Commands

Tail a run's event files (those present when the viewer starts; each file is written whole when its rollout finishes):

```bash
uv run -m platoon.visualization.cli tail --rdir /path/to/run_directory/rollouts
```

Replay recorded logs:

```bash
uv run -m platoon.visualization.cli replay --dir /path/to/events --delay 0.25
```

Replay OpenHands SDK trajectories with the OpenHands-focused renderer:

```bash
uv run -m platoon.visualization.cli replay --mode openhands /path/to/events.jsonl --delay 0
```

Launch with terminal text selection enabled:

```bash
uv run -m platoon.visualization.cli replay --mode openhands --selectable-text /path/to/events.jsonl --delay 0
```

For cleaner selection, press `d` to switch to details-only view before selecting text. Terminal selection is row-based, so details-only view avoids selecting tree text from the same rows.

`--mode auto` is the default. It keeps the CodeAct-oriented view for `code` / `thought` / `output` steps and switches to OpenHands rendering when a step contains `action_events` or `observation_events`.

Show serialized `TrajectoryCollection` dumps:

```bash
uv run -m platoon.visualization.cli show-dump /path/to/dump.jsonl
```

The `tail`, `replay`, and `show-dump` commands accept `--mode auto|codeact|openhands` and `--selectable-text`.

## OpenHands Rendering

OpenHands trajectories are rendered by Platoon step index. Each step groups OpenHands action events with their matching observation events, paired by `tool_call_id` or `action_id`.

The OpenHands mode:

- Summarizes step 0 setup without dumping the full system prompt.
- Resolves `call_tool` wrappers to catalog tool names such as `excel.create_workbook`.
- Collapses repeated tool calls in one model turn, for example `emails.send_email x3`.
- Surfaces observation errors in the tree label.
- Shows final `claim_done` evaluation payloads from `misc.reward_misc.openreward/final_payload`.

## TUI Keys

- `q`: quit
- `space`: play or pause replay
- `right` / `n`: next replay step
- `r`: restart replay
- `ctrl+f`: toggle search
- `m`: toggle mouse capture; when disabled, terminal drag-selection works
- `d`: toggle details-only view for cleaner terminal selection
- `f3` / `shift+f3`: next or previous search result
- `escape`: close search

Event files are JSONL streams. Multi-file replay merges records by `ts` when timestamps are present.
