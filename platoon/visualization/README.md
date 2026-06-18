# Trajectory Visualization

The visualization CLI tails, replays, and analyzes Platoon trajectory event logs.

```bash
uv run -m platoon.visualization.cli --help
```

## Common Commands

Tail live event logs:

```bash
uv run -m platoon.visualization.cli tail --rdir /path/to/events
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

Compare two methods:

```bash
uv run -m platoon.visualization.cli analyze-compare baseline candidate \
  --a-dir /path/to/baseline \
  --b-dir /path/to/candidate
```

Analyze failures:

```bash
uv run -m platoon.visualization.cli analyze-errors candidate \
  --dir /path/to/events
```

Add `--no-ui` to analysis commands to print JSON only.

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
