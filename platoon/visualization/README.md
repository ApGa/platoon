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

Show serialized `TrajectoryCollection` dumps:

```bash
uv run -m platoon.visualization.cli show-dump /path/to/dump.jsonl
```

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
- `f3` / `shift+f3`: next or previous search result
- `escape`: close search

Event files are JSONL streams. Multi-file replay merges records by `ts` when timestamps are present.
