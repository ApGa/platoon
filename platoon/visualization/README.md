## Trajectory visualization CLI

Use the visualization CLI via the module entrypoint. Examples use `uv run` as preferred in this repo.

```
uv run -m platoon.visualization.cli --help | cat
```

- **Tail live events**: Watch one or more JSONL event files.

```
# Tail a directory of JSONL event logs
uv run -m platoon.visualization.cli tail --dir /path/to/logs

# Tail JSONL event logs recursively from a directory root
uv run -m platoon.visualization.cli tail --rdir /path/to/logs

# Tail specific files
uv run -m platoon.visualization.cli tail /path/to/run1.jsonl /path/to/run2.jsonl
```

- **Replay recorded events**: Load from start and step through with a fixed delay.

```
# Replay a single file (0.25s between events)
uv run -m platoon.visualization.cli replay /path/to/run.jsonl --delay 0.25

# Replay multiple files together (merged by timestamp)
uv run -m platoon.visualization.cli replay /path/to/a.jsonl /path/to/b.jsonl --delay 0.5

# Or point at a directory of JSONL files
uv run -m platoon.visualization.cli replay --dir /path/to/logs --delay 0.5
```

- **Show serialized dumps**: If you have serialized `TrajectoryCollection` dumps as JSON or JSONL, convert and view them.

```
# Show a single JSON dump (one object)
uv run -m platoon.visualization.cli show-dump /path/to/dump.json

# Show a JSONL of multiple dumps (one per line)
uv run -m platoon.visualization.cli show-dump /path/to/dumps.jsonl

# Load all dumps from a directory (non-recursive)
uv run -m platoon.visualization.cli show-dump --dir /path/to/dumps
```

- **Analyze and compare two methods**: Summarize outcomes, optionally LLM-explain differences, and open a comparison UI.

```
# Provide labels and inputs for A and B via files and/or directories
uv run -m platoon.visualization.cli analyze-compare "baseline" "new" \
  --a /path/to/baseline_a.jsonl --a /path/to/baseline_b.jsonl \
  --a-dir /path/to/baseline_dir \
  --b /path/to/new_a.jsonl --b-dir /path/to/new_dir \
  --analysis-model gpt-4o-mini \
  --analysis-cache /tmp/platoon-compare-cache

# Print summary JSON only (no UI)
uv run -m platoon.visualization.cli analyze-compare "A" "B" --a-dir /dirA --b-dir /dirB --no-ui | jq .
```

- **Analyze errors for one method**: Cluster failures and open an error-analysis UI.

```
# Analyze errors from files and/or a directory
uv run -m platoon.visualization.cli analyze-errors "my-method" \
  --paths /path/to/run1.jsonl --paths /path/to/run2.jsonl \
  --dir /path/to/method_dir \
  --model gpt-4o-mini \
  --analysis-cache /tmp/platoon-errors-cache

# JSON-only output (no UI), optionally sample N failures
uv run -m platoon.visualization.cli analyze-errors "my-method" --dir /dir --no-ui --sample 100 --sample-seed 42 | jq .
```

## TUI keybindings

When the Textual UI opens:

- **q**: Quit
- **space**: Play/Pause (for replay modes)
- **right** / **n**: Next step (replay modes)
- **r**: Restart replay from beginning
- **ctrl+f**: Toggle search panel
- **f3** / **shift+f3**: Next/previous search result
- **escape**: Close search panel

Notes:
- Tail mode shows new events appended to `*.jsonl`. Replay mode reads from the start; use `--delay 0` with `replay` to load instantly then step manually.
- For multi-file inputs, records are merged by `ts` if present.
- `show-dump` accepts a single JSON object or JSONL with one dump per line and converts them to event streams under the hood.

