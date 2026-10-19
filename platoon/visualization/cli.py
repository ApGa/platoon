"""The trajectory viewer's command line: ``python -m platoon.visualization.cli {tail,replay,show-dump}``.

``tail`` and ``replay`` read rollout event files (JSONL, one record per event; see
:mod:`platoon.visualization.event_sinks`), which rollouts write under
``<output_dir>/events/``. ``show-dump`` converts serialized ``TrajectoryCollection``
dumps into the same events first.
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from platoon.utils.scratch import scratch_dir

from .tui import run_replay_from_jsonls, run_viewer_from_jsonls

VISUALIZATION_MODES = ("auto", "codeact", "openhands")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Trajectory visualization (Textual)")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # Tail one or more JSONL files
    tail_p = subparsers.add_parser("tail", help="Tail one or more JSONL event files")
    tail_p.add_argument("paths", nargs="*", help="JSONL files to tail; multiple supported")
    tail_p.add_argument("--dir", dest="dir", default=None, help="Directory of JSONL files to watch (non-recursive)")
    tail_p.add_argument(
        "--mode",
        choices=VISUALIZATION_MODES,
        default="auto",
        help="Rendering mode for trajectory steps (default: auto)",
    )
    tail_p.add_argument(
        "--selectable-text",
        action="store_true",
        help="Disable mouse capture so terminal drag-selection works; use keyboard navigation",
    )
    tail_p.add_argument(
        "--rdir",
        dest="rdir",
        default=None,
        help="Directory root to recursively find JSONL files to watch",
    )

    # Replay one or more JSONL files with a fixed delay between events
    replay_p = subparsers.add_parser("replay", help="Replay JSONL events from the start with a delay")
    replay_p.add_argument("paths", nargs="*", help="JSONL files to replay; multiple supported")
    replay_p.add_argument("--dir", dest="dir", default=None, help="Directory of JSONL files to replay (non-recursive)")
    replay_p.add_argument(
        "--mode",
        choices=VISUALIZATION_MODES,
        default="auto",
        help="Rendering mode for trajectory steps (default: auto)",
    )
    replay_p.add_argument(
        "--selectable-text",
        action="store_true",
        help="Disable mouse capture so terminal drag-selection works; use keyboard navigation",
    )
    replay_p.add_argument(
        "--delay", dest="delay", type=float, default=0.5, help="Seconds to wait between events (default: 0.5)"
    )

    # Show a saved dump (dict) by converting to events and running a temporary file
    dump_p = subparsers.add_parser(
        "show-dump", help="Show a serialized TrajectoryCollection dump JSON or JSONL file(s)"
    )
    dump_p.add_argument(
        "paths",
        nargs="*",
        help="Path(s) to JSON (single object) or JSONL (each line a TrajectoryCollection dump) files",
    )
    dump_p.add_argument(
        "--dir",
        dest="dir",
        default=None,
        help="Directory of JSON/JSONL dump files to load (non-recursive)",
    )
    dump_p.add_argument(
        "--mode",
        choices=VISUALIZATION_MODES,
        default="auto",
        help="Rendering mode for trajectory steps (default: auto)",
    )
    dump_p.add_argument(
        "--selectable-text",
        action="store_true",
        help="Disable mouse capture so terminal drag-selection works; use keyboard navigation",
    )

    args = parser.parse_args(argv)

    if args.cmd == "tail":
        paths = []
        if args.dir:
            d = Path(args.dir)
            if d.is_dir():
                paths.extend(sorted(p for p in d.iterdir() if p.suffix.lower() == ".jsonl"))
        if args.rdir:
            d = Path(args.rdir)
            if d.is_dir():
                paths.extend(sorted(p for p in d.rglob("*.jsonl") if p.is_file()))
        if args.paths:
            paths.extend(Path(p) for p in args.paths)
        if not paths:
            parser.error("tail: provide at least one path, --dir, or --rdir with JSONL files")
        run_viewer_from_jsonls(paths, mode=args.mode, selectable_text=args.selectable_text)

    elif args.cmd == "replay":
        paths = []
        if args.dir:
            d = Path(args.dir)
            if d.is_dir():
                paths.extend(sorted(p for p in d.iterdir() if p.suffix.lower() == ".jsonl"))
        if args.paths:
            paths.extend(Path(p) for p in args.paths)
        if not paths:
            parser.error("replay: provide at least one path or --dir with JSONL files")
        run_replay_from_jsonls(paths, delay=args.delay, mode=args.mode, selectable_text=args.selectable_text)

    elif args.cmd == "show-dump":
        # Collect input files from --dir and positional paths
        input_paths: list[Path] = []
        if getattr(args, "dir", None):
            d = Path(args.dir)
            if d.is_dir():
                input_paths.extend(sorted(p for p in d.iterdir() if p.suffix.lower() in {".json", ".jsonl"}))
        if getattr(args, "paths", None):
            input_paths.extend(Path(p) for p in args.paths)
        if not input_paths:
            parser.error("show-dump: provide at least one path or --dir with JSON/JSONL dump files")

        # Prepare temporary event JSONL files produced from the dumps
        tmp_dir = scratch_dir()
        tmp_event_paths: list[Path] = []
        from .event_sinks import (
            trajectory_collection_dump_to_events,
            write_events_from_dump_to_jsonl,
        )

        for src in input_paths:
            suffix = src.suffix.lower()
            tmp_path = tmp_dir / f"trajectory_dump_events_{uuid.uuid4().hex}.jsonl"
            if suffix == ".json":
                with src.open("r", encoding="utf-8") as f:
                    dump_obj = json.load(f)
                write_events_from_dump_to_jsonl(dump_obj, tmp_path)
                tmp_event_paths.append(tmp_path)
            elif suffix == ".jsonl":
                # Each line is a serialized TrajectoryCollection dump; expand all to events
                with src.open("r", encoding="utf-8") as f_in, tmp_path.open("w", encoding="utf-8") as f_out:
                    for line in f_in:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            dump_obj = json.loads(line)
                        except Exception:
                            continue
                        for rec in trajectory_collection_dump_to_events(dump_obj):
                            f_out.write(json.dumps(rec) + "\n")
                tmp_event_paths.append(tmp_path)
            else:
                # Skip unsupported extensions silently
                continue

        if not tmp_event_paths:
            parser.error("show-dump: no valid JSON/JSONL dump files were found")

        # Fast load: replay instantly when no delay specified
        run_replay_from_jsonls(tmp_event_paths, delay=0.0, mode=args.mode, selectable_text=args.selectable_text)


if __name__ == "__main__":
    main()
