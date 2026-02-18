from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
from typing import Any


def _parse_builder_spec(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise ValueError(
            f"Invalid builder spec '{spec}'. Expected format: module.path:ClassName"
        )
    module_name, class_name = spec.split(":", 1)
    if not module_name or not class_name:
        raise ValueError(
            f"Invalid builder spec '{spec}'. Expected format: module.path:ClassName"
        )
    return module_name, class_name


def _load_builder(builder_spec: str, prompt_mode: str, include_reasoning: bool) -> Any:
    module_name, class_name = _parse_builder_spec(builder_spec)
    module = importlib.import_module(module_name)
    builder_cls = getattr(module, class_name, None)
    if builder_cls is None:
        raise AttributeError(
            f"Builder class '{class_name}' not found in module '{module_name}'."
        )
    return builder_cls(prompt_mode=prompt_mode, include_reasoning=include_reasoning)


def _iter_dump_objects_from_file(path: Path) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    suffix = path.suffix.lower()

    if suffix == ".json":
        try:
            with path.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict) and "trajectories" in obj:
                objects.append(obj)
        except Exception:
            return objects
    elif suffix == ".jsonl":
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(obj, dict) and "trajectories" in obj:
                        objects.append(obj)
        except Exception:
            return objects

    return objects


def _iter_collection_dumps(input_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    files = sorted(
        [
            p
            for p in input_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in {".json", ".jsonl"}
        ]
    )
    dumps: list[tuple[Path, dict[str, Any]]] = []
    for file_path in files:
        for obj in _iter_dump_objects_from_file(file_path):
            dumps.append((file_path, obj))
    return dumps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively find trajectory collections and convert them to "
            "LLaMA-Factory style SFT JSONL."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory to recursively scan for trajectory collection dumps (.json/.jsonl).",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Output JSONL path (each line contains {'messages': [...]}).",
    )
    parser.add_argument(
        "--builder",
        default="platoon.appworld.agent:AppWorldRecursiveCodeActPromptBuilder",
        help=(
            "Prompt builder in module.path:ClassName format. "
            "Class must implement build_messages_from_traj_dump."
        ),
    )
    parser.add_argument(
        "--reward-threshold",
        type=float,
        default=1.0,
        help="Minimum trajectory reward passed to build_messages_from_traj_dump.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=["sequence_extension", "no_sequence_extension"],
        default="sequence_extension",
        help="Prompt mode used to initialize the prompt builder.",
    )
    parser.add_argument(
        "--include-reasoning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to include reasoning tags when building messages.",
    )
    parser.add_argument(
        "--appworld-root",
        default=None,
        help=(
            "Optional override for APPWORLD_ROOT. Required by AppWorld prompt builders "
            "to load task specs."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist or is not a directory: {input_dir}")

    if args.appworld_root:
        os.environ["APPWORLD_ROOT"] = str(Path(args.appworld_root).expanduser().resolve())

    if args.builder.startswith("platoon.appworld.agent:") and "APPWORLD_ROOT" not in os.environ:
        raise EnvironmentError(
            "APPWORLD_ROOT is not set. Set APPWORLD_ROOT or pass --appworld-root "
            "when using AppWorld prompt builders."
        )

    builder = _load_builder(
        builder_spec=args.builder,
        prompt_mode=args.prompt_mode,
        include_reasoning=args.include_reasoning,
    )

    dumps = _iter_collection_dumps(input_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    converted_examples = 0
    with output_path.open("w", encoding="utf-8") as out_f:
        for source_path, traj_collection_dump in dumps:
            try:
                conversations = builder.build_messages_from_traj_dump(
                    traj_collection_dump=traj_collection_dump,
                    reward_threshold=args.reward_threshold,
                )
            except Exception as e:
                print(f"[warn] failed to process {source_path}: {e}")
                continue

            for conversation in conversations:
                messages = conversation.get("messages", [])
                if not isinstance(messages, list) or len(messages) == 0:
                    continue
                out_f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
                converted_examples += 1

    print(f"Scanned trajectory collections: {len(dumps)}")
    print(f"Wrote SFT examples: {converted_examples}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
