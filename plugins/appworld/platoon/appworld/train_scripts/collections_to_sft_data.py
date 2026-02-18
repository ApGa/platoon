"""Convert trajectory collections to LLaMA-Factory SFT dataset format.

Recursively searches a directory for trajectory_collection.json files,
filters trajectories by reward threshold, and writes a JSONL file where
each line is {"messages": [{"role": ..., "content": ...}, ...]}.

Usage:
    python -m platoon.appworld.train_scripts.collections_to_sft_data \
        --input_dir /path/to/inference_results/appworld/qwen3-14b-instruct-recursive \
        --output_file /mnt/efs/LLaMA-Factory/data/my-sft-dataset.json \
        --recursive \
        --reward_threshold 1.0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from platoon.appworld.agent import (
    AppWorldCodeActPromptBuilder,
    AppWorldRecursiveCodeActPromptBuilder,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def find_trajectory_collections(input_dir: Path) -> list[Path]:
    return sorted(input_dir.rglob("trajectory_collection.json"))


def main(args: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Convert trajectory collections to LLaMA-Factory SFT JSONL format."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Root directory to search for trajectory_collection.json files.",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--reward_threshold",
        type=float,
        default=1.0,
        help="Minimum trajectory reward to include (default: 1.0).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=False,
        help="Use the recursive prompt builder (AppWorldRecursiveCodeActPromptBuilder).",
    )
    parser.add_argument(
        "--prompt_mode",
        choices=["sequence_extension", "no_sequence_extension"],
        default="sequence_extension",
        help="Prompt mode for the builder (default: sequence_extension).",
    )
    parser.add_argument(
        "--no_reasoning",
        action="store_true",
        default=False,
        help="Exclude reasoning (<thought> tags) from assistant messages.",
    )

    parsed = parser.parse_args(args)

    input_dir: Path = parsed.input_dir
    output_file: Path = parsed.output_file
    reward_threshold: float = parsed.reward_threshold
    include_reasoning: bool = not parsed.no_reasoning

    if not input_dir.exists():
        logger.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    if parsed.recursive:
        prompt_builder = AppWorldRecursiveCodeActPromptBuilder(
            prompt_mode=parsed.prompt_mode,
            include_reasoning=include_reasoning,
        )
        logger.info("Using AppWorldRecursiveCodeActPromptBuilder")
    else:
        prompt_builder = AppWorldCodeActPromptBuilder(
            prompt_mode=parsed.prompt_mode,
            include_reasoning=include_reasoning,
        )
        logger.info("Using AppWorldCodeActPromptBuilder")

    collection_paths = find_trajectory_collections(input_dir)
    logger.info(
        "Found %d trajectory_collection.json files in %s",
        len(collection_paths),
        input_dir,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    total_collections = 0
    total_examples = 0
    skipped_collections = 0

    with output_file.open("w", encoding="utf-8") as out_f:
        for collection_path in collection_paths:
            try:
                with collection_path.open("r", encoding="utf-8") as f:
                    traj_collection_dump = json.load(f)
            except Exception as e:
                logger.warning("Failed to load %s: %s", collection_path, e)
                skipped_collections += 1
                continue

            try:
                conversations = prompt_builder.build_messages_from_traj_dump(
                    traj_collection_dump, reward_threshold
                )
            except Exception as e:
                logger.warning("Failed to process %s: %s", collection_path, e)
                skipped_collections += 1
                continue

            total_collections += 1
            for conv in conversations:
                record = {"messages": conv["messages"]}
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_examples += 1

    logger.info(
        "Done. Processed %d collections, skipped %d. Wrote %d examples to %s",
        total_collections,
        skipped_collections,
        total_examples,
        output_file,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
