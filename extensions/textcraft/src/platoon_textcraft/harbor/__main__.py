"""``python -m platoon_textcraft.harbor export --variant synth --out DIR --version 1.0``: see :mod:`.export`."""

from __future__ import annotations

import argparse
import sys

from platoon_textcraft.harbor.export import VARIANTS, export_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m platoon_textcraft.harbor", description="Export TextCraft as a Harbor dataset."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="write the dataset's task directories, registry, index and card")
    export.add_argument("--variant", choices=sorted(VARIANTS), default="synth", help="synth (default) or original")
    export.add_argument("--out", required=True, help="a new directory for the dataset")
    export.add_argument("--version", default="1.0", help="the registry datasets' version (default 1.0)")
    export.add_argument("--splits", help="comma-separated splits (default: all of the environment's)")
    export.add_argument("--limit", type=int, help="export only the first N tasks of each split")
    export.add_argument("--repo-id", default="apurvaga/textcraft-synth", help="the Hub repository the card names")
    export.add_argument("--source-revision", help="the platoon commit to name in the card")
    export.add_argument("--force", action="store_true", help="replace --out if it exists")
    options = parser.parse_args(argv)
    tasks = export_dataset(
        options.out,
        variant=options.variant,
        version=options.version,
        splits=options.splits.split(",") if options.splits else None,
        limit=options.limit,
        repo_id=options.repo_id,
        source_revision=options.source_revision,
        force=options.force,
        log=lambda message: print(message, file=sys.stderr),
    )
    print(f"wrote {len(tasks)} tasks to {options.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
