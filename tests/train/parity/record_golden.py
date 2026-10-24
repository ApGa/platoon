"""Records the golden trace's files again after a change of behavior that is meant, or compares them with this tree's
traces (README.md).  Run it from the repository root, in the environment the tests run in::

    uv run python tests/train/parity/record_golden.py                        # compare every scenario
    uv run python tests/train/parity/record_golden.py --write [NAME ...]     # record every golden, or NAME's, again
    uv run python tests/train/parity/record_golden.py --out DIR [NAME ...]   # write the traces to DIR instead

Each scenario runs twice and is checked, as the test does.  ``--write`` also runs every scenario once more in a fresh
process with another hash seed, and writes nothing for a scenario whose runs differ, whose check fails, or whose
trace in that process differs.  Without ``--write`` or ``--out`` it writes nothing, and exits with 1 when a scenario
does not match its golden files.  ``--out`` writes the traces of two trees for ``diff -r``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from trace_format import GOLDEN_DIR, describe_difference, golden_names
from trace_scenarios import SCENARIOS
from training_trace import TraceResult, trace_scenario


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("scenarios", nargs="*", metavar="NAME", help="scenarios to run (default: every one)")
    where = parser.add_mutually_exclusive_group()
    where.add_argument("--write", action="store_true", help="record the golden files again")
    where.add_argument("--out", type=Path, metavar="DIR", help="write the traces to DIR instead of golden/")
    args = parser.parse_args(argv)
    names = args.scenarios or sorted(SCENARIOS)
    if unknown := sorted(set(names) - set(SCENARIOS)):
        parser.error(f"no such scenario: {', '.join(unknown)} (scenarios: {', '.join(sorted(SCENARIOS))})")

    with tempfile.TemporaryDirectory(prefix="training-trace-") as directory:
        results = [trace_scenario(name, Path(directory, name)) for name in names]
        elsewhere = _in_another_process(names, Path(directory, "elsewhere")) if args.write else {}
    status = 0
    for result in results:
        took = f"{result.seconds:.2f} s for two runs"
        failed = not result.deterministic or result.check_error is not None
        if args.write or args.out is not None:
            if failed:
                print(f"{result.name}: {result.problems(compare=False)}")
                status = 1
            if args.write:
                status |= _record(result, elsewhere, failed, took)
                continue
            result.write(args.out)
            if not result.deterministic:  # the second run's files too, to see what varies
                result.write(args.out / "second", second=True)
            print(f"{result.name}: wrote {', '.join(result.files)} to {args.out} ({took})")
        elif not result.ok:
            print(f"{result.name}: {result.problems()}")
            status = 1
        else:
            print(f"{result.name}: matches its golden files ({took})")
    return status


def _record(result: TraceResult, elsewhere: dict[str, str], failed: bool, took: str) -> int:
    """Write ``result``'s golden files, unless its runs differ, its check fails or another process traced it
    otherwise; 1 when it was not written."""

    other = {file: elsewhere[file] for file in result.files if file in elsewhere}
    if not other:
        print(f"{result.name}: the fresh process did not trace it (above); nothing written")
        return 1
    if other != result.files:
        print(f"{result.name}: a fresh process with another hash seed traced it otherwise; nothing written")
        print(describe_difference(result.files, other, labels=("this process", "another process")))
        return 1
    if failed:
        return 1  # never a golden that varies or tests less
    changed = result.golden != result.files
    result.write(GOLDEN_DIR)
    size = sum(map(len, result.files.values()))
    print(f"{result.name}: {'recorded' if changed else 'unchanged'} ({size} bytes, {took})")
    return 0


def _in_another_process(names: list[str], directory: Path) -> dict[str, str]:
    """The golden files of scenarios ``names`` as this script writes them in a fresh process whose hash seed differs
    from this one's, by file name (none of a scenario that process could not run)."""

    seed = "2" if os.environ.get("PYTHONHASHSEED") == "1" else "1"
    command = [sys.executable, __file__, "--out", str(directory), *names]
    done = subprocess.run(command, env={**os.environ, "PYTHONHASHSEED": seed}, capture_output=True, text=True)
    if done.returncode:
        print(f"the run in another process failed:\n{done.stdout}{done.stderr}")
    return {
        file: (directory / file).read_text(encoding="utf-8")
        for name in names
        for file in golden_names(name)
        if (directory / file).exists()
    }


if __name__ == "__main__":
    sys.exit(main())
