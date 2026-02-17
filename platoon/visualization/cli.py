from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

from .tui import run_replay_from_jsonls, run_viewer_from_jsonls


def main() -> None:
    parser = argparse.ArgumentParser(description="Trajectory visualization (Textual)")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # Tail one or more JSONL files
    tail_p = subparsers.add_parser("tail", help="Tail one or more JSONL event files")
    tail_p.add_argument("paths", nargs="*", help="JSONL files to tail; multiple supported")
    tail_p.add_argument("--dir", dest="dir", default=None, help="Directory of JSONL files to watch (non-recursive)")
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

    # Analyze and compare two sets of runs (dumps or event JSONLs)
    ac_p = subparsers.add_parser(
        "analyze-compare",
        help="Compare two methods' results (dirs or files) and open an analysis UI",
    )
    ac_p.add_argument("method_a", help="Label for method A (e.g., baseline)")
    ac_p.add_argument("method_b", help="Label for method B (e.g., new)")
    ac_p.add_argument("--a", dest="paths_a", action="append", default=[], help="File for method A (repeatable)")
    ac_p.add_argument("--a-dir", dest="dir_a", default=None, help="Directory for method A inputs")
    ac_p.add_argument("--b", dest="paths_b", action="append", default=[], help="File for method B (repeatable)")
    ac_p.add_argument("--b-dir", dest="dir_b", default=None, help="Directory for method B inputs")
    ac_p.add_argument(
        "--analysis-model", dest="analysis_model", default=None, help="Optional LLM model for right-pane explanations"
    )
    ac_p.add_argument(
        "--analyze-both-failed",
        dest="analyze_both_failed",
        action="store_true",
        help="Also run LLM analysis for pairs where both methods failed",
    )
    ac_p.add_argument(
        "--analysis-cache", dest="analysis_cache", default=None, help="Directory to cache/store LLM analyses"
    )
    ac_p.add_argument("--no-ui", dest="no_ui", action="store_true", help="Only print summary JSON without opening UI")

    # Error analysis for a single method
    ae_p = subparsers.add_parser(
        "analyze-errors",
        help="Analyze failures/behaviors for one method (dirs/files), perform hierarchical clustering, open UI",
    )
    ae_p.add_argument("label", help="Label for the method/run")
    ae_p.add_argument("--paths", dest="paths", action="append", default=[], help="File path (repeatable)")
    ae_p.add_argument("--dir", dest="dir", default=None, help="Directory of inputs (json/jsonl)")
    ae_p.add_argument("--model", dest="model", default=None, help="Optional LLM model")
    ae_p.add_argument(
        "--analysis-cache", dest="analysis_cache", default=None, help="Directory to cache/store LLM analyses"
    )
    ae_p.add_argument("--no-cluster", dest="no_cluster", action="store_true", help="Skip clustering step")
    ae_p.add_argument("--sample", dest="sample", type=int, default=None, help="Randomly sample N failures to analyze")
    ae_p.add_argument("--sample-seed", dest="sample_seed", type=int, default=None, help="Random seed for sampling")
    ae_p.add_argument("--passes", dest="passes", type=int, default=2, help="Clustering passes (default: 2)")
    ae_p.add_argument("--no-ui", dest="no_ui_e", action="store_true", help="Print JSON only")
    ae_p.add_argument(
        "--include-successes",
        dest="include_successes",
        action="store_true",
        help="Include successful collections in error analysis",
    )
    ae_p.add_argument(
        "--llm-issues",
        dest="llm_issues",
        action="store_true",
        help="Use LLM to extract issues per collection (slower)",
    )
    ae_p.add_argument(
        "--precompute-analyses",
        dest="precompute_analyses",
        action="store_true",
        help="Precompute and cache per-issue analyses before opening UI",
    )

    args = parser.parse_args()

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
        run_viewer_from_jsonls(paths)

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
        run_replay_from_jsonls(paths, delay=args.delay)

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
        tmp_dir = Path(os.getenv("TMPDIR", "/tmp"))
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
        run_replay_from_jsonls(tmp_event_paths, delay=0.0)

    elif args.cmd == "analyze-compare":
        from platoon.analysis.compare import (
            batch_explain_compare,
            compare_methods,
            llm_cluster_analyses,
            write_clusters_cache,
        )
        from platoon.analysis.compare import (
            discover_input_paths as discover_compare_paths,
        )
        from platoon.visualization.compare_tui import run_compare_ui

        paths_a = discover_compare_paths(getattr(args, "dir_a", None), list(getattr(args, "paths_a", []) or []))
        paths_b = discover_compare_paths(getattr(args, "dir_b", None), list(getattr(args, "paths_b", []) or []))
        if not paths_a or not paths_b:
            parser.error("analyze-compare: provide inputs for both A and B (files or --dir-*)")

        summary = compare_methods(args.method_a, paths_a, args.method_b, paths_b)

        # Prepare LLM explanations per settings (default only for winners)
        explain_items = summary.a_better + summary.b_better
        if getattr(args, "analyze_both_failed", False):
            explain_items += [it for it in summary.ties if "Both failed" in it.rationale]
        analyses = batch_explain_compare(
            explain_items,
            model=getattr(args, "analysis_model", None),
            cache_dir=getattr(args, "analysis_cache", None),
        )
        # Precompute clusters from available analyses and persist to cache
        try:
            clusters = llm_cluster_analyses(
                explain_items,
                analyses,
                model=getattr(args, "analysis_model", None),
            )
            # If clustering is trivial (each item its own cluster), do a heuristic pass
            trivial = len(clusters) == len(explain_items) and all(len(v) == 1 for v in clusters.values())
            if trivial:
                from platoon.analysis.compare import (
                    _heuristic_cluster_from_analyses,  # type: ignore
                )

                ordered = [(it, analyses.get(it.task_id, "")) for it in explain_items]
                clusters = _heuristic_cluster_from_analyses(ordered)
            write_clusters_cache(clusters, cache_dir=getattr(args, "analysis_cache", None))
        except Exception:
            pass

        if getattr(args, "no_ui", False):
            import json as _json

            print(
                _json.dumps(
                    {
                        "counts": {
                            "a_better": len(summary.a_better),
                            "b_better": len(summary.b_better),
                            "ties": len(summary.ties),
                            "unmatched": len(summary.unmatched),
                        },
                        "analyses": analyses,
                    },
                    indent=2,
                )
            )
        else:
            run_compare_ui(
                summary,
                a_label=args.method_a,
                b_label=args.method_b,
                analysis_cache_dir=getattr(args, "analysis_cache", None),
            )

    elif args.cmd == "analyze-errors":
        from platoon.analysis.error_analysis import (
            analyze_errors,
            batch_explain_errors,
            llm_cluster_issue_analyses,
        )
        from platoon.analysis.error_analysis import (
            discover_input_paths as discover_err_paths,
        )
        from platoon.visualization.error_tui import run_error_ui

        paths = discover_err_paths(getattr(args, "dir", None), list(getattr(args, "paths", []) or []))
        if not paths:
            parser.error("analyze-errors: provide files or --dir")
        issues, _ = analyze_errors(
            args.label,
            paths,
            model=getattr(args, "model", None),
            passes=getattr(args, "passes", 2),
            include_successes=getattr(args, "include_successes", False),
            llm_issues=getattr(args, "llm_issues", False),
        )
        # Optionally sample failures to analyze
        try:
            if getattr(args, "sample", None):
                import random

                def _is_failure(it):
                    try:
                        title = (it.title or "").lower()
                        reason = (it.reason or "").strip()
                        return ("fail" in title) or bool(reason)
                    except Exception:
                        return True

                failures = [it for it in issues if _is_failure(it)]
                n = max(0, int(args.sample))
                if n and len(failures) > n:
                    rng = random.Random(getattr(args, "sample_seed", None))
                    failures = rng.sample(failures, n)
                issues = failures
                try:
                    import sys as _sys

                    _sys.stderr.write(
                        f"[analyze-errors] Sampling enabled: selected {len(issues)} failures for analysis.\n"
                    )
                    _sys.stderr.flush()
                except Exception:
                    pass
        except Exception:
            pass
        # Generate analyses and optionally cluster
        analyses = {}
        if getattr(args, "precompute_analyses", False):
            analyses = batch_explain_errors(
                issues,
                model=getattr(args, "model", None),
                cache_dir=getattr(args, "analysis_cache", None),
            )
        if getattr(args, "no_cluster", False):
            from platoon.analysis.error_analysis import ErrorClusters

            clusters = ErrorClusters(clusters={})
        else:
            try:
                clusters = llm_cluster_issue_analyses(
                    issues, analyses if analyses else None, model=getattr(args, "model", None)
                )
            except Exception:
                from platoon.analysis.error_analysis import ErrorClusters

                clusters = ErrorClusters(clusters={})
        if getattr(args, "no_ui_e", False):
            import json as _json

            print(
                _json.dumps(
                    {
                        "issues": [
                            {
                                "task_id": it.task_id,
                                "collection_id": it.collection_id,
                                "title": it.title,
                                "reason": it.reason,
                                "step_refs": it.step_refs,
                                "source": str(it.source_path),
                            }
                            for it in issues
                        ],
                        "clusters": {
                            label: [
                                {
                                    "task_id": it.task_id,
                                    "title": it.title,
                                }
                                for it in items
                            ]
                            for label, items in clusters.clusters.items()
                        },
                    },
                    indent=2,
                )
            )
        else:
            run_error_ui(issues, clusters, analysis_cache_dir=getattr(args, "analysis_cache", None))


if __name__ == "__main__":
    main()
