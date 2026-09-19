"""Token-weighted sequential paths through recorded fork/join execution.

Timestamps establish ordering only: GPU queuing, tool latency, and verifier
latency never become reward costs. Direct child calls are weighted by their
own recursively computed critical paths. Overlapping operations are parallel;
successive operations add, including successive children in the same PTC call.
"""

import math
import time
from bisect import bisect_right

EXECUTION_SPAN_KEY = "execution_span"
MODEL_CALL_SPANS_KEY = "model_call_spans"


def record_model_span(completion_id: str, started_at: float) -> None:
    # Keep attribution utilities importable without loading agent dependencies.
    from platoon.episode.context import current_trajectory

    trajectory = current_trajectory.get(None)
    if trajectory is not None:
        trajectory.misc.setdefault(MODEL_CALL_SPANS_KEY, {})[completion_id] = [started_at, time.perf_counter()]


def _span(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    start, end = value
    if not all(isinstance(x, (float, int)) and math.isfinite(x) for x in value):
        return None
    return (start, end) if end > start else None


def critical_path_tokens(
    trajectory: dict,
    own_counts: dict[str, tuple[int, int]],
    children: list[tuple[dict, tuple[int, int]]],
    *,
    input_weight: float,
    output_weight: float,
) -> tuple[int, int, int]:
    """Return input/output tokens on one path and missing-timing count.

    Old saved trajectories have no execution trace. Missing operations are
    conservatively charged sequentially, never silently declared parallel.
    Input and output counts come from the SAME maximizing path.
    """
    spans = trajectory.get("misc", {}).get(MODEL_CALL_SPANS_KEY, {})
    operations = [(spans.get(key), counts) for key, counts in own_counts.items()]
    operations.extend((child.get("misc", {}).get(EXECUTION_SPAN_KEY), counts) for child, counts in children)
    timed = []
    untimed_input = untimed_output = missing = 0
    for interval, (input_tokens, output_tokens) in operations:
        span = _span(interval)
        if span is None:
            untimed_input += input_tokens
            untimed_output += output_tokens
            missing += 1
        else:
            timed.append((*span, input_tokens, output_tokens))
    timed.sort(key=lambda item: (item[1], item[0]))
    ends = [item[1] for item in timed]
    best = [(0, 0)]

    def cost(counts):
        return input_weight * counts[0] + output_weight * counts[1]

    for index, (start, _, inputs, outputs) in enumerate(timed):
        predecessor = bisect_right(ends, start, hi=index)
        candidate = (best[predecessor][0] + inputs, best[predecessor][1] + outputs)
        previous = best[-1]
        best.append(candidate if cost(candidate) > cost(previous) else previous)
    return best[-1][0] + untimed_input, best[-1][1] + untimed_output, missing
