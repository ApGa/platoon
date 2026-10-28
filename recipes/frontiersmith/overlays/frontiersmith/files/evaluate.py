#!/usr/bin/env python3
"""FrontierSmith's verifier on platoon: submits /app/solution.cpp to the node's judge pool through the rollout's scope
($JUDGE_URL, http://judge:8081, which the scope names the task's problem for) and writes /logs/verifier/reward.json:

- {"reward": score / 100} for a grade, 0 for code that is missing, empty or does not compile;
- {"invalid": 1, "failure_kind": ...} when the judge cannot grade it (it answered an error, gave no result, or could
  not be reached), which platoon records as invalid, never as a score (the adapter's verifier wrote 0).

Standard library only.  $JUDGE_WAIT_S bounds the wait for the grade (default one hour): the task's own verifier timeout,
which platoon scales, ends a longer one."""

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

JUDGE_URL = os.environ.get("JUDGE_URL", "http://judge:8081").rstrip("/")
WAIT_S = float(os.environ.get("JUDGE_WAIT_S", "3600"))
SOLUTION = Path("/app/solution.cpp")
REWARD = Path("/logs/verifier/reward.json")


def write(payload: dict) -> None:
    REWARD.parent.mkdir(parents=True, exist_ok=True)
    REWARD.write_text(json.dumps(payload))


def invalid(kind: str, detail: str) -> None:
    print(f"the judge cannot grade the submission ({kind}): {detail}")
    write({"invalid": 1, "failure_kind": kind})


def main() -> None:
    print(f"Frontier-CS problem {os.environ.get('PROBLEM_ID', '?')}, judged through {JUDGE_URL}")
    code = SOLUTION.read_text(errors="replace") if SOLUTION.is_file() else ""
    if not code.strip():
        print("no solution: /app/solution.cpp is missing or empty")
        write({"reward": 0.0})
        return
    request = urllib.request.Request(
        f"{JUDGE_URL}/evaluate",
        data=json.dumps({"code": code, "timeout": WAIT_S}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=WAIT_S + 60) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        invalid("judge_error", f"HTTP {exc.code}: {exc.read()[:500]!r}")
        return
    except (OSError, ValueError) as exc:
        invalid("judge_unreachable", f"{type(exc).__name__}: {exc}")
        return
    score = result.get("score") if isinstance(result, dict) else None
    if result.get("status") != "done" or isinstance(score, bool) or not isinstance(score, (int, float)):
        invalid("judge_malformed", repr(result)[:500])
        return
    if not 0 <= score <= 100:
        invalid("judge_malformed", f"score {score!r}")
        return
    cases = f"{result.get('perfect_cases', '?')} of {result.get('cases', '?')} cases perfect"
    print(f"score {score}/100 ({result.get('message') or cases})")
    write({"reward": float(score) / 100.0})


if __name__ == "__main__":
    main()
