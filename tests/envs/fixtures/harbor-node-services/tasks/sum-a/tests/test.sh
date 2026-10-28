#!/bin/bash
# Runs in main: submits /app/answer.txt to the judge; a judge that cannot grade it makes the reading invalid, never 0.
root="${PLATOON_SANDBOX_ROOT:-}"
python3 - "$JUDGE_URL" "$root" <<'PY'
import json, sys, urllib.request
url, root = sys.argv[1], sys.argv[2]
try:
    code = open(f"{root}/app/answer.txt").read()
except OSError:
    code = ""
reward = {"invalid": 1, "failure_kind": "judge_unavailable"}
try:
    request = urllib.request.Request(url + "/evaluate", data=json.dumps({"code": code}).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=20) as response:
        result = json.load(response)
    if result.get("status") == "done":
        reward = {"reward": float(result["score"]) / 100.0}
except Exception as exc:
    print(f"the judge cannot grade: {exc}")
open(f"{root}/logs/verifier/reward.json", "w").write(json.dumps(reward))
PY
