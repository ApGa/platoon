#!/bin/bash
# Grade a TextCraft episode: replay the agent's action log (/app/textcraft/actions.jsonl) from the task's starting
# inventory with the game's rules, using this directory's own copies of the game, the recipes and the task.  The
# reward is 1 only when the log is an episode the rules allow that ends in `textcraft finish` with every target
# crafted (the verdict goes to textcraft-verdict.json).  It runs in the agent's container (a shared verifier) or in a
# separate verifier that received the log as an artifact.
# PLATOON_SANDBOX_ROOT: platoon's process provider runs this on the host, with the sandbox's files below that
# directory; sandboxes leave it unset.
set -u
root="${PLATOON_SANDBOX_ROOT:-}"
tests="$root/tests"
out="$root/logs/verifier"
mkdir -p "$out"
# A fresh verdict: nothing the agent may have left in the verifier's directory counts.
rm -rf -- "$out/reward.json" "$out/reward.txt" "$out/textcraft-verdict.json"
exec python3 -I "$tests/textcraft_game.py" verify --task "$tests/task.json" --recipes "$tests/recipes.json" \
  --root "$root/" --log app/textcraft/actions.jsonl --out "$out"
