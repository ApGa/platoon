#!/bin/bash
# Runs in the separate verifier's sandbox, which holds only the artifacts collected from the api sidecar.
# PLATOON_SANDBOX_ROOT is set only by platoon's process provider.
root="${PLATOON_SANDBOX_ROOT:-}"
python3 - "$root" <<'PY'
import json
import sys
from pathlib import Path

root = sys.argv[1]
reward = Path(f"{root}/logs/verifier/reward.txt")
orders, stats = Path(f"{root}/tmp/api/orders.log"), Path(f"{root}/tmp/stats.json")
if not orders.is_file() or not stats.is_file():
    print("the api's evidence was not collected")
    reward.write_text("0")
    sys.exit(0)
items = [json.loads(line)["item"] for line in orders.read_text().splitlines() if line]
counted = json.loads(stats.read_text()).get("orders")
passed = len(items) == 3 and len(set(items)) == 3 and counted == 3
print(f"orders: {items}; the api counted {counted}")
reward.write_text("1" if passed else "0")
PY
