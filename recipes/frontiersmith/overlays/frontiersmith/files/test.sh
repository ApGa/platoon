#!/bin/bash
# FrontierSmith's tests on platoon: evaluate.py writes the reward, or marks the reading invalid when the judge cannot
# grade; nothing writes 0 in its place (the adapter's test.sh did, for a verifier that crashed).
mkdir -p /logs/verifier
python3 /tests/evaluate.py 2>&1 | tee /logs/verifier/evaluation.log
