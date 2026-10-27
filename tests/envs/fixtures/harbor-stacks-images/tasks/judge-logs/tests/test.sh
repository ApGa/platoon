#!/bin/bash
if [ "$(cat /logs/verifier/judge/record.txt 2>/dev/null)" = "recorded" ]; then
  echo "the judge's record reached the tests"
  echo 1 > /logs/verifier/reward.txt
else
  echo "no record of the judge's"
  echo 0 > /logs/verifier/reward.txt
fi
