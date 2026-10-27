#!/bin/bash
if [ "$(cat /app/answer.txt 2>/dev/null)" = "forty-two" ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
