#!/bin/bash
stored=$(python3 /tests/pg.py "SELECT value FROM answers" 2>&1)
if [ "$stored" = "forty-two" ]; then
  echo "the server holds the row"
  echo 1 > /logs/verifier/reward.txt
else
  echo "the server answered: $stored"
  echo 0 > /logs/verifier/reward.txt
fi
