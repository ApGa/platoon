#!/bin/bash
# Runs as root in the agent's sandbox: 1 only when the agent wrote its files as uid 1000, with its own HOME, could
# not read the answer kept to root, and could append to the copy given to it.
found="result=$(cat /app/result.txt 2>/dev/null) owner=$(stat -c %u /app/result.txt 2>/dev/null)"
found="$found whoami=$(cat /app/whoami.txt 2>/dev/null) secret=$(cat /app/secret.txt 2>/dev/null)"
found="$found notes=$(stat -c %u /app/notes.txt 2>/dev/null):$(tail -n 1 /app/notes.txt 2>/dev/null)"
echo "$found"
if [ "$found" = "result=done owner=1000 whoami=1000 /srv/agent secret=denied notes=1000:solved" ]; then echo 1
else echo 0; fi > /logs/verifier/reward.txt
