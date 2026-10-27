#!/bin/bash
# Runs as the image's user, as Docker runs it: 1 when that user is 65534 and wrote the result itself.
found="uid=$(id -u) result=$(cat /app/output/result.txt 2>/dev/null) owner=$(stat -c %u /app/output/result.txt 2>/dev/null)"
echo "$found"
if [ "$found" = "uid=65534 result=ok owner=65534" ]; then echo 1; else echo 0; fi > /logs/verifier/reward.txt
