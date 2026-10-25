#!/bin/bash
# Runs as the image's USER, in its working directory: 1 when the agent wrote out.txt there as that user, in the
# directories that the deferred WORKDIR made and gave it, while those the image had keep their owners and modes.
found="uid=$(id -u) pwd=$(pwd) out=$(cat out.txt 2>/dev/null) owner=$(stat -c %u out.txt 2>/dev/null)"
found="$found dirs=$(stat -c %u:%g:%a /srv /srv/secret /srv/work /srv/work/task 2>&1 | tr '\n' ' ')"
echo "$found"
expected="uid=1000 pwd=/srv/work/task out=done owner=1000 dirs=0:0:755 0:0:700 1000:1000:755 1000:1000:755 "
if [ "$found" = "$expected" ]; then echo 1; else echo 0; fi > /logs/verifier/reward.txt
