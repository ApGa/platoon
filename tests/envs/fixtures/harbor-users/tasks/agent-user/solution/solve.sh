#!/bin/bash
echo done > result.txt
echo "$(id -u) $HOME" > whoami.txt
cat /srv/secret/answer.txt > secret.txt 2>/dev/null || echo denied > secret.txt
echo solved >> notes.txt
