#!/bin/bash
# A reference solution, for checking the task itself; the agent never sees it.
grep -o -w -i harbor notes.txt | wc -l > answer.txt
