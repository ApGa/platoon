#!/bin/bash
# A reference solution, for checking the task itself; the agent never sees it.
grep -c " ERROR " events.txt > answer.txt
