#!/bin/bash
# A reference solution, for checking the task itself; the agent never sees it.
grep -m 1 " ERROR " events.txt | cut -d " " -f 1 > answer.txt
