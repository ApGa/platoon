#!/bin/bash
tr a-z A-Z < input.txt > output.txt
echo written > "${PLATOON_SANDBOX_ROOT:-}/opt/scratch/note.txt"
