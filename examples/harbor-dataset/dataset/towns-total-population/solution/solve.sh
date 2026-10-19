#!/bin/bash
# A reference solution, for checking the task itself; the agent never sees it.
python3 -c 'import csv; print(sum(int(row["population"]) for row in csv.DictReader(open("towns.csv"))))' > answer.txt
