#!/bin/bash
# A reference solution, for checking the task itself; the agent never sees it.
python3 -c 'import csv; print(max(csv.DictReader(open("towns.csv")), key=lambda row: int(row["population"]))["town"])' > answer.txt
