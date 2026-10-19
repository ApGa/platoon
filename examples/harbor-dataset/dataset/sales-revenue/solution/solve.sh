#!/bin/bash
# A reference solution, for checking the task itself; the agent never sees it.
python3 -c 'import csv; print("%.2f" % sum(int(row["units"]) * float(row["unit_price"]) for row in csv.DictReader(open("sales.csv"))))' > answer.txt
