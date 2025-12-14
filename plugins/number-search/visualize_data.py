#!/usr/bin/env python
"""Quick script to visualize the number_search training data."""

import pandas as pd

# Load the parquet file
parquet_path = "platoon/number_search/number_search_train.parquet"
df = pd.read_parquet(parquet_path)

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")
print(f"\nData types:")
print(df.dtypes)

print("\n" + "=" * 60)
print("FIRST 3 ROWS (truncated)")
print("=" * 60)

for i, row in df.head(3).iterrows():
    print(f"\n--- Row {i} ---")
    for col in df.columns:
        val = row[col]
        val_str = str(val)
        if len(val_str) > 300:
            print(f"  {col}: {val_str[:300]}...")
        else:
            print(f"  {col}: {val_str}")

print("\n" + "=" * 60)
print("COLUMN STATISTICS")
print("=" * 60)

for col in df.columns:
    print(f"\n--- {col} ---")
    if df[col].dtype == "object":
        # String column - show length stats
        lengths = df[col].astype(str).str.len()
        print(f"  Type: string")
        print(f"  Length: min={lengths.min()}, max={lengths.max()}, mean={lengths.mean():.1f}")
        print(f"  Unique values: {df[col].nunique()}")
    else:
        # Numeric column
        print(f"  Type: {df[col].dtype}")
        print(f"  Min: {df[col].min()}, Max: {df[col].max()}, Mean: {df[col].mean():.2f}")

