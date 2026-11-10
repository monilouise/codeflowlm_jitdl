"""
Check whether, for each numeric column with the same name in both CSV files,
and for rows matched by author_date_unix_timestamp, the value in
brackets_vld_st.csv equals the corresponding value in backets.csv divided by 10.
"""
import numpy as np
import pandas as pd

CG_PATH = "dataset/commit_guru/backets.csv"
VLD_PATH = "dataset/data.inuse/brackets_vld_st.csv"
KEY = "author_date_unix_timestamp"
TOL = 1e-6  # absolute tolerance for floating-point comparison

cg = pd.read_csv(CG_PATH)
vld = pd.read_csv(VLD_PATH)

# Numeric columns common to both files (excluding the join key)
common_cols = set(cg.columns) & set(vld.columns)
numeric_cg = set(cg.select_dtypes(include=[np.number]).columns)
numeric_vld = set(vld.select_dtypes(include=[np.number]).columns)
cols_to_check = sorted((common_cols & numeric_cg & numeric_vld) - {KEY})

print(f"Common numeric columns to check: {cols_to_check}\n")

# Inner-join on the timestamp key
merged = cg.merge(vld, on=KEY, suffixes=("_cg", "_vld"))
print(f"Rows in commit_guru: {len(cg)}")
print(f"Rows in vld_st:      {len(vld)}")
print(f"Rows after merge on '{KEY}': {len(merged)}\n")

if merged.empty:
    print("No rows share the same author_date_unix_timestamp; nothing to check.")
    raise SystemExit(0)

summary = []
for col in cols_to_check:
    a = merged[f"{col}_cg"]
    b = merged[f"{col}_vld"]

    expected = a
    # Both NaN -> treat as match
    both_nan = a.isna() & b.isna()
    diff = (b - expected).abs()
    match = both_nan | (diff <= TOL)

    n_total = len(merged)
    n_match = int(match.sum())
    n_mismatch = n_total - n_match
    summary.append((col, n_total, n_match, n_mismatch))

    print(f"[{col}] match={n_match}/{n_total}  mismatch={n_mismatch}")
    if n_mismatch > 0:
        bad = merged.loc[~match, [KEY, f"{col}_cg", f"{col}_vld"]].head(5).copy()
        bad["expected_cg"] = bad[f"{col}_cg"]
        print("  first mismatches:")
        print(bad.to_string(index=False))
    print()

print("=" * 60)
print("SUMMARY (column, rows, matches, mismatches)")
for row in summary:
    print(f"  {row[0]:<30} rows={row[1]}  match={row[2]}  mismatch={row[3]}")

all_ok = all(m == 0 for _, _, _, m in [(c, t, mt, ms) for c, t, mt, ms in summary])
print("\nResult:", "ALL columns satisfy vld == cg" if all_ok
      else "Some columns DO NOT satisfy vld == cg")
