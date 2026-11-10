import math
import re
import sys

import pandas as pd

GURU_PATH = "dataset/commit_guru/backets.csv"
VLD_PATH = "dataset/data.inuse/brackets_vld_st.csv"
TS_COL = "author_date_unix_timestamp"
TOL = 1e-4


def find_entropy_col(columns):
    pattern = re.compile(r"entr", re.IGNORECASE)
    matches = [c for c in columns if pattern.search(c)]
    if not matches:
        raise ValueError(f"No entropy-like column found in: {list(columns)}")
    return matches[0]


def main():
    guru = pd.read_csv(GURU_PATH)
    vld = pd.read_csv(VLD_PATH)

    guru_ent_col = find_entropy_col(guru.columns)
    vld_ent_col = find_entropy_col(vld.columns)
    print(f"commit_guru entropy column: {guru_ent_col!r}")
    print(f"data.inuse entropy column:  {vld_ent_col!r}")

    guru = guru[[TS_COL, guru_ent_col]].copy()
    vld = vld[[TS_COL, vld_ent_col]].copy()

    guru[TS_COL] = pd.to_numeric(guru[TS_COL], errors="coerce").astype("Int64")
    vld[TS_COL] = pd.to_numeric(vld[TS_COL], errors="coerce").astype("Int64")
    guru[guru_ent_col] = pd.to_numeric(guru[guru_ent_col], errors="coerce")
    vld[vld_ent_col] = pd.to_numeric(vld[vld_ent_col], errors="coerce")

    # Report duplicate timestamps inside each file (ambiguous matches).
    for name, df in (("commit_guru", guru), ("data.inuse", vld)):
        dups = df[df.duplicated(TS_COL, keep=False)]
        if not dups.empty:
            n_ts = dups[TS_COL].nunique()
            print(f"[warn] {name}: {len(dups)} rows share {n_ts} duplicated timestamps")

    merged = vld.merge(guru, on=TS_COL, how="left", suffixes=("_vld", "_guru"))

    missing = merged[merged[guru_ent_col].isna()]
    compared = merged.dropna(subset=[guru_ent_col, vld_ent_col])

    diff = (compared[vld_ent_col] - compared[guru_ent_col]).abs()
    mismatches = compared[diff > TOL]
    matches = compared[diff <= TOL]

    print(f"\nRows in vld file: {len(vld)}")
    print(f"Rows with timestamp not found in commit_guru: {len(missing)}")
    print(f"Rows compared: {len(compared)}")
    print(f"Mismatches (|diff| > {TOL}): {len(mismatches)}")
    print(f"Matches (|diff| <= {TOL}): {len(matches)}")

    if not mismatches.empty:
        print("\nFirst mismatches:")
        print(mismatches.head(20).to_string(index=False))
        
    if not matches.empty:
        print("\nFirst matches:")
        print(matches.head(20).to_string(index=False))

    if len(mismatches) == 0 and len(missing) == 0:
        print("\nOK: every vld row has a matching entropy value in commit_guru.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
