"""Check if all CSV files in dataset/data.inuse are sorted by
author_date_unix_timestamp in ascending order.
"""
import glob
import os
import sys

import pandas as pd

COLUMN = "author_date_unix_timestamp"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.inuse")


def check_file(path: str) -> tuple[bool, str]:
    try:
        df = pd.read_csv(path, usecols=[COLUMN])
    except ValueError:
        return False, f"column '{COLUMN}' not found"
    except Exception as exc:
        return False, f"error reading file: {exc}"

    series = df[COLUMN]
    if series.is_monotonic_increasing:
        return True, "sorted ascending"

    diffs = series.diff()
    bad = diffs[diffs < 0]
    first_idx = bad.index[0]
    return (
        False,
        f"not sorted: {len(bad)} out-of-order rows; "
        f"first violation at row {first_idx} "
        f"({series.iloc[first_idx - 1]} -> {series.iloc[first_idx]})",
    )


def main() -> int:
    pattern = os.path.join(DATA_DIR, "*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No CSV files found in {DATA_DIR}")
        return 1

    all_ok = True
    for path in files:
        ok, msg = check_file(path)
        status = "OK " if ok else "FAIL"
        print(f"[{status}] {os.path.basename(path)}: {msg}")
        if not ok:
            all_ok = False

    print()
    print("All files sorted ascending." if all_ok else "Some files are NOT sorted ascending.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
