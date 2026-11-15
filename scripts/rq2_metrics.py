"""RQ2 - Line-level JIT-DL metrics of the LLM answers (Table 5).

Evaluates every stored answer file (one per model and run) over the true-positive commits of
CodeT5+ and writes:
  results/rq2/llm_runs.csv     one row per (model, run);
  results/rq2/summary.csv      mean and standard deviation per model, together with the
                               attention-based baselines found in results/rq2/baselines/*.csv.

Answer files are looked up as <responses-dir>/responses-<model>_exec-<run>.pkl or
<responses-dir>/responses-<model>.pkl (a single run), as produced by scripts/rq2_llm.py.

Example:
    python scripts/rq2_metrics.py
    python scripts/rq2_metrics.py --models gpt-5
"""
import argparse
import glob
import os
import pickle
import re

import pandas as pd

from jitdl_lib import (DATA_DIR, METRICS, RESPONSES_DIR, RESULTS_DIR, evaluate_responses, load_buggy_changes,
                       load_line_labels, load_true_positives)

# file name stem -> name used in the paper
MODEL_NAMES = {"gpt-5": "GPT-5", "gemini": "Gemini 2.5 Pro", "claude": "Claude Sonnet 4.5"}
METRIC_LABELS = {"top5": "Top-5", "top10": "Top-10", "recall20": "Recall@20%", "effort20": "Effort@20%", "ifa": "IFA"}


def response_files(responses_dir, models):
    files = []
    for path in sorted(glob.glob(os.path.join(responses_dir, "responses-*.pkl"))):
        m = re.fullmatch(r"responses-(.+?)(?:_exec-(\d+))?\.pkl", os.path.basename(path))
        stem, run = m.group(1), int(m.group(2) or 0)
        if models and stem not in models:
            continue
        files.append((MODEL_NAMES.get(stem, stem), run, path))
    return files


def summarize(runs):
    """Mean and (sample) standard deviation over runs, per model."""
    grouped = runs.groupby("model")[METRICS]
    summary = grouped.mean().join(grouped.std(ddof=1), rsuffix="_std")
    summary["runs"] = runs.groupby("model").size()
    return summary.reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", help=f"Response file stems to evaluate (default: all), e.g. {list(MODEL_NAMES)}.")
    parser.add_argument("--responses-dir", default=RESPONSES_DIR)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--out-dir", default=RESULTS_DIR)
    args = parser.parse_args()

    files = response_files(args.responses_dir, args.models)
    if not files:
        raise SystemExit(f"No responses-*.pkl files found in {args.responses_dir}")

    tp = load_true_positives(args.data_dir)
    line_labels = load_line_labels(args.data_dir)
    buggy_changes = load_buggy_changes(args.data_dir)

    rows = []
    for model, run, path in files:
        with open(path, "rb") as f:
            responses = pickle.load(f)
        if len(responses) != len(tp):
            raise SystemExit(f"{path}: {len(responses)} answers, expected {len(tp)} (one per true-positive commit)")
        res = evaluate_responses(responses, tp, line_labels, buggy_changes)
        rows.append({"model": model, "run": run, **res})
        print(f"{model} run {run}: " + ", ".join(f"{METRIC_LABELS[m]} = {res[m]:.3f}" for m in METRICS)
              + f" ({res['n_commits']} commits)")

    os.makedirs(args.out_dir, exist_ok=True)
    runs = pd.DataFrame(rows)
    runs.to_csv(os.path.join(args.out_dir, "llm_runs.csv"), index=False)

    baselines = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(args.out_dir, "baselines", "*.csv")))]
    all_runs = pd.concat([runs] + baselines, ignore_index=True)
    summary = summarize(all_runs).sort_values("top5", ascending=False)
    summary.to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)

    print("\nTable 5 (mean, std over runs when available):")
    print(f"{'Model':<20}" + "".join(f"{METRIC_LABELS[m]:>16}" for m in METRICS) + f"{'runs':>6}")
    for _, row in summary.iterrows():
        cells = []
        for m in METRICS:
            std = row[f"{m}_std"]
            cells.append(f"{row[m]:.3f}" + (f" (±{std:.2f})" if pd.notna(std) else "        "))
        print(f"{row['model']:<20}" + "".join(f"{c:>16}" for c in cells) + f"{int(row['runs']):>6}")


if __name__ == "__main__":
    main()
