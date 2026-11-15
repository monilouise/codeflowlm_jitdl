"""RQ2 - JIT-SmartATTN baseline (Section 4.3.2, Table 5).

JIT-Smart (baselines/JIT-Smart, a copy of https://github.com/JIT-A/JIT-Smart with the changes listed in
baselines/JIT-Smart/README.md) trained for defect prediction only (--dl_loss_weight 0) and evaluated by
ranking the changed lines with the attention scores (JIT-SmartATTN). For each seed the model is trained and
then tested with the same arguments used in the paper, and one row per seed is written to
results/rq2/baselines/JIT-SmartATTN.csv, which rq2_metrics.py adds to Table 5.

Data: data/changes_{train,valid,test}_lst.pkl and data/features_{train,valid,test}.pkl (the features are the
same files as in JIT-Smart's data.zip) and the line labels of JIT-Smart
(baselines/JIT-Smart/JITSmart/process and label defect code lines/*_buggy_commit_lines_df.pkl).

Requires a GPU (train and test took about 5 h per seed in Google Colab).

Examples:
    python scripts/rq2_jitsmart.py
    python scripts/rq2_jitsmart.py --seeds 33 --dry-run
    python scripts/rq2_jitsmart.py --skip-train          # only evaluate existing checkpoints
"""
import argparse
import os
import re
import subprocess
import sys

import pandas as pd

from jitdl_lib import DATA_DIR, METRICS, REPO_ROOT, RESULTS_DIR

SEEDS = [33, 23, 99, 72, 22]
JITSMART_DIR = os.path.join(REPO_ROOT, "baselines", "JIT-Smart")
LINES_DIR = os.path.join(JITSMART_DIR, "JITSmart", "process and label defect code lines")
CONFIG = "JIT-SmartATTN"
METRICS_LINE = re.compile(r"Top-10-ACC: ([\d.]+),\s*Top-5-ACC: ([\d.]+), Recall20%Effort: ([\d.]+), "
                          r"Effort@20%LOC: ([\d.]+), IFA: ([\d.]+)")
CODEBERT = ["--config_name=microsoft/codebert-base", "--model_name_or_path=microsoft/codebert-base",
            "--tokenizer_name=microsoft/codebert-base"]
# Arguments of the train and test cells of the original notebook (JIT-Smart.ipynb).
TRAIN_ARGS = ["--epoch", "50", "--max_seq_length", "512", "--max_msg_length", "64", "--train_batch_size", "24",
              "--eval_batch_size", "128", "--learning_rate", "1e-5", "--max_grad_norm", "1.0",
              "--evaluate_during_training", "--feature_size", "14", "--patience", "10",
              "--max_codeline_length", "256", "--max_codeline_token_length", "64",
              "--dp_loss_weight", "1.0", "--dl_loss_weight", "0.0"]
TEST_ARGS = ["--epoch", "50", "--max_seq_length", "512", "--max_msg_length", "64", "--train_batch_size", "256",
             "--eval_batch_size", "24", "--learning_rate", "2e-5", "--max_grad_norm", "1.0",
             "--evaluate_during_training", "--only_adds", "--max_codeline_length", "256",
             "--max_codeline_token_length", "64", "--dp_loss_weight", "1.0", "--dl_loss_weight", "0"]


def command(args, seed, run_dir, train):
    d = os.path.abspath(args.data_dir)
    lines = [os.path.join(args.lines_dir, f"{split}_buggy_commit_lines_df.pkl") for split in ("train", "valid", "test")]
    cmd = [sys.executable, "-m", "JITSmart.concat.run", "--seed", str(seed),
           "--output_dir", os.path.join(run_dir, "checkpoints"), *CODEBERT,
           "--do_train" if train else "--do_test",
           "--train_data_file", os.path.join(d, "changes_train_lst.pkl"), os.path.join(d, "features_train.pkl"),
           "--eval_data_file", os.path.join(d, "changes_valid_lst.pkl"), os.path.join(d, "features_valid.pkl"),
           "--test_data_file", os.path.join(d, "changes_test_lst.pkl"), os.path.join(d, "features_test.pkl"),
           "--buggy_lines_file", *lines]
    if train:
        return cmd + TRAIN_ARGS
    return cmd + ["--buggy_line_filepath", lines[2]] + TEST_ARGS


def run(cmd, cwd, log_path):
    """Run one JIT-Smart step, echoing and saving its output; returns the output text."""
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")
        output = []
        for line in proc.stdout:
            log.write(line)
            output.append(line)
            if "INFO" in line or "Error" in line:
                print(line, end="", flush=True)
        proc.wait()
    if proc.returncode != 0:
        sys.exit(f"JIT-Smart failed (exit code {proc.returncode}); see {log_path}")
    return "".join(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--skip-train", action="store_true", help="Only evaluate existing checkpoints.")
    parser.add_argument("--jitsmart-dir", default=JITSMART_DIR)
    parser.add_argument("--lines-dir", default=LINES_DIR)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--work-dir", default=os.path.join(RESULTS_DIR, "work"))
    parser.add_argument("--out", default=os.path.join(RESULTS_DIR, "baselines", f"{CONFIG}.csv"))
    parser.add_argument("--dry-run", action="store_true", help="Only print the commands.")
    args = parser.parse_args()

    rows = []
    for seed in args.seeds:
        run_dir = os.path.abspath(os.path.join(args.work_dir, CONFIG, f"seed{seed}"))
        steps = ([] if args.skip_train else [True]) + [False]
        if args.dry_run:
            for train in steps:
                print(" ".join(command(args, seed, run_dir, train)))
            continue
        os.makedirs(run_dir, exist_ok=True)
        output = ""
        for train in steps:
            output = run(command(args, seed, run_dir, train), args.jitsmart_dir,
                         os.path.join(run_dir, "train.log" if train else "test.log"))
        found = METRICS_LINE.findall(output)
        if not found:
            sys.exit(f"seed {seed}: no localization metrics in {run_dir}/test.log")
        top10, top5, recall20, effort20, ifa = map(float, found[-1])
        rows.append({"model": CONFIG, "config": "concat", "run": seed, "top5": top5, "top10": top10,
                     "recall20": recall20, "effort20": effort20, "ifa": ifa})
        print(f"seed {seed}: " + ", ".join(f"{m} = {rows[-1][m]:.4f}" for m in METRICS), flush=True)

    if rows:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        df = pd.DataFrame(rows)
        if os.path.exists(args.out):
            old = pd.read_csv(args.out)
            df = pd.concat([old[~old["run"].isin(df["run"])], df], ignore_index=True)
        df.sort_values("run").to_csv(args.out, index=False)
        print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
