"""RQ2 - JIT-Fine baseline (Section 4.3.2, Table 5).

JIT-Fine (baselines/JIT-Fine, a copy of https://github.com/jacknichao/JIT-Fine with the changes listed in
baselines/JIT-Fine/README.md) is trained and tested for each seed with the arguments of the original notebook
(JIT-Fine.ipynb), and one row per seed is written to results/rq2/baselines/JIT-Fine.csv, which rq2_metrics.py
adds to Table 5.

As in the notebook, the model is trained on data/changes_{train,valid}_lst.pkl (changed lines as ordered lists)
and tested on JIT-Fine's own data/jitfine/changes_test.pkl from baselines/JIT-Fine/data.zip (same commits,
labels and messages, but changed lines stored as sets). The features (data/features_*.pkl) and the line labels
(data/changes_complete_buggy_line_level.pkl) are identical to the files in data.zip.
JIT-Fine caches the tokenized datasets next to the data files, so all of them are placed in
results/rq2/work/JIT-Fine/data (the data/jitfine folder of the notebook), shared by all seeds.

Requires a GPU (several hours per seed in Google Colab).

Examples:
    python scripts/rq2_jitfine.py
    python scripts/rq2_jitfine.py --seeds 33 --dry-run
    python scripts/rq2_jitfine.py --skip-train          # only evaluate existing checkpoints
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import zipfile

import pandas as pd

from jitdl_lib import DATA_DIR, METRICS, REPO_ROOT, RESULTS_DIR

SEEDS = [33, 23, 99, 72, 22]
JITFINE_DIR = os.path.join(REPO_ROOT, "baselines", "JIT-Fine")
CONFIG = "JIT-Fine"
METRICS_LINE = re.compile(r"Top-10-ACC: ([\d.]+),\s*Top-5-ACC: ([\d.]+), Recall20%Effort: ([\d.]+), "
                          r"Effort@20%LOC: ([\d.]+), IFA: ([\d.]+)")
CODEBERT = ["--config_name=microsoft/codebert-base", "--model_name_or_path=microsoft/codebert-base",
            "--tokenizer_name=microsoft/codebert-base"]
# Arguments of the train and test cells of the original notebook.
TRAIN_ARGS = ["--epoch", "50", "--max_seq_length", "512", "--max_msg_length", "64", "--train_batch_size", "24",
              "--eval_batch_size", "128", "--learning_rate", "1e-5", "--max_grad_norm", "1.0",
              "--evaluate_during_training", "--feature_size", "14", "--patience", "10"]
TEST_ARGS = ["--epoch", "50", "--max_seq_length", "512", "--max_msg_length", "64", "--train_batch_size", "256",
             "--eval_batch_size", "25", "--learning_rate", "2e-5", "--max_grad_norm", "1.0",
             "--evaluate_during_training", "--only_adds"]
FROM_REPO = ["changes_train_lst.pkl", "changes_valid_lst.pkl", "changes_test_lst.pkl", "features_train.pkl",
             "features_valid.pkl", "features_test.pkl", "changes_complete_buggy_line_level.pkl"]
FROM_ZIP = ["changes_train.pkl", "changes_valid.pkl", "changes_test.pkl"]


def prepare_data(args, data_dir):
    """Build the data/jitfine folder of the notebook (copies, so that JIT-Fine's caches stay out of data/)."""
    os.makedirs(data_dir, exist_ok=True)
    for name in FROM_REPO:
        if not os.path.exists(os.path.join(data_dir, name)):
            shutil.copy(os.path.join(args.data_dir, name), data_dir)
    missing = [n for n in FROM_ZIP if not os.path.exists(os.path.join(data_dir, n))]
    if missing:
        with zipfile.ZipFile(os.path.join(args.jitfine_dir, "data.zip")) as z:
            for name in missing:
                with z.open(f"data/jitfine/{name}") as src, open(os.path.join(data_dir, name), "wb") as dst:
                    shutil.copyfileobj(src, dst)


def command(data_dir, seed, run_dir, train):
    d = data_dir
    if train:
        files = ["--train_data_file", os.path.join(d, "changes_train_lst.pkl"), os.path.join(d, "features_train.pkl"),
                 "--eval_data_file", os.path.join(d, "changes_valid_lst.pkl"), os.path.join(d, "features_valid.pkl"),
                 "--test_data_file", os.path.join(d, "changes_test_lst.pkl"), os.path.join(d, "features_test.pkl")]
    else:
        files = ["--train_data_file", os.path.join(d, "changes_train.pkl"), os.path.join(d, "features_train.pkl"),
                 "--eval_data_file", os.path.join(d, "changes_valid.pkl"), os.path.join(d, "features_valid.pkl"),
                 "--test_data_file", os.path.join(d, "changes_test.pkl"), os.path.join(d, "features_test.pkl"),
                 "--buggy_line_filepath=" + os.path.join(d, "changes_complete_buggy_line_level.pkl")]
    return [sys.executable, "-m", "JITFine.concat.run", "--output_dir", os.path.join(run_dir, "checkpoints"),
            *CODEBERT, "--do_train" if train else "--do_test", *files,
            *(TRAIN_ARGS if train else TEST_ARGS), "--seed", str(seed)]


def run(cmd, cwd, log_path):
    """Run one JIT-Fine step, echoing and saving its output; returns the output text."""
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
        sys.exit(f"JIT-Fine failed (exit code {proc.returncode}); see {log_path}")
    return "".join(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--skip-train", action="store_true", help="Only evaluate existing checkpoints.")
    parser.add_argument("--jitfine-dir", default=JITFINE_DIR)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--work-dir", default=os.path.join(RESULTS_DIR, "work"))
    parser.add_argument("--out", default=os.path.join(RESULTS_DIR, "baselines", f"{CONFIG}.csv"))
    parser.add_argument("--dry-run", action="store_true", help="Only print the commands.")
    args = parser.parse_args()

    data_dir = os.path.abspath(os.path.join(args.work_dir, CONFIG, "data"))
    if not args.dry_run:
        prepare_data(args, data_dir)
    rows = []
    for seed in args.seeds:
        run_dir = os.path.abspath(os.path.join(args.work_dir, CONFIG, f"seed{seed}"))
        steps = ([] if args.skip_train else [True]) + [False]
        if args.dry_run:
            for train in steps:
                print(" ".join(command(data_dir, seed, run_dir, train)))
            continue
        os.makedirs(run_dir, exist_ok=True)
        output = ""
        for train in steps:
            output = run(command(data_dir, seed, run_dir, train), args.jitfine_dir,
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
