"""RQ2 - Attention-based JIT-DL baselines CodeT5+ and UniXCoder (Section 4.3.2, Table 5).

For each seed, the model is fine-tuned with LoRA on the JIT-Defects4J split with PEFT4CC
(deps/PEFT4CC @ ae42f8b, see scripts/setup_rq1.sh) and then tested; the lines of each true-positive commit
are ranked by attention and the line-level metrics are averaged over those commits. The arguments are those
of the "Best results" (CodeT5+) and "Normalizacao por comprimento de linha (Best execution)" (UniXCoder)
cells of the original notebook (JIT-DL.ipynb). One row per seed is written to
results/rq2/baselines/<config>.csv, which rq2_metrics.py adds to Table 5.

Each run uses its own folder (results/rq2/work/<config>/seed<seed>), including a private copy of the
line-level labels: PEFT4CC caches their tokenization next to that file
(changes_complete_buggy_line_level_cache.pkl), and the cache depends on the tokenizer. In the original
notebook a single cache, created on 2025-09-14, was shared by all runs; pass it with --labels-cache to
reproduce that setting.

Requires a GPU (in the original Colab runs, CodeT5+ took about 15 min per epoch, 10 epochs per seed).

Examples:
    python scripts/rq2_attention.py --config CodeT5+-LoRA
    python scripts/rq2_attention.py --config UniXCoder-LoRA --seeds 23 --dry-run
    python scripts/rq2_attention.py --config UniXCoder-LoRA --skip-train   # only evaluate existing checkpoints
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

import pandas as pd

from jitdl_lib import DATA_DIR, METRICS, REPO_ROOT, RESULTS_DIR

SEEDS = [33, 23, 99, 72, 22]
LABELS = "changes_complete_buggy_line_level.pkl"
LABELS_CACHE = "changes_complete_buggy_line_level_cache.pkl"
# name -> (row name in Table 5, pretrained model, arguments of the notebook cells besides data, seed and step)
CONFIGS = {
    "CodeT5+-LoRA": ("CodeT5+", "codet5p-770m",
                     ["--learning_rate", "1e-4", "--epochs", "10", "--batch_size", "8", "--hidden_size", "1024",
                      "--loc_last_k_layers", "3", "--loc_attn_temp", "0.85"]),
    "UniXCoder-LoRA": ("UniXCoder", "unixcoder",
                       ["--learning_rate", "1e-4", "--epochs", "10", "--batch_size", "16",
                        "--only_adds", "False", "--loc_deleted_weight", "0.5"]),
}
METRICS_LINE = re.compile(r"Top-10-ACC: ([\d.]+),\s*Top-5-ACC: ([\d.]+), Recall20%Effort: ([\d.]+), "
                          r"Effort@20%LOC: ([\d.]+), IFA: ([\d.]+)")


def command(args, seed, run_dir, train):
    _, model, extra = CONFIGS[args.config]
    d = os.path.abspath(args.data_dir)
    cmd = [
        sys.executable, os.path.join(os.path.abspath(args.peft4cc_dir), "just-in-time", "run_lora.py"),
        "--train_data_file", os.path.join(d, "changes_train_lst.pkl"), os.path.join(d, "features_train.pkl"),
        "--eval_data_file", os.path.join(d, "changes_valid_lst.pkl"), os.path.join(d, "features_valid.pkl"),
        "--test_data_file", os.path.join(d, "changes_test_lst.pkl"), os.path.join(d, "features_test.pkl"),
        "--output_dir", os.path.join(run_dir, "checkpoints"),
        "--pretrained_model", model, "--seed", str(seed),
        "--do_train" if train else "--do_test", "--use_lora", "--calculate_metrics",
        "--buggy_line_filepath", os.path.join(run_dir, LABELS), *extra,
    ]
    # In PEFT4CC @ ae42f8b the localization metrics are only computed with --do_locate_defects.
    return cmd if train else cmd + ["--do_locate_defects"]


def run(cmd, cwd, log_path):
    """Run one PEFT4CC step, echoing and saving its output; returns the output text."""
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")
        output = []
        for line in proc.stdout:
            log.write(line)
            output.append(line)
            if "INFO" in line or "Error" in line or "Top-10-ACC" in line:
                print(line, end="", flush=True)
        proc.wait()
    if proc.returncode != 0:
        sys.exit(f"PEFT4CC failed (exit code {proc.returncode}); see {log_path}")
    return "".join(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", choices=list(CONFIGS), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--skip-train", action="store_true", help="Only evaluate existing checkpoints.")
    parser.add_argument("--labels-cache", help=f"Existing {LABELS_CACHE} to use in every run "
                                               "(default: each run builds its own with its tokenizer).")
    parser.add_argument("--peft4cc-dir", default=os.path.join(REPO_ROOT, "deps", "PEFT4CC"))
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--work-dir", default=os.path.join(RESULTS_DIR, "work"))
    parser.add_argument("--out", help="Output CSV (default: results/rq2/baselines/<config>.csv).")
    parser.add_argument("--dry-run", action="store_true", help="Only print the commands.")
    args = parser.parse_args()

    table_name = CONFIGS[args.config][0]
    out_csv = args.out or os.path.join(RESULTS_DIR, "baselines", f"{args.config}.csv")
    rows = []
    for seed in args.seeds:
        run_dir = os.path.abspath(os.path.join(args.work_dir, args.config, f"seed{seed}"))
        steps = ([] if args.skip_train else [True]) + [False]
        if args.dry_run:
            for train in steps:
                print(" ".join(command(args, seed, run_dir, train)))
            continue
        os.makedirs(run_dir, exist_ok=True)
        if not os.path.exists(os.path.join(run_dir, LABELS)):
            shutil.copy(os.path.join(args.data_dir, LABELS), run_dir)
        if args.labels_cache:
            shutil.copy(args.labels_cache, os.path.join(run_dir, LABELS_CACHE))
        output = ""
        for train in steps:
            # PEFT4CC writes results.csv, predictions.pkl, line_df.csv, ... to the current directory
            output = run(command(args, seed, run_dir, train), run_dir,
                         os.path.join(run_dir, "train.log" if train else "test.log"))
        found = METRICS_LINE.findall(output)
        if not found:
            sys.exit(f"seed {seed}: no localization metrics in {run_dir}/test.log")
        top10, top5, recall20, effort20, ifa = map(float, found[-1])
        rows.append({"model": table_name, "config": args.config, "run": seed, "top5": top5, "top10": top10,
                     "recall20": recall20, "effort20": effort20, "ifa": ifa})
        print(f"seed {seed}: " + ", ".join(f"{m} = {rows[-1][m]:.4f}" for m in METRICS), flush=True)

    if rows:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
        df = pd.DataFrame(rows)
        if os.path.exists(out_csv):
            old = pd.read_csv(out_csv)
            df = pd.concat([old[~old["run"].isin(df["run"])], df], ignore_index=True)
        df.sort_values("run").to_csv(out_csv, index=False)
        print(f"Results written to {out_csv}")


if __name__ == "__main__":
    main()
