"""RQ1 - CodeFlowLM incremental JIT-SDP, within-project (WP) or cross-project (CP).

Runs every (configuration, project, seed) combination of the paper with a single command and
collects one row per run into a CSV that scripts/rq1_stats.py consumes.

Each run gets its own working directory (results/rq1/work/<setting>/<config>/seed<seed>/<project>),
so intermediate files never collide and an interrupted run resumes from its last training step
(codeflowlm stops after 10 hours and stores training_status.pickle; rerun the same command to resume).
Runs whose result already exists are skipped unless --force is given.

Requirements: the codeflowlm and PEFT4CC repositories (see scripts/setup_rq1.sh) and a GPU
(A100 or >= 24 GB of VRAM for CodeT5+; a T4 is enough for UniXCoder).

Examples:
    python scripts/rq1_codeflowlm.py --setting wp --config CodeT5+-LoRA
    python scripts/rq1_codeflowlm.py --setting cp --config UniXCoder-PreT --projects gora --seeds 33
    python scripts/rq1_codeflowlm.py --setting wp --config all --dry-run
"""
import argparse
import glob
import os
import shutil
import sys

import pandas as pd

from rq1_config import (CODEFLOWLM_CONFIGS, DATA_DIR, REPO_ROOT, RESULT_COLUMNS, RESULTS_DIR, SEEDS,
                        project_list)

DEPS_DIR = os.path.join(REPO_ROOT, "deps")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--setting", choices=["wp", "cp"], required=True)
    parser.add_argument("--config", nargs="+", default=["all"],
                        help=f"One or more of {list(CODEFLOWLM_CONFIGS)}, or 'all'.")
    parser.add_argument("--projects", nargs="+", default=["all"])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--codeflowlm-dir", default=os.path.join(DEPS_DIR, "codeflowlm"),
                        help="Clone of github.com/monilouise/codeflowlm.")
    parser.add_argument("--peft4cc-parent", default=DEPS_DIR,
                        help="Directory that contains the PEFT4CC clone (codeflowlm calls PEFT4CC/just-in-time/*.py).")
    parser.add_argument("--work-dir", default=os.path.join(RESULTS_DIR, "work"))
    parser.add_argument("--out", help="Results CSV (default: results/rq1/runs/<setting>/codeflowlm.csv).")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--keep-checkpoints", action="store_true",
                        help="Keep model checkpoints after a run finishes (they take several GB per run).")
    parser.add_argument("--force", action="store_true", help="Rerun combinations that already have results.")
    parser.add_argument("--dry-run", action="store_true", help="Only list the runs that would be executed.")
    return parser.parse_args()


def selected_configs(names, setting):
    names = list(CODEFLOWLM_CONFIGS) if names == ["all"] else names
    unknown = sorted(set(names) - set(CODEFLOWLM_CONFIGS))
    if unknown:
        raise SystemExit(f"Unknown config(s): {unknown}. Valid: {list(CODEFLOWLM_CONFIGS)}")
    if setting == "cp" and "UniXCoder-LoRA" in names:
        # UniXCoder-LoRA was evaluated only within-project (Section 4.4.1).
        names = [n for n in names if n != "UniXCoder-LoRA"]
        print("Skipping UniXCoder-LoRA: the paper evaluates it only in the WP setting.")
    return names


def results_csv(run_dir, pretrained_model, project):
    return os.path.join(run_dir, pretrained_model, "concat", "online", "baseline", f"{project}_results_wp.csv")


def upsert(out_csv, row):
    """Insert or replace the row of one (method, setting, project, seed) in the consolidated CSV."""
    key = ["method", "setting", "project", "seed"]
    new = pd.DataFrame([row], columns=RESULT_COLUMNS)
    if os.path.exists(out_csv):
        df = pd.read_csv(out_csv)
        df = df[~(df[key] == pd.Series(row)[key]).all(axis=1)]
        new = pd.concat([df, new], ignore_index=True)
    df = new
    df.sort_values(key).to_csv(out_csv, index=False)


def cleanup(run_dir, pretrained_model, project):
    baseline_dir = os.path.join(run_dir, pretrained_model, "concat", "online", "baseline")
    shutil.rmtree(os.path.join(baseline_dir, f"{project}_best_gmean"), ignore_errors=True)
    for pattern in ("df_train_*.csv", "df_test_*.csv", "*_online_*.pkl", "global_training_pool.pkl"):
        for f in glob.glob(os.path.join(run_dir, pattern)):
            os.remove(f)


def run_one(args, train, config, project, seed, out_csv):
    pretrained_model, peft_alg = CODEFLOWLM_CONFIGS[config]
    run_dir = os.path.join(os.path.abspath(args.work_dir), args.setting, config, f"seed{seed}", project)
    result = results_csv(run_dir, pretrained_model, project)
    label = f"[{args.setting}] {config} / {project} / seed {seed}"

    if os.path.exists(result) and not args.force:
        print(f"{label}: already done, skipping")
    elif args.dry_run:
        print(f"{label}: would run in {run_dir}")
        return True
    else:
        print(f"{label}: running in {run_dir}")
        os.makedirs(run_dir, exist_ok=True)
        if args.force and os.path.exists(result):
            shutil.rmtree(os.path.join(run_dir, pretrained_model), ignore_errors=True)
        data = os.path.abspath(args.data_dir)
        cwd = os.getcwd()
        os.chdir(run_dir)  # codeflowlm writes intermediate files to the current directory
        try:
            train.train_project_with_lat_ver(
                os.path.abspath(args.peft4cc_parent), run_dir, run_dir + os.sep,
                os.path.join(data, "commit_guru") + os.sep,
                os.path.join(data, "ord_cross_features_train.pkl"),
                os.path.join(data, "ord_cross_features_valid.pkl"),
                os.path.join(data, "ord_cross_features_test.pkl"),
                os.path.join(data, "ord_cross_changes_train_lst.pkl"),
                os.path.join(data, "ord_cross_changes_valid_lst.pkl"),
                os.path.join(data, "ord_cross_changes_test_lst.pkl"),
                project=project, early_stop_metric="gmean", adjust_th=False, skewed_oversample=True,
                peft_alg=peft_alg, seed=seed, pretrained_model=pretrained_model,
                batch_size=args.batch_size, cross_project=args.setting == "cp",
                do_eval_with_all_negative=False)
        finally:
            os.chdir(cwd)
        if not os.path.exists(result):
            print(f"{label}: stopped before the end of the stream (time limit). "
                  "Run the same command again to resume.")
            return False
        if not args.keep_checkpoints:
            cleanup(run_dir, pretrained_model, project)

    if not args.dry_run:
        res = pd.read_csv(result).iloc[0]
        upsert(out_csv, {"method": config, "setting": args.setting, "project": project, "seed": seed,
                         "gmean": res["g_mean"], "rdiff": res["|R0-R1|"]})
    return True


def main():
    args = parse_args()
    configs = selected_configs(args.config, args.setting)
    projects = project_list(args.projects)
    out_csv = args.out or os.path.join(RESULTS_DIR, "runs", args.setting, "codeflowlm.csv")
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)

    train = None
    if not args.dry_run:
        os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "True")
        sys.path.insert(0, os.path.abspath(args.codeflowlm_dir))
        try:
            import codeflowlm.train as train
        except ImportError as e:
            raise SystemExit(f"Cannot import codeflowlm from {args.codeflowlm_dir} ({e}). "
                             "Run scripts/setup_rq1.sh first or pass --codeflowlm-dir.")

    for config in configs:
        for seed in args.seeds:
            for project in projects:
                if not run_one(args, train, config, project, seed, out_csv):
                    sys.exit(3)
    if not args.dry_run:
        print(f"Results written to {out_csv}")


if __name__ == "__main__":
    main()
