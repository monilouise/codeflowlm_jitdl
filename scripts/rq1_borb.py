"""RQ1 - BORB baseline (Cabral et al., 2023) on JIT-Defects4J, WP or CP.

Two subcommands:
  run     runs `jitsdp borb` for every (base learner, project, seed) with the default BORB
          hyperparameters, storing the MLflow runs under results/rq1/work/borb/mlruns-<setting>;
  export  reads an MLflow directory and writes one row per run (method = BORB-<learner>)
          to results/rq1/runs/<setting>/borb.csv. rq1_stats.py keeps the best learner per project.

`run` needs the jit-sdp-nn fork (github.com/monilouise/jit-sdp-nn) installed in its own
environment (see baselines/jit-sdp-nn/README.md); `export` only needs pandas.

Examples:
    python scripts/rq1_borb.py run --setting wp
    python scripts/rq1_borb.py run --setting cp --models lr nb --projects gora --seeds 33
    python scripts/rq1_borb.py export --setting wp
    python scripts/rq1_borb.py export --setting wp --mlruns baselines/jit-sdp-nn/mlruns_rep5_wp \
        baselines/jit-sdp-nn/mlruns_rep5_commons-jcs-nb
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

import pandas as pd

from rq1_config import (BASELINES_DIR, BORB_MLP_EPOCHS, BORB_MODELS, RESULT_COLUMNS, RESULTS_DIR, SEEDS,
                        project_list)

BORB_DIR = os.path.join(BASELINES_DIR, "jit-sdp-nn")
MLFLOW_FINISHED = "3"


def default_mlruns(setting):
    return os.path.join(RESULTS_DIR, "work", "borb", f"mlruns-{setting}")


def jitsdp_executable():
    """The `jitsdp` entry point of the environment running this script (its Scripts/ or bin/ folder
    is not on PATH when the interpreter is called by its full path), falling back to PATH."""
    folder = os.path.dirname(sys.executable)
    for candidate in (os.path.join(folder, "Scripts", "jitsdp.exe"), os.path.join(folder, "jitsdp.exe"),
                      os.path.join(folder, "jitsdp")):
        if os.path.isfile(candidate):
            return candidate
    return shutil.which("jitsdp") or "jitsdp"


def borb_command(model, project, seed, setting):
    cmd = [jitsdp_executable(), "borb", "--cross-project", "1" if setting == "cp" else "0", "--model", model]
    if model == "mlp":
        cmd += ["--mlp-n-epochs", str(BORB_MLP_EPOCHS)]
    return cmd + ["--dataset", project, "--seed", str(seed)]


def read_file(path):
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def last_metric(run_dir, name):
    # MLflow file store: one "<timestamp> <value> <step>" line per logged value.
    return float(read_file(os.path.join(run_dir, "metrics", name)).splitlines()[-1].split()[1])


def finished_runs(mlruns):
    rows = []
    for meta in glob.glob(os.path.join(mlruns, "*", "*", "meta.yaml")):
        run_dir = os.path.dirname(meta)
        info = dict(line.split(": ", 1) for line in read_file(meta).splitlines() if ": " in line)
        if info.get("status") != MLFLOW_FINISHED or not os.path.exists(os.path.join(run_dir, "metrics", "g-mean")):
            continue
        params = {p: read_file(os.path.join(run_dir, "params", p))
                  for p in ("dataset", "model", "seed", "cross_project")}
        rows.append({
            "method": f"BORB-{params['model']}",
            "setting": "cp" if params["cross_project"] == "1" else "wp",
            "project": params["dataset"],
            "seed": int(params["seed"]),
            "gmean": last_metric(run_dir, "g-mean"),
            "rdiff": last_metric(run_dir, "r0-r1"),
            "end_time": int(info.get("end_time", 0)),
        })
    return pd.DataFrame(rows)


def cmd_run(args):
    mlruns = os.path.abspath(args.mlruns or default_mlruns(args.setting))
    env = dict(os.environ, MLFLOW_TRACKING_URI="file:" + mlruns.replace(os.sep, "/"))
    # Like `conda activate`: without the env's DLL folders on PATH, Windows fails to load _ssl.
    prefix = os.path.dirname(sys.executable)
    env_dirs = [os.path.join(prefix, *d) for d in
                [(), ("Library", "mingw-w64", "bin"), ("Library", "usr", "bin"), ("Library", "bin"), ("Scripts",),
                 ("bin",)]]
    env["PATH"] = os.pathsep.join([d for d in env_dirs if os.path.isdir(d)] + [env.get("PATH", "")])
    done = set()
    if os.path.isdir(mlruns) and not args.force:
        df = finished_runs(mlruns)
        if len(df):
            done = set(zip(df["method"], df["setting"], df["project"], df["seed"]))

    for project in project_list(args.projects):
        for model in args.models:
            for seed in args.seeds:
                cmd = borb_command(model, project, seed, args.setting)
                if (f"BORB-{model}", args.setting, project, seed) in done:
                    print(f"skip (done): {' '.join(cmd)}")
                    continue
                print(" ".join(cmd), flush=True)
                if not args.dry_run:
                    # jitsdp caches the downloaded dataset in ./data, so it runs inside its own folder.
                    subprocess.run(cmd, cwd=args.borb_dir, env=env, check=True)

    if not args.dry_run:
        args.mlruns = [mlruns]
        args.merge = True
        cmd_export(args)


def cmd_export(args):
    dirs = args.mlruns or [default_mlruns(args.setting)]
    df = pd.concat([finished_runs(d) for d in dirs], ignore_index=True)
    if df.empty:
        sys.exit(f"No finished BORB runs found in {dirs}")
    df = df[df["setting"] == args.setting]
    # Keep only the latest run of a repeated (learner, project, seed) combination.
    key = ["method", "setting", "project", "seed"]
    n = len(df)
    df = df.sort_values("end_time").drop_duplicates(key, keep="last")
    if len(df) < n:
        print(f"Ignored {n - len(df)} repeated run(s), keeping the latest of each combination.")
    out = args.out or os.path.join(RESULTS_DIR, "runs", args.setting, "borb.csv")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    if getattr(args, "merge", False) and os.path.exists(out):
        # `run` may cover only some projects/seeds: keep the other rows already in the CSV.
        old = pd.read_csv(out)
        old = old[~old.set_index(key).index.isin(df.set_index(key).index)]
        df = pd.concat([old[RESULT_COLUMNS], df[RESULT_COLUMNS]], ignore_index=True)
    df[RESULT_COLUMNS].sort_values(key).to_csv(out, index=False)
    print(f"{len(df)} runs written to {out}")
    counts = df.groupby("method")["project"].nunique()
    print("Projects per learner:", counts.to_dict())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run BORB and export the results.")
    run.add_argument("--setting", choices=["wp", "cp"], required=True)
    run.add_argument("--models", nargs="+", choices=BORB_MODELS, default=BORB_MODELS)
    run.add_argument("--projects", nargs="+", default=["all"])
    run.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    run.add_argument("--borb-dir", default=BORB_DIR)
    run.add_argument("--mlruns", help="MLflow directory (default: results/rq1/work/borb/mlruns-<setting>).")
    run.add_argument("--out", help="Results CSV (default: results/rq1/runs/<setting>/borb.csv).")
    run.add_argument("--force", action="store_true", help="Rerun combinations that already have a finished run.")
    run.add_argument("--dry-run", action="store_true", help="Only print the commands.")
    run.set_defaults(func=cmd_run)

    export = sub.add_parser("export", help="Export finished MLflow runs to the RQ1 results CSV.")
    export.add_argument("--setting", choices=["wp", "cp"], required=True)
    export.add_argument("--mlruns", nargs="+",
                        help="One or more MLflow directories (default: results/rq1/work/borb/mlruns-<setting>).")
    export.add_argument("--out", help="Results CSV (default: results/rq1/runs/<setting>/borb.csv).")
    export.set_defaults(func=cmd_export)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
