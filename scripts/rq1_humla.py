"""RQ1 - HumLa and Eco-HumLa baselines (Song et al., 2023) on JIT-Defects4J, WP or CP.

Replaces editing baselines/HumLa-fse2023/codes/run_RQs.py by hand: every project, method and
seed is run with a single command, and one row per (method, project, seed) is written to
results/rq1/runs/<setting>/humla.csv.

Settings follow Section 4.4.1: 90-day waiting time, pre-training on the first 50 commits, and
the classifier parameters tuned by grid search stored in codes/data_stream/para_bst_30sd.csv.
  * HumLa:     human labeling with no noise and full effort (RQ1.1 of the HumLa paper);
  * Eco-HumLa: human_eff = "auto_ecohumla2" (RQ2.1 of the HumLa paper).
CP uses a single stream with the commits of all projects ("all_projects") and evaluates the
predictions of each target project.

Requires the environment of the HumLa repository (Python 3.9, numpy, scikit-learn, scikit-multiflow).

Examples:
    python scripts/rq1_humla.py --setting wp
    python scripts/rq1_humla.py --setting cp --methods Eco-HumLa --projects gora
"""
import argparse
import os
import sys

import pandas as pd

from rq1_config import BASELINES_DIR, RESULT_COLUMNS, RESULTS_DIR, project_list

HUMLA_DIR = os.path.join(BASELINES_DIR, "HumLa-fse2023")
METHODS = {
    "HumLa": {"has_human": True, "human_err": 0, "human_eff": 1},
    "Eco-HumLa": {"has_human": True, "human_err": 0, "human_eff": "auto_ecohumla2"},
}
ALL_PROJECTS_ID = 33  # "all_projects" stream of the CP setting
SEEDS = list(range(5))  # HumLa's own seeds (0..4), as in the experiments of the paper


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--setting", choices=["wp", "cp"], required=True)
    parser.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS))
    parser.add_argument("--projects", nargs="+", default=["all"])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--n-para-tune", type=int, default=50, help="Commits used for pre-training (default: 50).")
    parser.add_argument("--wait-days", type=int, default=90)
    parser.add_argument("--humla-dir", default=HUMLA_DIR)
    parser.add_argument("--work-dir", default=os.path.join(RESULTS_DIR, "work", "humla"),
                        help="Where HumLa stores its per-seed predictions (rslt.save).")
    parser.add_argument("--out", help="Results CSV (default: results/rq1/runs/<setting>/humla.csv).")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_humla(humla_dir):
    """Import HumLa's main module; it expects to run from codes/ with the repository root on sys.path."""
    codes = os.path.join(os.path.abspath(humla_dir), "codes")
    sys.path[:0] = [codes, os.path.abspath(humla_dir)]
    os.chdir(codes)  # HumLa reads ../dataset and writes ../results relative to codes/
    import main as humla
    from data_stream.real_data_stream import data_id_2name
    name_to_id = {}
    for project_id in range(ALL_PROJECTS_ID):
        name_to_id[data_id_2name(project_id)] = project_id
    return humla, name_to_id


def main():
    args = parse_args()
    projects = project_list(args.projects)
    out_csv = os.path.abspath(args.out or os.path.join(RESULTS_DIR, "runs", args.setting, "humla.csv"))
    work_dir = os.path.abspath(args.work_dir)
    cp = args.setting == "cp"

    if args.dry_run:
        for method in args.methods:
            for project in projects:
                stream = "all_projects" if cp else project
                print(f"[{args.setting}] {method} / {project} (stream {stream}) / seeds {args.seeds}")
        return

    humla, name_to_id = load_humla(args.humla_dir)
    humla.configure(n_para_tune=args.n_para_tune, wait_days=args.wait_days,
                    results_dir=os.path.join(work_dir, "rslt.save"))

    rows = []
    for method in args.methods:
        for project in projects:
            print(f"[{args.setting}] {method} / {project}", flush=True)
            if cp:
                res = humla.run_HumLa(METHODS[method], ALL_PROJECTS_ID, None, args.seeds, 0, False,
                                      cp=True, target_project_id=name_to_id[project])
            else:
                res = humla.run_HumLa(METHODS[method], name_to_id[project], None, args.seeds, 0, False, cp=False)
            for seed, gmean, rdiff in zip(res["seeds"], res["gmean"], res["rdiff"]):
                rows.append({"method": method, "setting": args.setting, "project": project, "seed": seed,
                             "gmean": gmean, "rdiff": rdiff})

    df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    if os.path.exists(out_csv):
        # keep results of methods/projects that were not rerun now
        old = pd.read_csv(out_csv)
        key = ["method", "setting", "project", "seed"]
        old = old[~old.set_index(key).index.isin(df.set_index(key).index)]
        df = pd.concat([old, df], ignore_index=True)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.sort_values(["method", "project", "seed"]).to_csv(out_csv, index=False)
    print(f"Results written to {out_csv}")


if __name__ == "__main__":
    main()
