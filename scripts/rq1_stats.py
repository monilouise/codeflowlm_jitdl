"""RQ1 statistical analysis (Section 4.2.1 / 5.1 of the paper).

Reads one or more result CSVs with the columns
    method,setting,project,seed,gmean,rdiff
averages each (method, project) over seeds and runs, following Demsar (2006):
  * Friedman test + Iman-Davenport correction;
  * Nemenyi critical difference (all-pairs post-hoc);
  * Kendall's W;
  * Cliff's delta between each proposed configuration and every other method.

BORB rows may be named "BORB-<learner>" (one per base learner). With --borb-best (default),
they are collapsed into a single "BORB" method that keeps, for each project, the learner
with the highest mean G-mean, as reported in the paper.

Examples:
    python scripts/rq1_stats.py results/rq1/paper/wp.csv
    python scripts/rq1_stats.py results/rq1/runs/wp/*.csv --cd-plot results/rq1/cd_wp.png
"""
import argparse
import glob
import sys

import numpy as np
import pandas as pd
from scipy import stats

# Critical values q_alpha for the two-tailed Nemenyi test (Demsar, 2006, Table 5), k = 2..10.
NEMENYI_Q = {
    0.05: [1.960, 2.343, 2.569, 2.728, 2.850, 2.949, 3.031, 3.102, 3.164],
    0.10: [1.645, 2.052, 2.291, 2.459, 2.589, 2.693, 2.780, 2.855, 2.920],
}
HIGHER_IS_BETTER = {"gmean": True, "rdiff": False}


def load_results(paths):
    files = [f for p in paths for f in (glob.glob(p) or [p])]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    missing = {"method", "project", "gmean"} - set(df.columns)
    if missing:
        sys.exit(f"Missing columns in input: {missing}")
    return df


def collapse_borb(means):
    """Keep, per project, the BORB base learner with the highest mean G-mean."""
    is_borb = means["method"].str.startswith("BORB-")
    if not is_borb.any():
        return means
    borb = means[is_borb]
    best = borb.loc[borb.groupby("project")["gmean"].idxmax()].copy()
    print("BORB best base learner per project:")
    for _, row in best.sort_values("project").iterrows():
        print(f"  {row['project']:<24} {row['method']}")
    best["method"] = "BORB"
    return pd.concat([means[~is_borb], best], ignore_index=True)


def friedman(table, higher_is_better):
    """table: projects x methods. Returns average ranks and test statistics."""
    n, k = table.shape
    ranks = table.rank(axis=1, ascending=not higher_is_better, method="average")
    avg_ranks = ranks.mean(axis=0).sort_values()
    chi2 = 12 * n / (k * (k + 1)) * (np.sum(avg_ranks ** 2) - k * (k + 1) ** 2 / 4)
    df1, df2 = k - 1, (k - 1) * (n - 1)
    ff = (n - 1) * chi2 / (n * (k - 1) - chi2)
    return {
        "avg_ranks": avg_ranks,
        "chi2": chi2,
        "p_chi2": stats.chi2.sf(chi2, k - 1),
        "ff": ff,
        "df1": df1,
        "df2": df2,
        "p_ff": stats.f.sf(ff, df1, df2),
        "f_crit_05": stats.f.ppf(0.95, df1, df2),
        "f_crit_10": stats.f.ppf(0.90, df1, df2),
        "kendall_w": chi2 / (n * (k - 1)),
        "n": n,
        "k": k,
    }


def nemenyi_cd(k, n, alpha):
    return NEMENYI_Q[alpha][k - 2] * np.sqrt(k * (k + 1) / (6 * n))


def cliffs_delta(x, y):
    x, y = np.asarray(x), np.asarray(y)
    diff = x[:, None] - y[None, :]
    return (np.sum(diff > 0) - np.sum(diff < 0)) / (len(x) * len(y))


def cliffs_magnitude(d):
    # Romano et al. (2006) thresholds.
    d = abs(d)
    if d < 0.147:
        return "negligible"
    if d < 0.33:
        return "small"
    if d < 0.474:
        return "medium"
    return "large"


def plot_cd(avg_ranks, cd, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Demsar-style layout: rank axis on top (best on the right), best half labelled on the right.
    k = len(avg_ranks)
    half = (k + 1) // 2
    fig, ax = plt.subplots(figsize=(9, 1.6 + 0.4 * half))
    ax.set_xlim(k + 0.9, 0.1)
    ax.set_ylim(half + 0.6, -1.4)
    ax.axis("off")
    ax.plot([1, k], [0, 0], color="black", lw=1)
    for r in range(1, k + 1):
        ax.plot([r, r], [0, -0.15], color="black", lw=1)
        ax.text(r, -0.25, str(r), ha="center", va="bottom")
    ax.plot([k, k - cd], [-1.0, -1.0], color="red", lw=2)
    ax.text(k - cd / 2, -1.1, f"CD = {cd:.2f}", ha="center", va="bottom", color="red")
    for i, (method, rank) in enumerate(avg_ranks.items()):
        best_half = i < half
        y = (i + 1) if best_half else (k - i)
        end = 0.6 if best_half else k + 0.4
        ax.plot([rank, rank, end], [0, y, y], color="black", lw=0.8)
        ax.text(end, y, f" {method} ({rank:.2f}) " if best_half else f" ({rank:.2f}) {method} ",
                va="center", ha="left" if best_half else "right")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print(f"Critical difference diagram saved to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="Result CSV files (globs accepted).")
    parser.add_argument("--metric", choices=sorted(HIGHER_IS_BETTER), default="gmean")
    parser.add_argument("--setting", choices=["wp", "cp"], help="Filter rows by setting.")
    parser.add_argument("--alpha", type=float, choices=[0.05, 0.10], default=0.05)
    parser.add_argument("--no-borb-best", dest="borb_best", action="store_false",
                        help="Keep each BORB base learner as a separate method.")
    parser.add_argument("--proposed", nargs="+",
                        help="Methods treated as proposed for Cliff's delta (default: CodeT5+*/UniXCoder*).")
    parser.add_argument("--cd-plot", help="Path of an optional critical difference diagram (PNG).")
    parser.add_argument("--out", help="Optional CSV with the per-project table used in the tests.")
    args = parser.parse_args()

    df = load_results(args.inputs)
    if args.setting:
        df = df[df["setting"] == args.setting]

    means = df.groupby(["method", "project"], as_index=False)[["gmean", "rdiff"]].mean()
    if args.borb_best:
        means = collapse_borb(means)

    table = means.pivot(index="project", columns="method", values=args.metric)
    incomplete = table[table.isna().any(axis=1)]
    if len(incomplete):
        print(f"Dropping projects with missing methods: {list(incomplete.index)}")
        table = table.dropna()
    if args.out:
        table.to_csv(args.out)

    higher = HIGHER_IS_BETTER[args.metric]
    res = friedman(table, higher)
    n, k = res["n"], res["k"]
    cd = nemenyi_cd(k, n, args.alpha)

    print(f"\nMetric: {args.metric} ({'higher' if higher else 'lower'} is better); N = {n} projects, k = {k} methods")
    print("\nAverage ranks (lower is better):")
    for method, rank in res["avg_ranks"].items():
        print(f"  {method:<20} {rank:.2f}")
    print(f"\nFriedman chi2_F = {res['chi2']:.2f} (p = {res['p_chi2']:.4f})")
    print(f"Iman-Davenport F_F({res['df1']}, {res['df2']}) = {res['ff']:.2f}, p = {res['p_ff']:.4g}")
    print(f"  critical F: {res['f_crit_05']:.2f} (alpha = 0.05), {res['f_crit_10']:.2f} (alpha = 0.10)")
    rejected = res["p_ff"] < args.alpha
    print(f"  null hypothesis {'REJECTED' if rejected else 'not rejected'} at alpha = {args.alpha}")
    print(f"Kendall's W = {res['kendall_w']:.2f}")
    print(f"Nemenyi CD (alpha = {args.alpha}) = {cd:.2f}")

    ranks = res["avg_ranks"]
    print("\nPairwise average-rank gaps (* = above CD):")
    methods = list(ranks.index)
    for i, a in enumerate(methods):
        for b in methods[i + 1:]:
            gap = abs(ranks[a] - ranks[b])
            print(f"  {a} vs {b}: {gap:.2f}{' *' if gap > cd else ''}")

    proposed = args.proposed or [m for m in methods if m.startswith(("CodeT5", "UniXCoder"))]
    if proposed:
        print("\nCliff's delta over per-project values (positive = row method better):")
        sign = 1 if higher else -1
        for i, a in enumerate(proposed):
            others = [m for m in methods if m not in proposed] + proposed[i + 1:]
            for b in others:
                d = sign * cliffs_delta(table[a], table[b])
                print(f"  {a} vs {b}: {d:.2f} ({cliffs_magnitude(d)})")

    if args.cd_plot:
        plot_cd(ranks, cd, args.cd_plot, f"{args.metric} ({args.setting or 'all'})")


if __name__ == "__main__":
    main()
