"""AP x N scaling curve for the perception data-scaling ladder.

Reads the per-step seed-sweep ledgers written by the trainer
(runs/sweep_<model>_n<N>_<tag>.csv, column `best_ap`), aggregates best_ap over
seeds (mean +/- std) per arm, and plots AP against the number of training
segments N -- the headline data-scaling figure: "how does val-AP evolve as we
scale the number of scenarios". One line per arm, error band = +/- 1 std over
seeds (the distribution IS the result, not best-of-seed).

The x-axis is the segment count on a log2 scale (the ladder doubles: 6,12,...).
A scaling_table.csv with the aggregated numbers is written alongside the figure.

Palette matches the rest of the study (pretrained=blue, from-scratch=orange) and
the FrozenBN-vs-GN confound is stamped in the footer of every figure.

Run (PYTHONPATH not needed; pure stdlib + matplotlib):
    python plot_scaling.py --runs-root runs --steps "6 12 24 48 96" --tag s8 \
        --out-dir runs/plots \
        --arm "pretrained=faster_rcnn_pt" --arm "from-scratch=faster_rcnn"
"""

import argparse
import csv
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Study palette (ABERTURA): pretrained = blue, from-scratch = orange.
ARM_COLORS = {"pretrained": "#1f6fb2", "from-scratch": "#e08a1e"}
DEFAULT_COLORS = ["#1f6fb2", "#e08a1e", "#3a9e55", "#b23a3a"]
CONFOUND = ("confound: pretrained=FrozenBN vs from-scratch=GroupNorm "
            "(prior and norm coupled)")


def parse_arm(spec):
    """'label=model' -> (label, model)."""
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"--arm must be label=model, got {spec!r}")
    label, model = spec.split("=", 1)
    return label.strip(), model.strip()


def read_best_aps(csv_path):
    """Return the list of finite best_ap values in a sweep ledger (skip nan)."""
    aps = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            v = row.get("best_ap", "").strip()
            if not v or v.lower() == "nan":
                continue
            try:
                aps.append(float(v))
            except ValueError:
                continue
    return aps


def collect(runs_root, arms, steps, tag):
    """{label: {N: {mean,std,n,min,max}}} from the per-step ledgers."""
    out = {}
    for label, model in arms:
        out[label] = {}
        for n in steps:
            csv_path = runs_root / f"sweep_{model}_n{n}_{tag}.csv"
            if not csv_path.is_file():
                continue
            aps = read_best_aps(csv_path)
            if not aps:
                continue
            out[label][n] = {
                "mean": statistics.fmean(aps),
                "std": statistics.pstdev(aps) if len(aps) > 1 else 0.0,
                "n": len(aps),
                "min": min(aps),
                "max": max(aps),
            }
    return out


def main():
    p = argparse.ArgumentParser(description="AP x N data-scaling curve + table.")
    p.add_argument("--runs-root", type=Path, default=Path("runs"))
    p.add_argument("--steps", default="6 12 24 48 96",
                   help="space-separated segment counts (ladder x-axis)")
    p.add_argument("--tag", default="s8", help="stride tag in the ledger filenames")
    p.add_argument("--out-dir", type=Path, default=Path("runs/plots"))
    p.add_argument("--arm", action="append", type=parse_arm, dest="arms",
                   help="label=model; repeatable. Default: the two study arms.")
    p.add_argument("--title", default="Data-scaling: val-AP vs number of segments")
    p.add_argument("--dark", action="store_true")
    args = p.parse_args()

    arms = args.arms or [("pretrained", "faster_rcnn_pt"),
                         ("from-scratch", "faster_rcnn")]
    steps = [int(x) for x in args.steps.split()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = collect(args.runs_root, arms, steps, args.tag)

    # --- table (always written, even if a step is still missing) ------------
    table_path = args.out_dir / "scaling_table.csv"
    with open(table_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "N_segments", "n_seeds", "mean_ap", "std_ap", "min", "max"])
        for label, _ in arms:
            for n in steps:
                s = data.get(label, {}).get(n)
                if s:
                    w.writerow([label, n, s["n"], f"{s['mean']:.4f}",
                                f"{s['std']:.4f}", f"{s['min']:.4f}", f"{s['max']:.4f}"])
    print(f"[scaling] wrote {table_path}")

    # --- figure -------------------------------------------------------------
    if args.dark:
        plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    plotted = False
    for i, (label, _) in enumerate(arms):
        xs = [n for n in steps if n in data.get(label, {})]
        if not xs:
            continue
        plotted = True
        means = [data[label][n]["mean"] for n in xs]
        stds = [data[label][n]["std"] for n in xs]
        color = ARM_COLORS.get(label, DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
        ax.plot(xs, means, marker="o", color=color, label=label, zorder=3)
        ax.fill_between(xs,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        color=color, alpha=0.18, zorder=1)
        for n in xs:
            s = data[label][n]
            ax.annotate(f"{s['mean']:.3f}", (n, s["mean"]),
                        textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=7, color=color)

    ax.set_xscale("log", base=2)
    ax.set_xticks(steps)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("training segments N (log2)")
    ax.set_ylabel("val-AP @[.50:.95]  (mean ± std over seeds)")
    ax.set_title(args.title)
    ax.grid(True, which="both", alpha=0.25)
    if plotted:
        ax.legend(title="arm", frameon=False)
    else:
        ax.text(0.5, 0.5, "no ledgers found yet", transform=ax.transAxes,
                ha="center", va="center", color="gray")
    fig.text(0.5, 0.005, CONFOUND, ha="center", fontsize=7, style="italic",
             color="gray")

    out_png = args.out_dir / "scaling_ap.png"
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"[scaling] wrote {out_png}")


if __name__ == "__main__":
    main()
