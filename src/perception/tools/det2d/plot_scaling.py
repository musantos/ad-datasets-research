"""AP x N scaling curve for the perception data-scaling ladder.

Reads the per-step seed-sweep ledgers written by the trainer
(<runs_base>/<exp>/sweep_<model>_n<N>_<tag>.csv, column `best_ap`), aggregates
best_ap over seeds per arm, and plots AP against the number of training segments
N -- the headline data-scaling figure.

What changed vs the first version (why the cross figure came out wrong):
  * IDENTITY BY MODEL, not by role token. Colour, marker and legend name come
    from model_style (hue=family, shade=prior), so RetinaNet is always blue and
    SSD always green across every figure, regardless of which campaign/GPU ran
    them. Arms are named by `exp_id:model`, so a comparison can pull each model
    from whichever experiment folder holds it (e.g. retinanet from cross_g0,
    retinanet_pt from cross_g1) in one figure.
  * MEDIAN + band instead of mean +/- std. A diverged seed is logged as
    best_ap≈0, and a raw mean mixes training-failure rate into the performance
    number. Median is robust to that; the band is the IQR (or min-max) over the
    surviving seeds.
  * DIVERGENCE FILTER. Seeds that hit NaN are dropped before aggregation via
    model_style.seed_is_diverged (reads the `diverged` column when present, else
    the pre-fix epoch-1 heuristic). n_converged/n_total is annotated so a
    majority-diverged point (e.g. cross RetinaNet n6) is never read as "AP≈0".
  * CONFOUND FOOTER configurable (--caption). The old hard-coded FrozenBN-vs-GN
    note is true only for the same-model pretrained-vs-scratch ladder; it is
    false for a cross-architecture comparison. Default is a neutral note.

x-axis is the segment count on a log2 scale. A scaling_table.csv with the
aggregated numbers is written alongside the figure.

Run (pure stdlib + matplotlib):
    python plot_scaling.py --runs-base /workspace/runs --steps "6 24 96" --tag s8 \
        --out-dir runs/plots \
        --arm cross_g0:retinanet --arm cross_g1:retinanet_pt \
        --arm cross_g1:ssd --arm cross_g0:ssd_pt
"""

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

import model_style as ms

DEFAULT_CAPTION = ("median over converged seeds; band = IQR. diverged seeds "
                   "(NaN / epoch-1 collapse) excluded; n_converged/n_total annotated.")


def read_aps(csv_path, drop_diverged=True):
    """(converged best_ap list, n_total) from a sweep ledger."""
    aps, n_total = [], 0
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            v = (row.get("best_ap") or "").strip()
            if not v:
                continue
            n_total += 1
            if drop_diverged and ms.seed_is_diverged(row):
                continue
            try:
                aps.append(float(v))
            except ValueError:
                continue
    return aps, n_total


def aggregate(aps, band):
    """Central value + lo/hi band over a seed list, per the chosen band mode."""
    center = statistics.median(aps)
    if len(aps) < 2:
        return {"center": center, "lo": center, "hi": center}
    if band == "minmax":
        lo, hi = min(aps), max(aps)
    elif band == "std":
        s = statistics.pstdev(aps)
        lo, hi = center - s, center + s
    else:  # iqr
        q = statistics.quantiles(aps, n=4, method="inclusive")
        lo, hi = q[0], q[2]
    return {"center": center, "lo": lo, "hi": hi}


def collect(runs_base, arms, steps, tag, band, drop_diverged):
    """{arm_index: {N: {center,lo,hi,n_conv,n_total}}}."""
    out = {}
    for i, arm in enumerate(arms):
        out[i] = {}
        for n in steps:
            path = arm.ledger_path(runs_base, f"n{n}_{tag}")
            if not Path(path).is_file():
                continue
            aps, n_total = read_aps(path, drop_diverged)
            if not aps:
                out[i][n] = {"center": None, "lo": None, "hi": None,
                             "n_conv": 0, "n_total": n_total}
                continue
            agg = aggregate(aps, band)
            agg.update(n_conv=len(aps), n_total=n_total)
            out[i][n] = agg
    return out


def main():
    p = argparse.ArgumentParser(description="AP x N data-scaling curve + table.")
    p.add_argument("--runs-base", "--runs-root", dest="runs_base",
                   default="/workspace/runs",
                   help="Base dir; arm exp_ids resolve under it. --runs-root is "
                        "kept as an alias for the existing pipeline.")
    p.add_argument("--steps", default="6 12 24 48 96",
                   help="space-separated segment counts (ladder x-axis)")
    p.add_argument("--tag", default="s8", help="stride tag in the ledger filenames")
    p.add_argument("--out-dir", type=Path, default=Path("runs/plots"))
    p.add_argument("--arm", action="append", dest="arms", metavar="EXP:MODEL",
                   help="exp_id:model (model-level); repeatable.")
    p.add_argument("--arm-name", action="append", dest="arm_names", default=[],
                   help="Override display name for the arm at the same position.")
    p.add_argument("--arm-color", action="append", dest="arm_colors", default=[],
                   help="Override colour (named, e.g. blue) for the same-position arm.")
    p.add_argument("--arm-linestyle", action="append", dest="arm_linestyles",
                   default=[], help="Override linestyle (solid/dashed/dotted).")
    p.add_argument("--band", choices=["iqr", "minmax", "std"], default="iqr")
    p.add_argument("--no-diverge-filter", action="store_true",
                   help="Keep diverged seeds in the aggregation (debug).")
    p.add_argument("--title", default="Data-scaling: val-AP vs number of segments")
    p.add_argument("--caption", default=DEFAULT_CAPTION)
    p.add_argument("--dark", action="store_true")
    args = p.parse_args()

    if not args.arms:
        raise SystemExit("at least one --arm exp_id:model is required")
    arms = ms.resolve_arms(args.arms, args.arm_names, args.arm_colors,
                           args.arm_linestyles)
    steps = [int(x) for x in args.steps.split()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = collect(args.runs_base, arms, steps, args.tag, args.band,
                   not args.no_diverge_filter)

    # --- table --------------------------------------------------------------
    table_path = args.out_dir / "scaling_table.csv"
    with open(table_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "model", "N_segments", "n_converged", "n_total",
                    "center_ap", "band_lo", "band_hi"])
        for i, arm in enumerate(arms):
            for n in steps:
                s = data.get(i, {}).get(n)
                if s and s["center"] is not None:
                    w.writerow([arm.label, arm.model or "", n, s["n_conv"],
                                s["n_total"], f"{s['center']:.4f}",
                                f"{s['lo']:.4f}", f"{s['hi']:.4f}"])
    print(f"[scaling] wrote {table_path}")

    # --- figure -------------------------------------------------------------
    if args.dark:
        plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    plotted = False
    for i, arm in enumerate(arms):
        xs = [n for n in steps
              if n in data.get(i, {}) and data[i][n]["center"] is not None]
        if not xs:
            continue
        plotted = True
        center = [data[i][n]["center"] for n in xs]
        lo = [data[i][n]["lo"] for n in xs]
        hi = [data[i][n]["hi"] for n in xs]
        ax.plot(xs, center, marker=arm.marker, color=arm.color,
                linestyle=arm.linestyle, label=arm.label, zorder=3)
        ax.fill_between(xs, lo, hi, color=arm.color, alpha=0.18, zorder=1)
        for n in xs:
            s = data[i][n]
            note = f"{s['center']:.3f}"
            if s["n_conv"] != s["n_total"]:
                note += f"\n({s['n_conv']}/{s['n_total']})"
            ax.annotate(note, (n, s["center"]), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=7, color=arm.color)

    ax.set_xscale("log", base=2)
    ax.set_xticks(steps)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("training segments N (log2)")
    ax.set_ylabel("val-AP @[.50:.95]  (median over seeds)")
    ax.set_title(args.title)
    ax.grid(True, which="both", alpha=0.25)
    if plotted:
        ax.legend(title="model", frameon=False)
    else:
        ax.text(0.5, 0.5, "no ledgers found yet", transform=ax.transAxes,
                ha="center", va="center", color="gray")
    fig.text(0.5, 0.005, args.caption, ha="center", fontsize=7, style="italic",
             color="gray")

    out_png = args.out_dir / "scaling_ap.png"
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"[scaling] wrote {out_png}")


if __name__ == "__main__":
    main()