"""Plot mean +/- std training curves (loss, AP/AP50/AP75) vs epoch across arms.

Reads ``train_log.jsonl`` from every run in an arm and aggregates across seeds --
the project invariant is "the distribution is the result", so each arm is a mean
line with a +/-1 std band over its runs, never a single cherry-picked seed. Two
or more arms are overlaid so the comparison is read directly.

Identity (name/colour/marker/linestyle) comes from model_style, keyed off the
MODEL, so the same model looks the same here as in the scaling / PR / confusion
figures. Arms are given as ``exp_id:model_nN_tag`` (resolved to the per-seed run
glob under --runs-base); the legacy ``label=glob`` form is still accepted.

No re-score and no checkpoint needed: loss_total, AP, AP50, AP75 are logged per
epoch by the trainer. PR / confusion live in a separate script.

The confound footer is configurable (--caption). The old hard-coded
FrozenBN-vs-GN note is true only for the same-model pretrained-vs-scratch ladder
and false for a cross-architecture comparison; the default is neutral.

Run:
    python plot_curves.py --runs-base /workspace/runs \
        --arm cross_g0:retinanet_n96_s8 --arm cross_g1:retinanet_pt_n96_s8 \
        --out-dir runs/plots
"""

import argparse
import json
import sys
from glob import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

import model_style as ms  # noqa: E402

PALETTE = {
    "light": {"surface": "#fcfcfb", "primary": "#0b0b0b", "secondary": "#52514e",
              "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7"},
    "dark": {"surface": "#1a1a19", "primary": "#f0efec", "secondary": "#c3c2b7",
             "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835"},
}

METRIC_LABELS = {
    "loss_total": ("training loss (total)", "Training loss vs epoch"),
    "AP": ("AP@[.50:.95]", "AP@[.50:.95] vs epoch"),
    "AP50": ("AP50", "AP50 vs epoch"),
    "AP75": ("AP75", "AP75 vs epoch"),
    "loss_box_reg": ("loss_box_reg", "Box-regression loss"),
    "loss_classifier": ("loss_classifier", "Classifier loss"),
    "loss_objectness": ("loss_objectness", "Objectness loss"),
    "loss_rpn_box_reg": ("loss_rpn_box_reg", "RPN box-reg loss"),
    # one-stage (retinanet/ssd/fcos) loss keys, so their panels are labelled too
    "bbox_regression": ("bbox_regression", "Box-regression loss"),
    "classification": ("classification", "Classification loss"),
}

DEFAULT_CAPTION = "dev-val, mean +/- 1 std over seeds."


class Arm:
    """One comparison arm: its resolved style + loaded runs."""

    def __init__(self, style, runs):
        self.style = style            # model_style.Arm
        self.runs = runs              # list of (run_id, DataFrame)

    @property
    def label(self):
        return self.style.label

    @property
    def n(self):
        return len(self.runs)


def load_run(run_dir):
    """Load one run's per-epoch log into a DataFrame (epoch made 1-based)."""
    run_dir = Path(run_dir)
    rows = []
    with (run_dir / "train_log.jsonl").open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"empty log: {run_dir}")
    df = pd.DataFrame(rows)
    df["epoch"] = (df["epoch"] + 1) if "epoch" in df else range(1, len(df) + 1)
    model = None
    args_path = run_dir / "args.json"
    if args_path.exists():
        try:
            model = json.loads(args_path.read_text()).get("model")
        except (json.JSONDecodeError, OSError):
            model = None
    return run_dir.name, df, model


def collect_arm(style, runs_base):
    """Load all runs matching a resolved arm's run glob into an Arm."""
    pattern = style.run_glob(runs_base)
    runs, models = [], set()
    for d in sorted(glob(pattern)):
        p = Path(d)
        if not (p / "train_log.jsonl").exists():
            continue
        run_id, df, model = load_run(p)
        runs.append((run_id, df))
        if model is not None:
            models.add(model)
    if not runs:
        raise SystemExit(f"[{style.label}] no runs with train_log.jsonl match: {pattern}")
    if len(models) > 1:
        print(f"[warn] arm '{style.label}' mixes models {sorted(models)} -- "
              f"glob too loose?", file=sys.stderr)
    print(f"[arm] {style.label}: {len(runs)} runs, model={sorted(models) or 'unknown'}")
    return Arm(style, runs)


def aggregate(arm, metric):
    """Per-epoch mean / std / n over an arm's runs for one metric."""
    series = [df[["epoch", metric]] for _, df in arm.runs if metric in df]
    if not series:
        return None
    long = pd.concat(series, ignore_index=True)
    g = long.groupby("epoch")[metric]
    return pd.DataFrame({"mean": g.mean(), "std": g.std(ddof=1), "n": g.count()})


def _style_axes(ax, colors, *, xlabel, ylabel, title):
    ax.set_title(title, color=colors["primary"], fontsize=12, pad=10, loc="left")
    ax.set_xlabel(xlabel, color=colors["secondary"], fontsize=10)
    ax.set_ylabel(ylabel, color=colors["secondary"], fontsize=10)
    ax.grid(True, color=colors["grid"], linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(colors["axis"])
    ax.tick_params(colors=colors["muted"], labelsize=9)


def draw_metric(ax, arms, metric, colors, *, show_seeds):
    """Draw one metric for all arms: mean line (model colour/linestyle) + band."""
    for arm in arms:
        color, ls, marker = arm.style.color, arm.style.linestyle, arm.style.marker
        agg = aggregate(arm, metric)
        if agg is None:
            print(f"[warn] metric '{metric}' absent for arm '{arm.label}'",
                  file=sys.stderr)
            continue
        if show_seeds:
            for _, df in arm.runs:
                if metric in df:
                    ax.plot(df["epoch"], df[metric], color=color,
                            linewidth=0.7, alpha=0.22, zorder=1)
        lo = agg["mean"] - agg["std"]
        hi = agg["mean"] + agg["std"]
        ax.fill_between(agg.index, lo, hi, color=color, alpha=0.16,
                        linewidth=0, zorder=2)
        markevery = max(1, len(agg.index) // 6)  # sparse markers keep dense curves clean
        ax.plot(agg.index, agg["mean"], color=color, linestyle=ls, linewidth=2.0,
                marker=marker, markevery=markevery, markersize=5,
                label=f"{arm.label} (n={arm.n})", zorder=3)
    ylabel, _ = METRIC_LABELS.get(metric, (metric, metric))
    return "epoch", ylabel


def figure_for_metric(arms, metric, colors, cfg):
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=cfg["dpi"])
    fig.patch.set_facecolor(colors["surface"])
    ax.set_facecolor(colors["surface"])
    xlabel, ylabel = draw_metric(ax, arms, metric, colors, show_seeds=cfg["show_seeds"])
    _, title = METRIC_LABELS.get(metric, (metric, metric))
    _style_axes(ax, colors, xlabel=xlabel, ylabel=ylabel, title=title)
    if len(arms) >= 2 or metric != "loss_total":
        ax.legend(frameon=False, fontsize=9, labelcolor=colors["primary"])
    fig.text(0.01, 0.01, cfg["caption"], fontsize=7, color=colors["muted"],
             ha="left", va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    return fig


def figure_ap_panel(arms, colors, cfg):
    metrics = ["AP", "AP50", "AP75"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), dpi=cfg["dpi"], sharex=True)
    fig.patch.set_facecolor(colors["surface"])
    for ax, metric in zip(axes, metrics):
        ax.set_facecolor(colors["surface"])
        xlabel, ylabel = draw_metric(ax, arms, metric, colors,
                                     show_seeds=cfg["show_seeds"])
        _, title = METRIC_LABELS[metric]
        _style_axes(ax, colors, xlabel=xlabel, ylabel=ylabel, title=title)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=colors["primary"])
    fig.text(0.01, 0.01, cfg["caption"], fontsize=7, color=colors["muted"],
             ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def figure_loss_components(arms, colors, cfg):
    comps = ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]
    present = [m for m in comps
               if any(m in df for arm in arms for _, df in arm.runs)]
    if not present:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), dpi=cfg["dpi"], sharex=True)
    fig.patch.set_facecolor(colors["surface"])
    for ax, metric in zip(axes.ravel(), comps):
        ax.set_facecolor(colors["surface"])
        if metric not in present:
            ax.set_visible(False)
            continue
        xlabel, ylabel = draw_metric(ax, arms, metric, colors,
                                     show_seeds=cfg["show_seeds"])
        _, title = METRIC_LABELS[metric]
        _style_axes(ax, colors, xlabel=xlabel, ylabel=ylabel, title=title)
    axes[0, 0].legend(frameon=False, fontsize=9, labelcolor=colors["primary"])
    fig.text(0.01, 0.01, cfg["caption"], fontsize=7, color=colors["muted"],
             ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def save(fig, out_dir, stem, formats):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
        print(f"[plot] wrote {path}")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", action="append", dest="arms", required=True,
                   metavar="EXP:MODEL_nN_tag",
                   help="exp_id:model_nN_tag (or legacy label=glob); repeatable.")
    p.add_argument("--arm-name", action="append", dest="arm_names", default=[])
    p.add_argument("--arm-color", action="append", dest="arm_colors", default=[])
    p.add_argument("--arm-linestyle", action="append", dest="arm_linestyles", default=[])
    p.add_argument("--runs-base", "--runs-root", dest="runs_base",
                   default="/workspace/runs",
                   help="Base dir that arm exp_ids / relative globs resolve "
                        "against. --runs-root kept as an alias for the pipeline.")
    p.add_argument("--out-dir", type=Path, default=Path("runs/plots"))
    p.add_argument("--metrics", default="loss_total,AP,AP50,AP75")
    p.add_argument("--show-seeds", action="store_true")
    p.add_argument("--loss-components", action="store_true")
    p.add_argument("--no-ap-panel", action="store_true")
    p.add_argument("--dark", action="store_true")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--formats", default="png,svg")
    p.add_argument("--caption", default=DEFAULT_CAPTION)
    return p.parse_args()


def main():
    args = parse_args()
    colors = PALETTE["dark" if args.dark else "light"]
    plt.rcParams["font.family"] = ["DejaVu Sans"]

    styles = ms.resolve_arms(args.arms, args.arm_names, args.arm_colors,
                             args.arm_linestyles)
    arms = [collect_arm(s, args.runs_base) for s in styles]
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    cfg = {"dpi": args.dpi, "show_seeds": args.show_seeds, "caption": args.caption}

    for metric in metrics:
        save(figure_for_metric(arms, metric, colors, cfg), args.out_dir,
             f"curve_{metric}", formats)

    if not args.no_ap_panel and {"AP", "AP50", "AP75"}.issubset(metrics):
        save(figure_ap_panel(arms, colors, cfg), args.out_dir, "panel_ap", formats)

    if args.loss_components:
        fig = figure_loss_components(arms, colors, cfg)
        if fig is not None:
            save(fig, args.out_dir, "panel_loss_components", formats)

    print("[plot] done")


if __name__ == "__main__":
    main()