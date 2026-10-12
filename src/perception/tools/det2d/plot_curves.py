"""Plot mean +/- std training curves (loss, AP/AP50/AP75) vs epoch across arms.

Reads ``train_log.jsonl`` from every run in an arm's glob and aggregates across
seeds -- the project invariant is "the distribution is the result, not the best
seed", so each arm is drawn as a mean line with a +/-1 std band over its runs,
never a single cherry-picked seed. Two (or more) arms are overlaid on the same
axes so pretrained-vs-scratch is read directly.

No re-score and no checkpoint needed: loss_total, AP, AP50 and AP75 are logged
per epoch by the trainer (eval-ap-every=1), so these curves come straight off
disk. PR / confusion live in a separate script that consumes predictions.json.

Arm membership is given by an explicit glob (``label=glob``); the ``model`` key
in each run's args.json is read only to assert the arm is homogeneous and warn
if a glob accidentally mixes faster_rcnn with faster_rcnn_pt.

Confound, stated on every figure: pretrained uses FrozenBN, from-scratch uses
GroupNorm (no pretrained GN weights exist). The comparison is pt+FrozenBN vs
scratch+GN -- prior and normalization are coupled, not isolated.

Run (PYTHONPATH not required; pure pandas + matplotlib):

    python plot_curves.py \
        --runs-root runs \
        --arm "pretrained=faster_rcnn_pt_n48_s8_seed*" \
        --arm "from-scratch=faster_rcnn_n48_s8_seed*" \
        --out-dir runs/plots

Add ``--show-seeds`` for faint per-seed lines behind each mean, ``--dark`` for
dark-surface figures, ``--loss-components`` for the 4-panel RPN/head loss
breakdown.
"""

import argparse
import json
import sys
from glob import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; write files, never a display
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# --- palette (data-viz reference instance; pt=slot1 blue, scratch=slot2 orange).
# Fixed order: color follows the entity, assigned by arm order, never cycled.
PALETTE = {
    "light": {
        "surface": "#fcfcfb", "primary": "#0b0b0b", "secondary": "#52514e",
        "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
    },
    "dark": {
        "surface": "#1a1a19", "primary": "#f0efec", "secondary": "#c3c2b7",
        "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500"],
    },
}

# Metric -> (axis label, title). Driven by what the advisor asked for.
METRIC_LABELS = {
    "loss_total": ("training loss (total)", "Training loss vs epoch"),
    "AP": ("AP@[.50:.95]", "AP@[.50:.95] vs epoch"),
    "AP50": ("AP50", "AP50 vs epoch"),
    "AP75": ("AP75", "AP75 vs epoch"),
    "loss_box_reg": ("loss_box_reg", "Box-regression loss"),
    "loss_classifier": ("loss_classifier", "Classifier loss"),
    "loss_objectness": ("loss_objectness", "Objectness loss"),
    "loss_rpn_box_reg": ("loss_rpn_box_reg", "RPN box-reg loss"),
}

DEFAULT_CAPTION = (
    "dev-val (284 imgs), mean +/- 1 std over seeds. CONFOUND: pretrained = "
    "FrozenBN, from-scratch = GroupNorm -- prior and norm are coupled."
)


class Arm:
    """One comparison arm: a label, its loaded runs, and the model it holds."""

    def __init__(self, label, runs, model):
        self.label = label
        self.runs = runs  # list of (run_id, DataFrame)
        self.model = model

    @property
    def n(self):
        return len(self.runs)


def load_run(run_dir):
    """Load one run's per-epoch log into a DataFrame (epoch made 1-based).

    Returns (run_id, DataFrame, model_name). model_name comes from args.json
    (authoritative, robust to folder renames) or None if absent.
    """
    run_dir = Path(run_dir)
    log_path = run_dir / "train_log.jsonl"
    rows = []
    with log_path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"empty log: {log_path}")
    df = pd.DataFrame(rows)
    # Log epoch is 0-based; plot against 1..N.
    df["epoch"] = (df["epoch"] + 1) if "epoch" in df else range(1, len(df) + 1)

    model = None
    args_path = run_dir / "args.json"
    if args_path.exists():
        try:
            model = json.loads(args_path.read_text()).get("model")
        except (json.JSONDecodeError, OSError):
            model = None
    return run_dir.name, df, model


def collect_arm(spec, runs_root):
    """Expand one ``label=glob`` spec into an Arm, asserting homogeneity.

    Globs are resolved relative to runs_root unless already absolute. A run dir
    counts only if it holds a train_log.jsonl. If the matched runs carry more
    than one distinct args.json model, a warning is printed -- the glob is too
    loose (e.g. it swept in the quick probe or the other arm).
    """
    if "=" not in spec:
        raise SystemExit(f"--arm must be 'label=glob', got: {spec!r}")
    label, pattern = spec.split("=", 1)
    label, pattern = label.strip(), pattern.strip()

    base = Path(pattern)
    pattern = str(base if base.is_absolute() else Path(runs_root) / base)

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
        raise SystemExit(f"[{label}] no runs with train_log.jsonl match: {pattern}")
    if len(models) > 1:
        print(f"[warn] arm '{label}' mixes models {sorted(models)} -- "
              f"glob too loose? Each arm should be one model.", file=sys.stderr)

    print(f"[arm] {label}: {len(runs)} runs, model={sorted(models) or 'unknown'}")
    return Arm(label, runs, sorted(models)[0] if models else None)


def aggregate(arm, metric):
    """Per-epoch mean / std / n over an arm's runs for one metric.

    Sample std (ddof=1); epochs present in fewer than 2 runs get std = NaN
    (band collapses there rather than lying). Runs missing the metric are
    skipped for that metric only.
    """
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
    """Draw one metric for all arms onto ax: mean line + std band (+ seeds)."""
    for i, arm in enumerate(arms):
        color = colors["series"][i % len(colors["series"])]
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
        ax.plot(agg.index, agg["mean"], color=color, linewidth=2.0,
                label=f"{arm.label} (n={arm.n})", zorder=3)
    xlabel = "epoch"
    ylabel, _ = METRIC_LABELS.get(metric, (metric, metric))
    return xlabel, ylabel


def figure_for_metric(arms, metric, colors, cfg):
    """One standalone figure for a single metric. Returns the Figure."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=cfg["dpi"])
    fig.patch.set_facecolor(colors["surface"])
    ax.set_facecolor(colors["surface"])

    xlabel, ylabel = draw_metric(ax, arms, metric, colors,
                                 show_seeds=cfg["show_seeds"])
    _, title = METRIC_LABELS.get(metric, (metric, metric))
    _style_axes(ax, colors, xlabel=xlabel, ylabel=ylabel, title=title)
    if len(arms) >= 2 or metric != "loss_total":
        ax.legend(frameon=False, fontsize=9, labelcolor=colors["primary"])

    fig.text(0.01, 0.01, cfg["caption"], fontsize=7, color=colors["muted"],
             ha="left", va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    return fig


def figure_ap_panel(arms, colors, cfg):
    """Combined AP | AP50 | AP75 row sharing the epoch axis."""
    metrics = ["AP", "AP50", "AP75"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), dpi=cfg["dpi"],
                             sharex=True)
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
    """2x2 panel of the four head/RPN loss terms, if present in the logs."""
    comps = ["loss_classifier", "loss_box_reg",
             "loss_objectness", "loss_rpn_box_reg"]
    present = [m for m in comps
               if any(m in df for arm in arms for _, df in arm.runs)]
    if not present:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), dpi=cfg["dpi"],
                             sharex=True)
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
    p.add_argument("--arm", action="append", required=True, metavar="LABEL=GLOB",
                   help="Arm as 'label=glob' (repeatable). First arm = color "
                        "slot 1, etc. Glob is relative to --runs-root.")
    p.add_argument("--runs-root", default=".",
                   help="Base dir that relative arm globs resolve against.")
    p.add_argument("--out-dir", type=Path, default=Path("runs/plots"))
    p.add_argument("--metrics", default="loss_total,AP,AP50,AP75",
                   help="Comma list of per-epoch metrics to plot individually.")
    p.add_argument("--show-seeds", action="store_true",
                   help="Draw faint per-seed lines behind each mean.")
    p.add_argument("--loss-components", action="store_true",
                   help="Also emit the 2x2 head/RPN loss breakdown panel.")
    p.add_argument("--no-ap-panel", action="store_true",
                   help="Skip the combined AP|AP50|AP75 row.")
    p.add_argument("--dark", action="store_true", help="Dark-surface figures.")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--formats", default="png,svg",
                   help="Comma list of output formats (png,svg,pdf).")
    p.add_argument("--caption", default=DEFAULT_CAPTION,
                   help="Footnote stamped on every figure (confound note).")
    return p.parse_args()


def main():
    args = parse_args()
    colors = PALETTE["dark" if args.dark else "light"]
    plt.rcParams["font.family"] = ["DejaVu Sans"]  # consistent, always present

    arms = [collect_arm(spec, args.runs_root) for spec in args.arm]
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    cfg = {"dpi": args.dpi, "show_seeds": args.show_seeds,
           "caption": args.caption}

    for metric in metrics:
        fig = figure_for_metric(arms, metric, colors, cfg)
        save(fig, args.out_dir, f"curve_{metric}", formats)

    if not args.no_ap_panel and {"AP", "AP50", "AP75"}.issubset(metrics):
        save(figure_ap_panel(arms, colors, cfg), args.out_dir,
             "panel_ap", formats)

    if args.loss_components:
        fig = figure_loss_components(arms, colors, cfg)
        if fig is not None:
            save(fig, args.out_dir, "panel_loss_components", formats)

    print("[plot] done")


if __name__ == "__main__":
    main()
