"""Plot Precision-Recall curves per class, one series per model/arm.

The defensible detection analog of a ROC curve. ROC needs a fixed count of
negatives for its FPR axis; in detection the background is unbounded, so FPR is
ill-defined. PR has no such problem, and its area IS the COCO AP.

Aggregation over seeds is exact: COCOeval scores precision on a FIXED 101-point
recall grid, so precision at each recall level is averaged across seeds with no
interpolation. Each arm is a mean PR curve with a +/-1 std band over its seeds.

Identity (colour, legend name) comes from model_style, keyed off the MODEL, so a
model reads the same here as in the other figures. Arms are
``exp_id:model_nN_tag`` (resolved to the per-seed predictions.json glob under
--runs-base); legacy ``label=glob`` is still accepted.

The confound footer is configurable (--caption); default is neutral.

Run (PYTHONPATH=/workspace/src, pycocotools available -- i.e. gpu_env):
    python plot_pr.py --runs-base /workspace/runs \
        --arm cross_g0:retinanet_n96_s8 --arm cross_g1:ssd_n96_s8 \
        --iou 0.5 --out-dir runs/plots
"""

import argparse
import json
import sys
from glob import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# eval_ap lives beside this script; reuse its COCO builders (one ruler).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_ap import (  # noqa: E402
    build_coco_gt,
    build_detections,
    DEFAULT_CLASS_NAMES,
)
import model_style as ms  # noqa: E402

PALETTE = {
    "light": {"surface": "#fcfcfb", "primary": "#0b0b0b", "secondary": "#52514e",
              "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7"},
    "dark": {"surface": "#1a1a19", "primary": "#f0efec", "secondary": "#c3c2b7",
             "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835"},
}

DEFAULT_CAPTION = ("mean +/- 1 std over seeds. AP = area under the mean curve.")


def load_precision(json_path, class_names=None):
    """Run COCOeval on one predictions.json; return its precision tensor + axes."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    doc = json.loads(Path(json_path).read_text())
    frames = doc["frames"]
    class_names = class_names or doc.get("meta", {}).get(
        "class_names", DEFAULT_CLASS_NAMES)
    gt_dict = build_coco_gt(frames, class_names)
    dets = build_detections(frames)
    id_to_name = {c["id"]: c["name"] for c in gt_dict["categories"]}

    coco_gt = COCO()
    coco_gt.dataset = gt_dict
    coco_gt.createIndex()
    if not dets:
        raise ValueError(f"{json_path}: no detections -> no PR curve")
    coco_dt = coco_gt.loadRes(dets)
    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev.evaluate()
    ev.accumulate()
    return (ev.eval["precision"], ev.params.recThrs, ev.params.iouThrs,
            coco_gt.getCatIds(), id_to_name)


def pr_curve(precision, k, iou_mode, iou_thrs):
    arr = precision[:, :, k, 0, 2].copy()
    arr[arr < 0] = 0.0
    if iou_mode == "mean":
        return arr.mean(axis=0)
    t = int(np.argmin(np.abs(iou_thrs - float(iou_mode))))
    return arr[t]


def collect_arm(style, runs_base, iou_mode):
    """Resolved arm -> dict with style, recThrs, names, per-class stacked curves."""
    pattern = style.pred_glob(runs_base)
    paths = sorted(glob(pattern))
    if not paths:
        raise SystemExit(f"[{style.label}] no predictions.json match: {pattern}")
    per_class, rec, names = {}, None, None
    for p in paths:
        precision, recThrs, iouThrs, cat_ids, id_to_name = load_precision(p)
        rec = recThrs
        names = [id_to_name[c] for c in cat_ids]
        for k, cid in enumerate(cat_ids):
            per_class.setdefault(id_to_name[cid], []).append(
                pr_curve(precision, k, iou_mode, iouThrs))
    curves = {name: np.vstack(rows) for name, rows in per_class.items()}
    print(f"[arm] {style.label}: {len(paths)} seeds, classes={names}")
    return {"style": style, "rec": rec, "names": names, "curves": curves}


def _style_axes(ax, colors, *, title):
    ax.set_title(title, color=colors["primary"], fontsize=12, pad=10, loc="left")
    ax.set_xlabel("recall", color=colors["secondary"], fontsize=10)
    ax.set_ylabel("precision", color=colors["secondary"], fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, color=colors["grid"], linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(colors["axis"])
    ax.tick_params(colors=colors["muted"], labelsize=9)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", action="append", dest="arms", required=True,
                   metavar="EXP:MODEL_nN_tag",
                   help="exp_id:model_nN_tag (or legacy label=glob); repeatable.")
    p.add_argument("--arm-name", action="append", dest="arm_names", default=[])
    p.add_argument("--arm-color", action="append", dest="arm_colors", default=[])
    p.add_argument("--arm-linestyle", action="append", dest="arm_linestyles", default=[])
    p.add_argument("--runs-base", default="/workspace/runs")
    p.add_argument("--iou", default="0.5", help="0.5, 0.75, or 'mean'.")
    p.add_argument("--out-dir", type=Path, default=Path("runs/plots"))
    p.add_argument("--dark", action="store_true")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--formats", default="png,svg")
    p.add_argument("--caption", default=DEFAULT_CAPTION)
    args = p.parse_args()

    iou_mode = "mean" if args.iou.lower() == "mean" else float(args.iou)
    colors = PALETTE["dark" if args.dark else "light"]
    plt.rcParams["font.family"] = ["DejaVu Sans"]

    styles = ms.resolve_arms(args.arms, args.arm_names, args.arm_colors,
                             args.arm_linestyles)
    arms = [collect_arm(s, args.runs_base, iou_mode) for s in styles]
    names = arms[0]["names"]

    fig, axes = plt.subplots(1, len(names), figsize=(4.7 * len(names), 4.6),
                             dpi=args.dpi, squeeze=False)
    axes = axes[0]
    fig.patch.set_facecolor(colors["surface"])

    for ax, name in zip(axes, names):
        ax.set_facecolor(colors["surface"])
        for arm in arms:
            color, ls = arm["style"].color, arm["style"].linestyle
            stack = arm["curves"].get(name)
            if stack is None:
                continue
            mean = stack.mean(axis=0)
            std = stack.std(axis=0, ddof=1) if stack.shape[0] > 1 else None
            ap = mean.mean()
            if std is not None:
                ax.fill_between(arm["rec"], mean - std, mean + std,
                                color=color, alpha=0.16, linewidth=0, zorder=2)
            ax.plot(arm["rec"], mean, color=color, linestyle=ls, linewidth=2.0,
                    zorder=3,
                    label=f"{arm['style'].label} (AP={ap:.3f}, n={stack.shape[0]})")
        iou_txt = "mean@[.50:.95]" if iou_mode == "mean" else f"@{iou_mode:.2f}"
        _style_axes(ax, colors, title=f"{name} - PR {iou_txt}")
    axes[0].legend(frameon=False, fontsize=9, labelcolor=colors["primary"],
                   loc="lower left")

    fig.text(0.01, 0.01, args.caption, fontsize=7, color=colors["muted"],
             ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    iou_tag = "mean" if iou_mode == "mean" else f"{iou_mode:.2f}".replace(".", "")
    for ext in (f.strip() for f in args.formats.split(",") if f.strip()):
        path = args.out_dir / f"pr_per_class_iou{iou_tag}.{ext}"
        fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
        print(f"[plot] wrote {path}")
    plt.close(fig)
    print("[plot] done")


if __name__ == "__main__":
    main()