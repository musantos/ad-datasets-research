"""Plot Precision-Recall curves per class, pretrained vs from-scratch.

The defensible detection analog of a ROC curve. ROC needs a fixed count of
negatives for its FPR axis; in detection the background is unbounded, so FPR is
ill-defined. PR has no such problem -- precision and recall both read only off
what was detected and what should have been -- and its area IS the COCO AP the
curves plot numerically elsewhere. So this is the "ROC" figure, done right.

Aggregation over seeds is exact here, not approximate: COCOeval scores precision
on a FIXED 101-point recall grid (``recThrs``), so precision at each recall level
can be averaged across seeds with no interpolation. Each arm is therefore drawn
as a mean PR curve with a +/-1 std band over its seeds -- the project invariant
("the distribution is the result"), carried from the training curves into PR.

Reuses eval_ap's COCO builders (one ruler: identical xyxy->xywh conversion and
category declaration), then reads ``ev.eval['precision']`` directly -- the slice
``precision[t, :, k, 0, 2]`` is precision over recall for IoU threshold t, class
k, area=all, maxDet=100. Unachieved recall levels (-1) are shown as precision 0.

Confound, stated on every figure: pretrained uses FrozenBN, from-scratch uses
GroupNorm -- prior and normalization are coupled, not isolated.

Run (PYTHONPATH=/workspace/src, pycocotools available -- i.e. gpu_env):

    python plot_pr.py \
        --arm "pretrained=runs/faster_rcnn_pt_n48_s8_seed*/infer_val/predictions.json" \
        --arm "from-scratch=runs/faster_rcnn_n48_s8_seed*/infer_val/predictions.json" \
        --iou 0.5 \
        --out-dir runs/plots

Each arm's glob must match the predictions.json files written by
inference_frcnn.py (one per seed), produced at the SAME --score-thresh for both
arms. ``--iou`` is 0.5, 0.75, or "mean" (averaged over .50:.95).
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

# eval_ap lives beside this script (tools/det2d/); reuse its COCO builders so PR
# is computed through the exact same ruler as the AP numbers.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_ap import (  # noqa: E402
    build_coco_gt,
    build_detections,
    DEFAULT_CLASS_NAMES,
)

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

DEFAULT_CAPTION = (
    "dev-val (284 imgs), mean +/- 1 std over seeds. AP = area under the mean "
    "curve. CONFOUND: pretrained = FrozenBN, from-scratch = GroupNorm -- prior "
    "and norm are coupled."
)


def load_precision(json_path, class_names=None):
    """Run COCOeval on one predictions.json; return its precision tensor + axes.

    Returns (precision[T,R,K,A,M], recThrs, iouThrs, cat_ids, id_to_name).
    pycocotools is imported lazily inside here (gpu_env only).
    """
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
    """Precision over the recall grid for class index k at the chosen IoU.

    iou_mode: 0.5 / 0.75 (single threshold) or "mean" (averaged over .50:.95).
    -1 (unachieved recall) is shown as precision 0. Area under the returned
    curve equals the COCO AP for that class/IoU.
    """
    arr = precision[:, :, k, 0, 2].copy()  # [T, R]
    arr[arr < 0] = 0.0
    if iou_mode == "mean":
        return arr.mean(axis=0)
    t = int(np.argmin(np.abs(iou_thrs - float(iou_mode))))
    return arr[t]


def collect_arm(spec, iou_mode):
    """Expand 'label=glob' into (label, per-class stacked PR arrays, meta).

    Returns dict with label, recThrs, names (ordered by cat id), and
    curves: {class_name: ndarray[n_seeds, 101]}.
    """
    if "=" not in spec:
        raise SystemExit(f"--arm must be 'label=glob', got: {spec!r}")
    label, pattern = (s.strip() for s in spec.split("=", 1))
    paths = sorted(glob(pattern))
    if not paths:
        raise SystemExit(f"[{label}] no predictions.json match: {pattern}")

    per_class, rec, names = {}, None, None
    for p in paths:
        precision, recThrs, iouThrs, cat_ids, id_to_name = load_precision(p)
        rec = recThrs
        names = [id_to_name[c] for c in cat_ids]
        for k, cid in enumerate(cat_ids):
            name = id_to_name[cid]
            per_class.setdefault(name, []).append(
                pr_curve(precision, k, iou_mode, iouThrs))

    curves = {name: np.vstack(rows) for name, rows in per_class.items()}
    print(f"[arm] {label}: {len(paths)} seeds, classes={names}")
    return {"label": label, "rec": rec, "names": names, "curves": curves}


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
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", action="append", required=True, metavar="LABEL=GLOB",
                   help="Arm as 'label=glob' over predictions.json (repeatable).")
    p.add_argument("--iou", default="0.5",
                   help="IoU threshold for the curve: 0.5, 0.75, or 'mean'.")
    p.add_argument("--out-dir", type=Path, default=Path("runs/plots"))
    p.add_argument("--dark", action="store_true")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--formats", default="png,svg")
    p.add_argument("--caption", default=DEFAULT_CAPTION)
    args = p.parse_args()

    iou_mode = "mean" if args.iou.lower() == "mean" else float(args.iou)
    colors = PALETTE["dark" if args.dark else "light"]
    plt.rcParams["font.family"] = ["DejaVu Sans"]

    arms = [collect_arm(spec, iou_mode) for spec in args.arm]
    names = arms[0]["names"]  # class order from the first arm's categories

    fig, axes = plt.subplots(1, len(names), figsize=(4.7 * len(names), 4.6),
                             dpi=args.dpi, squeeze=False)
    axes = axes[0]
    fig.patch.set_facecolor(colors["surface"])

    for ax, name in zip(axes, names):
        ax.set_facecolor(colors["surface"])
        for i, arm in enumerate(arms):
            color = colors["series"][i % len(colors["series"])]
            stack = arm["curves"].get(name)
            if stack is None:
                continue
            mean = stack.mean(axis=0)
            std = stack.std(axis=0, ddof=1) if stack.shape[0] > 1 else None
            ap = mean.mean()  # area under mean curve = COCO AP at this IoU
            if std is not None:
                ax.fill_between(arm["rec"], mean - std, mean + std,
                                color=color, alpha=0.16, linewidth=0, zorder=2)
            ax.plot(arm["rec"], mean, color=color, linewidth=2.0, zorder=3,
                    label=f"{arm['label']} (AP={ap:.3f}, n={stack.shape[0]})")
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
