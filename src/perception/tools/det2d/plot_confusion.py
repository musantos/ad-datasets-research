"""Plot detection confusion matrices (per arm), pretrained vs from-scratch.

Consumes the same predictions.json dumps as plot_pr.py; needs no pycocotools --
IoU matching is done here. For each image, predictions are matched to GT greedily
by descending score, class-agnostically, at IoU >= --iou (so a box that localizes
an object but mislabels it shows up as off-diagonal confusion, not as a miss).

The matrix is (K+1) x (K+1): the K classes plus a ``background`` row/column.
  rows = ground-truth class (+ background),
  cols = predicted class    (+ background).
  M[g, p]  (g, p real classes): a true-g object detected and labelled p
  M[g, bg]: a true-g object with no matching prediction -> false negative (miss)
  M[bg, p]: a prediction of class p matching no GT -> false positive
  M[bg, bg]: undefined -- the detection "true negative" is the unbounded image
             background, so this cell is left blank (not zero).

Unlike AP / PR (which sweep the score threshold), a confusion matrix is read at
one operating point, so predictions are first filtered at --score-thresh. That
threshold, like the comparison itself, must match across arms; it is stamped on
the figure.

Aggregation: each seed's matrix is row-normalized, then averaged across the
arm's seeds (the distribution is the result). Cells show the mean; with more
than one seed a small +/-std is annotated under it.

Confound, stated on the figure: pretrained uses FrozenBN, from-scratch uses
GroupNorm -- prior and normalization are coupled, not isolated.

Run (pure numpy + matplotlib; no torch, no pycocotools -- runs in cpu_env too):

    python plot_confusion.py \
        --arm "pretrained=runs/faster_rcnn_pt_n48_s8_seed*/infer_val/predictions.json" \
        --arm "from-scratch=runs/faster_rcnn_n48_s8_seed*/infer_val/predictions.json" \
        --iou 0.5 --score-thresh 0.5 --out-dir runs/plots
"""

import argparse
import json
from glob import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import numpy as np  # noqa: E402

DEFAULT_CLASS_NAMES = {"1": "Vehicle", "2": "Pedestrian", "3": "Cyclist"}

PALETTE = {
    "light": {"surface": "#fcfcfb", "primary": "#0b0b0b", "secondary": "#52514e",
              "muted": "#898781", "grid": "#e1e0d9",
              "seq": ["#fcfcfb", "#cde2fb", "#6da7ec", "#2a78d6", "#104281"]},
    "dark": {"surface": "#1a1a19", "primary": "#f0efec", "secondary": "#c3c2b7",
             "muted": "#898781", "grid": "#2c2c2a",
             "seq": ["#1a1a19", "#184f95", "#2a78d6", "#6da7ec", "#cde2fb"]},
}

DEFAULT_CAPTION = (
    "dev-val (284 imgs). IoU-match >= {iou}, score >= {score}, row-normalized, "
    "mean over {n} seeds. rows = true, cols = predicted. CONFOUND: pretrained = "
    "FrozenBN, from-scratch = GroupNorm -- prior and norm are coupled."
)


def iou_xyxy(a, b):
    """IoU of two xyxy boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def confusion_counts(frames, class_ids, iou_thr, score_thr):
    """(K+1)x(K+1) raw count matrix for one predictions.json's frames.

    rows = GT class (+ background last), cols = predicted class (+ background).
    Greedy, class-agnostic matching by descending score.
    """
    id2idx = {cid: i for i, cid in enumerate(class_ids)}
    bg = len(class_ids)
    M = np.zeros((bg + 1, bg + 1), dtype=float)

    for fr in frames:
        gt_boxes, gt_labels = fr["gt"]["boxes"], fr["gt"]["labels"]
        pred = fr["pred"]
        order = sorted(range(len(pred["scores"])),
                       key=lambda k: pred["scores"][k], reverse=True)
        gt_used = [False] * len(gt_boxes)

        for k in order:
            if pred["scores"][k] < score_thr:
                continue
            pl = int(pred["labels"][k])
            if pl not in id2idx:
                continue  # label outside declared classes -> ignore
            pbox = pred["boxes"][k]
            best_iou, best_g = 0.0, -1
            for gi, gbox in enumerate(gt_boxes):
                if gt_used[gi]:
                    continue
                v = iou_xyxy(pbox, gbox)
                if v >= iou_thr and v > best_iou:
                    best_iou, best_g = v, gi
            if best_g >= 0:
                g = id2idx[int(gt_labels[best_g])]
                M[g, id2idx[pl]] += 1
                gt_used[best_g] = True
            else:
                M[bg, id2idx[pl]] += 1  # false positive

        for gi, used in enumerate(gt_used):
            if not used:
                gl = int(gt_labels[gi])
                if gl in id2idx:
                    M[id2idx[gl], bg] += 1  # false negative (miss)
    return M


def row_normalize(M):
    """Row-normalize; zero-sum rows stay zero (no division)."""
    out = M.astype(float).copy()
    sums = out.sum(axis=1, keepdims=True)
    nz = sums.squeeze(-1) > 0
    out[nz] = out[nz] / sums[nz]
    return out


def collect_arm(spec, class_ids, iou_thr, score_thr):
    """'label=glob' -> (label, mean matrix, std matrix, n_seeds)."""
    if "=" not in spec:
        raise SystemExit(f"--arm must be 'label=glob', got: {spec!r}")
    label, pattern = (s.strip() for s in spec.split("=", 1))
    paths = sorted(glob(pattern))
    if not paths:
        raise SystemExit(f"[{label}] no predictions.json match: {pattern}")

    mats = []
    for p in paths:
        frames = json.loads(Path(p).read_text())["frames"]
        mats.append(row_normalize(confusion_counts(
            frames, class_ids, iou_thr, score_thr)))
    stack = np.stack(mats)  # [n, K+1, K+1]
    print(f"[arm] {label}: {len(paths)} seeds")
    return (label, stack.mean(axis=0),
            stack.std(axis=0, ddof=1) if len(paths) > 1 else None, len(paths))


def draw_matrix(ax, mean, std, labels, colors, cmap, *, title, first=True):
    """Render one row-normalized confusion matrix as an annotated heatmap.

    ``first`` controls the shared y axis: only the leftmost panel carries the
    "true" label and the class y-ticks (identical across panels otherwise).
    """
    n = len(labels)
    masked = np.ma.array(mean, mask=False)
    masked.mask[n - 1, n - 1] = True  # background/background is undefined
    cmap = cmap.copy()
    cmap.set_bad(colors["grid"])
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="equal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9,
                       color=colors["secondary"])
    if first:
        ax.set_yticklabels(labels, fontsize=9, color=colors["secondary"])
        ax.set_ylabel("true", color=colors["secondary"], fontsize=10)
    else:
        ax.set_yticklabels([])
    ax.set_xlabel("predicted", color=colors["secondary"], fontsize=10)
    ax.set_title(title, color=colors["primary"], fontsize=12, pad=10, loc="left")
    ax.tick_params(colors=colors["muted"], length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color=colors["surface"], linewidth=2)
    ax.tick_params(which="minor", length=0)

    for i in range(n):
        for j in range(n):
            if i == n - 1 and j == n - 1:
                ax.text(j, i, "--", ha="center", va="center",
                        color=colors["muted"], fontsize=9)
                continue
            val = mean[i, j]
            txt_color = "#ffffff" if val > 0.55 else colors["primary"]
            label = f"{val:.2f}"
            if std is not None:
                label += f"\n±{std[i, j]:.2f}"
            ax.text(j, i, label, ha="center", va="center",
                    color=txt_color, fontsize=8)
    return im


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", action="append", required=True, metavar="LABEL=GLOB",
                   help="Arm as 'label=glob' over predictions.json (repeatable).")
    p.add_argument("--iou", type=float, default=0.5,
                   help="IoU threshold for matching pred to GT.")
    p.add_argument("--score-thresh", type=float, default=0.5,
                   help="Keep predictions above this (operating point). Must "
                        "match across arms.")
    p.add_argument("--out-dir", type=Path, default=Path("runs/plots"))
    p.add_argument("--dark", action="store_true")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--formats", default="png,svg")
    p.add_argument("--caption", default=None)
    args = p.parse_args()

    class_names = DEFAULT_CLASS_NAMES
    class_ids = sorted(int(k) for k in class_names)
    labels = [class_names[str(c)] for c in class_ids] + ["background"]
    colors = PALETTE["dark" if args.dark else "light"]
    cmap = LinearSegmentedColormap.from_list("seq", colors["seq"])
    plt.rcParams["font.family"] = ["DejaVu Sans"]

    # Fail-fast on empty arms before any matching.
    for spec in args.arm:
        _, pattern = (s.strip() for s in spec.split("=", 1))
        if not glob(pattern):
            raise SystemExit(f"no predictions.json match: {pattern}")

    arms = [collect_arm(spec, class_ids, args.iou, args.score_thresh)
            for spec in args.arm]

    fig, axes = plt.subplots(1, len(arms), figsize=(5.4 * len(arms), 5.0),
                             dpi=args.dpi, squeeze=False)
    axes = axes[0]
    fig.patch.set_facecolor(colors["surface"])
    im = None
    for idx, (ax, (label, mean, std, n)) in enumerate(zip(axes, arms)):
        ax.set_facecolor(colors["surface"])
        im = draw_matrix(ax, mean, std, labels, colors, cmap, title=label,
                         first=(idx == 0))

    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("row-normalized rate", color=colors["secondary"], fontsize=9)
    cbar.ax.tick_params(colors=colors["muted"], labelsize=8)
    cbar.outline.set_visible(False)

    n_seeds = arms[0][3]
    caption = args.caption or DEFAULT_CAPTION.format(
        iou=args.iou, score=args.score_thresh, n=n_seeds)
    fig.subplots_adjust(bottom=0.2)
    fig.text(0.01, 0.015, caption, fontsize=7, color=colors["muted"],
             ha="left", va="bottom")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"confusion_iou{str(args.iou).replace('.', '')}_s{str(args.score_thresh).replace('.', '')}"
    for ext in (f.strip() for f in args.formats.split(",") if f.strip()):
        path = args.out_dir / f"{stem}.{ext}"
        fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
        print(f"[plot] wrote {path}")
    plt.close(fig)
    print("[plot] done")


if __name__ == "__main__":
    main()
