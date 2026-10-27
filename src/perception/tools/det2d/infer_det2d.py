"""Run a trained detector over a split and report COCO AP -- scalable eval path.

Model-agnostic: the detector is rebuilt from the ``model`` key saved in the
checkpoint via the shared MODELS registry, so this one script scores every
architecture in the study (faster_rcnn, retinanet, ssd -- scratch and
pretrained). Renamed from ``inference_frcnn.py``: the old name predated the
multi-architecture registry and was misleading -- it was never FRCNN-only.

Unlike dump_predictions.py (which copies every frame to a PNG for the renderer),
this writes NO pixels: it runs inference, feeds predictions + GT straight through
the shared eval_ap ruler, and reports AP@[.50:.95] overall and per class. A lean
predictions.json (boxes/scores/labels + GT, no image copies) is written by
default, so the exact run can be re-scored later without re-running the model --
the same decoupling as dump_predictions, minus the pixel duplication that does
not scale to full validation packs.

For GT-vs-pred videos, use dump_predictions.py on a small subset; this script is
the metrics path for large splits.

--split is a free-form subdir name (not restricted to train/val), so a fresh
validation pack dropped at <cache-root>/<name> is usable without renaming.

Run (PYTHONPATH=/workspace/src):

    python .../infer_det2d.py \
        --ckpt runs/<run_id>/best.pt \
        --cache-root /workspace/datasets/waymo/perception_cache \
        --split val
"""

import argparse
import json
import time
from pathlib import Path

import torch

# --- adjust these imports to your package layout ---------------------------
from perception.waymo.tools.dataset import PerceptionCacheDataset
from perception.models.faster_rcnn import (
    build_faster_rcnn,
    build_faster_rcnn_pretrained,
)
from perception.models.retinanet import (
    build_retinanet,
    build_retinanet_pretrained,
)
from perception.models.ssd import (
    build_ssd,
    build_ssd_pretrained,
)
from perception.models.fcos import (
    build_fcos,
    build_fcos_pretrained,
)
from perception.tools.det2d.eval_ap import evaluate_predictions  # one ruler
# ---------------------------------------------------------------------------

# Must cover every model the trainer can produce, since load_model() rebuilds
# the detector from the `model` key saved in the checkpoint. Keep in sync with
# train_vision_model.MODELS, or a non-FRCNN checkpoint fails with KeyError here.
MODELS = {
    "faster_rcnn": build_faster_rcnn,
    "faster_rcnn_pt": build_faster_rcnn_pretrained,
    "retinanet": build_retinanet,
    "retinanet_pt": build_retinanet_pretrained,
    "ssd": build_ssd,
    "ssd_pt": build_ssd_pretrained,
    "fcos": build_fcos,
    "fcos_pt": build_fcos_pretrained,
}
CLASS_NAMES = {1: "Vehicle", 2: "Pedestrian", 3: "Cyclist"}


def load_model(ckpt_path, device, *, min_size=None, max_size=None):
    """Rebuild the detector from a checkpoint and load its weights.

    The builder is selected from the ``model`` key saved in the checkpoint args,
    so from-scratch and pretrained checkpoints both load without a CLI flag.
    weights_only=False is required: the saved args dict holds non-tensor objects
    (e.g. pathlib.Path). Safe here -- the file is our own training output.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved = ckpt.get("args", {})
    name = saved.get("model", "faster_rcnn")
    ms = min_size or int(saved.get("min_size", 800))
    xs = max_size or int(saved.get("max_size", 1333))
    model = MODELS[name](min_size=ms, max_size=xs)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    return model, name, ms, xs


def _as_py(v):
    """Unwrap a tensor / numpy scalar to a plain Python value."""
    if torch.is_tensor(v):
        return v.item() if v.ndim == 0 else v.tolist()
    return v


def _list(v):
    """Tensor -> nested python list; None/missing -> []."""
    if v is None:
        return []
    return v.tolist() if torch.is_tensor(v) else list(v)


def parse_args():
    p = argparse.ArgumentParser(
        description="Infer over a split and report COCO AP (no pixel dump).")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--split", default="val",
                   help="Subdir under cache-root (free-form, e.g. val, "
                        "val_packs). Not restricted to train/val.")
    p.add_argument("--score-thresh", type=float, default=0.05,
                   help="Keep predictions above this before scoring. Must be "
                        "IDENTICAL across arms for a fair comparison; lower "
                        "(down to 0.0) only raises recall-limited AP slightly.")
    p.add_argument("--out", type=Path, default=None,
                   help="Lean predictions.json path. Default: "
                        "<ckpt_dir>/infer_<split>/predictions.json. "
                        "Use --skip-dump to not write it.")
    p.add_argument("--metrics-out", type=Path, default=None,
                   help="Optional: also write the metrics dict as JSON here.")
    p.add_argument("--skip-dump", action="store_true",
                   help="Do not persist predictions.json (metrics only).")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress the pycocotools 12-line summary table.")
    p.add_argument("--min-size", type=int, default=None, help="Override ckpt.")
    p.add_argument("--max-size", type=int, default=None, help="Override ckpt.")
    p.add_argument("--limit", type=int, default=None, help="Cap #frames (debug).")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA unavailable; using CPU")
        args.device = "cpu"
    device = torch.device(args.device)

    model, name, ms, xs = load_model(args.ckpt, device,
                                     min_size=args.min_size, max_size=args.max_size)
    ds = PerceptionCacheDataset(args.cache_root / args.split, drop_empty=False)
    n = len(ds) if args.limit is None else min(args.limit, len(ds))

    print(f"[infer] model={name} split={args.split} frames={n} "
          f"thr={args.score_thresh} device={device}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()

    frames = []
    for i in range(n):
        image, target = ds[i]
        preds = model([image.to(device)])[0]
        keep = preds["scores"] >= args.score_thresh

        _, h, w = image.shape  # [C,H,W]; no PNG needed for COCO bbox eval
        frames.append({
            "image_id": _as_py(target.get("image_id", i)),
            "segment": _as_py(target.get("segment")),
            "frame_idx": int(_as_py(target.get("frame_idx", i))),
            "camera": _as_py(target.get("camera")),
            "width": int(w), "height": int(h),
            "gt": {
                "boxes": _list(target.get("boxes")),
                "labels": _list(target.get("labels")),
                "area": _list(target.get("area")),
                "iscrowd": _list(target.get("iscrowd")),
            },
            "pred": {
                "boxes": preds["boxes"][keep].cpu().tolist(),
                "labels": preds["labels"][keep].cpu().tolist(),
                "scores": preds["scores"][keep].cpu().tolist(),
            },
        })
        if (i % 50) == 0:
            print(f"  {i:5d}/{n}")

    dt = time.time() - t0
    peak_gb = (torch.cuda.max_memory_allocated(device) / 1024**3
               if device.type == "cuda" else 0.0)
    print(f"[infer] {n} frames in {dt:.1f}s ({n / dt:.1f} img/s), "
          f"peak VRAM {peak_gb:.2f} GB")

    class_names = {str(k): v for k, v in CLASS_NAMES.items()}
    metrics = evaluate_predictions(frames, class_names, verbose=not args.quiet)

    print(f"\n[AP] {name}/{args.split}   AP@[.50:.95]={metrics['AP']:.4f}   "
          f"AP50={metrics['AP50']:.4f}   AP75={metrics['AP75']:.4f}")
    for cname, v in metrics["per_class"].items():
        is_nan = isinstance(v, float) and v != v
        print(f"     {cname:12s} AP={'  n/a (no GT)' if is_nan else f'{v:.4f}'}")

    if not args.skip_dump:
        out = args.out or args.ckpt.parent / f"infer_{args.split}" / "predictions.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        doc = {
            "meta": {
                "model": name, "ckpt": str(args.ckpt), "split": args.split,
                "score_thresh": args.score_thresh,
                "min_size": ms, "max_size": xs,
                "class_names": class_names,
                "frames": n, "peak_vram_gb": round(peak_gb, 3),
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "frames": frames,
        }
        out.write_text(json.dumps(doc))
        print(f"[infer] wrote lean predictions -> {out}")

    if args.metrics_out:
        args.metrics_out.write_text(json.dumps(metrics, indent=2))
        print(f"[infer] wrote metrics -> {args.metrics_out}")


if __name__ == "__main__":
    main()