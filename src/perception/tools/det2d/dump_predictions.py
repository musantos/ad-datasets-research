"""Run a trained detector once over a split and dump predictions + GT to JSON.

Decouples inference from visualization and metrics (decision E): the model runs
exactly once and produces a self-contained artifact that both the comparison
video and the future pycocotools AP module consume -- no checkpoint and no torch
needed downstream.

Each frame image is also exported as a PNG next to the JSON, so the renderer is
fully standalone. This duplicates pixels -- fine for the small warm-up ladder;
swap the container (.pt / COCO-json with image refs) once the dataset scales.

Run (PYTHONPATH=/workspace/src):

    python .../dump_predictions.py \
        --ckpt runs/<run_id>/last.pt \
        --cache-root /workspace/datasets/waymo/perception_cache \
        --split val --dump-score-thresh 0.05 --out-dir runs/<run_id>/dump_val
"""

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

# --- adjust these two imports to your package layout -----------------------
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
# ---------------------------------------------------------------------------

# Must cover every model the trainer can produce (load_model rebuilds from the
# checkpoint's `model` key). Keep in sync with train_vision_model.MODELS.
MODELS = {
    "faster_rcnn": build_faster_rcnn,
    "faster_rcnn_pt": build_faster_rcnn_pretrained,
    "retinanet": build_retinanet,
    "retinanet_pt": build_retinanet_pretrained,
    "ssd": build_ssd,
    "ssd_pt": build_ssd_pretrained,
}
CLASS_NAMES = {1: "Vehicle", 2: "Pedestrian", 3: "Cyclist"}


def load_model(ckpt_path, device, *, min_size=None, max_size=None):
    """Rebuild the detector from a checkpoint and load its weights.

    weights_only=False is required: the checkpoint stores an `args` dict that
    may contain non-tensor objects (e.g. pathlib.Path). Safe here -- the file is
    our own training output.
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


def save_png(image, path):
    """[C,H,W] float in [0,1] -> PNG on disk. Returns (width, height)."""
    arr = (image.permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()
    # Pillow 13 (Oct 2025) dropped the `mode=` arg from fromarray; the mode is
    # inferred from the array shape/dtype (HxWx3 uint8 -> RGB), so pass none.
    img = Image.fromarray(arr)
    img.save(path)
    return img.width, img.height


def parse_args():
    p = argparse.ArgumentParser(description="Dump detector predictions + GT to JSON.")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--dump-score-thresh", type=float, default=0.05,
                   help="Keep predictions above this; video filters higher.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Default: <ckpt_dir>/dump_<split>.")
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

    out_dir = args.out_dir or args.ckpt.parent / f"dump_{args.split}"
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"[dump] model={name} split={args.split} frames={n} "
          f"thr={args.dump_score_thresh} device={device} -> {out_dir}")

    frames = []
    for i in range(n):
        image, target = ds[i]
        preds = model([image.to(device)])[0]
        keep = preds["scores"] >= args.dump_score_thresh

        png_name = f"{args.split}_{i:05d}.png"
        w, h = save_png(image, frames_dir / png_name)

        frames.append({
            "image_id": _as_py(target.get("image_id", i)),
            "segment": _as_py(target.get("segment")),
            "frame_idx": int(_as_py(target.get("frame_idx", i))),
            "camera": _as_py(target.get("camera")),
            "width": w, "height": h,
            "image_file": f"frames/{png_name}",
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
        if (i % 20) == 0:
            print(f"  {i:4d}/{n}")

    doc = {
        "meta": {
            "model": name,
            "ckpt": str(args.ckpt),
            "split": args.split,
            "dump_score_thresh": args.dump_score_thresh,
            "min_size": ms, "max_size": xs,
            "class_names": {str(k): v for k, v in CLASS_NAMES.items()},
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "frames": frames,
    }
    json_path = out_dir / "predictions.json"
    json_path.write_text(json.dumps(doc))
    print(f"[dump] wrote {len(frames)} frames -> {json_path}")


if __name__ == "__main__":
    main()