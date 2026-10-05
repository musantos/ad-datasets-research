"""Render a GT-vs-prediction comparison video from a trained 2D detector.

For each frame of a (front-only) Waymo perception segment, draws the
ground-truth boxes from the cache in BLACK and the model's predicted boxes in
per-class colors, then writes one video per segment. A visual sanity tool for
the Stage-1 warm-up -- not a metric.

Reuses the same model registry as training (single source of truth). The model
is rebuilt from the checkpoint's saved args, so the video always matches how the
run was trained.

Run (PYTHONPATH=/workspace/src, perception package __init__.py in place):

    python /workspace/src/perception/models/make_gt_vs_pred_video.py \
        --ckpt runs/<run_id>/last.pt \
        --cache-root /workspace/datasets/waymo/perception_cache \
        --split val --score-thresh 0.5 --fps 5
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from perception.waymo.tools.dataset import PerceptionCacheDataset
from perception.models.train_vision_model import MODELS, CLASS_NAMES


GT_COLOR = (0, 0, 0)                 # black, as requested
PRED_COLORS = {                      # per-class, predictions only
    1: (0, 200, 0),                  # Vehicle   -> green
    2: (255, 140, 0),                # Pedestrian -> orange
    3: (0, 200, 255),                # Cyclist   -> cyan
}
PRED_FALLBACK = (255, 0, 255)        # magenta for unexpected labels


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def load_model(ckpt_path, device, *, min_size=None, max_size=None):
    """Rebuild the detector from a checkpoint and load its weights."""
    #ckpt = torch.load(ckpt_path, map_location=device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved = ckpt.get("args", {})
    name = saved.get("model", "faster_rcnn")
    ms = min_size or int(saved.get("min_size", 800))
    xs = max_size or int(saved.get("max_size", 1333))
    model = MODELS[name](min_size=ms, max_size=xs)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    return model, name


# ---------------------------------------------------------------------------
# Frame ordering: group records by segment, order by frame_idx
# ---------------------------------------------------------------------------
def build_order(ds, segment_filter=None):
    """Return {segment: [dataset_index, ...]} ordered temporally by frame_idx."""
    groups = {}
    for i in range(len(ds)):
        _, t = ds[i]
        seg = _as_py(t.get("segment"))
        if segment_filter is not None and str(seg) != str(segment_filter):
            continue
        frame = int(_as_py(t.get("frame_idx", i)))
        groups.setdefault(seg, []).append((frame, i))
    return {seg: [idx for _, idx in sorted(pairs)]
            for seg, pairs in groups.items()}


def _as_py(v):
    """Unwrap tensors / numpy scalars to a plain Python value."""
    if torch.is_tensor(v):
        return v.item() if v.ndim == 0 else v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return v


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def tensor_to_pil(image):
    """[C,H,W] float in [0,1] -> RGB PIL.Image (uint8)."""
    arr = (image.permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()
    return Image.fromarray(arr, mode="RGB")


def draw_boxes(draw, boxes, labels, color_fn, *, width, scores=None, font=None):
    """Draw boxes; optionally annotate predictions with class + score."""
    for j, box in enumerate(boxes):
        x1, y1, x2, y2 = (float(v) for v in box)
        color = color_fn(int(labels[j]))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        if scores is not None:
            name = CLASS_NAMES.get(int(labels[j]), str(int(labels[j])))
            txt = f"{name} {float(scores[j]):.2f}"
            ty = max(y1 - 12, 0)
            draw.rectangle([x1, ty, x1 + 7 * len(txt), ty + 12], fill=color)
            draw.text((x1 + 1, ty), txt, fill=(0, 0, 0), font=font)


def render_frame(image, gt, pred, *, layout, font, banner):
    """Compose one RGB frame (uint8 np array) with GT and predictions."""
    def paint_pred(canvas):
        d = ImageDraw.Draw(canvas)
        draw_boxes(d, pred["boxes"], pred["labels"],
                   lambda c: PRED_COLORS.get(c, PRED_FALLBACK),
                   width=2, scores=pred["scores"], font=font)

    def paint_gt(canvas):
        d = ImageDraw.Draw(canvas)
        draw_boxes(d, gt["boxes"], gt["labels"], lambda _c: GT_COLOR, width=3)

    base = tensor_to_pil(image)
    if layout == "sidebyside":
        left, right = base.copy(), base.copy()
        paint_gt(left)
        paint_pred(right)
        _banner(left, "GT (black)", font)
        _banner(right, banner, font)
        out = Image.new("RGB", (base.width * 2, base.height))
        out.paste(left, (0, 0))
        out.paste(right, (base.width, 0))
    else:  # overlay
        paint_gt(base)
        paint_pred(base)
        _banner(base, "GT=black  pred=color  " + banner, font)
        out = base
    return np.asarray(out)


def _banner(canvas, text, font):
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, 8 * len(text) + 6, 16], fill=(0, 0, 0))
    d.text((3, 2), text, fill=(255, 255, 255), font=font)


# ---------------------------------------------------------------------------
# Video writer with fallbacks: imageio -> cv2 -> PNG dump
# ---------------------------------------------------------------------------
class VideoSink:
    def __init__(self, path, fps, png_dir):
        self.path, self.fps, self.png_dir = path, fps, png_dir
        self.backend, self.writer, self._n = None, None, 0
        try:
            import imageio.v2 as imageio
            self.writer = imageio.get_writer(str(path), fps=fps,
                                             macro_block_size=None)
            self.backend = "imageio"
            return
        except Exception:
            pass
        try:
            import cv2
            self._cv2 = cv2
            self.backend = "cv2"  # writer opened lazily (needs frame size)
            return
        except Exception:
            pass
        self.backend = "png"
        png_dir.mkdir(parents=True, exist_ok=True)

    def append(self, frame):
        if self.backend == "imageio":
            self.writer.append_data(frame)
        elif self.backend == "cv2":
            if self.writer is None:
                h, w = frame.shape[:2]
                fourcc = self._cv2.VideoWriter_fourcc(*"mp4v")
                self.writer = self._cv2.VideoWriter(str(self.path), fourcc,
                                                    self.fps, (w, h))
            self.writer.write(frame[:, :, ::-1])  # RGB -> BGR
        else:  # png
            Image.fromarray(frame).save(self.png_dir / f"frame_{self._n:04d}.png")
        self._n += 1

    def close(self):
        if self.backend == "imageio" and self.writer is not None:
            self.writer.close()
        elif self.backend == "cv2" and self.writer is not None:
            self.writer.release()
        elif self.backend == "png":
            print(f"[fallback] no video lib found; wrote {self._n} PNGs to "
                  f"{self.png_dir}\n          assemble with: ffmpeg -framerate "
                  f"{self.fps} -i {self.png_dir}/frame_%04d.png -c:v libx264 "
                  f"-pix_fmt yuv420p {self.path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="GT-vs-prediction comparison video.")
    p.add_argument("--ckpt", type=Path, required=True, help="Path to last.pt.")
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--segment", default=None, help="Render only this segment.")
    p.add_argument("--score-thresh", type=float, default=0.5)
    p.add_argument("--layout", default="overlay", choices=["overlay", "sidebyside"])
    p.add_argument("--fps", type=int, default=5)
    p.add_argument("--max-frames", type=int, default=None,
                   help="Cap frames per segment (quick preview).")
    p.add_argument("--min-size", type=int, default=None, help="Override ckpt.")
    p.add_argument("--max-size", type=int, default=None, help="Override ckpt.")
    p.add_argument("--out-dir", type=Path, default=Path("runs/videos"))
    p.add_argument("--device", default="cuda")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA unavailable; using CPU")
        args.device = "cpu"
    device = torch.device(args.device)

    model, name = load_model(args.ckpt, device,
                             min_size=args.min_size, max_size=args.max_size)
    ds = PerceptionCacheDataset(args.cache_root / args.split, drop_empty=False)
    order = build_order(ds, args.segment)
    if not order:
        raise SystemExit(f"no records for split={args.split} segment={args.segment}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    print(f"[video] model={name} split={args.split} segments={len(order)} "
          f"thr={args.score_thresh} layout={args.layout} device={device}")

    for seg, indices in order.items():
        if args.max_frames:
            indices = indices[:args.max_frames]
        safe_seg = str(seg).replace("/", "_")[:40]
        out_path = args.out_dir / f"{args.split}_{safe_seg}.mp4"
        sink = VideoSink(out_path, args.fps, args.out_dir / f"{safe_seg}_frames")

        for i in indices:
            image, target = ds[i]
            preds = model([image.to(device)])[0]
            keep = preds["scores"] >= args.score_thresh
            pred = {k: preds[k][keep].cpu() for k in ("boxes", "labels", "scores")}
            banner = f"{safe_seg} | f{int(_as_py(target.get('frame_idx', i)))}"
            frame = render_frame(image, target, pred,
                                 layout=args.layout, font=font, banner=banner)
            sink.append(frame)

        sink.close()
        print(f"[video]   {out_path}  ({len(indices)} frames)")


if __name__ == "__main__":
    main()
