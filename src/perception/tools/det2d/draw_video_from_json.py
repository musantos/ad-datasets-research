"""Render a GT-vs-prediction comparison video from a predictions.json dump.

Fully standalone: reads the JSON produced by dump_predictions.py plus the PNG
frames beside it, draws GT boxes in BLACK and predicted boxes in per-class
colors, and writes one video per segment. No torch, no checkpoint, no dataset --
runs anywhere PIL is available (e.g. cpu_env).

Changing --score-thresh / --layout / colors never re-runs the model: inference
was frozen at dump time.

Run:

    python draw_video_from_json.py \
        --json runs/<run_id>/dump_val/predictions.json \
        --score-thresh 0.5 --fps 5
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


GT_COLOR = (0, 0, 0)                 # black
PRED_COLORS = {                      # keyed by int label
    1: (0, 200, 0),                  # Vehicle    -> green
    2: (255, 140, 0),                # Pedestrian -> orange
    3: (0, 200, 255),                # Cyclist    -> cyan
}
PRED_FALLBACK = (255, 0, 255)        # magenta


def draw_boxes(draw, boxes, labels, color_fn, *, width,
               scores=None, names=None, font=None):
    """Draw boxes; annotate predictions with class name + score when given."""
    for j, box in enumerate(boxes):
        x1, y1, x2, y2 = (float(v) for v in box)
        lbl = int(labels[j])
        color = color_fn(lbl)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        if scores is not None:
            name = names.get(str(lbl), str(lbl)) if names else str(lbl)
            txt = f"{name} {float(scores[j]):.2f}"
            ty = max(y1 - 12, 0)
            draw.rectangle([x1, ty, x1 + 7 * len(txt), ty + 12], fill=color)
            draw.text((x1 + 1, ty), txt, fill=(0, 0, 0), font=font)


def banner(canvas, text, font):
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, 8 * len(text) + 6, 16], fill=(0, 0, 0))
    d.text((3, 2), text, fill=(255, 255, 255), font=font)


def render_frame(img, gt, pred, *, layout, names, font, label):
    """Compose one RGB frame (uint8 np array) from a PIL image + box dicts."""
    def paint_gt(c):
        draw_boxes(ImageDraw.Draw(c), gt["boxes"], gt["labels"],
                   lambda _l: GT_COLOR, width=3)

    def paint_pred(c):
        draw_boxes(ImageDraw.Draw(c), pred["boxes"], pred["labels"],
                   lambda l: PRED_COLORS.get(l, PRED_FALLBACK),
                   width=2, scores=pred["scores"], names=names, font=font)

    if layout == "sidebyside":
        left, right = img.copy(), img.copy()
        paint_gt(left)
        paint_pred(right)
        banner(left, "GT (black)", font)
        banner(right, label, font)
        out = Image.new("RGB", (img.width * 2, img.height))
        out.paste(left, (0, 0))
        out.paste(right, (img.width, 0))
    else:  # overlay
        paint_gt(img)
        paint_pred(img)
        banner(img, "GT=black  pred=color  " + label, font)
        out = img
    return np.asarray(out)


class VideoSink:
    """mp4 writer with fallbacks: imageio -> cv2 -> PNG dump + ffmpeg hint."""

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
            self.backend = "cv2"
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
            self.writer.write(frame[:, :, ::-1])
        else:
            Image.fromarray(frame).save(self.png_dir / f"frame_{self._n:04d}.png")
        self._n += 1

    def close(self):
        if self.backend == "imageio" and self.writer is not None:
            self.writer.close()
        elif self.backend == "cv2" and self.writer is not None:
            self.writer.release()
        elif self.backend == "png":
            print(f"[fallback] no video lib found; wrote {self._n} PNGs to "
                  f"{self.png_dir}\n          ffmpeg -framerate {self.fps} -i "
                  f"{self.png_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p "
                  f"{self.path}")


def parse_args():
    p = argparse.ArgumentParser(description="GT-vs-pred video from predictions.json.")
    p.add_argument("--json", type=Path, required=True)
    p.add_argument("--score-thresh", type=float, default=0.5,
                   help="Display threshold for predictions (filter only).")
    p.add_argument("--segment", default=None, help="Render only this segment.")
    p.add_argument("--layout", default="overlay", choices=["overlay", "sidebyside"])
    p.add_argument("--fps", type=int, default=5)
    p.add_argument("--max-frames", type=int, default=None,
                   help="Cap frames per segment (quick preview).")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Default: <json_dir>/videos.")
    return p.parse_args()


def main():
    args = parse_args()
    doc = json.loads(args.json.read_text())
    base = args.json.parent
    names = doc["meta"].get("class_names", {})
    thr = args.score_thresh

    if thr < doc["meta"].get("dump_score_thresh", 0.0):
        print(f"[warn] display thr {thr} < dump thr "
              f"{doc['meta']['dump_score_thresh']}: boxes below dump thr were "
              f"never stored; re-dump lower to see them.")

    # group by segment, order by frame_idx
    groups = {}
    for fr in doc["frames"]:
        seg = fr.get("segment")
        if args.segment is not None and str(seg) != str(args.segment):
            continue
        groups.setdefault(seg, []).append(fr)
    if not groups:
        raise SystemExit(f"no frames for segment={args.segment}")
    for seg in groups:
        groups[seg].sort(key=lambda f: f.get("frame_idx", 0))

    out_dir = args.out_dir or base / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    print(f"[video] segments={len(groups)} thr={thr} layout={args.layout} "
          f"-> {out_dir}")

    for seg, frs in groups.items():
        if args.max_frames:
            frs = frs[:args.max_frames]
        safe = str(seg).replace("/", "_")[:40]
        out_path = out_dir / f"{doc['meta'].get('split', 'x')}_{safe}.mp4"
        sink = VideoSink(out_path, args.fps, out_dir / f"{safe}_frames")

        for fr in frs:
            img = Image.open(base / fr["image_file"]).convert("RGB")
            pred = fr["pred"]
            keep = [k for k, s in enumerate(pred["scores"]) if s >= thr]
            pred_f = {
                "boxes": [pred["boxes"][k] for k in keep],
                "labels": [pred["labels"][k] for k in keep],
                "scores": [pred["scores"][k] for k in keep],
            }
            label = f"{safe} | f{fr.get('frame_idx', 0)}"
            frame = render_frame(img, fr["gt"], pred_f,
                                 layout=args.layout, names=names,
                                 font=font, label=label)
            sink.append(frame)

        sink.close()
        print(f"[video]   {out_path}  ({len(frs)} frames)")


if __name__ == "__main__":
    main()
