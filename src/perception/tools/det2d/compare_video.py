"""Render a model-vs-model comparison video: arm A vs arm B over the same frames.

draw_video_from_json.py compares GT against ONE model (its "sidebyside" is
GT|pred). This script instead puts TWO models on the same frames -- e.g.
pretrained vs from-scratch -- so the qualitative difference is read directly.

Inputs are two predictions.json dumps from dump_predictions.py (NOT
inference_frcnn.py -- that one writes no pixels). Both must be dumped on the SAME
--split so frame i aligns; alignment here is by image_id, not list position, so
it is robust to ordering. The PNG pixels are read from arm A's dump (both arms
dumped the same underlying frames).

Pick the MEDIAN seed of each arm, not the best -- the distribution is the result,
a best seed is cherry-picking. For N=48/s8 that is pt seed9 (0.1543 ~ mean) and
scratch seed1 (0.0825 ~ mean).

Layouts:
  3up           GT | arm-A | arm-B, three panels, per-class colors in each.
  dual-overlay  one image: GT in black, arm-A and arm-B each in one arm color
                (A=blue, B=orange, matching the plots), so the two models are
                told apart at a glance.

--score-thresh filters both arms identically (display only; inference was frozen
at dump time, so changing it never re-runs the model).

Run:

    python compare_video.py \
        --json-a runs/faster_rcnn_pt_n48_s8_seed9/dump_val/predictions.json \
        --json-b runs/faster_rcnn_n48_s8_seed1/dump_val/predictions.json \
        --label-a pretrained --label-b from-scratch \
        --layout 3up --score-thresh 0.5 --fps 5
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

GT_COLOR = (0, 0, 0)                   # black
PER_CLASS = {                          # 3up: color by class, keyed by int label
    1: (0, 200, 0),                    # Vehicle    -> green
    2: (255, 140, 0),                  # Pedestrian -> orange
    3: (0, 200, 255),                  # Cyclist    -> cyan
}
CLASS_FALLBACK = (255, 0, 255)         # magenta
ARM_COLORS = ((42, 120, 214),          # arm A -> blue  (#2a78d6, pt in the plots)
              (235, 104, 52))          # arm B -> orange(#eb6834, scratch)


def draw_boxes(draw, boxes, labels, color_fn, *, width,
               scores=None, names=None, font=None):
    """Draw boxes; annotate with class name + score when scores given."""
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


def banner(canvas, text, font, color=(0, 0, 0)):
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, 8 * len(text) + 6, 16], fill=color)
    d.text((3, 2), text, fill=(255, 255, 255), font=font)


def _filter(pred, thr):
    keep = [k for k, s in enumerate(pred["scores"]) if s >= thr]
    return ([pred["boxes"][k] for k in keep],
            [pred["labels"][k] for k in keep],
            [pred["scores"][k] for k in keep])


def render_3up(img, gt, pred_a, pred_b, *, names, font, labels):
    """GT | arm-A | arm-B, three panels with per-class colored predictions."""
    la, lb = labels
    gt_panel, a_panel, b_panel = img.copy(), img.copy(), img.copy()

    draw_boxes(ImageDraw.Draw(gt_panel), gt["boxes"], gt["labels"],
               lambda _l: GT_COLOR, width=3)
    for panel, pred in ((a_panel, pred_a), (b_panel, pred_b)):
        draw_boxes(ImageDraw.Draw(panel), pred[0], pred[1],
                   lambda l: PER_CLASS.get(l, CLASS_FALLBACK),
                   width=2, scores=pred[2], names=names, font=font)
    banner(gt_panel, "GT", font)
    banner(a_panel, la, font)
    banner(b_panel, lb, font)

    out = Image.new("RGB", (img.width * 3, img.height))
    out.paste(gt_panel, (0, 0))
    out.paste(a_panel, (img.width, 0))
    out.paste(b_panel, (img.width * 2, 0))
    return np.asarray(out)


def render_dual(img, gt, pred_a, pred_b, *, names, font, labels):
    """One image: GT black, arm-A in arm color A, arm-B in arm color B."""
    la, lb = labels
    d = ImageDraw.Draw(img)
    draw_boxes(d, gt["boxes"], gt["labels"], lambda _l: GT_COLOR, width=3)
    draw_boxes(d, pred_a[0], pred_a[1], lambda _l: ARM_COLORS[0], width=2)
    draw_boxes(d, pred_b[0], pred_b[1], lambda _l: ARM_COLORS[1], width=2)
    banner(img, f"GT=black  {la}=blue  {lb}=orange", font)
    return np.asarray(img)


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
            print(f"[fallback] no video lib; wrote {self._n} PNGs to "
                  f"{self.png_dir}\n          ffmpeg -framerate {self.fps} -i "
                  f"{self.png_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p "
                  f"{self.path}")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json-a", type=Path, required=True,
                   help="dump_predictions.py predictions.json for arm A.")
    p.add_argument("--json-b", type=Path, required=True,
                   help="dump_predictions.py predictions.json for arm B.")
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    p.add_argument("--layout", default="3up", choices=["3up", "dual-overlay"])
    p.add_argument("--score-thresh", type=float, default=0.5,
                   help="Display threshold, applied identically to both arms.")
    p.add_argument("--segment", default=None, help="Render only this segment.")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Cap frames per segment (quick preview).")
    p.add_argument("--fps", type=int, default=5)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Default: <json-a dir>/compare_videos.")
    return p.parse_args()


def main():
    args = parse_args()
    doc_a = json.loads(args.json_a.read_text())
    doc_b = json.loads(args.json_b.read_text())
    base_a = args.json_a.parent
    names = doc_a["meta"].get("class_names", {})
    thr = args.score_thresh

    for tag, doc in (("A", doc_a), ("B", doc_b)):
        dthr = doc["meta"].get("dump_score_thresh", 0.0)
        if thr < dthr:
            print(f"[warn] arm {tag}: display thr {thr} < dump thr {dthr}; "
                  f"boxes below dump thr were never stored.")

    # Align by image_id (robust to ordering); B indexed for lookup.
    b_by_id = {int(fr["image_id"]): fr for fr in doc_b["frames"]}

    # Group A's frames by segment, keep only those with a B match.
    groups, missing = {}, 0
    for fr in doc_a["frames"]:
        seg = fr.get("segment")
        if args.segment is not None and str(seg) != str(args.segment):
            continue
        if int(fr["image_id"]) not in b_by_id:
            missing += 1
            continue
        groups.setdefault(seg, []).append(fr)
    if not groups:
        raise SystemExit(f"no aligned frames (segment={args.segment}, "
                         f"{missing} A-frames had no B match)")
    if missing:
        print(f"[warn] {missing} A-frames had no matching image_id in B")
    for seg in groups:
        groups[seg].sort(key=lambda f: f.get("frame_idx", 0))

    out_dir = args.out_dir or base_a / "compare_videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    render = render_3up if args.layout == "3up" else render_dual
    labels = (args.label_a, args.label_b)
    split = doc_a["meta"].get("split", "x")
    print(f"[video] segments={len(groups)} layout={args.layout} thr={thr} "
          f"-> {out_dir}")

    for seg, frs in groups.items():
        if args.max_frames:
            frs = frs[:args.max_frames]
        safe = str(seg).replace("/", "_")[:40]
        out_path = out_dir / f"cmp_{split}_{safe}.mp4"
        sink = VideoSink(out_path, args.fps, out_dir / f"{safe}_frames")
        for fr_a in frs:
            fr_b = b_by_id[int(fr_a["image_id"])]
            img = Image.open(base_a / fr_a["image_file"]).convert("RGB")
            pa = _filter(fr_a["pred"], thr)
            pb = _filter(fr_b["pred"], thr)
            frame = render(img, fr_a["gt"], pa, pb,
                           names=names, font=font, labels=labels)
            sink.append(frame)
        sink.close()
        print(f"[video]   {out_path}  ({len(frs)} frames)")


if __name__ == "__main__":
    main()
