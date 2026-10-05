"""COCO-style AP evaluation for 2D detection dumps (decision E metric module).

Standalone: consumes a predictions.json produced by dump_predictions.py and
reports AP@[.50:.95], AP50, AP75 -- overall and per class -- via pycocotools
(pure Python, runs in gpu_env; no TF / cpu_env needed). Decoupled from training:
run it on any dumped checkpoint, on demand.

The core (`evaluate_predictions`) is importable, so the training loop can feed
its in-memory val predictions per epoch through the SAME metric -- one ruler,
two feeders (JSON here, in-memory later).

Conventions enforced here (silent-bug guards):
  * boxes are converted xyxy -> xywh (COCO expects xywh);
  * all classes are declared as categories even with zero GT, so a class is
    reported as n/a (no GT) rather than silently vanishing;
  * image_id is taken as-is from the dump (dataset.py emits the manifest index,
    unique per split -- no synthesis needed).

Run:

    python eval_ap.py --json runs/<run_id>/dump_val/predictions.json
"""

import argparse
import contextlib
import io
import json
import math
from pathlib import Path

DEFAULT_CLASS_NAMES = {"1": "Vehicle", "2": "Pedestrian", "3": "Cyclist"}


def _xywh(box):
    """xyxy -> xywh."""
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


def build_coco_gt(frames, class_names):
    """Assemble a COCO ground-truth dict from dump frames."""
    images, annotations = [], []
    ann_id = 1
    for fr in frames:
        img_id = int(fr["image_id"])
        images.append({"id": img_id,
                       "width": int(fr.get("width", 0)),
                       "height": int(fr.get("height", 0))})
        gt = fr["gt"]
        boxes, labels = gt["boxes"], gt["labels"]
        areas = gt.get("area") or []
        crowd = gt.get("iscrowd") or []
        for k, box in enumerate(boxes):
            w, h = box[2] - box[0], box[3] - box[1]
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": int(labels[k]),
                "bbox": _xywh(box),
                "area": float(areas[k]) if k < len(areas) else float(w * h),
                "iscrowd": int(crowd[k]) if k < len(crowd) else 0,
            })
            ann_id += 1
    categories = [{"id": cid, "name": name} for cid, name in
                  sorted((int(k), v) for k, v in class_names.items())]
    return {"images": images, "annotations": annotations, "categories": categories}


def build_detections(frames):
    """Flatten dump predictions into a COCO results list."""
    dets = []
    for fr in frames:
        img_id = int(fr["image_id"])
        pred = fr["pred"]
        for k, box in enumerate(pred["boxes"]):
            dets.append({
                "image_id": img_id,
                "category_id": int(pred["labels"][k]),
                "bbox": _xywh(box),
                "score": float(pred["scores"][k]),
            })
    return dets


def evaluate_predictions(frames, class_names=None, *, verbose=True):
    """Run COCOeval over dump frames. Returns a metrics dict.

    frames: list of {image_id, width, height, gt:{boxes,labels,area,iscrowd},
            pred:{boxes,labels,scores}} with plain-list boxes in xyxy pixels.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import numpy as np

    class_names = class_names or DEFAULT_CLASS_NAMES
    gt_dict = build_coco_gt(frames, class_names)
    dets = build_detections(frames)

    coco_gt = COCO()
    coco_gt.dataset = gt_dict
    coco_gt.createIndex()

    keys = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large"]
    id_to_name = {c["id"]: c["name"] for c in gt_dict["categories"]}

    if not dets:
        if verbose:
            print("[ap] no detections above dump threshold -> all AP = 0")
        return {k: 0.0 for k in keys} | {
            "per_class": {n: float("nan") for n in id_to_name.values()}}

    coco_dt = coco_gt.loadRes(dets)
    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev.evaluate()
    ev.accumulate()

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ev.summarize()
    if verbose:
        print(buf.getvalue(), end="")

    stats = ev.stats
    out = {k: float(stats[i]) for i, k in enumerate(keys)}

    # per-class AP@[.50:.95]: precision[T, R, K, A=0(all), M=2(maxDet100)].
    # -1 entries are unachieved recall levels; a class with no GT is all -1 -> nan.
    precision = ev.eval["precision"]
    out["per_class"] = {}
    for k, cid in enumerate(coco_gt.getCatIds()):
        p = precision[:, :, k, 0, 2]
        p = p[p > -1]
        out["per_class"][id_to_name[cid]] = float(p.mean()) if p.size else float("nan")
    return out


def _fmt(v):
    return "  n/a (no GT)" if (isinstance(v, float) and math.isnan(v)) else f"{v:.4f}"


def main():
    ap = argparse.ArgumentParser(description="COCO-AP over a predictions.json dump.")
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None,
                    help="optional: also write the metrics as JSON here.")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress the pycocotools 12-line summary table.")
    args = ap.parse_args()

    doc = json.loads(args.json.read_text())
    frames = doc["frames"]
    class_names = doc.get("meta", {}).get("class_names", DEFAULT_CLASS_NAMES)

    m = evaluate_predictions(frames, class_names, verbose=not args.quiet)

    print(f"\n[AP] overall   AP@[.50:.95]={m['AP']:.4f}   "
          f"AP50={m['AP50']:.4f}   AP75={m['AP75']:.4f}")
    for name, v in m["per_class"].items():
        print(f"     {name:12s} AP={_fmt(v)}")

    if args.out:
        args.out.write_text(json.dumps(m, indent=2))
        print(f"[AP] wrote {args.out}")


if __name__ == "__main__":
    main()
