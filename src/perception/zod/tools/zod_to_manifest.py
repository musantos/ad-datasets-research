"""ZOD Frames -> perception cache (sampled JPG copy + manifest, no TensorFlow).

Analogue of the Waymo ``preprocessor.py`` for the ZOD side. Both emit the exact
same cache contract that the shared ``dataset.py`` consumes:

    <cache_dir>/
    ├── images/        the frame JPGs actually used
    └── manifest.json  list of records, image paths RELATIVE to <cache_dir>

so the cache is self-contained and portable (moving it never breaks the paths).

Two structural differences from Waymo, both inherent to the data, not choices:
  * ZOD ships already unpacked, so there is no TFRecord to decode: the raw JPG is
    copied straight from the (HDD) root into ``images/``. No re-encode to PNG --
    the JPG is the original; PNG would only inflate it ~3-5x for no quality gain.
  * ZOD 2D boxes are the image projection of the 3D box, stored as a rotated
    quadrilateral (4 points). The 2D box is just the axis-aligned envelope of
    those 4 points -- nothing to reproject, no devkit, no fisheye handling.

The split index (``trainval-frames-full.json``, ~470 MB) is read in streaming
mode so it never sits in RAM whole; only a light per-frame tuple is kept to drive
sampling. Clamp, degenerate-box drop and int labels are left to ``dataset.py`` at
__getitem__ time, so the manifest stays raw on purpose.

Sampling note: ZOD frames are independent (one keyframe per scene, not video), so
there is no temporal correlation to break. Shuffling still matters, for a
different reason -- the index may be ordered by collection batch (date, fleet,
country), so taking the first N would yield a biased slice. A seeded shuffle
makes the sample representative of the full set and reproducible for the paper.

Usage (on the host, where the data and ijson live):
    # real sample: 2000 frames, shuffled with a fixed seed, empties skipped
    python zod_to_manifest.py \
        --root  ~/autonomous_drive/datasets/zod/frames \
        --split train \
        --cache ~/autonomous_drive/datasets/zod/perception_cache/train \
        --sample-size 2000 --seed 42

    # smoke test: first 20 frames, no shuffle, no copy guarantees
    python zod_to_manifest.py \
        --root  ~/autonomous_drive/datasets/zod/frames \
        --split train \
        --cache ~/autonomous_drive/datasets/zod/perception_cache/_smoke \
        --limit 20
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import json
from collections import Counter

# ijson streams the 470 MB index item by item. If it is missing, we fall back to
# a whole-file json.load (needs the RAM, but works) so the script is never blocked.
try:
    import ijson
    _HAVE_IJSON = True
except ImportError:
    _HAVE_IJSON = False


# ZOD object_detection `class` -> contiguous index (0 reserved for background by
# the detector). Matches the Waymo map {Vehicle:1, Pedestrian:2}. Cyclist is left
# out on purpose: it is not yet confirmed how it surfaces in the ZOD schema
# (likely a VulnerableVehicle class/subtype), so we do not invent index 3 until a
# real "class"/"type" value is confirmed in an object_detection.json.
ZOD_TO_IDX = {"Vehicle": 1, "Pedestrian": 2}
IDX_TO_NAME = {1: "vehicle", 2: "pedestrian"}

# Image blur folders are split by id range; the index path omits this prefix.
IMAGE_RANGE_PREFIXES = ("images_blur_000000_049999", "images_blur_050000_099999")


def image_prefix_for_id(frame_id: str) -> str:
    """Return the images_blur_* folder that holds a given frame id."""
    n = int(frame_id)
    return IMAGE_RANGE_PREFIXES[0] if n < 50000 else IMAGE_RANGE_PREFIXES[1]


def envelope(coords) -> list[float]:
    """Axis-aligned envelope [x1, y1, x2, y2] of a MultiPoint's 4 corners."""
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    return [min(xs), min(ys), max(xs), max(ys)]


def extract_boxes(obj_detection_path: str) -> list[list[float]]:
    """Read one object_detection.json -> list of [x1, y1, x2, y2, class_idx].

    Keeps only classes in ZOD_TO_IDX; everything else (TrafficSign, PoleObject,
    ...) is dropped to background. Defensive: an object may carry only geometry +
    class, so every property access tolerates absence.
    """
    boxes: list[list[float]] = []
    with open(obj_detection_path) as f:
        objects = json.load(f)  # per-frame file is small

    for obj in objects:
        cls = obj.get("properties", {}).get("class")
        idx = ZOD_TO_IDX.get(cls)
        if idx is None:
            continue
        coords = obj.get("geometry", {}).get("coordinates")
        if not coords:
            continue
        x1, y1, x2, y2 = envelope(coords)
        boxes.append([x1, y1, x2, y2, idx])
    return boxes


def iter_index_frames(index_path: str, split: str):
    """Yield each frame dict of the chosen split, streaming when possible."""
    if _HAVE_IJSON:
        with open(index_path, "rb") as f:
            yield from ijson.items(f, f"{split}.item")
    else:
        with open(index_path) as f:
            data = json.load(f)
        yield from data[split]


def collect_frame_refs(index_path: str, split: str) -> list[dict]:
    """Stream the index once and keep only the light fields needed per frame.

    Returns one dict per frame with: id, annotation path, image path, width,
    height. Dropping the rest keeps memory modest even across the full 100k, so
    the whole split can be shuffled in place.
    """
    refs: list[dict] = []
    n_skipped = 0
    for frame in iter_index_frames(index_path, split):
        frame_id = frame["id"]
        ann_rel = frame.get("annotations", {}).get("object_detection", {}).get("filepath")
        cam = frame.get("camera_frames", {}).get("front_blur", [])
        if not ann_rel or not cam:
            n_skipped += 1
            continue
        refs.append({
            "id": frame_id,
            "ann_rel": ann_rel,
            "img_rel": cam[0]["filepath"],
            "width": cam[0]["width"],
            "height": cam[0]["height"],
        })
    if n_skipped:
        print(f"INFO: {n_skipped} index entries dropped (no annotation or no front_blur).")
    return refs


def build_cache(root: str, split: str, cache_dir: str, sample_size: int | None,
                seed: int, limit: int | None, skip_empty: bool) -> None:
    index_path = os.path.join(root, "infos", "trainval-frames-full.json")
    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"split index not found: {index_path}")

    img_dir = os.path.join(cache_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    print(f"INFO: reading split '{split}' from {index_path}")
    print(f"INFO: ijson streaming = {_HAVE_IJSON}")
    print(f"INFO: writing cache to {cache_dir}")

    # --- select the frames to materialize -------------------------------------
    if limit is not None:
        # Smoke test: first N in index order, no shuffle. Stop streaming early.
        refs = []
        for frame in iter_index_frames(index_path, split):
            frame_id = frame["id"]
            ann_rel = frame.get("annotations", {}).get("object_detection", {}).get("filepath")
            cam = frame.get("camera_frames", {}).get("front_blur", [])
            if not ann_rel or not cam:
                continue
            refs.append({
                "id": frame_id, "ann_rel": ann_rel, "img_rel": cam[0]["filepath"],
                "width": cam[0]["width"], "height": cam[0]["height"],
            })
            if len(refs) >= limit:
                break
        print(f"INFO: smoke test -> first {len(refs)} frame(s), no shuffle.")
    else:
        # Real sample: collect the whole split, shuffle with seed, cut to size.
        refs = collect_frame_refs(index_path, split)
        print(f"INFO: split holds {len(refs)} usable frame(s).")
        if sample_size is not None:
            random.Random(seed).shuffle(refs)
            refs = refs[:sample_size]
            print(f"INFO: sampled {len(refs)} frame(s) with seed={seed}.")
        else:
            print("INFO: no --sample-size -> processing the full split.")

    # --- materialize: copy JPG (HDD -> SSD cache) + build manifest -------------
    manifest = []
    class_counter: Counter[str] = Counter()
    n_missing_ann = n_missing_img = n_empty_skipped = 0

    for ref in refs:
        frame_id = ref["id"]

        ann_abs = os.path.join(root, "annotations", ref["ann_rel"])
        if not os.path.isfile(ann_abs):
            n_missing_ann += 1
            continue

        img_abs = os.path.join(root, image_prefix_for_id(frame_id), ref["img_rel"])
        if not os.path.isfile(img_abs):
            n_missing_img += 1
            continue

        boxes = extract_boxes(ann_abs)
        if skip_empty and not boxes:
            n_empty_skipped += 1
            continue

        # Copy the raw JPG into the cache. Basename is unique (<id>_<fleet>_<ts>)
        # and preserves provenance in the filename.
        fname = os.path.basename(ref["img_rel"])
        shutil.copy2(img_abs, os.path.join(img_dir, fname))

        for b in boxes:
            class_counter[IDX_TO_NAME[b[4]]] += 1

        manifest.append({
            "image": f"images/{fname}",  # relative: cache is self-contained
            "width": int(ref["width"]),
            "height": int(ref["height"]),
            "segment": frame_id,         # provenance: the ZOD frame id
            "frame_idx": 0,              # one keyframe per ZOD frame
            "camera": "front_blur",
            "boxes": boxes,
        })

    with open(os.path.join(cache_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f)

    # --- sanity report (same spirit as the Waymo preprocessor) ----------------
    print(f"INFO: done. records={len(manifest)}, "
          f"empty_skipped={n_empty_skipped}, "
          f"missing_ann={n_missing_ann}, missing_img={n_missing_img}")
    print(f"INFO: box counts by class: {dict(class_counter)}")
    for name in IDX_TO_NAME.values():
        if class_counter[name] == 0:
            print(f"WARNING: zero '{name}' boxes in this extraction.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a perception cache (sampled JPG copy + manifest) from ZOD Frames."
    )
    parser.add_argument("--root", required=True,
                        help="ZOD frames root (holds annotations/, infos/, images_blur_*/)")
    parser.add_argument("--split", default="train", choices=["train", "val"],
                        help="official ZOD split to process")
    parser.add_argument("--cache", required=True,
                        help="output cache dir (gets images/ + manifest.json)")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="sample N frames from the split (seeded shuffle first)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for the sample shuffle (reproducibility)")
    parser.add_argument("--limit", type=int, default=None,
                        help="SMOKE TEST: first N frames, no shuffle (ignores --sample-size)")
    parser.add_argument("--keep-empty", action="store_true",
                        help="keep frames with no target boxes (default: skip them)")
    args = parser.parse_args()

    if args.limit is not None and args.sample_size is not None:
        parser.error("--limit (smoke, no shuffle) and --sample-size (real sample) "
                     "are mutually exclusive; pick one.")

    build_cache(
        root=os.path.expanduser(args.root),
        split=args.split,
        cache_dir=os.path.expanduser(args.cache),
        sample_size=args.sample_size,
        seed=args.seed,
        limit=args.limit,
        skip_empty=not args.keep_empty,
    )
