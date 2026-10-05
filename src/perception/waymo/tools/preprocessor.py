"""Waymo Perception preprocessor (warm-up / Stage 1).

Reads the frozen split (perception_splits.json), slices the first N segments,
and converts (image, 2D boxes) per camera into a PNG + JSON-manifest cache.
Runs in the CPU container (TF read side). The training side stays TF-free:
it consumes only PNG + manifest, never TF or the raw dump.

Mirrors the Motion preprocessor's shape (official split, run_extraction, argparse,
sanity warnings) but for the Perception camera-detection track.
"""

import argparse
import glob
import json
import os
from collections import Counter

import numpy as np
import tensorflow as tf
from PIL import Image
from waymo_open_dataset import dataset_pb2 as open_dataset

# --- Paths inside the container (mount drops the ".disks"; see ways-of-working) ---
DATA_ROOT = "/data/hdd16/waymo_perception/waymo_open_dataset_v_1_4_3/individual_files"
SPLITS_FILE = "/workspace/datasets/waymo/perception_splits.json"
SUFFIX = "_with_camera_labels.tfrecord"

# Map our split name -> (source subfolder, destination cache)
SPLIT_CFG = {
    "train": {"dir": "training",   "cache": "/workspace/datasets/waymo/perception_cache/train"},
    "val":   {"dir": "validation", "cache": "/workspace/datasets/waymo/perception_cache/val"},
}

# Short alias -> Waymo camera enum. Single source of truth for camera selection.
CAMERA_ALIASES = {
    "f":  open_dataset.CameraName.FRONT,
    "fl": open_dataset.CameraName.FRONT_LEFT,
    "fr": open_dataset.CameraName.FRONT_RIGHT,
    "sl": open_dataset.CameraName.SIDE_LEFT,
    "sr": open_dataset.CameraName.SIDE_RIGHT,
}

# Waymo class enum -> contiguous index (0 reserved for background).
# Vehicle=1, Pedestrian=2, Cyclist=4. Sign=3 and Unknown=0 are dropped -> background.
WAYMO_TO_IDX = {1: 1, 2: 2, 4: 3}
IDX_TO_NAME = {1: "vehicle", 2: "pedestrian", 3: "cyclist"}


def resolve_cameras(spec):
    """Turn a camera spec ('all' or ['f','sl',...]) into a list of Waymo enums."""
    if spec == "all":
        return list(CAMERA_ALIASES.values())
    return [CAMERA_ALIASES[a] for a in spec]   # KeyError here = typo caught early


def extract_boxes(labels_proto):
    """Convert Waymo camera labels -> list of [x1, y1, x2, y2, class_idx]."""
    boxes = []
    if labels_proto is None:
        return boxes
    for label in labels_proto.labels:
        if label.type not in WAYMO_TO_IDX:   # Sign / Unknown -> background, skip
            continue
        b = label.box
        # Waymo: length -> x (horizontal), width -> y (vertical). Center -> corners.
        x1 = b.center_x - b.length / 2
        y1 = b.center_y - b.width / 2
        x2 = b.center_x + b.length / 2
        y2 = b.center_y + b.width / 2
        boxes.append([x1, y1, x2, y2, WAYMO_TO_IDX[label.type]])
    return boxes


def run_extraction(split, n_segments, stride, cameras_spec, skip_empty):
    cfg = SPLIT_CFG[split]
    cameras = resolve_cameras(cameras_spec)
    cache_path = cfg["cache"]
    img_dir = os.path.join(cache_path, "images")
    os.makedirs(img_dir, exist_ok=True)
    print(f"INFO: split='{split}' -> writing cache to {cache_path}")

    # Read the frozen split and slice [:N] (nested-safe: order is fixed in the file).
    with open(SPLITS_FILE) as f:
        split_names = json.load(f)[split]
    names = split_names[:n_segments]
    print(f"INFO: processing {len(names)} segment(s), stride={stride}, "
          f"cameras={cameras_spec}, skip_empty={skip_empty}")

    manifest = []
    class_counter = Counter()
    n_empty_skipped = 0

    for seg_name in names:
        seg_path = os.path.join(DATA_ROOT, cfg["dir"], seg_name + SUFFIX)
        if not os.path.exists(seg_path):
            print(f"WARNING: segment not found on disk, skipping: {seg_path}")
            continue

        raw_dataset = tf.data.TFRecordDataset(seg_path)
        for frame_idx, raw_record in enumerate(raw_dataset):
            if frame_idx % stride != 0:
                continue
            frame = open_dataset.Frame()
            frame.ParseFromString(raw_record.numpy())

            for cam_enum in cameras:
                # Match image and labels by .name (never by index).
                img_proto = next((im for im in frame.images if im.name == cam_enum), None)
                lbl_proto = next((cl for cl in frame.camera_labels if cl.name == cam_enum), None)
                if img_proto is None:
                    continue

                boxes = extract_boxes(lbl_proto)
                if skip_empty and not boxes:
                    n_empty_skipped += 1
                    continue

                pixels = tf.image.decode_jpeg(img_proto.image).numpy()
                h, w = pixels.shape[:2]
                cam_short = next(k for k, v in CAMERA_ALIASES.items() if v == cam_enum)
                fname = f"{seg_name}_f{frame_idx:03d}_{cam_short}.png"
                Image.fromarray(pixels).save(os.path.join(img_dir, fname))

                for b in boxes:
                    class_counter[IDX_TO_NAME[b[4]]] += 1

                manifest.append({
                    "image": f"images/{fname}",
                    "width": int(w),
                    "height": int(h),
                    "segment": seg_name,
                    "frame_idx": int(frame_idx),
                    "camera": cam_short,
                    "boxes": boxes,
                })

    with open(os.path.join(cache_path, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # --- Sanity report (the Perception analogue of Motion's SDC/target warnings) ---
    print(f"INFO: extraction complete. records={len(manifest)}, "
          f"empty_skipped={n_empty_skipped}")
    print(f"INFO: box counts by class: {dict(class_counter)}")
    for name in IDX_TO_NAME.values():
        if class_counter[name] == 0:
            print(f"WARNING: zero '{name}' boxes in this extraction.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-process Waymo Perception segments into a PNG + manifest cache."
    )
    parser.add_argument("--split", default="train", choices=list(SPLIT_CFG),
                        help="official Waymo split to process")
    parser.add_argument("--n-segments", type=int, default=6,
                        help="number of segments from the frozen split (slices [:N])")
    parser.add_argument("--stride", type=int, default=10,
                        help="temporal subsampling: keep 1 frame every STRIDE")
    parser.add_argument("--cameras", default="f",
                        help="'all' or comma-separated aliases, e.g. 'f' or 'f,sl,sr'")
    parser.add_argument("--keep-empty", action="store_true",
                        help="keep frames with no target boxes (default: skip them)")
    args = parser.parse_args()

    cameras_spec = "all" if args.cameras == "all" else args.cameras.split(",")
    run_extraction(
        split=args.split,
        n_segments=args.n_segments,
        stride=args.stride,
        cameras_spec=cameras_spec,
        skip_empty=not args.keep_empty,
    )