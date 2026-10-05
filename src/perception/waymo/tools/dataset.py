"""PyTorch Dataset for the perception cache (PNG + manifest.json).

Dataset-agnostic by design: it reads the generic cache contract produced by the
Waymo preprocessor (and, later, any other dataset whose preprocessor emits the
same format), with no TensorFlow and no Waymo-specific knowledge. Each record is
one image plus its 2D boxes, already in corner format and absolute pixels.

Cache layout expected (one such directory per split):

    <cache_dir>/
    ├── images/        PNGs
    └── manifest.json  list of records

Manifest record schema (confirmed against the real artifact):

    {
      "image": "images/<name>.png",       # path relative to <cache_dir>
      "width": W, "height": H,
      "segment": "<base name>",            # provenance
      "frame_idx": N, "camera": "f",       # provenance
      "boxes": [[x1, y1, x2, y2, class_idx], ...]
    }

Box class indices use the contiguous map {Vehicle: 1, Pedestrian: 2, Cyclist: 3};
index 0 is reserved for background by the detector, so no class 0 appears here.

The target follows the torchvision detection contract: a dict with at least
``boxes`` (FloatTensor[N, 4], xyxy, pixels) and ``labels`` (Int64Tensor[N]).
Extra keys (``image_id``, ``area``, ``iscrowd``) are included so the same samples
feed a COCO-style mAP evaluator unchanged. Provenance keys are carried through
untouched, letting a prediction be traced back to its exact source frame.

Images are returned as FloatTensor[C, H, W] in [0, 1], UNNORMALIZED: the
torchvision detector normalizes and resizes internally via its
GeneralizedRCNNTransform, so normalizing here would double-normalize.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def image_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert a PIL RGB image to a float tensor [C, H, W] in [0, 1]."""
    arr = np.asarray(img, dtype=np.uint8)  # [H, W, C]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return tensor.float().div_(255.0)


class PerceptionCacheDataset(Dataset):
    """Reads a perception cache split (manifest.json + images/) for detection.

    Args:
        cache_dir: directory holding ``manifest.json`` and ``images/``.
        transform: optional callable ``(PIL.Image) -> Tensor[C, H, W]``.
            Defaults to :func:`image_to_tensor` (0-1, no normalization).
        drop_empty: if True, records with zero boxes are removed. Intended for
            the training split (torchvision's RPN expects targets); keep False
            for validation so empty scenes can measure false positives.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        drop_empty: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"manifest not found: {manifest_path}")

        with manifest_path.open() as f:
            records = json.load(f)

        if drop_empty:
            records = [r for r in records if r["boxes"]]
        self.records = records
        self.transform = transform or image_to_tensor

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]

        img = Image.open(self.cache_dir / rec["image"]).convert("RGB")
        image = self.transform(img)

        width, height = rec["width"], rec["height"]
        boxes_raw = rec["boxes"]

        if boxes_raw:
            arr = torch.tensor(boxes_raw, dtype=torch.float32)  # [N, 5]
            boxes = arr[:, :4].clone()
            labels = arr[:, 4].to(torch.int64)
            # Clamp to image bounds, then drop degenerate boxes (x2<=x1 or y2<=y1);
            # torchvision raises on degenerate boxes during training.
            boxes[:, 0::2].clamp_(min=0, max=width)
            boxes[:, 1::2].clamp_(min=0, max=height)
            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes, labels = boxes[keep], labels[keep]
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        if len(boxes):
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        else:
            area = torch.zeros((0,), dtype=torch.float32)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(idx, dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
            # Provenance — carried through untouched by the detector's transform.
            "segment": rec["segment"],
            "frame_idx": rec["frame_idx"],
            "camera": rec["camera"],
        }
        return image, target


def collate_fn(batch):
    """Collate detection samples, keeping images and targets as tuples.

    Detection images may differ in size, so they are not stacked into a single
    tensor; torchvision's detectors accept a list of images and a list of
    targets.
    """
    return tuple(zip(*batch))