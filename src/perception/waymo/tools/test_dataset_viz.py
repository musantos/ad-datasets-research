"""Visual test for PerceptionCacheDataset.

Validates the dataset's *output* (not the manifest): it draws the target boxes on
the very image tensor the dataset returns, so any error in the tensor conversion
or the box pipeline (clamp / degenerate filter / corner format) shows up as a
misaligned box. Writes annotated PNGs to inspect on the host; no model involved.

Usage (in gpu_env):
    python test_dataset_viz.py --cache /workspace/datasets/waymo/perception_cache/train \
        --out /workspace/_viz_dataset --n 6
"""

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from dataset import PerceptionCacheDataset

CLASS_NAMES = {1: "vehicle", 2: "pedestrian", 3: "cyclist"}
CLASS_COLORS = {1: (0, 200, 0), 2: (255, 160, 0), 3: (0, 160, 255)}


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    """Invert the dataset transform: FloatTensor[C,H,W] in [0,1] -> PIL RGB."""
    arr = (image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy())
    return Image.fromarray(arr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="cache split dir (has manifest.json + images/)")
    ap.add_argument("--out", default="_viz_dataset", help="output dir for annotated PNGs")
    ap.add_argument("--n", type=int, default=6, help="number of samples to render")
    args = ap.parse_args()

    ds = PerceptionCacheDataset(args.cache)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"dataset: {len(ds)} records | rendering {min(args.n, len(ds))}")

    for i in range(min(args.n, len(ds))):
        image, target = ds[i]
        pil = tensor_to_pil(image)
        draw = ImageDraw.Draw(pil)

        for box, label in zip(target["boxes"].tolist(), target["labels"].tolist()):
            x1, y1, x2, y2 = box
            color = CLASS_COLORS.get(label, (255, 0, 0))
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text((x1 + 2, max(0, y1 - 12)), CLASS_NAMES.get(label, str(label)), fill=color)

        name = f"{i:03d}_{target['segment']}_f{target['frame_idx']}_{target['camera']}.png"
        pil.save(out / name)
        print(f"  [{i}] {tuple(image.shape)} | {len(target['boxes'])} boxes -> {name}")

    print(f"done -> {out.resolve()}")


if __name__ == "__main__":
    main()
