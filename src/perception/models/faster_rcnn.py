"""Faster R-CNN model builder — from scratch, GroupNorm backbone.

Returns a torchvision Faster R-CNN with a ResNet-50-FPN backbone, trained from
scratch (NO pretrained weights, backbone or head). This is a deliberate choice:
no ImageNet/COCO prior contaminates the specialist-vs-generalist comparison.

Why not the one-liner ``fasterrcnn_resnet50_fpn(weights=None)``: that factory
always builds the backbone with ``FrozenBatchNorm2d``. Frozen BN without
pretrained statistics is a normalization layer stuck at init that never updates,
so the backbone cannot learn. From scratch we must use a trainable norm. Detection
at full resolution fits only ~1-2 images per batch on 16 GB, and BatchNorm is
unstable at such tiny batches, so we use GroupNorm — the recipe from
"Rethinking ImageNet Pre-training" (He et al., 2018) for training detectors from
scratch. GroupNorm is batch-size independent.

Class index map (contiguous, background-implicit at 0):
    0 = background (reserved by the detector)
    1 = Vehicle, 2 = Pedestrian, 3 = Cyclist
=> num_classes = 4.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

NUM_CLASSES = 4  # 3 foreground classes + background


def _group_norm(num_channels: int, num_groups: int = 32) -> nn.GroupNorm:
    """GroupNorm factory for use as a backbone norm_layer.

    ResNet-50 channel counts (64, 128, 256, 512, 1024, 2048) are all divisible by
    32, so 32 groups is valid throughout the network.
    """
    return nn.GroupNorm(num_groups, num_channels)


def build_faster_rcnn(
    num_classes: int = NUM_CLASSES,
    trainable_backbone_layers: int = 5,
    min_size: int = 800,
    max_size: int = 1333,
) -> FasterRCNN:
    """Build a from-scratch Faster R-CNN (ResNet-50-FPN, GroupNorm).

    Args:
        num_classes: detector classes including background (default 4).
        trainable_backbone_layers: ResNet stages left trainable; 5 = all, as
            required when training from scratch (the factory default of 3 would
            freeze early stages).
        min_size / max_size: the detector's internal resize bounds
            (GeneralizedRCNNTransform). Lower them to speed up the warm-up.

    Returns:
        A ``torchvision.models.detection.FasterRCNN`` with no pretrained weights.
    """
    backbone = resnet_fpn_backbone(
        backbone_name="resnet50",
        weights=None,                 # from scratch — no ImageNet backbone
        norm_layer=_group_norm,       # GroupNorm instead of FrozenBatchNorm2d
        trainable_layers=trainable_backbone_layers,
    )
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,      # builds the box predictor head from scratch
        min_size=min_size,
        max_size=max_size,
    )
    return model
