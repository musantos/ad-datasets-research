"""Faster R-CNN model builders — from-scratch (GroupNorm) and ImageNet-pretrained.

Two sibling builders for the specialist-vs-generalist comparison, differing only
in the backbone prior:

  * ``build_faster_rcnn``            — ResNet-50-FPN from scratch, GroupNorm.
  * ``build_faster_rcnn_pretrained`` — ResNet-50-FPN, ImageNet-1k backbone,
                                       FrozenBatchNorm2d.

Why not the one-liner ``fasterrcnn_resnet50_fpn(weights=None)`` for the scratch
arm: that factory always builds the backbone with ``FrozenBatchNorm2d``. Frozen
BN without pretrained statistics is a normalization layer stuck at init that
never updates, so the backbone cannot learn. From scratch we must use a trainable
norm. Detection at full resolution fits only ~1-2 images per batch on 16 GB, and
BatchNorm is unstable at such tiny batches, so we use GroupNorm — the recipe from
"Rethinking ImageNet Pre-training" (He et al., 2018) for training detectors from
scratch. GroupNorm is batch-size independent.

Class index map (contiguous, background-implicit at 0):
    0 = background (reserved by the detector)
    1 = Vehicle, 2 = Pedestrian, 3 = Cyclist
=> num_classes = 4.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet50_Weights
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


def build_faster_rcnn_pretrained(
    num_classes: int = NUM_CLASSES,
    trainable_backbone_layers: int = 3,
    min_size: int = 800,
    max_size: int = 1333,
) -> FasterRCNN:
    """Build an ImageNet-pretrained Faster R-CNN (ResNet-50-FPN, FrozenBN).

    Sibling of ``build_faster_rcnn`` differing only in the backbone prior: the
    ResNet-50 stem+body start from ImageNet-1k weights instead of random init.
    FPN, RPN and the box head are still built from scratch, isolating the
    backbone prior as the arm's single intended variable -- except the norm
    layer, see below.

    NORM CONFOUND (declared, not hidden): torchvision's pretrained ResNet
    weights carry BatchNorm running stats, exposed as ``FrozenBatchNorm2d`` (the
    factory default used here). The from-scratch arm uses GroupNorm, because
    trainable BatchNorm is unstable at bs=2. FrozenBN sidesteps that (it never
    updates from batches), but the comparison is therefore pretrained+FrozenBN
    vs scratch+GN -- two coupled variables (prior + norm). Pragmatic and
    reported; a scratch+BN control arm would be needed to fully isolate norm.

    Args:
        num_classes: detector classes including background (default 4).
        trainable_backbone_layers: ResNet stages left trainable, counted from
            the top. 3 (default) fine-tunes layer2/3/4 and freezes conv1+layer1
            -- the classic partial fine-tune; 5 would unfreeze everything.
        min_size / max_size: detector resize bounds, kept identical to the
            from-scratch arm for protocol parity.

    Returns:
        A ``torchvision.models.detection.FasterRCNN`` with an ImageNet backbone.

    Note:
        Input normalization is left to the detector's default
        GeneralizedRCNNTransform, whose image_mean/image_std ARE the ImageNet
        stats the pretrained features expect. The dataset must keep emitting
        unnormalized [0,1] tensors (as it does) -- do NOT pre-normalize upstream.
    """
    backbone = resnet_fpn_backbone(
        backbone_name="resnet50",
        weights=ResNet50_Weights.IMAGENET1K_V1,  # ImageNet prior (backbone only)
        # norm_layer omitted -> FrozenBatchNorm2d (factory default), carrying the
        # pretrained BN running stats; stable at tiny batch size.
        trainable_layers=trainable_backbone_layers,
    )
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,      # box predictor head still from scratch
        min_size=min_size,
        max_size=max_size,
    )
    return model