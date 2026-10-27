"""RetinaNet model builders — from-scratch (GroupNorm) and ImageNet-pretrained.

One-stage, anchor-based counterpart to the Faster R-CNN two-stage pair, kept on
the SAME backbone (ResNet-50-FPN) so the only intended cross-family variable is
the detector head, not the feature extractor. Mirrors ``faster_rcnn.py``:

  * ``build_retinanet``            — ResNet-50-FPN from scratch, GroupNorm.
  * ``build_retinanet_pretrained`` — ResNet-50-FPN, ImageNet-1k backbone,
                                     FrozenBatchNorm2d.

Same scratch-vs-pretrained rationale as Faster R-CNN (see that file): the
torchvision factory ``retinanet_resnet50_fpn`` always builds the backbone with
FrozenBatchNorm2d, which cannot learn from random init, so the scratch arm is
hand-built with GroupNorm — the "Rethinking ImageNet Pre-training" recipe
(He et al., 2018), batch-size independent and stable at the bs=1-2 we are forced
into at full resolution. The SAME declared norm confound therefore carries over:
pretrained+FrozenBN vs scratch+GroupNorm (prior + norm, two coupled variables).

FPN difference from Faster R-CNN (architectural, NOT a protocol knob): RetinaNet's
head and default anchor generator expect the 5-level P3-P7 pyramid, so the
backbone is built with ``returned_layers=[2, 3, 4]`` and an ``LastLevelP6P7``
extra block — exactly what the torchvision factory does internally. Faster R-CNN
instead uses the default P2-P5 + LastLevelMaxPool. This is intrinsic to the
architecture and is kept identical across both RetinaNet arms.

Class index map — identical to Faster R-CNN on purpose (shared dataset, labels
and eval ruler):
    0 = background, 1 = Vehicle, 2 = Pedestrian, 3 = Cyclist
=> num_classes = 4. torchvision counts the background slot in RetinaNet's
num_classes as well (per the official docs), so channel 0 simply never receives a
positive target — one unused channel, and the label space stays {1,2,3} across
every model in the study.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops.feature_pyramid_network import LastLevelP6P7

NUM_CLASSES = 4  # 3 foreground classes + background (shared with faster_rcnn.py)

# RetinaNet's P3-P7 pyramid: skip C1 (returned_layers starts at 2) and append
# P6/P7 on top of the FPN. NOTE (version-sensitive, the one line to check if the
# smoke-test throws a channel mismatch): current torchvision feeds P6 from the
# 256-channel FPN output -> LastLevelP6P7(256, 256). Older torchvision fed it
# from C5 -> LastLevelP6P7(2048, 256). Flip these two ints if needed.
_RETURNED_LAYERS = [2, 3, 4]
_P6P7_IN, _P6P7_OUT = 256, 256


def _group_norm(num_channels: int, num_groups: int = 32) -> nn.GroupNorm:
    """GroupNorm factory for use as a backbone norm_layer.

    ResNet-50 channel counts (64, 128, 256, 512, 1024, 2048) are all divisible by
    32, so 32 groups is valid throughout the network.
    """
    return nn.GroupNorm(num_groups, num_channels)


def build_retinanet(
    num_classes: int = NUM_CLASSES,
    trainable_backbone_layers: int = 5,
    min_size: int = 800,
    max_size: int = 1333,
) -> RetinaNet:
    """Build a from-scratch RetinaNet (ResNet-50-FPN, GroupNorm, P3-P7).

    Args:
        num_classes: detector classes including background (default 4).
        trainable_backbone_layers: ResNet stages left trainable; 5 = all, as
            required when training from scratch.
        min_size / max_size: detector resize bounds (GeneralizedRCNNTransform),
            kept identical to the Faster R-CNN arms for protocol parity.

    Returns:
        A ``torchvision.models.detection.RetinaNet`` with no pretrained weights.
    """
    backbone = resnet_fpn_backbone(
        backbone_name="resnet50",
        weights=None,                 # from scratch — no ImageNet backbone
        norm_layer=_group_norm,       # GroupNorm instead of FrozenBatchNorm2d
        trainable_layers=trainable_backbone_layers,
        returned_layers=_RETURNED_LAYERS,
        extra_blocks=LastLevelP6P7(_P6P7_IN, _P6P7_OUT),
    )
    model = RetinaNet(
        backbone,
        num_classes=num_classes,      # builds the classification/regression heads
        min_size=min_size,
        max_size=max_size,
    )
    return model


def build_retinanet_pretrained(
    num_classes: int = NUM_CLASSES,
    trainable_backbone_layers: int = 3,
    min_size: int = 800,
    max_size: int = 1333,
) -> RetinaNet:
    """Build an ImageNet-pretrained RetinaNet (ResNet-50-FPN, FrozenBN, P3-P7).

    Sibling of ``build_retinanet`` differing only in the backbone prior: the
    ResNet-50 stem+body start from ImageNet-1k weights. FPN (incl. P6/P7) and the
    detection heads are still built from scratch, isolating the backbone prior as
    the arm's single intended variable — except the norm layer (see below).

    NORM CONFOUND (declared, not hidden): the pretrained ResNet weights carry
    BatchNorm running stats, exposed as FrozenBatchNorm2d (the resnet_fpn_backbone
    default used here when a norm_layer is not given). The from-scratch arm uses
    GroupNorm. The comparison is therefore pretrained+FrozenBN vs scratch+GN —
    the same coupled pair reported for Faster R-CNN.

    Args:
        num_classes: detector classes including background (default 4).
        trainable_backbone_layers: ResNet stages left trainable, from the top.
            3 (default) fine-tunes layer2/3/4 and freezes conv1+layer1.
        min_size / max_size: detector resize bounds, identical to the scratch arm.

    Returns:
        A ``torchvision.models.detection.RetinaNet`` with an ImageNet backbone.

    Note:
        Input normalization is left to the detector's default
        GeneralizedRCNNTransform, whose image_mean/image_std ARE the ImageNet
        stats the pretrained features expect. Keep emitting unnormalized [0,1]
        tensors upstream — do NOT pre-normalize.
    """
    backbone = resnet_fpn_backbone(
        backbone_name="resnet50",
        weights=ResNet50_Weights.IMAGENET1K_V1,  # ImageNet prior (backbone only)
        # norm_layer omitted -> FrozenBatchNorm2d (default), carrying pretrained
        # BN running stats; stable at tiny batch size.
        trainable_layers=trainable_backbone_layers,
        returned_layers=_RETURNED_LAYERS,
        extra_blocks=LastLevelP6P7(_P6P7_IN, _P6P7_OUT),
    )
    model = RetinaNet(
        backbone,
        num_classes=num_classes,      # detection heads still from scratch
        min_size=min_size,
        max_size=max_size,
    )
    return model