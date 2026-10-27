"""SSD model builders — scratch and ImageNet-pretrained-backbone (SSD300-VGG16).

One-stage, anchor-based detector requested for the architecture sweep. UNLIKE
``faster_rcnn.py`` and ``retinanet.py``, torchvision's only SSD is SSD300 on a
VGG16 backbone at a fixed 300x300 input — there is no ResNet-50-FPN SSD in the
library. That makes SSD a DELIBERATELY confounded comparison in this study; the
confounds are declared here so results are never read as a clean one-stage result:

  1. Backbone: VGG16 (not ResNet-50-FPN) — different feature extractor entirely.
  2. Input size: SSD resizes every frame to 300x300 internally, regardless of the
     --min-size/--max-size passed to the other detectors (which is why this builder
     ACCEPTS those args and ignores them). Downscaling a 1920x1280 Waymo or
     3848x2168 ZOD frame to 300px is severe; SSD300 is known-weak on the small,
     distant objects that dominate driving scenes. Expect low AP, especially for
     pedestrian/cyclist — this is the detector, not a bug.
  3. From-scratch SSD300 is a notoriously hard baseline (long schedules); the
     scratch arm here is for completeness, not a fair scratch-vs-pretrained point.

Use SSD as a "classic SSD data point", not as the matched one-stage counterpart to
Faster R-CNN — RetinaNet (same ResNet-50-FPN backbone) is that counterpart. If a
param-budget specialist is wanted instead (committee "version A"), prefer SSDlite
(MobileNetV3) via its own builder.

Builders:
  * ``build_ssd``            — VGG16 backbone from scratch (weights_backbone=None).
  * ``build_ssd_pretrained`` — VGG16 backbone from ImageNet-1k features.

Class index map — shared with the rest of the study:
    0 = background, 1 = Vehicle, 2 = Pedestrian, 3 = Cyclist
=> num_classes = 4. SSD uses a softmax multibox head with an explicit background
class at index 0, so the "+background" convention is literal here.
"""

from __future__ import annotations

from torchvision.models import VGG16_Weights
# Import both from the submodule: some torchvision versions don't re-export the
# `SSD` class at the `torchvision.models.detection` package level (only the
# `ssd300_vgg16` factory), which breaks a package-level `import SSD`. The
# submodule defines both, so this path is version-robust. `SSD` is used only as
# a return type hint here.
from torchvision.models.detection.ssd import SSD, ssd300_vgg16

NUM_CLASSES = 4  # 3 foreground classes + background (shared across the study)


def build_ssd(
    num_classes: int = NUM_CLASSES,
    trainable_backbone_layers: int = 5,
    min_size: int = 800,   # accepted for registry parity, IGNORED (SSD fixes 300)
    max_size: int = 1333,  # accepted for registry parity, IGNORED (SSD fixes 300)
) -> SSD:
    """Build a from-scratch SSD300 (VGG16, random init).

    Args:
        num_classes: detector classes including background (default 4).
        trainable_backbone_layers: VGG stages left trainable; 5 = all, for scratch.
        min_size / max_size: accepted so the registry can call every builder
            uniformly, but IGNORED — SSD300 resizes internally to 300x300.

    Returns:
        A ``torchvision.models.detection.SSD`` with no pretrained weights.
    """
    return ssd300_vgg16(
        weights=None,
        weights_backbone=None,        # from scratch — no ImageNet backbone
        num_classes=num_classes,
        trainable_backbone_layers=trainable_backbone_layers,
    )


def build_ssd_pretrained(
    num_classes: int = NUM_CLASSES,
    trainable_backbone_layers: int = 4,
    min_size: int = 800,   # accepted for registry parity, IGNORED (SSD fixes 300)
    max_size: int = 1333,  # accepted for registry parity, IGNORED (SSD fixes 300)
) -> SSD:
    """Build an SSD300 with an ImageNet-pretrained VGG16 backbone.

    Sibling of ``build_ssd`` differing only in the backbone prior: VGG16 starts
    from the ImageNet-1k FEATURES weights (the SSD-style, classifier-free VGG
    weights torchvision ships for exactly this purpose). The SSD extra layers and
    multibox head are still built from scratch.

    No GroupNorm/FrozenBN confound here: stock torchvision VGG16 has no BatchNorm
    layers, so the scratch-vs-pretrained contrast for SSD is a cleaner "prior
    only" difference than the ResNet arms — but the backbone/input-size confounds
    in the module docstring still dominate any cross-architecture reading.

    Args:
        num_classes: detector classes including background (default 4).
        trainable_backbone_layers: VGG stages left trainable (0-5); 4 fine-tunes
            most of VGG while leaving the earliest block frozen.
        min_size / max_size: accepted for registry parity, IGNORED (SSD fixes 300).

    Returns:
        A ``torchvision.models.detection.SSD`` with an ImageNet VGG16 backbone.

    Note:
        Input normalization is handled by SSD's own transform. Keep emitting
        unnormalized [0,1] tensors upstream — do NOT pre-normalize.
    """
    return ssd300_vgg16(
        weights=None,
        weights_backbone=VGG16_Weights.IMAGENET1K_FEATURES,  # ImageNet prior
        num_classes=num_classes,
        trainable_backbone_layers=trainable_backbone_layers,
    )