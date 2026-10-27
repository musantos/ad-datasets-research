"""Central style + arm-resolution for the det2d plotting scripts.

Single source of truth so every figure (scaling, curves, PR, confusion) shows
the *same* model with the *same* display name, colour and marker. Previously
each plot keyed colour/legend off the arm's role token (``base`` / ``pretrained``),
so a cross-campaign comparison of different architectures came out with the
wrong names and collapsed colours. Here the identity is derived from the model
key itself (``retinanet``, ``ssd_pt``, ...), which is stable across campaigns
and already encoded in every run dir and ledger filename.

Two things this module provides:

1. Per-model STYLE, by rule, so a new model (FCOS) needs no per-figure edit:
     * hue  = architecture family  (retinanet=blue, ssd=green, frcnn=orange, fcos=purple)
     * shade= prior                (pretrained = dark, from-scratch = light)
     * marker = prior              (pretrained = filled circle, scratch = triangle)
     * linestyle = solid (always, unless overridden per arm)
   Any unmapped key falls back to a neutral grey + its raw name, never crashing.

2. ARM RESOLUTION from a compact spec, so a figure names arms by WHERE they ran
   and WHICH model, not by a hand-typed glob:
     * ``exp_id:model``            -> model-level arm (scaling: N comes from --steps)
     * ``exp_id:model_nN_tag``     -> point-level arm (curves/PR/confusion)
     * ``label=glob``              -> legacy form, still accepted (model inferred
                                      from the glob for colour when possible)
   Name / colour / linestyle can be overridden per arm by the caller.

Colours use matplotlib's named colours (blue, orange, ...) so a human override
on the command line is a word, not a hex code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# --- family table -----------------------------------------------------------
# key family -> (display name, light colour [from-scratch], dark colour [pretrained]).
# To mark SSD visually as the declared confound, swap its row for a grey pair,
# e.g. ("SSD", "lightgrey", "dimgrey") -- one line, no other change needed.
FAMILY_STYLE = {
    "faster_rcnn": ("Faster R-CNN", "sandybrown",     "darkorange"),
    "retinanet":   ("RetinaNet",    "cornflowerblue", "navy"),
    "ssd":         ("SSD",          "mediumseagreen", "darkgreen"),
    "fcos":        ("FCOS",         "mediumpurple",   "indigo"),
}
FALLBACK_COLOR = "grey"
MARKER_PRETRAINED = "o"   # filled circle
MARKER_SCRATCH = "^"      # triangle
DEFAULT_LINESTYLE = "solid"

# A model key is "<family>" or "<family>_pt". Split the prior suffix off the key.
_PT_SUFFIX = "_pt"
# Detect a point token "..._n<N>_<tag>" inside an arm spec (e.g. retinanet_n96_s8).
_POINT_RE = re.compile(r"^(?P<model>.+?)_n(?P<n>\d+)_(?P<tag>[^_/]+)$")


def split_prior(model: str) -> tuple[str, bool]:
    """'retinanet_pt' -> ('retinanet', True);  'ssd' -> ('ssd', False)."""
    if model.endswith(_PT_SUFFIX):
        return model[: -len(_PT_SUFFIX)], True
    return model, False


def display_name(model: str) -> str:
    """Pretty label for a model key, e.g. 'retinanet_pt' -> 'RetinaNet (pretrained)'."""
    family, pretrained = split_prior(model)
    base = FAMILY_STYLE.get(family, (family, None, None))[0]
    return f"{base} (pretrained)" if pretrained else base


def color_for(model: str) -> str:
    """Named colour: hue by family, shade by prior. Grey for unmapped families."""
    family, pretrained = split_prior(model)
    row = FAMILY_STYLE.get(family)
    if row is None:
        return FALLBACK_COLOR
    _, light, dark = row
    return dark if pretrained else light


def marker_for(model: str) -> str:
    """Filled circle for pretrained, triangle for from-scratch."""
    _, pretrained = split_prior(model)
    return MARKER_PRETRAINED if pretrained else MARKER_SCRATCH


# --- arm resolution ---------------------------------------------------------
@dataclass
class Arm:
    """A resolved comparison arm: its identity, style and how to find it on disk.

    ``model`` may be None for a legacy ``label=glob`` arm whose model could not
    be inferred from the glob; style then falls back to grey + the given label.
    """
    label: str
    model: Optional[str]
    color: str
    marker: str
    linestyle: str
    exp_id: Optional[str] = None       # new-format arms
    point: Optional[str] = None        # "nN_tag" for point-level arms
    legacy_glob: Optional[str] = None  # set only for label=glob arms
    _raw: str = field(default="", repr=False)

    # -- path flavours (new-format arms) --
    def run_glob(self, runs_base) -> str:
        """Per-seed run dirs: <runs_base>/<exp>/<model>_<point>_seed*."""
        if self.legacy_glob is not None:
            return _resolve_legacy(self.legacy_glob, runs_base)
        return str(Path(runs_base) / self.exp_id / f"{self.model}_{self.point}_seed*")

    def ledger_path(self, runs_base, point: str) -> Path:
        """Sweep ledger for a given point: <runs_base>/<exp>/sweep_<model>_<point>.csv.

        ``point`` is passed explicitly because scaling arms are model-level and
        iterate over several N; point-level arms ignore the argument's N and use
        their own if set.
        """
        pt = self.point or point
        model = self.model or self.legacy_glob  # legacy scaling: --arm label=model
        base = Path(runs_base) / self.exp_id if self.exp_id else Path(runs_base)
        return base / f"sweep_{model}_{pt}.csv"

    def pred_glob(self, runs_base) -> str:
        """predictions.json per seed: <run_glob>/infer_val/predictions.json."""
        if self.legacy_glob is not None:
            return _resolve_legacy(self.legacy_glob, runs_base)
        return str(Path(runs_base) / self.exp_id
                   / f"{self.model}_{self.point}_seed*" / "infer_val" / "predictions.json")


def _resolve_legacy(glob_str: str, runs_base) -> str:
    """Legacy globs: an absolute path is used as-is; a relative one resolves
    against runs_base, matching the old --runs-root behaviour so the existing
    pipeline's per-step auto-plots keep working unchanged."""
    p = Path(glob_str)
    return glob_str if p.is_absolute() else str(Path(runs_base) / p)


def _infer_model_from_glob(glob_str: str) -> Optional[str]:
    """Best-effort: pull a known model key out of a legacy glob for colouring."""
    # Prefer the longest match so 'retinanet_pt' wins over 'retinanet'.
    candidates = []
    for family in FAMILY_STYLE:
        for key in (f"{family}{_PT_SUFFIX}", family):
            if key in glob_str:
                candidates.append(key)
    return max(candidates, key=len) if candidates else None


def resolve_arm(spec: str, *, name=None, color=None, linestyle=None) -> Arm:
    """Parse one arm spec into an Arm, applying any per-arm overrides.

    Accepts:
      * ``exp_id:model``          -> model-level (scaling)
      * ``exp_id:model_nN_tag``   -> point-level (curves/PR/confusion)
      * ``label=glob``            -> legacy
    Precedence for name/colour/linestyle: explicit override > derived-from-model
    > neutral fallback.
    """
    spec = spec.strip()

    # Legacy "label=glob": keep working; infer model only to pick a colour.
    if "=" in spec and ":" not in spec.split("=", 1)[0]:
        label, glob_str = (s.strip() for s in spec.split("=", 1))
        model = _infer_model_from_glob(glob_str)
        return Arm(
            label=name or label,
            model=model,
            color=color or (color_for(model) if model else FALLBACK_COLOR),
            marker=marker_for(model) if model else MARKER_PRETRAINED,
            linestyle=linestyle or DEFAULT_LINESTYLE,
            legacy_glob=glob_str,
            _raw=spec,
        )

    # New "exp_id:..." form.
    if ":" not in spec:
        raise ValueError(
            f"arm spec must be 'exp_id:model[_nN_tag]' or legacy 'label=glob', got {spec!r}")
    exp_id, rest = (s.strip() for s in spec.split(":", 1))

    m = _POINT_RE.match(rest)
    if m:                                   # point-level: exp:model_nN_tag
        model = m.group("model")
        point = f"n{m.group('n')}_{m.group('tag')}"
    else:                                   # model-level: exp:model (scaling)
        model = rest
        point = None

    return Arm(
        label=name or display_name(model),
        model=model,
        color=color or color_for(model),
        marker=marker_for(model),
        linestyle=linestyle or DEFAULT_LINESTYLE,
        exp_id=exp_id,
        point=point,
        _raw=spec,
    )


def resolve_arms(specs, names=None, colors=None, linestyles=None) -> list[Arm]:
    """Resolve a list of specs with positional override lists (None-padded)."""
    specs = list(specs or [])
    def pad(xs):
        xs = list(xs or [])
        return xs + [None] * (len(specs) - len(xs))
    names, colors, linestyles = pad(names), pad(colors), pad(linestyles)
    return [resolve_arm(s, name=n, color=c, linestyle=ls)
            for s, n, c, ls in zip(specs, names, colors, linestyles)]


# --- shared divergence heuristic --------------------------------------------
# Pre-fix ledgers have no `diverged` column: a run that hit NaN was saved as
# best_ap≈0 @ best_epoch==1 (the AP=0 first eval beat the -1 init and wrote
# best.pt at epoch 1). Post-fix ledgers carry `diverged` explicitly. This reads
# the column when present, else applies the epoch-1 heuristic.
DIVERGE_AP_EPS = 5e-4   # best_ap at or below this AND best_epoch==1 => dead


def seed_is_diverged(row: dict) -> bool:
    """True if a sweep-ledger row represents a diverged/dead training run."""
    col = (row.get("diverged") or "").strip()
    if col:                                  # post-fix: trust the column
        return col == "1"
    # pre-fix heuristic
    try:
        ap = float((row.get("best_ap") or "nan").strip())
    except ValueError:
        return True
    if ap != ap:                             # literal nan
        return True
    try:
        best_epoch = int(float((row.get("best_epoch") or "-1").strip()))
    except ValueError:
        best_epoch = -1
    return ap <= DIVERGE_AP_EPS and best_epoch <= 1
