"""Training entry point for torchvision 2D detectors (Perception warm-up).

Model-agnostic *within the torchvision contract*: builds a detector from the
model registry, trains on the Waymo perception cache, and evaluates COCO-AP on
val per --eval-ap-every epochs -- driving best.pt selection and AP-based
early-stopping. The green end-to-end gate (loss decreasing, inference yielding
predictions) still holds; AP is the real ruler on top of it.

Decision E: the metric stays in an independent module
(perception.tools.det2d.eval_ap). This loop imports its core and feeds it the
in-memory val predictions -- one ruler, two feeders (JSON dump there, in-memory
here). No metric logic is duplicated in the training loop.

Run (with PYTHONPATH=/workspace/src and the perception package __init__.py in
place):

    python /workspace/src/perception/models/train_vision_model.py \
        --model faster_rcnn --cache-root /workspace/datasets/waymo/perception_cache \
        --epochs 5 --batch-size 2
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset

# evaluate() retains every val target across the split; with the default
# file_descriptor sharing strategy each worker-origin tensor holds an open FD,
# exhausting `ulimit -n` on non-trivial val sets. file_system backs shared
# tensors by name instead -- the fix the RuntimeError itself recommends.
mp.set_sharing_strategy("file_system")

from perception.waymo.tools.dataset import PerceptionCacheDataset, collate_fn
from perception.models.faster_rcnn import (
    build_faster_rcnn,
    build_faster_rcnn_pretrained,
)
from perception.models.retinanet import (
    build_retinanet,
    build_retinanet_pretrained,
)
from perception.models.ssd import (
    build_ssd,
    build_ssd_pretrained,
)
from perception.models.fcos import (
    build_fcos,
    build_fcos_pretrained,
)
from perception.tools.det2d.eval_ap import evaluate_predictions


# ---------------------------------------------------------------------------
# Model registry: single extension point. Add a new detector with one line,
# e.g. "retinanet": build_retinanet. Builders must accept min_size/max_size.
# ---------------------------------------------------------------------------
MODELS = {
    "faster_rcnn": build_faster_rcnn,
    "faster_rcnn_pt": build_faster_rcnn_pretrained,
    "retinanet": build_retinanet,
    "retinanet_pt": build_retinanet_pretrained,
    "ssd": build_ssd,
    "ssd_pt": build_ssd_pretrained,
    "fcos": build_fcos,
    "fcos_pt": build_fcos_pretrained,
}

# Models carrying a pretrained backbone. Retained for reference/documentation;
# LR safety is now handled by the per-model default below, not by a guard.
PRETRAINED_MODELS = {"faster_rcnn_pt", "retinanet_pt", "ssd_pt", "fcos_pt"}

# Per-model default base LR, scaled to the locked bs=2 ladder protocol. Each
# entry is the architecture's published recipe LR (at the reference batch 16)
# transported to bs=2 by the linear scaling rule (lr proportional to batch,
# Goyal et al. 2017, "Accurate, Large Minibatch SGD"):
#   Faster R-CNN  0.02  (Detectron/torchvision ref)   / 8 -> 2.5e-3
#   RetinaNet     0.01  (Lin et al. 2017, Focal Loss)  / 8 -> 1.25e-3
#   FCOS          0.01  (Tian et al. 2019, FCOS)       / 8 -> 1.25e-3
#   SSD300-VGG16  1e-3  (Liu et al. 2016)              / 8 -> 1.25e-4
# Pretrained and from-scratch share the per-arch value: He et al. 2018
# ("Rethinking ImageNet Pre-training") trains from scratch at the SAME LR,
# needing only trainable norm (GroupNorm) + warmup, not a smaller step. The
# previous single default (2.5e-3) was the Faster R-CNN number applied blindly
# to every arch, which drove from-scratch RetinaNet to NaN. --lr overrides this
# per run; --lr-scale multiplies whatever base LR is resolved (for ablations).
MODEL_DEFAULT_LR = {
    "faster_rcnn": 2.5e-3,
    "faster_rcnn_pt": 2.5e-3,
    "retinanet": 1.25e-3,
    "retinanet_pt": 1.25e-3,
    "fcos": 1.25e-3,
    "fcos_pt": 1.25e-3,
    "ssd": 1.25e-4,
    "ssd_pt": 1.25e-4,
}

CLASS_NAMES = {1: "Vehicle", 2: "Pedestrian", 3: "Cyclist"}


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and Torch RNGs for single-seed reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(params, name: str, lr: float, momentum: float,
                    weight_decay: float):
    """Return an optimizer.

    From-scratch + GroupNorm pairs well with SGD; AdamW is the lower-variance
    fallback for tiny batches.
    """
    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum,
                               weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer: {name!r}")


def targets_to_device(targets, device):
    """Move tensor fields to the device; leave metadata untouched."""
    moved = []
    for t in targets:
        moved.append({
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in t.items()
        })
    return moved


def train_one_epoch(model, loader, optimizer, device, epoch, *,
                    scaler=None, use_amp=False, amp_dtype=None,
                    warmup_sched=None, warmup_iters=0,
                    global_step=0, grad_clip=0.0, log_every=20):
    """Run one training epoch. Returns (mean_losses_dict, global_step, n_nonfinite).

    AMP is decoupled from the scaler: BF16 autocast keeps FP32 dynamic range
    and needs no GradScaler, so `use_amp` can be True with `scaler is None`.
    A GradScaler is passed only for FP16.

    Stability nets (new): a non-finite (NaN/Inf) loss step is skipped instead of
    backpropagated -- otherwise it poisons every weight and the run never
    recovers. Finite steps are gradient-clipped to `grad_clip` global norm
    (0 disables). The count of skipped non-finite steps is returned so the
    caller can flag the run as diverged.
    """
    model.train()
    running = {}
    n = 0
    nonfinite_steps = 0
    for it, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        targets = targets_to_device(targets, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

        # Non-finite guard: a NaN/Inf loss would propagate through backprop into
        # every parameter and never clear. Skip the step, flag the run.
        if not torch.isfinite(loss):
            nonfinite_steps += 1
            if (it % log_every) == 0:
                print(f"  epoch {epoch} it {it:4d}/{len(loader)} "
                      f"NON-FINITE loss -> step skipped")
            global_step += 1
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                # Unscale BEFORE clipping so the norm is measured on true
                # (un-scaled) gradients -- clipping scaled grads is meaningless.
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if warmup_sched is not None and global_step < warmup_iters:
            warmup_sched.step()
        global_step += 1

        for k, v in loss_dict.items():
            running[k] = running.get(k, 0.0) + float(v.detach())
        running["loss_total"] = running.get("loss_total", 0.0) + float(loss.detach())
        n += 1

        if (it % log_every) == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  epoch {epoch} it {it:4d}/{len(loader)} "
                  f"lr {lr:.2e} loss {float(loss.detach()):.4f}")

    mean = {k: v / max(n, 1) for k, v in running.items()}
    mean.setdefault("loss_total", float("nan"))  # whole epoch skipped -> nan, no crash
    return mean, global_step, nonfinite_steps


@torch.no_grad()
def evaluate(model, loader, device, *, score_thresh=0.05):
    """Eval forward pass.

    Returns a list of {image_id, prediction, target}, eval-ready for an
    independent AP module (decision E seam). Predictions are filtered by a low
    score threshold and moved to CPU.
    """
    model.eval()
    results = []
    for images, targets in loader:
        images = [img.to(device) for img in images]
        preds = model(images)
        for img, pred, tgt in zip(images, preds, targets):
            keep = pred["scores"] >= score_thresh
            results.append({
                "image_id": tgt.get("image_id"),
                "height": int(img.shape[-2]),
                "width": int(img.shape[-1]),
                "prediction": {
                    "boxes": pred["boxes"][keep].cpu(),
                    "labels": pred["labels"][keep].cpu(),
                    "scores": pred["scores"][keep].cpu(),
                },
                "target": tgt,
            })
    return results


def results_to_frames(results):
    """Convert evaluate() output to eval_ap frames (plain lists, xyxy px).

    The single new seam between the training loop and the COCO-AP core: same
    ruler as the standalone eval_ap.py, fed in-memory instead of from JSON.
    """
    def _as_list(v):
        return v.tolist() if torch.is_tensor(v) else (v or [])

    frames = []
    for r in results:
        tgt, pred = r["target"], r["prediction"]
        frames.append({
            "image_id": int(tgt["image_id"]),
            "height": r.get("height", 0),
            "width": r.get("width", 0),
            "gt": {
                "boxes": tgt["boxes"].tolist(),
                "labels": tgt["labels"].tolist(),
                "area": _as_list(tgt.get("area")),
                "iscrowd": _as_list(tgt.get("iscrowd")),
            },
            "pred": {
                "boxes": pred["boxes"].tolist(),
                "labels": pred["labels"].tolist(),
                "scores": pred["scores"].tolist(),
            },
        })
    return frames


def save_pred_viz(image, prediction, out_path, *, score_thresh=0.5):
    """Minimal prediction overlay.

    Convenience only -- swap for the test_dataset_viz logic if preferred. Draws
    boxes above score_thresh on the unnormalized [C,H,W] image in [0,1].
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    img = image.permute(1, 2, 0).cpu().numpy()
    fig, ax = plt.subplots(1, figsize=(10, 7))
    ax.imshow(img)
    for box, label, score in zip(prediction["boxes"], prediction["labels"],
                                 prediction["scores"]):
        if float(score) < score_thresh:
            continue
        x1, y1, x2, y2 = (float(v) for v in box)
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                     fill=False, linewidth=2, edgecolor="lime"))
        name = CLASS_NAMES.get(int(label), str(int(label)))
        ax.text(x1, max(y1 - 4, 0), f"{name} {float(score):.2f}",
                color="black", fontsize=8,
                bbox=dict(facecolor="lime", edgecolor="none", pad=1))
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Perception warm-up trainer (Stage 1).")
    p.add_argument("--model", default="faster_rcnn", choices=sorted(MODELS))
    p.add_argument("--cache-root", type=Path, required=True,
                   help="Directory containing train/ and val/ perception caches.")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=None,
                   help="Base LR. Default: per-model value (MODEL_DEFAULT_LR), "
                        "i.e. the arch's reference recipe scaled to bs=2.")
    p.add_argument("--lr-scale", type=float, default=1.0,
                   help="Multiply the resolved base LR (explicit --lr or the "
                        "per-model default) by this factor. For LR-stability "
                        "ablations, e.g. --lr-scale 2.5.")
    p.add_argument("--grad-clip-norm", type=float, default=10.0,
                   help="Max global gradient norm for clipping (0 disables). "
                        "The stability net against NaN; AMP-aware (unscales "
                        "FP16 grads before measuring the norm).")
    p.add_argument("--optimizer", default="sgd", choices=["sgd", "adamw"])
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-iters", type=int, default=50,
                   help="Linear warmup length; clamped to one epoch.")
    p.add_argument("--min-size", type=int, default=800)
    p.add_argument("--max-size", type=int, default=1333)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--amp", action="store_true", help="Enable mixed precision.")
    p.add_argument("--amp-dtype", default="bfloat16",
                   choices=["bfloat16", "float16"],
                   help="Autocast dtype when --amp is set. bfloat16 (default, "
                        "Blackwell/5060 Ti-preferred, no GradScaler) or float16 "
                        "(adds a GradScaler). AP should match FP32 either way; "
                        "the difference is numeric robustness, not speed.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-id", default=None)
    p.add_argument("--runs-root", type=Path, default=Path("runs"))
    p.add_argument("--sweep-csv", type=Path, default=None,
                   help="If set, append one summary row (run_id,seed,best_ap,"
                        "best_epoch,epochs,early_stopped,diverged) here. One "
                        "line per run -- the seed-sweep ledger; header written "
                        "if absent.")
    p.add_argument("--save-epochs", default=None,
                   help="Comma-separated epoch numbers to snapshot, e.g. "
                        "'5,10,15,20' (1-based). Saves epoch_<e>.pt alongside "
                        "last.pt, so one run can be inspected at several stages "
                        "-- the SAME trajectory, not independent trainings. "
                        "Ensure --epochs >= max(save-epochs).")
    p.add_argument("--eval-ap-every", type=int, default=1,
                   help="Run COCO-AP on val every K epochs (always also on the "
                        "last). Small val -> 1; large val -> 2-5.")
    p.add_argument("--patience", type=int, default=3,
                   help="Early-stop after P consecutive evals with no AP gain "
                        ">min-delta. Counted in EVALUATIONS, not epochs. 0 off.")
    p.add_argument("--min-delta", type=float, default=1e-3,
                   help="Minimum AP gain counted as improvement; rides out "
                        "micro-plateaus (e.g. the +0.0017 e10->e15 dip).")
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-val", type=int, default=None)
    p.add_argument("--viz-pred", type=int, default=0,
                   help="Save N val prediction overlays after training.")
    p.add_argument("--viz-score-thresh", type=float, default=0.5)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # --- learning rate resolution -------------------------------------------
    # Base LR: explicit --lr wins; otherwise the per-model default (SGD-
    # calibrated, see MODEL_DEFAULT_LR). AdamW keeps a single conservative
    # default since the table encodes the SGD recipe. --lr-scale then multiplies
    # whatever base was chosen (ablation knob). The effective LR is stored in
    # args.json below, so every run records exactly what it trained with.
    if args.lr is not None:
        base_lr, lr_source = args.lr, "explicit --lr"
    elif args.optimizer == "sgd":
        base_lr, lr_source = MODEL_DEFAULT_LR[args.model], f"{args.model} default"
    else:
        base_lr, lr_source = 1e-4, "adamw default"
    args.lr = base_lr * args.lr_scale
    print(f"[lr] {lr_source} {base_lr:.3e}"
          + (f" x {args.lr_scale} -> {args.lr:.3e}"
             if args.lr_scale != 1.0 else f" -> {args.lr:.3e}"))

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA unavailable; falling back to CPU")
        args.device = "cpu"
    device = torch.device(args.device)

    run_id = args.run_id or f"{args.model}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = args.runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(
        json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    # --- data ---------------------------------------------------------------
    # drop_empty True on train (RPN dislikes empty targets), False on val
    # (empty images measure false positives). Decision C.
    train_ds = PerceptionCacheDataset(args.cache_root / "train", drop_empty=True)
    val_ds = PerceptionCacheDataset(args.cache_root / "val", drop_empty=False)
    if args.limit_train:
        train_ds = Subset(train_ds, range(min(args.limit_train, len(train_ds))))
    if args.limit_val:
        val_ds = Subset(val_ds, range(min(args.limit_val, len(val_ds))))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn)
    # num_workers=0 on val: evaluate() holds the whole split in memory, so
    # worker shared-memory tensors spike RAM (and swap) each eval. Single-process
    # removes the spike; val is small and GPU-bound, so throughput is unaffected.
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=0, collate_fn=collate_fn)

    # --- model / optim ------------------------------------------------------
    build = MODELS[args.model]
    model = build(min_size=args.min_size, max_size=args.max_size).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(params, args.optimizer, args.lr,
                                args.momentum, args.weight_decay)

    warmup_iters = (min(args.warmup_iters, len(train_loader))
                    if args.warmup_iters > 0 else 0)
    warmup_sched = None
    if warmup_iters > 0:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_iters)

    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    # GradScaler only for FP16; bfloat16 keeps FP32 range and needs none.
    scaler = (torch.amp.GradScaler("cuda")
              if (args.amp and args.amp_dtype == "float16") else None)

    print(f"[run] {run_id} | model={args.model} opt={args.optimizer} "
          f"lr={args.lr} bs={args.batch_size} epochs={args.epochs} "
          f"device={device} amp={args.amp}"
          f"({args.amp_dtype if args.amp else '-'}) warmup={warmup_iters} "
          f"grad_clip={args.grad_clip_norm}")
    print(f"[data] train={len(train_ds)} val={len(val_ds)}")

    # --- train --------------------------------------------------------------
    log_path = run_dir / "train_log.jsonl"
    save_epochs = {int(x) for x in args.save_epochs.split(",")
                   if x.strip()} if args.save_epochs else set()
    global_step = 0
    first_total = last_total = None
    best_ap, best_epoch, no_improve = -1.0, -1, 0
    total_nonfinite = 0   # non-finite (skipped) steps across the whole run
    last_eval_results = None  # reused post-loop to avoid a redundant val pass
    stopped_early = False

    for epoch in range(args.epochs):
        t0 = time.time()
        mean, global_step, nonfinite = train_one_epoch(
            model, train_loader, optimizer, device, epoch,
            scaler=scaler, use_amp=args.amp,
            amp_dtype=(amp_dtype if args.amp else None),
            warmup_sched=warmup_sched,
            warmup_iters=warmup_iters, global_step=global_step,
            grad_clip=args.grad_clip_norm)
        total_nonfinite += nonfinite
        dt = time.time() - t0
        mean["epoch"] = epoch
        mean["time_s"] = round(dt, 1)
        if nonfinite:
            mean["nonfinite_steps"] = nonfinite

        # --- val AP (decision E ruler; in memory, no per-epoch dump) --------
        is_last = (epoch + 1) == args.epochs
        do_eval = is_last or ((epoch + 1) % args.eval_ap_every == 0)
        ap = None
        if do_eval:
            te0 = time.time()
            last_eval_results = evaluate(model, val_loader, device)
            m = evaluate_predictions(results_to_frames(last_eval_results),
                                     verbose=False)
            # eval runs the full val every K epochs and is NOT part of time_s
            # (which is train-only); logging it separately makes the wall-clock
            # honest -- this is what made the n6 run "mysteriously" take 2h.
            mean["eval_time_s"] = round(time.time() - te0, 1)
            ap = m["AP"]
            mean["AP"], mean["AP50"], mean["AP75"] = ap, m["AP50"], m["AP75"]

        with log_path.open("a") as f:
            f.write(json.dumps(mean) + "\n")
        comp = " ".join(f"{k}={mean[k]:.4f}" for k in sorted(mean)
                        if k.startswith("loss_") and k != "loss_total")
        ap_str = f" AP={ap:.4f}" if ap is not None else ""
        nf_str = f" nonfinite={nonfinite}" if nonfinite else ""
        print(f"[epoch {epoch}] total={mean['loss_total']:.4f} {comp}"
              f"{ap_str}{nf_str} ({dt:.1f}s)")
        first_total = mean["loss_total"] if first_total is None else first_total
        last_total = mean["loss_total"]

        # stage snapshot: same run, same trajectory, fixed at this epoch.
        # epoch is 0-based here, so epoch_<e>.pt is saved after the e-th epoch.
        if (epoch + 1) in save_epochs:
            snap = run_dir / f"epoch_{epoch + 1}.pt"
            torch.save({"model": model.state_dict(), "args": vars(args)}, snap)
            print(f"[ckpt] saved {snap.name}")

        # --- best + early-stopping (AP, patience in evaluations) -----------
        if ap is not None:
            if ap > best_ap + args.min_delta:
                best_ap, best_epoch, no_improve = ap, epoch + 1, 0
                torch.save({"model": model.state_dict(), "args": vars(args)},
                           run_dir / "best.pt")
                print(f"[best] AP={best_ap:.4f} @ epoch {best_epoch} -> best.pt")
            else:
                no_improve += 1
                if args.patience and no_improve >= args.patience:
                    print(f"[early-stop] AP flat for {no_improve} evals "
                          f"(best={best_ap:.4f} @ epoch {best_epoch})")
                    stopped_early = True
                    break

    # --- checkpoint ---------------------------------------------------------
    torch.save({"model": model.state_dict(), "args": vars(args)},
               run_dir / "last.pt")

    # --- eval forward pass (green = inference runs and yields predictions) ---
    # reuse the last in-loop AP eval if present; only re-run if we never evaluated.
    results = (last_eval_results if last_eval_results is not None
               else evaluate(model, val_loader, device))
    n_preds = sum(len(r["prediction"]["boxes"]) for r in results)
    print(f"[eval] images={len(results)} kept_preds={n_preds}")

    # --- optional prediction viz -------------------------------------------
    if args.viz_pred > 0:
        viz_dir = run_dir / "viz_pred"
        viz_dir.mkdir(exist_ok=True)
        k = min(args.viz_pred, len(results))
        for i in range(k):
            image, _ = val_ds[i]  # index-aligned with results (shuffle=False)
            save_pred_viz(image, results[i]["prediction"],
                          viz_dir / f"pred_{i:03d}.png",
                          score_thresh=args.viz_score_thresh)
        print(f"[viz] wrote {k} overlays to {viz_dir}")

    # --- green verdict ------------------------------------------------------
    if first_total is not None and last_total is not None:
        delta = first_total - last_total
        verdict = "GREEN (loss decreased)" if delta > 0 else "loss did NOT decrease"
        print(f"[verdict] first={first_total:.4f} last={last_total:.4f} "
              f"delta={delta:+.4f} -> {verdict}")
    if total_nonfinite:
        print(f"[warn] {total_nonfinite} non-finite step(s) skipped this run "
              f"-> flagged diverged in the ledger")

    # --- AP summary (decision E ruler, now wired in-loop) ------------------
    # The metric lives in perception.tools.det2d.eval_ap (one ruler); the loop
    # feeds it in-memory per --eval-ap-every and tracks best.pt + early-stop.
    if best_epoch > 0:
        tail = " | early-stopped" if stopped_early else ""
        print(f"[summary] best AP={best_ap:.4f} @ epoch {best_epoch} "
              f"(best.pt){tail}")

    # --- sweep ledger: one row per run, appended to a shared CSV -----------
    # Keeps all seeds' best AP in one place so ranking is a sort, not a grep
    # across N logs. Checkpoints are never pruned -- the full distribution is
    # the citable result (mean/std over seeds), not the single best. The
    # `diverged` column flags runs that hit >=1 non-finite step (even if the
    # guard let them recover), so plots can separate training-failure from
    # genuine low AP instead of averaging a dead seed in as 0.
    if args.sweep_csv is not None:
        ap_str = f"{best_ap:.4f}" if best_epoch > 0 else "nan"
        diverged = int(total_nonfinite > 0)
        row = (f"{run_id},{args.seed},{ap_str},{best_epoch},"
               f"{args.epochs},{int(stopped_early)},{diverged}")
        args.sweep_csv.parent.mkdir(parents=True, exist_ok=True)
        new_file = not args.sweep_csv.exists()
        with args.sweep_csv.open("a") as f:
            if new_file:
                f.write("run_id,seed,best_ap,best_epoch,epochs,"
                        "early_stopped,diverged\n")
            f.write(row + "\n")
        print(f"[sweep] appended row to {args.sweep_csv}")


if __name__ == "__main__":
    main()