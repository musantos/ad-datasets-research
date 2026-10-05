"""Training entry point for torchvision 2D detectors (Perception warm-up).

Model-agnostic *within the torchvision contract*: builds a detector from the
model registry, trains on the Waymo perception cache, and evaluates COCO-AP on
val per --eval-ap-every epochs -- driving best.pt selection and AP-based
early-stopping. The green end-to-end gate (loss decreasing, inference yielding
predictions) still holds; AP is the real ruler on top of it.

Scope: torchvision detectors only (shared train/eval contract). LiDAR / 3D
detection does not fit this harness -- it lives in a separate framework
(e.g. OpenPCDet, mmdetection3d) with its own training loop. Hence the explicit
`vision_model` in the filename rather than a generic `train.py`.

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
from torch.utils.data import DataLoader, Subset

from perception.waymo.tools.dataset import PerceptionCacheDataset, collate_fn
from perception.models.faster_rcnn import build_faster_rcnn
from perception.tools.det2d.eval_ap import evaluate_predictions


# ---------------------------------------------------------------------------
# Model registry: single extension point. Add a new detector with one line,
# e.g. "retinanet": build_retinanet. Builders must accept min_size/max_size.
# ---------------------------------------------------------------------------
MODELS = {
    "faster_rcnn": build_faster_rcnn,
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
                    scaler=None, warmup_sched=None, warmup_iters=0,
                    global_step=0, log_every=20):
    """Run one training epoch. Returns (mean_losses_dict, global_step)."""
    model.train()
    running = {}
    n = 0
    use_amp = scaler is not None
    for it, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        targets = targets_to_device(targets, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
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
    return mean, global_step


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
                   help="Base LR. Defaults to 2.5e-3 (sgd) / 1e-4 (adamw).")
    p.add_argument("--optimizer", default="sgd", choices=["sgd", "adamw"])
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-iters", type=int, default=50,
                   help="Linear warmup length; clamped to one epoch.")
    p.add_argument("--min-size", type=int, default=800)
    p.add_argument("--max-size", type=int, default=1333)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--amp", action="store_true", help="Enable mixed precision.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-id", default=None)
    p.add_argument("--runs-root", type=Path, default=Path("runs"))
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

    if args.lr is None:
        args.lr = 2.5e-3 if args.optimizer == "sgd" else 1e-4

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
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn)

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

    scaler = torch.amp.GradScaler("cuda") if args.amp else None

    print(f"[run] {run_id} | model={args.model} opt={args.optimizer} "
          f"lr={args.lr} bs={args.batch_size} epochs={args.epochs} "
          f"device={device} amp={bool(scaler)} warmup={warmup_iters}")
    print(f"[data] train={len(train_ds)} val={len(val_ds)}")

    # --- train --------------------------------------------------------------
    log_path = run_dir / "train_log.jsonl"
    save_epochs = {int(x) for x in args.save_epochs.split(",")
                   if x.strip()} if args.save_epochs else set()
    global_step = 0
    first_total = last_total = None
    best_ap, best_epoch, no_improve = -1.0, -1, 0
    last_eval_results = None  # reused post-loop to avoid a redundant val pass
    stopped_early = False

    for epoch in range(args.epochs):
        t0 = time.time()
        mean, global_step = train_one_epoch(
            model, train_loader, optimizer, device, epoch,
            scaler=scaler, warmup_sched=warmup_sched,
            warmup_iters=warmup_iters, global_step=global_step)
        dt = time.time() - t0
        mean["epoch"] = epoch
        mean["time_s"] = round(dt, 1)

        # --- val AP (decision E ruler; in memory, no per-epoch dump) --------
        is_last = (epoch + 1) == args.epochs
        do_eval = is_last or ((epoch + 1) % args.eval_ap_every == 0)
        ap = None
        if do_eval:
            last_eval_results = evaluate(model, val_loader, device)
            m = evaluate_predictions(results_to_frames(last_eval_results),
                                     verbose=False)
            ap = m["AP"]
            mean["AP"], mean["AP50"], mean["AP75"] = ap, m["AP50"], m["AP75"]

        with log_path.open("a") as f:
            f.write(json.dumps(mean) + "\n")
        comp = " ".join(f"{k}={mean[k]:.4f}" for k in sorted(mean)
                        if k.startswith("loss_") and k != "loss_total")
        ap_str = f" AP={ap:.4f}" if ap is not None else ""
        print(f"[epoch {epoch}] total={mean['loss_total']:.4f} {comp}"
              f"{ap_str} ({dt:.1f}s)")
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

    # --- AP summary (decision E ruler, now wired in-loop) ------------------
    # The metric lives in perception.tools.det2d.eval_ap (one ruler); the loop
    # feeds it in-memory per --eval-ap-every and tracks best.pt + early-stop.
    if best_epoch > 0:
        tail = " | early-stopped" if stopped_early else ""
        print(f"[summary] best AP={best_ap:.4f} @ epoch {best_epoch} "
              f"(best.pt){tail}")


if __name__ == "__main__":
    main()