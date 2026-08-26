import os
import sys
import time
import csv
import random
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from src.core.waymo_pytorch_dataset_agentcentric import WaymoMotionDatasetAgentCentric
from src.motion.sequential_model import SequentialTrajectoryPredictor

# Mapping of the Object.Type enum from the Waymo proto (same as the baseline).
OBJECT_TYPE_NAMES = {
    0: "UNSET",
    1: "VEHICLE",
    2: "PEDESTRIAN",
    3: "CYCLIST",
    4: "OTHER",
}

NUM_MODES = 6          # matches max_predictions=6 from the official config
EPOCHS = 300           # ceiling only; early stopping ends training before this
PATIENCE = 10          # stop if Val(best mode) does not improve for this many epochs
BATCH_SIZE = 64
LR = 1e-3

# ITEM 4 -- rich input, FIXED across both grid arms so the ONLY difference
# between {SDC-centric, agent-centric} is the normalization, not the input
# richness. heading expands to (sin, cos) inside the dataset, so this is 6
# channels; n_features is read off the dataset, never hard-coded.
FEATURES = ("x", "y", "heading", "vx", "vy")

# OFFICIAL WAYMO SPLIT -- there is no random_split.
# Training and validation come from DIFFERENT shards of DIFFERENT splits, so
# there is no possibility of leakage: they are distinct scenarios. This also
# makes the numbers comparable with the literature, which reports on the
# 'validation' split.
TRAIN_CACHE = "/workspace/datasets/waymo/cache_train"
VAL_CACHE = "/workspace/datasets/waymo/cache_val"

# Root of the checkpoints. The final folder includes the cls_weight AND the
# grid arm (sdc/agent), so sweep runs never overwrite each other.
CHECKPOINT_ROOT = os.environ.get("CHECKPOINT_ROOT", "/workspace/experiments/checkpoints")

# Root of the per-epoch training logs (one CSV per run, named by
# model + cls_weight + arm + seed + timestamp so runs never overwrite).
LOG_ROOT = os.environ.get("LOG_ROOT", "/workspace/experiments/logs")


def masked_mse_per_mode(outputs, targets, mask):
    """
    Masked mean squared error of EACH mode, per example in the batch.

    outputs: [batch, K, 80, 2]  -- the K hypotheses
    targets: [batch, 80, 2]     -- the ground truth (single)
    mask:    [batch, 80]        -- 1 where the future frame is valid

    Returns: [batch, K]

    The mask is essential: agents that leave visibility have invalid frames
    zeroed out in the dataset, and without masking the model would be
    punished for not predicting zeros.
    """
    targets = targets.unsqueeze(1)                 # [B, 1, 80, 2]
    mask_exp = mask.unsqueeze(1).unsqueeze(-1)     # [B, 1, 80, 1]

    diff2 = (outputs - targets) ** 2               # [B, K, 80, 2]
    masked = diff2 * mask_exp

    per_mode_sum = masked.sum(dim=(2, 3))                        # [B, K]
    denom = (mask.sum(dim=1) * 2).clamp(min=1.0).unsqueeze(1)    # [B, 1]

    return per_mode_sum / denom


def wta_loss(outputs, scores, targets, mask, cls_weight):
    """
    Winner-Takes-All (WTA) loss.

    Idea: of the K modes, only the one closest to the ground truth receives
    regression gradient. The others are free to cover alternative hypotheses
    instead of being pulled toward the average -- which is exactly what you
    want in a multimodal problem (at an intersection, "turn" and "go
    straight" are both correct, and the average of the two is not a
    plausible trajectory).

    In parallel, the score head learns via cross-entropy which mode was the
    winner. Without this term the model would have K trajectories but no
    notion of which is the most likely -- and the official metric (mAP) uses
    the score.

    NOTE ON cls_weight: the two terms live on VERY different scales. The
    regression is MSE in m^2 (tens to hundreds); the cross-entropy with 6
    classes starts at ln(6) ~= 1.79. With cls_weight=1.0 the classification
    is worth ~1% of the total loss and the score_head's gradient gets
    drowned out -- which is exactly what happened in the first run.

    Returns a dictionary so as not to multiply the number of return values
    with each new diagnostic.
    """
    per_mode = masked_mse_per_mode(outputs, targets, mask)   # [B, K]

    # argmin does not propagate gradient (it's an index) -- the gradient
    # flows only through the selected value via gather.
    best_idx = per_mode.argmin(dim=1)                        # [B]
    reg = per_mode.gather(1, best_idx.unsqueeze(1)).squeeze(1)

    cls = F.cross_entropy(scores, best_idx, reduction="none")  # [B]

    total = reg + cls_weight * cls

    # --- Diagnostics (do not enter the loss) ---
    with torch.no_grad():
        top1_idx = scores.argmax(dim=1)
        top1 = per_mode.gather(1, top1_idx.unsqueeze(1)).squeeze(1)

        # Mean error of the K modes = what you would get by choosing at random.
        # If top1 ~= mean_all, the score head learned nothing.
        # If top1 > mean_all, it learned something INVERTED (sign of a bug).
        mean_all = per_mode.mean(dim=1)

        # Position of the chosen mode in the real quality ranking.
        # 0 = picked the best of the 6; 5 = picked the worst.
        # Pure chance would give a mean of 2.5.
        rank = (per_mode < top1.unsqueeze(1)).sum(dim=1).float()

    return {
        "total": total,        # [B] what gets optimized
        "reg": reg,            # [B] error of the best mode (proxy for minADE)
        "cls": cls,            # [B] pure cross-entropy, unweighted
        "top1": top1,          # [B] error of the best-scored mode
        "mean_all": mean_all,  # [B] mean error of the modes (chance baseline)
        "rank": rank,          # [B] position of the chosen mode in the real ranking
        "best_idx": best_idx,  # [B] which mode won (collapse diagnostic)
    }


def train(cls_weight, agent_centric, seed=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reproducibility: when --seed is given, seed every RNG. The seed also
    # goes into the run tag (folder + log name) so multiple seeds coexist
    # instead of overwriting each other. WITHOUT --seed the paths keep the
    # base tag, so tooling stays predictable.
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Grid arm goes into the tag so {sdc, agent} never share a checkpoint dir.
    arm = "agent" if agent_centric else "sdc"
    tag = f"cls{cls_weight:g}_{arm}"
    run_tag = tag if seed is None else f"{tag}_seed{seed}"
    checkpoint_dir = os.path.join(CHECKPOINT_ROOT, f"sequential_{run_tag}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("=" * 78)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (no GPU detected)"
    print(f"[*] SEQUENTIAL (GRU) TRAINING (K={NUM_MODES}, cls_weight={cls_weight}, "
          f"arm={arm}) ON GPU: {gpu_name}")
    print("=" * 78)

    try:
        train_dataset = WaymoMotionDatasetAgentCentric(
            TRAIN_CACHE, agent_centric=agent_centric, features=FEATURES)
        val_dataset = WaymoMotionDatasetAgentCentric(
            VAL_CACHE, agent_centric=agent_centric, features=FEATURES)
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        sys.exit(1)

    if len(train_dataset) == 0:
        print(f"[ERROR] Empty training cache: {TRAIN_CACHE}")
        sys.exit(1)
    if len(val_dataset) == 0:
        print(f"[ERROR] Empty validation cache: {VAL_CACHE}")
        print("       Run first, in the METRICS container:")
        print("       python3 -m src.core.waymo_preprocessor --split validation --shards 0,1,2")
        sys.exit(1)

    n_features = train_dataset.n_features
    print(f"[OK] Training (official 'training' split):      {len(train_dataset)} examples")
    print(f"[OK] Validation (official 'validation' split): {len(val_dataset)} examples")
    print(f"[OK] Features {FEATURES} -> n_features={n_features} (agent_centric={agent_centric})")

    # Input-bound training (the GPU sits mostly idle): the win is in the
    # dataloader, not the model. More workers + pinned memory + persistent
    # workers keep the GPU fed. BATCH_SIZE stays 64 on purpose -- changing it
    # would change the optimization and add a confound to the controlled grid.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=8, pin_memory=True, persistent_workers=True)

    model = SequentialTrajectoryPredictor(
        input_steps=11, output_steps=80, num_modes=NUM_MODES, n_features=n_features
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] Sequential model: {n_params:,} parameters")
    print(f"[OK] Checkpoints in: {checkpoint_dir}")
    if seed is not None:
        print(f"[OK] Seed: {seed}")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    best_val_reg = float("inf")
    best_epoch = 0
    epochs_no_improve = 0

    # --- Per-epoch CSV log (in addition to the terminal display) ---
    # One file per run: <model>_<run_tag>_<timestamp>.csv. The timestamp
    # keeps re-trains from overwriting each other. cum_time_s on the last
    # is_best=1 row is the "time until the best epoch".
    os.makedirs(LOG_ROOT, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(LOG_ROOT, f"sequential_{run_tag}_{stamp}.csv")
    log_file = open(log_path, "w", newline="")
    log_file.write(f"# model=sequential,cls_weight={cls_weight},arm={arm},"
                   f"n_features={n_features},seed={seed},checkpoint_dir={checkpoint_dir}\n")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "epoch", "train_loss", "val_best_mode", "val_top1", "val_chance",
        "val_rank", "val_ce", "epoch_time_s", "cum_time_s",
        "is_best", "best_epoch", "epochs_no_improve",
    ])
    print(f"[OK] Per-epoch log: {log_path}")

    train_start = time.time()
    best_cum_time = 0.0

    for epoch in range(EPOCHS):
        # ---------------- Training ----------------
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        start_time = time.time()

        for history, future_gt, future_mask, agent_type in train_loader:
            history = history.to(device)
            future_gt = future_gt.to(device)
            future_mask = future_mask.to(device)

            optimizer.zero_grad()
            outputs, scores = model(history)

            out = wta_loss(outputs, scores, future_gt, future_mask, cls_weight)
            loss = out["total"].mean()
            loss.backward()
            optimizer.step()

            train_loss_sum += out["total"].sum().item()
            train_count += out["total"].numel()

        avg_train_loss = train_loss_sum / train_count

        # ---------------- Validation ----------------
        model.eval()
        sums = {"reg": 0.0, "top1": 0.0, "mean_all": 0.0, "rank": 0.0, "cls": 0.0}
        val_count = 0
        per_type_stats = {}                      # {type: [top1_sum, count]}
        mode_usage = [0] * NUM_MODES             # mode-collapse diagnostic

        with torch.no_grad():
            for history, future_gt, future_mask, agent_type in val_loader:
                history = history.to(device)
                future_gt = future_gt.to(device)
                future_mask = future_mask.to(device)

                outputs, scores = model(history)
                out = wta_loss(outputs, scores, future_gt, future_mask, cls_weight)

                for k in sums:
                    sums[k] += out[k].sum().item()
                val_count += out["reg"].numel()

                for m in out["best_idx"].cpu().tolist():
                    mode_usage[m] += 1

                for t, l in zip(agent_type.tolist(), out["top1"].cpu().tolist()):
                    if t not in per_type_stats:
                        per_type_stats[t] = [0.0, 0]
                    per_type_stats[t][0] += l
                    per_type_stats[t][1] += 1

        avg = {k: v / val_count for k, v in sums.items()}

        # Best-only checkpoint. The "best" criterion is the error of the
        # BEST mode, the direct proxy for what the official metric rewards
        # (minADE/minFDE are best-of-6). We persist ONLY when the epoch
        # improves it, overwriting a single sequential_best.pth. The old
        # per-epoch save (one .pth per epoch) was never read by anything and
        # cost ~4.6 G across the sweep -- and every best is reproducible in
        # ~2 min/run anyway.
        if avg["reg"] < best_val_reg:
            best_val_reg = avg["reg"]
            best_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save(model.state_dict(),
                       os.path.join(checkpoint_dir, "sequential_best.pth"))
            marker = "  <- best so far"
            is_best = 1
        else:
            epochs_no_improve += 1
            marker = ""
            is_best = 0

        duration = time.time() - start_time
        cum_time = time.time() - train_start
        if is_best:
            best_cum_time = cum_time
        print(f"[OK] Epoch {epoch+1}/{EPOCHS} | "
              f"Train: {avg_train_loss:.4f} | "
              f"Val(best mode): {avg['reg']:.4f}{marker} | "
              f"Val(top-1): {avg['top1']:.4f} | "
              f"Time: {duration:.2f}s")

        breakdown = []
        for t, (loss_sum, n) in sorted(per_type_stats.items()):
            name = OBJECT_TYPE_NAMES.get(t, f"TYPE_{t}")
            breakdown.append(f"{name}: {loss_sum / n:.2f} (n={n})")
        print(f"         Val top-1 by type -> {' | '.join(breakdown)}")

        # --- Score head quality ---
        # If top-1 ~= chance and rank ~= 2.5, the score_head learned nothing.
        # If top-1 < chance and rank < 2.5, it is genuinely ranking.
        # If top-1 > chance and rank > 2.5, it learned INVERTED -- suspect a bug.
        if avg["rank"] < 2.3:
            verdict = "ranking"
        elif avg["rank"] > 2.7:
            verdict = "INVERTED (?)"
        else:
            verdict = "~chance"
        print(f"         Score head -> top-1: {avg['top1']:.1f} | "
              f"chance: {avg['mean_all']:.1f} | "
              f"mean rank: {avg['rank']:.2f}/5 ({verdict}) | "
              f"CE: {avg['cls']:.3f}")

        # If a single mode wins almost always, the model collapsed into a
        # disguised unimodal one -- and the experiment loses its point.
        usage_pct = [100.0 * c / val_count for c in mode_usage]
        usage_str = " | ".join(f"m{i}: {p:.0f}%" for i, p in enumerate(usage_pct))
        print(f"         Mode usage -> {usage_str}")
        if max(usage_pct) > 90.0:
            print("         [WARNING] mode collapse: one mode won >90% of the time.")

        # Persist this epoch's diagnostics to the CSV (the terminal display
        # above is unchanged). Written and flushed per epoch so the log
        # survives a crash or a manual interrupt mid-training.
        log_writer.writerow([
            epoch + 1,
            f"{avg_train_loss:.6f}",
            f"{avg['reg']:.6f}",
            f"{avg['top1']:.6f}",
            f"{avg['mean_all']:.6f}",
            f"{avg['rank']:.6f}",
            f"{avg['cls']:.6f}",
            f"{duration:.2f}",
            f"{cum_time:.2f}",
            is_best,
            best_epoch,
            epochs_no_improve,
        ])
        log_file.flush()

        # Early stopping: once Val(best mode) stops improving for PATIENCE
        # epochs we have reached the plateau. This is what makes the
        # architecture comparison fair -- each model is measured AT
        # CONVERGENCE, not at a fixed 25-epoch budget that silently favors
        # whichever one happens to converge faster (the MLP did).
        if epochs_no_improve >= PATIENCE:
            print(f"\n[EARLY STOP] No Val(best mode) improvement for {PATIENCE} "
                  f"epochs. Stopping at epoch {epoch+1}/{EPOCHS}.")
            break

    log_file.close()

    total_time = time.time() - train_start
    print(f"\n[SUCCESS] Sequential training finished (cls_weight={cls_weight}, arm={arm}).")
    print(f"Best checkpoint: sequential_best.pth (epoch {best_epoch}, Val best mode: {best_val_reg:.4f})")
    print(f"Time to best epoch: {best_cum_time:.1f}s | Total training time: {total_time:.1f}s")
    print(f"Saved at: {checkpoint_dir}")
    print(f"Per-epoch log: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trains the sequential (GRU) K=6 model with Winner-Takes-All loss"
    )
    parser.add_argument("--cls-weight", type=float, default=50.0,
                        help="weight of the classification term in the total loss")
    parser.add_argument("--agent-centric", action="store_true",
                        help="agent-centric normalization arm; omit for the SDC-centric arm")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed; also suffixes the run folder/log (..._seed<n>)")
    args = parser.parse_args()

    train(cls_weight=args.cls_weight, agent_centric=args.agent_centric, seed=args.seed)