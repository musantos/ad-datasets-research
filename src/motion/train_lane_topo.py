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

# SIBLING of train_map (V3 = V2 social+map PLUS lane topology). Changes vs V2 are
# ALL mechanical:
#   (1) dataset -> WaymoMotionDatasetMapTopo (V3 loader, 10-tuple with adjacency);
#   (2) model   -> LaneTopoTrajectoryPredictor (V2 subclass + typed lane graph);
#   (3) batch   -> 10-tuple (map_adjacency after map_mask), both loops;
#   (4) call    -> model(history, neighbors, nmask, map_polylines, map_type,
#                        map_mask, map_adjacency).
# The SCIENCE is intact: same WTA, best=Val(best mode), patience, cls_weight,
# BATCH_SIZE=64, LR, official split, K/M/Np frozen a priori. Agent-centric arm is
# IMPLICIT (as in V1/V2). V3 runs raw-only (MODELS_CFG variants=["raw"]);
# --standardize is kept for sibling parity but is not used by the grid.
from src.core.waymo_pytorch_dataset_map_topo import WaymoMotionDatasetMapTopo
from src.motion.lane_topo_model import LaneTopoTrajectoryPredictor

OBJECT_TYPE_NAMES = {0: "UNSET", 1: "VEHICLE", 2: "PEDESTRIAN", 3: "CYCLIST", 4: "OTHER"}

NUM_MODES = 6
EPOCHS = 300
PATIENCE = 10
BATCH_SIZE = 64
LR = 1e-3
N_NEIGHBORS = 16              # K fixed a priori (social branch), == V1/V2
N_MAP_POLYLINES = 128         # M fixed a priori (== V2; the topo alignment depends on M)
N_POINTS_PER_POLYLINE = 20    # Np fixed a priori (== V2)

FEATURES = ("x", "y", "heading", "vx", "vy")

# V3 reads the *_map_topo caches (strict superset of the V2 map caches: same
# geometry + the per-lane adjacency lists). The V2 loader runs on them unchanged.
TRAIN_CACHE = "/workspace/datasets/waymo/cache_train_map_topo"
VAL_CACHE = "/workspace/datasets/waymo/cache_val_map_topo"

CHECKPOINT_ROOT = os.environ.get("CHECKPOINT_ROOT", "/workspace/experiments/checkpoints")
LOG_ROOT = os.environ.get("LOG_ROOT", "/workspace/experiments/logs")
STATS_PATH = "/workspace/experiments/feature_stats.npy"

# Folder/log prefix. NOT a superset of any earlier prefix ('vectorized_social_map'
# would be), so the consolidation regexes cannot cross-match V2 vs V3 folders.
MODEL_NAME = "lane_topo"
CHECKPOINT_NAME = "lane_topo_best.pth"        # checkpoint name inside the folder


def masked_mse_per_mode(outputs, targets, mask):
    """Masked MSE of EACH mode, per batch example. [B,K]. Same as V0/V1/V2."""
    targets = targets.unsqueeze(1)                 # [B,1,80,2]
    mask_exp = mask.unsqueeze(1).unsqueeze(-1)     # [B,1,80,1]
    diff2 = (outputs - targets) ** 2
    masked = diff2 * mask_exp
    per_mode_sum = masked.sum(dim=(2, 3))                        # [B,K]
    denom = (mask.sum(dim=1) * 2).clamp(min=1.0).unsqueeze(1)    # [B,1]
    return per_mode_sum / denom


def wta_loss(outputs, scores, targets, mask, cls_weight):
    """Winner-Takes-All + cross-entropy of the winning mode. Same as V0/V1/V2."""
    per_mode = masked_mse_per_mode(outputs, targets, mask)       # [B,K]
    best_idx = per_mode.argmin(dim=1)                            # [B]
    reg = per_mode.gather(1, best_idx.unsqueeze(1)).squeeze(1)
    cls = F.cross_entropy(scores, best_idx, reduction="none")    # [B]
    total = reg + cls_weight * cls

    with torch.no_grad():
        top1_idx = scores.argmax(dim=1)
        top1 = per_mode.gather(1, top1_idx.unsqueeze(1)).squeeze(1)
        mean_all = per_mode.mean(dim=1)
        rank = (per_mode < top1.unsqueeze(1)).sum(dim=1).float()

    return {"total": total, "reg": reg, "cls": cls, "top1": top1,
            "mean_all": mean_all, "rank": rank, "best_idx": best_idx}


def train(cls_weight, seed=None, standardize=False, stats_path=STATS_PATH,
          n_neighbors=N_NEIGHBORS, n_map_polylines=N_MAP_POLYLINES,
          n_points_per_polyline=N_POINTS_PER_POLYLINE, use_map_type=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Agent-centric arm is implicit in V3 -> arm fixed to "agent". run_tag follows
    # the SAME format as V0/V1/V2 (cls<w>_agent_seed<s>[_std][_type]) to reuse the
    # grid / sentinel logic; only the model PREFIX changes (lane_topo).
    arm = "agent"
    tag = f"cls{cls_weight:g}_{arm}"
    run_tag = tag if seed is None else f"{tag}_seed{seed}"
    if standardize:
        run_tag = f"{run_tag}_std"
    if use_map_type:
        run_tag = f"{run_tag}_type"
    checkpoint_dir = os.path.join(CHECKPOINT_ROOT, f"{MODEL_NAME}_{run_tag}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("=" * 78)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (no GPU)"
    print(f"[*] LANE-TOPO (social+map+topology) TRAINING (K={NUM_MODES}, "
          f"cls_weight={cls_weight}, arm={arm}, n_neighbors={n_neighbors}, "
          f"M={n_map_polylines}, Np={n_points_per_polyline}, "
          f"use_map_type={use_map_type}) ON: {gpu_name}")
    print("=" * 78)

    try:
        train_dataset = WaymoMotionDatasetMapTopo(
            TRAIN_CACHE, n_neighbors=n_neighbors, n_map_polylines=n_map_polylines,
            n_points_per_polyline=n_points_per_polyline, features=FEATURES)
        val_dataset = WaymoMotionDatasetMapTopo(
            VAL_CACHE, n_neighbors=n_neighbors, n_map_polylines=n_map_polylines,
            n_points_per_polyline=n_points_per_polyline, features=FEATURES)
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        sys.exit(1)

    if len(train_dataset) == 0:
        print(f"[ERROR] Empty training cache: {TRAIN_CACHE}")
        sys.exit(1)
    if len(val_dataset) == 0:
        print(f"[ERROR] Empty validation cache: {VAL_CACHE}")
        sys.exit(1)

    n_features = train_dataset.n_features
    print(f"[OK] Training examples:   {len(train_dataset)}")
    print(f"[OK] Validation examples: {len(val_dataset)}")
    print(f"[OK] Features {FEATURES} -> n_features={n_features} | n_neighbors={n_neighbors} "
          f"| M={n_map_polylines} | Np={n_points_per_polyline}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=8, pin_memory=True, persistent_workers=True)

    model = LaneTopoTrajectoryPredictor(
        input_steps=11, output_steps=80, num_modes=NUM_MODES, n_features=n_features,
        use_map_type=use_map_type
    ).to(device)

    if standardize:
        if not os.path.exists(stats_path):
            print(f"[ERROR] --standardize asked for stats but the file is missing: {stats_path}")
            print("        Generate it ONCE with: python3 -m src.motion.compute_feature_stats")
            sys.exit(1)
        model.load_feature_stats(stats_path)
        fm = model.feat_mean.detach().cpu().tolist()
        fs = model.feat_std.detach().cpu().tolist()
        print(f"[OK] Standardization ON (std). stats: {stats_path}")
        print(f"     mean/channel = {[round(v,3) for v in fm]}")
        print(f"     std /channel = {[round(v,3) for v in fs]}  (channels 2,3=sin,cos passthrough)")
    else:
        print("[OK] Standardization OFF (raw): raw input, identity buffers.")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] Lane-topo model: {n_params:,} parameters")
    print(f"[OK] Checkpoints in: {checkpoint_dir}")
    if seed is not None:
        print(f"[OK] Seed: {seed}")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    best_val_reg = float("inf")
    best_epoch = 0
    epochs_no_improve = 0

    os.makedirs(LOG_ROOT, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(LOG_ROOT, f"{MODEL_NAME}_{run_tag}_{stamp}.csv")
    log_file = open(log_path, "w", newline="")
    log_file.write(f"# model={MODEL_NAME},cls_weight={cls_weight},arm={arm},"
                   f"n_features={n_features},n_neighbors={n_neighbors},"
                   f"n_map_polylines={n_map_polylines},"
                   f"n_points_per_polyline={n_points_per_polyline},"
                   f"use_map_type={int(use_map_type)},seed={seed},"
                   f"standardize={int(standardize)},checkpoint_dir={checkpoint_dir}\n")
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

        # 10-tuple (V3): map_adjacency comes after map_mask, before the labels.
        for (history, neighbors, nmask, map_polylines, map_type, map_mask,
             map_adjacency, future_gt, future_mask, agent_type) in train_loader:
            history = history.to(device)
            neighbors = neighbors.to(device)
            nmask = nmask.to(device)
            map_polylines = map_polylines.to(device)
            map_type = map_type.to(device)
            map_mask = map_mask.to(device)
            map_adjacency = map_adjacency.to(device)
            future_gt = future_gt.to(device)
            future_mask = future_mask.to(device)

            optimizer.zero_grad()
            outputs, scores = model(history, neighbors, nmask,
                                    map_polylines, map_type, map_mask, map_adjacency)

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
        per_type_stats = {}
        mode_usage = [0] * NUM_MODES

        with torch.no_grad():
            for (history, neighbors, nmask, map_polylines, map_type, map_mask,
                 map_adjacency, future_gt, future_mask, agent_type) in val_loader:
                history = history.to(device)
                neighbors = neighbors.to(device)
                nmask = nmask.to(device)
                map_polylines = map_polylines.to(device)
                map_type = map_type.to(device)
                map_mask = map_mask.to(device)
                map_adjacency = map_adjacency.to(device)
                future_gt = future_gt.to(device)
                future_mask = future_mask.to(device)

                outputs, scores = model(history, neighbors, nmask,
                                        map_polylines, map_type, map_mask, map_adjacency)
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

        if avg["reg"] < best_val_reg:
            best_val_reg = avg["reg"]
            best_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save(model.state_dict(),
                       os.path.join(checkpoint_dir, CHECKPOINT_NAME))
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
        print(f"[OK] Epoch {epoch+1}/{EPOCHS} | Train: {avg_train_loss:.4f} | "
              f"Val(best mode): {avg['reg']:.4f}{marker} | "
              f"Val(top-1): {avg['top1']:.4f} | Time: {duration:.2f}s")

        breakdown = []
        for t, (loss_sum, n) in sorted(per_type_stats.items()):
            name = OBJECT_TYPE_NAMES.get(t, f"TYPE_{t}")
            breakdown.append(f"{name}: {loss_sum / n:.2f} (n={n})")
        print(f"         Val top-1 by type -> {' | '.join(breakdown)}")

        if avg["rank"] < 2.3:
            verdict = "ranking"
        elif avg["rank"] > 2.7:
            verdict = "INVERTED (?)"
        else:
            verdict = "~chance"
        print(f"         Score head -> top-1: {avg['top1']:.1f} | "
              f"chance: {avg['mean_all']:.1f} | "
              f"mean rank: {avg['rank']:.2f}/5 ({verdict}) | CE: {avg['cls']:.3f}")

        usage_pct = [100.0 * c / val_count for c in mode_usage]
        usage_str = " | ".join(f"m{i}: {p:.0f}%" for i, p in enumerate(usage_pct))
        print(f"         Mode usage -> {usage_str}")
        if max(usage_pct) > 90.0:
            print("         [WARNING] mode collapse: one mode won >90% of the time.")

        log_writer.writerow([
            epoch + 1, f"{avg_train_loss:.6f}", f"{avg['reg']:.6f}",
            f"{avg['top1']:.6f}", f"{avg['mean_all']:.6f}", f"{avg['rank']:.6f}",
            f"{avg['cls']:.6f}", f"{duration:.2f}", f"{cum_time:.2f}",
            is_best, best_epoch, epochs_no_improve,
        ])
        log_file.flush()

        if epochs_no_improve >= PATIENCE:
            print(f"\n[EARLY STOP] No Val(best mode) improvement for {PATIENCE} "
                  f"epochs. Stopping at epoch {epoch+1}/{EPOCHS}.")
            break

    log_file.close()

    total_time = time.time() - train_start
    print(f"\n[SUCCESS] Lane-topo training finished (cls_weight={cls_weight}, arm={arm}).")
    print(f"Best checkpoint: {CHECKPOINT_NAME} (epoch {best_epoch}, "
          f"Val best mode: {best_val_reg:.4f})")
    print(f"Time to best epoch: {best_cum_time:.1f}s | Total training time: {total_time:.1f}s")
    print(f"Saved at: {checkpoint_dir}")
    print(f"Per-epoch log: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trains the vectorized+social+map+topology K=6 model with Winner-Takes-All loss"
    )
    parser.add_argument("--cls-weight", type=float, default=20.0,
                        help="weight of the classification term in the total loss")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed; also suffixes the run folder/log (..._seed<n>)")
    parser.add_argument("--standardize", action="store_true",
                        help="std: standardize the input (target AND neighbors) with "
                             "frozen cache_train stats (sin/cos passthrough). Suffixes _std. "
                             "Omit = raw. NOTE: V3 grid is raw-only; kept for parity.")
    parser.add_argument("--stats-path", default=STATS_PATH,
                        help=f"path of the stats .npy (default: {STATS_PATH}). "
                             "Only used with --standardize.")
    parser.add_argument("--n-neighbors", type=int, default=N_NEIGHBORS,
                        help=f"K nearest neighbors per target (default: {N_NEIGHBORS}). "
                             "MUST match inference. Fixed a priori in the grid.")
    parser.add_argument("--n-map-polylines", type=int, default=N_MAP_POLYLINES,
                        help=f"M nearest map polylines per target (default: {N_MAP_POLYLINES}). "
                             "MUST match inference. Fixed a priori in the grid. The topo "
                             "alignment depends on M, so keep it at the V2 value.")
    parser.add_argument("--n-points-per-polyline", type=int, default=N_POINTS_PER_POLYLINE,
                        help=f"Np points each polyline is resampled to (default: "
                             f"{N_POINTS_PER_POLYLINE}). MUST match inference. Fixed a priori.")
    parser.add_argument("--use-map-type", action="store_true",
                        help="type ablation: feed the 3-way entity type embedding. "
                             "Suffixes _type. Default OFF = geometry + topology (the V3 variable).")
    args = parser.parse_args()

    train(cls_weight=args.cls_weight, seed=args.seed, standardize=args.standardize,
          stats_path=args.stats_path, n_neighbors=args.n_neighbors,
          n_map_polylines=args.n_map_polylines,
          n_points_per_polyline=args.n_points_per_polyline,
          use_map_type=args.use_map_type)
