import os
import time
import csv
import random
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# SIBLING de train_vectorized (V1 = social). Mudanças vs V0, TODAS mecânicas:
#   (1) dataset/model sociais;  (2) batch de 6 campos (neighbors + nmask);
#   (3) model(history, neighbors, nmask);  (4) flag --n-neighbors.
# A CIÊNCIA é intacta: mesma WTA, best=Val(best mode), patience, cls_weight,
# BATCH_SIZE=64, LR, split oficial. Braço agente-cêntrico é IMPLÍCITO no V1.
from src.core.waymo_pytorch_dataset_social import WaymoMotionDatasetSocial
from src.motion.vectorized_social_model import VectorizedSocialTrajectoryPredictor

OBJECT_TYPE_NAMES = {0: "UNSET", 1: "VEHICLE", 2: "PEDESTRIAN", 3: "CYCLIST", 4: "OTHER"}

NUM_MODES = 6
EPOCHS = 300
PATIENCE = 10
BATCH_SIZE = 64
LR = 1e-3
N_NEIGHBORS = 16          # K fixo a priori (calibrado empiricamente no smoke do dataset)

FEATURES = ("x", "y", "heading", "vx", "vy")

TRAIN_CACHE = "/workspace/datasets/waymo/cache_train"
VAL_CACHE = "/workspace/datasets/waymo/cache_val"

CHECKPOINT_ROOT = "/workspace/experiments/checkpoints"
LOG_ROOT = "/workspace/experiments/logs"
STATS_PATH = "/workspace/experiments/feature_stats.npy"

MODEL_NAME = "vectorized_social"      # prefixo de pasta/log (não colide com V0)
CHECKPOINT_NAME = "social_best.pth"   # nome do checkpoint dentro da pasta


def masked_mse_per_mode(outputs, targets, mask):
    """MSE mascarado de CADA modo, por exemplo do batch. [B,K]. Idêntico ao V0."""
    targets = targets.unsqueeze(1)                 # [B,1,80,2]
    mask_exp = mask.unsqueeze(1).unsqueeze(-1)     # [B,1,80,1]
    diff2 = (outputs - targets) ** 2
    masked = diff2 * mask_exp
    per_mode_sum = masked.sum(dim=(2, 3))                        # [B,K]
    denom = (mask.sum(dim=1) * 2).clamp(min=1.0).unsqueeze(1)    # [B,1]
    return per_mode_sum / denom


def wta_loss(outputs, scores, targets, mask, cls_weight):
    """Winner-Takes-All + cross-entropy do modo vencedor. Idêntico ao V0."""
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
          n_neighbors=N_NEIGHBORS):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Braço agente-cêntrico é implícito no V1 -> arm fixo em "agent". O run_tag
    # segue o MESMO formato do V0 (cls<w>_agent_seed<s>[_std]) para reaproveitar
    # a lógica de grade/sentinela; só o PREFIXO de modelo muda (vectorized_social).
    arm = "agent"
    tag = f"cls{cls_weight:g}_{arm}"
    run_tag = tag if seed is None else f"{tag}_seed{seed}"
    if standardize:
        run_tag = f"{run_tag}_std"
    checkpoint_dir = os.path.join(CHECKPOINT_ROOT, f"{MODEL_NAME}_{run_tag}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("=" * 78)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (no GPU)"
    print(f"[*] SOCIAL TRAINING (K={NUM_MODES}, cls_weight={cls_weight}, arm={arm}, "
          f"n_neighbors={n_neighbors}) ON: {gpu_name}")
    print("=" * 78)

    try:
        train_dataset = WaymoMotionDatasetSocial(
            TRAIN_CACHE, n_neighbors=n_neighbors, features=FEATURES)
        val_dataset = WaymoMotionDatasetSocial(
            VAL_CACHE, n_neighbors=n_neighbors, features=FEATURES)
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        return

    if len(train_dataset) == 0:
        print(f"[ERROR] Empty training cache: {TRAIN_CACHE}")
        return
    if len(val_dataset) == 0:
        print(f"[ERROR] Empty validation cache: {VAL_CACHE}")
        return

    n_features = train_dataset.n_features
    print(f"[OK] Training examples:   {len(train_dataset)}")
    print(f"[OK] Validation examples: {len(val_dataset)}")
    print(f"[OK] Features {FEATURES} -> n_features={n_features} | n_neighbors={n_neighbors}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=8, pin_memory=True, persistent_workers=True)

    model = VectorizedSocialTrajectoryPredictor(
        input_steps=11, output_steps=80, num_modes=NUM_MODES, n_features=n_features
    ).to(device)

    if standardize:
        if not os.path.exists(stats_path):
            print(f"[ERROR] --standardize pediu stats mas o arquivo não existe: {stats_path}")
            print("        Gere-o UMA vez com: python3 -m src.motion.compute_feature_stats")
            return
        model.load_feature_stats(stats_path)
        fm = model.feat_mean.detach().cpu().tolist()
        fs = model.feat_std.detach().cpu().tolist()
        print(f"[OK] Padronização ATIVA (std). stats: {stats_path}")
        print(f"     mean/canal = {[round(v,3) for v in fm]}")
        print(f"     std /canal = {[round(v,3) for v in fs]}  (canais 2,3=sin,cos passthrough)")
    else:
        print("[OK] Padronização OFF (raw): entrada crua, buffers identidade.")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] Social model: {n_params:,} parameters")
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
                   f"n_features={n_features},n_neighbors={n_neighbors},seed={seed},"
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

        for history, neighbors, nmask, future_gt, future_mask, agent_type in train_loader:
            history = history.to(device)
            neighbors = neighbors.to(device)
            nmask = nmask.to(device)
            future_gt = future_gt.to(device)
            future_mask = future_mask.to(device)

            optimizer.zero_grad()
            outputs, scores = model(history, neighbors, nmask)

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
            for history, neighbors, nmask, future_gt, future_mask, agent_type in val_loader:
                history = history.to(device)
                neighbors = neighbors.to(device)
                nmask = nmask.to(device)
                future_gt = future_gt.to(device)
                future_mask = future_mask.to(device)

                outputs, scores = model(history, neighbors, nmask)
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
    print(f"\n[SUCCESS] Social training finished (cls_weight={cls_weight}, arm={arm}).")
    print(f"Best checkpoint: {CHECKPOINT_NAME} (epoch {best_epoch}, "
          f"Val best mode: {best_val_reg:.4f})")
    print(f"Time to best epoch: {best_cum_time:.1f}s | Total training time: {total_time:.1f}s")
    print(f"Saved at: {checkpoint_dir}")
    print(f"Per-epoch log: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trains the vectorized+social K=6 model with Winner-Takes-All loss"
    )
    parser.add_argument("--cls-weight", type=float, default=20.0,
                        help="weight of the classification term in the total loss")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed; also suffixes the run folder/log (..._seed<n>)")
    parser.add_argument("--standardize", action="store_true",
                        help="std: padroniza a entrada (alvo E vizinhos) com stats "
                             "congeladas do cache_train (sin/cos passthrough). Sufixa _std. "
                             "Omitir = raw.")
    parser.add_argument("--stats-path", default=STATS_PATH,
                        help=f"caminho do .npy de stats (default: {STATS_PATH}). "
                             "Só usado com --standardize.")
    parser.add_argument("--n-neighbors", type=int, default=N_NEIGHBORS,
                        help=f"K vizinhos mais próximos por alvo (default: {N_NEIGHBORS}). "
                             "MUST match inference. Fixo a priori na grade.")
    args = parser.parse_args()

    train(cls_weight=args.cls_weight, seed=args.seed, standardize=args.standardize,
          stats_path=args.stats_path, n_neighbors=args.n_neighbors)
