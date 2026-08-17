import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time

from src.core.waymo_pytorch_dataset import WaymoMotionDataset
from src.motion.simple_model import SimpleTrajectoryPredictor

# Mapping of the Object.Type enum from the Waymo proto.
# NOTE: based on the public WOD documentation -- if any type shows up as
# "unknown" in the output, we need to confirm against
# waymo_open_dataset/protos/scenario.proto (enum ObjectType) in your container.
OBJECT_TYPE_NAMES = {
    0: "UNSET",
    1: "VEHICLE",
    2: "PEDESTRIAN",
    3: "CYCLIST",
    4: "OTHER",
}


def masked_mse_per_example(outputs, targets, mask):
    """
    Returns the masked loss PER EXAMPLE in the batch (not the mean over the
    whole batch), so it can later be grouped by agent type.

    outputs, targets: [batch, 80, 2]
    mask: [batch, 80]
    Returns: tensor [batch] with the masked MSE of each example.
    """
    diff2 = (outputs - targets) ** 2          # [batch, 80, 2]
    mask_exp = mask.unsqueeze(-1)             # [batch, 80, 1]

    masked_diff = diff2 * mask_exp
    per_example_sum = masked_diff.sum(dim=(1, 2))                       # [batch]
    per_example_count = (mask_exp.sum(dim=(1, 2)) * 2).clamp(min=1.0)   # [batch]

    return per_example_sum / per_example_count


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # OFFICIAL WAYMO SPLIT -- there is no more random_split.
    # Training and validation come from DIFFERENT splits of the dataset
    # (folders scenario/training and scenario/validation), so there is no
    # possibility of leakage: they are distinct scenarios. This also makes
    # the metrics comparable with the literature, which reports on the
    # 'validation' split (the 'testing' one has no annotated future).
    #
    # These are exactly the same caches used by train_multimodal.py -- which
    # is what guarantees that the comparison between the two models isolates
    # multimodality as the only variable.
    train_cache = "/workspace/datasets/waymo/cache_train"
    val_cache = "/workspace/datasets/waymo/cache_val"

    # New directory: preserves the old checkpoints (home-made split, with
    # contamination) in checkpoints/baseline_3shards/ for historical
    # purposes, without mixing them with the valid results.
    checkpoint_dir = "/workspace/experiments/checkpoints/baseline_oficial"
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("=" * 50)
    print(f"[*] STARTING TRAINING ON GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (no GPU detected)'}")
    print("=" * 50)

    try:
        train_dataset = WaymoMotionDataset(train_cache)
        val_dataset = WaymoMotionDataset(val_cache)
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        return

    if len(train_dataset) == 0:
        print(f"[ERROR] Empty training cache: {train_cache}")
        print("       If you haven't renamed the old cache yet:")
        print("       mv datasets/waymo/cache datasets/waymo/cache_train")
        return

    if len(val_dataset) == 0:
        print(f"[ERROR] Empty validation cache: {val_cache}")
        print("       Run first, in the METRICS container:")
        print("       python3 -m src.core.waymo_preprocessor --split validation --shards 0,1,2")
        return

    print(f"[OK] Training (official 'training' split):      {len(train_dataset)} examples")
    print(f"[OK] Validation (official 'validation' split): {len(val_dataset)} examples")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    model = SimpleTrajectoryPredictor(input_steps=11, output_steps=80).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float("inf")

    epochs = 25
    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        start_time = time.time()

        for history, future_gt, future_mask, agent_type in train_loader:
            history = history.to(device)
            future_gt = future_gt.to(device)
            future_mask = future_mask.to(device)

            optimizer.zero_grad()
            outputs = model(history)

            per_ex_loss = masked_mse_per_example(outputs, future_gt, future_mask)
            loss = per_ex_loss.mean()
            loss.backward()
            optimizer.step()

            train_loss_sum += per_ex_loss.sum().item()
            train_count += per_ex_loss.numel()

        avg_train_loss = train_loss_sum / train_count

        # --- Validation (with breakdown by agent type) ---
        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        per_type_stats = {}  # {type: [loss_sum, count]}

        with torch.no_grad():
            for history, future_gt, future_mask, agent_type in val_loader:
                history = history.to(device)
                future_gt = future_gt.to(device)
                future_mask = future_mask.to(device)

                outputs = model(history)
                per_ex_loss = masked_mse_per_example(outputs, future_gt, future_mask)

                val_loss_sum += per_ex_loss.sum().item()
                val_count += per_ex_loss.numel()

                for t, l in zip(agent_type.tolist(), per_ex_loss.cpu().tolist()):
                    if t not in per_type_stats:
                        per_type_stats[t] = [0.0, 0]
                    per_type_stats[t][0] += l
                    per_type_stats[t][1] += 1

        avg_val_loss = val_loss_sum / val_count

        checkpoint_path = os.path.join(checkpoint_dir, f"motion_model_e{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(checkpoint_dir, "motion_model_best.pth")
            torch.save(model.state_dict(), best_path)
            marker = "  <- best so far"
        else:
            marker = ""

        duration = time.time() - start_time
        print(f"[OK] Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}{marker} | "
              f"Time: {duration:.2f}s")

        breakdown = []
        for t, (loss_sum, n) in sorted(per_type_stats.items()):
            name = OBJECT_TYPE_NAMES.get(t, f"TYPE_{t}")
            breakdown.append(f"{name}: {loss_sum / n:.2f} (n={n})")
        print(f"         Val by type -> {' | '.join(breakdown)}")

    print("\n[SUCCESS] Training finished.")
    print(f"Best checkpoint: motion_model_best.pth (Val Loss: {best_val_loss:.4f})")
    print(f"Saved at: {checkpoint_dir}")


if __name__ == "__main__":
    train()
