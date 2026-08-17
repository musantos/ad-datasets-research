import os
import numpy as np
import torch

from src.motion.simple_model import SimpleTrajectoryPredictor

# RUN THIS SCRIPT IN THE TRAINING CONTAINER (GPU) -- that's where PyTorch
# and the trained checkpoint exist.

# Inference ONLY over the validation cache (official Waymo split).
# The previous version scanned the entire cache, which also contained the
# training data -- the resulting metrics measured memorization together
# with generalization.
CACHE_DIR = "/workspace/datasets/waymo/cache_val"
PRED_DIR = "/workspace/datasets/waymo/predictions/baseline"
CHECKPOINT_PATH = "/workspace/experiments/checkpoints/baseline_oficial/motion_model_best.pth"


def run_inference():
    os.makedirs(PRED_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"[ERROR] Checkpoint not found: {CHECKPOINT_PATH}")
        print("       Run first: python3 -m src.motion.train_motion")
        return

    model = SimpleTrajectoryPredictor(input_steps=11, output_steps=80).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.npy')]
    if not files:
        print(f"[ERROR] Empty validation cache: {CACHE_DIR}")
        return

    print(f"INFO: generating predictions for up to {len(files)} scenarios...")

    n_done = 0
    n_agents = 0
    with torch.no_grad():
        for i, fname in enumerate(files):
            path = os.path.join(CACHE_DIR, fname)
            data = np.load(path, allow_pickle=True).item()

            preds = {}
            for agent in data['agents']:
                if not agent.get('is_target', False):
                    continue

                # Same logic of zeroing invalid past frames used in training
                # (waymo_pytorch_dataset.py).
                traj = agent['trajectory'].copy()
                mask = agent['mask']
                traj[~mask] = 0.0

                x_past = torch.tensor(traj[:11, :], dtype=torch.float32)
                x_past = x_past.unsqueeze(0).to(device)  # [1, 11, 2]

                pred = model(x_past)  # [1, 80, 2], at 10Hz (same rate as training)
                preds[agent['id']] = pred.squeeze(0).cpu().numpy()
                n_agents += 1

            if preds:
                out_path = os.path.join(PRED_DIR, fname)
                np.save(out_path, preds)
                n_done += 1

            if (i + 1) % 100 == 0:
                print(f"  ... {i+1}/{len(files)} scenarios processed")

    print(f"[SUCCESS] Predictions saved in {PRED_DIR}")
    print(f"          {n_done} scenarios with at least 1 target agent | "
          f"{n_agents} target trajectories.")


if __name__ == "__main__":
    run_inference()
