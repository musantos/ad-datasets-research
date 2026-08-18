import os
import argparse
import numpy as np
import torch

from src.motion.sequential_model import SequentialTrajectoryPredictor

# RUN THIS SCRIPT IN THE TRAINING CONTAINER (GPU) -- that's where PyTorch and
# the trained checkpoint exist.

NUM_MODES = 6

# Inference ONLY over the validation cache (official split). The old version
# scanned the entire cache, which contained the training data -- the
# resulting metrics measured memorization together with generalization.
CACHE_DIR = "/workspace/datasets/waymo/cache_val"
CHECKPOINT_ROOT = "/workspace/experiments/checkpoints"
PRED_ROOT = "/workspace/datasets/waymo/predictions"


def run_inference(tag):
    """
    tag: identifies the experiment, e.g. 'cls20'.
         Checkpoint: experiments/checkpoints/sequential_<tag>/sequential_best.pth
         Output:     datasets/waymo/predictions/sequential_<tag>/
    """
    checkpoint_path = os.path.join(
        CHECKPOINT_ROOT, f"sequential_{tag}", "sequential_best.pth"
    )
    pred_dir = os.path.join(PRED_ROOT, f"sequential_{tag}")

    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        available = [d for d in os.listdir(CHECKPOINT_ROOT) if d.startswith("sequential")]
        print(f"       Available folders: {sorted(available)}")
        return

    os.makedirs(pred_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SequentialTrajectoryPredictor(
        input_steps=11, output_steps=80, num_modes=NUM_MODES
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.npy')]
    if not files:
        print(f"[ERROR] Empty validation cache: {CACHE_DIR}")
        return

    print(f"INFO: checkpoint = {checkpoint_path}")
    print(f"INFO: output     = {pred_dir}")
    print(f"INFO: generating sequential predictions for up to {len(files)} scenarios...")

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

                traj_out, scores = model(x_past)
                # traj_out: [1, K, 80, 2] at 10Hz (same rate as training)
                # scores:   [1, K] in LOGITS

                # Softmax here, and not in the model: the official metric uses
                # the scores to rank the hypotheses in the mAP computation.
                probs = torch.softmax(scores, dim=1)

                # Sort by descending probability. The metric does not require
                # any ordering, but having the most likely mode at index 0
                # makes inspection and possible top-k cutting easier.
                order = torch.argsort(probs[0], descending=True)

                preds[agent['id']] = {
                    'trajectories': traj_out[0][order].cpu().numpy(),  # [K, 80, 2]
                    'scores': probs[0][order].cpu().numpy(),           # [K]
                }
                n_agents += 1

            if preds:
                np.save(os.path.join(pred_dir, fname), preds)
                n_done += 1

            if (i + 1) % 200 == 0:
                print(f"  ... {i+1}/{len(files)} scenarios processed")

    print(f"[SUCCESS] Predictions saved in {pred_dir}")
    print(f"          {n_done} scenarios | {n_agents} target trajectories "
          f"(x{NUM_MODES} modes each).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generates sequential (GRU) predictions over the validation split"
    )
    parser.add_argument("--tag", required=True,
                        help="experiment identifier, e.g. cls1, cls20, cls50, cls100")
    args = parser.parse_args()

    run_inference(tag=args.tag)
