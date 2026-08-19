import os
import argparse
from collections import defaultdict

import numpy as np
import torch

from src.motion.multimodal_model import MultimodalTrajectoryPredictor
from src.core.waymo_pytorch_dataset_agentcentric import (
    WaymoMotionDatasetAgentCentric, ANCHOR_FRAME, FS_X, FS_Y, FS_HEADING,
)

# RUN THIS SCRIPT IN THE TRAINING CONTAINER (GPU) -- that's where PyTorch and
# the trained checkpoint exist.

NUM_MODES = 6

# ITEM 4 -- rich input, IDENTICAL to train_*.py. Both must agree feature-for-
# feature so the model sees at inference exactly what it saw at training. The
# input tensor is built BY THE DATASET (same transform code), not rebuilt here,
# which is the whole point: it removes any chance of a divergent hand-rolled
# transform silently corrupting the metrics.
FEATURES = ("x", "y", "heading", "vx", "vy")

# Inference ONLY over the validation cache (official split). The old version
# scanned the entire cache, which contained the training data -- the
# resulting metrics measured memorization together with generalization.
CACHE_DIR = "/workspace/datasets/waymo/cache_val"
CHECKPOINT_ROOT = "/workspace/experiments/checkpoints"
PRED_ROOT = "/workspace/datasets/waymo/predictions"


def run_inference(tag, agent_centric, seed=None):
    """
    tag:           experiment identifier, e.g. 'cls20'.
    agent_centric: selects the grid arm. MUST match the training run:
                   True  -> agent-centric normalization (predictions come out
                            in the agent frame and are inverted back to SDC
                            here before saving).
                   False -> SDC-centric (predictions already in the SDC frame).
    seed:          optional; selects the seeded run folder, matching train_*.py.

    Paths (arm is folded into the tag so {sdc, agent} never collide):
        Checkpoint: checkpoints/multimodal_<tag>_<arm>[_seed<n>]/multimodal_best.pth
        Output:     predictions/multimodal_<tag>_<arm>[_seed<n>]/
    """
    arm = "agent" if agent_centric else "sdc"
    base_tag = f"{tag}_{arm}"
    run_tag = base_tag if seed is None else f"{base_tag}_seed{seed}"
    checkpoint_path = os.path.join(
        CHECKPOINT_ROOT, f"multimodal_{run_tag}", "multimodal_best.pth"
    )
    pred_dir = os.path.join(PRED_ROOT, f"multimodal_{run_tag}")

    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        available = [d for d in os.listdir(CHECKPOINT_ROOT) if d.startswith("multimodal")]
        print(f"       Available folders: {sorted(available)}")
        return

    os.makedirs(pred_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # The dataset both flattens (scenario, target) pairs into a linear index and
    # exposes n_features. We reuse ITS transform to build every model input.
    dataset = WaymoMotionDatasetAgentCentric(
        CACHE_DIR, agent_centric=agent_centric, features=FEATURES)
    if len(dataset) == 0:
        print(f"[ERROR] No target agents in validation cache: {CACHE_DIR}")
        return
    n_features = dataset.n_features

    model = MultimodalTrajectoryPredictor(
        input_steps=11, output_steps=80, num_modes=NUM_MODES, n_features=n_features
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Regroup the flat samples back by scenario file, so the output keeps the
    # baseline layout: one .npy per scenario, keyed by agent id.
    by_file = defaultdict(list)
    for idx, (path, aid) in enumerate(dataset.samples):
        by_file[path].append((idx, aid))

    print(f"INFO: checkpoint = {checkpoint_path}")
    print(f"INFO: output     = {pred_dir}")
    print(f"INFO: arm={arm} | agent_centric={agent_centric} | n_features={n_features}")
    print(f"INFO: generating multimodal predictions for {len(by_file)} scenarios "
          f"({len(dataset)} target agents)...")

    n_done = 0
    n_agents = 0
    with torch.no_grad():
        for i, path in enumerate(sorted(by_file)):
            fname = os.path.basename(path)
            data = np.load(path, allow_pickle=True).item()
            agents_by_id = {a['id']: a for a in data['agents']}

            preds = {}
            for idx, aid in by_file[path]:
                # Model input built by the SAME dataset transform used in
                # training -> train/inference input parity by construction.
                x_past, _, _, _ = dataset[idx]
                x_past = x_past.unsqueeze(0).to(device)  # [1, 11, n_features]

                traj_out, scores = model(x_past)
                # traj_out: [1, K, 80, 2] at 10Hz (same rate as training),
                #           in the AGENT frame when agent_centric=True.
                # scores:   [1, K] in LOGITS.
                traj = traj_out[0].cpu().numpy().astype(np.float64)  # [K, 80, 2]

                # INVERSION (agent arm only): map predictions back to the SDC
                # frame that validate_motion_official compares against. The
                # pose (p0, theta0) is re-read from the SAME cache file, and the
                # rotation matrix is the dataset's own _rotation_neg -- so the
                # convention cannot drift from the forward transform.
                #   forward:  xy_agent = (xy_sdc - p0) @ R.T
                #   inverse:  xy_sdc   =  xy_agent      @ R  + p0     (R.T @ R = I)
                if agent_centric:
                    fs = np.asarray(agents_by_id[aid]['full_state'], dtype=np.float64)
                    p0 = fs[ANCHOR_FRAME, [FS_X, FS_Y]]
                    theta0 = float(fs[ANCHOR_FRAME, FS_HEADING])
                    R = WaymoMotionDatasetAgentCentric._rotation_neg(theta0)
                    traj = traj @ R + p0                 # [K, 80, 2] back in SDC frame

                # Softmax here, and not in the model: the official metric uses
                # the scores to rank the hypotheses in the mAP computation.
                probs = torch.softmax(scores, dim=1)

                # Sort by descending probability. The metric does not require
                # any ordering, but keeping the most likely mode at index 0
                # makes inspection and possible top-k cutting easier.
                order = torch.argsort(probs[0], descending=True)
                order_np = order.cpu().numpy()

                preds[aid] = {
                    'trajectories': traj[order_np].astype(np.float32),  # [K, 80, 2]
                    'scores': probs[0][order].cpu().numpy(),            # [K]
                }
                n_agents += 1

            if preds:
                np.save(os.path.join(pred_dir, fname), preds)
                n_done += 1

            if (i + 1) % 200 == 0:
                print(f"  ... {i+1}/{len(by_file)} scenarios processed")

    print(f"[SUCCESS] Predictions saved in {pred_dir}")
    print(f"          {n_done} scenarios | {n_agents} target trajectories "
          f"(x{NUM_MODES} modes each).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generates multimodal predictions over the validation split"
    )
    parser.add_argument("--tag", required=True,
                        help="experiment identifier, e.g. cls1, cls20, cls50, cls100")
    parser.add_argument("--agent-centric", action="store_true",
                        help="agent-centric arm; MUST match training. Omit for SDC-centric.")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed used at training; selects the run folder (..._seed<n>)")
    args = parser.parse_args()

    run_inference(tag=args.tag, agent_centric=args.agent_centric, seed=args.seed)
