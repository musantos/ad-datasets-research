import os
import sys
import argparse
from collections import defaultdict

import numpy as np
import torch

# SIBLING of run_inference_vectorized (V1 = social). Changes vs V0, mechanical:
#   (1) social dataset/model;  (2) model input = target + neighbors + mask;
#   (3) flag --n-neighbors (MUST match training).
# The agent->SDC INVERSION is IDENTICAL: neighbors are ONLY input; the output stays
# in the TARGET's agent frame and is inverted back to SDC with the target's pose@10,
# exactly as in V0.
from src.motion.vectorized_social_model import VectorizedSocialTrajectoryPredictor
from src.core.waymo_pytorch_dataset_social import WaymoMotionDatasetSocial
from src.core.waymo_pytorch_dataset_agentcentric import (
    WaymoMotionDatasetAgentCentric, ANCHOR_FRAME, FS_X, FS_Y, FS_HEADING,
)

# RUN THIS SCRIPT IN THE GPU CONTAINER.

NUM_MODES = 6
N_NEIGHBORS = 16
FEATURES = ("x", "y", "heading", "vx", "vy")

CACHE_DIR = "/workspace/datasets/waymo/cache_val"
CHECKPOINT_ROOT = os.environ.get("CHECKPOINT_ROOT", "/workspace/experiments/checkpoints")
PRED_ROOT = os.environ.get("PRED_ROOT", "/workspace/datasets/waymo/predictions")

MODEL_NAME = "vectorized_social"
CHECKPOINT_NAME = "social_best.pth"


def run_inference(tag, seed=None, standardize=False, n_neighbors=N_NEIGHBORS):
    """
    tag:         experiment identifier, e.g. 'cls20'.
    seed:        selects the ..._seed<n> folder, matching train.
    standardize: selects the _std checkpoint. The stats come from the checkpoint
                 BUFFERS (load_state_dict); this flag only composes the path.
    n_neighbors: MUST match training. The top-K selection changes the distribution
                 of neighbors the model sees -> a K different from training is a
                 train/inference mismatch (silently degrades metrics).

    Agent-centric arm is IMPLICIT in V1 (arm='agent'):
        Checkpoint: checkpoints/vectorized_social_<tag>_agent[_seed<n>][_std]/social_best.pth
        Output:     predictions/vectorized_social_<tag>_agent[_seed<n>][_std]/
    """
    arm = "agent"
    base_tag = f"{tag}_{arm}"
    run_tag = base_tag if seed is None else f"{base_tag}_seed{seed}"
    if standardize:
        run_tag = f"{run_tag}_std"
    checkpoint_path = os.path.join(
        CHECKPOINT_ROOT, f"{MODEL_NAME}_{run_tag}", CHECKPOINT_NAME
    )
    pred_dir = os.path.join(PRED_ROOT, f"{MODEL_NAME}_{run_tag}")

    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        available = [d for d in os.listdir(CHECKPOINT_ROOT) if d.startswith(MODEL_NAME)]
        print(f"       Available folders: {sorted(available)}")
        sys.exit(1)

    os.makedirs(pred_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = WaymoMotionDatasetSocial(
        CACHE_DIR, n_neighbors=n_neighbors, features=FEATURES)
    if len(dataset) == 0:
        print(f"[ERROR] No target agents in validation cache: {CACHE_DIR}")
        sys.exit(1)
    n_features = dataset.n_features

    model = VectorizedSocialTrajectoryPredictor(
        input_steps=11, output_steps=80, num_modes=NUM_MODES, n_features=n_features
    ).to(device)

    # TOLERANT load of the standardization buffers (same logic as V0). A V1
    # checkpoint ALWAYS has the buffers (register_buffer in __init__), so the
    # 'missing' branch below does not fire for V1 -- kept for consistency and
    # defense. cross_attn/attn_norm ARE required (trained) and cannot be
    # missing; if they are, it's a real bug and the guard raises.
    STD_BUFFERS = {"feat_mean", "feat_std"}
    res = model.load_state_dict(torch.load(checkpoint_path, map_location=device),
                                strict=False)
    missing = set(res.missing_keys)
    unexpected = set(res.unexpected_keys)
    if not missing.issubset(STD_BUFFERS) or unexpected:
        raise RuntimeError(
            f"incompatible checkpoint: missing={sorted(missing)} "
            f"unexpected={sorted(unexpected)}. Only the buffers "
            f"{sorted(STD_BUFFERS)} may be missing, and nothing may be extra."
        )
    if missing:
        if standardize:
            raise RuntimeError(
                f"run --standardize but the checkpoint lacks {sorted(missing)}: "
                "standardization stats lost. Aborting."
            )
        print(f"[INFO] checkpoint without buffers {sorted(missing)} -> identity (raw).")
    model.eval()

    by_file = defaultdict(list)
    for idx, (path, aid) in enumerate(dataset.samples):
        by_file[path].append((idx, aid))

    print(f"INFO: checkpoint = {checkpoint_path}")
    print(f"INFO: output     = {pred_dir}")
    print(f"INFO: arm={arm} | n_features={n_features} | n_neighbors={n_neighbors}")
    print(f"INFO: generating social predictions for {len(by_file)} scenarios "
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
                # Input built by the SAME dataset transform (train/inference
                # parity by construction): target + neighbors + mask.
                x_past, neighbors, nmask, _, _, _ = dataset[idx]
                x_past = x_past.unsqueeze(0).to(device)       # [1,11,F]
                neighbors = neighbors.unsqueeze(0).to(device)  # [1,K,11,F]
                nmask = nmask.unsqueeze(0).to(device)          # [1,K]

                traj_out, scores = model(x_past, neighbors, nmask)
                traj = traj_out[0].cpu().numpy().astype(np.float64)  # [K,80,2] frame AGENTE

                # agent->SDC INVERSION (IDENTICAL to V0): target pose@10 re-read
                # from the same cache; neighbors do not enter here.
                #   forward:  xy_agent = (xy_sdc - p0) @ R.T
                #   inverse:  xy_sdc   =  xy_agent      @ R  + p0
                fs = np.asarray(agents_by_id[aid]['full_state'], dtype=np.float64)
                p0 = fs[ANCHOR_FRAME, [FS_X, FS_Y]]
                theta0 = float(fs[ANCHOR_FRAME, FS_HEADING])
                R = WaymoMotionDatasetAgentCentric._rotation_neg(theta0)
                traj = traj @ R + p0                          # [K,80,2] back in SDC

                probs = torch.softmax(scores, dim=1)
                order = torch.argsort(probs[0], descending=True)
                order_np = order.cpu().numpy()

                preds[aid] = {
                    'trajectories': traj[order_np].astype(np.float32),  # [K,80,2]
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
        description="Generates vectorized+social predictions over the validation split"
    )
    parser.add_argument("--tag", required=True,
                        help="experiment identifier, e.g. cls20")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed used at training; selects the run folder")
    parser.add_argument("--standardize", action="store_true",
                        help="std: selects the _std checkpoint. MUST match training.")
    parser.add_argument("--n-neighbors", type=int, default=N_NEIGHBORS,
                        help=f"K neighbors (default: {N_NEIGHBORS}). MUST match training.")
    args = parser.parse_args()

    run_inference(tag=args.tag, seed=args.seed, standardize=args.standardize,
                  n_neighbors=args.n_neighbors)