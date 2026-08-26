import os
import sys
import argparse
from collections import defaultdict

import numpy as np
import torch

# SIBLING of run_inference_social (V2 = social + map). Changes vs V1, mechanical:
#   (1) map dataset/model;  (2) model input = target + neighbors + mask + the 3
#       map tensors;  (3) flags --n-map-polylines / --n-points-per-polyline /
#       --use-map-type (MUST match training).
# The agent->SDC INVERSION is IDENTICAL: the map is ONLY input; the output stays
# in the target's agent frame and is inverted back to SDC with the target's
# pose@10, exactly as in V0/V1. The map never enters the inversion.
from src.motion.vectorized_social_map_model import VectorizedSocialMapTrajectoryPredictor
from src.core.waymo_pytorch_dataset_map import WaymoMotionDatasetMap
from src.core.waymo_pytorch_dataset_agentcentric import (
    WaymoMotionDatasetAgentCentric, ANCHOR_FRAME, FS_X, FS_Y, FS_HEADING,
)

# RUN THIS SCRIPT IN THE GPU CONTAINER.

NUM_MODES = 6
N_NEIGHBORS = 16
N_MAP_POLYLINES = 128
N_POINTS_PER_POLYLINE = 20
FEATURES = ("x", "y", "heading", "vx", "vy")

CACHE_DIR = "/workspace/datasets/waymo/cache_val_map"
CHECKPOINT_ROOT = os.environ.get("CHECKPOINT_ROOT", "/workspace/experiments/checkpoints")
PRED_ROOT = os.environ.get("PRED_ROOT", "/workspace/datasets/waymo/predictions")

MODEL_NAME = "vectorized_social_map"
CHECKPOINT_NAME = "map_best.pth"


def run_inference(tag, seed=None, standardize=False, n_neighbors=N_NEIGHBORS,
                  n_map_polylines=N_MAP_POLYLINES,
                  n_points_per_polyline=N_POINTS_PER_POLYLINE, use_map_type=False):
    """
    tag:              experiment identifier, e.g. 'cls20'.
    seed:             selects the ..._seed<n> folder, matching train.
    standardize:      selects the _std checkpoint. Stats come from the checkpoint
                      BUFFERS (load_state_dict); this flag only composes the path.
                      (V2 grid is raw-only; kept for parity.)
    n_neighbors:      MUST match training (social branch top-K).
    n_map_polylines:  MUST match training (map branch top-K). A different M changes
                      the polyline distribution the model saw -> train/inference
                      mismatch (silently degrades metrics).
    n_points_per_polyline: MUST match training (map resampling).
    use_map_type:     MUST match training. A mismatch makes map_type_emb.weight
                      missing/unexpected in load_state_dict -> the guard raises.

    Agent-centric arm is IMPLICIT in V2 (arm='agent'):
        Checkpoint: checkpoints/vectorized_social_map_<tag>_agent[_seed<n>][_std][_type]/map_best.pth
        Output:     predictions/vectorized_social_map_<tag>_agent[_seed<n>][_std][_type]/
    """
    arm = "agent"
    base_tag = f"{tag}_{arm}"
    run_tag = base_tag if seed is None else f"{base_tag}_seed{seed}"
    if standardize:
        run_tag = f"{run_tag}_std"
    if use_map_type:
        run_tag = f"{run_tag}_type"
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

    dataset = WaymoMotionDatasetMap(
        CACHE_DIR, n_neighbors=n_neighbors, n_map_polylines=n_map_polylines,
        n_points_per_polyline=n_points_per_polyline, features=FEATURES)
    if len(dataset) == 0:
        print(f"[ERROR] No target agents in validation cache: {CACHE_DIR}")
        sys.exit(1)
    n_features = dataset.n_features

    model = VectorizedSocialMapTrajectoryPredictor(
        input_steps=11, output_steps=80, num_modes=NUM_MODES, n_features=n_features,
        use_map_type=use_map_type
    ).to(device)

    # Tolerant load of the standardization buffers (same logic as V0/V1). A V2
    # checkpoint ALWAYS has the buffers (register_buffer in __init__), so the
    # 'missing' branch does not fire for V2 -- kept for consistency/defense. The
    # map modules (map_subgraph/map_proj/map_cross_attn) and, when use_map_type,
    # map_type_emb ARE required (trained) and cannot be missing; a mismatch of
    # use_map_type vs the checkpoint surfaces here as missing/unexpected -> raise.
    STD_BUFFERS = {"feat_mean", "feat_std"}
    res = model.load_state_dict(torch.load(checkpoint_path, map_location=device),
                                strict=False)
    missing = set(res.missing_keys)
    unexpected = set(res.unexpected_keys)
    if not missing.issubset(STD_BUFFERS) or unexpected:
        raise RuntimeError(
            f"incompatible checkpoint: missing={sorted(missing)} "
            f"unexpected={sorted(unexpected)}. Only the buffers "
            f"{sorted(STD_BUFFERS)} may be missing, and nothing may be extra. "
            f"(A use_map_type mismatch vs training triggers this.)"
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
    print(f"INFO: arm={arm} | n_features={n_features} | n_neighbors={n_neighbors} "
          f"| M={n_map_polylines} | Np={n_points_per_polyline} | use_map_type={use_map_type}")
    print(f"INFO: generating social+map predictions for {len(by_file)} scenarios "
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
                # Input built by the SAME dataset transform used in training
                # (train/inference parity by construction): target + neighbors +
                # mask + the 3 map tensors.
                (x_past, neighbors, nmask, map_polylines, map_type, map_mask,
                 _, _, _) = dataset[idx]
                x_past = x_past.unsqueeze(0).to(device)          # [1,11,F]
                neighbors = neighbors.unsqueeze(0).to(device)     # [1,K,11,F]
                nmask = nmask.unsqueeze(0).to(device)             # [1,K]
                map_polylines = map_polylines.unsqueeze(0).to(device)  # [1,M,Np,2]
                map_type = map_type.unsqueeze(0).to(device)       # [1,M]
                map_mask = map_mask.unsqueeze(0).to(device)       # [1,M]

                traj_out, scores = model(x_past, neighbors, nmask,
                                         map_polylines, map_type, map_mask)
                traj = traj_out[0].cpu().numpy().astype(np.float64)  # [K,80,2] AGENT frame

                # agent->SDC INVERSION (IDENTICAL to V0/V1): target pose@10 re-read
                # from the same cache; neighbors and map do not enter here.
                #   forward:  xy_agent = (xy_sdc - p0) @ R.T
                #   inverse:  xy_sdc   =  xy_agent      @ R  + p0
                fs = np.asarray(agents_by_id[aid]['full_state'], dtype=np.float64)
                p0 = fs[ANCHOR_FRAME, [FS_X, FS_Y]]
                theta0 = float(fs[ANCHOR_FRAME, FS_HEADING])
                R = WaymoMotionDatasetAgentCentric._rotation_neg(theta0)
                traj = traj @ R + p0                              # [K,80,2] back in SDC

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
        description="Generates vectorized+social+map predictions over the validation split"
    )
    parser.add_argument("--tag", required=True,
                        help="experiment identifier, e.g. cls20")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed used at training; selects the run folder")
    parser.add_argument("--standardize", action="store_true",
                        help="std: selects the _std checkpoint. MUST match training.")
    parser.add_argument("--n-neighbors", type=int, default=N_NEIGHBORS,
                        help=f"K neighbors (default: {N_NEIGHBORS}). MUST match training.")
    parser.add_argument("--n-map-polylines", type=int, default=N_MAP_POLYLINES,
                        help=f"M map polylines (default: {N_MAP_POLYLINES}). MUST match training.")
    parser.add_argument("--n-points-per-polyline", type=int, default=N_POINTS_PER_POLYLINE,
                        help=f"Np points/polyline (default: {N_POINTS_PER_POLYLINE}). MUST match training.")
    parser.add_argument("--use-map-type", action="store_true",
                        help="V2+type ablation. MUST match training (else load_state_dict raises).")
    args = parser.parse_args()

    run_inference(tag=args.tag, seed=args.seed, standardize=args.standardize,
                  n_neighbors=args.n_neighbors, n_map_polylines=args.n_map_polylines,
                  n_points_per_polyline=args.n_points_per_polyline,
                  use_map_type=args.use_map_type)