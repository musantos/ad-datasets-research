import os
import csv
import argparse
import datetime
import numpy as np
import tensorflow as tf
from google.protobuf import text_format

from waymo_open_dataset.metrics.ops import py_metrics_ops
from waymo_open_dataset.metrics.python import config_util_py as config_util
from waymo_open_dataset.protos import motion_metrics_pb2

# RUN THIS SCRIPT IN THE METRICS CONTAINER (CPU) -- that is where
# py_metrics_ops (TF/WOD) exists. It requires that run_inference*.py has
# already run in the training container and that the predictions folder is
# reachable here (the same shared volume as the cache).

# Evaluation is ALWAYS on the official validation split.
CACHE_DIR = "/workspace/datasets/waymo/cache_val"
DEFAULT_PRED_DIR = "/workspace/datasets/waymo/predictions/baseline"

# Cap on target agents per scenario. The official tutorial uses 128 (every
# possible agent per scenario, with a mask). Here we simplify: since we
# already filter to target agents (is_target), the largest count seen in
# the logs was 8 -- we keep a safety margin.
# This is mathematically equivalent to the official approach, because the
# padding slots have gt_is_valid=False and pred_mask=False, so they do not
# contribute to the metric -- it just is not byte-for-byte identical to the
# original code (which uses 128 fixed slots representing ALL agents in the
# scene, target or not).
MAX_AGENTS = 12

TRACK_STEPS_PER_SECOND = 10
PREDICTION_STEPS_PER_SECOND = 2
TG = 91  # track_history_samples(10) + 1 + track_future_samples(80)

# Number of prediction steps after downsampling to 2Hz.
PRED_STEPS = 16

# Cap on supported modes. Matches max_predictions in the official config.
MAX_MODES = 6


def build_config():
    """Official challenge config, taken from tutorial_motion_original.ipynb."""
    config = motion_metrics_pb2.MotionMetricsConfig()
    config_text = """
    track_steps_per_second: 10
    prediction_steps_per_second: 2
    track_history_samples: 10
    track_future_samples: 80
    speed_lower_bound: 1.4
    speed_upper_bound: 11.0
    speed_scale_lower: 0.5
    speed_scale_upper: 1.0
    step_configurations {
      measurement_step: 5
      lateral_miss_threshold: 1.0
      longitudinal_miss_threshold: 2.0
    }
    step_configurations {
      measurement_step: 9
      lateral_miss_threshold: 1.8
      longitudinal_miss_threshold: 3.6
    }
    step_configurations {
      measurement_step: 15
      lateral_miss_threshold: 3.0
      longitudinal_miss_threshold: 6.0
    }
    max_predictions: 6
    """
    text_format.Parse(config_text, config)
    return config


def unpack_prediction(entry):
    """
    Normalize an agent's prediction to (trajectories [K,80,2], scores [K]).

    Accepts the two formats written by the inference scripts:

      - UNIMODAL   (run_inference.py):
            np.ndarray [80, 2]
        -> becomes K=1 with score 1.0, which reproduces exactly the
           previous behavior of this script.

      - MULTIMODAL (run_inference_multimodal.py):
            {'trajectories': [K, 80, 2], 'scores': [K]}

    Detection is by type, not by flag: this way the same script evaluates
    both model types with no extra parameter and no risk of the user
    picking the wrong mode.
    """
    if isinstance(entry, dict):
        traj = np.asarray(entry['trajectories'], dtype=np.float32)  # [K, 80, 2]
        scores = np.asarray(entry['scores'], dtype=np.float32)      # [K]
        if traj.ndim != 3:
            raise ValueError(f"multimodal trajectories with ndim={traj.ndim}, expected 3")
        return traj, scores

    traj = np.asarray(entry, dtype=np.float32)                      # [80, 2]
    if traj.ndim != 2:
        raise ValueError(f"unimodal trajectory with ndim={traj.ndim}, expected 2")
    return traj[None, ...], np.ones((1,), dtype=np.float32)


def build_scenario_tensors(cache_path, pred_path, n_modes):
    data = np.load(cache_path, allow_pickle=True).item()
    preds = np.load(pred_path, allow_pickle=True).item()

    target_agents = [
        a for a in data['agents']
        if a.get('is_target', False) and a['id'] in preds
    ]
    n = len(target_agents)
    if n == 0:
        return None
    if n > MAX_AGENTS:
        print(f"WARNING: scenario {data['scenario_id']} has {n} target agents, "
              f"above MAX_AGENTS={MAX_AGENTS}. Truncating (increase MAX_AGENTS).")

    gt_traj = np.zeros((MAX_AGENTS, TG, 7), dtype=np.float32)
    gt_valid = np.zeros((MAX_AGENTS, TG), dtype=bool)
    obj_type = np.zeros((MAX_AGENTS,), dtype=np.int64)
    obj_id = np.zeros((MAX_AGENTS,), dtype=np.int64)

    # Now with a mode dimension: [MAX_AGENTS, K, 16, 2] and [MAX_AGENTS, K].
    pred_traj = np.zeros((MAX_AGENTS, n_modes, PRED_STEPS, 2), dtype=np.float32)
    pred_score = np.zeros((MAX_AGENTS, n_modes), dtype=np.float32)
    pred_mask = np.zeros((MAX_AGENTS,), dtype=bool)

    # Official downsampling formula (10Hz -> 2Hz), from the tutorial:
    # prediction_trajectory[..., (interval - 1)::interval, :]
    interval = TRACK_STEPS_PER_SECOND // PREDICTION_STEPS_PER_SECOND  # 5

    n_used = 0
    for i, agent in enumerate(target_agents[:MAX_AGENTS]):
        gt_traj[i] = agent['full_state']       # [91, 7]
        gt_valid[i] = agent['mask']             # [91]
        obj_type[i] = int(agent['type'])
        obj_id[i] = int(agent['id'])

        full_pred, scores = unpack_prediction(preds[agent['id']])   # [K,80,2], [K]
        k = min(full_pred.shape[0], n_modes)

        sub_pred = full_pred[:k, (interval - 1)::interval, :]        # [k, 16, 2]
        pred_traj[i, :k] = sub_pred
        pred_score[i, :k] = scores[:k]

        # If the model produced fewer modes than n_modes, the remaining
        # slots keep a zeroed trajectory and score 0. A score of 0 makes the
        # metric rank them last; since minADE/minFDE are best-of-K, an extra
        # bad mode does not worsen the result.
        pred_mask[i] = True
        n_used += 1

    return {
        'scenario_id': data['scenario_id'],
        'gt_trajectory': gt_traj,
        'gt_is_valid': gt_valid,
        'object_type': obj_type,
        'object_id': obj_id,
        'pred_trajectory': pred_traj,
        'pred_score': pred_score,
        'pred_mask': pred_mask,
        'n_agents': n_used,
    }


def detect_n_modes(pred_dir, files):
    """
    Determine how many modes the predictions have by looking at the first
    valid file. Avoids having to pass this as a parameter and get it wrong.
    """
    for fname in files:
        path = os.path.join(pred_dir, fname)
        if not os.path.exists(path):
            continue
        preds = np.load(path, allow_pickle=True).item()
        for entry in preds.values():
            traj, _ = unpack_prediction(entry)
            return int(traj.shape[0])
    return 1


def default_csv_path(pred_dir):
    """
    Build a default CSV path so that different runs do not overwrite each
    other: results/metrics_<tag>_<YYYY-MM-DD>.csv, where <tag> is the name
    of the predictions folder (e.g. 'baseline', 'multimodal_cls20').
    """
    tag = os.path.basename(os.path.normpath(pred_dir))
    date = datetime.date.today().isoformat()
    return os.path.join("results", f"metrics_{tag}_{date}.csv")


def write_metrics_csv(csv_path, pred_dir, kind, metric_names,
                      min_ade, min_fde, miss_rate, overlap_rate,
                      mean_average_precision):
    """
    Write one row per breakdown to a simple CSV. Columns:
    breakdown_name, minADE, minFDE, MissRate, OverlapRate, mAP.
    The predictions folder and detected format go in a header comment line.
    """
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"# pred_dir={pred_dir}", f"format={kind}"])
        writer.writerow(["breakdown_name", "minADE", "minFDE",
                         "MissRate", "OverlapRate", "mAP"])
        for i, name in enumerate(metric_names):
            writer.writerow([
                name,
                f"{float(min_ade[i]):.4f}",
                f"{float(min_fde[i]):.4f}",
                f"{float(miss_rate[i]):.4f}",
                f"{float(overlap_rate[i]):.4f}",
                f"{float(mean_average_precision[i]):.4f}",
            ])
    print(f"INFO: metrics written to {csv_path}")


def run_validation(pred_dir, csv_path=None):
    config = build_config()
    metric_names = config_util.get_breakdown_names_from_motion_config(config)

    if not os.path.isdir(pred_dir):
        print(f"ERROR: predictions folder not found: {pred_dir}")
        return

    files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.npy')]

    n_modes = detect_n_modes(pred_dir, files)
    if n_modes > MAX_MODES:
        print(f"WARNING: predictions with {n_modes} modes, above max_predictions="
              f"{MAX_MODES}. Using the first {MAX_MODES}.")
        n_modes = MAX_MODES

    kind = "UNIMODAL" if n_modes == 1 else f"MULTIMODAL (K={n_modes})"
    print(f"INFO: cache  = {CACHE_DIR}")
    print(f"INFO: preds  = {pred_dir}")
    print(f"INFO: detected format = {kind}")

    all_gt_traj, all_gt_valid = [], []
    all_obj_type, all_obj_id, all_scenario_id = [], [], []
    all_pred_traj, all_pred_score, all_pred_idx, all_pred_idx_mask = [], [], [], []

    n_scenarios = 0
    n_trajectories = 0
    for fname in files:
        cache_path = os.path.join(CACHE_DIR, fname)
        pred_path = os.path.join(pred_dir, fname)
        if not os.path.exists(pred_path):
            continue

        t = build_scenario_tensors(cache_path, pred_path, n_modes)
        if t is None:
            continue

        all_gt_traj.append(t['gt_trajectory'][None])   # [1, MAX_AGENTS, TG, 7]
        all_gt_valid.append(t['gt_is_valid'][None])     # [1, MAX_AGENTS, TG]
        all_obj_type.append(t['object_type'][None])     # [1, MAX_AGENTS]
        all_obj_id.append(t['object_id'][None])
        all_scenario_id.append(t['scenario_id'])

        # Shape expected by the API:
        #   [batch, groups, top_k, agents_per_group, steps, 2]
        # Here: groups = MAX_AGENTS (1 agent per group), top_k = n_modes.
        # The previous version of this script fixed top_k=1; this is the only
        # structural change needed to evaluate the multimodal model.
        pred = t['pred_trajectory'][None, :, :, None, :, :]  # [1, MAX_AGENTS, K, 1, 16, 2]
        score = t['pred_score'][None]                        # [1, MAX_AGENTS, K]
        idx = np.tile(np.arange(MAX_AGENTS, dtype=np.int64)[None, :, None], (1, 1, 1))
        idx_mask = t['pred_mask'][None, :, None]             # [1, MAX_AGENTS, 1]

        all_pred_traj.append(pred)
        all_pred_score.append(score)
        all_pred_idx.append(idx)
        all_pred_idx_mask.append(idx_mask)

        n_scenarios += 1
        n_trajectories += t['n_agents']

    if n_scenarios == 0:
        print("ERROR: no scenario with predictions found. "
              "Run run_inference first (in the training container) and "
              "confirm the predictions folder is visible here.")
        return

    print(f"INFO: validating {n_scenarios} scenarios / {n_trajectories} target trajectories "
          f"with the official Waymo Motion metrics...")

    gt_trajectory = np.concatenate(all_gt_traj, axis=0)
    gt_is_valid = np.concatenate(all_gt_valid, axis=0)
    object_type = np.concatenate(all_obj_type, axis=0)
    object_id = np.concatenate(all_obj_id, axis=0)
    scenario_id = np.array(all_scenario_id)

    prediction_trajectory = np.concatenate(all_pred_traj, axis=0)
    prediction_score = np.concatenate(all_pred_score, axis=0)
    prediction_ground_truth_indices = np.concatenate(all_pred_idx, axis=0)
    prediction_ground_truth_indices_mask = np.concatenate(all_pred_idx_mask, axis=0)

    print(f"INFO: prediction_trajectory shape = {prediction_trajectory.shape}")
    print(f"INFO: prediction_score      shape = {prediction_score.shape}")

    (min_ade, min_fde, miss_rate, overlap_rate,
     mean_average_precision) = py_metrics_ops.motion_metrics(
        config=config.SerializeToString(),
        prediction_trajectory=tf.constant(prediction_trajectory, dtype=tf.float32),
        prediction_score=tf.constant(prediction_score, dtype=tf.float32),
        ground_truth_trajectory=tf.constant(gt_trajectory, dtype=tf.float32),
        ground_truth_is_valid=tf.constant(gt_is_valid, dtype=tf.bool),
        prediction_ground_truth_indices=tf.constant(prediction_ground_truth_indices, dtype=tf.int64),
        prediction_ground_truth_indices_mask=tf.constant(prediction_ground_truth_indices_mask, dtype=tf.bool),
        object_type=tf.constant(object_type, dtype=tf.int64),
        object_id=tf.constant(object_id, dtype=tf.int64),
        scenario_id=tf.constant(scenario_id, dtype=tf.string),
    )

    print("\n" + "=" * 50)
    print("OFFICIAL VALIDATION RESULT (Waymo Motion Metrics)")
    print(f"Predictions: {pred_dir}  |  format: {kind}")
    print("=" * 50)
    for i, name in enumerate(metric_names):
        print(f"\n[{name}]")
        print(f"  minADE:      {float(min_ade[i]):.4f}")
        print(f"  minFDE:      {float(min_fde[i]):.4f}")
        print(f"  MissRate:    {float(miss_rate[i]):.4f}")
        print(f"  OverlapRate: {float(overlap_rate[i]):.4f}")
        print(f"  mAP:         {float(mean_average_precision[i]):.4f}")

    if csv_path is None:
        csv_path = default_csv_path(pred_dir)
    write_metrics_csv(csv_path, pred_dir, kind, metric_names,
                      min_ade, min_fde, miss_rate, overlap_rate,
                      mean_average_precision)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute the official Waymo Motion metrics over the validation split"
    )
    parser.add_argument("--pred-dir", default=DEFAULT_PRED_DIR,
                        help="folder with the predictions to evaluate")
    parser.add_argument("--csv", default=None,
                        help="output CSV path (default: results/metrics_<tag>_<date>.csv)")
    args = parser.parse_args()

    run_validation(pred_dir=args.pred_dir, csv_path=args.csv)