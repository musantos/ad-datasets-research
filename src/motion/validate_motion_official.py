import os
import numpy as np
import tensorflow as tf
from google.protobuf import text_format

from waymo_open_dataset.metrics.ops import py_metrics_ops
from waymo_open_dataset.metrics.python import config_util_py as config_util
from waymo_open_dataset.protos import motion_metrics_pb2

# RODAR ESTE SCRIPT NO CONTAINER DE METRICAS (CPU) -- e onde o
# py_metrics_ops (TF/WOD) existe. Requer que run_inference.py ja tenha
# rodado no container de treino e que a pasta de predicoes esteja
# acessivel aqui (mesmo volume compartilhado do cache).

CACHE_DIR = "/workspace/datasets/waymo/cache"
PRED_DIR = "/workspace/datasets/waymo/predictions"

# Teto de agentes-alvo por cenario. O tutorial oficial usa 128 (todos os
# agentes possiveis por cenario, com mascara). Aqui simplificamos: como
# ja filtramos para os agentes-alvo (is_target), o maior numero observado
# nos seus logs foi 8 -- deixamos margem de seguranca.
# Isso e matematicamente equivalente ao approach oficial, pois os slots
# de padding tem gt_is_valid=False e pred_mask=False, entao nao
# contribuem para a metrica -- so nao e byte-a-byte identico ao codigo
# original (que usa 128 slots fixos representando TODOS os agentes da
# cena, alvo ou nao).
MAX_AGENTS = 12

TRACK_STEPS_PER_SECOND = 10
PREDICTION_STEPS_PER_SECOND = 2
TG = 91  # track_history_samples(10) + 1 + track_future_samples(80)


def build_config():
    """Config oficial do challenge, extraido do tutorial_motion_original.ipynb."""
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


def build_scenario_tensors(cache_path, pred_path):
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
        print(f"AVISO: cenario {data['scenario_id']} tem {n} agentes-alvo, "
              f"acima do MAX_AGENTS={MAX_AGENTS}. Truncando (aumente MAX_AGENTS).")

    gt_traj = np.zeros((MAX_AGENTS, TG, 7), dtype=np.float32)
    gt_valid = np.zeros((MAX_AGENTS, TG), dtype=bool)
    obj_type = np.zeros((MAX_AGENTS,), dtype=np.int64)
    obj_id = np.zeros((MAX_AGENTS,), dtype=np.int64)
    pred_traj = np.zeros((MAX_AGENTS, 16, 2), dtype=np.float32)
    pred_mask = np.zeros((MAX_AGENTS,), dtype=bool)

    # Formula oficial de downsampling (10Hz -> 2Hz), do tutorial:
    # prediction_trajectory[..., (interval - 1)::interval, :]
    interval = TRACK_STEPS_PER_SECOND // PREDICTION_STEPS_PER_SECOND  # 5

    for i, agent in enumerate(target_agents[:MAX_AGENTS]):
        gt_traj[i] = agent['full_state']       # [91, 7]
        gt_valid[i] = agent['mask']             # [91]
        obj_type[i] = int(agent['type'])
        obj_id[i] = int(agent['id'])

        full_pred = preds[agent['id']]          # [80, 2] a 10Hz
        sub_pred = full_pred[(interval - 1)::interval, :]  # [16, 2] a 2Hz
        pred_traj[i] = sub_pred
        pred_mask[i] = True

    return {
        'scenario_id': data['scenario_id'],
        'gt_trajectory': gt_traj,
        'gt_is_valid': gt_valid,
        'object_type': obj_type,
        'object_id': obj_id,
        'pred_trajectory': pred_traj,
        'pred_mask': pred_mask,
    }


def run_validation():
    config = build_config()
    metric_names = config_util.get_breakdown_names_from_motion_config(config)

    files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.npy')]

    all_gt_traj, all_gt_valid = [], []
    all_obj_type, all_obj_id, all_scenario_id = [], [], []
    all_pred_traj, all_pred_score, all_pred_idx, all_pred_idx_mask = [], [], [], []

    n_scenarios = 0
    for fname in files:
        cache_path = os.path.join(CACHE_DIR, fname)
        pred_path = os.path.join(PRED_DIR, fname)
        if not os.path.exists(pred_path):
            continue

        t = build_scenario_tensors(cache_path, pred_path)
        if t is None:
            continue

        all_gt_traj.append(t['gt_trajectory'][None])   # [1, MAX_AGENTS, TG, 7]
        all_gt_valid.append(t['gt_is_valid'][None])     # [1, MAX_AGENTS, TG]
        all_obj_type.append(t['object_type'][None])     # [1, MAX_AGENTS]
        all_obj_id.append(t['object_id'][None])
        all_scenario_id.append(t['scenario_id'])

        pred = t['pred_trajectory'][None, :, None, None, :, :]  # [1, MAX_AGENTS, 1, 1, 16, 2]
        score = np.ones((1, MAX_AGENTS, 1), dtype=np.float32)
        idx = np.tile(np.arange(MAX_AGENTS, dtype=np.int64)[None, :, None], (1, 1, 1))
        idx_mask = t['pred_mask'][None, :, None]        # [1, MAX_AGENTS, 1]

        all_pred_traj.append(pred)
        all_pred_score.append(score)
        all_pred_idx.append(idx)
        all_pred_idx_mask.append(idx_mask)

        n_scenarios += 1

    if n_scenarios == 0:
        print("ERRO: nenhum cenario com predicoes encontrado. Rode run_inference.py primeiro "
              "(no container de treino) e confirme que PRED_DIR e visivel aqui.")
        return

    print(f"INFO: validando {n_scenarios} cenarios com as metricas oficiais do Waymo Motion...")

    gt_trajectory = np.concatenate(all_gt_traj, axis=0)
    gt_is_valid = np.concatenate(all_gt_valid, axis=0)
    object_type = np.concatenate(all_obj_type, axis=0)
    object_id = np.concatenate(all_obj_id, axis=0)
    scenario_id = np.array(all_scenario_id)

    prediction_trajectory = np.concatenate(all_pred_traj, axis=0)
    prediction_score = np.concatenate(all_pred_score, axis=0)
    prediction_ground_truth_indices = np.concatenate(all_pred_idx, axis=0)
    prediction_ground_truth_indices_mask = np.concatenate(all_pred_idx_mask, axis=0)

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
    print("RESULTADO DA VALIDACAO OFICIAL (Waymo Motion Metrics)")
    print("=" * 50)
    for i, name in enumerate(metric_names):
        print(f"\n[{name}]")
        print(f"  minADE:      {float(min_ade[i]):.4f}")
        print(f"  minFDE:      {float(min_fde[i]):.4f}")
        print(f"  MissRate:    {float(miss_rate[i]):.4f}")
        print(f"  OverlapRate: {float(overlap_rate[i]):.4f}")
        print(f"  mAP:         {float(mean_average_precision[i]):.4f}")


if __name__ == "__main__":
    run_validation()
