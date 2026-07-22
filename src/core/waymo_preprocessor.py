import os
import numpy as np
import tensorflow as tf
from src.core.waymo_decoder import parse_waymo_scenario

DATA_DIR = "/data/hdd3a/waymo_motion/waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario/training"
CACHE_PATH = "/workspace/datasets/waymo/cache"
TOTAL_SHARDS = 1000  # total de shards que existem no dataset completo (00000 a 00999)


def build_shard_paths(shard_indices):
    """
    Monta os caminhos completos dos shards a partir dos indices numericos.
    Ex: shard_indices=[0, 1, 2] -> training.tfrecord-00000-of-01000,
                                    training.tfrecord-00001-of-01000,
                                    training.tfrecord-00002-of-01000
    """
    paths = []
    for idx in shard_indices:
        fname = f"training.tfrecord-{idx:05d}-of-{TOTAL_SHARDS:05d}"
        full_path = os.path.join(DATA_DIR, fname)
        if os.path.exists(full_path):
            paths.append(full_path)
        else:
            print(f"AVISO: shard nao encontrado no disco, pulando: {full_path}")
    return paths


def preprocess_scenario(scenario):
    """
    (Sem mudanca de logica em relacao a versao anterior -- so movida para
    este arquivo v3, que agora suporta multiplos shards de entrada.)
    """
    sdc_idx = scenario.sdc_track_index
    sdc_state = scenario.tracks[sdc_idx].states[10]

    if not sdc_state.valid:
        return None

    origin_x = sdc_state.center_x
    origin_y = sdc_state.center_y
    angle = sdc_state.heading

    c, s = np.cos(-angle), np.sin(-angle)
    rotation_matrix = np.array([[c, -s], [s, c]])

    target_indices = {req.track_index for req in scenario.tracks_to_predict}

    processed_tracks = []

    for i, track in enumerate(scenario.tracks):
        xy = np.array([[st.center_x, st.center_y] for st in track.states])
        valid = np.array([st.valid for st in track.states])
        lengths = np.array([st.length for st in track.states])
        widths = np.array([st.width for st in track.states])
        headings = np.array([st.heading for st in track.states])
        vel = np.array([[st.velocity_x, st.velocity_y] for st in track.states])

        xy_rel = xy - np.array([origin_x, origin_y])
        xy_rot = np.dot(xy_rel, rotation_matrix)
        vel_rot = np.dot(vel, rotation_matrix)
        heading_rel = headings - angle

        full_state = np.concatenate([
            xy_rot, lengths[:, None], widths[:, None],
            heading_rel[:, None], vel_rot,
        ], axis=1)

        if np.any(valid):
            processed_tracks.append({
                'id': track.id,
                'type': track.object_type,
                'trajectory': xy_rot,
                'full_state': full_state,
                'mask': valid,
                'is_sdc': bool(i == sdc_idx),
                'is_target': bool(i in target_indices),
            })

    return {
        'scenario_id': scenario.scenario_id,
        'agents': processed_tracks,
    }


def run_extraction(shard_indices, num_scenarios=None):
    """
    shard_indices: lista de indices de shard a processar, ex: [0, 1, 2]
                   (0 = o shard que voce ja vinha usando).
    num_scenarios: limite TOTAL de cenarios a extrair somando todos os
                   shards. None = processa todos os cenarios disponiveis
                   nos shards informados.
    """
    if not os.path.exists(CACHE_PATH):
        os.makedirs(CACHE_PATH, exist_ok=True)

    shard_paths = build_shard_paths(shard_indices)
    if not shard_paths:
        print("ERRO: nenhum shard valido encontrado. Verifique shard_indices e DATA_DIR.")
        return

    print(f"INFO: Lendo {len(shard_paths)} shard(s): {[os.path.basename(p) for p in shard_paths]}")

    # TFRecordDataset aceita uma LISTA de arquivos diretamente -- concatena
    # a leitura de todos os shards em sequencia.
    dataset = tf.data.TFRecordDataset(shard_paths, compression_type='')

    count = 0
    for data in dataset:
        if num_scenarios is not None and count >= num_scenarios:
            break

        scenario = parse_waymo_scenario(data)
        processed = preprocess_scenario(scenario)

        if processed:
            n_sdc = sum(1 for a in processed['agents'] if a['is_sdc'])
            n_target = sum(1 for a in processed['agents'] if a['is_target'])
            if n_sdc != 1:
                print(f"AVISO: cenario {processed['scenario_id']} tem {n_sdc} agentes SDC (esperado 1).")
            if n_target == 0:
                print(f"AVISO: cenario {processed['scenario_id']} sem agentes-alvo.")

            file_path = os.path.join(CACHE_PATH, f"{processed['scenario_id']}.npy")
            np.save(file_path, processed)
            count += 1

            if count % 100 == 0:
                print(f"  ... {count} cenarios processados ate agora")

    print(f"INFO: Extracao concluida. Total de cenarios processados: {count}")


if __name__ == "__main__":
    # Exemplo: processar os shards 0, 1 e 2 (0 = o mesmo de antes),
    # sem limite de cenarios (pega tudo que existir nesses 3 shards).
    run_extraction(shard_indices=[0, 1, 2], num_scenarios=None)
