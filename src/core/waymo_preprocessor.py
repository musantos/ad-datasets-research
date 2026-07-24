import os
import numpy as np
import tensorflow as tf
from src.core.waymo_decoder import parse_waymo_scenario

# Raiz do dataset DENTRO do container.
# No host isto e /data/.disks/hdd3a/... ; o docker run monta
# -v /data/.disks:/data, entao aqui o caminho perde o ".disks".
SCENARIO_ROOT = "/data/hdd3a/waymo_motion/waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario"

# Configuracao por split OFICIAL do Waymo. Usar a divisao oficial (em vez
# de um split caseiro dos shards de treino) e o que torna as metricas
# comparaveis com os papers da area -- todos reportam sobre 'validation',
# ja que 'testing' nao tem futuro anotado (e so para o leaderboard).
SPLITS = {
    "training": {
        "dir": os.path.join(SCENARIO_ROOT, "training"),
        "prefix": "training",
        "total_shards": 1000,
        "cache": "/workspace/datasets/waymo/cache_train",
    },
    "validation": {
        "dir": os.path.join(SCENARIO_ROOT, "validation"),
        "prefix": "validation",
        "total_shards": 150,
        "cache": "/workspace/datasets/waymo/cache_val",
    },
}


def build_shard_paths(shard_indices, split):
    """
    Monta os caminhos completos dos shards de um split a partir dos
    indices numericos.

    Ex: split="validation", shard_indices=[0, 1, 2] ->
        validation.tfrecord-00000-of-00150
        validation.tfrecord-00001-of-00150
        validation.tfrecord-00002-of-00150
    """
    cfg = SPLITS[split]
    paths = []
    for idx in shard_indices:
        fname = f"{cfg['prefix']}.tfrecord-{idx:05d}-of-{cfg['total_shards']:05d}"
        full_path = os.path.join(cfg["dir"], fname)
        if os.path.exists(full_path):
            paths.append(full_path)
        else:
            print(f"AVISO: shard nao encontrado no disco, pulando: {full_path}")
    return paths


def preprocess_scenario(scenario):
    """
    Converte um Scenario proto num dicionario com as trajetorias de todos
    os agentes, em coordenadas relativas ao SDC (origem e rotacao tomadas
    do frame 10 = fim do passado / presente).

    (Logica inalterada em relacao as versoes anteriores.)
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


def run_extraction(shard_indices, split, num_scenarios=None):
    """
    shard_indices: lista de indices de shard a processar, ex: [0, 1, 2].
    split:         'training' ou 'validation' (split OFICIAL do Waymo).
                   Determina a pasta de origem, o prefixo dos arquivos, o
                   total de shards e a pasta de cache de destino.
    num_scenarios: limite TOTAL de cenarios a extrair somando todos os
                   shards. None = processa todos os cenarios disponiveis
                   nos shards informados.
    """
    if split not in SPLITS:
        print(f"ERRO: split invalido '{split}'. Use um de: {list(SPLITS)}")
        return

    cache_path = SPLITS[split]["cache"]
    os.makedirs(cache_path, exist_ok=True)
    print(f"INFO: split='{split}' -> gravando cache em {cache_path}")

    shard_paths = build_shard_paths(shard_indices, split)
    if not shard_paths:
        print("ERRO: nenhum shard valido encontrado. Verifique shard_indices e o split.")
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

            file_path = os.path.join(cache_path, f"{processed['scenario_id']}.npy")
            np.save(file_path, processed)
            count += 1

            if count % 100 == 0:
                print(f"  ... {count} cenarios processados ate agora")

    print(f"INFO: Extracao concluida. Split='{split}', "
          f"cenarios processados: {count}, destino: {cache_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Pre-processa shards do Waymo Motion para o cache .npy"
    )
    parser.add_argument("--split", default="validation", choices=list(SPLITS),
                        help="split oficial do Waymo a processar")
    parser.add_argument("--shards", default="0,1,2",
                        help="indices dos shards, separados por virgula")
    parser.add_argument("--limit", type=int, default=None,
                        help="limite de cenarios (para teste rapido)")
    args = parser.parse_args()

    indices = [int(s) for s in args.shards.split(",") if s.strip() != ""]
    run_extraction(shard_indices=indices, split=args.split, num_scenarios=args.limit)
