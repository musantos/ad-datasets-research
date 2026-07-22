import os
import numpy as np
import matplotlib.pyplot as plt

from src.core.waymo_decoder import parse_waymo_scenario

# NOTA: import ajustado para usar o decoder compartilhado do projeto
# (src.core.waymo_decoder), em vez de reimplementar o parsing do proto
# aqui. Mantem consistencia com o resto do pipeline.

DATA_PATH = "/data/hdd3a/waymo_motion/waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario/training/training.tfrecord-00002-of-01000" # 00000


def visualize_scenario(scenario_idx=0):
    import tensorflow as tf  # import local, so quando a funcao roda de fato

    if not os.path.exists(DATA_PATH):
        print(f"Erro: Arquivo nao encontrado em {DATA_PATH}")
        return

    dataset = tf.data.TFRecordDataset(DATA_PATH, compression_type='')

    for i, data in enumerate(dataset.skip(scenario_idx).take(1)):
        scenario = parse_waymo_scenario(data)
        print(f"Visualizando Cenario ID: {scenario.scenario_id}")

        # CORRECAO: indices de agentes-alvo (tracks_to_predict), para
        # tambem destaca-los no plot -- sao os agentes que realmente
        # importam para o problema de Motion Prediction.
        target_indices = {req.track_index for req in scenario.tracks_to_predict}
        sdc_idx = scenario.sdc_track_index

        plt.figure(figsize=(12, 12))

        # 1. Desenhar o mapa (Roadgraph)
        print("[*] Desenhando mapa...")
        for map_feature in scenario.map_features:
            if map_feature.HasField('lane'):
                polyline = np.array([[p.x, p.y] for p in map_feature.lane.polyline])
                plt.plot(polyline[:, 0], polyline[:, 1], 'gray', alpha=0.3, linewidth=1)
            elif map_feature.HasField('road_edge'):
                polyline = np.array([[p.x, p.y] for p in map_feature.road_edge.polyline])
                plt.plot(polyline[:, 0], polyline[:, 1], 'black', alpha=0.5, linewidth=1.5)
            elif map_feature.HasField('road_line'):
                polyline = np.array([[p.x, p.y] for p in map_feature.road_line.polyline])
                plt.plot(polyline[:, 0], polyline[:, 1], 'gray', linestyle='--', alpha=0.5, linewidth=1)

        # 2. Desenhar trajetorias dos agentes
        print("[*] Desenhando trajetorias...")
        sdc_plotted = False
        target_plotted = False

        # CORRECAO: usamos enumerate() para ter o INDICE de cada track na
        # lista original, e comparamos esse indice com sdc_idx e com
        # target_indices. Antes o codigo comparava "track.id ==
        # scenario.sdc_track_index" -- id e indice sao coisas diferentes,
        # entao essa comparacao praticamente nunca era verdadeira.
        for idx, track in enumerate(scenario.tracks):
            states = [s for s in track.states if s.valid]
            if not states:
                continue

            traj = np.array([[s.center_x, s.center_y] for s in states])

            if idx == sdc_idx:
                label = 'SDC (Ego)' if not sdc_plotted else None
                plt.plot(traj[:, 0], traj[:, 1], 'red', linewidth=3, label=label, zorder=5)
                sdc_plotted = True
            elif idx in target_indices:
                label = 'Agente-alvo (a prever)' if not target_plotted else None
                plt.plot(traj[:, 0], traj[:, 1], 'orange', linewidth=2, label=label, zorder=4)
                target_plotted = True
            else:
                plt.plot(traj[:, 0], traj[:, 1], 'blue', alpha=0.3, linewidth=1, zorder=1)

        plt.title(f"Waymo Open Motion Dataset - Scenario {scenario.scenario_id}")
        plt.xlabel("X (metros)")
        plt.ylabel("Y (metros)")
        plt.axis('equal')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='best')

        output_path = f"/workspace/src/scenario_{scenario.scenario_id}.png"
        plt.savefig(output_path)
        print(f"Visualizacao salva em: {output_path}")
        plt.close()


if __name__ == "__main__":
    visualize_scenario(0)
