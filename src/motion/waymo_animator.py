import os
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# CORRECAO: import estava "from core.waymo_decoder import parse_waymo_scenario",
# inconsistente com a convencao do projeto (todos os outros modulos usam
# "from src.core..."). Isso provavelmente so nao quebrava porque o script
# era executado de dentro da pasta motion/ com core/ acessivel via
# PYTHONPATH relativo -- mas quebraria se executado como modulo
# (python3 -m src.motion.waymo_animator), igual os outros scripts do projeto.
from src.core.waymo_decoder import parse_waymo_scenario

DATA_PATH = "/data/hdd3a/waymo_motion/waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario/training/training.tfrecord-00002-of-01000" #00000


def create_animation(scenario_idx=0, output_name="scenario_animation_02.mp4"):
    if not os.path.exists(DATA_PATH):
        print("ERRO: Arquivo nao encontrado")
        return

    dataset = tf.data.TFRecordDataset(DATA_PATH, compression_type='')

    for i, data in enumerate(dataset.skip(scenario_idx).take(1)):
        scenario = parse_waymo_scenario(data)
        print(f"Processando Cenario: {scenario.scenario_id}")

        # NOVO: indices dos agentes-alvo (tracks_to_predict), para
        # destaca-los na animacao alem do SDC.
        target_indices = {req.track_index for req in scenario.tracks_to_predict}

        fig, ax = plt.subplots(figsize=(10, 10))

        print("[*] Extraindo mapa...")
        road_pts = []
        for feature in scenario.map_features:
            if feature.HasField('lane'):
                road_pts.extend([[p.x, p.y] for p in feature.lane.polyline])
            elif feature.HasField('road_edge'):
                road_pts.extend([[p.x, p.y] for p in feature.road_edge.polyline])

        road_pts = np.array(road_pts)

        print("[*] Preparando trajetorias...")
        tracks_data = []
        for track in scenario.tracks:
            states = [[s.center_x, s.center_y, s.valid] for s in track.states]
            tracks_data.append(states)

        tracks_data = np.array(tracks_data)  # [num_agents, num_frames, 3]
        num_agents = tracks_data.shape[0]
        num_frames = tracks_data.shape[1]

        # NOTA: a comparacao "a == scenario.sdc_track_index" abaixo ja
        # estava CORRETA na versao original -- "a" vem de range(num_agents)
        # sobre uma lista construida na mesma ordem de scenario.tracks,
        # entao "a" e de fato o indice, nao o id. Mantida sem alteracao,
        # apenas adicionado o destaque para agentes-alvo.
        def update(frame):
            ax.clear()
            if len(road_pts) > 0:
                ax.scatter(road_pts[:, 0], road_pts[:, 1], s=0.5, c='gray', alpha=0.2)

            for a in range(num_agents):
                x, y, valid = tracks_data[a, frame]
                if valid:
                    if a == scenario.sdc_track_index:
                        color, size, z = 'red', 30, 5
                    elif a in target_indices:
                        color, size, z = 'orange', 20, 4
                    else:
                        color, size, z = 'royalblue', 10, 1
                    ax.scatter(x, y, c=color, s=size, zorder=z)

            ax.set_title(f"Waymo Scenario: {scenario.scenario_id} | Frame: {frame}")
            ax.axis('equal')

        print(f"Gerando animacao ({num_frames} frames)...")
        anim = animation.FuncAnimation(fig, update, frames=num_frames, interval=100)

        output_path = f"/workspace/src/motion/{output_name}"
        try:
            anim.save(output_path, writer='ffmpeg', fps=10)
            print(f"SUCESSO: Animacao salva em {output_path}")
        except Exception as e:
            print(f"ERRO ao salvar: {e}. DICA: 'apt install ffmpeg' no container.")

        plt.close()


if __name__ == "__main__":
    create_animation(0)
