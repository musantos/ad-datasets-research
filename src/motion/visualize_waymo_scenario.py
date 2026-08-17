import os
import numpy as np
import matplotlib.pyplot as plt

from src.core.waymo_decoder import parse_waymo_scenario

# NOTE: import adjusted to use the project's shared decoder
# (src.core.waymo_decoder), instead of reimplementing the proto parsing
# here. Keeps consistency with the rest of the pipeline.

DATA_PATH = "/data/hdd3a/waymo_motion/waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario/training/training.tfrecord-00002-of-01000" # 00000


def visualize_scenario(scenario_idx=0):
    import tensorflow as tf  # local import, only when the function actually runs

    if not os.path.exists(DATA_PATH):
        print(f"Error: File not found at {DATA_PATH}")
        return

    dataset = tf.data.TFRecordDataset(DATA_PATH, compression_type='')

    for i, data in enumerate(dataset.skip(scenario_idx).take(1)):
        scenario = parse_waymo_scenario(data)
        print(f"Visualizing Scenario ID: {scenario.scenario_id}")

        # FIX: indices of the target agents (tracks_to_predict), to also
        # highlight them in the plot -- they are the agents that really
        # matter for the Motion Prediction problem.
        target_indices = {req.track_index for req in scenario.tracks_to_predict}
        sdc_idx = scenario.sdc_track_index

        plt.figure(figsize=(12, 12))

        # 1. Draw the map (Roadgraph)
        print("[*] Drawing map...")
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

        # 2. Draw the agents' trajectories
        print("[*] Drawing trajectories...")
        sdc_plotted = False
        target_plotted = False

        # FIX: we use enumerate() to get the INDEX of each track in the
        # original list, and compare that index with sdc_idx and with
        # target_indices. Before, the code compared "track.id ==
        # scenario.sdc_track_index" -- id and index are different things,
        # so that comparison was almost never true.
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
                label = 'Target agent (to predict)' if not target_plotted else None
                plt.plot(traj[:, 0], traj[:, 1], 'orange', linewidth=2, label=label, zorder=4)
                target_plotted = True
            else:
                plt.plot(traj[:, 0], traj[:, 1], 'blue', alpha=0.3, linewidth=1, zorder=1)

        plt.title(f"Waymo Open Motion Dataset - Scenario {scenario.scenario_id}")
        plt.xlabel("X (meters)")
        plt.ylabel("Y (meters)")
        plt.axis('equal')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='best')

        output_path = f"/workspace/src/scenario_{scenario.scenario_id}.png"
        plt.savefig(output_path)
        print(f"Visualization saved at: {output_path}")
        plt.close()


if __name__ == "__main__":
    visualize_scenario(0)
