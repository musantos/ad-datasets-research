import os
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# FIX: the import was "from core.waymo_decoder import parse_waymo_scenario",
# inconsistent with the project convention (all other modules use
# "from src.core..."). This probably only did not break because the script
# was run from inside the motion/ folder with core/ accessible via a
# relative PYTHONPATH -- but it would break if run as a module
# (python3 -m src.motion.waymo_animator), like the other project scripts.
from src.core.waymo_decoder import parse_waymo_scenario

DATA_PATH = "/data/hdd3a/waymo_motion/waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario/training/training.tfrecord-00002-of-01000" #00000


def create_animation(scenario_idx=0, output_name="scenario_animation_02.mp4"):
    if not os.path.exists(DATA_PATH):
        print("ERROR: File not found")
        return

    dataset = tf.data.TFRecordDataset(DATA_PATH, compression_type='')

    for i, data in enumerate(dataset.skip(scenario_idx).take(1)):
        scenario = parse_waymo_scenario(data)
        print(f"Processing Scenario: {scenario.scenario_id}")

        # NEW: indices of the target agents (tracks_to_predict), to
        # highlight them in the animation besides the SDC.
        target_indices = {req.track_index for req in scenario.tracks_to_predict}

        fig, ax = plt.subplots(figsize=(10, 10))

        print("[*] Extracting map...")
        road_pts = []
        for feature in scenario.map_features:
            if feature.HasField('lane'):
                road_pts.extend([[p.x, p.y] for p in feature.lane.polyline])
            elif feature.HasField('road_edge'):
                road_pts.extend([[p.x, p.y] for p in feature.road_edge.polyline])

        road_pts = np.array(road_pts)

        print("[*] Preparing trajectories...")
        tracks_data = []
        for track in scenario.tracks:
            states = [[s.center_x, s.center_y, s.valid] for s in track.states]
            tracks_data.append(states)

        tracks_data = np.array(tracks_data)  # [num_agents, num_frames, 3]
        num_agents = tracks_data.shape[0]
        num_frames = tracks_data.shape[1]

        # NOTE: the comparison "a == scenario.sdc_track_index" below was
        # already CORRECT in the original version -- "a" comes from
        # range(num_agents) over a list built in the same order as
        # scenario.tracks, so "a" is indeed the index, not the id. Kept
        # unchanged, only the highlight for target agents was added.
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

        print(f"Generating animation ({num_frames} frames)...")
        anim = animation.FuncAnimation(fig, update, frames=num_frames, interval=100)

        output_path = f"/workspace/src/motion/{output_name}"
        try:
            anim.save(output_path, writer='ffmpeg', fps=10)
            print(f"SUCCESS: Animation saved at {output_path}")
        except Exception as e:
            print(f"ERROR while saving: {e}. HINT: 'apt install ffmpeg' in the container.")

        plt.close()


if __name__ == "__main__":
    create_animation(0)
