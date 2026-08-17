import os
import numpy as np
import tensorflow as tf
from src.core.waymo_decoder import parse_waymo_scenario

# Dataset root INSIDE the container.
# On the host this is /data/.disks/hdd3a/... ; the docker run mounts
# -v /data/.disks:/data, so here the path loses the ".disks".
SCENARIO_ROOT = "/data/hdd3a/waymo_motion/waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario"

# Configuration per OFFICIAL Waymo split. Using the official split (instead
# of a homemade split of the training shards) is what makes the metrics
# comparable with the papers in the field -- they all report on 'validation',
# since 'testing' has no annotated future (it is only for the leaderboard).
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
    Builds the full shard paths of a split from the numeric indices.

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
            print(f"WARNING: shard not found on disk, skipping: {full_path}")
    return paths


def preprocess_scenario(scenario):
    """
    Converts a Scenario proto into a dictionary with the trajectories of all
    agents, in coordinates relative to the SDC (origin and rotation taken
    from frame 10 = end of history / present).

    (Logic unchanged relative to previous versions.)
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
    shard_indices: list of shard indices to process, e.g. [0, 1, 2].
    split:         'training' or 'validation' (OFFICIAL Waymo split).
                   Determines the source folder, the file prefix, the
                   total number of shards and the destination cache folder.
    num_scenarios: TOTAL limit of scenarios to extract across all
                   shards. None = processes all scenarios available
                   in the given shards.
    """
    if split not in SPLITS:
        print(f"ERROR: invalid split '{split}'. Use one of: {list(SPLITS)}")
        return

    cache_path = SPLITS[split]["cache"]
    os.makedirs(cache_path, exist_ok=True)
    print(f"INFO: split='{split}' -> writing cache to {cache_path}")

    shard_paths = build_shard_paths(shard_indices, split)
    if not shard_paths:
        print("ERROR: no valid shard found. Check shard_indices and the split.")
        return

    print(f"INFO: Reading {len(shard_paths)} shard(s): {[os.path.basename(p) for p in shard_paths]}")

    # TFRecordDataset accepts a LIST of files directly -- concatenates
    # the reading of all shards in sequence.
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
                print(f"WARNING: scenario {processed['scenario_id']} has {n_sdc} SDC agents (expected 1).")
            if n_target == 0:
                print(f"WARNING: scenario {processed['scenario_id']} has no target agents.")

            file_path = os.path.join(cache_path, f"{processed['scenario_id']}.npy")
            np.save(file_path, processed)
            count += 1

            if count % 100 == 0:
                print(f"  ... {count} scenarios processed so far")

    print(f"INFO: Extraction complete. Split='{split}', "
          f"scenarios processed: {count}, destination: {cache_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Pre-processes Waymo Motion shards into the .npy cache"
    )
    parser.add_argument("--split", default="validation", choices=list(SPLITS),
                        help="official Waymo split to process")
    parser.add_argument("--shards", default="0,1,2",
                        help="shard indices, comma-separated")
    parser.add_argument("--limit", type=int, default=None,
                        help="scenario limit (for quick testing)")
    args = parser.parse_args()

    indices = [int(s) for s in args.shards.split(",") if s.strip() != ""]
    run_extraction(shard_indices=indices, split=args.split, num_scenarios=args.limit)