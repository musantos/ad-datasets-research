import os
import numpy as np
import tensorflow as tf
from src.core.waymo_decoder import parse_waymo_scenario

# SIBLING of waymo_preprocessor (V2 = map-aware). The V0/V1 preprocessor is NOT
# edited: this file adds the roadgraph to the .npy contract and writes to a
# SEPARATE, versioned cache so the closed V0/V1 science keeps its own cache
# untouched (invariant of the V2 step). The agent branch is byte-for-byte the
# same logic as the original -- only the roadgraph extraction and the cache
# paths differ, so any downstream delta is attributable to the map alone.
#
# The dataset root inside the container is unchanged (the docker run mounts
# -v /data/.disks:/data, so the path loses the ".disks").
SCENARIO_ROOT = "/data/hdd3a/waymo_motion/waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario"

# Per OFFICIAL Waymo split. Same source folders/prefixes/shard counts as the
# V0/V1 preprocessor; ONLY the destination cache differs (new versioned dir,
# suffix "_map"), so the map cache can never overwrite the agent-only cache the
# closed V0/V1 grids depend on.
SPLITS = {
    "training": {
        "dir": os.path.join(SCENARIO_ROOT, "training"),
        "prefix": "training",
        "total_shards": 1000,
        "cache": "/workspace/datasets/waymo/cache_train_map",
    },
    "validation": {
        "dir": os.path.join(SCENARIO_ROOT, "validation"),
        "prefix": "validation",
        "total_shards": 150,
        "cache": "/workspace/datasets/waymo/cache_val_map",
    },
}

# Roadgraph layers kept in the V2 cache. Confirmed against the 1.3.1 proto on a
# real validation shard: these three carry a `polyline` (open line of {x,y,z})
# and an integer `type` enum. We keep geometry only (2D polyline + type) and
# drop:
#   * z            -- the model is 2D, like the agents;
#   * lane topology -- speed_limit_mph / interpolating / exit_lanes /
#                      {left,right}_neighbors / {left,right}_boundaries. Consuming
#                      the lane graph is the MTR/Wayformer step (lane-graph
#                      attention) = a DIFFERENT architectural change. V2 stays
#                      VectorNet-style (polylines as vector sets), so storing the
#                      graph now would only bloat the cache with data the V2
#                      model never reads. Topology is future work (a clean,
#                      isolated V3 hypothesis if map-as-polyline doesn't close MR).
#   * crosswalk / speed_bump / driveway -- POLYGONS (area, field `polygon`), not
#                      via polylines; they are pedestrian/area context, not the
#                      travel-lane geometry the V2 hypothesis tests (structure of
#                      the road closing 8s coverage). Kept out to avoid mixing
#                      two geometry kinds in the subgraph. Future work.
#   * stop_sign -- a single position, not a polyline. Future work.
# So the V2 roadgraph is the pure via-polyline set: the three below.
_MAP_POLYLINE_LAYERS = ("lane", "road_edge", "road_line")


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


def extract_roadgraph(scenario, origin, rotation_matrix):
    """
    Extracts the scene roadgraph and projects it into the SAME SDC frame the
    agents live in, so map and agents share one frame in the cache and the
    downstream agent-centric re-frame (in the loader) applies ONE rotation to
    both -> zero cross-frame drift by construction.

    Transform per polyline point: (xy_world - origin) @ rotation_matrix. This is
    IDENTICAL to the agent position transform in preprocess_scenario (same
    origin, same R(+angle)); a map point is geometrically just a position. The
    map has no heading and no velocity, so there is no `- angle` step and no
    velocity rotation -- orientation is implicit in the point order (the tangent
    = successive-point difference), and rotating the points rotates the tangents
    consistently. Fewer moving parts than the agent case, hence less bug surface.

    Returns a list of dicts, one per polyline map feature:
        id            int    -- proto feature id
        feature_type  str    -- 'lane' | 'road_edge' | 'road_line' | 'crosswalk'
        type          int    -- proto enum of the internal `type` field (raw,
                                 not translated; loader decides what to do)
        polyline      [P,2]  -- SDC-frame xy (z dropped), float64

    Empty polylines (defensive: a feature with zero points) are skipped.
    """
    roadgraph = []
    for mf in scenario.map_features:
        kind = mf.WhichOneof("feature_data")
        if kind not in _MAP_POLYLINE_LAYERS:
            continue
        feat = getattr(mf, kind)

        # Guard: _MAP_POLYLINE_LAYERS must contain ONLY polyline features. If a
        # polygon layer (crosswalk/speed_bump/driveway, field `polygon`) is ever
        # added by mistake, fail loud HERE rather than crash mid-extraction with
        # a bare AttributeError -- this is exactly the trap that a wrong entry
        # caused once. (Cheap: one hasattr per feature.)
        assert hasattr(feat, "polyline"), (
            f"map layer '{kind}' has no `polyline` field (is it a polygon?); "
            f"_MAP_POLYLINE_LAYERS must list polyline features only."
        )

        pts = feat.polyline
        if len(pts) == 0:
            continue

        xy = np.array([[p.x, p.y] for p in pts], dtype=np.float64)   # [P,2] world
        xy_rot = (xy - origin) @ rotation_matrix                     # [P,2] SDC frame

        # `type` is present on every polyline layer confirmed in 1.3.1 (lane,
        # road_edge, road_line, crosswalk). Read defensively so a layer without
        # it (should not happen for these four) does not crash extraction.
        ftype = int(getattr(feat, "type", 0))

        roadgraph.append({
            'id': int(mf.id),
            'feature_type': kind,
            'type': ftype,
            'polyline': xy_rot,
        })
    return roadgraph


def preprocess_scenario(scenario):
    """
    Converts a Scenario proto into a dictionary with the trajectories of all
    agents PLUS the scene roadgraph, all in coordinates relative to the SDC
    (origin and rotation taken from frame 10 = end of history / present).

    The agent branch is unchanged relative to the V0/V1 preprocessor; the only
    addition is data['roadgraph'] (see extract_roadgraph), built from the SAME
    (origin, rotation_matrix) so map and agents are co-framed.
    """
    sdc_idx = scenario.sdc_track_index
    sdc_state = scenario.tracks[sdc_idx].states[10]

    if not sdc_state.valid:
        return None

    origin_x = sdc_state.center_x
    origin_y = sdc_state.center_y
    angle = sdc_state.heading

    # State arrays are row-vectors [N, 2], so a point p transforms as `p @ R`,
    # which applies R^T to p. To express world vectors in the SDC frame at
    # frame 10 the point must be rotated by -angle_SDC; for the row-vector
    # convention that means R itself must be R(+angle) so that R^T = R(-angle).
    # This keeps xy_rot / vel_rot in the SAME frame as heading_rel (= headings
    # - angle, below), AND in the same frame as the roadgraph (extract_roadgraph
    # uses this exact rotation_matrix). The previous version built R(-angle)
    # directly, which under `p @ R` rotated positions/velocity by +angle instead
    # -- leaving heading offset from xy/vel by 2*angle_SDC. Harmless while only
    # x,y were used (Part A), but wrong the moment heading is co-used (item 4);
    # the same trap applies to the map, hence the shared matrix.
    #   R(+angle) = [[cos, -sin], [sin, cos]]
    c, s = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([[c, -s], [s, c]])
    origin = np.array([origin_x, origin_y])

    target_indices = {req.track_index for req in scenario.tracks_to_predict}

    processed_tracks = []

    for i, track in enumerate(scenario.tracks):
        xy = np.array([[st.center_x, st.center_y] for st in track.states])
        valid = np.array([st.valid for st in track.states])
        lengths = np.array([st.length for st in track.states])
        widths = np.array([st.width for st in track.states])
        headings = np.array([st.heading for st in track.states])
        vel = np.array([[st.velocity_x, st.velocity_y] for st in track.states])

        xy_rel = xy - origin
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

    roadgraph = extract_roadgraph(scenario, origin, rotation_matrix)

    return {
        'scenario_id': scenario.scenario_id,
        'agents': processed_tracks,
        'roadgraph': roadgraph,
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
    print(f"INFO: split='{split}' -> writing MAP cache to {cache_path}")

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
            n_map = len(processed['roadgraph'])
            if n_sdc != 1:
                print(f"WARNING: scenario {processed['scenario_id']} has {n_sdc} SDC agents (expected 1).")
            if n_target == 0:
                print(f"WARNING: scenario {processed['scenario_id']} has no target agents.")
            if n_map == 0:
                print(f"WARNING: scenario {processed['scenario_id']} has an empty roadgraph.")

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
        description="Pre-processes Waymo Motion shards into the map-aware .npy "
                    "cache (V2): agents + roadgraph, co-framed in the SDC frame."
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