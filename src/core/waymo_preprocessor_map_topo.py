import os
import numpy as np
import tensorflow as tf
from src.core.waymo_decoder import parse_waymo_scenario

# SIBLING of waymo_preprocessor_map (V3 = lane topology). Neither the V0/V1
# preprocessor nor the V2 map preprocessor is edited. This file adds lane
# ADJACENCY (the single new V3 variable) to the roadgraph contract and writes to
# a SEPARATE, versioned cache (suffix "_map_topo") so the closed V0/V1 and the
# V2 map cache both keep their own caches untouched.
#
# The agent branch and the map GEOMETRY transform are byte-for-byte the V2 map
# preprocessor (preprocess_scenario / extract_roadgraph); the ONLY addition is
# the four adjacency lists per lane. So the resulting cache is a STRICT SUPERSET
# of cache_*_map: the V2 loader (which ignores the extra keys) runs on it
# unchanged, and the V3 loader additionally reads the adjacency. One cache serves
# both V2 and V3 in the final mega-run.
#
# Field names/id space are NOT taken from memory: probe_lane_topology.py verified
# on a real 1.3.1 shard that (a) entry_lanes/exit_lanes are repeated int64
# feature ids, (b) {left,right}_neighbors are repeated LaneNeighbor messages whose
# `feature_id` is the referenced lane id, (c) every referenced id lives in the
# scene's MapFeature id space (== int(mf.id), what the roadgraph stores as 'id').
SCENARIO_ROOT = ("/data/hdd3a/waymo_motion/waymo_open_dataset_motion_v_1_3_1"
                 "/uncompressed/scenario")

# Same source folders/prefixes/shard counts as the V2 map preprocessor; ONLY the
# destination cache differs (new versioned dir, suffix "_map_topo").
SPLITS = {
    "training": {
        "dir": os.path.join(SCENARIO_ROOT, "training"),
        "prefix": "training",
        "total_shards": 1000,
        "cache": "/workspace/datasets/waymo/cache_train_map_topo",
    },
    "validation": {
        "dir": os.path.join(SCENARIO_ROOT, "validation"),
        "prefix": "validation",
        "total_shards": 150,
        "cache": "/workspace/datasets/waymo/cache_val_map_topo",
    },
}

# Same three polyline layers as V2 (geometry contract unchanged).
_MAP_POLYLINE_LAYERS = ("lane", "road_edge", "road_line")

# Adjacency is a LANE-only concept. Non-lane polyline entries (road_edge,
# road_line) carry EMPTY lists so every roadgraph entry has the same keys and the
# loader reads them without .get(). Relations kept separate (not merged into one
# "connected" list) so the V3 MODEL decides how many edge channels to use --
# Stage A stores all four; Stage B (the GNN) chooses the encoding.
_ADJ_KEYS = ("entry_lanes", "exit_lanes", "left_neighbors", "right_neighbors")


def build_shard_paths(shard_indices, split):
    """Builds the full shard paths of a split from the numeric indices (identical
    to the V2 preprocessor; duplicated so this sibling stands alone)."""
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


def _lane_adjacency(feat):
    """Reads the four adjacency lists off a LaneCenter sub-message as raw feature
    ids (int). entry_lanes/exit_lanes are repeated int64; {left,right}_neighbors
    are repeated LaneNeighbor messages -> take `.feature_id`. Read defensively so
    a lane missing a field (dead-end etc.) yields an empty list, not a crash."""
    return {
        "entry_lanes": [int(x) for x in getattr(feat, "entry_lanes", [])],
        "exit_lanes": [int(x) for x in getattr(feat, "exit_lanes", [])],
        "left_neighbors": [int(nb.feature_id)
                           for nb in getattr(feat, "left_neighbors", [])],
        "right_neighbors": [int(nb.feature_id)
                            for nb in getattr(feat, "right_neighbors", [])],
    }


def extract_roadgraph(scenario, origin, rotation_matrix):
    """V2 geometry extraction PLUS lane adjacency. The geometry block (layer
    filter, polyline transform, `type`) is byte-for-byte the V2 preprocessor; the
    ONLY addition is the adjacency dict on lanes (empty lists on non-lanes).

    Transform per polyline point: (xy_world - origin) @ rotation_matrix -- the
    exact agent position transform (same origin, same R(+angle)); a map point is
    geometrically just a position. Adjacency is id-based and frame-invariant, so
    it is unaffected by the rotation (no geometry to get wrong here).

    Returns a list of dicts, one per polyline map feature:
        id            int    -- proto feature id (== the ids adjacency references)
        feature_type  str    -- 'lane' | 'road_edge' | 'road_line'
        type          int    -- raw proto enum of the internal `type` field
        polyline      [P,2]  -- SDC-frame xy (z dropped), float64
        entry_lanes   [int]  -- predecessor lane ids   (empty on non-lane)
        exit_lanes    [int]  -- successor lane ids      (empty on non-lane)
        left_neighbors  [int] -- left adjacent lane ids  (empty on non-lane)
        right_neighbors [int] -- right adjacent lane ids (empty on non-lane)

    Empty polylines are skipped (defensive), same as V2.
    """
    roadgraph = []
    for mf in scenario.map_features:
        kind = mf.WhichOneof("feature_data")
        if kind not in _MAP_POLYLINE_LAYERS:
            continue
        feat = getattr(mf, kind)

        assert hasattr(feat, "polyline"), (
            f"map layer '{kind}' has no `polyline` field (is it a polygon?); "
            f"_MAP_POLYLINE_LAYERS must list polyline features only."
        )

        pts = feat.polyline
        if len(pts) == 0:
            continue

        xy = np.array([[p.x, p.y] for p in pts], dtype=np.float64)   # [P,2] world
        xy_rot = (xy - origin) @ rotation_matrix                     # [P,2] SDC frame
        ftype = int(getattr(feat, "type", 0))

        entry = {
            'id': int(mf.id),
            'feature_type': kind,
            'type': ftype,
            'polyline': xy_rot,
        }
        if kind == "lane":
            entry.update(_lane_adjacency(feat))
        else:
            for k in _ADJ_KEYS:                       # uniform contract
                entry[k] = []
        roadgraph.append(entry)
    return roadgraph


def preprocess_scenario(scenario):
    """Scenario proto -> dict of agent trajectories PLUS the roadgraph with lane
    adjacency, all in the SDC frame (origin/rotation from frame 10). The agent
    branch is byte-for-byte the V2 map preprocessor; only data['roadgraph'] now
    carries adjacency (see extract_roadgraph)."""
    sdc_idx = scenario.sdc_track_index
    sdc_state = scenario.tracks[sdc_idx].states[10]

    if not sdc_state.valid:
        return None

    origin_x = sdc_state.center_x
    origin_y = sdc_state.center_y
    angle = sdc_state.heading

    # R(+angle) so that under the row-vector convention `p @ R` applies R(-angle)
    # to points, putting xy/vel in the SAME frame as heading_rel AND the
    # roadgraph (extract_roadgraph uses this exact matrix). Byte-for-byte with
    # the V2 preprocessor; see its comment for the full derivation.
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
    """Extract shards into cache_*_map_topo (agents + roadgraph-with-adjacency)."""
    if split not in SPLITS:
        print(f"ERROR: invalid split '{split}'. Use one of: {list(SPLITS)}")
        return

    cache_path = SPLITS[split]["cache"]
    os.makedirs(cache_path, exist_ok=True)
    print(f"INFO: split='{split}' -> writing MAP+TOPO cache to {cache_path}")

    shard_paths = build_shard_paths(shard_indices, split)
    if not shard_paths:
        print("ERROR: no valid shard found. Check shard_indices and the split.")
        return

    print(f"INFO: Reading {len(shard_paths)} shard(s): "
          f"{[os.path.basename(p) for p in shard_paths]}")

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
                print(f"WARNING: scenario {processed['scenario_id']} has {n_sdc} "
                      f"SDC agents (expected 1).")
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


def verify_contract(shard_indices, split, limit):
    """GATE (writes nothing): process `limit` scenarios in memory and check the
    topology cache contract before the expensive real extraction.

    HARD asserts (structure): every roadgraph entry has all four adjacency keys
    as lists; non-lane entries carry empty lists. REPORTED (not asserted, so data
    quirks do not fail the gate): fraction of adjacency ids that resolve to a KEPT
    roadgraph entry, and of those, the fraction resolving to a LANE entry. A high
    resolution rate confirms the id space matches (== the probe's premise, now
    re-checked through the actual cache dict, not the raw proto). Low rates would
    flag a wrong id field or a bad layer filter."""
    if split not in SPLITS:
        print(f"ERROR: invalid split '{split}'. Use one of: {list(SPLITS)}")
        return

    shard_paths = build_shard_paths(shard_indices, split)
    if not shard_paths:
        print("ERROR: no valid shard found.")
        return

    dataset = tf.data.TFRecordDataset(shard_paths, compression_type='')

    n_scen = n_lane = n_nonlane = 0
    edge_total = {k: 0 for k in _ADJ_KEYS}
    resolved = 0
    resolved_total = 0
    to_lane = 0

    for data in dataset:
        if n_scen >= limit:
            break
        scenario = parse_waymo_scenario(data)
        processed = preprocess_scenario(scenario)
        if not processed:
            continue
        rg = processed['roadgraph']

        # id -> feature_type over the KEPT entries (what the loader will see).
        id_type = {e['id']: e['feature_type'] for e in rg}

        for e in rg:
            # HARD: structural contract.
            for k in _ADJ_KEYS:
                assert k in e and isinstance(e[k], list), (
                    f"entry id={e['id']} ({e['feature_type']}) missing/!list key {k}")
            if e['feature_type'] == 'lane':
                n_lane += 1
            else:
                n_nonlane += 1
                for k in _ADJ_KEYS:
                    assert e[k] == [], (
                        f"non-lane id={e['id']} ({e['feature_type']}) has "
                        f"non-empty {k}={e[k]} (adjacency must be lane-only).")

            # REPORTED: id-space resolution against the kept roadgraph.
            for k in _ADJ_KEYS:
                edge_total[k] += len(e[k])
                for nid in e[k]:
                    resolved_total += 1
                    ftype = id_type.get(nid)
                    if ftype is not None:
                        resolved += 1
                        if ftype == 'lane':
                            to_lane += 1
        n_scen += 1

    print(f"\n[verify] scenarios={n_scen}  lane_entries={n_lane}  "
          f"non_lane_entries={n_nonlane}")
    print(f"[verify] edges per relation: "
          + "  ".join(f"{k}={edge_total[k]}" for k in _ADJ_KEYS))
    if resolved_total:
        print(f"[verify] adjacency ids resolving to a kept entry: "
              f"{resolved}/{resolved_total} ({100.0*resolved/resolved_total:.1f}%)")
        print(f"[verify] ...of resolved, pointing to a LANE entry: "
              f"{to_lane}/{resolved} ({100.0*to_lane/max(resolved,1):.1f}%)")
        print("[verify] NOTE: <100% resolution is expected & handled downstream -- "
              "the loader drops edges to entries outside the kept top-M (and to "
              "any empty-polyline lane skipped here). The gate only requires the "
              "STRUCTURE asserts above to pass; resolution is a health signal.")
    else:
        print("[verify] no adjacency edges in the sample (increase --limit).")
    print("[verify] STRUCTURE contract OK (keys present as lists; non-lanes empty).")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Pre-process Waymo Motion shards into the map+topology .npy "
                    "cache (V3): agents + roadgraph WITH lane adjacency, "
                    "co-framed in the SDC frame."
    )
    parser.add_argument("--split", default="validation", choices=list(SPLITS),
                        help="official Waymo split to process")
    parser.add_argument("--shards", default="0,1,2",
                        help="shard indices, comma-separated")
    parser.add_argument("--limit", type=int, default=None,
                        help="scenario limit (for quick testing / --verify-only)")
    parser.add_argument("--verify-only", action="store_true",
                        help="write NOTHING: check the topology cache contract on "
                             "--limit scenarios (default 20) and exit. Run this "
                             "gate before the real extraction.")
    args = parser.parse_args()

    indices = [int(s) for s in args.shards.split(",") if s.strip() != ""]
    if args.verify_only:
        verify_contract(indices, args.split, limit=args.limit or 20)
    else:
        run_extraction(shard_indices=indices, split=args.split,
                       num_scenarios=args.limit)
