#!/usr/bin/env python3
"""
STAGE A / step 0 -- PROTO PROBE (read-only, writes NO cache).

Confirms, against a REAL WOMD 1.3.1 shard, the lane-topology field names and the
id space BEFORE the topology extractor is written against them. Project
invariant: adjacency field names come from the real proto, never from memory.

Run in the CPU/TF container (has waymo-open-dataset):
    docker exec -w /workspace cpu_env \
        python3 scripts/probe_lane_topology.py --split validation --shards 0 --limit 5

Reports, for the first few `lane` features across the first scenarios:
  * every DECLARED and every POPULATED field on the lane sub-message (descriptor
    names -- so nothing is assumed from memory);
  * the adjacency fields (entry_lanes / exit_lanes / left_neighbors /
    right_neighbors): how many, sample values, and -- for neighbors -- which
    sub-field carries the referenced id (discovered, not assumed);
  * an ID-SPACE check: every referenced neighbor id is looked up among the
    scene's MapFeature ids, confirming adjacency references the SAME id space the
    map preprocessor already stores as roadgraph[i]['id'] (= int(mf.id)).

Nothing is cached; this only prints. Paste the output back to lock the extractor.
"""
import argparse
import os

import tensorflow as tf
from src.core.waymo_decoder import parse_waymo_scenario

# Same dataset root/layout as waymo_preprocessor_map.py.
SCENARIO_ROOT = ("/data/hdd3a/waymo_motion/waymo_open_dataset_motion_v_1_3_1"
                 "/uncompressed/scenario")
SPLITS = {
    "training":   dict(dir=os.path.join(SCENARIO_ROOT, "training"),
                       prefix="training",   total=1000),
    "validation": dict(dir=os.path.join(SCENARIO_ROOT, "validation"),
                       prefix="validation", total=150),
}

# The fields we EXPECT on a LaneCenter in 1.3.1 -- probed defensively (hasattr),
# never assumed. The probe reports which are actually present on the build.
ADJ_CANDIDATES = ["entry_lanes", "exit_lanes", "left_neighbors", "right_neighbors"]


def shard_paths(indices, split):
    cfg = SPLITS[split]
    out = []
    for idx in indices:
        fname = f"{cfg['prefix']}.tfrecord-{idx:05d}-of-{cfg['total']:05d}"
        p = os.path.join(cfg["dir"], fname)
        if os.path.exists(p):
            out.append(p)
        else:
            print(f"WARNING: shard not found: {p}")
    return out


def declared_fields(msg):
    return list(msg.DESCRIPTOR.fields_by_name.keys())


def populated_fields(msg):
    return [f.name for f, _ in msg.ListFields()]


def neighbor_ids(repeated_msg):
    """Discover the id-bearing sub-field of a LaneNeighbor without assuming its
    name. Prefer 'feature_id'; else fall back to the first int64 sub-field."""
    ids, subfields, id_field = [], None, None
    for nb in repeated_msg:
        if subfields is None:
            subfields = declared_fields(nb)
            if "feature_id" in subfields:
                id_field = "feature_id"
            else:
                for f in nb.DESCRIPTOR.fields_by_name.values():
                    if f.type in (f.TYPE_INT64, f.TYPE_INT32, f.TYPE_UINT32,
                                  f.TYPE_UINT64):
                        id_field = f.name
                        break
        if id_field is not None:
            ids.append(int(getattr(nb, id_field)))
    return ids, subfields, id_field


def main():
    ap = argparse.ArgumentParser(description="Read-only lane-topology proto probe.")
    ap.add_argument("--split", default="validation", choices=list(SPLITS))
    ap.add_argument("--shards", default="0", help="comma-separated shard indices")
    ap.add_argument("--limit", type=int, default=3,
                    help="number of scenarios to inspect")
    ap.add_argument("--max-lanes", type=int, default=3,
                    help="lanes to dump per scenario")
    args = ap.parse_args()

    idx = [int(s) for s in args.shards.split(",") if s.strip() != ""]
    paths = shard_paths(idx, args.split)
    if not paths:
        print("ERROR: no shard found on disk.")
        return

    ds = tf.data.TFRecordDataset(paths, compression_type="")
    n_scen = 0
    for raw in ds:
        if n_scen >= args.limit:
            break
        scen = parse_waymo_scenario(raw)

        all_ids = {int(mf.id) for mf in scen.map_features}
        kinds, lanes = {}, []
        for mf in scen.map_features:
            k = mf.WhichOneof("feature_data")
            kinds[k] = kinds.get(k, 0) + 1
            if k == "lane":
                lanes.append(mf)

        print(f"\n===== scenario {scen.scenario_id} "
              f"(map_features={len(scen.map_features)}) =====")
        print(f"  feature kinds  : {kinds}")
        print(f"  #lanes={len(lanes)}  #map_ids={len(all_ids)}")

        for mf in lanes[:args.max_lanes]:
            lane = mf.lane
            print(f"  --- lane mf.id={int(mf.id)} ---")
            print(f"      declared : {declared_fields(lane)}")
            print(f"      populated: {populated_fields(lane)}")
            for name in ADJ_CANDIDATES:
                if not hasattr(lane, name):
                    print(f"      [{name}] ABSENT on this proto build")
                    continue
                val = getattr(lane, name)
                try:
                    n = len(val)
                except TypeError:
                    n = "?"
                if name in ("entry_lanes", "exit_lanes"):
                    ids = [int(x) for x in val]
                    inn = sum(1 for i in ids if i in all_ids)
                    print(f"      {name}: n={n} ids={ids[:8]} "
                          f"in_scene={inn}/{len(ids)}")
                else:
                    ids, subfields, id_field = neighbor_ids(val)
                    inn = sum(1 for i in ids if i in all_ids)
                    print(f"      {name}: n={n} subfields={subfields} "
                          f"id_field={id_field!r} ids={ids[:8]} "
                          f"in_scene={inn}/{len(ids)}")
        n_scen += 1

    print("\n[probe] done. To lock the extractor, confirm from the output above:")
    print("  (1) which of entry_lanes/exit_lanes/left_neighbors/right_neighbors "
          "are POPULATED on real lanes;")
    print("  (2) neighbors expose an id sub-field (id_field, expected "
          "'feature_id');")
    print("  (3) in_scene == total for the adjacency ids -> they share the "
          "map-feature id space the preprocessor stores as roadgraph[i]['id'].")


if __name__ == "__main__":
    main()
