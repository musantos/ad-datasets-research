"""
Empirical frame smoke test for the V2 map cache (anti-item4 diagnostic).

Runs in the CPU container after a small extraction, e.g.:
    python3 waymo_preprocessor_map.py --split validation --shards 0 --limit 50
    python3 smoke_map_frame.py /workspace/datasets/waymo/cache_val_map

WHAT IT CHECKS (measured in the cache, not trusted from the code):
The roadgraph is stored in the SDC frame, same as the agents. Re-frame BOTH
onto a target agent's frame-10 pose with ONE rotation R(-theta0), exactly as the
loader will. Then, for map points lying under/near the target, the lane tangent
should align with the target heading -- which is 0 (+x axis) in the target
frame. A systematic offset of ~2*theta0 is the item4 signature (rotation applied
with the wrong convention). We report the tangent-vs-+x angle distribution for
near-target polyline segments; it should cluster near {0, pi} (a lane runs along
or against travel), NOT be uniformly spread and NOT centered on a target-varying
2*theta0.

This does not "prove" correctness the way the code comments claim it; it is the
same std-of-angle probe that caught the silent sign error in item4.
"""
import os
import sys
import numpy as np

FS_X, FS_Y, FS_HEADING = 0, 1, 4
ANCHOR = 10


def _rotation_neg(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, s], [-s, c]], dtype=np.float64)   # R(-theta), same as loader


def main(cache_dir, near_radius=10.0, max_scenes=50):
    files = [f for f in os.listdir(cache_dir) if f.endswith(".npy")][:max_scenes]
    if not files:
        print(f"no .npy in {cache_dir}")
        return

    map_counts = []
    seg_angles = []          # tangent-vs-+x angle for near-target segments
    n_targets = 0

    for fname in files:
        data = np.load(os.path.join(cache_dir, fname), allow_pickle=True).item()
        rg = data.get("roadgraph", [])
        map_counts.append(len(rg))
        if not rg:
            continue

        for agent in data["agents"]:
            if not agent.get("is_target", False):
                continue
            fs = np.asarray(agent["full_state"], dtype=np.float64)
            m = np.asarray(agent["mask"]).astype(bool)
            if not bool(m[ANCHOR]):
                continue
            n_targets += 1

            p0 = fs[ANCHOR, [FS_X, FS_Y]].copy()
            theta0 = float(fs[ANCHOR, FS_HEADING])
            R = _rotation_neg(theta0)

            for feat in rg:
                if feat["feature_type"] != "lane":
                    continue
                poly = np.asarray(feat["polyline"], dtype=np.float64)   # SDC frame
                if poly.shape[0] < 2:
                    continue
                poly_t = (poly - p0) @ R.T                              # target frame
                # segments whose midpoint is near the target
                mid = 0.5 * (poly_t[:-1] + poly_t[1:])
                d = np.hypot(mid[:, 0], mid[:, 1])
                sel = d < near_radius
                if not np.any(sel):
                    continue
                seg = poly_t[1:] - poly_t[:-1]
                seg = seg[sel]
                ang = np.arctan2(seg[:, 1], seg[:, 0])                  # tangent angle
                # fold to [0, pi): a lane along +x or -x are both "aligned"
                ang_folded = np.mod(ang, np.pi)
                seg_angles.extend(ang_folded.tolist())

    map_counts = np.array(map_counts)
    print(f"scenes={len(files)}  targets_probed={n_targets}")
    print(f"roadgraph per scene: min={map_counts.min()} "
          f"mean={map_counts.mean():.1f} max={map_counts.max()}")
    if map_counts.min() == 0:
        print("  NOTE: some scenes have empty roadgraph (rare but possible).")

    if not seg_angles:
        print("no near-target lane segments found -- widen near_radius or "
              "extract more scenes.")
        return

    a = np.array(seg_angles)
    # distance to nearest of {0, pi} on the folded circle
    dist0 = np.minimum(a, np.pi - a)
    print(f"\nnear-target lane tangent alignment (folded to [0,pi)):")
    print(f"  n_segments={a.size}")
    print(f"  |angle to nearest of 0/pi|: "
          f"median={np.median(dist0):.3f} rad  mean={dist0.mean():.3f} rad")
    frac_aligned = float(np.mean(dist0 < np.radians(30)))
    print(f"  fraction within 30deg of a lane direction: {frac_aligned:.2f}")
    print("\nEXPECT: median small (lanes near the target tend to run along its")
    print("travel axis) and NO dependence on the target's own theta0. If instead")
    print("the offset tracks 2*theta0 across targets, the map rotation used the")
    print("wrong convention (item4 signature) -- re-check extract_roadgraph.")


if __name__ == "__main__":
    cache = sys.argv[1] if len(sys.argv) > 1 else "/workspace/datasets/waymo/cache_val_map"
    main(cache)
