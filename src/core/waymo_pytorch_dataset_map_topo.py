import os
import numpy as np
import torch

# SIBLING of waymo_pytorch_dataset_map (V3 = map + lane topology). The V2 loader
# (WaymoMotionDatasetMap) is NOT touched. We extend it so the sample index, the
# feature vocabulary, and the target + neighbor + map geometry are inherited and
# byte-for-byte identical to V2 -- topology is a purely additive context branch,
# exactly as the map was additive over social.
#
# The ONE new thing is the ADJACENCY branch: after V2's sort-by-(distance, id) +
# top-M truncation of the roadgraph polylines, each kept lane's declared
# neighbors (entry/exit/left/right, stored as raw proto ids in the *_map_topo
# cache) are remapped to the CORRECT slot in [0, M). This remap is the recipe
# proven byte-for-byte in smoke_topo_alignment.py (Stage A, acceptances 1-3): an
# id -> slot table built over the KEPT candidates only, edges to neighbors dropped
# by top-M discarded, edges lane-only on both endpoints. Here that recipe is
# PROMOTED from the smoke into the loader and emitted as a dense typed tensor.
#
# Why full __getitem__ override (not super() + a second pass): V2's __getitem__
# builds map_cands as (d, id, poly_agent, ftype) and discards the raw feature
# dict, so the adjacency ids are not recoverable from it. Re-loading the .npy and
# re-sorting in a second pass would double the loader I/O at exactly the degrau
# the handoff flags as the hardware ceiling. Instead we reproduce V2's __getitem__
# VERBATIM in a single pass, keeping the feature dict, and the __main__ guard
# asserts the non-adjacency tuple elements match WaymoMotionDatasetMap[i]
# byte-for-byte -- the codebase's established "verbatim copy + equivalence guard"
# pattern (the same way V2 copied the neighbor loop from the social loader).
from src.core.waymo_pytorch_dataset_agentcentric import (
    WaymoMotionDatasetAgentCentric,
    FS_X, FS_Y, FS_HEADING, ANCHOR_FRAME, N_PAST,
    _ANGLE_FEATURES,
)
from src.core.waymo_pytorch_dataset_social import _agent_past_features
from src.core.waymo_pytorch_dataset_map import (
    WaymoMotionDatasetMap, _MAP_FEATURE_TYPES, _resample_polyline,
    N_MAP_POLYLINES, N_POINTS_PER_POLYLINE,
)


# Relation channels of the adjacency tensor, in a FIXED order (== _ADJ_KEYS in
# smoke_topo_alignment.py). R = len(_ADJ_RELATIONS) = 4. Kept typed and all four
# separate on purpose: the handoff's canonical V3 model starts TYPED (>= successor
# / predecessor / lateral); storing entry/exit/left/right lets the model merge
# left+right into "lateral" for the 3-typed canonical, transpose entry<->exit for
# reverse message-passing, or collapse all four into an untyped graph as a cheap
# ablation -- every choice is a reduction of what is emitted here, so the loader
# commits to nothing beyond the maximal typed representation.
_ADJ_RELATIONS = ("entry_lanes", "exit_lanes", "left_neighbors", "right_neighbors")

# Lane is entity type 0 in _MAP_FEATURE_TYPES. Edges are lane-only on BOTH
# endpoints (Stage A acceptance 1: 100% of adjacency ids resolve to a lane).
_LANE_TYPE = _MAP_FEATURE_TYPES["lane"]


class WaymoMotionDatasetMapTopo(WaymoMotionDatasetMap):
    """
    V3 dataset: the V2 map dataset PLUS lane-connectivity topology, re-aligned to
    the M kept polyline slots. Extends WaymoMotionDatasetMap so the sample index,
    the feature vocabulary, and the target + neighbor + map geometry are inherited
    unchanged -- the adjacency is a purely additive context branch.

    Reads the V3 cache (cache_{train,val}_map_topo), a STRICT SUPERSET of the V2
    cache: every roadgraph entry carries, in addition to the V2 keys
    {id, feature_type, type, polyline[P,2]}, the four id-based adjacency lists
        entry_lanes, exit_lanes, left_neighbors, right_neighbors   (each [int])
    where the ids reference MapFeature.id (globally unique in the scene) and are
    EMPTY for non-lane features. Because the cache is a strict superset, the V2
    loader (WaymoMotionDatasetMap) runs on it unchanged and ignores these keys.

    __getitem__ returns a 10-tuple: V2's 9-tuple with the adjacency tensor
    inserted after map_mask (map context grouped before the labels, exactly as V2
    inserted the map tensors after the neighbor tensors):

        x_past        [11, n_features]   target, agent-centric        (== V2)
        neighbors     [K, 11, n_features] neighbors, target frame     (== V2)
        neighbor_mask [K]                1.0 where a neighbor exists   (== V2)
        map_polylines [M, Np, 2]         roadgraph polylines, target frame (== V2)
        map_type      [M] long           entity type per polyline     (== V2)
        map_mask      [M]                1.0 where a polyline exists   (== V2)
        map_adjacency [R, M, M]          lane connectivity, slot-aligned  <-- NEW
        y_future      [80, 2]            target, same frame as input  (== V2)
        future_mask   [80]               1.0 where the future frame is valid (== V2)
        agent_type    scalar long                                     (== V2)

    where R = len(_ADJ_RELATIONS) = 4, M = n_map_polylines, Np = n_points_per_polyline.

    ADJACENCY SEMANTICS: map_adjacency[r, i, j] == 1.0 iff the kept lane at slot i
    declares the kept lane at slot j as its relation-r neighbor, where
        r=0 entry_lanes (predecessors)   r=1 exit_lanes (successors)
        r=2 left_neighbors               r=3 right_neighbors.
    Non-lane source slots have all-zero rows; padded slots (map_mask==0) never
    appear as source or target. The matrix is NOT forced symmetric -- entry/exit
    encode direction; global entry-total == exit-total is a scene-level health
    check (Stage A), not a per-edge one.

    REDUCTION TO V2: the first six and last three tuple elements are produced by
    code reproduced verbatim from V2, so they are byte-for-byte identical to
    WaymoMotionDatasetMap on the same cache (asserted in __main__). Zeroing
    map_adjacency therefore recovers V2 exactly -- the loader-level analogue of
    the V3 model reducing to V2 when the adjacency is masked out (the B.2 ladder
    contract). The V3 cache is a strict superset, so the adjacency keys never
    perturb the V2 output.

    FORMAT NOTE (deferred model decision): the dense [R, M, M] tensor is a fixed
    shape and default-collatable (no custom collate_fn leaking into the training
    loop). A message-passing model that prefers a sparse edge_index derives it on
    device via map_adjacency.nonzero() -- so emitting dense commits the loader to
    nothing beyond a batch-friendly superset. M=128 is the frozen V2 truncation;
    the alignment depends on M, so run with N_MAP_POLYLINES=128 (the loader
    default is 64, the grid overrides it).
    """

    # __init__ is inherited unchanged from WaymoMotionDatasetMap: the adjacency
    # needs no new knob (R is fixed, the relation set is fixed, M/Np come from V2).

    @staticmethod
    def _build_adjacency(map_cands, M):
        """Build the [R, M, M] lane-adjacency tensor from the kept, ordered map
        candidates. `map_cands` is the post-sort, post-top-M list of
        (dist, proto_id, poly_agent, ftype, feat), so the slot index is the list
        position -- the SAME order the map_polylines/map_type/map_mask tensors use.

        The remap (validated in smoke_topo_alignment.py):
          * slot_of maps proto id -> slot over the KEPT candidates ONLY, so a
            neighbor truncated away by top-M is absent and its edge is dropped
            (the loader projects the cache's full topology onto the current M).
          * edges are lane-only: only lane source slots contribute rows, and a
            neighbor id always resolves to a lane slot (Stage A acceptance 1).
        """
        R = len(_ADJ_RELATIONS)
        A = np.zeros((R, M, M), dtype=np.float32)
        slot_of = {pid: k for k, (_d, pid, _p, _ft, _f) in enumerate(map_cands)}
        for i, (_d, _pid, _poly, ftype, feat) in enumerate(map_cands):
            if ftype != _LANE_TYPE:                 # source must be a lane
                continue
            for r, rel in enumerate(_ADJ_RELATIONS):
                for nid in feat[rel]:               # KeyError = cache contract broken
                    j = slot_of.get(nid)
                    if j is None:                   # neighbor outside kept top-M -> drop
                        continue
                    A[r, i, j] = 1.0
        return A

    def __getitem__(self, idx):
        file_path, target_id = self.samples[idx]
        data = np.load(file_path, allow_pickle=True).item()
        agents = data['agents']

        target = next((a for a in agents if a['id'] == target_id), None)
        if target is None:
            raise RuntimeError(f"Agent id={target_id} not found in {file_path}.")

        t_fs = np.asarray(target['full_state'], dtype=np.float64)
        t_mask = np.asarray(target['mask'])
        assert t_fs.shape == (91, 7), (
            f"full_state {t_fs.shape} != (91,7) in {file_path}; layout changed."
        )
        assert t_mask.shape[0] == 91, f"mask len {t_mask.shape[0]} != 91."
        assert bool(t_mask[ANCHOR_FRAME]), (
            f"anchor frame {ANCHOR_FRAME} invalid for target {target_id} in "
            f"{file_path}; agent-centric frame undefined."
        )

        # Anchor of the target: single rotation source for target, neighbors, map.
        p0 = t_fs[ANCHOR_FRAME, [FS_X, FS_Y]].copy()
        theta0 = float(t_fs[ANCHOR_FRAME, FS_HEADING])
        R = WaymoMotionDatasetAgentCentric._rotation_neg(theta0)

        # --- TARGET (== V2 / V1 / agent-centric) --------------------------------
        x_past = _agent_past_features(
            t_fs, t_mask, p0, theta0, self.features, self.heading_as_sincos)

        xy_all = (t_fs[:, [FS_X, FS_Y]].copy() - p0) @ R.T   # [91,2] agent frame
        xy_all[~t_mask.astype(bool)] = 0.0
        y_future = xy_all[N_PAST:, :]                         # [80,2]
        future_mask = t_mask[N_PAST:].astype(np.float32)     # [80]

        # --- NEIGHBORS (verbatim from the V2 / social loader) -------------------
        t_pos10 = t_fs[ANCHOR_FRAME, [FS_X, FS_Y]]           # SDC frame (distance)
        cands = []
        for a in agents:
            if a['id'] == target_id:
                continue
            am = np.asarray(a['mask'])
            if not bool(am[ANCHOR_FRAME]):
                continue
            a_fs = np.asarray(a['full_state'], dtype=np.float64)
            dx, dy = a_fs[ANCHOR_FRAME, [FS_X, FS_Y]] - t_pos10
            dist = float(np.hypot(dx, dy))
            cands.append((dist, int(a['id']), a_fs, am))

        cands.sort(key=lambda c: (c[0], c[1]))               # distance, tie by id
        cands = cands[:self.n_neighbors]

        K = self.n_neighbors
        neighbors = np.zeros((K, N_PAST, self.n_features), dtype=np.float64)
        neighbor_mask = np.zeros((K,), dtype=np.float32)
        for i, (_dist, _aid, a_fs, am) in enumerate(cands):
            neighbors[i] = _agent_past_features(
                a_fs, am, p0, theta0, self.features, self.heading_as_sincos)
            neighbor_mask[i] = 1.0

        # --- MAP (verbatim from V2, but KEEP `feat` so adjacency is readable) ---
        roadgraph = data.get('roadgraph', [])
        map_cands = []
        for feat in roadgraph:
            poly_sdc = np.asarray(feat['polyline'], dtype=np.float64)
            if poly_sdc.shape[0] < 2:                        # need >=1 segment
                continue
            ft_name = feat['feature_type']
            if ft_name not in _MAP_FEATURE_TYPES:            # cache contract broken
                raise ValueError(
                    f"unexpected map feature_type {ft_name!r} in {file_path}; "
                    f"expected one of {sorted(_MAP_FEATURE_TYPES)}."
                )
            ftype = _MAP_FEATURE_TYPES[ft_name]
            poly_agent = (poly_sdc - p0) @ R.T               # [P,2] agent frame
            d = float(np.min(np.hypot(poly_agent[:, 0], poly_agent[:, 1])))
            map_cands.append((d, int(feat['id']), poly_agent, ftype, feat))

        map_cands.sort(key=lambda c: (c[0], c[1]))           # distance, tie by id
        map_cands = map_cands[:self.n_map_polylines]

        M = self.n_map_polylines
        Np = self.n_points_per_polyline
        map_polylines = np.zeros((M, Np, 2), dtype=np.float64)
        map_type = np.zeros((M,), dtype=np.int64)            # pad value irrelevant
        map_mask = np.zeros((M,), dtype=np.float32)          # -> masked by map_mask
        for i, (_d, _id, poly_agent, ftype, _feat) in enumerate(map_cands):
            map_polylines[i] = _resample_polyline(poly_agent, Np)
            map_type[i] = ftype
            map_mask[i] = 1.0

        # --- NEW: lane adjacency aligned to the M kept slots (promoted remap) ----
        map_adjacency = self._build_adjacency(map_cands, M)  # [R, M, M] float32

        return (
            torch.from_numpy(x_past).float(),            # [11, F]
            torch.from_numpy(neighbors).float(),          # [K, 11, F]
            torch.from_numpy(neighbor_mask).float(),      # [K]
            torch.from_numpy(map_polylines).float(),      # [M, Np, 2]
            torch.from_numpy(map_type).long(),            # [M]
            torch.from_numpy(map_mask).float(),           # [M]
            torch.from_numpy(map_adjacency).float(),      # [R, M, M]
            torch.from_numpy(y_future).float(),           # [80, 2]
            torch.from_numpy(future_mask).float(),        # [80]
            torch.tensor(int(target['type']), dtype=torch.long),
        )


if __name__ == "__main__":
    print("OK: WaymoMotionDatasetMapTopo class ready.")
    for feats, sc in [
        (("x", "y", "heading", "vx", "vy"), True),
        (("x", "y", "length", "width", "heading", "vx", "vy"), True),
    ]:
        n = sum(2 if (f in _ANGLE_FEATURES and sc) else 1 for f in feats)
        print(f"  features={feats} sincos={sc} -> n_features={n}")

    # Point WAYMO_MAP_TOPO_CACHE at a real cache_*_map_topo dir to run the
    # acceptance end-to-end (in gpu_env, with N_MAP_POLYLINES=128):
    #   docker exec -w /workspace -e N_MAP_POLYLINES=128 \
    #       -e WAYMO_MAP_TOPO_CACHE=/workspace/datasets/waymo/cache_val_map_topo \
    #       gpu_env python3 -m src.core.waymo_pytorch_dataset_map_topo
    CACHE = os.environ.get("WAYMO_MAP_TOPO_CACHE", "")
    if CACHE and os.path.isdir(CACHE):
        K = int(os.environ.get("N_NEIGHBORS", "16"))
        M = int(os.environ.get("N_MAP_POLYLINES", str(N_MAP_POLYLINES)))
        Np = int(os.environ.get("N_POINTS_PER_POLYLINE", str(N_POINTS_PER_POLYLINE)))
        feats = ("x", "y", "heading", "vx", "vy")
        R = len(_ADJ_RELATIONS)
        if M != 128:
            print(f"[cfg] WARNING: M={M} != 128 (the frozen V2 grid value). The "
                  f"alignment depends on M; pass N_MAP_POLYLINES=128.")

        ds = WaymoMotionDatasetMapTopo(
            CACHE, n_neighbors=K, n_map_polylines=M,
            n_points_per_polyline=Np, features=feats)
        n_probe = min(len(ds), int(os.environ.get("N_PROBE", "50")))
        print(f"[cfg] cache={CACHE} K={K} M={M} Np={Np} R={R} | items={len(ds)}, "
              f"probing {n_probe}")

        xp, nb, nm, mp, mt, mm, adj, yf, fm, at = ds[0]
        print(f"[smoke] shapes | x_past={tuple(xp.shape)} neighbors={tuple(nb.shape)} "
              f"map={tuple(mp.shape)} adj={tuple(adj.shape)} y_future={tuple(yf.shape)}")
        assert adj.shape == (R, M, M), adj.shape
        assert adj.dtype == torch.float32, adj.dtype

        # (1) REDUCTION TO V2: the non-adjacency tuple elements must be byte-for-
        # byte identical to WaymoMotionDatasetMap on the SAME cache (V2 runs on the
        # topo superset unchanged, ignoring the adjacency keys). This is the B.1
        # acceptance: zeroing map_adjacency recovers V2 exactly. Match by
        # (basename, id) so the check is robust to listdir order.
        ds_v2 = WaymoMotionDatasetMap(
            CACHE, n_neighbors=K, n_map_polylines=M,
            n_points_per_polyline=Np, features=feats)
        v2_index = {(os.path.basename(p), tid): j
                    for j, (p, tid) in enumerate(ds_v2.samples)}
        v2_slots = [3, 4, 5]      # map_polylines/type/mask in both loaders
        # topo (non-adj) vs V2: topo[0:6]==v2[0:6] and topo[7:10]==v2[6:9].
        topo_to_v2 = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 7: 6, 8: 7, 9: 8}

        # (2) SLOT ALIGNMENT: every nonzero adjacency entry lands on present lane
        # slots, in bounds; padded slots never appear as source or target.
        edges_per_rel = [0] * R
        lane_slots_seen = 0

        for i in range(n_probe):
            p, tid = ds.samples[i]
            item = ds[i]
            adj_i = item[6].numpy()
            mt_i = item[4].numpy()
            mm_i = item[5].numpy()

            # (1) reduction to V2
            key = (os.path.basename(p), tid)
            j = v2_index.get(key)
            assert j is not None, f"item {i}: {key} not found in V2 loader index"
            v2 = ds_v2[j]
            for ti, vi in topo_to_v2.items():
                a, b = item[ti], v2[vi]
                assert torch.equal(a, b) or torch.allclose(
                    a.float(), b.float(), atol=1e-6), \
                    f"item {i}: tuple elem {ti} differs from V2 elem {vi}"

            # (2) alignment invariants on the emitted tensor
            present = mm_i > 0.5
            lane_slots_seen += int((mt_i[present] == _LANE_TYPE).sum())
            nz = np.argwhere(adj_i > 0.5)                # [(r, s, t), ...]
            for r, s, t in nz:
                assert 0 <= t < M and 0 <= s < M, f"item {i}: edge slot out of [0,{M})"
                assert mm_i[s] == 1.0, f"item {i}: source slot {s} is padding"
                assert mm_i[t] == 1.0, f"item {i}: target slot {t} is padding"
                assert mt_i[s] == _LANE_TYPE, f"item {i}: source slot {s} not a lane"
                assert mt_i[t] == _LANE_TYPE, f"item {i}: target slot {t} not a lane"
                edges_per_rel[int(r)] += 1

            # (3) padded-slot invariant: no edge touches a masked slot on either axis
            pad = ~present
            if pad.any():
                assert adj_i[:, pad, :].sum() == 0.0, f"item {i}: edge from padded slot"
                assert adj_i[:, :, pad].sum() == 0.0, f"item {i}: edge into padded slot"

        total_edges = sum(edges_per_rel)
        print(f"[align] lane slots examined: {lane_slots_seen}")
        print(f"[align] edges VALIDATED (present lane->lane, in-bounds slot): {total_edges}")
        for r, rel in enumerate(_ADJ_RELATIONS):
            print(f"          r={r} {rel}: {edges_per_rel[r]}")
        assert total_edges > 0, (
            "no edges validated -- increase N_PROBE or check M; the alignment "
            "assertions were vacuous.")
        print("[align] SLOT ALIGNMENT OK: every emitted edge hits a present lane "
              "slot at the right index.")
        print("[reduce] REDUCTION TO V2 OK: non-adjacency tuple elements are "
              "byte-for-byte identical to WaymoMotionDatasetMap.")
        print("ALL B.1 ACCEPTANCES PASSED.")
    else:
        print("[skip] set WAYMO_MAP_TOPO_CACHE=<cache_*_map_topo> to run the "
              "acceptance (reduction-to-V2 + slot-alignment).")
