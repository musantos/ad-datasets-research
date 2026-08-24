import os
import numpy as np
import torch

# SIBLING of waymo_pytorch_dataset_social (V2 = map-aware). Neither the V0
# (agent-centric) nor the V1 (social) loader is touched. The target and neighbor
# branches are NOT reimplemented here: we import `_agent_past_features` and
# `_rotation_neg` from the existing modules so the frame convention has ONE
# source of truth. The neighbor-selection loop is replicated verbatim from the
# social loader (orchestration, not geometry); the __main__ equivalence guard
# asserts the target+neighbor outputs match WaymoMotionDatasetSocial byte-for-
# byte on the same cache, so any accidental drift in the copy fails loudly.
#
# The ONLY new thing is the MAP branch: the roadgraph polylines (stored in the
# SDC frame by waymo_preprocessor_map) are re-framed onto the target's frame-10
# pose with the SAME (p0, theta0) and the SAME _rotation_neg used for the
# neighbors -> a map point is geometrically just a position, so it undergoes the
# exact transform the target's xy undergoes. Zero cross-frame drift by
# construction (this is the item-4 lesson institutionalized: one rotation, not
# two independent ones that must "happen" to agree). The map has no heading and
# no velocity -- orientation is implicit in point order, and rotating the points
# rotates the tangents consistently (confirmed empirically by smoke_map_frame:
# median tangent-vs-+x = 0.048 rad, no 2*theta0 signature).
from src.core.waymo_pytorch_dataset_agentcentric import (
    WaymoMotionDatasetAgentCentric,
    FS_X, FS_Y, FS_HEADING, ANCHOR_FRAME, N_PAST,
    _SCALAR_FEATURES, _ANGLE_FEATURES,
)
from src.core.waymo_pytorch_dataset_social import (
    WaymoMotionDatasetSocial, _agent_past_features,
)


# --- a-priori knobs (fix before the grid, calibrate with the __main__ smoke) --
# Mirror the discipline used for n_neighbors=16 (calibrated empirically on the
# dataset smoke, then frozen). These two are the map representation and live in
# the LOADER on purpose (re-tunable without re-extracting the cache, which keeps
# the FULL polyline):
#   * N_MAP_POLYLINES: top-K nearest polylines kept per target (by min distance
#     to the target, tie-break by proto id -> deterministic, seed-independent).
#   * N_POINTS_PER_POLYLINE: each kept polyline is resampled to exactly this many
#     points, so the map subgraph sees fixed-shape polylines with NO intra-
#     polyline padding/mask (structurally identical to the 11-frame agent
#     polyline). Masking is polyline-level only (presence), like the neighbors.
N_MAP_POLYLINES = 64
N_POINTS_PER_POLYLINE = 20

# Map feature_type -> categorical int, exposed in the tuple as map_type. Kept at
# the ENTITY level (lane / road_edge / road_line), NOT the raw per-layer proto
# enum: the raw enum's meaning changes per layer (value k means different things
# under lane vs road_line), so a single embedding table indexed by it would
# collapse distinct semantics -- a silent representation bug. The 3-way entity
# distinction (drivable lane vs boundary vs marking) is the first-order signal
# the map hypothesis rests on and matches what VectorNet-family models use. The
# model consumes this behind a flag (off by default -> V2 = pure geometry); when
# on, it embeds the 3 categories. Finer sub-typing (marking style, lane class)
# is future work, same shelf as topology/V3.
_MAP_FEATURE_TYPES = {"lane": 0, "road_edge": 1, "road_line": 2}


def _resample_polyline(poly, n):
    """Resample an [P,2] polyline to exactly [n,2] by nearest-index selection.

    Deterministic and coordinate-preserving (no synthetic/interpolated points):
    picks n indices evenly spaced over [0, P-1] and rounds. For P >= n this is a
    subsample; for P < n some points repeat (harmless for the permutation-
    invariant maxpool -- a repeated point yields a zero-length segment, no NaN).
    P >= 2 is guaranteed by the caller (shorter polylines are skipped).
    """
    P = poly.shape[0]
    if P == n:
        return poly
    idx = np.round(np.linspace(0, P - 1, n)).astype(int)
    return poly[idx]


class WaymoMotionDatasetMap(WaymoMotionDatasetSocial):
    """
    V2 dataset: the social dataset PLUS the scene roadgraph re-framed onto the
    target agent's frame. Extends WaymoMotionDatasetSocial so the sample index
    (one (scene, target) per item), the feature vocabulary, and the target +
    neighbor geometry are inherited unchanged -- the map is a purely additive
    context branch, exactly as social was additive over V0.

    Reads the V2 cache (cache_{train,val}_map), whose .npy carries the extra key
    data['roadgraph'] = list of {id, feature_type, type, polyline[P,2]} in the
    SDC frame. (The V0/V1 caches lack this key; point this loader at a *_map
    cache.) The agent branch of the *_map cache is byte-for-byte the V0/V1 agent
    branch, so scene parity with V1 holds by construction.

    __getitem__ returns a 9-tuple: the social 6-tuple with three map tensors
    inserted after the neighbor tensors (context grouped before labels):

        x_past        [11, n_features]      target, agent-centric      (== V1)
        neighbors     [K, 11, n_features]   neighbors, target frame    (== V1)
        neighbor_mask [K]                   1.0 where a neighbor exists (== V1)
        map_polylines [M, Np, 2]            roadgraph polylines, target frame
        map_type      [M] long             entity type per polyline (see below)
        map_mask      [M]                   1.0 where a polyline exists (0 = pad)
        y_future      [80, 2]               target, same frame as input (== V1)
        future_mask   [80]                  1.0 where the future frame is valid
        agent_type    scalar long                                      (== V1)

    where M = n_map_polylines and Np = n_points_per_polyline.

    REDUCTION TO V1: a scene with an empty roadgraph (rare but possible) yields
    map_mask all-zero and map_polylines all-zero. The V2 model's map guard
    (mirroring the zero-neighbor guard of V1) then zeroes the map context, so V2
    collapses to V1 on map-less scenes. The map branch never corrupts the social
    output -- the two are independent context sources fused downstream.

    Map representation note: geometry (xy) AND the entity type (map_type) are
    delivered, but V2 stays a single isolated variable by having the MODEL gate
    the type behind a flag (off -> pure geometry, the V2 hypothesis "does road
    GEOMETRY close MR?"; on -> V2+type ablation "does lane semantics add more?").
    Delivering the type here (near-free, already in the cache contract) means the
    ablation is a CLI flag, not a second loader/sibling. See _MAP_FEATURE_TYPES
    for why the entity level is used and not the raw per-layer proto enum.
    """

    def __init__(
        self,
        cache_dir,
        n_neighbors=16,
        n_map_polylines=N_MAP_POLYLINES,
        n_points_per_polyline=N_POINTS_PER_POLYLINE,
        features=("x", "y", "heading", "vx", "vy"),
        heading_as_sincos=True,
    ):
        # Inherit sample indexing, feature validation, n_features, neighbor setup.
        super().__init__(
            cache_dir,
            n_neighbors=n_neighbors,
            features=features,
            heading_as_sincos=heading_as_sincos,
        )
        self.n_map_polylines = int(n_map_polylines)
        self.n_points_per_polyline = int(n_points_per_polyline)

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

        # Anchor of the target: defines the agent-centric frame of EVERYTHING
        # (target, neighbors, AND the map polylines) -> single rotation source.
        p0 = t_fs[ANCHOR_FRAME, [FS_X, FS_Y]].copy()
        theta0 = float(t_fs[ANCHOR_FRAME, FS_HEADING])
        R = WaymoMotionDatasetAgentCentric._rotation_neg(theta0)

        # --- TARGET (== agent-centric / V1) -------------------------------------
        x_past = _agent_past_features(
            t_fs, t_mask, p0, theta0, self.features, self.heading_as_sincos)

        xy_all = (t_fs[:, [FS_X, FS_Y]].copy() - p0) @ R.T   # [91,2] agent frame
        xy_all[~t_mask.astype(bool)] = 0.0
        y_future = xy_all[N_PAST:, :]                         # [80,2]
        future_mask = t_mask[N_PAST:].astype(np.float32)     # [80]

        # --- NEIGHBORS (verbatim from the social loader) ------------------------
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

        # --- MAP: roadgraph polylines in the SAME target frame ------------------
        # Each polyline is re-framed with the SAME (p0, R) as the neighbors: a
        # map point is geometrically a position, so this is the neighbor xy
        # transform with no heading/velocity terms. Rank by min distance from the
        # target (at the origin in the agent frame), tie-break by proto id ->
        # deterministic. Then resample each kept polyline to a fixed point count.
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
            map_cands.append((d, int(feat['id']), poly_agent, ftype))

        map_cands.sort(key=lambda c: (c[0], c[1]))           # distance, tie by id
        map_cands = map_cands[:self.n_map_polylines]

        M = self.n_map_polylines
        Np = self.n_points_per_polyline
        map_polylines = np.zeros((M, Np, 2), dtype=np.float64)
        map_type = np.zeros((M,), dtype=np.int64)            # pad value irrelevant
        map_mask = np.zeros((M,), dtype=np.float32)          # -> masked by map_mask
        for i, (_d, _id, poly_agent, ftype) in enumerate(map_cands):
            map_polylines[i] = _resample_polyline(poly_agent, Np)
            map_type[i] = ftype
            map_mask[i] = 1.0

        return (
            torch.from_numpy(x_past).float(),           # [11, F]
            torch.from_numpy(neighbors).float(),         # [K, 11, F]
            torch.from_numpy(neighbor_mask).float(),     # [K]
            torch.from_numpy(map_polylines).float(),     # [M, Np, 2]
            torch.from_numpy(map_type).long(),           # [M]
            torch.from_numpy(map_mask).float(),          # [M]
            torch.from_numpy(y_future).float(),          # [80, 2]
            torch.from_numpy(future_mask).float(),       # [80]
            torch.tensor(int(target['type']), dtype=torch.long),
        )


if __name__ == "__main__":
    print("OK: WaymoMotionDatasetMap class ready.")
    for feats, sc in [
        (("x", "y", "heading", "vx", "vy"), True),
        (("x", "y", "length", "width", "heading", "vx", "vy"), True),
    ]:
        n = sum(2 if (f in _ANGLE_FEATURES and sc) else 1 for f in feats)
        print(f"  features={feats} sincos={sc} -> n_features={n}")

    # Point WAYMO_MAP_CACHE at a real cache_*_map dir to exercise end-to-end.
    CACHE = os.environ.get("WAYMO_MAP_CACHE", "")
    if CACHE and os.path.isdir(CACHE):
        K = int(os.environ.get("N_NEIGHBORS", "16"))
        M = int(os.environ.get("N_MAP_POLYLINES", str(N_MAP_POLYLINES)))
        Np = int(os.environ.get("N_POINTS_PER_POLYLINE", str(N_POINTS_PER_POLYLINE)))
        feats = ("x", "y", "heading", "vx", "vy")

        ds = WaymoMotionDatasetMap(
            CACHE, n_neighbors=K, n_map_polylines=M,
            n_points_per_polyline=Np, features=feats)
        xp, nb, nm, mp, mt, mm, yf, fm, at = ds[0]
        print(f"[smoke] n_features={ds.n_features} | x_past={tuple(xp.shape)} "
              f"neighbors={tuple(nb.shape)} nmask={tuple(nm.shape)} "
              f"map={tuple(mp.shape)} mtype={tuple(mt.shape)} mmask={tuple(mm.shape)} "
              f"(map valid={int(mm.sum())}/{M}) y_future={tuple(yf.shape)} "
              f"mask={tuple(fm.shape)} type={int(at)}")
        assert xp.shape == (11, ds.n_features)
        assert nb.shape == (K, 11, ds.n_features)
        assert nm.shape == (K,)
        assert mp.shape == (M, Np, 2)
        assert mt.shape == (M,) and mt.dtype == torch.long
        assert mm.shape == (M,)
        assert yf.shape == (80, 2)

        # map_type is valid (0..2) exactly where a polyline is present, and the
        # padded slots carry the (irrelevant) default 0 -- the model masks them.
        present = mm.bool()
        assert (mt[present] >= 0).all() and (mt[present] <= 2).all(), \
            "map_type out of the {lane,road_edge,road_line} range on a real polyline"
        type_hist = {t: int((mt[present] == t).sum()) for t in (0, 1, 2)}
        print(f"[smoke] map_type histogram (0=lane,1=road_edge,2=road_line): {type_hist}")

        # EQUIVALENCE OF TARGET + SOCIAL: x_past / neighbors / masks / y_future
        # must match WaymoMotionDatasetSocial byte-for-byte on the SAME cache
        # (the social loader ignores the extra 'roadgraph' key) -> proves the
        # verbatim neighbor-loop copy did not drift and the map branch is purely
        # additive.
        social = WaymoMotionDatasetSocial(CACHE, n_neighbors=K, features=feats)
        want = ds.samples[0]
        j = social.samples.index(want)
        sxp, snb, snm, syf, sfm, sat = social[j]
        assert torch.allclose(xp, sxp, atol=1e-6), "x_past drifted from social!"
        assert torch.allclose(nb, snb, atol=1e-6), "neighbors drifted from social!"
        assert torch.allclose(nm, snm, atol=1e-6), "neighbor_mask drifted!"
        assert torch.allclose(yf, syf, atol=1e-6), "y_future drifted from social!"
        assert torch.allclose(fm, sfm, atol=1e-6), "future_mask drifted!"
        assert int(at) == int(sat), "agent_type drifted!"
        print("[smoke] target+social equivalence vs WaymoMotionDatasetSocial OK.")

        # MAP FRAME SANITY (mirrors smoke_map_frame): the nearest kept polyline
        # should have a point close to the origin (the target sits at (0,0) in
        # the agent frame), i.e. the map really shares the target frame.
        if int(mm.sum()) > 0:
            nearest = mp[0][mm[0].bool() if mm.ndim > 1 else slice(None)]
            d0 = float(np.min(np.hypot(mp[0, :, 0].numpy(), mp[0, :, 1].numpy())))
            print(f"[smoke] nearest kept polyline min-dist to target = {d0:.2f} m "
                  f"(should be small; large -> map not co-framed).")

        # CALIBRATION: distribution of valid polylines/points across a sample of
        # items, to fix N_MAP_POLYLINES / N_POINTS_PER_POLYLINE a priori (same
        # discipline as n_neighbors=16). Reports how often the top-K cap is hit.
        import numpy as _np
        n_probe = min(len(ds), int(os.environ.get("N_PROBE", "200")))
        valid_counts = []
        raw_polyline_lengths = []
        capped = 0
        for i in range(n_probe):
            path, tid = ds.samples[i]
            d = _np.load(path, allow_pickle=True).item()
            rg = [f for f in d.get('roadgraph', [])
                  if _np.asarray(f['polyline']).shape[0] >= 2]
            valid_counts.append(len(rg))
            if len(rg) >= M:
                capped += 1
            for f in rg:
                raw_polyline_lengths.append(int(_np.asarray(f['polyline']).shape[0]))
        vc = _np.array(valid_counts)
        pl = _np.array(raw_polyline_lengths) if raw_polyline_lengths else _np.array([0])
        print(f"\n[calib] over {n_probe} items:")
        print(f"  polylines/scene (>=2 pts): min={vc.min()} mean={vc.mean():.1f} "
              f"median={int(_np.median(vc))} max={vc.max()}")
        print(f"  items hitting the top-K cap (M={M}): {capped}/{n_probe} "
              f"({100.0*capped/n_probe:.0f}%)")
        print(f"  raw polyline length (pts): min={pl.min()} mean={pl.mean():.1f} "
              f"median={int(_np.median(pl))} p95={int(_np.percentile(pl,95))} "
              f"max={pl.max()}  (resampling to Np={Np})")
        print("  -> if the cap is hit ~always, M may be too small (missing lanes);")
        print("     if raw lengths >> Np, Np subsampling may drop lane detail.")
