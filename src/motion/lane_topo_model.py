import torch
import torch.nn as nn

# SIBLING of vectorized_social_map_model (V3 = V2 + LANE TOPOLOGY). SUBCLASSES the
# V2 predictor so the ENTIRE V2 machine (target/neighbor/map subgraph encoders,
# both cross-attentions, the fusion LayerNorm, the heads, the standardization
# buffers) is inherited byte-for-byte. The ONLY architectural addition is a typed
# relational message-passing layer that refines the map polyline embeddings
# [B,M,H] along lane connectivity, inserted BEFORE the target->map cross-attention
# -> any metric delta over V2 is attributable to lane topology alone (the B.2
# ladder variable).
#
# build_vectors is imported only for the optional check_frame path (mirrors V2).
from src.motion.vectorized_model import build_vectors
from src.motion.vectorized_social_map_model import (
    VectorizedSocialMapTrajectoryPredictor,
)

# The loader (WaymoMotionDatasetMapTopo) emits R=4 RAW relations in the fixed
# order (entry_lanes, exit_lanes, left_neighbors, right_neighbors). The CANONICAL
# model is 3-TYPED: entry->predecessor, exit->successor, and left|right merged
# into a single lateral relation. Rationale: the successor/predecessor/lateral
# triple is the standard lane-graph typing; the left-vs-right DIRECTION of a
# lateral neighbor is not a driving-topology distinction at this granularity, and
# merging halves the lateral parameters. Untyped (all four collapsed into one
# relation) and 4-typed (keep left/right separate) are cheap UPWARD ablations,
# kept OUT of the canonical -- every one is a reduction of what the loader emits.
_N_TYPED_RELATIONS = 3          # predecessor, successor, lateral

# Raw-relation channel indices in the loader's [R=4, M, M] tensor.
_R_ENTRY, _R_EXIT, _R_LEFT, _R_RIGHT = 0, 1, 2, 3


class TypedLaneGraphLayer(nn.Module):
    """One relational message-passing (MP) layer over the M lane slots.

        h:     [B, M, H]       per-slot map polyline embeddings (V2 map subgraph)
        adj4:  [B, 4, M, M]    raw adjacency; adj4[b, r, i, j] == 1.0 iff slot i
                               declares slot j as its relation-r neighbor
                               (r: 0=entry, 1=exit, 2=left, 3=right).
        return [B, M, H]       refined embeddings, h + sum_r Ahat_r @ (h W_r)

    TYPED, R-GCN style: one linear per typed relation (predecessor / successor /
    lateral), NO bias -> the message is a pure function of the neighbor
    embeddings. MEAN aggregation with an in-degree clamp (>= 1): a slot with no
    relation-r neighbors -- an isolated lane, a non-lane slot, or a padded slot,
    all of which have all-zero adjacency rows (the loader guarantees no edge
    touches a padded slot) -- contributes a ZERO message.

    REDUCTION TO V2: with adj4 all-zero, every row-normalized Ahat_r is zero
    (0 / clamp(0, min=1) = 0), so every message is exactly zero and the residual
    returns h unchanged. The layer is then the identity, which is what makes the
    V3 forward reduce to V2 BIT-FOR-BIT (the B.2 acceptance) -- not merely a
    structural reduction. The clamp mirrors the masked-MSE denominator clamp
    already used in train_map.py (same defensive pattern in the codebase).
    """

    def __init__(self, hidden_dim):
        super().__init__()
        # bias=False so an all-zero adjacency yields an exact zero message (no
        # additive constant can leak into the residual).
        self.rel_lin = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(_N_TYPED_RELATIONS)
        )

    @staticmethod
    def _typed(adj4):
        """[B,4,M,M] raw (entry,exit,left,right) -> 3 typed [B,M,M]
        (predecessor, successor, lateral=left|right). The lateral union is
        clamped to 1.0 so a slot flagged as BOTH a left and a right neighbor
        (degenerate, should not occur) still counts as a single lateral edge."""
        pred = adj4[:, _R_ENTRY]
        succ = adj4[:, _R_EXIT]
        lat = (adj4[:, _R_LEFT] + adj4[:, _R_RIGHT]).clamp(max=1.0)
        return (pred, succ, lat)

    def forward(self, h, adj4):
        update = torch.zeros_like(h)                          # [B, M, H]
        for A_r, lin in zip(self._typed(adj4), self.rel_lin):
            deg = A_r.sum(dim=2, keepdim=True).clamp(min=1.0)  # [B, M, 1] in-degree
            A_hat = A_r / deg                                  # row-normalized mean
            update = update + torch.bmm(A_hat, lin(h))         # [B, M, H]
        return h + update                                      # residual (identity @ adj=0)


class LaneTopoTrajectoryPredictor(VectorizedSocialMapTrajectoryPredictor):
    """
    Vectorized encoder + SOCIAL context + MAP context + LANE TOPOLOGY (V3). Over
    V2, the SOLE architectural change is a typed lane-graph layer that refines the
    map polyline embeddings [B,M,H] using lane connectivity, applied AFTER the map
    subgraph and BEFORE the target->map cross-attention:

        MAP:  [B,M,Np,2] -> map_subgraph -> maxpool -> map_proj -> [B,M,H]
              (+ optional type embedding, == V2)
              -> lane_graph(map_emb, map_adjacency) -> [B,M,H]     <-- NEW (V3)
              -> target->map cross-attention (== V2)

    Everything else -- the target/neighbor branch, the social cross-attention, the
    fusion LayerNorm, the trajectory/score heads, the standardization buffers --
    is INHERITED unchanged from V2 (this class subclasses V2). The forward is a
    verbatim copy of V2's with the single lane_graph call inserted; it takes ONE
    extra argument, map_adjacency [B, R, M, M] (R=4, the loader's raw relations).

    REDUCTION TO V2 (bit-for-bit, the B.2 ladder contract): the lane_graph is a
    residual layer whose message is exactly zero when map_adjacency is all-zero
    (see TypedLaneGraphLayer). So V3.forward(..., map_adjacency=0) computes the
    IDENTICAL graph of operations as V2.forward(...) using the same shared
    weights -> the outputs are allclose to V2. This is STRONGER than the V2->V1
    reduction (which was structural, because V1 and V2 have distinct fusion
    LayerNorm instances): here, zeroing the new context recovers V2 numerically,
    so any V3-vs-V2 metric delta is 100% attributable to lane topology.

    Params NOT tied (efficiency axis, report in the table): the typed lane-graph
    adds 3 * H*H weights (one bias-free linear per typed relation) on top of the
    V2 count. The exact number is printed by the smoke below.

    Type flag (use_map_type) and standardization behave exactly as in V2 (the
    lane_graph sits after the type-embedding add and is orthogonal to it).
    """

    def __init__(self, input_steps=11, output_steps=80, num_modes=6,
                 hidden_dim=64, n_layers=2, n_features=6, n_heads=4,
                 use_map_type=False):
        super().__init__(
            input_steps=input_steps, output_steps=output_steps,
            num_modes=num_modes, hidden_dim=hidden_dim, n_layers=n_layers,
            n_features=n_features, n_heads=n_heads, use_map_type=use_map_type,
        )
        # The ONLY new module. One MP layer (1 hop): each lane hears its direct
        # predecessor/successor/lateral neighbors. A 2nd hop and an in-layer
        # non-linearity are documented UPWARD ablations, kept out of the canonical
        # minimal step (isolate "does topology help?" before spending compute on
        # reach). Message-passing over M polylines compounds V2's cost -- the
        # hardware ceiling the handoff flags lives here.
        self.lane_graph = TypedLaneGraphLayer(hidden_dim)

    def forward(self, x, neighbors, neighbor_mask,
                map_polylines, map_type, map_mask, map_adjacency,
                check_frame=False):
        """
        x:             [B, 11, 6]        target, raw agent-centric frame.
        neighbors:     [B, N, 11, 6]     neighbors in the SAME target frame (pad 0).
        neighbor_mask: [B, N]            1.0 where the neighbor exists, 0 = padding.
        map_polylines: [B, M, Np, 2]     map polylines in the target frame (pad 0).
        map_type:      [B, M] long       entity type per polyline (used iff flag on).
        map_mask:      [B, M]            1.0 where the polyline exists, 0 = padding.
        map_adjacency: [B, R, M, M]      raw lane adjacency (R=4: entry/exit/left/
                                         right), slot-aligned to map_polylines.
        """
        B = x.shape[0]
        N = neighbors.shape[1]
        M = map_polylines.shape[1]
        Np = map_polylines.shape[2]

        # Frame check on the RAW x (before standardizing, which would destroy the
        # frame). Only the target satisfies pos[10]=(0,0),(sin,cos)[10]=(0,1).
        if check_frame:
            build_vectors(x, check_frame=True)

        # Standardization (identity when buffers 0/1). Target AND neighbors; the
        # map is never standardized (== V2).
        xs = (x - self.feat_mean) / self.feat_std
        ns = (neighbors - self.feat_mean) / self.feat_std

        target_emb = self._encode_agent(xs)                              # [B,H]
        neigh_emb = self._encode_agent(
            ns.reshape(B * N, self.input_steps, self.n_features))
        neigh_emb = neigh_emb.reshape(B, N, self.hidden_dim)             # [B,N,H]

        # Map branch (raw positions), identical to V2 up to the type add.
        map_emb = self._encode_map(map_polylines.reshape(B * M, Np, 2))
        map_emb = map_emb.reshape(B, M, self.hidden_dim)                 # [B,M,H]
        if self.use_map_type:
            map_emb = map_emb + self.map_type_emb(map_type)              # [B,M,H]

        # --- NEW (V3): refine the map embeddings along lane topology BEFORE the
        # target->map cross-attention. Residual + clamped mean-agg => an all-zero
        # map_adjacency leaves map_emb untouched (reduction to V2 bit-for-bit).
        map_emb = self.lane_graph(map_emb, map_adjacency)               # [B,M,H]

        # Two mirrored, independently guarded cross-attentions (== V2).
        social_ctx = self._masked_cross_attn(
            self.social_cross_attn, target_emb, neigh_emb, neighbor_mask)  # [B,H]
        map_ctx = self._masked_cross_attn(
            self.map_cross_attn, target_emb, map_emb, map_mask)            # [B,H]

        fused = self.fusion_norm(target_emb + social_ctx + map_ctx)      # [B,H]

        trajectories = self.traj_head(fused).view(
            B, self.num_modes, self.output_steps, 2)                     # AGENT frame
        scores = self.score_head(fused)                                  # [B,K] logits
        return trajectories, scores


if __name__ == "__main__":
    from src.motion.vectorized_model import CH_X, CH_Y, CH_SIN, CH_COS
    from src.motion.vectorized_social_map_model import (
        VectorizedSocialMapTrajectoryPredictor as _V2, N_MAP_TYPES,
    )

    def _agent_centric(x):
        x[:, 10, CH_X] = 0.0
        x[:, 10, CH_Y] = 0.0
        x[:, 10, CH_SIN] = 0.0
        x[:, 10, CH_COS] = 1.0
        return x

    torch.manual_seed(0)
    B, N, M, Np, R = 4, 16, 64, 20, 4

    for hd in (32, 64, 128):
        m = LaneTopoTrajectoryPredictor(hidden_dim=hd, n_features=6)
        n_params = sum(p.numel() for p in m.parameters())
        print(f"OK: Lane-topo model loaded, hidden_dim={hd}, n_heads={m.n_heads}, "
              f"use_map_type={m.use_map_type} ({n_params:,} parameters).")

    # Parameter delta of the lane graph (reported in the table).
    p_v2 = sum(p.numel() for p in _V2(hidden_dim=64).parameters())
    p_v3 = sum(p.numel() for p in
               LaneTopoTrajectoryPredictor(hidden_dim=64).parameters())
    print(f"   lane-graph param delta: V2={p_v2:,} -> V3={p_v3:,} (+{p_v3 - p_v2:,}).")

    model = LaneTopoTrajectoryPredictor(hidden_dim=64, n_features=6)

    x = _agent_centric(torch.randn(B, 11, 6))
    neighbors = torch.randn(B, N, 11, 6)
    nmask = torch.ones(B, N)
    map_polys = torch.randn(B, M, Np, 2)
    mtype = torch.randint(0, N_MAP_TYPES, (B, M))
    mmask = torch.ones(B, M)
    # A sparse random raw adjacency (entry/exit/left/right). ~5% density leaves
    # many all-zero rows -> exercises the in-degree clamp / isolated nodes.
    adj = (torch.rand(B, R, M, M) < 0.05).float()
    zadj = torch.zeros(B, R, M, M)

    # Row 0: no neighbors AND no map (both presence guards) -- adjacency is moot
    # there (masked out at the cross-attention), but the lane_graph still runs.
    nmask[0] = 0.0
    mmask[0] = 0.0
    nmask[1, 3:] = 0.0
    mmask[1, 5:] = 0.0

    traj, scores = model(x, neighbors, nmask, map_polys, mtype, mmask, adj,
                         check_frame=True)
    print(f"   traj={tuple(traj.shape)} scores={tuple(scores.shape)}")
    assert traj.shape == (B, 6, 80, 2)
    assert scores.shape == (B, 6)
    assert torch.isfinite(traj).all() and torch.isfinite(scores).all(), \
        "NaN/inf in the output -> a guard or the degree clamp failed?"
    print("OK: shapes + finite output (in-degree clamp holds on isolated slots).")

    # === B.2 ACCEPTANCE: reduction to V2 bit-for-bit ==========================
    # With map_adjacency all-zero the lane_graph is the identity, so V3.forward
    # must equal V2.forward using the SAME shared weights. V3 subclasses V2, so
    # every V2 key exists in V3 by the same name -> copy the shared subset into a
    # fresh V2 and compare.
    m_v2 = _V2(hidden_dim=64, n_features=6, use_map_type=False)
    shared = {k: v for k, v in model.state_dict().items()
              if k in m_v2.state_dict()}
    res = m_v2.load_state_dict(shared, strict=False)
    assert not res.missing_keys, f"V2 keys not covered by V3: {res.missing_keys}"
    assert not res.unexpected_keys, f"unexpected V2 keys: {res.unexpected_keys}"

    t3, s3 = model(x, neighbors, nmask, map_polys, mtype, mmask, zadj)
    t2, s2 = m_v2(x, neighbors, nmask, map_polys, mtype, mmask)
    assert torch.allclose(t3, t2, atol=1e-6) and torch.allclose(s3, s2, atol=1e-6), \
        "map_adjacency=0 does NOT reduce V3 to V2 bit-for-bit (residual broken)."
    print("OK: REDUCTION TO V2 -> map_adjacency=0 gives V2 output bit-for-bit.")

    # Topology MOVES the output: real adjacency vs zero adjacency must differ.
    tA, _ = model(x, neighbors, nmask, map_polys, mtype, mmask, adj)
    tZ, _ = model(x, neighbors, nmask, map_polys, mtype, mmask, zadj)
    assert not torch.allclose(tA, tZ), \
        "lane topology did not change the output (message passing is dead)."
    print("OK: lane topology changes the output when edges are present.")

    # Adjacency CONTENT (not just presence) matters: two different graphs differ.
    adj2 = (torch.rand(B, R, M, M) < 0.05).float()
    tB, _ = model(x, neighbors, nmask, map_polys, mtype, mmask, adj2)
    assert not torch.allclose(tA, tB), \
        "different lane graphs gave the same output (adjacency ignored)."
    print("OK: different lane graphs give different outputs (topology is read).")

    # Masked map still does not leak (inherited V2 guard): with map_mask all-zero,
    # changing the map content OR the adjacency must not change the output.
    zmap = torch.zeros(B, M)
    g0, _ = model(x, neighbors, nmask, map_polys, mtype, zmap, adj)
    g1, _ = model(x, neighbors, nmask, torch.randn(B, M, Np, 2), mtype, zmap, adj2)
    assert torch.allclose(g0, g1, atol=1e-6), \
        "masked map/topology leaked into the output (cross-attn guard broken)."
    print("OK: map_mask=0 -> neither map content nor topology affects the output.")

    # Type flag OFF -> map_type ignored (inherited). The lane_graph is orthogonal.
    typeA = torch.zeros(B, M, dtype=torch.long)
    typeB = torch.full((B, M), 2, dtype=torch.long)
    tOffA, _ = model(x, neighbors, nmask, map_polys, typeA, torch.ones(B, M), adj)
    tOffB, _ = model(x, neighbors, nmask, map_polys, typeB, torch.ones(B, M), adj)
    assert torch.allclose(tOffA, tOffB, atol=1e-6), \
        "use_map_type=False but map_type changed the output (flag leaking)."
    print("OK: type flag OFF -> map_type ignored (pure geometry + topology).")

    # state_dict carries the lane_graph keys and reloads cleanly.
    sd = model.state_dict()
    assert any(k.startswith("lane_graph") for k in sd), "lane_graph keys missing!"
    m2 = LaneTopoTrajectoryPredictor(hidden_dim=64, n_features=6)
    m2.load_state_dict(sd)
    print("OK: state_dict carries lane_graph keys and reloads cleanly.")
    print("ALL SMOKES PASSED.")
