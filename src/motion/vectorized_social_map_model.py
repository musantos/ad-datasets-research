import torch
import torch.nn as nn

# SIBLING of vectorized_social_model (V2 = social context + MAP context). Neither
# V0 nor V1 is edited: build_vectors and the subgraph blocks are imported, so the
# encoder MACHINE (polyline construction, MLP+LayerNorm+ReLU, permutation-invariant
# maxpool) is BYTE-FOR-BYTE the V0 one. The ONLY architectural additions over V1
# are the MAP branch (separate map-subgraph + a second cross-attention) and its
# fusion -> any metric delta is attributable to the map alone.
#
# (This file is the PT->EN rewrite of the V1 sibling: text only, identical
# semantics, following the project rule for siblings whose base carried PT docs.)
from src.motion.vectorized_model import (
    build_vectors, _MLP, _SubgraphLayer,
    CH_X, CH_Y, CH_SIN, CH_COS, CH_VX, CH_VY,
)


# Map-vector width: [x_start, y_start, x_end, y_end] per polyline segment. The map
# has NO time / velocity / heading channels (unlike the 9-d agent vector); the
# segment orientation is the tangent (end - start), which is implicit in the point
# order and is carried into the target frame by the loader's rotation. So
# [start, end] is the minimal VectorNet-style map vector -- V2 = pure geometry.
# An explicit tangent channel (end - start) is a documented FUTURE ablation, kept
# OUT here to keep this the leanest possible geometry-only step.
F_MAP_VEC = 4

# Entity-level map types delivered by the loader (map_type): 0=lane, 1=road_edge,
# 2=road_line. Consumed ONLY when use_map_type=True (V2+type ablation).
N_MAP_TYPES = 3


def build_map_vectors(polylines):
    """Turn fixed-length map polylines into VectorNet-style segment vectors.

        polylines: [*, Np, 2]     points in the target (agent-centric) frame
        return:    [*, Np-1, 4]   -> [x_start, y_start, x_end, y_end] per segment

    Mirrors the POSITIONAL part of build_vectors (agent), minus the time / heading
    / velocity channels the map does not have. A padded (all-zero) polyline yields
    all-zero segments; those polylines are masked out at the polyline level by
    map_mask, so they never reach the cross-attention softmax. No check_frame: the
    map is NOT anchored at the origin (only the target satisfies pos[10]=(0,0)).
    """
    starts = polylines[..., :-1, :]                  # [*, Np-1, 2]  point i
    ends = polylines[..., 1:, :]                     # [*, Np-1, 2]  point i+1
    return torch.cat([starts, ends], dim=-1)         # [*, Np-1, 4]


class VectorizedSocialMapTrajectoryPredictor(nn.Module):
    """
    Vectorized encoder + SOCIAL context + MAP context (V2). Over V1:

      * TARGET:    [B,11,6]      -> build_vectors -> subgraph -> maxpool -> proj -> [B,H]
                   (identical to V0/V1).
      * NEIGHBORS: [B,N,11,6]    -> SAME shared subgraph -> [B,N,H]   (== V1).
      * MAP:       [B,M,Np,2]    -> SEPARATE map-subgraph -> maxpool -> map_proj
                   -> [B,M,H]. A second, independent cross-attention branch.
      * FUSION:    two cross-attentions (target=query; neighbors and map polylines
                   as the two key/value pools, each masked by presence) produce
                   social_ctx and map_ctx; a single LayerNorm fuses them onto the
                   target embedding:  fused = LN(target + social_ctx + map_ctx).

    Why the map is a SEPARATE branch (and not pooled with the neighbors under one
    softmax): keeping the map as a distinct, attributable modality isolates its
    metric delta ("does map close MissRate?") and makes the ablation clean --
    masking only the map branch reduces V2 to V1. Merging map polylines into the
    neighbor pool would blur each source's contribution.

    Contract preserved from V0/V1 (does not break the ladder):
      - target input: [B,11,n_features], n_features=6 (assert).
      - outputs: trajectories [B,K,80,2] + scores [B,K] (LOGITS), K=6, same heads.
      - agent-centric as the FRAME of the input (build_vectors assumes it);
        neighbors AND map come in the SAME target frame (done in the dataset).
      - standardization as persistent BUFFERS that travel in the checkpoint
        (train/inference parity). Applied to the target AND neighbors with the
        SAME stats; the MAP is NEVER standardized (raw positions, a different
        distribution). V2 runs raw-only, so the buffers stay identity in practice;
        they are kept for parity with V0/V1 and a stable state_dict.

    Type flag (item-2 ablation): use_map_type=False (default) -> map_type is
    IGNORED, V2 = pure geometry. use_map_type=True -> a 3-way entity embedding is
    added to each polyline embedding (V2+type). The embedding module exists only
    when the flag is on, so the OFF and ON checkpoints are distinct ablation arms.

    Reduction to V1: with map_mask all-zero (map-less scene), the map guard yields
    map_ctx=0 and fused = LN(target + social_ctx). With BOTH masks zero,
    fused = LN(target). The map branch is PURELY ADDITIVE. The reduction is
    structural (the fusion LayerNorm is a distinct learned instance from V1's),
    not bit-for-bit -- same convention as the V1->V0 reduction.

    Params NOT tied (efficiency axis, report in the table): the map subgraph +
    map projection + second cross-attention (+ the optional type embedding) sit on
    top of the ~97k V1 count. The exact number is printed by the smoke below.
    """

    def __init__(self, input_steps=11, output_steps=80, num_modes=6,
                 hidden_dim=64, n_layers=2, n_features=6, n_heads=4,
                 use_map_type=False):
        super().__init__()

        assert n_features == 6, (
            "VectorizedSocialMapTrajectoryPredictor needs the rich 6-channel input "
            "(x,y,sin,cos,vx,vy); same as V0/V1."
        )

        self.input_steps = input_steps
        self.output_steps = output_steps
        self.num_modes = num_modes
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.use_map_type = bool(use_map_type)

        # --- Standardization (V0-std), persistent buffers, IDENTITY default ---
        # Same as V0/V1. The buffer (not a Python flag) decides raw vs std and
        # travels in the state_dict -> inference gets the same stats via
        # load_state_dict, no divergent transform. Applied to the target AND the
        # neighbors (same channels, same frame family). The MAP branch does NOT
        # use these buffers. V2 grid is raw-only; buffers stay identity.
        self.register_buffer("feat_mean", torch.zeros(n_features))
        self.register_buffer("feat_std", torch.ones(n_features))

        # --- SHARED agent/neighbor polyline encoder (byte-for-byte V0/V1) ------
        f_vec = 9
        layers, cur = [], f_vec
        for _ in range(n_layers):
            layers.append(_SubgraphLayer(cur, hidden_dim))
            cur = 2 * hidden_dim
        self.subgraph = nn.ModuleList(layers)
        self.proj = nn.Linear(cur, hidden_dim)          # target/neighbor embedding = H

        # --- SEPARATE map polyline encoder (same machine, 4-d segment input) ---
        # Same _SubgraphLayer/_MLP class as the agent branch, only the input width
        # differs (F_MAP_VEC=4 vs 9) -> the map subgraph is machine-identical to
        # V0's, so any delta is the branch, not a different encoder.
        mlayers, mcur = [], F_MAP_VEC
        for _ in range(n_layers):
            mlayers.append(_SubgraphLayer(mcur, hidden_dim))
            mcur = 2 * hidden_dim
        self.map_subgraph = nn.ModuleList(mlayers)
        self.map_proj = nn.Linear(mcur, hidden_dim)     # map polyline embedding = H

        # --- Optional entity-type embedding (V2+type ablation) -----------------
        # Added to the polyline-level embedding AFTER the geometry subgraph, so the
        # geometry machine is untouched by the flag; only this additive path turns
        # on. Created only when the flag is on (OFF/ON = separate checkpoints).
        if self.use_map_type:
            self.map_type_emb = nn.Embedding(N_MAP_TYPES, hidden_dim)

        # --- Two mirrored cross-attention branches (social + map) --------------
        # batch_first=True -> [B, seq, H]. Each branch: target=query over one
        # key/value pool, masked by presence. A single fusion LayerNorm normalizes
        # the summed contexts on top of the target embedding.
        self.social_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads, batch_first=True)
        self.map_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads, batch_first=True)
        self.fusion_norm = nn.LayerNorm(hidden_dim)

        # --- Heads IDENTICAL to V0/V1 (only encoder+fusion above differ) -------
        self.traj_head = nn.Linear(hidden_dim, num_modes * output_steps * 2)
        self.score_head = nn.Linear(hidden_dim, num_modes)

    def load_feature_stats(self, path):
        """Load frozen per-channel stats (mean/std [6]) into the buffers.
        Same as V0/V1: sin/cos come passthrough (0/1) from the file. Called in
        training under --standardize; inference receives them via load_state_dict.
        NOTE: V2 runs raw-only, so this is normally unused -- kept for parity."""
        import numpy as _np
        blob = _np.load(path, allow_pickle=True).item()
        mean = torch.as_tensor(blob["mean"], dtype=self.feat_mean.dtype)
        std = torch.as_tensor(blob["std"], dtype=self.feat_std.dtype)
        assert mean.shape == self.feat_mean.shape == std.shape, (
            f"wrong stats shape: mean{tuple(mean.shape)} std{tuple(std.shape)} "
            f"!= {tuple(self.feat_mean.shape)}"
        )
        assert torch.all(std > 0), "feat_std has a channel <= 0 (division by zero)"
        self.feat_mean.copy_(mean)
        self.feat_std.copy_(std)

    def _encode_agent(self, x):
        """Agent/neighbor polyline -> embedding [*, H]. SHARED encoder. x already
        standardized, shape [*, 11, 6]. build_vectors without check_frame (only
        the target is at the origin; neighbors are not)."""
        vec = build_vectors(x, check_frame=False)       # [*, 10, 9]
        for layer in self.subgraph:
            vec = layer(vec)                             # [*, 10, 2H]
        pooled = vec.max(dim=1).values                   # [*, 2H]
        return self.proj(pooled)                         # [*, H]

    def _encode_map(self, polys):
        """Map polyline -> embedding [*, H] via the SEPARATE map subgraph.
        polys shape [*, Np, 2] in the target frame (raw, not standardized)."""
        vec = build_map_vectors(polys)                   # [*, Np-1, 4]
        for layer in self.map_subgraph:
            vec = layer(vec)                             # [*, Np-1, 2H]
        pooled = vec.max(dim=1).values                   # [*, 2H]
        return self.map_proj(pooled)                     # [*, H]

    @staticmethod
    def _masked_cross_attn(attn, query_emb, ctx_emb, presence_mask):
        """One guarded cross-attention.

            attn:          nn.MultiheadAttention (batch_first)
            query_emb:     [B, H]        the target embedding (query)
            ctx_emb:       [B, S, H]     the key/value pool (neighbors OR map)
            presence_mask: [B, S]        1.0 where the entry exists, 0 = padding

        Returns context [B, H]. GUARD: a row whose keys are ALL masked would make
        the attention softmax run over -inf -> NaN. So only rows with >=1 present
        entry pass through the attention; the rest get ZERO context (no signal
        from that branch on those rows). Mirrors the V1 zero-neighbor guard."""
        ctx = torch.zeros_like(query_emb)                # [B, H]
        has = presence_mask.sum(dim=1) > 0               # [B]
        if bool(has.any()):
            idx = has.nonzero(as_tuple=True)[0]
            q = query_emb[idx].unsqueeze(1)              # [n, 1, H]
            key_padding = (presence_mask[idx] < 0.5)     # [n, S] True = ignore
            out, _ = attn(q, ctx_emb[idx], ctx_emb[idx],
                          key_padding_mask=key_padding)  # [n, 1, H]
            ctx = ctx.index_copy(0, idx, out.squeeze(1))
        return ctx

    def forward(self, x, neighbors, neighbor_mask,
                map_polylines, map_type, map_mask, check_frame=False):
        """
        x:             [B, 11, 6]       target, raw agent-centric frame.
        neighbors:     [B, N, 11, 6]    neighbors in the SAME target frame (pad 0).
        neighbor_mask: [B, N]           1.0 where the neighbor exists, 0 = padding.
        map_polylines: [B, M, Np, 2]    map polylines in the target frame (pad 0).
        map_type:      [B, M] long      entity type per polyline (used iff flag on).
        map_mask:      [B, M]           1.0 where the polyline exists, 0 = padding.
        """
        B = x.shape[0]
        N = neighbors.shape[1]
        M = map_polylines.shape[1]
        Np = map_polylines.shape[2]

        # Frame check on the RAW x (before standardizing, which would destroy the
        # frame). Only the target satisfies pos[10]=(0,0),(sin,cos)[10]=(0,1).
        if check_frame:
            build_vectors(x, check_frame=True)

        # Standardization (identity when buffers 0/1). Broadcast [6] over the last
        # axis of both target [B,11,6] and neighbors [B,N,11,6]. Map is untouched.
        xs = (x - self.feat_mean) / self.feat_std
        ns = (neighbors - self.feat_mean) / self.feat_std

        target_emb = self._encode_agent(xs)                              # [B,H]
        neigh_emb = self._encode_agent(
            ns.reshape(B * N, self.input_steps, self.n_features))
        neigh_emb = neigh_emb.reshape(B, N, self.hidden_dim)             # [B,N,H]

        # Map branch (raw positions). Encode each polyline, then optionally add the
        # entity-type embedding at the polyline level (masked rows are irrelevant).
        map_emb = self._encode_map(map_polylines.reshape(B * M, Np, 2))
        map_emb = map_emb.reshape(B, M, self.hidden_dim)                 # [B,M,H]
        if self.use_map_type:
            map_emb = map_emb + self.map_type_emb(map_type)              # [B,M,H]

        # Two mirrored, independently guarded cross-attentions.
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
    def _agent_centric(x):
        x[:, 10, CH_X] = 0.0
        x[:, 10, CH_Y] = 0.0
        x[:, 10, CH_SIN] = 0.0
        x[:, 10, CH_COS] = 1.0
        return x

    torch.manual_seed(0)
    B, N, M, Np = 4, 16, 64, 20

    for hd in (32, 64, 128):
        m = VectorizedSocialMapTrajectoryPredictor(hidden_dim=hd, n_features=6)
        n_params = sum(p.numel() for p in m.parameters())
        print(f"OK: Social+Map model loaded, hidden_dim={hd}, n_heads={m.n_heads}, "
              f"use_map_type={m.use_map_type} ({n_params:,} parameters).")

    # Parameter delta of the type flag (reported in the table).
    m_on = VectorizedSocialMapTrajectoryPredictor(hidden_dim=64, use_map_type=True)
    p_off = sum(p.numel() for p in
                VectorizedSocialMapTrajectoryPredictor(hidden_dim=64).parameters())
    p_on = sum(p.numel() for p in m_on.parameters())
    print(f"   type flag param delta: OFF={p_off:,} -> ON={p_on:,} (+{p_on - p_off:,}).")

    model = VectorizedSocialMapTrajectoryPredictor(hidden_dim=64, n_features=6)

    x = _agent_centric(torch.randn(B, 11, 6))
    neighbors = torch.randn(B, N, 11, 6)
    nmask = torch.ones(B, N)
    map_polys = torch.randn(B, M, Np, 2)
    mtype = torch.randint(0, N_MAP_TYPES, (B, M))
    mmask = torch.ones(B, M)
    # Row 0: no neighbors AND no map (tests both guards at once).
    nmask[0] = 0.0
    mmask[0] = 0.0
    # Row 1: a few neighbors, a few polylines.
    nmask[1, 3:] = 0.0
    mmask[1, 5:] = 0.0

    traj, scores = model(x, neighbors, nmask, map_polys, mtype, mmask, check_frame=True)
    print(f"   traj={tuple(traj.shape)} scores={tuple(scores.shape)}")
    assert traj.shape == (B, 6, 80, 2)
    assert scores.shape == (B, 6)
    assert torch.isfinite(traj).all() and torch.isfinite(scores).all(), \
        "NaN/inf in the output -> a zero-context guard failed?"
    print("OK: shapes + both guards (zero-neighbor AND zero-map on row 0) -> finite.")

    # Reduction: with map_mask all-zero, the map branch contributes ZERO -> the
    # output must not depend on the (masked) map content.
    zmap = torch.zeros(B, M)
    t0, s0 = model(x, neighbors, nmask, map_polys, mtype, zmap)
    t1, s1 = model(x, neighbors, nmask, torch.randn(B, M, Np, 2), mtype, zmap)
    assert torch.allclose(t0, t1, atol=1e-6) and torch.allclose(s0, s1, atol=1e-6), \
        "masked map leaked into the output (map guard/attention broken)."
    print("OK: map guard -> masked polylines do not affect the output.")

    # Same reduction for the social branch (unchanged from V1).
    zmask = torch.zeros(B, N)
    t2, _ = model(x, neighbors, zmask, map_polys, mtype, mmask)
    t3, _ = model(x, torch.randn(B, N, 11, 6), zmask, map_polys, mtype, mmask)
    assert torch.allclose(t2, t3, atol=1e-6), \
        "masked neighbors leaked into the output (social guard broken)."
    print("OK: social guard -> masked neighbors do not affect the output.")

    # The MAP branch MOVES the output when polylines are present.
    tA, _ = model(x, neighbors, nmask, map_polys, mtype, torch.ones(B, M))
    tZ, _ = model(x, neighbors, nmask, map_polys, mtype, torch.zeros(B, M))
    assert not torch.allclose(tA, tZ), "map context did not change the output."
    print("OK: map context changes the output when polylines are present.")

    # The SOCIAL branch still moves the output.
    tS1, _ = model(x, neighbors, torch.ones(B, N), map_polys, mtype, mmask)
    tS0, _ = model(x, neighbors, torch.zeros(B, N), map_polys, mtype, mmask)
    assert not torch.allclose(tS1, tS0), "social context did not change the output."
    print("OK: social context changes the output when neighbors are present.")

    # Type flag OFF -> map_type is IGNORED (changing it must NOT change the output).
    typeA = torch.zeros(B, M, dtype=torch.long)
    typeB = torch.full((B, M), 2, dtype=torch.long)
    tOffA, _ = model(x, neighbors, nmask, map_polys, typeA, torch.ones(B, M))
    tOffB, _ = model(x, neighbors, nmask, map_polys, typeB, torch.ones(B, M))
    assert torch.allclose(tOffA, tOffB, atol=1e-6), \
        "use_map_type=False but map_type changed the output (flag leaking)."
    print("OK: type flag OFF -> map_type ignored (pure geometry).")

    # Type flag ON -> map_type CHANGES the output (semantics are used).
    tOnA, _ = m_on(x, neighbors, nmask, map_polys, typeA, torch.ones(B, M))
    tOnB, _ = m_on(x, neighbors, nmask, map_polys, typeB, torch.ones(B, M))
    assert not torch.allclose(tOnA, tOnB), \
        "use_map_type=True but map_type did not change the output (embedding dead)."
    print("OK: type flag ON -> map_type changes the output (V2+type live).")

    # Standardization: default identity == raw; fake stats (sin/cos passthrough)
    # change the output and do NOT trip the frame check (checked on raw x).
    assert torch.allclose(model.feat_mean, torch.zeros(6)) and \
        torch.allclose(model.feat_std, torch.ones(6)), \
        "buffer default is not identity (raw broken)"
    t_raw, _ = model(x.clone(), neighbors, torch.ones(B, N),
                     map_polys, mtype, torch.ones(B, M), check_frame=True)
    model.feat_mean.copy_(torch.tensor([1.5, -0.3, 0.0, 0.0, 2.0, 0.1]))
    model.feat_std.copy_(torch.tensor([8.0, 6.0, 1.0, 1.0, 3.0, 3.0]))
    t_std, _ = model(x.clone(), neighbors, torch.ones(B, N),
                     map_polys, mtype, torch.ones(B, M), check_frame=True)
    assert model.feat_mean[2] == 0 and model.feat_std[2] == 1 and \
        model.feat_mean[3] == 0 and model.feat_std[3] == 1, "sin/cos not passthrough"
    assert not torch.allclose(t_raw, t_std), "std did not change the output (buffer ignored?)"
    print("OK: standardization (raw==identity, std!=raw, sin/cos passthrough).")

    # Buffers travel in the state_dict (train/inference parity), and the map
    # modules load back cleanly.
    sd = model.state_dict()
    assert "feat_mean" in sd and "feat_std" in sd, "buffers missing from state_dict!"
    assert any(k.startswith("map_subgraph") for k in sd) and "map_proj.weight" in sd \
        and any(k.startswith("map_cross_attn") for k in sd), "map keys missing!"
    m2 = VectorizedSocialMapTrajectoryPredictor(hidden_dim=64, n_features=6)
    m2.load_state_dict(sd)
    assert torch.allclose(m2.feat_mean, model.feat_mean) and \
        torch.allclose(m2.feat_std, model.feat_std), "load_state_dict did not restore stats"
    print("OK: state_dict carries map keys + buffers and reloads cleanly.")
    print("ALL SMOKES PASSED.")
