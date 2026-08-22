import torch
import torch.nn as nn

# SIBLING de vectorized_model (V1 = contexto social). O V0 NÃO é editado:
# importamos build_vectors + os blocos do subgraph, então a MÁQUINA do encoder
# (construção da polyline, MLP+LayerNorm+ReLU, maxpool permutation-invariant) é
# BYTE-A-BYTE a mesma do V0. A ÚNICA diferença arquitetural entre V0 e V1 é o
# ramo social (cross-attention) -> qualquer Δ métrico é atribuível só a ele.
from src.motion.vectorized_model import (
    build_vectors, _MLP, _SubgraphLayer,
    CH_X, CH_Y, CH_SIN, CH_COS, CH_VX, CH_VY,
)


class VectorizedSocialTrajectoryPredictor(nn.Module):
    """
    Encoder vetorizado + contexto SOCIAL (V1). Sobre o V0:

      * ALVO: [B,11,6] -> build_vectors -> subgraph -> maxpool -> proj -> [B,H]
        (idêntico ao V0).
      * VIZINHOS: [B,N,11,6] -> MESMO subgraph compartilhado -> [B,N,H].
      * FUSÃO: cross-attention (alvo=query, vizinhos=key/value, mascarada por
        presença) + residual + LayerNorm -> [B,H]. É a PRIMEIRA vez que atenção
        aparece na escada (>=2 polylines). Mapa (V2) é outro degrau.

    Contrato preservado do V0 (não quebra a escada):
      - entrada do alvo: [B,11,n_features], n_features=6 (assert).
      - saída: trajectories [B,K,80,2] + scores [B,K] (LOGITS), K=6, mesmas heads.
      - agente-cêntrico como FRAME do input (build_vectors assume); vizinhos vêm
        no MESMO frame do alvo (feito no dataset).
      - padronização como BUFFERS persistentes que viajam no checkpoint
        (paridade train/inferência). Aplicada ao alvo E aos vizinhos com as
        MESMAS stats (subgraph compartilhado -> mesma distribuição de entrada).

    Redução ao V0: com neighbor_mask todo-zero (cena sem vizinhos), o contexto
    é zero e fused = LayerNorm(target_emb). O ramo social é PURAMENTE ADITIVO;
    a diferença vs V0 é o LayerNorm no embedding (documentado, não bit-a-bit).

    Params NÃO casados (eixo de eficiência, reportar na tabela): +~17k sobre o
    V0 (~80k) vindos da cross-attention + LN -> ~97k, ainda <1/3 do flatten.
    """

    def __init__(self, input_steps=11, output_steps=80, num_modes=6,
                 hidden_dim=64, n_layers=2, n_features=6, n_heads=4):
        super().__init__()

        assert n_features == 6, (
            "VectorizedSocialTrajectoryPredictor exige input rico de 6 canais "
            "(x,y,sin,cos,vx,vy); igual ao V0."
        )

        self.input_steps = input_steps
        self.output_steps = output_steps
        self.num_modes = num_modes
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads

        # --- Padronização (V0-std), buffers persistentes, default IDENTIDADE --
        # Idêntico ao V0. O buffer (não uma flag Python) determina raw vs std,
        # e viaja no state_dict -> a inferência recebe as mesmas stats via
        # load_state_dict, sem transform divergente. Aplicado ao alvo E aos
        # vizinhos (mesmos canais, mesmo frame family). Padding mascarado não
        # contamina: os embeddings de vizinhos ausentes são mascarados na atenção.
        self.register_buffer("feat_mean", torch.zeros(n_features))
        self.register_buffer("feat_std", torch.ones(n_features))

        # --- Encoder de polyline COMPARTILHADO (alvo + vizinhos) -------------
        f_vec = 9
        layers, cur = [], f_vec
        for _ in range(n_layers):
            layers.append(_SubgraphLayer(cur, hidden_dim))
            cur = 2 * hidden_dim
        self.subgraph = nn.ModuleList(layers)
        self.proj = nn.Linear(cur, hidden_dim)      # embedding da polyline = H

        # --- NOVO: ramo social (cross-attention alvo <- vizinhos) ------------
        # batch_first=True -> tensores [B, seq, H]. LN pós-residual (estilo
        # transformer) estabiliza a soma target+contexto num único bloco de
        # atenção. É a ÚNICA adição de parâmetros sobre o V0.
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # --- Heads IDÊNTICAS ao V0 (só o encoder+fusão acima diferem) --------
        self.traj_head = nn.Linear(hidden_dim, num_modes * output_steps * 2)
        self.score_head = nn.Linear(hidden_dim, num_modes)

    def load_feature_stats(self, path):
        """Carrega stats congeladas (mean/std por-canal [6]) para os buffers.
        Idêntico ao V0: sin/cos já vêm passthrough (0/1) do arquivo. Chamado no
        treino quando --standardize; a inferência recebe via load_state_dict."""
        import numpy as _np
        blob = _np.load(path, allow_pickle=True).item()
        mean = torch.as_tensor(blob["mean"], dtype=self.feat_mean.dtype)
        std = torch.as_tensor(blob["std"], dtype=self.feat_std.dtype)
        assert mean.shape == self.feat_mean.shape == std.shape, (
            f"stats shape errado: mean{tuple(mean.shape)} std{tuple(std.shape)} "
            f"!= {tuple(self.feat_mean.shape)}"
        )
        assert torch.all(std > 0), "feat_std tem canal <= 0 (divisão por zero)"
        self.feat_mean.copy_(mean)
        self.feat_std.copy_(std)

    def _encode(self, x):
        """Polyline -> embedding [*, H]. COMPARTILHADO entre alvo e vizinhos.
        x já padronizado, shape [*, 11, 6]. build_vectors sem check_frame (os
        vizinhos NÃO estão na origem; só o alvo satisfaz o frame check)."""
        vec = build_vectors(x, check_frame=False)   # [*, 10, 9]
        for layer in self.subgraph:
            vec = layer(vec)                          # [*, 10, 2H]
        pooled = vec.max(dim=1).values                # [*, 2H]
        return self.proj(pooled)                      # [*, H]

    def forward(self, x, neighbors, neighbor_mask, check_frame=False):
        """
        x:             [B, 11, 6]      alvo, frame agente-cêntrico CRU.
        neighbors:     [B, N, 11, 6]   vizinhos no MESMO frame do alvo (pad 0).
        neighbor_mask: [B, N]          1.0 onde o vizinho existe, 0 = padding.
        """
        B = x.shape[0]
        N = neighbors.shape[1]

        # Frame check no x CRU (antes de padronizar, que destruiria o frame).
        # Só o alvo satisfaz pos[10]=(0,0),(sin,cos)[10]=(0,1). Default False.
        if check_frame:
            build_vectors(x, check_frame=True)

        # Padronização (identidade quando buffers 0/1). Broadcast [6] sobre o
        # último eixo tanto do alvo [B,11,6] quanto dos vizinhos [B,N,11,6].
        xs = (x - self.feat_mean) / self.feat_std
        ns = (neighbors - self.feat_mean) / self.feat_std

        target_emb = self._encode(xs)                                   # [B,H]
        neigh_emb = self._encode(ns.reshape(B * N, self.input_steps, self.n_features))
        neigh_emb = neigh_emb.reshape(B, N, self.hidden_dim)            # [B,N,H]

        # Cross-attention: alvo (query) sobre vizinhos (key/value). Vizinhos
        # ausentes viram key_padding_mask=True (ignorados no softmax).
        q = target_emb.unsqueeze(1)                                    # [B,1,H]
        key_padding = (neighbor_mask < 0.5)                            # [B,N] True=ignora

        # GUARD zero-vizinhos: uma linha com TODOS os keys mascarados faria o
        # softmax da atenção operar sobre -inf -> NaN (falha silenciosa). Então
        # só passam pela atenção as linhas com >=1 vizinho; as demais recebem
        # contexto ZERO (fused = LN(target_emb), i.e. sem sinal social).
        attn_ctx = torch.zeros_like(target_emb)                        # [B,H]
        has_nb = neighbor_mask.sum(dim=1) > 0                          # [B]
        if bool(has_nb.any()):
            idx = has_nb.nonzero(as_tuple=True)[0]
            a_out, _ = self.cross_attn(
                q[idx], neigh_emb[idx], neigh_emb[idx],
                key_padding_mask=key_padding[idx],
            )                                                          # [n,1,H]
            attn_ctx = attn_ctx.index_copy(0, idx, a_out.squeeze(1))

        fused = self.attn_norm(target_emb + attn_ctx)                  # [B,H]

        trajectories = self.traj_head(fused).view(
            B, self.num_modes, self.output_steps, 2)                   # frame AGENTE
        scores = self.score_head(fused)                                # [B,K] logits
        return trajectories, scores


if __name__ == "__main__":
    def _agent_centric(x):
        x[:, 10, CH_X] = 0.0
        x[:, 10, CH_Y] = 0.0
        x[:, 10, CH_SIN] = 0.0
        x[:, 10, CH_COS] = 1.0
        return x

    torch.manual_seed(0)
    B, N = 4, 16

    for hd in (32, 64, 128):
        m = VectorizedSocialTrajectoryPredictor(hidden_dim=hd, n_features=6)
        n_params = sum(p.numel() for p in m.parameters())
        print(f"OK: Social model loaded, hidden_dim={hd}, n_heads={m.n_heads} "
              f"({n_params:,} parameters).")

    model = VectorizedSocialTrajectoryPredictor(hidden_dim=64, n_features=6)

    x = _agent_centric(torch.randn(B, 11, 6))
    neighbors = torch.randn(B, N, 11, 6)
    nmask = torch.ones(B, N)
    # linha 0 SEM vizinhos (testa o guard); linha 1 com só 3 vizinhos.
    nmask[0] = 0.0
    nmask[1, 3:] = 0.0

    traj, scores = model(x, neighbors, nmask, check_frame=True)
    print(f"   traj={tuple(traj.shape)} scores={tuple(scores.shape)}")
    assert traj.shape == (B, 6, 80, 2)
    assert scores.shape == (B, 6)
    assert torch.isfinite(traj).all() and torch.isfinite(scores).all(), \
        "NaN/inf na saída -> guard de zero-vizinhos falhou?"

    # Redução ao V0: com neighbor_mask todo-zero, o contexto é zero e a saída
    # tem de ser IDÊNTICA a um forward onde os vizinhos são irrelevantes.
    zmask = torch.zeros(B, N)
    t0, s0 = model(x, neighbors, zmask)
    t1, s1 = model(x, torch.randn(B, N, 11, 6), zmask)  # outros vizinhos, mask 0
    assert torch.allclose(t0, t1, atol=1e-6) and torch.allclose(s0, s1, atol=1e-6), \
        "vizinhos mascarados vazaram para a saída (guard/atenção furados)."
    print("OK: guard zero-vizinhos -> vizinhos mascarados não afetam a saída.")

    # O ramo social MOVE a saída quando há vizinhos (senão a atenção é inócua).
    tA, _ = model(x, neighbors, torch.ones(B, N))
    tZ, _ = model(x, neighbors, torch.zeros(B, N))
    assert not torch.allclose(tA, tZ), "contexto social não mudou a saída."
    print("OK: contexto social altera a saída quando há vizinhos.")

    # Padronização: default identidade == raw; stats fake (sin/cos passthrough)
    # muda a saída e NÃO dispara o frame check (checado no x cru).
    assert torch.allclose(model.feat_mean, torch.zeros(6)) and \
        torch.allclose(model.feat_std, torch.ones(6)), \
        "default dos buffers não é identidade (raw quebrado)"
    t_raw, _ = model(x.clone(), neighbors, torch.ones(B, N), check_frame=True)
    model.feat_mean.copy_(torch.tensor([1.5, -0.3, 0.0, 0.0, 2.0, 0.1]))
    model.feat_std.copy_(torch.tensor([8.0, 6.0, 1.0, 1.0, 3.0, 3.0]))
    t_std, _ = model(x.clone(), neighbors, torch.ones(B, N), check_frame=True)
    assert model.feat_mean[2] == 0 and model.feat_std[2] == 1 and \
        model.feat_mean[3] == 0 and model.feat_std[3] == 1, "sin/cos não passthrough"
    assert not torch.allclose(t_raw, t_std), "std não mudou a saída (buffer ignorado?)"
    print("OK: padronização (raw==identidade, std!=raw, sin/cos passthrough).")

    # Buffers viajam no state_dict (paridade train/inferência).
    sd = model.state_dict()
    assert "feat_mean" in sd and "feat_std" in sd, "buffers fora do state_dict!"
    m2 = VectorizedSocialTrajectoryPredictor(hidden_dim=64, n_features=6)
    m2.load_state_dict(sd)
    assert torch.allclose(m2.feat_mean, model.feat_mean) and \
        torch.allclose(m2.feat_std, model.feat_std), \
        "load_state_dict não restaurou as stats"
    print("OK: state_dict carrega chaves do V1 (incl. cross_attn/attn_norm) e buffers.")
    print("TODOS OS SMOKES PASSARAM.")
