import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# ÚNICA suposição a verificar contra o cache real: ordem dos 6 canais ricos.
# Memória do projeto: (x, y, heading->sin/cos, vx, vy). Se o cache usar outra
# ordem, corrigir SÓ aqui -- os asserts de frame em build_vectors pegam ordem
# trocada (o último (sin,cos) tem de bater (0,1)).
# ---------------------------------------------------------------------------
CH_X, CH_Y, CH_SIN, CH_COS, CH_VX, CH_VY = 0, 1, 2, 3, 4, 5


def build_vectors(hist, check_frame=False):
    """
    Constrói a polyline do histórico do alvo no estilo VectorNet: cada par de
    frames consecutivos (i -> i+1) vira UM vetor [start, end, atributos].

    Derivado on-the-fly do cache de 6 canais -- NÃO toca no preprocessor nem no
    contrato .npy. Atributos (sin/cos do heading, velocidade) ancorados no
    endpoint (frame i+1): o último vetor carrega o estado atual t=10, o mesmo
    sinal de "posição de partida" que ganhou na Parte A. dt=i/9 preserva a
    recência apesar do maxpool permutation-invariant do subgraph.

        hist:   [batch, 11, 6]  (frame agente-cêntrico: origem em pos[10], head[10]=0)
        return: [batch, 10, 9]  -> [xs, ys, xe, ye, sin, cos, vx, vy, dt]
    """
    B, T, C = hist.shape
    assert T == 11 and C == 6, f"esperado [B,11,6], recebido {tuple(hist.shape)}"

    pos = hist[..., [CH_X, CH_Y]]                 # [B,11,2]
    starts = pos[:, :-1, :]                        # [B,10,2]  frame i
    ends = pos[:, 1:, :]                           # [B,10,2]  frame i+1
    sincos = hist[:, 1:, [CH_SIN, CH_COS]]         # [B,10,2]  endpoint
    vel = hist[:, 1:, [CH_VX, CH_VY]]              # [B,10,2]  endpoint

    n = T - 1
    dt = (torch.arange(n, device=hist.device, dtype=hist.dtype) / (n - 1))
    dt = dt.view(1, n, 1).expand(B, n, 1)          # [B,10,1]

    vec = torch.cat([starts, ends, sincos, vel, dt], dim=-1)   # [B,10,9]

    if check_frame:
        last_end = vec[:, -1, 2:4]                 # xe, ye do último vetor
        last_sc = vec[:, -1, 4:6]                  # sin, cos do último vetor
        assert torch.allclose(last_end, torch.zeros_like(last_end), atol=1e-4), \
            f"último x_end != 0 -> frame/ordem de canal errado: {last_end[0].tolist()}"
        tgt = torch.tensor([0.0, 1.0], device=vec.device, dtype=vec.dtype)
        assert torch.allclose(last_sc, tgt.expand_as(last_sc), atol=1e-4), \
            f"último (sin,cos) != (0,1) -> heading não zerado: {last_sc[0].tolist()}"

    return vec


class _MLP(nn.Module):
    """Linear + LayerNorm + ReLU (bloco canônico do subgraph VectorNet)."""
    def __init__(self, d_in, d_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_out), nn.LayerNorm(d_out), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class _SubgraphLayer(nn.Module):
    """genc = MLP(nó); agg = maxpool_nós(genc); out = concat(genc, agg) -> 2*hidden."""
    def __init__(self, d_in, hidden):
        super().__init__()
        self.mlp = _MLP(d_in, hidden)

    def forward(self, x):                          # x: [B, N, d_in]
        h = self.mlp(x)                             # [B, N, hidden]
        agg = h.max(dim=1, keepdim=True).values     # [B, 1, hidden]
        return torch.cat([h, agg.expand_as(h)], dim=-1)   # [B, N, 2*hidden]


class VectorizedTrajectoryPredictor(nn.Module):
    """
    Vectorized-encoder version of MultimodalTrajectoryPredictor / SequentialTrajectoryPredictor.

    Conceptual difference from the MLP (flatten) and GRU (recurrence) siblings:
    the 11 history frames become a POLYLINE of 10 vectors and are aggregated by
    a VectorNet subgraph (shared node MLP + permutation-invariant maxpool). The
    ONLY thing that changes between the three experiments is how the temporal
    history is encoded (flatten -> recurrence -> polyline subgraph). Everything
    downstream is identical, so any metric difference is attributable to the
    encoder and nothing else:
        - same input:   [batch, 11, n_features]
        - same outputs: trajectories [batch, K, 80, 2]
                        scores       [batch, K]  (LOGITS, not probabilities)
        - same K=6, same output horizon, same two Linear heads.

    This is the V0 degrau of Phase 2: it isolates the vectorized MACHINERY, not
    context. Social agents (V1) and map polylines (V2) are later degraus. The
    agent-centric normalization already validated in item 4 enters here as the
    reference FRAME of the input (build_vectors assumes it), not as a branch.

    IMPORTANT: this model requires the rich 6-channel input (x, y, sin, cos, vx,
    vy). Unlike the MLP/GRU siblings, n_features=2 is NOT meaningful -- there is
    no heading/velocity to build vectors from. It is compared against the
    flatten-AGENT arm (also 6ch), not the 2ch Part A baseline.

    Design notes:
        - hidden_dim=64 (subgraph width d) with 2 subgraph layers is ~the
          canonical VectorNet scale, and ~matches the flatten-agent capacity so
          the comparison is "machine vs machine", not "capacity vs capacity".
          Param count is reported in the table, exactly like the GRU sibling;
          hidden_dim / n_layers are exposed as the knobs they are (32/128 sweep
          runs as a documented robustness sub-experiment, not to pick a winner).
        - The global maxpool at each subgraph layer is what generalizes to
          variable-length map polylines in V2.

    Shapes:
        input:   [batch, 11, n_features]  (n_features=6)
        outputs: trajectories [batch, K, 80, 2]
                 scores       [batch, K]
    """

    def __init__(self, input_steps=11, output_steps=80, num_modes=6,
                 hidden_dim=64, n_layers=2, n_features=6):
        super(VectorizedTrajectoryPredictor, self).__init__()

        assert n_features == 6, (
            "VectorizedTrajectoryPredictor exige input rico de 6 canais "
            "(x,y,sin,cos,vx,vy); n_features=2 não é suportado (sem heading/vel "
            "para construir os vetores). Compara-se ao braço flatten-AGENTE."
        )

        self.input_steps = input_steps
        self.output_steps = output_steps
        self.num_modes = num_modes
        self.n_features = n_features
        self.hidden_dim = hidden_dim

        # --- Padronização de features (V0-std) como BUFFERS persistentes -----
        # O comportamento é DETERMINADO pelo buffer, não por uma flag Python:
        # forward SEMPRE aplica (x - feat_mean) / feat_std. Com os defaults
        # abaixo (mean=0, std=1) isso é IDENTIDADE -> V0-raw, entrada crua,
        # bit-a-bit como antes. load_feature_stats() sobrescreve os buffers com
        # as stats congeladas do cache_train -> V0-std.
        #
        # Por que buffer e não pré-processamento no dataloader: buffers entram
        # no state_dict e VIAJAM COM O CHECKPOINT. A inferência reconstrói o
        # modelo, dá load_state_dict, e recebe as mesmas stats automaticamente
        # -> paridade train/inferência por construção, sem risco de transform
        # divergente (a falha silenciosa clássica). run_inference_vectorized NÃO
        # precisa saber de stats; só do sufixo _std no path do checkpoint.
        #
        # sin/cos ficam PASSTHROUGH (mean=0, std=1) por decisão: já ∈ [-1,1], e
        # padronizá-los não paga; o ganho é em x,y (espalhamento) e vx,vy. Essa
        # decisão é gravada no ARQUIVO de stats (canais 2,3 = 0/1), não aqui.
        self.register_buffer("feat_mean", torch.zeros(n_features))
        self.register_buffer("feat_std", torch.ones(n_features))

        # Encoder: subgraph VectorNet sobre a polyline de 10 vetores (9 feats/nó).
        f_vec = 9
        layers, cur = [], f_vec
        for _ in range(n_layers):
            layers.append(_SubgraphLayer(cur, hidden_dim))
            cur = 2 * hidden_dim                    # concat dobra a dim entre camadas
        self.subgraph = nn.ModuleList(layers)
        self.proj = nn.Linear(cur, hidden_dim)      # embedding da polyline = hidden_dim

        # Head 1: the K trajectories. K * 80 * 2 = 960 values for K=6.
        # IDÊNTICA aos siblings MLP/GRU -- só o encoder acima difere.
        self.traj_head = nn.Linear(hidden_dim, num_modes * output_steps * 2)

        # Head 2: the score of each mode. Output is LOGITS -- softmax left to the
        # consumer, exactly as in the MLP/GRU models (cross_entropy in training
        # applies it internally; run_inference applies it explicitly before
        # saving, because the official metric expects comparable scores).
        self.score_head = nn.Linear(hidden_dim, num_modes)

    def load_feature_stats(self, path):
        """Carrega stats congeladas (mean/std por-canal) para dentro dos buffers.

        O arquivo é um dict np.save'd com chaves 'mean' e 'std', ambas [6], já
        com sin/cos em passthrough (mean=0/std=1 nos canais 2,3). Chamado no
        treino quando --standardize; a inferência NÃO chama isto (recebe as
        stats via load_state_dict do checkpoint). Idempotente."""
        import numpy as _np
        blob = _np.load(path, allow_pickle=True).item()
        mean = torch.as_tensor(blob["mean"], dtype=self.feat_mean.dtype)
        std = torch.as_tensor(blob["std"], dtype=self.feat_std.dtype)
        assert mean.shape == self.feat_mean.shape == std.shape, (
            f"stats com shape errado: mean{tuple(mean.shape)} "
            f"std{tuple(std.shape)} != esperado {tuple(self.feat_mean.shape)}"
        )
        assert torch.all(std > 0), "feat_std tem canal <= 0 (divisão por zero)"
        # copy_ preserva o device/dtype dos buffers e mantém tudo no state_dict.
        self.feat_mean.copy_(mean)
        self.feat_std.copy_(std)

    def forward(self, x, check_frame=False):
        # x arrives as [batch, 11, n_features].
        batch_size = x.shape[0]

        # Padronização (identidade quando buffers = 0/1). Broadcasting [6] sobre
        # [B,11,6]. Feita ANTES de build_vectors: os vetores VectorNet passam a
        # ser derivados da entrada padronizada (x,y espalhados, vx,vy em z-score,
        # sin/cos intactos). A saída (traj_head) segue no frame agente em METROS
        # -- padroniza-se só a ENTRADA, então a inversão agente->SDC da
        # inferência é inalterada.
        if check_frame:
            # Asserts de frame pressupõem o frame agente CRU (pos[10]=(0,0),
            # (sin,cos)[10]=(0,1)); padronizar destrói isso. Então checa-se o
            # frame no x CRU (descartado), antes de padronizar. Só ocorre no
            # smoke/__main__ (default False no treino/inferência).
            build_vectors(x, check_frame=True)

        x = (x - self.feat_mean) / self.feat_std

        vec = build_vectors(x, check_frame=False)   # [B,10,9]
        for layer in self.subgraph:
            vec = layer(vec)                               # [B,10,2*hidden]
        pooled = vec.max(dim=1).values                     # [B,2*hidden]
        features = self.proj(pooled)                       # [B,hidden]  (embedding)

        trajectories = self.traj_head(features)
        trajectories = trajectories.view(
            batch_size, self.num_modes, self.output_steps, 2
        )
        # NOTA: trajectories está no FRAME AGENTE. Plugar a inversão agente->mundo
        # já validada (item 4) ANTES das métricas oficiais.

        scores = self.score_head(features)  # [batch, K]

        return trajectories, scores


if __name__ == "__main__":
    # frame agente-cêntrico no dado sintético p/ os asserts de frame passarem
    def _agent_centric(x):
        x[:, 10, CH_X] = 0.0
        x[:, 10, CH_Y] = 0.0
        x[:, 10, CH_SIN] = 0.0
        x[:, 10, CH_COS] = 1.0
        return x

    for hd in (32, 64, 128):   # varredura de robustez sai de graça (mesma classe)
        model = VectorizedTrajectoryPredictor(hidden_dim=hd, n_features=6)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"OK: Vectorized (subgraph) model loaded, hidden_dim={hd}, "
              f"n_features=6 ({n_params:,} parameters).")

        test_input = _agent_centric(torch.randn(2, 11, 6))
        traj, scores = model(test_input, check_frame=True)
        print(f"   Trajectories shape: {tuple(traj.shape)}")   # [2, 6, 80, 2]
        print(f"   Scores shape:       {tuple(scores.shape)}")  # [2, 6]

    # --- Buffers de padronização: default IDENTIDADE == raw ------------------
    m = VectorizedTrajectoryPredictor(hidden_dim=64, n_features=6)
    assert torch.allclose(m.feat_mean, torch.zeros(6)) and torch.allclose(m.feat_std, torch.ones(6)), \
        "default dos buffers não é identidade (raw quebrado)"
    xa = _agent_centric(torch.randn(2, 11, 6))
    t_raw, _ = m(xa.clone(), check_frame=True)   # identidade -> equivale a raw

    # --- Caminho std: stats fake com sin/cos PASSTHROUGH (canais 2,3 = 0/1) --
    # check_frame=True deve continuar passando: o frame é checado no x CRU
    # ANTES de padronizar, então padronizar não dispara o assert.
    fake_mean = torch.tensor([1.5, -0.3, 0.0, 0.0, 2.0, 0.1])
    fake_std  = torch.tensor([8.0,  6.0, 1.0, 1.0, 3.0, 3.0])
    m.feat_mean.copy_(fake_mean); m.feat_std.copy_(fake_std)
    t_std, _ = m(xa.clone(), check_frame=True)   # não deve levantar assert
    assert m.feat_mean[2] == 0 and m.feat_std[2] == 1 and m.feat_mean[3] == 0 and m.feat_std[3] == 1, \
        "sin/cos não estão em passthrough"
    assert not torch.allclose(t_raw, t_std), "std não mudou a saída (buffer ignorado?)"

    # --- Buffers viajam no state_dict (paridade train/inferência) ------------
    sd = m.state_dict()
    assert "feat_mean" in sd and "feat_std" in sd, "buffers fora do state_dict!"
    m2 = VectorizedTrajectoryPredictor(hidden_dim=64, n_features=6)
    m2.load_state_dict(sd)
    assert torch.allclose(m2.feat_mean, fake_mean) and torch.allclose(m2.feat_std, fake_std), \
        "load_state_dict não restaurou as stats -> inferência veria stats erradas"

    print("OK: shapes, frame agente-cêntrico, passthrough sin/cos, e "
          "paridade via state_dict validados (raw==identidade, std!=raw).")