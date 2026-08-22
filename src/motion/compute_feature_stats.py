#!/usr/bin/env python3
"""
Computa UMA vez as estatísticas de feature (mean/std por-canal) do cache_train,
no MESMO frame e MESMAS features que o modelo vê no treino, e salva num .npy
congelado. É o insumo do V0-std: train_vectorized --standardize carrega este
arquivo para os buffers do modelo (que viajam no checkpoint -> paridade
train/inferência por construção).

DECISÕES (fixas a priori, casam com o modelo):
  * Frame AGENTE (agent_centric=True). O V0 é agente-only; padronizar num frame
    diferente do treino seria confound.
  * Features ("x","y","heading","vx","vy") -> 6 canais (heading expande p/
    sin,cos DENTRO do dataset). Ordem = a assumida pelo modelo:
    (x, y, sin, cos, vx, vy).
  * sin/cos em PASSTHROUGH: canais 2,3 forçados a mean=0, std=1 DEPOIS do
    cálculo. Decisão do projeto (já ∈ [-1,1]; o ganho é em x,y e vx,vy). Assim o
    arquivo já sai "pronto pra usar" -- o modelo não precisa saber disso.
  * Stats sobre TODOS os pares (exemplo, frame) do histórico [11,6]. z-score
    padrão por canal.

SÓ o cache_train entra aqui. O cache_val NUNCA -- as stats têm de ser cegas à
validação (senão vaza informação da métrica pro pré-processamento).

Uso (container GPU, mesmo import de train_vectorized):
    python3 -m src.motion.compute_feature_stats
    # ou: python3 -m src.motion.compute_feature_stats --out /caminho/stats.npy

Saída: dict np.save'd com {'mean'[6], 'std'[6], 'features', 'channel_order',
'agent_centric', 'n_examples', 'n_frames', 'passthrough'} em --out.
"""

import os
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.core.waymo_pytorch_dataset_agentcentric import WaymoMotionDatasetAgentCentric

# Casa com train_vectorized.py (mesmas fontes de verdade).
TRAIN_CACHE = "/workspace/datasets/waymo/cache_train"
FEATURES = ("x", "y", "heading", "vx", "vy")
DEFAULT_OUT = "/workspace/experiments/feature_stats.npy"

# Índices dos canais sin/cos na saída de 6 canais do dataset (x,y,sin,cos,vx,vy).
# Mantidos em passthrough. Espelham CH_SIN, CH_COS de vectorized_model.py.
PASSTHROUGH_CHANNELS = (2, 3)

STD_FLOOR = 1e-6   # evita divisão por ~0 (nenhum canal real deve chegar aqui)


def compute_stats(cache_dir, features, batch_size=256, num_workers=8):
    ds = WaymoMotionDatasetAgentCentric(cache_dir, agent_centric=True,
                                        features=features)
    n = len(ds)
    if n == 0:
        raise SystemExit(f"[ERROR] cache vazio: {cache_dir}")
    n_features = ds.n_features
    assert n_features == 6, (
        f"esperado 6 canais (x,y,sin,cos,vx,vy), dataset deu {n_features}. "
        "Padronização do V0 pressupõe o input rico."
    )

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=False)

    # Acumuladores em float64 para estabilidade (soma de muitos quadrados).
    ch_sum = torch.zeros(n_features, dtype=torch.float64)
    ch_sqsum = torch.zeros(n_features, dtype=torch.float64)
    count = 0   # número de (exemplo, frame) somados por canal
    n_examples = 0

    for history, _future, _mask, _atype in loader:
        # history: [B, 11, 6]. Achata (B, 11) -> soma sobre exemplos e frames.
        h = history.to(torch.float64)
        B, T, C = h.shape
        flat = h.reshape(B * T, C)
        ch_sum += flat.sum(dim=0)
        ch_sqsum += (flat * flat).sum(dim=0)
        count += B * T
        n_examples += B

    mean = ch_sum / count
    var = ch_sqsum / count - mean * mean
    var = torch.clamp(var, min=0.0)          # numérico: var >= 0
    std = torch.sqrt(var).clamp(min=STD_FLOOR)

    mean = mean.numpy().astype(np.float64)
    std = std.numpy().astype(np.float64)

    # --- Passthrough sin/cos: canais 2,3 -> identidade -----------------------
    for c in PASSTHROUGH_CHANNELS:
        mean[c] = 0.0
        std[c] = 1.0

    return mean, std, n_examples, count


def main():
    ap = argparse.ArgumentParser(
        description="Computa stats de feature congeladas (V0-std) do cache_train")
    ap.add_argument("--cache", default=TRAIN_CACHE, help=f"default {TRAIN_CACHE}")
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"default {DEFAULT_OUT}")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()

    print("=" * 78)
    print("[*] Computando stats de feature (V0-std) -- frame AGENTE, cache_train")
    print(f"    cache    = {args.cache}")
    print(f"    features = {FEATURES}  (heading -> sin,cos)")
    print(f"    passthrough (identidade) nos canais {PASSTHROUGH_CHANNELS} = sin,cos")
    print("=" * 78)

    mean, std, n_examples, n_frames = compute_stats(
        args.cache, FEATURES, args.batch_size, args.num_workers)

    ch_names = ("x", "y", "sin", "cos", "vx", "vy")
    print(f"[OK] {n_examples} exemplos | {n_frames} pares (exemplo,frame) somados")
    print(f"{'canal':>6} {'mean':>12} {'std':>12}")
    for i, nm in enumerate(ch_names):
        tag = "  (passthrough)" if i in PASSTHROUGH_CHANNELS else ""
        print(f"{nm:>6} {mean[i]:>12.5f} {std[i]:>12.5f}{tag}")

    blob = {
        "mean": mean,
        "std": std,
        "features": list(FEATURES),
        "channel_order": list(ch_names),
        "agent_centric": True,
        "n_examples": int(n_examples),
        "n_frames": int(n_frames),
        "passthrough": list(PASSTHROUGH_CHANNELS),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, blob, allow_pickle=True)
    print(f"\n[SUCCESS] stats salvas em: {args.out}")
    print("          Use no treino: python3 -m src.motion.train_vectorized "
          "--agent-centric --cls-weight 20 --seed 0 --standardize")


if __name__ == "__main__":
    main()
