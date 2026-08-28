# Waymo Motion Prediction — Relatório destilado (etapa de aquecimento)

*Última atualização: 28 de agosto de 2026*

> **Sobre este documento.** Relatório generalizado da etapa de Motion, destilado a
> partir do registro técnico completo (`DOCUMENTACAO_PROJETO.md`, §0–§0.14) e do
> mega-run 8-seed. É o `docs/waymo_motion.md` previsto no README. Todos os números
> vêm de arquivos reais (grades consolidadas + `report.csv` do mega-run); onde algo
> depende de material ainda não dobrado na doc, está marcado como **[lacuna]**.

---

## 1. Papel no mestrado e tese metodológica

O foco do mestrado é **Perception** (comparação cross-dataset). O trabalho começou
pelo **Waymo Open Motion Dataset** como **aquecimento/hands-on** — setup de ambiente,
stack (TF/PyTorch/Docker), métricas oficiais — mas os resultados renderam o suficiente
para virar **entregável próprio possível** (paper de Motion).

A tese metodológica que distingue o trabalho: comparações na literatura frequentemente
misturam **dados + treino + arquitetura** de uma vez, impedindo atribuir um ganho a uma
causa. Aqui a disciplina é uma **escada controlada** que **isola uma variável
arquitetural por degrau**, avaliando cada um contra as **métricas oficiais** (minADE,
minFDE, MissRate, mAP a 3/5/8 s) e, como eixo secundário, contra o **custo** (tempo,
energia/potência de GPU). Regra de rigor: **N=8 seeds fixadas a priori = veredito**
(mean±std; nunca parar em número favorável).

## 2. Setup

- **Hardware:** RTX 5060 Ti 16 GB (desktop). Dois containers Docker: **GPU** (PyTorch,
  treino/inferência) e **CPU** (TensorFlow 2.11 + `waymo-open-dataset`, métricas
  oficiais) — separação necessária pela incompatibilidade CUDA/TF antigo × GPU moderna.
- **Dados:** WOMD v1.3.1. Partição fixa e **igual para todos os degraus**: train shards
  0–5 (`cache_train` ≈ 2972 cenários / ~13,2k exemplos-agente), val shards 0–2 (879
  cenários / ~3,8k). tfrecords brutos no HDD; cache `.npy` no SSD SATA.
- **Métricas:** oficiais do desafio, nos horizontes `_5`/`_9`/`_15` (≈ 3/5/8 s). Nunca
  `Val(top-1)`. "Overall" = média das 9 breakdowns (3 tipos × 3 horizontes).

## 3. Trajetória do baseline à escada

| Etapa | Data | O que mudou (1 variável) | Achado |
|-------|------|--------------------------|--------|
| **M1** | 16/jul | MLP unimodal, `random_split`, 3 shards | Pipeline mecânico; baseline ingênuo (MissRate ~92–98 %, §8 da doc) |
| **M2** | — | Splits **oficiais** + multimodal K=6 (WTA), sweep `cls_weight` | Contrato de métricas oficiais fechado |
| **Higiene (Parte A)** | 19/ago | Seeds + agregação (5→8 seeds) | flatten > sequential **robusto**; `cls_weight` = ruído |
| **Item 4** | 20/ago | Normalização **agente-cêntrica** (input rico 6-ch) | **Ganho grande**: agente >> SDC (H1 confirmada); fecha o gap flatten×GRU |
| **V0** | 21/ago | Encoder **vetorizado** (VectorNet) no lugar do flatten | Padronização ajuda pouco; **MissRate não move → falta contexto** |
| **V1** | 22–24/ago | **Social** (cross-attention sobre vizinhos) | Move a trajetória (Δ = 1.43 m real) mas **não fecha a cobertura longa** |
| **V2** | 26/ago build · 27–28 veredito | **Mapa** (cross-attention sobre roadgraph) | **Fecha o MissRate longo — hipótese CONFIRMADA** |
| **V3** | 27–28/ago | **Topologia de lane** (GNN de conectividade) | **Não** adiciona sobre o mapa — negativo limpo reportável |

Marcos pré-escada (agente-cêntrico, Item 4), overall 8 seeds: **flatten-agente**
minADE 1.394 ± 0.029, MissRate 0.576 ± 0.018 vs **flatten-SDC** 2.068 / 0.725 — a
normalização agente-cêntrica foi a maior alavanca isolada da etapa.

## 4. Resultados consolidados da escada (mega-run 8 seeds)

Braço `agent`, variante `raw`, N=8. O V0-raw reproduz o §0.10 **exato** (validação de
consistência do mega-run).

**Overall (média das 9 breakdowns), mean ± std:**

| degrau | minADE | minFDE | MissRate | mAP |
|--------|--------|--------|----------|-----|
| V0 vectorized | 1.468 ± 0.042 | 3.409 ± 0.085 | 0.615 ± 0.024 | 0.119 ± 0.034 |
| V1 social | 1.426 ± 0.039 | 3.178 ± 0.083 | 0.606 ± 0.029 | 0.116 ± 0.031 |
| V2 map | **1.251 ± 0.026** | **2.667 ± 0.075** | **0.505 ± 0.016** | **0.139 ± 0.029** |
| V3 lane_topo | 1.273 ± 0.041 | 2.700 ± 0.096 | 0.532 ± 0.036 | 0.127 ± 0.025 |

**MissRate a 8 s (`_15`) — métrica-hipótese:** V0 0.738 · V1 0.748 · **V2 0.649** ·
V3 0.673.

**Vereditos (teste pareado por seed, 8 s):**
- **V2 (mapa) — CONFIRMADO.** vs V1: MissRate −0.099 (p = 8.3e-6), minADE −0.34,
  minFDE −0.97, mAP +0.033 — melhora tudo. O mapa é o componente que fecha a cobertura.
- **V3 (topologia) — NÃO confirmado.** vs V2: nada significativo (MissRate +0.025,
  p = 0.11; mAP −0.015, p = 0.32). O ganho de mAP do seed0 era ruído; as distribuições
  de V2 e V3 se sobrepõem.

**Custo (do próprio mega-run):** util GPU 6.6 → 17 % ao longo da escada → **tudo
upstream-bound** (dataload/IO), não compute. V2/V3 custam ~6–10× a energia de V0/V1;
V3 custa ≈ V2 por ganho nulo.

## 5. Achados que "vendem" o trabalho

1. **Atribuição limpa por construção.** Cada degrau muda uma coisa (siblings de código
   com o encoder anterior importado intacto), então cada Δ é atribuível.
2. **O componente que fecha a cobertura é o mapa** — não o social (V1) nem a topologia
   de lane (V3). Resultado positivo forte e localizado.
3. **Negativos limpos e reportáveis** (raros na literatura): social sozinho não fecha
   cobertura longa; topologia de lane sobre o mapa não traz ganho mensurável. Blindados
   por N=8 (dispersão de seeds), não por 1 rodada.
4. **Eixo de custo explícito:** ganho por energia/tempo/potência, não só acurácia.

## 6. Limitações conhecidas (honestidade metodológica)

- **6 shards de treino = fração mínima** do WOMD (~1000 shards) → métricas **absolutas
  sub-treinadas**. A **curva de data-scaling** (métrica × tamanho, arquitetura fixa) é
  entregável próprio, ainda pendente.
- **Baseline de velocidade constante** (piso trivial) ainda não rodado — pré-requisito
  antes de reportar qualquer número absoluto.
- **Gate B.2 do V3** (redução a V2 via `torch.allclose`) não confirmado → V3 tem
  veredito de número mas não é degrau formalmente fechado quanto à atribuição.
- Simplificações herdadas (§9 da doc): `MAX_AGENTS=12` vs 128 slots; passado inválido
  zerado sem flag de ausência; geometria de mapa pura (sem tipo/tangente no V2 base).
- **[lacuna]** Detalhe fino do **mega-run smoke 26/ago** e da **Etapa B.1** (loader de
  topologia, contrato da 10-tupla) vive em handoffs ainda não dobrados na doc como
  `§0.X` — subir esses handovers fecharia a lacuna.

## 7. Estado e próximos passos

- **Infra: fechada (100 %)** — validada em produção pelo mega-run (6 modelos × 8 seeds,
  Guard 2 passou, run-id/rename/consolidadores OK). Pronta para rodar sem novo trabalho
  de pipeline.
- **Fase atual (teórica):** mapear os métodos do código aos nomes canônicos da
  literatura (VectorNet para V0, cross-attention social para V1, map-aware para V2,
  LaneGCN para V3, MultiPath para o multimodal) e ler o leaderboard oficial do WOMD —
  base do Related Work e da comparação de diferenças finas.
- **Volta aos resultados (depois):** fechar B.2 (V3 limpo); rodar velocidade constante;
  curva de data-scaling (com diagnóstico de gargalo antes, já que é IO-bound); decidir
  se multimodal/sequential entram como métodos de comparação ou ficam como legado.

---

### Nota de confiabilidade

Este relatório é **confiável por construção**: números vindos de `DOCUMENTACAO_PROJETO.md`
(§0–§0.14) e do `report.csv` do mega-run, ambos arquivos reais reconferidos. Onde há
**[lacuna]**, é porque a fonte está em handoffs não anexados nesta sessão — nada foi
reconstruído de memória. Subir esses handovers eleva as `[lacuna]` de "esqueleto" para
"completo".
