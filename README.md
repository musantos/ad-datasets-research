# ad-datasets-research

Research repository for evaluating autonomous driving **perception**
models across multiple datasets (Waymo, ZOD, A2D2, etc.). Focus on
benchmarking, cross-dataset comparison, and analysis of model performance
under different data distributions.

## Escopo e trajetória do projeto

O foco principal deste mestrado é **Perception** — comparar métodos e
desempenho de modelos entre datasets diferentes de condução autônoma, em
vez de otimizar um único modelo para um único dataset.

O trabalho começou pelo **Waymo Open Motion Dataset** como etapa de
**aquecimento/hands-on**: setup de ambiente, aprendizado da stack
(TensorFlow/PyTorch, Docker, métricas oficiais), e familiarização geral
com esse tipo de dado antes de migrar para Perception. Não é o objeto
central da pesquisa, mas os resultados obtidos aí são relevantes o
suficiente para render, possivelmente, um artigo próprio sobre Motion
Prediction — a decidir conforme o projeto avança.

## Status atual

🟢 **Fase de Perception em andamento (foco principal).** Estudo comparativo
**A×B controlado** em detecção 2D (câmera frontal única, Waymo Perception):
**pretrained (FrozenBN) × from-scratch (GroupNorm)** com o mesmo detector
Faster R-CNN, mesma régua (split/classes/score-thresh), distribuição sobre
seeds (nunca best-vs-best). Veredito no ponto n48/s8 (10 seeds): o pretrained
**vence de forma disjunta** (AP 0.1545 ± 0.0041 vs 0.0815 ± 0.0048; +0.073 AP,
+90%; p ≈ 5e-18). Em curso: **escada de data-scaling** AP×N sobre
**6 → 12 → 24 → 48 → 96** segmentos (`scripts/run_perception_pipeline.sh`,
10 seeds, E=32), com pipeline automático de treino → inferência → plots
(curves/PR/confusão/scaling) → vídeo comparativo, validado ponta-a-ponta.
Registro técnico completo em
[`docs/DOCUMENTACAO_PROJETO.md`](docs/DOCUMENTACAO_PROJETO.md) **§0.16**.

🧊 **Motion concluído e congelado** (etapa de aquecimento). Escada experimental
`MLP → VectorNet (V0) → social (V1) → mapa (V2) → topologia de lane (V3)`,
veredito **N=8**: **V2 (mapa) confirmado** (fecha o MissRate longo), **V3
(lane topo) não confirmado** (ganho de seed0 não sobreviveu a N=8). Data-scaling
de Motion **6→12→24** concluído: regime **data-limited** (nenhum método satura),
com a inversão `+lane_topo > +map` sobrevivendo e alargando. Registro em
§0.1–§0.15. *(Relatório destilado `docs/waymo_motion.md` planejado, caso renda
um paper próprio de Motion.)*

## Datasets

| Dataset | Papel no projeto | Status |
|---|---|---|
| Waymo Open Dataset (Motion) | Aquecimento / hands-on | Concluído/congelado — data-scaling 6→12→24 (N=8) |
| Waymo Open Dataset (Perception) | Foco principal | **Em andamento** — A×B transfer + data-scaling 6→96 |
| ZOD (Zenseact Open Dataset) | Benchmarking cross-dataset | Planejado |
| A2D2 (Audi Autonomous Driving Dataset) | Benchmarking cross-dataset | Planejado |

## Estrutura do repositório

```
.
├── docs/
│   ├── DOCUMENTACAO_PROJETO.md  # registro técnico (Motion §0.1–0.15, Perception §0.16)
│   └── waymo_motion.md          # (planejado) relatório destilado de Motion
├── docker/
│   ├── waymo-metrics/        # container CPU (cpu_env): leitura de dados + métricas oficiais (TF)
│   └── training-v1/          # container GPU (gpu_env): treino/inferência (PyTorch)
├── scripts/
│   └── run_perception_pipeline.sh  # orquestrador do ladder de Perception (train+infer+plots+vídeo)
├── src/
│   ├── core/                 # decodificação e pré-processamento (container de métricas)
│   ├── motion/               # modelos, treino, inferência e validação (Motion — aquecimento)
│   └── perception/           # FOCO PRINCIPAL
│       ├── waymo/tools/      # preprocessor.py (extração PNG+manifest), dataset.py
│       ├── models/           # train_vision_model.py, faster_rcnn.py
│       └── tools/det2d/      # inference_frcnn, dump_predictions, compare_video,
│                             #   plot_curves, plot_pr, plot_confusion, plot_scaling, eval_ap
└── tutorial_motion_original.ipynb   # tutorial oficial do Waymo (referência da etapa de Motion)
```

## Ambiente

O projeto usa **dois containers Docker separados**, pela incompatibilidade
entre as versões antigas de TensorFlow/CUDA exigidas pelas bibliotecas
oficiais de alguns datasets (ex: `waymo-open-dataset`) e hardware GPU
moderno:

- **Métricas / extração (CPU, `cpu_env`):** leitura de tfrecords brutos,
  pré-processamento, cálculo de métricas oficiais. TensorFlow 2.11 (CPU-only).
- **Treino / inferência (GPU, `gpu_env`):** treino e inferência dos modelos.
  PyTorch, CUDA. Perception roda inteiro aqui (o detector não precisa de TF).

Ver `docker/` para os Dockerfiles. A mesma estrutura de dois ambientes se
repete (ajustada conforme necessário) para os próximos datasets.

## Como reproduzir

**Perception (fase atual) — ladder de data-scaling 2D:**
```bash
# 1. Extração do cache (cpu_env) — PNG + manifest.json, stride 8, câmera frontal.
#    [:N] no split congelado => n96 é superconjunto aninhado de n48 (núcleo idêntico).
docker exec -it cpu_env python /workspace/src/perception/waymo/tools/preprocessor.py \
  --split train --n-segments 96 --stride 8 --cameras f
docker exec -it cpu_env python /workspace/src/perception/waymo/tools/preprocessor.py \
  --split val   --n-segments 12 --stride 8 --cameras f   # val FIXO (não re-extrair)

# 2. Ladder completo (host; faz docker exec no gpu_env). Smoke primeiro:
SMOKE=1 ./scripts/run_perception_pipeline.sh
# Run real (10 seeds, E=32, steps 6/12/24/48/96, ~105 h):
NTFY_TOPIC=<topico> nohup ./scripts/run_perception_pipeline.sh > ladder_$(date +%F_%H%M).log 2>&1 &
```

**Motion (etapa de aquecimento, congelada):**
```bash
python3 scripts/run_grid_gpu.py      --model <m> --seeds 0-7            # GPU — gera run-id
python3 scripts/run_grid_validate.py --model <m> --run-id <id> --seeds 0-7  # CPU/TF
python3 scripts/consolidate_seeds.py --model <m> --run-id <id> --seeds 0-7  # GPU
python3 scripts/consolidate_gpu.py                 --run-id <id>            # GPU
python3 scripts/report.py                          --run-id <id>            # GPU
```

Detalhes completos (formato de dados, protocolo, vereditos, limitações) em
[`docs/DOCUMENTACAO_PROJETO.md`](docs/DOCUMENTACAO_PROJETO.md).

## Próximos passos

- **Perception:** rodar o ladder completo 6→96 (10 seeds, E=32); curva AP×N +
  tabela; análise PR/confusão/vídeo nos pontos representativos. Depois: eixo
  "generalista × especialista" (mesma máquina de comparação, trocando os globs
  dos arms, desde que os dois usem a mesma régua).
- Expandir Perception para **ZOD** e **A2D2** (benchmarking cross-dataset — o
  objetivo central da dissertação).
- Motion: possível paper específico sobre o baseline (a decidir).