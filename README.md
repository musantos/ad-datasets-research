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

🟢 **Pipeline de Motion funcional de ponta a ponta** no Waymo Open Motion
Dataset — da leitura dos dados brutos à validação com as métricas oficiais
do desafio (minADE, minFDE, Miss Rate, mAP). O estudo segue uma **escada
experimental controlada** que isola **uma variável arquitetural por etapa**
(`MLP → VectorNet (V0) → contexto social (V1) → mapa (V2) → topologia de
lane (V3)`), permitindo atribuir cada ganho a um componente específico — ao
contrário de comparações que misturam dados, treino e arquitetura de uma vez.
Cada degrau é avaliado contra as métricas oficiais da Waymo e, como eixo
secundário, contra o custo computacional (tempo, potência de GPU). **Estado:** Motion **concluído e congelado** — infraestrutura de treino fechada; V0–V3 com veredito **N=8** (mega-run de 8 seeds). V2 (mapa) **confirmado** (fecha o MissRate longo); V3 (topologia de lane) **não confirmado** — o ganho de seed0 não sobreviveu a N=8 (negativo limpo). Estudo de **data-scaling 6→12→24** concluído: regime **data-limited** (nenhum método satura), com a inversão `+lane_topo > +map` sobrevivendo e alargando. Pendências não-bloqueantes: manifesto de validation e baseline constant-velocity. Registro técnico completo
(setup, bugs corrigidos, resultados por seed, limitações) em
[`docs/DOCUMENTACAO_PROJETO.md`](docs/DOCUMENTACAO_PROJETO.md). *(Um
relatório destilado `docs/waymo_motion.md` está planejado, caso a etapa
renda um paper próprio de Motion.)*

🔜 **Fase atual:** foco migrado para **Perception** (Motion congelado, aquecimento concluído). Estudo comparativo cross-dataset em Waymo/ZOD em definição; A2D2 a confirmar.

## Datasets

| Dataset | Papel no projeto | Status |
|---|---|---|
| Waymo Open Dataset (Motion) | Aquecimento / hands-on | Concluído/congelado — data-scaling 6→12→24 (N=8) |
| Waymo Open Dataset (Perception) | Foco principal | Planejado |
| ZOD (Zenseact Open Dataset) | Benchmarking cross-dataset | Planejado |
| A2D2 (Audi Autonomous Driving Dataset) | Benchmarking cross-dataset | Planejado |

## Estrutura do repositório

```
.
├── docs/
│   ├── DOCUMENTACAO_PROJETO.md  # registro técnico da etapa de Motion (setup, bugs, resultados por seed)
│   └── waymo_motion.md          # (planejado) relatório destilado de Motion
├── docker/
│   ├── waymo-metrics/        # container CPU: leitura de dados + métricas oficiais (TF)
│   └── training-v1/          # container GPU: treino de modelos (PyTorch)
├── src/
│   ├── core/                 # decodificação e pré-processamento de dados (roda no container de métricas)
│   ├── motion/                # modelos, treino, inferência e validação (Motion Prediction — aquecimento)
│   └── perception/            # foco principal do mestrado (planejado)
└── tutorial_motion_original.ipynb   # tutorial oficial do Waymo, usado como referência na etapa de Motion
```

## Ambiente

O projeto usa **dois containers Docker separados**, pela incompatibilidade
entre as versões antigas de TensorFlow/CUDA exigidas pelas bibliotecas
oficiais de alguns datasets (ex: `waymo-open-dataset`) e hardware GPU
moderno:

- **Métricas (CPU):** leitura de dados brutos, pré-processamento,
  cálculo de métricas oficiais. TensorFlow 2.11 (CPU-only).
- **Treino (GPU):** treinamento e inferência dos modelos. PyTorch,
  CUDA 13.0.

Ver `docker/` para os Dockerfiles de cada ambiente. A mesma estrutura de
dois ambientes deve se repetir (ajustada conforme necessário) para os
próximos datasets.

## Como reproduzir (Waymo Motion — etapa de aquecimento)

```bash
# Container de Métricas (CPU)
python3 -m src.core.waymo_preprocessor

# Container de Treino (GPU)
python3 -m src.motion.train_motionv4
python3 -m src.motion.run_inference

# Container de Métricas (CPU)
python3 -m src.motion.validate_motion_official
```

Detalhes completos (formato de dados, bugs corrigidos, resultados de
validação, limitações conhecidas) em
[`docs/DOCUMENTACAO_PROJETO.md`](docs/DOCUMENTACAO_PROJETO.md).

## Próximos passos

- Motion concluído/congelado (possível paper específico sobre esse baseline — a decidir).
- Migrar foco de pesquisa para **Perception** no Waymo Open Dataset.
- Expandir para ZOD e A2D2, com pipeline de benchmarking e comparação
  cross-dataset como objetivo central da dissertação.