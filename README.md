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

🟢 **Etapa de Motion concluída (aquecimento):** pipeline de Motion
Prediction funcional de ponta a ponta no Waymo Open Motion Dataset — desde
a leitura dos dados brutos até a validação com as métricas oficiais do
desafio (minADE, minFDE, Miss Rate, mAP). A espinha de experimentos
controlados `MLP → encoder sequencial → normalização agente-cêntrica` está
fechada e validada com múltiplas seeds; próximo degrau é o modelo
vetorizado com contexto de mapa. Registro técnico completo (setup, bugs
corrigidos, resultados por seed, limitações) em
[`docs/DOCUMENTACAO_PROJETO.md`](docs/DOCUMENTACAO_PROJETO.md). *(Um
relatório destilado `docs/waymo_motion.md` está planejado, caso a etapa
renda um paper próprio de Motion.)*

🔜 **Próxima fase:** migração de foco para Perception, com expansão para
outros datasets (ZOD, A2D2) visando benchmarking e comparação
cross-dataset.

## Datasets

| Dataset | Papel no projeto | Status |
|---|---|---|
| Waymo Open Dataset (Motion) | Aquecimento / hands-on | Baseline validado com métricas oficiais |
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

- Concluir/documentar a etapa de Motion (possível paper específico sobre
  esse baseline).
- Migrar foco de pesquisa para **Perception** no Waymo Open Dataset.
- Expandir para ZOD e A2D2, com pipeline de benchmarking e comparação
  cross-dataset como objetivo central da dissertação.