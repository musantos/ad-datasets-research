# Projeto de Mestrado — Motion Prediction (Waymo Open Dataset)
### Documento de contexto — Milestone 2 (split oficial + experimento multimodal)
Última atualização: 24 de julho de 2026

---

## 1. Objetivo da pesquisa

Projeto de mestrado em datasets automotivos / autonomous driving. Foco inicial:
**Waymo Open Motion Dataset**, explorando primeiro Motion Prediction como
familiarização técnica antes de migrar o foco principal da tese para
**Perception**. Comparação futura planejada com outros datasets (NuScenes, ZOD).

O objetivo não é propor um método novo, e sim demonstrar domínio teórico e
prático do problema — o que torna a **comparabilidade entre experimentos** o
requisito metodológico central, mais importante que o valor absoluto das
métricas.

## 2. Hardware

- **Laptop:** Ryzen 5 5600H, 32GB RAM, RTX 3060 6GB (dual boot, ~500GB Linux).
- **Desktop (PC principal do projeto):** Ryzen 5 8400F, 32GB RAM, **RTX 5060 Ti
  16GB** (arquitetura Blackwell, `sm_120`). Armazenamento: 500GB NVMe (SO),
  1TB SSD SATA (`/home`), 2x3TB WD Red + 1x16TB Toshiba N300 (datalake/cold
  storage).

**Divisão de armazenamento (deliberada):** o dataset bruto (ordem de TB) fica
no HDD porque é lido sequencialmente e uma única vez, no pré-processamento.
O cache `.npy` fica no SSD (`/home`) porque o `Dataset` faz acesso **aleatório**
a ele a cada `__getitem__`, milhares de vezes por época — em HDD isso viraria
o gargalo do treino.

## 3. Arquitetura de ambientes (por que dois containers)

O `waymo-open-dataset` tem binários C++ (`py_metrics_ops`) compilados para
TensorFlow/CUDA antigos, incompatíveis com a RTX 5060 Ti (que exige CUDA 12+).
Solução adotada: **dois containers Docker separados**, conectados só pela
pasta de cache compartilhada.

| Container | Função | Stack | Hardware |
|---|---|---|---|
| **Métricas** | Ler TFRecords, pré-processar, calcular métricas oficiais | Docker, Python 3.8, TF-CPU 2.11.0, `waymo-open-dataset-tf-2-11-0==1.6.1` | CPU only |
| **Treino** | Treinar o modelo, gerar predições | Docker, PyTorch 2.9.0, CUDA 13.0 | GPU (RTX 5060 Ti) |

Ponte entre os dois: pastas montadas como volume nos dois containers
(`datasets/waymo/cache_train`, `cache_val` e `predictions/`).

**Invocação típica do container de treino:**
```bash
docker run -it --rm --gpus all \
    -v ~/autonomous_drive:/workspace \
    -v /data/.disks:/data \
    --name train-gpu-container training-v1
```
Note o mount `-v /data/.disks:/data`: no host o dataset está em
`/data/.disks/hdd3a/...`, mas **dentro do container** o caminho é
`/data/hdd3a/...` (sem o `.disks`). É por isso que o `DATA_DIR` do
preprocessor não tem o ponto — não é erro.

### 3.1. Splits do dataset bruto

A raiz do dataset (dentro do container) é:
`/data/hdd3a/waymo_motion/waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario/`

| Pasta | Shards | Uso no projeto |
|---|---|---|
| `training/` | 1000 | **treino** (shards 0,1,2 processados) |
| `validation/` | 150 | **validação e métricas** (shards 0,1,2 processados) |
| `testing/` | 150 | não usado — sem futuro anotado (só leaderboard) |
| `training_20s/` | 1000 | não usado — versão de 20s |
| `*_interactive/` | 150 | não usado — tarefa de Interaction Prediction (futuro) |

Existe também `tf_example/` (formato pré-processado por agente) e
`visualization/` (HTML). O projeto usa o formato `scenario/`, que é o proto
`Scenario` bruto lido pelo `waymo_decoder.py`.

## 4. Estrutura de arquivos do pipeline (árvore real — 24/07)

```
~/autonomous_drive/
├── datasets/waymo/
│   ├── cache_train/         # .npy do split 'training'   (shards 0,1,2)
│   ├── cache_val/           # .npy do split 'validation' (shards 0,1,2)
│   ├── motion -> /data/.disks/hdd3a/waymo_motion/.../uncompressed/  (symlink)
│   └── predictions/
│       ├── baseline/        # predições do SimpleTrajectoryPredictor
│       └── multimodal/      # predições do MultimodalTrajectoryPredictor
├── docker/
│   ├── waymo-metrics/Dockerfile     # container de METRICAS (CPU)
│   ├── training-v1/Dockerfile       # container de TREINO (GPU)
│   └── _OLD_waymo-legacy/Dockerfile # descartado
├── experiments/checkpoints/
│   ├── baseline_3shards/    # HISTÓRICO — split caseiro, ver seção 8.1
│   ├── baseline_oficial/    # baseline com split oficial
│   └── multimodal/          # modelo multimodal K=6
├── src/
│   ├── core/
│   │   ├── waymo_decoder.py             # decodifica Scenario proto bruto
│   │   ├── waymo_preprocessor.py        # roda no container de METRICAS (CPU)
│   │   └── waymo_pytorch_dataset.py     # Dataset PyTorch (confirmado: src/core/)
│   └── motion/
│       ├── simple_model.py              # SimpleTrajectoryPredictor (MLP unimodal)
│       ├── train_motion.py              # treino do baseline           (GPU)
│       ├── run_inference.py             # inferência do baseline       (GPU)
│       ├── multimodal_model.py          # MultimodalTrajectoryPredictor (K=6)
│       ├── train_multimodal.py          # treino do multimodal          (GPU)
│       ├── run_inference_multimodal.py  # inferência do multimodal      (GPU)
│       └── validate_motion_official.py  # métricas oficiais             (CPU)
└── tutorial_motion_original.ipynb   # fonte do config oficial e da fórmula de downsample
```

**Limpeza realizada (julho/2026):** o projeto acumulava versões paralelas dos
mesmos scripts (`_old`, `v1`, `v2`, `v3`, `v4`, uma `(copy)`). Todas foram
removidas e o repositório foi publicado no GitHub. **Convenção adotada
daqui em diante: sufixo por experimento, nunca por versão** — `train_motion`
e `train_multimodal`, não `train_v5`. Versões ficam no histórico do git.

**Import consistente:** todos os módulos usam `from src.core...` / `from
src.motion...`. Scripts devem ser executados como módulo a partir de
`/workspace` dentro do container (precisa de `__init__.py`, mesmo vazio, em
`src/`, `src/core/`, `src/motion/`).

## 4.1. Sequência completa de execução do pipeline

```bash
# --- Container de METRICAS (CPU) ---
# Pré-processamento. O --split escolhe training ou validation.
python3 -m src.core.waymo_preprocessor --split training   --shards 0,1,2
python3 -m src.core.waymo_preprocessor --split validation --shards 0,1,2

# --- Container de TREINO (GPU) ---
python3 -m src.motion.train_motion                 # baseline unimodal
python3 -m src.motion.run_inference

python3 -m src.motion.train_multimodal             # multimodal K=6
python3 -m src.motion.run_inference_multimodal

# --- Container de METRICAS (CPU) ---
python3 -m src.motion.validate_motion_official \
    --pred-dir /workspace/datasets/waymo/predictions/baseline
```

**Atenção:** retreinar sobrescreve os checkpoints do experimento
correspondente. Cada experimento tem sua própria pasta em
`experiments/checkpoints/` justamente para que isso nunca destrua o
resultado de outro.

## 5. Histórico de bugs encontrados e corrigidos (herdados do trabalho com Manus)

O ambiente Docker (infra) sempre esteve correto e validado. Os bugs estavam
todos no código Python de dentro dos containers:

1. **Nomes de import incompatíveis:** `train_motion.py` importava
   `WaymoDataset`/`SimpleModel`, classes que não existiam (`waymo_pytorch_dataset.py`
   define `WaymoMotionDataset`; `simple_model.py` define
   `SimpleTrajectoryPredictor`).
2. **Interface incompatível:** o Dataset devolvia uma tupla, o script de
   treino esperava um dicionário (`data['history']`).
3. **Caminho de cache divergente:** `train_motion.py` apontava para um
   caminho diferente de onde `waymo_preprocessor.py` salvava.
4. **Identificação errada do agente principal:** o código pegava o
   **primeiro agente da lista** como se fosse o SDC (carro autônomo), sem
   checar de fato. Corrigido com uma flag explícita `is_sdc`, calculada por
   **índice** de posição em `scenario.tracks` (não por `track.id`, que é
   um identificador de objeto, não uma posição).
5. **Confusão conceitual mais importante:** a tarefa oficial do Waymo Motion
   é prever o futuro de **outros agentes** ao redor do SDC (carros,
   pedestres, ciclistas) — não do próprio SDC. Essa informação vem do campo
   `scenario.tracks_to_predict`, nunca lido antes. Corrigido: cada agente
   agora carrega uma flag `is_target`.
6. **Frames inválidos contaminando o treino:** agentes fora de visibilidade
   em parte do tempo têm `x=0, y=0` nesses frames (coordenadas absolutas).
   Depois de subtrair a origem do SDC, isso virava valores gigantes e sem
   sentido, inflando a loss para a casa de milhões. Corrigido com máscara:
   zera os frames inválidos do **passado** (entrada do modelo) e ignora os
   frames inválidos do **futuro** na loss (loss mascarada).

### 5.1. Falha metodológica corrigida em 24/07 (não é bug de código)

O `run_inference.py` original varria **todo** o cache para gerar predições,
mas o split treino/validação acontecia dentro do `train_motion.py`, em
memória, sem deixar rastro em disco. Consequência: **as métricas oficiais da
seção 8.1 foram calculadas sobre dados que incluíam os ~80% usados no
treino.** Elas mediam memorização junto com generalização.

Havia ainda um problema mais sutil de reprodutibilidade: o
`WaymoMotionDataset` monta a lista de amostras a partir de `os.listdir()`,
cuja ordem **não é garantida**. O `random_split` com `manual_seed(42)`
embaralha índices sobre essa lista — ou seja, a reprodutibilidade que a seed
parecia garantir era ilusória, e dois treinos em momentos diferentes podiam
usar splits diferentes sem nenhum sinal visível.

**Correção adotada: usar o split oficial do Waymo.** Treino vem de
`scenario/training`, validação de `scenario/validation`. São shards
distintos com cenários distintos — o vazamento se torna impossível por
construção, a discussão "split por agente vs por cenário" desaparece, e os
números passam a ser diretamente comparáveis com os papers da área (que
reportam sobre `validation`, já que `testing` não tem futuro anotado).

## 6. Formato de dado no cache (`waymo_preprocessor.py`)

Cada `.npy` salvo em `cache_train/<scenario_id>.npy` ou
`cache_val/<scenario_id>.npy` contém um dicionário:
```python
{
  'scenario_id': str,
  'agents': [
    {
      'id': int,                    # id do objeto/track
      'type': int,                  # 1=Veiculo, 2=Pedestre, 3=Ciclista (confirmado)
      'trajectory': np.array [91,2],   # x,y relativos ao SDC, rotacionados
      'full_state': np.array [91,7],   # x,y,length,width,heading,vx,vy (idem)
      'mask': np.array [91] bool,      # frame valido ou nao
      'is_sdc': bool,
      'is_target': bool,            # agente que deve ser previsto (tracks_to_predict)
    },
    ...
  ]
}
```
Origem/rotação: sempre relativa à posição e heading do SDC no frame 10
(fim do passado / início do presente). 91 frames = 10 passado + 1 presente +
80 futuro (10Hz).

**Nota:** a normalização é *SDC-cêntrica*, não *agente-cêntrica*. Um agente
distante do carro autônomo ainda tem coordenadas na casa das dezenas de
metros, e cada agente enxerga o próprio movimento num referencial diferente.
Centrar e rotacionar no próprio agente-alvo é prática padrão na área e fica
como melhoria futura (ver seção 11) — mas é ajuste fino, não o fator
dominante nas métricas atuais.

## 7. Modelos e treino

### 7.1. Baseline — `SimpleTrajectoryPredictor` (unimodal)
- MLP de 2 camadas escondidas (256 neurônios), entrada 22 valores
  (11 frames × x,y), saída 160 valores (80 frames × x,y). ~113k parâmetros.
- **Uma amostra de treino = um agente-alvo específico**, não uma cena
  inteira. Entrada: só a trajetória passada do próprio agente (SEM contexto
  de mapa ou de outros agentes — simplificação deliberada, ver seção 9).
- Loss: MSE mascarada por exemplo (`masked_mse_per_example`).
- Checkpoints em `experiments/checkpoints/baseline_oficial/`.

### 7.2. Experimento 2 — `MultimodalTrajectoryPredictor` (K=6)
- **Mesmo backbone** (2×256) de propósito: a única variável que muda entre
  os dois experimentos é a multimodalidade, para que qualquer diferença nas
  métricas seja atribuível a ela.
- Duas cabeças: trajetórias `[K,80,2]` e scores `[K]` (logits). ~320k parâmetros.
- **Motivação:** a métrica oficial avalia com `max_predictions=6`. O baseline
  unimodal estava sendo avaliado num regime multimodal com uma única
  hipótese. Além disso, motion prediction é ambíguo por natureza — num
  cruzamento, "virar" e "seguir reto" são ambas corretas, e um modelo
  unimodal treinado com MSE aprende a **média** das duas, que não é uma
  trajetória plausível.
- **Loss Winner-Takes-All:** dos K modos, só o mais próximo do ground truth
  recebe gradiente de regressão; os demais ficam livres para cobrir hipóteses
  alternativas. Em paralelo, cross-entropy ensina a cabeça de score a
  identificar qual modo venceu (a métrica mAP usa o score).
- **Diagnóstico obrigatório:** o script reporta a distribuição de vitórias
  entre os modos. Se um modo vencer >90% das vezes, houve **colapso de
  modos** e o modelo virou um unimodal disfarçado — o experimento perde o
  sentido e a saída seria inicialização diversificada ou EWTA.
- Checkpoints em `experiments/checkpoints/multimodal/`.

### 7.3. Como comparar os dois (importante)

| Grandeza | Baseline | Multimodal | Comparável? |
|---|---|---|---|
| Val Loss | `Val Loss` | `Val(top-1)` | **sim** — mesma definição |
| — | — | `Val(melhor modo)` | não — best-of-6, otimista por construção |
| minADE / minFDE / MissRate / mAP | oficial | oficial | **sim** |

O número honesto para "a multimodalidade ajudou?" é o **top-1** contra o
`Val Loss` do baseline. O `melhor modo` mede outra coisa: se as 6 hipóteses
cobrem o futuro real — útil, mas otimista, porque usa o ground truth para
escolher qual modo reportar.

**Assimetria a declarar na dissertação:** o baseline seleciona o melhor
checkpoint pela val loss unimodal; o multimodal, pelo erro do melhor modo
(proxy direto de minADE, que é best-of-6). Cada um otimiza o que sua métrica
premia. É defensável, mas é uma escolha explícita.

### Progressão de escala testada
| Cenários (shards) | Exemplos (agente-alvo) | Observação |
|---|---|---|
| 5 (shard 0, parcial) | 5 (só SDC, versão antiga) | Só validou o pipeline mecânico |
| 250 (shard 0, parcial) | 250 (só SDC, versão antiga) | Overfitting evidente |
| 400 (shard 0 completo, quase) | 1716 (agentes-alvo reais) | Loss estourou p/ milhões (bug de mask), depois corrigido |
| ~1500 (shards 0,1,2 de `training`) | 6836 | Baseline da seção 8.1 (com contaminação) |
| idem + ~1500 de `validation` | a medir | Split oficial — resultados pendentes |

## 8. Validação oficial (métricas Waymo Motion)

Metodologia: inferência no container de treino (`run_inference*.py`) →
tensores montados e métrica calculada no container de métricas
(`validate_motion_official.py`, chama `py_metrics_ops.motion_metrics`).

**Config oficial usado** (extraído de `tutorial_motion_original.ipynb`):
`track_steps_per_second: 10`, `prediction_steps_per_second: 2`,
`track_history_samples: 10`, `track_future_samples: 80`, thresholds de miss
a 3s/5s/8s (`measurement_step: 5, 9, 15`), `max_predictions: 6`.

**Downsample de predição** (10Hz → 2Hz), fórmula oficial:
`prediction[..., (interval-1)::interval, :]` com `interval = 10 // 2 = 5`.

### 8.1. Resultado HISTÓRICO — split caseiro, com contaminação

> ⚠️ **Estes números não são válidos como medida de generalização.** Foram
> obtidos antes da correção descrita na seção 5.1: a inferência rodou sobre
> o cache inteiro, que incluía os dados de treino. Ficam registrados como
> marco de validação **mecânica** do pipeline (a cadeia ponta a ponta
> funciona e as métricas oficiais rodam), não como resultado científico.

Baseline MLP unimodal sem contexto, 1530 trajetórias avaliadas, 3 shards:

| Categoria @ horizonte | minADE | minFDE | MissRate | mAP |
|---|---|---|---|---|
| Veículo @3s | 4.38m | 7.63m | 97.3% | 0.0006 |
| Veículo @5s | 7.62m | 15.76m | 97.6% | 0.0005 |
| Veículo @8s | 12.84m | 29.38m | 97.9% | 0.0006 |
| Pedestre @3s | 2.74m | 3.93m | 95.8% | 0.0019 |
| Pedestre @5s | 3.74m | 5.91m | 94.6% | 0.0031 |
| Pedestre @8s | 4.90m | 8.15m | 92.4% | 0.0052 |
| Ciclista @3s | 3.53m | 5.54m | 94.4% | 0.0026 |
| Ciclista @5s | 5.28m | 9.36m | 90.8% | 0.0068 |
| Ciclista @8s | 7.93m | 16.56m | 94.8% | 0.0022 |

**Ponto em aberto:** a origem do número **1530** nunca foi confirmada. Não
bate com os 6836 agentes-alvo do cache. Hipóteses: truncamento por
`MAX_AGENTS=12` por cenário, inferência parcial/interrompida, ou execução
sobre um subconjunto de shards. Como o experimento será refeito com o split
oficial, o número deixa de importar operacionalmente — fica como nota.

**Leitura:** números típicos de baseline ingênuo (extrapolação sem contexto),
bem distantes do estado da arte (faixa de 1-2m de minADE @8s, MissRate bem
abaixo de 50%).

### 8.2. Resultado com split oficial — A PREENCHER

Baseline e multimodal, ambos treinados em `cache_train` e avaliados em
`cache_val`.

> **Expectativa:** os números do baseline devem **piorar** em relação à
> seção 8.1. Isso não é regressão — é a medida honesta aparecendo pela
> primeira vez, sem contaminação treino/validação.

## 9. Simplificações deliberadas (documentar na tese como limitação conhecida)

- O modelo prevê cada agente **isoladamente**, usando só a trajetória
  passada dele mesmo — sem mapa/roadgraph, sem outros agentes como
  contexto. Modelos sérios da área (VectorNet, MTR) usam ambos.
- Entrada limitada a `x,y`. O `full_state` tem também velocidade, heading e
  dimensões, hoje não usados.
- A validação oficial usa um teto (`MAX_AGENTS=12`) de agentes-alvo por
  cenário em vez do padrão de 128 slots do tutorial oficial — equivalente
  matematicamente (slots de padding são mascarados), mas não é
  byte-a-byte idêntico ao código oficial.
- O passado inválido é zerado (não há flag de "frame ausente" como
  feature de entrada) — simplificação aceitável para o baseline atual.
- Normalização SDC-cêntrica, não agente-cêntrica (ver seção 6).
- Apenas 3 shards de cada split (~0,3% do `training` disponível).

## 10. Comandos úteis

```bash
# Contar cenarios em um shard
docker exec -it waymo-metrics-container python3 -c "
import tensorflow as tf
path = '<caminho_do_shard>'
print(sum(1 for _ in tf.data.TFRecordDataset(path, compression_type='')))
"

# Conferir tamanho dos caches
ls datasets/waymo/cache_train/*.npy | wc -l
ls datasets/waymo/cache_val/*.npy   | wc -l

# Monitorar GPU durante o treino
watch -n 1 nvidia-smi
nvidia-smi dmon -s um        # utilizacao + memoria, em stream
```

Para a sequência completa do pipeline, ver seção 4.1.

## 11. Próximos passos possíveis

**Em andamento:**
1. Multimodal K=6 com loss Winner-Takes-All (seção 7.2) — resultados pendentes.
2. Adaptar `validate_motion_official.py` para receber as 6 hipóteses e os
   scores reais (hoje ele monta `prediction_score` com valor único e assume
   1 modo).

**Fila, em ordem crescente de esforço:**
3. Normalização agente-cêntrica (centrar e rotacionar pelo próprio agente-alvo).
4. Usar mais campos do `full_state` como entrada (velocidade, heading).
5. Mais shards — ajuda generalização geral, não resolve MissRate alto
   (que é falta de contexto, não falta de dado).
6. Encoder sequencial (LSTM/GRU ou Transformer pequeno) sobre o histórico,
   substituindo o `flatten` dos 11 frames.
7. Incorporar outros agentes como contexto (salto arquitetural real).
8. Incorporar roadgraph/mapa. Duas linhagens: raster + CNN (MultiPath,
   CoverNet) ou vetorizado (VectorNet, LaneGCN). É aqui que a GPU
   efetivamente trabalha — com 320k parâmetros a RTX 5060 Ti está ociosa,
   e o gargalo real hoje é CPU/IO no dataloader.
9. Regularização (dropout, weight decay) e/ou early stopping automático
   se overfitting voltar a aparecer com mais dado.

**Referência de teto:** Wayformer, MTR e SceneTransformer são o estado da
arte no leaderboard do WOMD. Provavelmente fora de escopo para reimplementar,
mas obrigatórios como citação e ponto de comparação.

## 12. Nota sobre este documento

Este arquivo é a fonte de verdade de contexto do projeto e está versionado
no repositório (`docs/`), junto com o código. Ao usar assistentes de IA com
o repositório sincronizado, sincronize antes de começar — código e
documentação defasados são a principal causa de respostas erradas com
aparência de confiança.
