# Projeto de Mestrado — Motion Prediction (Waymo Open Dataset)
### Documento de contexto — Milestone 2 (split oficial + multimodal K=6 validado)
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
│       ├── baseline/            # predições do SimpleTrajectoryPredictor
│       └── multimodal_cls*/     # predições do multimodal, uma por cls_weight
├── docker/
│   ├── waymo-metrics/Dockerfile     # container de METRICAS (CPU)
│   ├── training-v1/Dockerfile       # container de TREINO (GPU)
│   └── _OLD_waymo-legacy/Dockerfile # descartado
├── experiments/checkpoints/
│   ├── baseline_3shards/    # HISTÓRICO — split caseiro, ver seção 8.1
│   ├── baseline_oficial/    # baseline com split oficial
│   └── multimodal_cls*/     # multimodal K=6, uma pasta por cls_weight
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

# multimodal: cls_weight vira parte do nome da pasta de checkpoint,
# entao rodadas da varredura nao se sobrescrevem
python3 -m src.motion.train_multimodal --cls-weight 20
python3 -m src.motion.run_inference_multimodal --tag cls20

# --- Container de METRICAS (CPU) ---
# O validate detecta sozinho se as predicoes sao unimodais ou multimodais.
python3 -m src.motion.validate_motion_official \
    --pred-dir /workspace/datasets/waymo/predictions/baseline

python3 -m src.motion.validate_motion_official \
    --pred-dir /workspace/datasets/waymo/predictions/multimodal_cls20
```

**Checagem obrigatoria antes de confiar em qualquer numero:** o script
imprime o shape de `prediction_trajectory` antes de chamar a metrica.
Unimodal deve dar `(N, 12, 1, 1, 16, 2)`; multimodal K=6 deve dar
`(N, 12, 6, 1, 16, 2)`. Se a terceira posicao nao for 6 no multimodal, o
empacotamento dos modos falhou e o resultado nao vale nada.

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
| minADE / minFDE / MissRate / mAP (oficiais) | sim | sim | **sim — é a comparação válida** |
| Val Loss (treino) | `Val Loss` | `Val(melhor modo)` | não — best-of-6 vs best-of-1 |
| Val Loss (treino) | `Val Loss` | `Val(top-1)` | não — ver abaixo |

**Correção registrada em 24/07.** Uma versão anterior deste documento
afirmava que `Val(top-1)` era o número honesto para comparar com o
baseline. **Está errado**, e o motivo é conceitual:

para MSE com uma única predição, o ótimo matemático é a **média
condicional** do futuro — e é exatamente isso que o baseline unimodal
aprende. O WTA faz o oposto de propósito: empurra cada modo para uma
hipótese específica, longe da média. Medido: melhor modo ~47, modo
escolhido ~230, média dos 6 ~715. Os modos são deliberadamente
espalhados, então o top-1 de um modelo multimodal **nunca** vai bater o
unimodal em MSE — e isso não é falha, é a consequência esperada de
otimizar cobertura em vez de erro médio.

A comparação válida é pelas métricas oficiais, que existem justamente
para isso: minADE/minFDE/MissRate medem cobertura (best-of-6) e mAP mede
a qualidade do ranqueamento.

**Uso correto dos diagnósticos de treino:**
- `Val(melhor modo)` — proxy de minADE, útil para seleção de checkpoint
- `Val(top-1)` e `rank medio` — saúde da cabeça de score, não comparação
- `Uso dos modos` — detecção de colapso

**Assimetria a declarar na dissertação:** o baseline seleciona o melhor
checkpoint pela val loss unimodal; o multimodal, pelo erro do melhor modo
(proxy direto de minADE, que é best-of-6). Cada um otimiza o que sua
métrica premia. É defensável, mas é uma escolha explícita.

### 7.4. Calibração do `cls_weight` (loss WTA)

Os dois termos da loss vivem em escalas muito diferentes: a regressão é
MSE em m² (dezenas a centenas), enquanto a cross-entropy com 6 classes
parte de ln(6) ≈ 1,79. Com `cls_weight=1.0` a classificação vale ~1% da
loss total e o gradiente da `score_head` fica afogado.

Diagnóstico implementado em `wta_loss`: além do erro do modo escolhido,
o script reporta o erro **médio** dos K modos (o que se obteria escolhendo
ao acaso) e o **rank médio** do modo escolhido no ranking real de
qualidade (0 = melhor dos 6, 5 = pior, 2,5 = acaso puro).

Medido com `cls_weight=20..100`: rank médio entre 1,0 e 1,5 e erro do
escolhido ~3x melhor que o acaso. A cabeça de score ranqueia de verdade.

### Progressão de escala testada
| Cenários (shards) | Exemplos (agente-alvo) | Observação |
|---|---|---|
| 5 (shard 0, parcial) | 5 (só SDC, versão antiga) | Só validou o pipeline mecânico |
| 250 (shard 0, parcial) | 250 (só SDC, versão antiga) | Overfitting evidente |
| 400 (shard 0 completo, quase) | 1716 (agentes-alvo reais) | Loss estourou p/ milhões (bug de mask), depois corrigido |
| ~1500 (shards 0,1,2 de `training`) | 6836 | Baseline da seção 8.1 (com contaminação) |
| 1530 de `training` + 879 de `validation` | 6836 treino / 3837 validação | Split oficial — seção 8.2 |

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

**Ponto RESOLVIDO em 24/07:** o número **1530** eram **cenários**, não
trajetórias. O log do op C++ do Waymo imprime `Computing motion metrics
for N trajectories`, onde N é o tamanho do batch — ou seja, a contagem de
cenários. O rótulo errado veio do próprio código do Waymo e foi copiado
para este documento. Não houve inferência parcial nem truncamento.

**Leitura:** números típicos de baseline ingênuo (extrapolação sem contexto),
bem distantes do estado da arte (faixa de 1-2m de minADE @8s, MissRate bem
abaixo de 50%).

### 8.2. Resultado com split oficial (24/07)

Treino em `cache_train` (shards 0-2 de `training`, 6836 agentes-alvo),
avaliação em `cache_val` (shards 0-2 de `validation`, 879 cenários /
3837 agentes-alvo). Sem contaminação treino/validação.

#### Veículos @8s (n=3324 — a única categoria com amostra sólida)

| Modelo | minADE | minFDE | MissRate | mAP |
|---|---|---|---|---|
| Baseline unimodal | 12,94 | 29,72 | 97,7% | 0,0006 |
| Multimodal cls=1 | 7,78 | 15,81 | 93,0% | 0,0025 |
| Multimodal cls=20 | **7,29** | **14,91** | 93,8% | 0,0026 |
| Multimodal cls=50 | 7,38 | 15,29 | 93,1% | 0,0050 |
| Multimodal cls=100 | 7,47 | 15,29 | 93,0% | **0,0058** |

#### Pedestres @8s (n=417)

| Modelo | minADE | minFDE | MissRate | mAP |
|---|---|---|---|---|
| Baseline unimodal | 5,02 | 8,70 | 90,1% | 0,0125 |
| Multimodal cls=1 | 4,79 | 8,65 | 94,9% | 0,0053 |
| Multimodal cls=20 | 4,33 | 7,76 | 92,7% | 0,0133 |
| Multimodal cls=50 | **3,92** | **7,04** | 91,7% | 0,0076 |
| Multimodal cls=100 | 4,35 | 7,56 | 88,5% | **0,0141** |

#### Ciclistas @8s (n=96 — ⚠️ amostra pequena, números não confiáveis)

| Modelo | minADE | minFDE | MissRate | mAP |
|---|---|---|---|---|
| Baseline unimodal | 7,34 | 17,25 | 96,3% | 0,0012 |
| Multimodal cls=1 | 6,07 | 11,32 | 93,9% | 0,0032 |
| Multimodal cls=20 | 5,72 | 10,97 | 95,1% | 0,0019 |
| Multimodal cls=50 | **5,55** | 11,07 | 84,2% | 0,0516 |
| Multimodal cls=100 | 5,69 | 11,79 | 96,3% | 0,0025 |

> ⚠️ Com 96 exemplos, as métricas de ciclista oscilam sem padrão entre
> configurações (MissRate varia de 84% a 96% sem relação com o peso). O
> mAP de 0,0516 do cls=50 é quase certamente ruído amostral, não um
> resultado. Reportar esta categoria apenas com o n explícito, ou
> aumentar o número de shards antes de afirmar qualquer coisa sobre ela.

#### Leitura dos resultados

**1. A multimodalidade entregou um ganho grande e real.** Veículos @8s:
minADE cai 44% (12,94 → 7,29) e minFDE 50% (29,72 → 14,91). Obtido sem
alterar o backbone, sem adicionar dado e sem contexto novo — apenas
mudando a formulação da saída (K=6) e a loss (WTA). Confirma que parte
substancial do erro do baseline vinha de ele ser forçado a prever a média
de futuros mutuamente incompatíveis.

**2. O trade-off cobertura × ranqueamento é real e controlável.** O mAP
de veículo cresce monotonicamente com o `cls_weight` (0,0025 → 0,0026 →
0,0050 → 0,0058, mais que dobrando), enquanto o minADE tem mínimo em
cls=20 e piora levemente depois. Peso maior na classificação melhora a
ordenação das hipóteses e degrada um pouco a diversidade delas. É um
resultado limpo, medido por métrica oficial, sobre duas capacidades que
a literatura costuma reportar juntas.

**3. O MissRate mal se moveu** (97,7% → 93%). Os thresholds oficiais @8s
são 3m lateral / 6m longitudinal; com minADE em 7,3m, quase tudo continua
sendo "miss". As 6 hipóteses cobrem melhor o espaço, mas nenhuma chega
perto o bastante. **Isso confirma que o gargalo dominante é a ausência de
contexto (mapa e outros agentes), não a formulação da saída.**

**4. O mAP absoluto continua irrelevante** (~0,006 contra ~0,4 do estado
da arte). A *tendência* com o `cls_weight` é informativa; o valor, não.

**5. A contaminação anterior teve efeito nulo neste baseline.** Veículo
@8s: 12,84 (contaminado, seção 8.1) vs 12,94 (limpo). Praticamente
idêntico — porque um MLP de 113k parâmetros vendo apenas 11 pontos (x,y)
não tem capacidade de memorizar 6836 trajetórias; ele aprende algo
próximo de "extrapole a velocidade média" e aplica igualmente a dados
vistos e não vistos. Contaminação só distorce métrica quando há
memorização. Isso **não** invalida a correção da seção 5.1: o pipeline
anterior era indefensável e distorceria os resultados assim que o modelo
ganhasse capacidade.

**6. `OverlapRate` passou a ser reportado** (não constava na seção 8.1).
Veículos @8s ficam em ~22% de sobreposição com outros agentes —
coerente com um modelo que ignora completamente os demais agentes da cena.

#### Ressalvas metodológicas a declarar

- **Uma única seed por configuração.** A diferença entre cls=20 e cls=50
  em minADE (7,29 vs 7,38) está dentro do que pode ser ruído. Para
  afirmação forte, repetir com 3 seeds (~2,5 min por configuração).
- **O `cls_weight` foi escolhido olhando o conjunto de validação**, que é
  uma forma leve de ajuste ao conjunto de teste. Com 4 configurações o
  efeito é pequeno, mas a seleção precisa ser registrada como tal.
- **3 shards de cada split** (~0,3% do `training` disponível).

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

**Concluído em 24/07:** split oficial, multimodal K=6 com WTA, varredura
de `cls_weight`, e `validate_motion_official.py` adaptado a K modos.

**Fila, em ordem crescente de esforço:**
1. **Repetir com 3 seeds** por configuração — barato (~2,5 min cada) e
   necessário para afirmar que a diferença entre cls=20 e cls=50 é real.
2. **Mais shards**, prioritariamente para ciclista (n=96 hoje, insuficiente
   para qualquer conclusão sobre a categoria).
3. Normalização agente-cêntrica (centrar e rotacionar pelo próprio
   agente-alvo, em vez de pelo SDC).
4. Usar mais campos do `full_state` como entrada (velocidade, heading).
5. Encoder sequencial (LSTM/GRU ou Transformer pequeno) sobre o histórico,
   substituindo o `flatten` dos 11 frames.
6. Incorporar outros agentes como contexto (salto arquitetural real).
7. Incorporar roadgraph/mapa. Duas linhagens: raster + CNN (MultiPath,
   CoverNet) ou vetorizado (VectorNet, LaneGCN). **É o item que a seção 8.2
   aponta como decisivo:** o MissRate travado em ~93% indica falta de
   contexto, não de formulação. É também onde a GPU efetivamente trabalha —
   com 320k parâmetros a RTX 5060 Ti fica ociosa, e o gargalo real hoje é
   CPU/IO no dataloader.
8. Regularização (dropout, weight decay) e/ou early stopping automático
   se overfitting voltar a aparecer com mais dado.

**Dívida técnica conhecida:** o `WaymoMotionDataset.__getitem__` carrega o
cenário inteiro (todos os agentes, com `full_state [91,7]` de cada) para
extrair **um** agente — o mesmo arquivo é desserializado ~4,5 vezes por
época. Com 6836 amostras pequenas, cachear tudo em RAM no `__init__`
eliminaria o IO. Não vale otimizar enquanto o treino leva 2s por época.

**Referência de teto:** Wayformer, MTR e SceneTransformer são o estado da
arte no leaderboard do WOMD. Provavelmente fora de escopo para reimplementar,
mas obrigatórios como citação e ponto de comparação.

## 12. Nota sobre este documento

Este arquivo é a fonte de verdade de contexto do projeto e está versionado
no repositório (`docs/`), junto com o código. Ao usar assistentes de IA com
o repositório sincronizado, sincronize antes de começar — código e
documentação defasados são a principal causa de respostas erradas com
aparência de confiança.