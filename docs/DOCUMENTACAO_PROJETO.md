# Projeto de Mestrado — Motion Prediction (Waymo Open Dataset)
### Documento de contexto — Milestone 1 (pipeline validado com métricas oficiais)
Última atualização: 16 de julho de 2026

---

## 1. Objetivo da pesquisa

Projeto de mestrado em datasets automotivos / autonomous driving. Foco inicial:
**Waymo Open Motion Dataset**, explorando primeiro Motion Prediction como
familiarização técnica antes de migrar o foco principal da tese para
**Perception**. Comparação futura planejada com outros datasets (NuScenes, ZOD).

## 2. Hardware

- **Laptop:** Ryzen 5 5600H, 32GB RAM, RTX 3060 6GB (dual boot, ~500GB Linux).
- **Desktop (PC principal do projeto):** Ryzen 5 8400F, 32GB RAM, **RTX 5060 Ti
  16GB** (arquitetura Blackwell, `sm_120`). Armazenamento: 500GB NVMe (SO),
  1TB SSD SATA (`/home`), 2x3TB WD Red + 1x16TB Toshiba N300 (datalake/cold
  storage).

## 3. Arquitetura de ambientes (por que dois containers)

O `waymo-open-dataset` tem binários C++ (`py_metrics_ops`) compilados para
TensorFlow/CUDA antigos, incompatíveis com a RTX 5060 Ti (que exige CUDA 12+).
Solução adotada: **dois containers Docker separados**, conectados só pela
pasta de cache compartilhada.

| Container | Função | Stack | Hardware |
|---|---|---|---|
| **Métricas** | Ler TFRecords, pré-processar, calcular métricas oficiais | Docker, Python 3.8, TF-CPU 2.11.0, `waymo-open-dataset-tf-2-11-0==1.6.1` | CPU only |
| **Treino** | Treinar o modelo, gerar predições | Docker, PyTorch 2.9.0, CUDA 13.0 | GPU (RTX 5060 Ti) |

Ponte entre os dois: pasta de cache montada como volume nos dois containers
(`/workspace/datasets/waymo/cache` e `/workspace/datasets/waymo/predictions`).

**Dataset bruto:** Waymo Motion, `training.tfrecord-XXXXX-of-01000`
(1000 shards no total, ~496 cenários por shard, confirmado por contagem direta).
Hoje processados: shards 0, 1 e 2 (~1500 cenários brutos → 6836 exemplos de
agente-alvo).

## 4. Estrutura de arquivos do pipeline (árvore real confirmada em 16/07)

```
~/autonomous_drive/
├── datasets/waymo/
│   ├── cache/              # .npy gerados pelo preprocessor (ground truth processado)
│   ├── motion -> /data/.disks/hdd3a/waymo_motion/.../uncompressed/  (symlink)
│   └── predictions/        # .npy gerados pelo run_inference (predições do modelo)
├── docker/
│   ├── waymo-metrics/Dockerfile     # container de METRICAS (CPU)
│   ├── training-v1/Dockerfile       # container de TREINO (GPU)
│   └── _OLD_waymo-legacy/Dockerfile # descartado
├── experiments/checkpoints/         # motion_model_eN.pth + motion_model_best.pth
├── src/
│   ├── core/
│   │   ├── waymo_decoder.py             # decodifica Scenario proto bruto
│   │   ├── waymo_preprocessor.py        # ATUAL — roda no container de METRICAS (CPU)
│   │   └── (waymo_preprocessor_old.py, v1, v2, v2(copy) — versões antigas, não usar)
│   └── motion/
│       ├── simple_model.py              # SimpleTrajectoryPredictor (MLP)
│       ├── waymo_pytorch_dataset.py      # nota: fisicamente pode estar em src/core/
│       │                                  # dependendo de quando foi movido — confirmar
│       │                                  # com `find` se dor de cabeça com import
│       ├── train_motionv4.py            # ATUAL — roda no container de TREINO (GPU)
│       │                                  # (confirmado: salva motion_model_best.pth)
│       ├── run_inference.py             # ATUAL — roda no container de TREINO (GPU)
│       ├── validate_motion_official.py  # ATUAL — roda no container de METRICAS (CPU)
│       │                                  # (localização real: src/motion/, não src/core/)
│       └── (train_motion.py, v2, v3, _old, test_metrics.py — versões antigas/não
│           confirmadas, revisar antes de usar)
└── tutorial_motion_original.ipynb   # fonte do config oficial e da fórmula de downsample
```

**⚠️ Observação importante:** o projeto acumulou várias versões paralelas dos
mesmos scripts (`_old`, `v1`, `v2`, `v3`, `v4`, até uma `(copy)`). Isso é
esperado numa fase exploratória, mas é uma fonte real de risco de confusão
(inclusive para uma IA lendo o projeto do zero). **Recomendação para quando
houver um respiro:** consolidar em um único arquivo "canônico" por
função e apagar/arquivar as versões antigas, ou pelo menos renomeá-las para
algo como `archive_train_motion_v1.py` deixando claro que não são para uso.

**Import consistente:** todos os módulos usam `from src.core...` / `from
src.motion...`. Scripts devem ser executados como módulo a partir de
`/workspace` dentro do container, ex: `python3 -m src.motion.train_motionv4`
(precisa de `__init__.py`, mesmo vazio, em `src/`, `src/core/`, `src/motion/`).

## 4.1. Sequência completa de execução do pipeline (confirmada e usada)

```bash
# 1. Container de METRICAS (CPU) — gera/atualiza o cache de ground truth
python3 -m src.core.waymo_preprocessor

# 2. Container de TREINO (GPU) — treina o modelo
python3 -m src.motion.train_motionv4

# 3. Container de TREINO (GPU) — gera predições com o melhor checkpoint
python3 -m src.motion.run_inference

# 4. Container de METRICAS (CPU) — calcula as métricas oficiais
python3 -m src.motion.validate_motion_official
```

**Atenção:** rodar `train_motionv4` novamente SOBRESCREVE os checkpoints
anteriores (`motion_model_e*.pth`, `motion_model_best.pth`) em
`experiments/checkpoints/`. Se quiser preservar um resultado específico
(ex: o marco de 1530 trajetórias documentado na seção 8), copie essa pasta
para outro local antes de treinar de novo.

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

## 6. Formato de dado no cache (`waymo_preprocessor.py`, versão atual)

Cada `.npy` salvo em `/workspace/datasets/waymo/cache/<scenario_id>.npy`
contém um dicionário:
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

## 7. Modelo e treino

- **Modelo:** `SimpleTrajectoryPredictor` — MLP de 2 camadas escondidas
  (256 neurônios), entrada 22 valores (11 frames × x,y), saída 160 valores
  (80 frames × x,y).
- **Uma amostra de treino = um agente-alvo específico**, não uma cena
  inteira. Entrada: só a trajetória passada do próprio agente (SEM contexto
  de mapa ou de outros agentes — simplificação deliberada, ver seção 9).
- **Split treino/validação:** 80/20, seed fixa (42), via `random_split`.
- **Checkpoints:** salvo por época + `motion_model_best.pth` (menor Val Loss).

### Progressão de escala testada
| Cenários (shards) | Exemplos (agente-alvo) | Observação |
|---|---|---|
| 5 (shard 0, parcial) | 5 (só SDC, versão antiga) | Só validou o pipeline mecânico |
| 250 (shard 0, parcial) | 250 (só SDC, versão antiga) | Overfitting evidente |
| 400 (shard 0 completo, quase) | 1716 (agentes-alvo reais) | Loss estourou p/ milhões (bug de mask), depois corrigido |
| ~1500 (shards 0,1,2) | 6836 | Melhor resultado: overfitting reduzido, breakdown por tipo revelou pouco dado de ciclista |

## 8. Validação oficial (métricas Waymo Motion) — RESULTADO ATUAL

Metodologia: inferência no container de treino (`run_inference.py`, usa
`motion_model_best.pth`) → tensores montados e métrica calculada no
container de métricas (`validate_motion_official.py`, chama
`py_metrics_ops.motion_metrics`).

**Config oficial usado** (extraído de `tutorial_motion_original.ipynb`):
`track_steps_per_second: 10`, `prediction_steps_per_second: 2`,
`track_history_samples: 10`, `track_future_samples: 80`, thresholds de miss
a 3s/5s/8s (`measurement_step: 5, 9, 15`).

**Downsample de predição** (10Hz → 2Hz), fórmula oficial:
`prediction[..., (interval-1)::interval, :]` com `interval = 10 // 2 = 5`.

**Resultado (1530 trajetórias avaliadas, 3 shards, baseline MLP sem contexto):**

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

**Leitura:** números típicos de baseline ingênuo (extrapolação sem contexto),
bem distantes do estado da arte (que fica na faixa de 1-2m de minADE @8s,
MissRate bem abaixo de 50%). Isso é esperado, não indica bug — serve como
piso de comparação documentado para a dissertação.

## 9. Simplificações deliberadas (documentar na tese como limitação conhecida)

- O modelo prevê cada agente **isoladamente**, usando só a trajetória
  passada dele mesmo — sem mapa/roadgraph, sem outros agentes como
  contexto. Modelos sérios da área (VectorNet, MTR) usam ambos.
- A validação oficial usa um teto (`MAX_AGENTS=12`) de agentes-alvo por
  cenário em vez do padrão de 128 slots do tutorial oficial — equivalente
  matematicamente (slots de padding são mascarados), mas não é
  byte-a-byte idêntico ao código oficial.
- O passado inválido é zerado (não há flag de "frame ausente" como
  feature de entrada) — simplificação aceitável para o baseline atual.

## 10. Comandos úteis

```bash
# Contar cenarios em um shard
docker exec -it waymo-metrics-container python3 -c "
import tensorflow as tf
path = '<caminho_do_shard>'
print(sum(1 for _ in tf.data.TFRecordDataset(path, compression_type='')))
"

# Preprocessor (container METRICAS)
cd /workspace && python3 -m src.core.waymo_preprocessor

# Treino (container TREINO)
cd /workspace && python3 -m src.motion.train_motion

# Inferencia para validacao oficial (container TREINO)
cd /workspace && python3 -m src.motion.run_inference

# Validacao oficial (container METRICAS)
cd /workspace && python3 -m src.core.validate_motion_official

# Monitorar GPU durante o treino
watch -n 1 nvidia-smi
```

## 11. Próximos passos possíveis (não iniciados)

1. Mais shards (mais dado) — ajuda generalização geral, não resolve
   MissRate alto (que é falta de contexto, não falta de dado).
2. Usar mais campos do `full_state` como entrada (velocidade, heading),
   não só x,y.
3. Incorporar outros agentes como contexto (passo maior).
4. Incorporar roadgraph/mapa (arquitetura mais séria: GNN, VectorNet-like).
5. Regularização (dropout, weight decay) e/ou early stopping automático
   se overfitting voltar a aparecer com mais dado.
