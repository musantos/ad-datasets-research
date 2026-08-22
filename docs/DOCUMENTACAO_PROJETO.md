# Projeto de Mestrado — Motion Prediction (Waymo Open Dataset)
### Documento de contexto
Última atualização: 22 de agosto de 2026

> **Nota de leitura.** As seções **1–11** abaixo são o registro **histórico do
> Milestone 1** (16/jul/2026: baseline MLP *unimodal*, `random_split`, 3 shards).
> O estado corrente do projeto está na **seção 0**, logo abaixo. Onde houver
> conflito (ex.: split, formato de saída, scripts em uso), **vale a seção 0**.

---

## 0. Estado atual — Encoder sequencial (GRU) × Flatten (MLP multimodal)
*(18/ago/2026)*

### 0.1. O que aconteceu desde o Milestone 1
Entre o M1 (corpo histórico deste documento) e este experimento, dois marcos
intermediários foram concluídos:

- **Milestone 2:** os splits **oficiais** do Waymo substituíram o
  `random_split` do M1; treinado um modelo **multimodal K=6 (WTA)** em quatro
  valores de `cls_weight` (1, 20, 50, 100); métricas oficiais rodadas em todos.
- **Tradução PT→EN** do codebase concluída (comentários/docstrings/logs e nomes
  locais; sem mudança de lógica; contrato `.npy` preservado verbatim).

Este experimento é o **primeiro degrau real** da espinha
`MLP → encoder → agente-cêntrica → vetorizado com mapa`: troca o *flatten*
(MLP multimodal) por um **encoder sequencial (GRU)**, mantendo tudo o mais igual.

### 0.2. Setup do experimento
- **Flatten (multimodal):** MLP K=6, entrada achatada `[B,22]`.
- **Sequential:** GRU drop-in, mesma entrada **sem** flatten `[B,11,2]`, mesmas
  saídas `[B,K,80,2]`+`[B,K]`, ~175k params, lê só o último hidden state
  (`hidden_dim=128`). Scripts `sequential_model.py` / `train_sequential.py` /
  `run_inference_sequential.py` (cópias fiéis dos irmãos multimodal).
- **Comparação justa:** mesmo dado/split/hardware; ambos treinados **até
  convergência** com early stopping (`EPOCHS=300`, `PATIENCE=10`), não a um
  orçamento fixo de épocas. Lote `2026-08-18_12-38-00`.
- **Fonte de verdade do rótulo:** cabeçalho `pred_dir` de cada CSV (o nome do
  arquivo é só conveniência). CSVs em `results/metrics_<modelo>_<cls>_<stamp>.csv`.

### 0.3. Resultado principal — médias das 9 breakdowns
Valores conferidos direto nas 8 CSVs do lote:

| cls | modelo | minADE | minFDE | MissRate | OverlapRate | mAP |
|-----|--------|--------|--------|----------|-------------|--------|
| 1   | flatten    | 3.73 | 6.83 | 0.89 | 0.125 | 0.0131 |
| 1   | sequential | 4.52 | 8.25 | 0.96 | 0.132 | 0.0055 |
| 20  | flatten    | 3.50 | 6.22 | 0.86 | 0.117 | 0.0539 |
| 20  | sequential | 4.20 | 7.79 | 0.94 | 0.130 | 0.0106 |
| 50  | flatten    | 3.48 | 6.57 | 0.89 | 0.115 | 0.0310 |
| 50  | sequential | 5.25 | 9.63 | 0.96 | 0.151 | 0.0053 |
| 100 | flatten    | 3.40 | 6.18 | 0.87 | 0.114 | 0.0454 |
| 100 | sequential | 4.57 | 8.65 | 0.95 | 0.135 | 0.0209 |

**O flatten vence em TODAS as métricas, nos 4 pesos.** Não é bug: ambos
convergiram, mesmo dado/split/hardware, rótulos conferidos pelo `pred_dir`.

### 0.4. Interpretação (resultado reportável, não falha)
- Com só `x,y` SDC-cêntrico e 11 frames de passado, o MLP tem acesso **linear
  direto** à posição de largada (frame 10); "continue de onde parou" é trivial.
  O GRU comprime isso num hidden state e perde essa âncora, pagando o custo da
  recorrência sem estrutura temporal que compense num horizonte tão curto.
- **Custo de treino do GRU é maior:** converge bem mais devagar (mais épocas até
  o early stop) para chegar num número **pior**. Entra como coluna "custo" do
  trade-off.
- **Conclusão que arma o próximo degrau:** recorrência **sozinha** não ajuda. A
  hipótese testável passa a ser "recorrência **+ estado rico**
  (velocidade/heading) **+ normalização agente-cêntrica** ajuda (ou não)".

**Ressalva metodológica:** **sem seeds, não ranquear `cls` entre si** (ex.:
sequential cls50 saiu pior que cls1 — cheiro de ruído de rodada). A ordem
*flatten > sequential* é robusta; a ordem *entre pesos*, não.

### 0.5. Armadilhas registradas neste ciclo
- **Rótulo de CSV silencioso.** CSVs sem modelo/data no nome geraram confusão
  "flatten vs sequential". Fix: `metrics_<modelo>_<cls>_<timestamp>.csv`; fonte
  de verdade = `pred_dir` no cabeçalho.
- **Re-treino sobrescreve** `checkpoints/<model>_<tag>/` e
  `predictions/<model>_<tag>/` sem avisar (um re-treino não-intencional já apagou
  predições antigas). Usar `--seed` no nome quando quiser múltiplas rodadas.
- **Comparar a épocas fixas é injusto** (favorece quem converge rápido, o MLP).
  Princípio adotado: comparar **at convergence** (early stopping). CSVs antigas
  de 25 épocas estão sub-treinadas — **não** usar na tabela final nem como seeds.
- **Fronteira de contêiner.** `validate_motion_official` → contêiner de MÉTRICAS
  (CPU/TF). `train_*`/`run_inference_*` → contêiner de TREINO (GPU).
  `ModuleNotFoundError: tensorflow` = contêiner errado.
- GPU segue **IO-bound** (util ~6%); util%/VRAM só valem medir na Fase 2 (mapa).
  Registrar como custo: tempo/época, nº de épocas até early stop, nº de params.

### 0.6. Próximos passos
1. ✅ **`--seed`** nos `train_*` **e** `run_inference_*` (semeia
   random/numpy/torch + sufixo `<model>_<tag>_seed<n>` em checkpoints, predições
   e log). Retrocompatível: sem `--seed`, caminhos idênticos aos de hoje.
   ✅ **5 seeds rodadas e agregadas** (média ± desvio) — resultados em **0.8**.
2. ✅ **`best_epoch`** registrado na linha final do log + no CSV por época
   (também adicionado neste ciclo: log por época em
   `/workspace/experiments/logs/<model>_<run_tag>_<stamp>.csv`, com `epoch_time_s`,
   `cum_time_s`, `is_best` → tempo-até-best = `cum_time_s` da última linha `is_best=1`).
3. **Degrau seguinte:** normalização **agente-cêntrica** + `full_state`
   (velocidade/heading). Mora em `src/core` (preprocessor + `WaymoMotionDataset`).
   O `full_state` no cache já tem `heading, vx, vy` → dá para re-centrar/rotacionar
   **no dataload**, sem reprocessar os 2,1 TB. Grade alvo:
   `{flatten, encoder} × {SDC, agente-cêntrica}` + input enriquecido.

### 0.7. Comandos atuais (substituem os da seção 10 para este ciclo)

**Rodada única (sem seed) — comportamento de hoje.** Caminhos sem sufixo
(`multimodal_cls50`, etc.); é o que gerou o lote `2026-08-18_12-38-00`.
```bash
# ===== CONTÊINER DE TREINO (GPU) — treina + infere, 2 modelos × 4 pesos =====
for t in 1 20 50 100; do
    python3 -m src.motion.train_multimodal         --cls-weight $t
    python3 -m src.motion.run_inference_multimodal --tag cls$t
    python3 -m src.motion.train_sequential         --cls-weight $t
    python3 -m src.motion.run_inference_sequential --tag cls$t
done

# ===== CONTÊINER DE MÉTRICAS (CPU, TF 2.11) =====
mkdir -p results
STAMP=$(date +%Y-%m-%d_%H-%M-%S)          # 1 carimbo p/ o lote inteiro
for m in multimodal sequential; do
  for t in cls1 cls20 cls50 cls100; do
    python3 -m src.motion.validate_motion_official \
        --pred-dir /workspace/datasets/waymo/predictions/${m}_$t \
        --csv results/metrics_${m}_${t}_${STAMP}.csv
  done
done
```

**Multi-seed (Fase 0 de hygiene) — 3 rodadas por configuração.** O `--seed`
sufixa `_seed<n>` na pasta de checkpoints, nas predições e no log por época
(`<model>_cls<t>_seed<s>`), então as seeds coexistem em vez de se sobrescrever.
Treino e inferência precisam do **mesmo** `--seed`; as métricas apontam para a
pasta sufixada no `--pred-dir`. Ajustar `SEEDS` conforme quantas rodadas.
```bash
SEEDS="0 1 2"

# ===== CONTÊINER DE TREINO (GPU) — 2 modelos × 4 pesos × N seeds =====
for s in $SEEDS; do
  for t in 1 20 50 100; do
    python3 -m src.motion.train_multimodal         --cls-weight $t --seed $s
    python3 -m src.motion.run_inference_multimodal --tag cls$t     --seed $s
    python3 -m src.motion.train_sequential         --cls-weight $t --seed $s
    python3 -m src.motion.run_inference_sequential --tag cls$t     --seed $s
  done
done

# ===== CONTÊINER DE MÉTRICAS (CPU, TF 2.11) =====
mkdir -p results
STAMP=$(date +%Y-%m-%d_%H-%M-%S)          # 1 carimbo p/ o lote inteiro
for s in $SEEDS; do
  for m in multimodal sequential; do
    for t in cls1 cls20 cls50 cls100; do
      python3 -m src.motion.validate_motion_official \
          --pred-dir /workspace/datasets/waymo/predictions/${m}_${t}_seed${s} \
          --csv results/metrics_${m}_${t}_seed${s}_${STAMP}.csv
    done
  done
done
```
Agregação: com as N CSVs por config, reportar **média ± desvio** das métricas
sobre as seeds (é o que separa ruído de treino de efeito real; sem isso, não
ranquear `cls` entre si — ver ressalva em 0.4).

### 0.8. Consolidação das seeds *(19/ago/2026)* — fecha a Parte A

Rodadas **5 seeds** por configuração (não 3; 5 dá barra de erro mais confiável),
lote de treino/inferência de 19/ago. Grade completa: **2 modelos × 4 pesos × 5
seeds = 40 runs**. **Todos convergiram por early stopping** (`epochs_after_best`
== `PATIENCE=10` em todos os 40; nenhum parou por teto de épocas). A tabela 0.3
acima (lote single-seed 18/ago) fica como **registro histórico**; a leitura
válida com variância é a de baixo.

**Métricas oficiais — overall (média das 9 breakdowns), média ± desvio das 5 seeds:**

| modelo | cls | minADE | minFDE | MissRate | mAP |
|--------|-----|--------|--------|----------|-----|
| flatten    | 1   | 3.572 ± 0.128 | 6.581 ± 0.230 | 0.881 ± 0.010 | 0.024 ± 0.016 |
| flatten    | 20  | 3.481 ± 0.141 | 6.380 ± 0.149 | 0.878 ± 0.012 | 0.020 ± 0.011 |
| flatten    | 50  | 3.538 ± 0.116 | 6.552 ± 0.152 | 0.889 ± 0.015 | 0.020 ± 0.013 |
| flatten    | 100 | 3.519 ± 0.082 | 6.368 ± 0.106 | 0.880 ± 0.013 | 0.023 ± 0.009 |
| sequential | 1   | 4.742 ± 0.371 | 8.724 ± 0.518 | 0.947 ± 0.009 | 0.008 ± 0.003 |
| sequential | 20  | 4.502 ± 0.370 | 8.381 ± 0.550 | 0.948 ± 0.007 | 0.016 ± 0.008 |
| sequential | 50  | 4.442 ± 0.129 | 8.259 ± 0.264 | 0.945 ± 0.006 | 0.006 ± 0.002 |
| sequential | 100 | 4.415 ± 0.155 | 8.144 ± 0.164 | 0.944 ± 0.010 | 0.013 ± 0.008 |

**Veredicto 1 — flatten > sequential é robusto.** O gap entre modelos (~0.9–1.2
em minADE, ~1.7–2.1 em minFDE) é **várias vezes maior que o desvio entre seeds**
(~0.08–0.37). Pela regra "diferença menor que a barra = ruído", a ordem é efeito
real. Isto **promove o resultado do M1/seção 0.3** de single-seed para
estatisticamente sustentado.

**Veredicto 2 — `cls_weight` é indistinguível de ruído (resolve a ressalva de
0.4).** Dentro de cada modelo, as 4 configs de peso **se sobrepõem dentro do
desvio** em todas as métricas (ex.: flatten minADE varia 3.481→3.538 com desvios
~0.12–0.14). A ressalva "sem seeds, não ranquear `cls`" agora tem veredicto:
**não há ranking detectável entre pesos** neste regime — é empate dentro da barra
de erro, não uma ordem. `mAP` é ruidoso demais para ordenar (desvio
frequentemente ~ metade da média); não usar como critério de desempate aqui.

**Veredicto 3 — custo assimétrico (trade-off da seção 0.4, agora quantificado):**

| modelo | tempo total médio (s) | épocas até convergir | s/época |
|--------|----------------------|---------------------|---------|
| flatten    | 127.1 | 52.0  | 2.05 |
| sequential | 342.5 | 167.2 | 1.94 |

O GRU **não é mais lento por época** (~2 s nos dois); ele precisa de **~3.2× mais
épocas** para convergir e termina **pior**. Custo/benefício condenatório
(~2.7× wall-clock por métricas piores) — reforça a hipótese que arma o item 4:
recorrência sozinha não paga o próprio custo. As 40 rodadas somaram ~2.6 h.

**Veredicto 4 — GPU subutilizada (dado de infra p/ planejar a Fase 2).** RTX 5060
Ti 16 GB: utilização **mediana 4%** (pico 44%), memória **~1.3 GB de 16 GB**,
potência ~28 W, processo Python ~240 MiB. Os experimentos **não são GPU-bound** —
o limitador é o pipeline de dados (dataload/CPU), não o compute. Implica:
(a) folga enorme de VRAM/compute para o modelo vetorizado + mapa/social (Fase 2);
(b) mais seeds / batch maior são ~grátis em GPU; (c) se o tempo incomodar, o alvo
de otimização é o I/O de dados, não a placa.

**Decisão para o item 4:** como nenhum `cls_weight` se distingue, **não há config
"vencedora" a carregar**. Fixar um peso razoável (cls20 teve o menor minADE médio
no flatten — serve de default) e concentrar a variável experimental do item 4
(SDC-cêntrico × agente-cêntrico + estado rico). Isso **encolhe a grade** (dispensa
varrer `cls` de novo).

**Contexto de dados (relevante daqui pra frente):** os treinos atuais usam um
**subconjunto** cacheado, não os 2,1 TB. Pipeline: full dataset em **HDD (cold
storage)** → script copia tfrecords para **SSD SATA 1 TB** (throughput) → cache
`.npy` lido no treino. Os ~2 s/época confirmam subconjunto pequeno. "Usar mais
dados" só vira alavanca real **depois** da mudança arquitetural (item 4 → Fase 2),
que é o que dá capacidade/entrada para explorá-los; antes disso, mais dados
esbarram no teto de entrada pobre (`[B,11,2]`, sem mapa/social). *(Nº exato de
tfrecords em uso: a confirmar.)*

---

### 0.9. Item 4 — normalização agente-cêntrica *(20/ago/2026)* — fecha o item 4

Executada a grade do item 4 (**opção C**): `{flatten, sequential} × {SDC,
agente} × seed 0..7`, **`cls_weight=20` fixo** (herdado da Parte A; ver decisão
em 0.8). 4 configs × 8 seeds = **32 runs (train+infer)**. **Todas convergiram
por early stopping** (`epochs_after_best == PATIENCE=10` nas 32; nenhuma parou
por teto de épocas). Mesmo split/hardware. `cache_train=2972`, `cache_val=879`,
**volume fixo** (os 4 configs veem exatamente o mesmo dado).

**Importante — o que este experimento isola:** o braço `SDC` **não** é o baseline
de 2 canais da Parte A. Aqui os **dois** braços usam input rico
`(x,y,heading,vx,vy)` com heading→sin/cos ⇒ **6 canais**. A única variável é a
**normalização** (SDC-cêntrico × agente-cêntrico). Predições do braço agente são
remapeadas ao frame SDC antes de gravar (inversão validada numericamente,
round-trip 10000/10000 exato).

**Métricas oficiais — overall (média das 9 breakdowns), média ± desvio das 8 seeds:**

| modelo | arm | minADE | minFDE | MissRate | mAP |
|--------|-----|--------|--------|----------|-----|
| flatten    | agente | **1.394 ± 0.029** | **3.293 ± 0.060** | **0.576 ± 0.018** | 0.117 ± 0.018 |
| flatten    | SDC    | 2.068 ± 0.069 | 3.705 ± 0.140 | 0.725 ± 0.035 | 0.055 ± 0.020 |
| sequential | agente | 1.408 ± 0.032 | 3.360 ± 0.113 | 0.581 ± 0.017 | **0.140 ± 0.023** |
| sequential | SDC    | 2.252 ± 0.123 | 4.065 ± 0.194 | 0.759 ± 0.020 | 0.056 ± 0.008 |

**Veredicto 1 — H1 confirmada, forte (agente-cêntrico vence).** Teste **pareado
por seed** (mesma seed nos dois braços): o agente vence SDC em **minADE em 9/9
breakdowns nos dois modelos, todos p < 0.001**; em MissRate, 9/9 no flatten
(7/9 significativos). A vantagem em minADE cai de ~0.76 (horizonte curto `_5`)
para ~0.41 (`_15`) no flatten — grande no curto, encolhe no longo, exatamente a
assinatura física esperada. O smoke-test de 1 seed se sustenta com IC de 8 seeds.

**Veredicto 2 — H2 resolvida: os flips do smoke-test eram ruído.** Os dois casos
que invertiam no n=1 (VEHICLE_15, CYCLIST_15 em minFDE) **não são reais** no
flatten: VEHICLE_15 Δ(SDC−agente)=−0.027 (p=0.71), CYCLIST_15 Δ=−0.085 (p=0.61)
— IC cruza zero. No horizonte de 8s a vantagem do agente **converge para empate
estatístico** nos endpoints difíceis (veículo/ciclista), mas **nunca reverte
significativamente** para o SDC. No braço sequential nem empate há (agente ainda
vence VEHICLE_15 minFDE, p=0.03).

**Veredicto 3 — H3, achado novo: a normalização fecha o gap flatten×GRU.** No
braço **SDC**, o resultado da Parte A se repete limpo: flatten > sequential em
**9/9** (minADE/minFDE/MR). No braço **agente**, os modelos ficam **essencialmente
empatados** (flatten ganha 4–6/9 conforme a métrica, com vitórias e derrotas
significativas equilibradas); em mAP o **sequential-agente é o melhor de todos**
(0.140 vs 0.117). Leitura: a recorrência **sozinha** não pagava o próprio custo
(Parte A), mas recorrência **+ estado rico + normalização agente-cêntrica** deixa
o GRU competitivo. Reportável nos dois sentidos.

**Veredicto 4 — o agente-cêntrico também estabiliza/barateia o treino.** Épocas
até convergir (média ± desvio das 8 seeds) e energia por run (fase train,
integrada do log da placa):

| modelo | arm | épocas até best | tempo/run (s) | energia/run (Wh) |
|--------|-----|-----------------|---------------|------------------|
| flatten    | SDC    | 36.0 ± 19.3 | 84.8  | 0.64 |
| flatten    | agente | 43.1 ± 5.5  | 99.8  | 0.73 |
| sequential | agente | 65.9 ± 9.2  | 142.3 | 1.18 |
| sequential | SDC    | 92.4 ± 21.6 | 185.6 | 1.51 |

O `sequential-SDC` é o mais caro (mais épocas, maior energia) e o pior em
métricas; o `sequential-agente` cai para ~66 épocas e metade da variância. Ou
seja, a mesma intervenção metodológica melhora **acurácia e custo de treino** do
GRU simultaneamente. A grade inteira (32 treinos) somou **~0.033 kWh / ~68 min**.

**Veredicto 5 — confirmação quantitativa de input-bound (infra p/ Fase 2).**
Telemetria da RTX 5060 Ti durante toda a grade: utilização **média 5.6%, p95 9%,
pico 13%**; VRAM **máx 1766 MiB (11% de 16 GB)**; potência **média 29.5 W** (TDP
~180 W). Os experimentos **não são GPU-bound** — o limitador é o dataload
(CPU/IO em HDD). Implica folga enorme para a Fase 2 (vetorizado + mapa/social,
atenção O(n²)) e legitima o próximo eixo **curva de dados**: provavelmente
data-starved, então escalar cenários deve ser barato e reportável.

**Caveats carregados (honestidade metodológica):**
- `cls_weight=20` foi **herdado** do regime SDC e não re-varrido no item 4 —
  possível leve viés pró-SDC. O agente venceu mesmo assim ⇒ conclusão **mais
  forte** (ganhou em desvantagem).
- **N=8 fixado a priori.** Não adicionar seeds após ver o resultado (p-hacking).
- Comparação sobre métricas oficiais (minADE/minFDE/MR/mAP), **nunca** `Val(top-1)`.
- `mAP` é a métrica mais ruidosa (desvio ~ metade da média em vários breakdowns);
  a direção agente > SDC é consistente e significativa nos breakdowns fortes
  (VEH_5, PED_5, PED_15), mas não usar como critério fino de ranking.

**Decisão para a Fase 2:** a melhor config do item 4 é **flatten-agente** em
distância/miss (com sequential-agente competitivo e à frente em mAP). Fixar
flatten-agente como base para (a) a **curva de dados** e (b) o salto arquitetural
para o modelo **vetorizado com mapa** — cada eixo isolado, sem misturar com esta
grade (seria confound).

---

### 0.10. V0 — encoder vetorizado (VectorNet) *(21/ago/2026)* — fecha o V0

Primeiro degrau da **Fase 2 (vetorizado)**. Substitui o encoder *flatten* (MLP)
pelo **subgraph VectorNet** (polyline + agregação), mantendo tudo o mais igual
(saída `[B,6,80,2]`+`[B,6]`, K=6, WTA, métricas oficiais, mesmo cache). NÃO
adiciona contexto — social é o V1, mapa é o V2. A normalização agente-cêntrica
(item 4) entra como **frame de referência da entrada**, não como braço separado.

Grade completa: **1 modelo × 1 braço (agente) × 2 variantes {raw, std} × 8 seeds
= 16 runs**, `cls_weight=20`, at convergence (`patience=10`), mesmo cache
(`cache_train=2972`/13204 exemplos-agente; `cache_val=879`/3837), métricas
oficiais. Média ± desvio (populacional) sobre as 8 seeds; cada seed = média das 9
breakdowns.

**Arquitetura (decisões a priori):** subgraph VectorNet `hidden_dim=64`,
`n_layers=2`, 10 vetores por agente, entrada 9-dim (`xs,ys,xe,ye,sin,cos,vx,vy,dt`
derivados on-the-fly do histórico `[B,11,6]`). **~80k params** (¼ do flatten
~320k) — NÃO casados de propósito (eixo de eficiência). Padronização = incremento
medido e rotulado: stats por-canal congeladas do `cache_train`, **sin/cos
passthrough** (mean=0/std=1), buffers persistentes que **viajam no checkpoint** —
o buffer (identidade=raw, stats reais=std), não uma flag Python, determina o
regime, garantindo paridade train/inferência por construção.

**Métricas oficiais — overall (média das 9 breakdowns), média ± desvio das 8 seeds:**

| métrica | RAW (8) | STD (8) | flatten-agente (8) | std−raw | std−flatten |
|---|---|---|---|---|---|
| minADE   | 1.468 ± 0.040 | 1.439 ± 0.038 | **1.394 ± 0.029** | −0.029 | +0.045 |
| minFDE   | 3.409 ± 0.080 | 3.364 ± 0.083 | **3.293 ± 0.060** | −0.045 | +0.071 |
| MissRate | 0.615 ± 0.022 | 0.613 ± 0.023 | **0.576 ± 0.018** | −0.002 | +0.037 |
| mAP      | 0.119 ± 0.032 | 0.107 ± 0.016 | **0.117 ± 0.018** | −0.012 | −0.010 |

**minADE por seed** (para variância): raw = [1.538, 1.429, 1.493, 1.493, 1.468,
1.454, 1.400, 1.465]; std = [1.413, 1.422, 1.458, 1.413, 1.423, 1.511, 1.390,
1.477]. (O seed0-raw = 1.538 foi o **pior** dos 8 — foi o que inflou o smoke de
seed0 e encolheu o efeito aparente da padronização.)

**Convergência** (train_summary): raw best_epoch 35–74 (total 83–158s); std
best_epoch 47–112 (total 107–238s). Todos `epochs_after_best=10` → **todos
pararam por early stopping no platô**, nenhum por budget. Grade metodologicamente
sólida.

**Veredicto 1 — a padronização ajuda pouco e não uniformemente.** Melhora minADE
(−0.029) e minFDE (−0.045) de forma consistente — as métricas de *erro de
posição*, onde espalhar x,y e normalizar vx,vy deveria pagar. Mas **MissRate quase
não move** (−0.002) e **mAP piora** (−0.012, ruidoso: desvio raw ±0.032 alto).
Padronizar refina a *geometria da trajetória*, mas não toca *cobertura*
(MissRate) nem *ranqueamento* (mAP).

**Veredicto 2 — std NÃO alcança o flatten-agente.** Continua atrás em 3/4 (minADE
+0.045, minFDE +0.071, MissRate +0.037), com barras que **não se sobrepõem** em
minADE/minFDE. Empata em mAP (dentro do ruído). A padronização **estreitou** o gap
mas não o **fechou**.

**Veredicto 3 — o V0 fez seu trabalho.** Isolou a máquina vetorizada e mostrou que
o gargalo de cobertura **não está no encoder nem na escala da entrada**. Se
normalizar a entrada não move MissRate, o que falta é **contexto** (social/mapa).
Isso **motiva o V1/V2** — é a ponte lógica pro próximo degrau.

**Veredicto 4 — eixo de eficiência (params NÃO casados).** Tudo isso com **~80k
params** (¼ do flatten). "Perde por +0.045 em minADE com 4× menos parâmetros" é
uma frase de **eficiência**, não de derrota. Reportar params na tabela.

**Pendência analítica opcional (não bloqueia):** teste t pareado por seed nas 4
métricas (raw↔std, std↔flatten) formalizaria quais Δ passam de ruído. Os 16 runs
consolidados já existem; a conclusão qualitativa NÃO depende disso.

---

### 0.11. V1 — contexto social (cross-attention) *(22/ago/2026)* — DIAGNÓSTICO EM ANDAMENTO

> **Status: degrau NÃO fechado.** Apenas o **seed0 (raw)** foi medido — vale para
> validar o pipeline, **não** para concluir sobre o efeito social. A grade de 8
> seeds (`{raw,std}×8`) é o que decide, e está **pendente** (junto dos 3 scripts
> de orquestração da grade). Os números abaixo são diagnóstico de 1 seed, com a
> ressalva "seeds before conclusions". Não promover a resultado final.

Segundo degrau da Fase 2. Adiciona os **outros agentes** da cena como polylines +
**cross-attention** do alvo sobre eles (primeira vez que atenção aparece na
escada). NÃO adiciona mapa (isso é o V2). Hipótese, sustentada pelo V0: o gap de
**MissRate** vem de falta de contexto → V1 é o primeiro candidato a movê-lo.

**Arquitetura (siblings limpos; o V0 NÃO foi editado):** encoder V0 + ramo social.
Subgraph **compartilhado** alvo+vizinhos (importado do V0 → máquina idêntica por
construção, o Δ métrico é atribuível só ao ramo social); vizinhos transformados
para o **frame agente-cêntrico do alvo** (mesmo `p0,θ0` → zero drift de frame);
cross-attention (`nn.MultiheadAttention`, 4 heads) + residual + LayerNorm;
seleção determinística top-K por `(distância@10, id)`, **K=16** fixo a priori
(flag `--n-neighbors`, fora do nome da pasta); **guard de zero-vizinhos** (evita
NaN de softmax tudo-mascarado). **~97k params.** O cache atual **já guarda todos
os agentes** da cena (`data['agents']`) → V1 não precisou reprocessar cache.

**Seed0 (raw, cls20, K=16, at convergence) — média das 9 breakdowns vs âncoras V0:**

| métrica | V1-social seed0 (raw) | V0-raw (8 seeds) | flatten-agente |
|---|---|---|---|
| minADE   | 1.488 | 1.468 ± 0.040 | **1.394** |
| minFDE   | **3.260** | 3.409 ± 0.080 | 3.293 |
| MissRate | 0.625 | 0.615 ± 0.022 | **0.576** |
| mAP      | 0.056 | 0.119 ± 0.032 | **0.117** |

(O seed0 do V0-raw era 1.538 em minADE — o **pior** dos 8. Comparar seed0-a-seed0
é ruído; a comparação honesta é contra a **média** V0-raw.)

**O achado — mAP pela METADE do V0, e por que NÃO é bug.** O mAP agregado (0.056)
é ~½ do V0 (0.119). Quatro checagens descartam bug e uma localiza a causa:

1. **Convergência sadia.** best_epoch=45, parou por platô (patience), não por
   budget. V0 convergia em 47–112. Não é convergência lenta.
2. **Score head ranqueia.** `val_rank`≈1.0–1.4 durante todo o treino (chance=2.5;
   <2.3 = ranqueia). No best: rank 1.27, top1≈52 vs chance≈128 (~2.5× melhor que a
   média). **Não é colapso de modo.**
3. **Calibração razoável.** Scores das predições: max-score médio 0.402, mediana
   0.365 (K=6; uniforme=0.167). Sub-confiante moderado, mas longe do achatamento
   (~0.20) que explicaria a queda sozinho. Causa no máximo **parcial**.
4. **Atenção viva — o teste decisivo.** Forward com-vs-sem vizinhos no mesmo
   checkpoint (mascarando os vizinhos): **|Δtraj| = 1.427 m** numa amostra com 3
   vizinhos. Os vizinhos chegam às heads e movem a predição em >1 m. **Não é
   atenção morta / bug de plumbing.**

**Causa real (localizada por breakdown):** o mAP morre nos horizontes **longos**.
mAP a 3s ok (VEHICLE_5=0.095, CYCLIST_5=0.100, comparável ao V0); a 8s despenca
(VEHICLE_15=0.025, CYCLIST_15=0.016). Isso segue o MissRate a 8s altíssimo
(VEHICLE_15=0.74, CYCLIST_15=0.84): quando quase tudo é "miss", não há
verdadeiros-positivos para ranquear e o mAP colapsa. **E o V1 NÃO moveu o
MissRate agregado** (0.625 vs V0-raw 0.615) — que era a hipótese central do V1.

**Enquadramento (reportável SE a grade confirmar).** O V1-raw seed0 diz,
honestamente: **agentes vizinhos sem mapa movem a trajetória (Δ=1.43 m real) mas
não bastam para fechar a cobertura a longo horizonte.** O mAP baixo é o *sintoma
agregado* disso, não um artefato. Se a grade confirmar, **fortalece a motivação do
V2 (mapa)** e é a ponte lógica: "isolamos o contexto social, medimos, não basta
para a cobertura longa → o próximo degrau é o mapa". O **Δ=1.43 m é a evidência a
guardar para a escrita** — separa "social não ajuda" de "social bugado" (revisores
vão querer essa distinção).

**Disciplina (o que NÃO fazer):** não rerodar o seed0 "pra ver se melhora" (o
número é informação, não erro); não ajustar `cls_weight`/temperatura com base em 1
seed (qualquer ajuste é sub-experimento MEDIDO e ROTULADO, feito DEPOIS da grade
confirmar a queda). O "cls_weight indistinguível" valia **no V0** e NÃO transfere
automaticamente pro V1 (a fusão mudou a escala do embedding que chega às heads).

**Pendências para fechar o degrau:** (1) escrever os 3 siblings de orquestração da
grade (`run_grid_gpu_social`, `run_grid_validate_social`,
`consolidate_seeds_social`); (2) rodar a grade `{raw,std}×8` e consolidar; (3)
baseline de velocidade constante (o "chão" verdadeiro, pendência antiga) antes de
reportar em absoluto.

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
cd /workspace && python3 -m src.motion.validate_motion_official

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