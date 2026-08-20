#!/usr/bin/env python3
"""
Consolida os CSVs por-seed da GRADE V0 (vetorizado) em dois arquivos enxutos.

DIFERENCA vs a versao da Parte A: o item 4 tem o eixo do BRACO (sdc|agent) no
nome do arquivo, e cls fica fixo em 20. Os nomes agora sao:

  treino:   <model>_cls<w>_<arm>_seed<s>_<ts>.csv
            ex: vectorized_cls20_agent_seed0_2026-08-19_04-54-06.csv
  metricas: metrics_<model>_cls<w>_<arm>_seed<s>[_<ts>].csv
            ex: metrics_vectorized_cls20_agent_seed0_2026-08-19.csv

Como o regex casa SO 'vectorized', os arquivos do item 4 e da Parte A caem no
'ignored' de proposito -- cada regime tem sua tabela (a comparacao V0 vs flatten
e feita ENTRE tabelas consolidadas). Para reconsolidar o item 4, use a versao item4.

Saidas:
  1. metrics_all.csv    (mantem os 9 breakdowns por arquivo -> 8 x 9 = 72 linhas)
  2. train_summary.csv  (1 linha-resumo por run -> 8 linhas)

Uso:
    python3 consolidate_seeds_vectorized.py \
        --logs-dir experiments/logs \
        --results-dir results \
        --out-dir .

Sem dependencias externas (so stdlib). Roda em qualquer container.
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Padroes de nome (estritos -> ignoram lixo antigo e arquivos da Parte A).
# O braco (sdc|agent) e OBRIGATORIO. O timestamp e OPCIONAL e tolera tanto
# data-so (2026-08-19) quanto data+hora (2026-08-19_07-54-08), porque o
# validate pode gravar num formato ou no outro dependendo de --cleanup/--csv.
# ---------------------------------------------------------------------------
_TS = r"\d{4}-\d{2}-\d{2}(?:_\d{2}-\d{2}-\d{2})?"

RE_TRAIN = re.compile(
    r"^(?P<model>vectorized)_cls(?P<cls>\d+)_(?P<arm>sdc|agent)"
    r"_seed(?P<seed>\d+)(?:_(?P<ts>" + _TS + r"))?\.csv$"
)
RE_METRICS = re.compile(
    r"^metrics_(?P<model>vectorized)_cls(?P<cls>\d+)_(?P<arm>sdc|agent)"
    r"_seed(?P<seed>\d+)(?:_(?P<ts>" + _TS + r"))?\.csv$"
)


def pick_latest(files, regex):
    """Casa cada arquivo pelo regex; se houver varios timestamps para a mesma
    config (model,cls,arm,seed), mantem o mais recente e avisa. Arquivos sem
    timestamp no nome recebem '' (ordenam antes de qualquer datado)."""
    by_cfg = defaultdict(list)
    ignored = []
    for f in files:
        m = regex.match(f.name)
        if not m:
            ignored.append(f.name)
            continue
        cfg = (m["model"], int(m["cls"]), m["arm"], int(m["seed"]))
        ts = m["ts"] or ""
        by_cfg[cfg].append((ts, f))

    chosen = {}
    for cfg, lst in by_cfg.items():
        lst.sort(key=lambda x: x[0])  # timestamp ISO ordena lexicograficamente
        if len(lst) > 1:
            print(f"  [dup] {cfg}: {len(lst)} arquivos, usando o mais recente "
                  f"({lst[-1][0] or 'sem-ts'})", file=sys.stderr)
        chosen[cfg] = lst[-1][1]
    return chosen, ignored


def read_csv_skip_comments(path):
    """Le um CSV pulando linhas de comentario (#) do inicio."""
    with open(path, newline="") as fh:
        lines = [ln for ln in fh if not ln.lstrip().startswith("#")]
    return list(csv.DictReader(lines))


# ---------------------------------------------------------------------------
# 1) METRICAS
# ---------------------------------------------------------------------------
def build_metrics(results_dir, out_path):
    files = sorted(Path(results_dir).glob("*.csv"))
    chosen, ignored = pick_latest(files, RE_METRICS)
    print(f"[metricas] {len(chosen)} runs casados; {len(ignored)} arquivos ignorados")

    rows = []
    for (model, cls, arm, seed), f in sorted(chosen.items()):
        for r in read_csv_skip_comments(f):
            rows.append({
                "model": model, "cls_weight": cls, "arm": arm, "seed": seed,
                "breakdown_name": r["breakdown_name"],
                "minADE": r["minADE"], "minFDE": r["minFDE"],
                "MissRate": r["MissRate"], "OverlapRate": r["OverlapRate"],
                "mAP": r["mAP"],
            })

    fields = ["model", "cls_weight", "arm", "seed", "breakdown_name",
              "minADE", "minFDE", "MissRate", "OverlapRate", "mAP"]
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[metricas] -> {out_path}  ({len(rows)} linhas, "
          f"{len(chosen)} runs x {len(rows)//max(len(chosen),1)} breakdowns)")
    return len(chosen)


# ---------------------------------------------------------------------------
# 2) RESUMO DE TREINO
# ---------------------------------------------------------------------------
def build_train_summary(logs_dir, out_path):
    files = sorted(Path(logs_dir).glob("*.csv"))
    chosen, ignored = pick_latest(files, RE_TRAIN)
    print(f"[treino]   {len(chosen)} runs casados; {len(ignored)} arquivos ignorados")

    rows = []
    for (model, cls, arm, seed), f in sorted(chosen.items()):
        data = read_csv_skip_comments(f)
        if not data:
            print(f"  [vazio] {f.name}", file=sys.stderr)
            continue
        last = data[-1]
        best_epoch = int(last["best_epoch"])
        # tempo ate a melhor epoca = cum_time_s da linha cujo epoch == best_epoch
        t_best = next((float(r["cum_time_s"]) for r in data
                       if int(r["epoch"]) == best_epoch), float("nan"))
        total_epochs = int(last["epoch"])
        total_time = float(last["cum_time_s"])
        rows.append({
            "model": model, "cls_weight": cls, "arm": arm, "seed": seed,
            "best_epoch": best_epoch,
            "time_to_best_s": round(t_best, 2),
            "total_epochs": total_epochs,
            "total_time_s": round(total_time, 2),
            # convergencia: se == patience, parou por early stopping
            "epochs_after_best": total_epochs - best_epoch,
        })

    fields = ["model", "cls_weight", "arm", "seed", "best_epoch", "time_to_best_s",
              "total_epochs", "total_time_s", "epochs_after_best"]
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda x: (x["model"], x["cls_weight"],
                                                x["arm"], x["seed"])))
    print(f"[treino]   -> {out_path}  ({len(rows)} linhas)")
    return len(chosen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir", default="experiments/logs")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--expected", type=int, default=8,
                    help="numero esperado de runs por tipo (V0: 1 modelo x "
                         "1 braco x 8 seeds = 8)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    n_m = build_metrics(args.results_dir, out / "metrics_all.csv")
    n_t = build_train_summary(args.logs_dir, out / "train_summary.csv")

    print("\n--- verificacao ---")
    for label, n in [("metricas", n_m), ("treino", n_t)]:
        flag = "OK" if n == args.expected else "!! CONFERIR"
        print(f"  {label}: {n} runs (esperado {args.expected})  [{flag}]")


if __name__ == "__main__":
    main()
