#!/usr/bin/env python3
"""
Consolida os CSVs por-seed em dois arquivos enxutos para analise da Parte A:

  1. metrics_all.csv     -> results/metrics_<model>_cls<w>_seed<s>_*.csv
                            (mantem os 9 breakdowns por arquivo -> 40 x 9 = 360 linhas)
  2. train_summary.csv   -> experiments/logs/<model>_cls<w>_seed<s>_*.csv
                            (1 linha-resumo por run -> 40 linhas)

Uso:
    python3 consolidate_seeds.py \
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
# Padroes de nome de arquivo (estritos -> ignoram lixo antigo nas pastas)
# ---------------------------------------------------------------------------
# treino:  multimodal_cls1_seed0_2026-08-19_04-54-06.csv
RE_TRAIN = re.compile(
    r"^(?P<model>multimodal|sequential)_cls(?P<cls>\d+)_seed(?P<seed>\d+)"
    r"_(?P<ts>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.csv$"
)
# metricas: metrics_multimodal_cls20_seed0_2026-08-19_07-54-08.csv
RE_METRICS = re.compile(
    r"^metrics_(?P<model>multimodal|sequential)_cls(?P<cls>\d+)_seed(?P<seed>\d+)"
    r"_(?P<ts>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.csv$"
)


def pick_latest(files, regex):
    """Casa cada arquivo pelo regex; se houver varios timestamps para a mesma
    config (model,cls,seed), mantem o mais recente e avisa."""
    by_cfg = defaultdict(list)
    ignored = []
    for f in files:
        m = regex.match(f.name)
        if not m:
            ignored.append(f.name)
            continue
        cfg = (m["model"], int(m["cls"]), int(m["seed"]))
        by_cfg[cfg].append((m["ts"], f))

    chosen = {}
    for cfg, lst in by_cfg.items():
        lst.sort(key=lambda x: x[0])  # timestamp ISO ordena lexicograficamente
        if len(lst) > 1:
            print(f"  [dup] {cfg}: {len(lst)} arquivos, usando o mais recente "
                  f"({lst[-1][0]})", file=sys.stderr)
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
    for (model, cls, seed), f in sorted(chosen.items()):
        for r in read_csv_skip_comments(f):
            rows.append({
                "model": model, "cls_weight": cls, "seed": seed,
                "breakdown_name": r["breakdown_name"],
                "minADE": r["minADE"], "minFDE": r["minFDE"],
                "MissRate": r["MissRate"], "OverlapRate": r["OverlapRate"],
                "mAP": r["mAP"],
            })

    fields = ["model", "cls_weight", "seed", "breakdown_name",
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
    for (model, cls, seed), f in sorted(chosen.items()):
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
            "model": model, "cls_weight": cls, "seed": seed,
            "best_epoch": best_epoch,
            "time_to_best_s": round(t_best, 2),
            "total_epochs": total_epochs,
            "total_time_s": round(total_time, 2),
            # convergencia: se == patience, parou por early stopping
            "epochs_after_best": total_epochs - best_epoch,
        })

    fields = ["model", "cls_weight", "seed", "best_epoch", "time_to_best_s",
              "total_epochs", "total_time_s", "epochs_after_best"]
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda x: (x["model"], x["cls_weight"], x["seed"])))
    print(f"[treino]   -> {out_path}  ({len(rows)} linhas)")
    return len(chosen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir", default="experiments/logs")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--expected", type=int, default=40,
                    help="numero esperado de runs por tipo (default 40)")
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
