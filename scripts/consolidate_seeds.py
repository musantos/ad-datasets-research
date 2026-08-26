#!/usr/bin/env python3
"""
GENERIC per-seed consolidator. Unifies consolidate_seeds_item4.py and
consolidate_seeds_vectorized.py into a single script; the method becomes an
ARGUMENT (--model), like in the runners (run_grid_gpu.py / run_grid_validate.py).
The MODELS_CFG registry (same concept as the runners) defines, per method, the
FOLDER PREFIX and the axes (arms/variants) used only to compute --expected.

  python3 consolidate_seeds.py --model vectorized
  python3 consolidate_seeds.py --model social
  python3 consolidate_seeds.py --model multimodal

Reads the per-seed CSVs and writes two lean files:
  1. metrics_all.csv    (9 breakdowns per run)
  2. train_summary.csv  (1 summary row per run)

Expected names (identical to the source scripts):
  train:    <folder>_cls<w>_<arm>_seed<s>[_std][_<ts>].csv
  metrics:  metrics_<folder>_cls<w>_<arm>_seed<s>[_std][_<ts>].csv

The GOTCHA the registry solves (§3.1 of the transition): 'vectorized_social'
contains 'vectorized' as a prefix. The regex anchors the FULL folder NAME followed
by '_cls', so a social file NEVER matches as vectorized (after 'vectorized' the
social file carries '_social', not '_cls') and vice-versa. Confirmed by smoke test.

No external dependencies (stdlib only). Runs in any container.
"""

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# --- MODEL REGISTRY ----------------------------------------------------------
# In sync with the runners' MODELS_CFG. folder = folder prefix / file name;
# arms/variants only serve to compute the default --expected.
MODELS_CFG = {
    "multimodal": dict(folder="multimodal",        arms=["sdc", "agent"], variants=["raw"]),
    "sequential": dict(folder="sequential",        arms=["sdc", "agent"], variants=["raw"]),
    "vectorized": dict(folder="vectorized",        arms=["agent"],        variants=["raw", "std"]),
    "social":     dict(folder="vectorized_social", arms=["agent"],        variants=["raw", "std"]),
    "map":        dict(folder="vectorized_social_map", arms=["agent"],    variants=["raw"]),
}

CLS_WEIGHT_DEFAULT = 20

# Absolute roots (today's roots on their respective disks). --run-id resolves the
# read/write dirs against these, matching apply_run_id in run_grid_validate.py.
# The runners write into LOG_ROOT/runs/<id>, METRICS_DIR/runs/<id>; the
# consolidator reads from the SAME place (write/read parity is the whole point).
LOG_ROOT = "/workspace/experiments/logs"
METRICS_DIR = "/workspace/results"


def apply_run_id(run_id):
    """Resolve the run-scoped dirs from --run-id against the absolute roots
    (same convention as run_grid_validate.apply_run_id): logs and metrics live
    under runs/<id>, and the consolidated output goes into
    METRICS_DIR/runs/<id>/consolidated. run-id is a layer ABOVE run_tag; the
    regex (anchored on the folder name) and file names are unchanged, just
    relocated. Returns (logs_dir, results_dir, out_dir)."""
    logs_dir = os.path.join(LOG_ROOT, "runs", run_id)
    results_dir = os.path.join(METRICS_DIR, "runs", run_id)
    out_dir = os.path.join(results_dir, "consolidated")
    return logs_dir, results_dir, out_dir


# OPTIONAL timestamp: tolerates date-only (2026-08-19) and date+time
# (2026-08-19_07-54-08).
_TS = r"\d{4}-\d{2}-\d{2}(?:_\d{2}-\d{2}-\d{2})?"


def build_regexes(folder):
    """Strict regexes for (train, metrics), anchored on the FULL folder name.
    The _std suffix is OPTIONAL (absent -> raw; present -> std)."""
    lit = re.escape(folder)
    tail = (r"_cls(?P<cls>\d+)_(?P<arm>sdc|agent)"
            r"_seed(?P<seed>\d+)(?P<std>_std)?(?:_(?P<ts>" + _TS + r"))?\.csv$")
    re_train = re.compile(r"^(?P<model>" + lit + r")" + tail)
    re_metrics = re.compile(r"^metrics_(?P<model>" + lit + r")" + tail)
    return re_train, re_metrics


def pick_latest(files, regex):
    """Match each file; if there are several timestamps for the same config
    (model,cls,arm,variant,seed), keep the most recent one and warn. Files without
    a timestamp get '' (they sort before any dated one)."""
    by_cfg = defaultdict(list)
    ignored = []
    for f in files:
        m = regex.match(f.name)
        if not m:
            ignored.append(f.name)
            continue
        variant = "std" if m.group("std") else "raw"
        cfg = (m["model"], int(m["cls"]), m["arm"], variant, int(m["seed"]))
        by_cfg[cfg].append((m["ts"] or "", f))

    chosen = {}
    for cfg, lst in by_cfg.items():
        lst.sort(key=lambda x: x[0])  # ISO timestamp sorts lexicographically
        if len(lst) > 1:
            print(f"  [dup] {cfg}: {len(lst)} files, using the most recent "
                  f"({lst[-1][0] or 'no-ts'})", file=sys.stderr)
        chosen[cfg] = lst[-1][1]
    return chosen, ignored


def read_csv_skip_comments(path):
    """Read a CSV skipping the leading comment lines (#)."""
    with open(path, newline="") as fh:
        lines = [ln for ln in fh if not ln.lstrip().startswith("#")]
    return list(csv.DictReader(lines))


def build_metrics(results_dir, out_path, re_metrics):
    files = sorted(Path(results_dir).glob("*.csv"))
    chosen, ignored = pick_latest(files, re_metrics)
    print(f"[metrics] {len(chosen)} runs matched; {len(ignored)} files ignored")

    rows = []
    for (model, cls, arm, variant, seed), f in sorted(chosen.items()):
        for r in read_csv_skip_comments(f):
            rows.append({
                "model": model, "cls_weight": cls, "arm": arm,
                "variant": variant, "seed": seed,
                "breakdown_name": r["breakdown_name"],
                "minADE": r["minADE"], "minFDE": r["minFDE"],
                "MissRate": r["MissRate"], "OverlapRate": r["OverlapRate"],
                "mAP": r["mAP"],
            })

    fields = ["model", "cls_weight", "arm", "variant", "seed", "breakdown_name",
              "minADE", "minFDE", "MissRate", "OverlapRate", "mAP"]
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[metrics] -> {out_path}  ({len(rows)} rows, "
          f"{len(chosen)} runs x {len(rows)//max(len(chosen),1)} breakdowns)")
    return len(chosen)


def build_train_summary(logs_dir, out_path, re_train):
    files = sorted(Path(logs_dir).glob("*.csv"))
    chosen, ignored = pick_latest(files, re_train)
    print(f"[train]   {len(chosen)} runs matched; {len(ignored)} files ignored")

    rows = []
    for (model, cls, arm, variant, seed), f in sorted(chosen.items()):
        data = read_csv_skip_comments(f)
        if not data:
            print(f"  [empty] {f.name}", file=sys.stderr)
            continue
        last = data[-1]
        best_epoch = int(last["best_epoch"])
        # time to best epoch = cum_time_s of the row whose epoch == best_epoch
        t_best = next((float(r["cum_time_s"]) for r in data
                       if int(r["epoch"]) == best_epoch), float("nan"))
        total_epochs = int(last["epoch"])
        total_time = float(last["cum_time_s"])
        rows.append({
            "model": model, "cls_weight": cls, "arm": arm,
            "variant": variant, "seed": seed,
            "best_epoch": best_epoch,
            "time_to_best_s": round(t_best, 2),
            "total_epochs": total_epochs,
            "total_time_s": round(total_time, 2),
            # convergence: if == patience, it stopped by early stopping
            "epochs_after_best": total_epochs - best_epoch,
        })

    fields = ["model", "cls_weight", "arm", "variant", "seed", "best_epoch",
              "time_to_best_s", "total_epochs", "total_time_s", "epochs_after_best"]
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda x: (x["model"], x["cls_weight"],
                                                x["arm"], x["variant"], x["seed"])))
    print(f"[train]   -> {out_path}  ({len(rows)} rows)")
    return len(chosen)


def parse_seeds(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",") if x.strip() != ""]


def main():
    ap = argparse.ArgumentParser(description="Generic per-seed consolidator.")
    ap.add_argument("--model", required=True, choices=sorted(MODELS_CFG))
    ap.add_argument("--run-id", required=True,
                    help="grid run-id (REQUIRED). Resolves logs<-LOG_ROOT/runs/<id>, "
                         "results<-METRICS_DIR/runs/<id>, out->results/consolidated. "
                         "The --*-dir flags below OVERRIDE this when passed.")
    ap.add_argument("--logs-dir", default=None,
                    help="override: read per-epoch CSVs from here instead of "
                         "LOG_ROOT/runs/<run-id>.")
    ap.add_argument("--results-dir", default=None,
                    help="override: read metrics CSVs from here instead of "
                         "METRICS_DIR/runs/<run-id>.")
    ap.add_argument("--out-dir", default=None,
                    help="override: write metrics_all.csv/train_summary.csv here "
                         "instead of METRICS_DIR/runs/<run-id>/consolidated.")
    ap.add_argument("--cls-weights", default=str(CLS_WEIGHT_DEFAULT),
                    help="weight(s) to compute --expected: '20' or '1,20,50'.")
    ap.add_argument("--seeds", default="0-7",
                    help="seeds for --expected: '0-7' (range) or '0,1,2'.")
    ap.add_argument("--expected", type=int, default=None,
                    help="override of the expected run count. If omitted, computes "
                         "weights x variants x arms x seeds from the registry.")
    args = ap.parse_args()

    # --run-id resolves the run-scoped defaults; explicit --*-dir wins when passed.
    rid_logs, rid_results, rid_out = apply_run_id(args.run_id)
    logs_dir = args.logs_dir if args.logs_dir is not None else rid_logs
    results_dir = args.results_dir if args.results_dir is not None else rid_results
    out_dir = args.out_dir if args.out_dir is not None else rid_out

    cfg = MODELS_CFG[args.model]
    re_train, re_metrics = build_regexes(cfg["folder"])

    if args.expected is not None:
        expected = args.expected
    else:
        n_cls = len([w for w in args.cls_weights.split(",") if w.strip() != ""])
        expected = n_cls * len(cfg["variants"]) * len(cfg["arms"]) * len(parse_seeds(args.seeds))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[run-id] {args.run_id}")
    print(f"         logs    <- {logs_dir}")
    print(f"         metrics <- {results_dir}")
    print(f"         out     -> {out_dir}")
    print(f"[model] {args.model} -> folder='{cfg['folder']}'  "
          f"arms={cfg['arms']} variants={cfg['variants']}  expected={expected}")

    n_m = build_metrics(results_dir, out / "metrics_all.csv", re_metrics)
    n_t = build_train_summary(logs_dir, out / "train_summary.csv", re_train)

    print("\n--- check ---")
    for label, n in [("metrics", n_m), ("train", n_t)]:
        flag = "OK" if n == expected else "!! REVIEW"
        print(f"  {label}: {n} runs (expected {expected})  [{flag}]")


if __name__ == "__main__":
    main()