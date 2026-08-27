#!/usr/bin/env python3
"""
MASTER TABLE: one row per run, joining the three consolidated files
(train_summary.csv + gpu_all.csv + metrics_all.csv) by the RUN KEY
(model,cls_weight,arm,variant,seed). ADDITIVE output -- only reads CSV, runs in
ANY container (no TF needed).

  python3 report.py --in-dir . --out report.csv

Metric columns are PER HORIZON (mean over the 3 classes VEHICLE/PED/CYC): the
breakdowns TYPE_<class>_<h> become minADE_<h>, minFDE_<h>, MissRate_<h>, mAP_<h>.
The largest <h> is the longest horizon (8 s) -- the point of V1/V2 (coverage at
8 s). OverlapRate is left out of the expansion (secondary); easy to add if needed.

GPU per run: energy_wh = SUM of the phases (train+infer); util_gpu_train = mean
util of the TRAIN phase (infer is ~7 s of noise); vram_max_mib = max across phases.

Join by UNION of the three files' keys: nothing is dropped silently; runs with a
missing piece come out with empty cells and are counted in the final warning.

No external dependencies (stdlib only).
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

RUN_KEY = ["model", "cls_weight", "arm", "variant", "seed"]
METRICS = ["minADE", "minFDE", "MissRate", "mAP"]

# Absolute root (today's results root). --run-id resolves --in-dir against the
# shared consolidated/ dir (same convention as consolidate_seeds.py /
# consolidate_gpu.py), where the 3 consolidated CSVs live. report.csv is written
# INTO that same dir (final master table, next to its inputs).
METRICS_DIR = "/workspace/results"


def apply_run_id(run_id):
    """Resolve --in-dir from --run-id against the absolute results root: the
    consolidated/ dir holding metrics_all.csv, train_summary.csv, gpu_all.csv.
    Returns the in-dir (report.csv is written into it by default)."""
    return os.path.join(METRICS_DIR, "runs", run_id, "consolidated")


def read_csv_skip_comments(path):
    if not Path(path).is_file():
        return []
    with open(path, newline="") as fh:
        lines = [ln for ln in fh if not ln.lstrip().startswith("#")]
    return list(csv.DictReader(lines))


def key_of(r):
    """Run key with stable types (cls_weight/seed as int)."""
    return (r["model"], int(r["cls_weight"]), r["arm"], r["variant"], int(r["seed"]))


def fmean(vals, nd):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals), nd)


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def horizon_of(breakdown_name):
    """'TYPE_VEHICLE_15' -> '15'. Last token after '_'."""
    return breakdown_name.rsplit("_", 1)[-1]


def load_metrics(path):
    """runkey -> {metric -> {horizon -> [values over classes]}}."""
    agg = defaultdict(lambda: {m: defaultdict(list) for m in METRICS})
    horizons = set()
    for r in read_csv_skip_comments(path):
        k = key_of(r)
        h = horizon_of(r["breakdown_name"])
        horizons.add(h)
        for m in METRICS:
            agg[k][m][h].append(to_float(r.get(m)))
    return agg, horizons


def load_train(path):
    out = {}
    for r in read_csv_skip_comments(path):
        out[key_of(r)] = r
    return out


def load_gpu(path):
    """runkey -> {energy_wh(sum), util_gpu_train, vram_max_mib}."""
    by_run = defaultdict(list)
    for r in read_csv_skip_comments(path):
        by_run[key_of(r)].append(r)

    out = {}
    for k, rows in by_run.items():
        energy = [to_float(r.get("energy_wh")) for r in rows]
        energy = [e for e in energy if e is not None]
        vram = [to_float(r.get("vram_max_mib")) for r in rows]
        vram = [v for v in vram if v is not None]
        train_rows = [r for r in rows if r.get("phase") == "train"]
        util_src = train_rows[0] if train_rows else (rows[0] if rows else None)
        out[k] = {
            "energy_wh": round(sum(energy), 4) if energy else None,
            "util_gpu_train": to_float(util_src.get("util_gpu_mean")) if util_src else None,
            "vram_max_mib": int(max(vram)) if vram else None,
        }
    return out


def main():
    ap = argparse.ArgumentParser(description="Master table (join of the 3 consolidated files).")
    ap.add_argument("--run-id", required=True,
                    help="grid run-id (REQUIRED). Resolves --in-dir <- "
                         "METRICS_DIR/runs/<id>/consolidated (the 3 consolidated "
                         "CSVs). --in-dir/--out OVERRIDE this when passed.")
    ap.add_argument("--in-dir", default=None,
                    help="override: dir with metrics_all.csv, train_summary.csv, "
                         "gpu_all.csv. Defaults to METRICS_DIR/runs/<run-id>/consolidated.")
    ap.add_argument("--out", default=None,
                    help="override: path of the master table. Defaults to "
                         "report.csv inside --in-dir.")
    ap.add_argument("--metrics", default=None,
                    help="override input name. Default metrics_all_<run-id>.csv.")
    ap.add_argument("--train", default=None,
                    help="override input name. Default train_summary_<run-id>.csv.")
    ap.add_argument("--gpu", default=None,
                    help="override input name. Default gpu_all_<run-id>.csv.")
    args = ap.parse_args()

    # --run-id resolves the run-scoped in-dir; explicit flags win when passed.
    in_dir = args.in_dir if args.in_dir is not None else apply_run_id(args.run_id)
    out_path = args.out if args.out is not None else os.path.join(in_dir, f"report_{args.run_id}.csv")
    # Consolidated input names carry the run-id too (write/read parity with the
    # consolidators), so no artifact collides once pulled out of runs/<id>/.
    metrics_name = args.metrics if args.metrics is not None else f"metrics_all_{args.run_id}.csv"
    train_name = args.train if args.train is not None else f"train_summary_{args.run_id}.csv"
    gpu_name = args.gpu if args.gpu is not None else f"gpu_all_{args.run_id}.csv"

    d = Path(in_dir)
    print(f"[run-id] {args.run_id}")
    print(f"         in  <- {in_dir}")
    print(f"         out -> {out_path}")
    metrics, horizons = load_metrics(d / metrics_name)
    train = load_train(d / train_name)
    gpu = load_gpu(d / gpu_name)

    if not (metrics or train or gpu):
        print(f"[error] none of the 3 CSVs found in {d}/", file=sys.stderr)
        sys.exit(1)

    # sort horizons numerically when possible
    def hkey(h):
        try:
            return (0, int(h))
        except ValueError:
            return (1, h)
    horizons = sorted(horizons, key=hkey)

    metric_cols = [f"{m}_{h}" for m in METRICS for h in horizons]
    fields = (RUN_KEY +
              ["best_epoch", "time_to_best_s", "energy_wh", "util_gpu_train",
               "vram_max_mib"] + metric_cols)

    all_keys = sorted(set(metrics) | set(train) | set(gpu))
    rows = []
    n_partial = 0
    for k in all_keys:
        model, cls, arm, variant, seed = k
        row = {"model": model, "cls_weight": cls, "arm": arm,
               "variant": variant, "seed": seed}

        tr = train.get(k)
        row["best_epoch"] = tr.get("best_epoch") if tr else None
        row["time_to_best_s"] = tr.get("time_to_best_s") if tr else None

        gp = gpu.get(k)
        row["energy_wh"] = gp["energy_wh"] if gp else None
        row["util_gpu_train"] = gp["util_gpu_train"] if gp else None
        row["vram_max_mib"] = gp["vram_max_mib"] if gp else None

        mt = metrics.get(k)
        nd = {"minADE": 4, "minFDE": 4, "MissRate": 4, "mAP": 4}
        for m in METRICS:
            for h in horizons:
                col = f"{m}_{h}"
                row[col] = fmean(mt[m][h], nd[m]) if mt else None

        if not (tr and gp and mt):
            n_partial += 1
        rows.append(row)

    outp = Path(out_path)
    with open(outp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"[report] -> {outp}  ({len(rows)} runs, horizons={horizons})")
    print(f"[report] sources: metrics={len(metrics)} train={len(train)} gpu={len(gpu)} runs")
    if n_partial:
        print(f"[report] WARNING: {n_partial} runs with a missing piece "
              f"(no metrics, train OR gpu) -> empty cells.", file=sys.stderr)


if __name__ == "__main__":
    main()