#!/usr/bin/env python3
"""
Assigns GPU telemetry to each (run, phase). Joins:
  * phase_index_*.csv  -> windows [start,end] per (run, phase={train,infer})
  * gpulog_state_*.csv -> nvidia-smi samples (util/VRAM/power/temp) ~every 5 s
by TIME, and aggregates. ADDITIVE output (only reads CSV; never touches
train/infer/validate).

  python3 consolidate_gpu.py --phase-dir experiments/gpu_logs --out-dir .

Emits gpu_all.csv: ONE row per (run, phase) -- same pattern as metrics_all.csv.
report.py sums/selects phases afterwards (train is what matters; infer is ~7 s).

--- DESIGN DECISIONS (formats confirmed against a real sample) -----------------

1. JOIN BY WALL CLOCK (naive), not by epoch. phase_index (start_iso) and gpulog
   (timestamp) are written by the SAME machine/clock, only in different formats:
   phase_index '2026-08-20 16:00:37' (hyphen, no ms) vs gpulog
   '2026/08/20 16:00:37.722' (slash, ms). Parsing both as naive datetime and
   comparing, the join is exact WITHOUT calibrating timezone (same clock).
   start_epoch exists and works as a cross-check, but is not required.

2. phase_index has TWO formats; we tolerate both (backward-compat rule from §4):
   * NEW (current runners): 'cls_weight' column present.
   * OLD (item4/V0): NO 'cls_weight' -> default read from the comment
     (# ... cls_weight=20 ...) or 20.

3. gpulog: units embedded in the VALUE ('7 %', '1257 MiB', '14.70 W') and a space
   after the comma. num() strips unit/space; '[N/A]' -> None. pstate ('P5') is not
   numeric (ignored in aggregates).

4. PAIRING phase_index <-> gpulog: the 2nd comment line of phase_index points to
   the file ('# gpu_state=gpulog_state_<ts>.csv ...'). We load that name from
   --gpu-dir (default = --phase-dir). Fallback: same <ts> from the name.

No external dependencies (stdlib only). Runs in any container.
"""

import argparse
import csv
import glob
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path

NOMINAL_INTERVAL_S = 5.0          # nvidia-smi -l 5
CLS_WEIGHT_DEFAULT = 20

# Absolute root (today's gpu_logs root). --run-id resolves the phase/gpu dir and
# the output against this, matching apply_run_id in run_grid_validate.py and
# consolidate_seeds.py. run_grid_gpu.py writes phase_index_* and gpulog_state_*
# into GPULOG_DIR/runs/<id>; this reads from the SAME place. The output goes into
# the shared consolidated/ (METRICS_DIR/runs/<id>/consolidated) so report.py finds
# gpu_all.csv next to metrics_all.csv/train_summary.csv.
GPULOG_DIR = "/workspace/experiments/gpu_logs"
METRICS_DIR = "/workspace/results"


def apply_run_id(run_id):
    """Resolve the run-scoped dirs from --run-id against the absolute roots
    (same convention as consolidate_seeds.apply_run_id). phase_index and
    gpulog_state both live under GPULOG_DIR/runs/<id>; the output joins the
    shared consolidated/ dir. Returns (phase_dir, out_dir)."""
    phase_dir = os.path.join(GPULOG_DIR, "runs", run_id)
    out_dir = os.path.join(METRICS_DIR, "runs", run_id, "consolidated")
    return phase_dir, out_dir


RUN_KEY = ["model", "cls_weight", "arm", "variant", "seed"]

OUT_FIELDS = RUN_KEY + [
    "phase", "n_samples", "dur_s",
    "util_gpu_mean", "util_gpu_p95", "util_gpu_max",
    "util_mem_mean", "vram_max_mib",
    "power_mean_w", "power_max_w", "energy_wh",
    "temp_max_c", "returncode",
]


# --- parsing helpers ---------------------------------------------------------
def num(s):
    """'7 %'->7.0, '1257 MiB'->1257.0, '14.70 W'->14.70, '42'->42.0,
    '[N/A]'/'P5'/''->None."""
    if s is None:
        return None
    tok = s.strip().split()
    if not tok:
        return None
    try:
        return float(tok[0])
    except ValueError:
        return None


def parse_phase_dt(s):
    return datetime.strptime(s.strip(), "%Y-%m-%d %H:%M:%S")


def parse_gpu_dt(s):
    return datetime.strptime(s.strip(), "%Y/%m/%d %H:%M:%S.%f")


def percentile(sorted_vals, p):
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return sorted_vals[int(k)]
    return sorted_vals[lo] * (hi - k) + sorted_vals[hi] * (k - lo)


def read_comment_and_rows(path):
    """Read a CSV with comment lines (#) on top. Returns (comments, rows-dict)."""
    with open(path, newline="") as fh:
        raw = fh.readlines()
    comments = [ln for ln in raw if ln.lstrip().startswith("#")]
    body = [ln for ln in raw if not ln.lstrip().startswith("#")]
    rows = list(csv.DictReader(body))
    return comments, rows


def cls_from_comments(comments):
    for ln in comments:
        m = re.search(r"cls_weights?\s*=\s*\[?\s*(\d+)", ln)
        if m:
            return int(m.group(1))
    return CLS_WEIGHT_DEFAULT


def gpu_state_name_from_comments(comments):
    for ln in comments:
        m = re.search(r"gpu_state\s*=\s*(\S+)", ln)
        if m and m.group(1).upper() != "NA":
            return m.group(1)
    return None


# --- gpulog loading ----------------------------------------------------------
def load_gpu_state(path):
    """Returns a time-sorted list of dicts: {t, util_gpu, util_mem, vram, power,
    temp}. Skips samples without a parseable timestamp."""
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if not header:
            return []
        # columns by POSITION (nvidia-smi names vary with [units]):
        # 0 timestamp, 1 util.gpu, 2 util.mem, 3 mem.used, 4 power, 5 temp, 6 pstate
        out = []
        for r in reader:
            if len(r) < 6:
                continue
            try:
                t = parse_gpu_dt(r[0])
            except (ValueError, IndexError):
                continue
            out.append(dict(
                t=t,
                util_gpu=num(r[1]), util_mem=num(r[2]),
                vram=num(r[3]), power=num(r[4]), temp=num(r[5]),
            ))
    out.sort(key=lambda x: x["t"])
    return out


def aggregate_window(samples, t0, t1, dur_s):
    """Aggregate samples with t0 <= t <= t1. energy_wh by trapezoid over the real
    time between consecutive samples; fallback = power_mean * dur_s if <2 samples."""
    win = [s for s in samples if t0 <= s["t"] <= t1]
    n = len(win)
    agg = dict.fromkeys(
        ["util_gpu_mean", "util_gpu_p95", "util_gpu_max", "util_mem_mean",
         "vram_max_mib", "power_mean_w", "power_max_w", "energy_wh", "temp_max_c"],
        None)
    agg["n_samples"] = n
    if n == 0:
        return agg

    ug = sorted(v["util_gpu"] for v in win if v["util_gpu"] is not None)
    um = [v["util_mem"] for v in win if v["util_mem"] is not None]
    vr = [v["vram"] for v in win if v["vram"] is not None]
    pw = [v["power"] for v in win if v["power"] is not None]
    tp = [v["temp"] for v in win if v["temp"] is not None]

    if ug:
        agg["util_gpu_mean"] = round(sum(ug) / len(ug), 1)
        agg["util_gpu_p95"] = round(percentile(ug, 95), 1)
        agg["util_gpu_max"] = round(ug[-1], 1)
    if um:
        agg["util_mem_mean"] = round(sum(um) / len(um), 1)
    if vr:
        agg["vram_max_mib"] = int(max(vr))
    if pw:
        agg["power_mean_w"] = round(sum(pw) / len(pw), 2)
        agg["power_max_w"] = round(max(pw), 2)
    if tp:
        agg["temp_max_c"] = int(max(tp))

    # energy (Wh) = integral of power dt / 3600
    energy_j = 0.0
    have_energy = False
    for i in range(n - 1):
        a, b = win[i], win[i + 1]
        if a["power"] is None or b["power"] is None:
            continue
        dt = (b["t"] - a["t"]).total_seconds()
        if dt <= 0 or dt > 3 * NOMINAL_INTERVAL_S:      # gap -> do not integrate the hole
            continue
        energy_j += 0.5 * (a["power"] + b["power"]) * dt
        have_energy = True
    if not have_energy and pw and dur_s:                # fallback for a short window
        energy_j = (sum(pw) / len(pw)) * float(dur_s)
    agg["energy_wh"] = round(energy_j / 3600.0, 4) if (have_energy or pw) else None
    return agg


# --- main --------------------------------------------------------------------
def process_phase_index(pi_path, gpu_dir):
    comments, rows = read_comment_and_rows(pi_path)
    if not rows:
        print(f"  [empty] {os.path.basename(pi_path)}", file=sys.stderr)
        return []

    has_cls_col = "cls_weight" in rows[0]
    default_cls = cls_from_comments(comments)

    missing = [c for c in ("cls_weight", "arm", "variant") if c not in rows[0]]
    if missing:
        print(f"  [compat] {os.path.basename(pi_path)}: missing columns {missing} "
              f"-> defaults (cls={default_cls}, arm=agent, variant=raw)", file=sys.stderr)

    state_name = gpu_state_name_from_comments(comments)
    samples = []
    if state_name:
        sp = os.path.join(gpu_dir, state_name)
        if os.path.isfile(sp):
            samples = load_gpu_state(sp)
        else:
            print(f"  [warn] gpu_state missing: {sp} (rows without telemetry)",
                  file=sys.stderr)
    else:
        print(f"  [warn] {os.path.basename(pi_path)}: no gpu_state ref in the "
              f"comment (rows without telemetry)", file=sys.stderr)

    out = []
    for r in rows:
        try:
            t0 = parse_phase_dt(r["start_iso"])
            t1 = parse_phase_dt(r["end_iso"])
        except (ValueError, KeyError):
            print(f"  [warn] row with invalid iso in "
                  f"{os.path.basename(pi_path)}; skipped", file=sys.stderr)
            continue
        dur_s = r.get("duration_s")
        cls = int(float(r["cls_weight"])) if has_cls_col else default_cls

        agg = aggregate_window(samples, t0, t1, dur_s)
        row = {
            "model": r["model"], "cls_weight": cls,
            "arm": r.get("arm", "agent"),          # old formats may not have it
            "variant": r.get("variant", "raw"),    # missing -> raw (was raw-only)
            "seed": int(r["seed"]),
            "phase": r["phase"],
            "dur_s": round(float(dur_s), 1) if dur_s else None,
            "returncode": r.get("returncode"),
        }
        row.update({k: agg[k] for k in
                    ["n_samples", "util_gpu_mean", "util_gpu_p95", "util_gpu_max",
                     "util_mem_mean", "vram_max_mib", "power_mean_w",
                     "power_max_w", "energy_wh", "temp_max_c"]})
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser(description="Consolidate GPU telemetry per (run,phase).")
    ap.add_argument("--run-id", required=True,
                    help="grid run-id (REQUIRED). Resolves phase-dir<-GPULOG_DIR/"
                         "runs/<id> and out->METRICS_DIR/runs/<id>/consolidated. "
                         "--phase-dir/--gpu-dir/--out-dir OVERRIDE this when passed.")
    ap.add_argument("--phase-dir", default=None,
                    help="override: dir with phase_index_*.csv (and, by default, "
                         "the gpulog_state_*). Defaults to GPULOG_DIR/runs/<run-id>.")
    ap.add_argument("--gpu-dir", default=None,
                    help="dir of the gpulog_state_*.csv (default = --phase-dir).")
    ap.add_argument("--out-dir", default=None,
                    help="override: write gpu_all.csv here instead of "
                         "METRICS_DIR/runs/<run-id>/consolidated.")
    ap.add_argument("--phase-glob", default="phase_index_*.csv",
                    help="glob for the phase_index files inside --phase-dir.")
    args = ap.parse_args()

    # --run-id resolves the run-scoped defaults; explicit flags win when passed.
    rid_phase, rid_out = apply_run_id(args.run_id)
    phase_dir = args.phase_dir if args.phase_dir is not None else rid_phase
    out_dir = args.out_dir if args.out_dir is not None else rid_out
    gpu_dir = args.gpu_dir or phase_dir

    print(f"[run-id] {args.run_id}")
    print(f"         phase   <- {phase_dir}")
    print(f"         gpu     <- {gpu_dir}")
    print(f"         out     -> {out_dir}")

    pi_files = sorted(glob.glob(os.path.join(phase_dir, args.phase_glob)))
    if not pi_files:
        print(f"[error] no phase_index in {phase_dir}/{args.phase_glob}",
              file=sys.stderr)
        sys.exit(1)

    # dedup by key (run,phase): if the same config appears in several phase_index
    # files (re-run), the LAST file (lexicographic order of the ts in the name) wins.
    by_key = {}
    for pi in pi_files:
        print(f"[phase-index] {os.path.basename(pi)}")
        for row in process_phase_index(pi, gpu_dir):
            k = tuple(row[c] for c in RUN_KEY) + (row["phase"],)
            by_key[k] = row

    rows = sorted(by_key.values(),
                  key=lambda x: (x["model"], x["cls_weight"], x["arm"],
                                 x["variant"], x["seed"], x["phase"]))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / f"gpu_all_{args.run_id}.csv"
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=OUT_FIELDS)
        w.writeheader()
        w.writerows(rows)

    n_train = sum(1 for r in rows if r["phase"] == "train")
    n_infer = sum(1 for r in rows if r["phase"] == "infer")
    n_empty = sum(1 for r in rows if r["n_samples"] == 0)
    print(f"[gpu] -> {out_path}  ({len(rows)} rows: {n_train} train, {n_infer} infer)")
    if n_empty:
        print(f"[gpu] WARNING: {n_empty} rows without telemetry samples "
              f"(gpu_state missing or window outside the log).", file=sys.stderr)


if __name__ == "__main__":
    main()